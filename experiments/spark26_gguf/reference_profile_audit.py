"""Audit shape-linked GGUF launches for the negative reference-patch diagnosis."""
import argparse
import gzip
from hashlib import file_digest
from collections import defaultdict
import json
from pathlib import Path
import re

from gguf_trace_audit import audit as audit_trace, summarize_trace

SOURCE_LOCK = Path(__file__).with_name('reference_pair_workload.json')
PREFILL = re.compile(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),')
STAGE_SHAPES = {'short1/prefill': 130, 'long2/prefill': 1536, 'short1/decode': 1, 'long2/decode': 2}


def projection(n, k):
    if k == 2048 and n in (512,2048,3072):
        # The N=2048 path also includes the common output projection, and a
        # profiler shape alone cannot separate it from the original Q call.
        return 'qkv_plus_attention_out'
    if k == 2048 and n in (6656,13312):
        return 'ffn_gate_up'
    if k == 2048 and n == 8:
        return 'unchanged_attention_gate'
    if k == 6656 and n == 2048:
        return 'unchanged_ffn_down'
    return 'other'


def summarize_arm(rows, m):
    groups = defaultdict(lambda: dict(matmul_calls=0, matmul_us=0.0, activation_quant_calls=0, activation_quant_us=0.0))
    matrix = []
    for r in rows:
        if r['m'] != m:
            continue
        label = projection(r['n'],r['k'])
        p = groups[label]
        if re.search(r'\b(?:mul_mat_q8_0|mul_mat_vec_q)<', r['kernel']):
            p['matmul_calls'] += r['calls']
            p['matmul_us'] += r['sum_us']
            matrix.append(r)
        elif 'quantize_q8_1<' in r['kernel']:
            p['activation_quant_calls'] += r['calls']
            p['activation_quant_us'] += r['sum_us']
    anchor = groups['unchanged_attention_gate']['matmul_calls']
    if anchor <= 0 or anchor % 28:
        raise ValueError(f'M={m}: no known unchanged 28-layer anchor for normalization')
    forwards = anchor//28
    for p in groups.values():
        p['matmul_us_per_forward'] = p['matmul_us']/forwards
        p['activation_quant_us_per_forward'] = p['activation_quant_us']/forwards
    return dict(forward_count=forwards, groups=groups, kernel_launch_shapes=matrix)


def check_log(log, arm, digest):
    markers = [line for line in log.splitlines() if line.startswith('EE_REFERENCE_PROFILE ')]
    if markers != [f'EE_REFERENCE_PROFILE {arm} {digest}']:
        raise ValueError(f'{arm}: wrong source marker')
    windows = dict(short1=[], long2=[])
    shape = None
    for line in log.splitlines():
        if line.startswith('EE_CALIBRATION '):
            shape = line.split()[1]
        elif shape in windows and (m := PREFILL.search(line)):
            windows[shape].append(tuple(map(int,m.groups())))
    # The client marks each profile after its two warm-up waves, so the next
    # shape's warm-ups are logged between the two profile markers.
    if windows != {'short1': [(1,130,0)]*4+[(2,1536,0)]*2, 'long2': [(2,1536,0)]*4}:
        raise ValueError(f'{arm}: expected four real prefill waves per profiled M, got {windows}')


def audit(root):
    lock = json.loads(SOURCE_LOCK.read_bytes())
    res = {}
    for arm in ('a','b'):
        path = root/arm
        check_log((path/'profile.server.log').read_text(encoding='utf-8'), arm, lock['source_sha256'][arm])
        grps = audit_trace(path)['groups']
        out = {}
        for stage, m in STAGE_SHAPES.items():
            items = grps[stage]
            if len(items) != 1 or not any((d['computeMajor'],d['computeMinor'])==(12,1) for d in items[0]['device_properties']):
                raise ValueError(f'{arm}/{stage}: trace ambiguity or unexpected GPU')
            row = summarize_arm(items[0]['shape_linked_kernel_launches'], m)
            if 'ffn_gate_up' not in row['groups'] or 'qkv_plus_attention_out' not in row['groups']:
                raise ValueError(f'{arm}/{stage}: missing coalesced projection families')
            out[stage] = dict(m=m, profiled_sum_cuda_kernel_us=items[0]['sum_kernel_us'], **row)
        res[arm] = out
    return dict(arms=res,note='Separate instrumented shape samples; summed kernel duration is not end-to-end latency.')


def audit_surviving_prefill(root):
    """Explicitly partial diagnosis after B stalled; never substitute for audit()."""
    lock = json.loads(SOURCE_LOCK.read_bytes())
    result = {}
    for arm, manifest in (('a','profiles.sha256'),('b','partial-profiles.sha256')):
        path = root/arm
        log = (path/'profile.server.log').read_text(encoding='utf-8')
        digest = lock['source_sha256'][arm]
        if [line for line in log.splitlines() if line.startswith('EE_REFERENCE_PROFILE ')] != [f'EE_REFERENCE_PROFILE {arm} {digest}']:
            raise ValueError('Wrong partial-trace source')
        marker = 'EE_CALIBRATION short1 profile'
        if log.count(marker) != 1:
            raise ValueError('Missing short1 stage marker')
        trail = log.partition(marker)[2].partition('EE_CALIBRATION long2 profile')[0]
        shape = PREFILL.search(trail)
        if not shape or shape.groups() != ('1','130','0'):
            raise ValueError('Wrong surviving scheduler prefill shape')
        traces = [line.split('  ',1) for line in (path/manifest).read_text(encoding='utf-8').splitlines()]
        (checksum, relative), = [(h,p) for h,p in traces
                                   if re.fullmatch(r'profiles/short1-[0-9.]+-TP-0-EXTEND\.trace\.json\.gz',p)]
        target = path/relative
        with target.open('rb') as f:
            if file_digest(f,'sha256').hexdigest() != checksum:
                raise ValueError('Wrong partial-trace checksum')
        with gzip.open(target,'rt',encoding='utf-8') as f:
            trace = summarize_trace(json.load(f))
        if not any((g['computeMajor'],g['computeMinor'])==(12,1) for g in trace['device_properties']):
            raise ValueError('Wrong device')
        out = summarize_arm(trace['shape_linked_kernel_launches'],130)
        if out['forward_count'] != 1:
            raise ValueError('Expected one captured prefill forward')
        result[arm] = dict(checksum=checksum, **out, profiled_sum_cuda_kernel_us=trace['sum_kernel_us'])
    return dict(arms=result, note='Partial diagnosis: short1 prefill only; B profiling timed out in decode. This is not a performance sample.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence_dir',type=Path)
    p.add_argument('--surviving-prefill',action='store_true',help='explicitly scope recovery of a failed B trace to short1 prefill')
    args = p.parse_args()
    result = audit_surviving_prefill(args.evidence_dir) if args.surviving_prefill else audit(args.evidence_dir)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
