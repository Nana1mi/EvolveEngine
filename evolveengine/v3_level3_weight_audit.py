"""Host-only Q8 engineering audit packets. Signals are not legality tests or scores."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from .v3_harness import canonical, digest

POLICY = 'q8-external-review-v1'
PROVISIONAL = 'provisional_pending_weight_review'
AGENT_NOTICE = ('Memory measurements are telemetry, never a development score or an E0-relative limit. '
    'You may trade memory for time using CUDA Graph, KV cache, speculative buffers, transient per-tile '
    'dequantization and bit-exact rearrangement of the original Q8_0 codes and scales. Do not persist any '
    'alternate decoded or requantized target weights, including partial BF16/FP16, FP8 or Q4 copies. '
    'Hardware/OOM, isolation and correctness checks remain in force. All gains are provisional until '
    'independent host review of the actual built runtime; a file hash or a clean scan alone is not approval.')
CAST = re.compile(r'dequant|requant|quantize|float8|\bfp8|\bq4|bfloat16|float16|\.half\(|\.to\s*\(', re.I)
STORE = re.compile(r'register_buffer|(?:self|module|model|layer)\.[\w.]+\s*=|(?:cache|table|weights?)\w*\s*\[|\b(?:static|global)\b', re.I)
SHAPE = re.compile(r'(?:empty|zeros|ones|full)_like\s*\(|(?:weight|qweight)\w*\.(?:shape|size)', re.I)
DTYPE = re.compile(r'dtype|float|half|\bq4|\bfp8', re.I)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(value):
    need(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid audit digest')
    return value


def wire(obj):
    return canonical(obj) + b'\n'


def store(path, raw):
    """Publish a complete immutable host artifact; readers never see partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        tmp.write(raw)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            need(path.read_bytes() == raw, 'different evidence already exists at this identity')
    finally:
        temp.unlink()


TOKEN_RUN = re.compile(rb'(?<![.\w])(?:0[xX][a-fA-F0-9]+|[0-9]+)[uUlL]*\s*,\s*')
INPUT_HASH = re.compile(r'(?:prompt|input|token).{0,40}(?:hash|digest)|(?:hash|digest).{0,60}(?:prompt|input|token)', re.I)
LOOKUP = re.compile(r'\b(?:if|switch|case|match|lookup|find|contains|get)\b|\[[^]\n]{0,120}(?:hash|digest)', re.I)


def static_scan(base, edits):
    findings, total = [], 0
    for path, data in edits:
        old = Counter(base.get(path, b'').splitlines()); lines = data.splitlines()
        literal_count = 0
        literal_start = None
        for i, line in enumerate(lines):
            if old[line]:
                literal_start, literal_count = None, 0
                old[line] -= 1
                continue
            nearby = '\n'.join(l[:2048].decode('utf8', 'replace') for l in lines[max(0,i-3):i+4])
            tags = []
            # Numeric sequence across changed lines, including native initializers and JSON.
            # Report once per run; no token values are exported and no finding is a verdict.
            count = 0
            for _ in TOKEN_RUN.finditer(line):
                count += 1
                if count >= 32: break
            if count:
                if literal_start is None: literal_start = i
                if literal_count < 32 <= literal_count + count: tags.append('embedded_integer_table')
                literal_count += count
            else:
                literal_start, literal_count = None, 0
            if INPUT_HASH.search(nearby) and LOOKUP.search(nearby): tags.append('input_fingerprint_dispatch')
            if CAST.search(nearby) and STORE.search(nearby): tags.append('conversion_near_persistent_storage')
            if SHAPE.search(nearby) and DTYPE.search(nearby): tags.append('possibly_same_shape_different_dtype')
            if tags:
                total += 1
                if len(findings) < 128:
                    findings.append(dict(path=path, line=i+1, hints=tags))
    return dict(changed_line_hits=total, locations=findings, complete_source_available=True,
                scope='lexical hints, including native text; no dataflow or runtime proof')


def memory_hint(candidate, reference):
    a, b = (v.get('weight_observation', {}) for v in (candidate, reference))
    if not a.get('workload') or a['workload'] != b.get('workload'):
        return dict(status='no_matched_comparison')
    x,y = (v['sampled_bytes'] for v in (a,b))
    if not x or x.keys() != y.keys(): return dict(status='no_matched_comparison')
    delta = max(x.values()) - max(y.values())
    return dict(status='matched', peak_delta_bytes=delta, stages_delta_bytes={k:x[k]-y[k] for k in x},
                investigate_growth_over_300_mib=delta > 300*(1<<20))


def bandwidth_hint(data):
    out = {}
    for scene in ('W1','W2'):
        diag = (data.get('speculation') or {}).get('timed', {}).get(scene, {})
        spec = ('reported_speculation' if diag.get('proposed',0) or diag.get('verifies',0) else
                'reported_no_proposals' if diag.get('requests') and diag.get('reported') == diag['requests'] else 'unknown')
        tpot = data.get('scenes',{}).get(scene,{}).get('tpot_ms')
        out[scene] = dict(tpot_ms=tpot, speculation=spec, below_screening_estimate=(tpot < 6.7 if tpot is not None else None),
            investigate=tpot is not None and tpot < 6.7 and spec != 'reported_speculation')
    return out


def packet(folder, contract, engine, chosen, build, quality, reference, timings, *, trigger, context, source_cache=None):
    need(trigger in ('historical_best','checkpoint','final_nomination'), 'unknown weight audit trigger')
    need(chosen['engine'] == engine.digest == quality['engine'] == build['engine'] and
         chosen['build'] == build['evidence'] == quality['build'] and chosen['quality'] == quality['evidence'],
         'audit inputs belong to different deployments')
    cached = (source_cache or {}).get(engine.digest)
    if cached is None:
        source = wire(contract.to_args(engine)); source_pin = digest(source)
        store(folder/'sources'/f'{source_pin}.json', source)
        findings = static_scan(contract.base, engine.edits)
        if source_cache is not None: source_cache[engine.digest] = (source_pin, findings)
    else:
        source_pin, findings = cached
    doc = dict(policy=POLICY, gain_status=PROVISIONAL, trigger=trigger, context=context, chosen=chosen,
        baseline=contract.baseline, source_sha256=source_pin, static=findings,
        build=build, quality=quality, e0_quality=reference,
        memory=memory_hint(quality['data'], reference['data']),
        paired_timing=[dict(receipt=j['evidence'], comparison=j['performance'], bandwidth=bandwidth_hint(j['data'])) for j in timings],
        limitations=['Observed stages are snapshots, not continuous allocator peaks.',
                     '6.7 ms/token is an uncalibrated single-stream screening estimate, not a physical bound.',
                     'Speculation diagnostics are self-reported; unknown is not disabled.',
                     'Hashes and static analysis cannot exclude hidden Native copies. Independently inspect runtime representation and lifetime.'])
    raw = wire(doc); key = digest(raw); store(folder/'packets'/f'{key}.json', raw); return key


CHECKS = ('gguf_integrity','runtime_representation','persistent_alternate_copies')


def packet_read(folder, key):
    raw=(folder/'packets'/f'{pin(key)}.json').read_bytes(); need(digest(raw)==key,'audit packet hash changed')
    doc=json.loads(raw); need(doc['policy']==POLICY,'unexpected audit policy')
    raw_source=(folder/'sources'/f"{pin(doc['source_sha256'])}.json").read_bytes()
    need(digest(raw_source)==doc['source_sha256'],'audit candidate source changed')
    return doc


def check_report(doc, key, report):
    need(report.get('packet_sha256') == key and report.get('chosen') == doc['chosen'] and
         report.get('decision') in ('approved','rejected','inconclusive') and
         isinstance(report.get('reviewer'),str) and report['reviewer'].strip(), 'review not bound to packet and candidate')
    checks=report.get('checks'); need(isinstance(checks,dict) and set(checks)==set(CHECKS) and
         all(isinstance(checks[k],str) and checks[k].strip() for k in CHECKS), 'expected independent findings for all three checks')
    pin(report.get('runtime_evidence_sha256'))


def file_digest(path):
    result=sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20),b''): result.update(block)
    return result.hexdigest()


def check_runtime(path, chosen):
    # An external reviewer's observation, not candidate telemetry or an original GGUF hash.
    evidence=json.loads(path.read_bytes()); need(isinstance(evidence,dict) and
        evidence.get('schema')=='q8-runtime-witness-v1' and evidence.get('chosen')==chosen and
        isinstance(evidence.get('method'),str) and evidence['method'].strip(), 'wrong or missing runtime witness binding')
    stages=evidence.get('observations'); need(isinstance(stages,list) and stages and all(isinstance(s,dict) and
        all(isinstance(s.get(k),str) and s[k].strip() for k in ('stage','representation','lifetime','raw_observation'))
        for s in stages) and {'weights_loaded','post_requests'} <= {s['stage'] for s in stages},
        'runtime review requires live load and post-request representation/lifetime observations')


def record_review(folder, key, report, runtime_evidence):
    """Operator command, deliberately not an Agent tool. Archives external runtime evidence by hash."""
    import shutil
    doc=packet_read(folder,key); check_report(doc,key,report)
    folder.mkdir(exist_ok=True); blobs=folder/'runtime-evidence'; blobs.mkdir(exist_ok=True)
    with NamedTemporaryFile(dir=blobs,delete=False) as tmp:
        temp=Path(tmp.name)
        try:
            with runtime_evidence.open('rb') as src: shutil.copyfileobj(src,tmp)
        except BaseException:
            tmp.close();temp.unlink();raise
    try:
        h=report['runtime_evidence_sha256']; need(temp.stat().st_size and file_digest(temp)==h,'runtime evidence missing or changed')
        if report['decision']=='approved': check_runtime(temp,doc['chosen'])
        try: os.link(temp,blobs/h)
        except FileExistsError: need(file_digest(blobs/h)==h,'archived runtime evidence changed')
    finally: temp.unlink()
    raw=wire(report)
    store(folder/'review-history'/f'{digest(raw)}.json',raw)
    if report['decision']=='inconclusive':
        return  # No terminal decision yet. Further independent evidence can be submitted.
    if report['decision']=='rejected':
        chosen=doc['chosen']; store(folder/'rejections'/pin(chosen['engine'])/pin(chosen['build']),wire(dict(rejected=True)))
    store(folder/'reviews'/f'{key}.json',raw)


def approved(folder, key, chosen):
    doc=packet_read(folder,key); path=folder/'reviews'/f'{key}.json'; need(path.is_file(),'independent weight review pending')
    need(not (folder/'rejections'/pin(chosen['engine'])/pin(chosen['build'])).exists(), 'independent reviewer rejected this build')
    raw=path.read_bytes(); report=json.loads(raw);check_report(doc,key,report)
    need(doc['trigger']=='final_nomination' and doc['chosen']==chosen and report['decision']=='approved', 'final weight review not approved for this build')
    blob=folder/'runtime-evidence'/report['runtime_evidence_sha256'];need(blob.is_file() and blob.stat().st_size and file_digest(blob)==blob.name,'runtime evidence missing or changed')
    check_runtime(blob,chosen)
    return report,digest(raw)


def main():
    ap=argparse.ArgumentParser(description='Host-only asynchronous Q8 engineering review; never an Agent RPC.')
    commands=ap.add_subparsers(dest='command',required=True)
    commands.add_parser('pending').add_argument('folder',type=Path)
    record=commands.add_parser('record'); record.add_argument('folder',type=Path); record.add_argument('packet');
    record.add_argument('report',type=Path);record.add_argument('runtime_evidence',type=Path)
    args=ap.parse_args()
    if args.command=='record':
        record_review(args.folder,args.packet,json.loads(args.report.read_bytes()),args.runtime_evidence)
        return
    for path in sorted((args.folder/'packets').glob('*.json')):
        if not (args.folder/'reviews'/path.name).exists():
            doc=packet_read(args.folder,path.stem)
            print(canonical(dict(packet=path.stem, trigger=doc['trigger'], chosen=doc['chosen'], context=doc['context'])).decode())


if __name__ == '__main__':
    main()
