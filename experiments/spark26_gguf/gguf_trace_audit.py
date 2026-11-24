"""Summarize separate staged E0 traces without pretending that profiling is timing."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

OPS = ('ggml_mul_mat_a8', 'ggml_mul_mat_vec_a8', 'ggml_dequantize')
FILENAME = re.compile(r'^(short1|long2).*?\b(extend|prefill|decode)\b.*\.trace\.json\.gz$')
HEX = re.compile(r'[0-9a-f]{64}')


def event_arg(event, name):
    return event.get('args', {}).get(name)


def ggml_op(event):
    name = event.get('name', '').partition('::')
    if name[0] == 'sgl_kernel' and name[2].split('.')[0] in OPS:
        return name[2].split('.')[0]
    return None


def summarize_trace(trace):
    ops = Counter()
    external = defaultdict(set)
    launches = defaultdict(set)
    counts = Counter()
    durations = Counter()
    linked = defaultdict(Counter)
    shapes_by_external = defaultdict(set)
    shapes_linked = defaultdict(lambda: [0, 0.0])
    events = trace['traceEvents']
    for event in events:
        if event.get('ph') != 'X':
            continue
        cat, name = event.get('cat'), event.get('name', '')
        if cat == 'cpu_op' and (op := ggml_op(event)):
            ops[op] += 1
            if (ext := event_arg(event, 'External id')) is not None:
                external[ext].add(op)
                dims = event_arg(event, 'Input Dims') or ()
                if op != 'ggml_dequantize' and len(dims) >= 2:
                    weight, activation = dims[:2]
                    if all(isinstance(shape, (list, tuple)) and len(shape) == 2 for shape in (weight, activation)):
                        n, packed_stride = weight
                        m, k = activation
                        if all(type(d) is int and d > 0 for d in (m,n,k,packed_stride)):
                            shapes_by_external[ext].add((op,m,n,k,packed_stride))
        if cat in ('cuda_runtime', 'cuda_driver') and ('Launch' in name or 'launch' in name):
            ext, corr = event_arg(event, 'External id'), event_arg(event, 'correlation')
            if ext is not None and corr is not None:
                launches[corr].add(ext)
    for event in events:
        if (event.get('ph'), event.get('cat')) != ('X', 'kernel'):
            continue
        name, duration = event['name'], event['dur']
        if not math.isfinite(duration) or duration < 0 or not name:
            raise ValueError(f'Invalid GPU event {name!r}')
        counts[name] += 1
        durations[name] += duration
        ids = {*launches.get(event_arg(event, 'correlation'), ()), event_arg(event, 'External id')}
        matches = set().union(*(external.get(i, ()) for i in ids))
        if len(matches) == 1:
            linked[matches.pop()][name] += 1
            shapes = set().union(*(shapes_by_external.get(i, ()) for i in ids))
            if len(shapes) == 1:
                key = (*shapes.pop(), name, tuple(event_arg(event, 'grid') or ()), tuple(event_arg(event, 'block') or ()))
                stat = shapes_linked[key]
                stat[0] += 1
                stat[1] += duration
    if not counts:
        raise ValueError('No CUDA kernel events in staged trace')
    total = sum(durations.values())
    if total <= 0:
        raise ValueError('No positive CUDA duration in staged trace')
    names = sorted(counts, key=lambda k: (-durations[k], k))
    shaped = [dict(op=op, m=m, n=n, k=k, packed_stride=stride, kernel=name, grid=grid, block=block,
                   calls=stat[0], sum_us=stat[1])
              for (op,m,n,k,stride,name,grid,block), stat in sorted(shapes_linked.items())]
    return dict(shape_linked_kernel_launches=shaped, device_properties=trace.get('deviceProperties', []), gpu_kernel_events=sum(counts.values()),
                sum_kernel_us=total, operator_calls=ops, linked_operator_kernels=linked,
                kernels=[dict(name=k, calls=counts[k], sum_us=durations[k], partial_share_pct=100*durations[k]/total)
                         for k in names])


def audit(root):
    index = (root/'profiles.sha256').read_text(encoding='utf-8').splitlines()
    groups = defaultdict(list)
    seen = set()
    for line in index:
        digest, rel = line.split('  ', 1)
        path = Path(rel)
        if not HEX.fullmatch(digest) or path.parts != ('profiles', path.name) or path.name in seen:
            raise ValueError(f'Untrusted or duplicate trace path {rel}')
        match = FILENAME.fullmatch(path.name.lower().replace('_', '-'))
        if not match:
            raise ValueError(f'Cannot establish profile shape and stage for {rel}')
        seen.add(path.name)
        with (root/path).open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != digest:
                raise ValueError(f'Invalid trace checksum: {rel}')
        with gzip.open(root/path, 'rt', encoding='utf-8') as f:
            summary = summarize_trace(json.load(f))
        shape, stage = match.groups()
        stage = 'prefill' if stage == 'extend' else stage
        groups[f'{shape}/{stage}'].append(dict(file=rel, sha256=digest, **summary))
    if set(groups) != {f'{shape}/{stage}' for shape in ('short1','long2') for stage in ('prefill','decode')}:
        raise ValueError('Missing short1/long2 prefill/decode coverage')
    return dict(note='Nested CPU ops or missing correlations can leave kernels unattributed. Summed GPU duration is not device wall time.',
                groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence_dir), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
