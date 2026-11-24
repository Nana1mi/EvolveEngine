"""Audit warmed synthetic E0/E0 evidence; requests are not independent replicates."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

LOCK = Path(__file__).with_name('calibration_workload.json')
PREFILL = re.compile(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),')
MARKER = 'EE_CALIBRATION '
HASH = re.compile(r'[0-9a-f]{64}')


def load_lock(path):
    data = path.read_bytes()
    spec = json.loads(data)
    if spec['schema'] != 1 or spec['purpose'] != 'synthetic_same_image_e0_noise_not_business_baseline':
        raise ValueError('Wrong calibration lock')
    return spec, hashlib.sha256(data).hexdigest()


def schedule(lock, order):
    return [(name, phase, i) for name in order for phase, count in (
        ('warmup', lock['warmup_waves']), ('timed', lock['timed_waves'])) for i in range(count)]


def check_prefills(text, lock, expected):
    observed = []
    for line in text.splitlines():
        if line.startswith(MARKER):
            observed.append((line[len(MARKER):], []))
        elif observed and (m := PREFILL.search(line)):
            observed[-1][1].append(tuple(map(int, m.groups())))
    if len(observed) != len(expected) + 1 or observed[-1] != ('complete', []):
        raise ValueError('Missing, extra, or incomplete calibration markers')
    for (mark, batches), (name, phase, i) in zip(observed, expected):
        w, n = (lock['shapes'][name][k] for k in ('width', 'input_tokens'))
        if mark != f'{name} {phase} wave-{i}' or batches != [(w, w*n, 0)]:
            raise ValueError(f'{mark}: expected exactly one cold-prefix prefill {(w, w*n, 0)}, got {batches}')


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def audit_run(root, lock, digest, label, order):
    expected = schedule(lock, order)
    records = [json.loads(line) for line in (root/f'{label}.stdout').read_text(encoding='utf-8').splitlines()
               if line.startswith('{')]
    if len(records) != len(expected)+1 or records[0] != {
            'kind': 'run_identity', 'label': label, 'workload_sha256': digest}:
        raise ValueError(f'{label}: missing/wrong run identity or record count')
    check_prefills((root/f'{label}.server.log').read_text(encoding='utf-8'), lock, expected)
    timed = defaultdict(list)
    tokens = defaultdict(set)
    for row, (name, phase, i) in zip(records[1:], expected):
        shape = lock['shapes'][name]
        width, n, out = (shape[k] for k in ('width', 'input_tokens', 'output_tokens'))
        if (row['kind'], row['shape'], row['phase'], row['index']) != ('wave', name, phase, i):
            raise ValueError(f'{label}: misplaced record for {name}/{phase}/{i}')
        wall, reqs = row['wall_seconds'], row['requests']
        if len(reqs) != width or not positive(wall):
            raise ValueError(f'{label}/{name}/{phase}/{i}: invalid wave width/time')
        for lane, req in enumerate(reqs):
            if ((req['rid'], req['input_tokens'], req['output_tokens'], req['finish']) != (
                    f'{label}-{name}-{phase}-{i}-{lane}', n, out, 'length')
                    or not HASH.fullmatch(req['output_sha256'])
                    or not positive(req['first_token_ms']) or req['first_token_ms'] > wall*1000):
                raise ValueError(f'{label}/{name}/{phase}/{i}: invalid request {lane}')
            if phase == 'timed':
                tokens[name, lane].add(req['output_sha256'])
        if phase == 'timed':
            timed[name].append((max(r['first_token_ms'] for r in reqs), wall*1000, width*out))
    return timed, tokens


def summarize(root, lock, digest):
    pairs = []
    all_tokens = defaultdict(set)
    per_run = {}
    count = 0
    metrics = ('wave_max_ttft_ms', 'wave_wall_ms', 'wave_output_tps')
    for pair in lock['pairs']:
        for label in pair['labels']:
            timed, tokens = audit_run(root, lock, digest, label, pair['shape_order'])
            count += sum(len(waves)*lock['shapes'][name]['width'] for name, waves in timed.items())
            for key, values in tokens.items():
                all_tokens[key].update(values)
            per_run[label] = {name: dict(zip(metrics, (
                median(x[0] for x in t), median(x[1] for x in t), median(x[2]*1000/x[1] for x in t))))
                for name, t in timed.items()}
        a, b = sorted(pair['labels'])
        values = {name: {k: 200*(per_run[b][name][k]-va)/(per_run[b][name][k]+va)
                         for k, va in per_run[a][name].items()} for name in lock['shapes']}
        pairs.append(dict(run_order=pair['labels'], signed_b_minus_a_pct=values))
    max_abs = {name: {k: max(abs(p['signed_b_minus_a_pct'][name][k]) for p in pairs)
                      for k in metrics} for name in lock['shapes']}
    scopes = {f'{name}/lane{lane}': len(hashes) for (name, lane), hashes in sorted(all_tokens.items())}
    cross_width = {f'{p}/lane0_identical': len(all_tokens[p+'1', 0] | all_tokens[p+'2', 0]) == 1
                   for p in ('short', 'long')}
    return dict(purpose=lock['purpose'], workload_sha256=digest, n_pairs=len(pairs), timed_requests=count,
                method='median within run; signed percent 200*(b-a)/(a+b); descriptive, no production threshold',
                per_run_medians=per_run, pairs=pairs, max_abs_pct=max_abs,
                distinct_timed_token_hashes=scopes, cross_width_diagnostic=cross_width)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence_dir', type=Path)
    p.add_argument('--lock', type=Path, default=LOCK)
    args = p.parse_args()
    lock, digest = load_lock(args.lock)
    print(json.dumps(summarize(args.evidence_dir, lock, digest), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
