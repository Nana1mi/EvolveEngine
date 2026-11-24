"""Check shape coverage, exact-output witnesses and residency for one candidate diagnostic."""
import argparse
import json
from pathlib import Path
import re

from runtime_residency_audit import audit as residency
from synthetic_baseline_audit import audit_run, load_lock, LOCK

SHAPES = (1, 2, 130, 260, 768, 1536)
LAYERS = ('QKVParallelLinear', 'MergedColumnParallelLinear')
EXACT = re.compile(r'^EE_COALESCE_EXACT (\w+) M=(\d+)$')


def audit(candidate, e0_residency_log, e0_calibration, expected_layers=LAYERS):
    lock, digest = load_lock(LOCK)
    log = (candidate/'residency.server.log').read_text(encoding='utf-8')
    candidate_residency = residency(log)
    expected_residency = residency(e0_residency_log.read_text(encoding='utf-8'))
    for a, b in zip(expected_residency['snapshots'], candidate_residency['snapshots']):
        for metric in ('storage_bytes_by_role', 'known_parallel_rank_bytes'):
            if a[metric] != b[metric]:
                raise ValueError(f'{a["stage"]}: different {metric}')
        # Cache reservation may differ with the shadow operation's temporaries.
        if a['allocator']['live_bytes'] != b['allocator']['live_bytes']:
            raise ValueError(f'{a["stage"]}: live allocated bytes changed versus E0')
    witnesses = [(m[1], int(m[2])) for line in log.splitlines() if (m := EXACT.fullmatch(line))]
    if len(witnesses) != len(expected_layers)*len(SHAPES) or set(witnesses) != {(layer, m) for layer in expected_layers for m in SHAPES}:
        raise ValueError(f'Missing, duplicated or out-of-scope shadow shape coverage: {witnesses}')
    e0 = {}
    for pair in lock['pairs']:
        for label in pair['labels']:
            _, tokens = audit_run(e0_calibration, lock, digest, label, pair['shape_order'])
            for key, options in tokens.items():
                e0.setdefault(key, set()).update(options)
    rows = [json.loads(line) for line in (candidate/'residency.stdout').read_text(encoding='utf-8').splitlines()
            if line.startswith('{')]
    if len(rows) != len(lock['shapes']) or {r['shape'] for r in rows} != set(lock['shapes']):
        raise ValueError('Missing or duplicated workload shapes')
    checked = 0
    for r in rows:
        shape, outputs = r['shape'],r['outputs']
        if r['kind'] != 'residency_workload' or len(outputs) != lock['shapes'][shape]['width']:
            raise ValueError(f'Invalid workload record: {r}')
        for lane, output in enumerate(outputs):
            if {output} != e0[shape, lane]:
                raise ValueError(f'Candidate output differs from calibrated E0 at {shape}/{lane}')
            checked += 1
    return dict(exact_shadow_checks=witnesses, distinct_workload_shapes=len(rows),
                candidate_requests_with_e0_token_match=checked, candidate_residency=candidate_residency,
                scope='One shadow/probe run. No speed, general quality or blind acceptance conclusion.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path)
    p.add_argument('e0_residency_log', type=Path)
    p.add_argument('e0_calibration', type=Path)
    p.add_argument('--qkv-only', action='store_true')
    a = p.parse_args()
    layers = ('QKVParallelLinear',) if a.qkv_only else LAYERS
    print(json.dumps(audit(a.candidate, a.e0_residency_log, a.e0_calibration, layers), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
