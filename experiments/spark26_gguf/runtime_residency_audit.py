"""Validate the diagnostic E0 residence snapshots, not opaque native allocations."""
import argparse
from collections import Counter
import json
from pathlib import Path

from runtime_residency_probe import expected_layout

STAGES = ['after_load', 'ready', 'post_requests']


def audit(log, *, steady_live_tolerance=0):
    snaps = [json.loads(x.removeprefix('EE_RESIDENCY ')) for x in log.splitlines() if x.startswith('EE_RESIDENCY ')]
    if [s['label'] for s in snaps] != STAGES:
        raise ValueError('Missing/duplicated residence checkpoints')
    packed, norms = expected_layout()
    report = []
    for s in snaps:
        inv = s['inventory']
        if s['issues']:
            raise ValueError(f'{s["label"]}: {s["issues"]}')
        if len({v['address'] for v in inv.values()}) != len(inv):
            raise ValueError('Duplicate declared storage')
        expected_roles = {**dict.fromkeys(packed, 'Q8_0'), **dict.fromkeys(norms, 'native_float_norm'),
                          **dict.fromkeys((n+'.qweight_type' for n in packed), 'gguf_type_metadata'),
                          **dict.fromkeys(('shared_rotary_cos_sin_cache_256',
                                           'shared_rotary_cos_sin_cache_64'), 'derived_rope_cache')}
        if {k:v['role'] for k,v in inv.items()} != expected_roles:
            raise ValueError('Wrong declared layout/coverage')
        if inv != snaps[0]['inventory']:
            raise ValueError(f'{s["label"]}: model storage identity changed')
        types = Counter()
        for item in inv.values():
            types[item['role']] += item['bytes']
        if types['Q8_0'] != s['packed_bytes'] or s['packed_bytes'] != sum(n*(k//32)*34 for (n,k), _ in packed.values()):
            raise ValueError('Incorrect total packed weight bytes')
        unknown_bytes = 0
        for storage in s['model_owned_unclassified']:
            for info in storage:
                if ((info['dtype'], info['shape'], info['bytes']) != ('torch.int32', [1], 4)
                        or not info['path'].endswith("_group']['active_ranks']")):
                    raise ValueError(f'{s["label"]}: unattributed model-owned storage {info}')
            unknown_bytes += storage[0]['bytes']
        report.append(dict(stage=s['label'], storage_bytes_by_role=types,
                           known_parallel_rank_bytes=unknown_bytes, allocator=s['allocator'],
                           allocator_live_bytes_minus_observed_storage=s['allocator']['live_bytes']-sum(types.values())-unknown_bytes))
    if abs(snaps[1]['allocator']['live_bytes'] - snaps[2]['allocator']['live_bytes']) > steady_live_tolerance:
        raise ValueError('Steady live allocated bytes changed; investigate before approving')
    return dict(packed_weights=len(packed), native_norms=len(norms), snapshots=report,
                scope='Known Python model-owned tensor storage and Torch CUDA allocator; opaque/native and transient buffers are not proven by this audit.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('server_log', type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.server_log.read_text(encoding='utf-8')), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
