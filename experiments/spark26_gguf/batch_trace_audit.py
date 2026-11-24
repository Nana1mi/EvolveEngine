"""Fail closed if the cold/reuse comparison lacks evidence of actual co-prefill."""
import ast
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

TRIALS = ('cold1', 'cold2', 'reused')
PAIRS = tuple(f'paired-{i}' for i in range(1, 7))
PICKED = re.compile(r'\bpicked-rids=(\[[^\]\r\n]*\])')


def validate_picks(text, stages=PAIRS):
    groups = [tuple(ast.literal_eval(match[1])) for match in PICKED.finditer(text)]
    for stage in stages:
        expected = {f'{stage}-0', f'{stage}-1'}
        associated = [group for group in groups if expected.intersection(group)]
        if len(associated) != 1 or len(associated[0]) != 2 or set(associated[0]) != expected:
            raise ValueError(f'{stage}: missing, split, duplicated, or contaminated schedule pick')


def read_trial(root, name):
    def file(suffix):
        return (root/f'{name}.{suffix}').read_text(encoding='utf-8')

    validate_picks(file('server.log'))
    rows = [json.loads(line) for line in file('stdout').splitlines() if line.startswith('{')]
    requests = {row['rid']: row for row in rows if row.get('kind') == 'request'}
    cache = {row['stage']: row for row in rows if row.get('kind') == 'cache'}
    if len(rows) != 27 or len(requests) != 16 or len(cache) != 10:
        raise ValueError(f'{name}: missing or duplicate probe records')
    flags = [row for row in rows if row.get('kind') == 'token_ids_check']
    if len(flags) != 1:
        raise ValueError(f'{name}: missing token ID comparison')
    pairs = []
    for stage in PAIRS:
        pair = [requests[f'{stage}-{i}'] for i in range(2)]
        for row in pair:
            if row['stage'] != stage or (row['width'], row['prompt_tokens'], row['completion_tokens']) != (2, 130, 16):
                raise ValueError(f'{name}/{stage}: unexpected measured shape')
            for key in ('client_ttft_ms', 'first_token_latency_ms', 'prefill_latency_ms'):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f'{name}/{stage}: missing or invalid {key}')
        pairs.append({'stage': stage, 'rows': pair, 'cache': cache[stage]})
    pre = set(file('precache.txt').splitlines())
    post = set(file('postcache.txt').splitlines())
    if not pre.issubset(post):
        raise ValueError(f'{name}: cache snapshot shrank')
    return dict(pairs=pairs, flags=flags[0], pre=pre, post=post)


def comparison(root):
    readings = {trial: read_trial(root, trial) for trial in TRIALS}
    if readings['cold1']['pre'] or readings['cold2']['pre']:
        raise ValueError('Both cold trials must start with an empty metadata cache')
    if not readings['cold1']['post'] or readings['cold1']['post'] != readings['reused']['pre']:
        raise ValueError('Restart did not reuse exactly the populated cold1 metadata paths')
    out = {}
    for trial, reading in readings.items():
        pairs = reading['pairs']
        first = pairs[0]
        warm = [row for p in pairs[1:] for row in p['rows']]
        keys = ('client_ttft_ms', 'first_token_latency_ms', 'prefill_latency_ms')
        out[trial] = dict(
            first={key: max(r[key] for r in first['rows']) for key in keys},
            repeat_median={key: median(r[key] for r in warm) for key in keys},
            repeat_max={key: max(r[key] for r in warm) for key in keys},
            first_new=first['cache']['new'], repeat_new=[p['cache']['new'] for p in pairs[1:]],
            pre_metadata_count=len(reading['pre']), post_metadata_count=len(reading['post']),
            tokens=reading['flags'], pairs=pairs,
        )
    return out


def main(root):
    out = comparison(root)
    for trial, row in out.items():
        print(f"{trial}: client TTFT {row['first']['client_ttft_ms']:.2f} -> "
              f"{row['repeat_median']['client_ttft_ms']:.2f} ms; prefill "
              f"{row['first']['prefill_latency_ms']:.2f} -> {row['repeat_median']['prefill_latency_ms']:.2f} ms; "
              f"first new JSON: {len(row['first_new'])}; tokens: {row['tokens']}")
    (root/'audited-comparison.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n', encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: batch_trace_audit.py <run directory with cold1/cold2/reused files>')
    main(Path(sys.argv[1]))
