"""Fail-closed audit and engineering-screen decision for the pinned synthetic AB diagnostic."""
import argparse
from datetime import datetime
from hashlib import file_digest, sha256
import json
from math import exp, log
from pathlib import Path
import re
from statistics import median

from synthetic_baseline_audit import summarize

LOCK_PATH = Path(__file__).with_name('reference_pair_workload.json')
HEX = re.compile(r'[0-9a-f]{64}')
PURPOSE = 'e0_vs_python_coalesce_synthetic_diagnostic'
ORDER = ('a1','b1','b2','a2','b3','a3','a4','b4')
BUNDLE_NAMES = {'REFERENCE_PAIR_PLAN.md', 'reference_pair_workload.json', 'reference_pair_entry.sh',
                'reference_coalesce.py', 'synthetic_baseline.py', 'run_reference_pair_diagnostic.sh'}


def checked_manifest(root, index, names):
    manifest = {}
    for line in (root/index).read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if not HEX.fullmatch(digest) or name not in names or name in manifest:
            raise ValueError(f'Unknown, duplicate or tampered manifest entry: {line}')
        with (root/name).open('rb') as f:
            if file_digest(f, 'sha256').hexdigest() != digest:
                raise ValueError(f'Hash mismatch: {name}')
        manifest[name] = digest
    if set(manifest) != names:
        raise ValueError(f'Incomplete {index}')
    return manifest


def signed_symmetric_pct(ratio):
    return 200*(ratio-1)/(ratio+1)


def screen(lock, summary):
    thresholds = lock['screen']
    expected_request_count = 2*len(lock['pairs'])*lock['timed_waves']*sum(s['width'] for s in lock['shapes'].values())
    if (summary['n_pairs'], summary['timed_requests']) != (len(lock['pairs']), expected_request_count):
        raise ValueError('Incomplete valid process pairs')
    gain = []
    for p in summary['pairs']:
        a,b = sorted(p['run_order'])
        ar, br = (summary['per_run_medians'][k] for k in (a,b))
        ratio = exp(sum(log(br[s]['wave_output_tps']/ar[s]['wave_output_tps'])
                        for s in lock['shapes'])/len(lock['shapes']))
        gain.append(signed_symmetric_pct(ratio))
    by_shape = {s: {k: median(p['signed_b_minus_a_pct'][s][k] for p in summary['pairs'])
                   for k in ('wave_output_tps', 'wave_max_ttft_ms', 'wave_wall_ms')} for s in lock['shapes']}
    token_counts = summary['distinct_timed_token_hashes']
    token_scopes = {f'{name}/lane{n}' for name, shape in lock['shapes'].items() for n in range(shape['width'])}
    checks = {'primary_median_exceeds_min': median(gain) > thresholds['primary_min_exclusive_pct'],
              'positive_pairs': sum(g > 0 for g in gain) >= thresholds['minimum_positive_pairs'],
              'all_timed_tokens_unchanged': set(token_counts)==token_scopes and all(v == 1 for v in token_counts.values())}
    for shape, effect in by_shape.items():
        checks[f'{shape}_throughput_guard'] = effect['wave_output_tps'] >= -thresholds['tps_median_regression_limit_pct'][shape]
        checks[f'{shape}_ttft_guard'] = effect['wave_max_ttft_ms'] <= thresholds['ttft_median_regression_limit_pct'][shape]
    return dict(engineering_screen_passed=all(checks.values()), checks=checks, pair_primary_signed_pct=gain,
                primary_median_signed_pct=median(gain), shape_median_signed_pct=by_shape, thresholds=thresholds)


def audit(root, expected_bundle_hash):
    bundle_hash = sha256((root/'bundle.sha256').read_bytes()).hexdigest()
    if bundle_hash != expected_bundle_hash:
        raise ValueError('Bundle differs from the pre-timing commitment')
    qkv_variant = (root/'qkv_service_workload.json').is_file()
    if qkv_variant:
        if (root/LOCK_PATH.name).exists():
            raise ValueError('Ambiguous workload')
        names = BUNDLE_NAMES-{'REFERENCE_PAIR_PLAN.md',LOCK_PATH.name,'reference_coalesce.py'}
        names |= {'QKV_SERVICE_PLAN.md','qkv_service_workload.json','reference_qkv.py'}
    else:
        names = BUNDLE_NAMES
    files = checked_manifest(root, 'bundle.sha256', names)
    evidence_hashes = checked_manifest(root, 'raw-evidence.sha256',
        {f'{label}.{part}' for label in ORDER for part in ('stdout','server.log','pre-gpu.csv','post-gpu.csv')})
    workload = root/('qkv_service_workload.json' if qkv_variant else LOCK_PATH.name)
    lock = json.loads(workload.read_bytes())
    order = tuple(label for p in lock['pairs'] for label in p['labels'])
    purpose = 'e0_vs_qkv_only_synthetic_diagnostic' if qkv_variant else PURPOSE
    if lock['schema'] != 1 or lock['purpose'] != purpose or order != ORDER:
        raise ValueError('Invalid reference-pair workload')
    events = [line.split('\t') for line in (root/'run-order.tsv').read_text(encoding='utf-8').splitlines()]
    if [(stage, label) for stage,label,ts in events] != [(s,l) for l in order for s in ('start','done')]:
        raise ValueError('Missing, duplicated or reordered start/done evidence')
    times = [datetime.fromisoformat(ts) for _,_,ts in events]
    closed, exit_code = (root/'run.closed').read_text(encoding='utf-8').strip().split('\t')
    if times != sorted(times) or exit_code != '0' or datetime.fromisoformat(closed) < times[-1]:
        raise ValueError('Run chronology/termination is invalid')
    if (root/'post-compute.csv').read_text(encoding='utf-8').strip():
        raise ValueError('GPU compute state not clear at teardown')
    for label in order:
        log_text = (root/f'{label}.server.log').read_text(encoding='utf-8')
        rows = [l for l in log_text.splitlines() if l.startswith('EE_REFERENCE_PAIR ')]
        want = f'EE_REFERENCE_PAIR {label} {lock["source_sha256"][label[0]]}'
        if rows != [want] or re.search(r'^EE_(?:RESIDENCY|COALESCE_EXACT) ', log_text, re.MULTILINE):
            raise ValueError(f'{label}: wrong arm code or unexpected diagnostic probes')
    report = summarize(root, lock, files[workload.name])
    return dict(bundle_sha256=bundle_hash, raw_evidence_manifest=evidence_hashes,
                elapsed_s=(times[-1]-times[0]).total_seconds(),
                outcome=screen(lock, report), measurements=report,
                note='Synthetic same-node engineering screen; no significance test or production conclusion.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence_dir', type=Path)
    p.add_argument('--bundle-sha256', required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.evidence_dir, args.bundle_sha256), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
