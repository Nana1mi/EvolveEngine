"""Compare QKV-only deterministic service output and built-in API quality checks to E0."""
import argparse
import json
from pathlib import Path

from deterministic_probe import ROUNDS, audit
from reference_pair_audit import checked_manifest

LOCK = Path(__file__).with_name('qkv_service_workload.json')
RAW_FILES = {f'{label}.{suffix}' for label in ROUNDS for suffix in ('stdout', 'server.log')}


def verify_candidate_artifacts(root):
    raw_hashes = checked_manifest(root, 'raw-evidence.sha256', RAW_FILES)
    if (root/'post-compute.csv').read_text(encoding='utf-8').strip():
        raise ValueError('GPU compute state not clear at deterministic teardown')
    return raw_hashes


def read_semantic_gate(log, label, wanted=None):
    markers = [line for line in log.splitlines() if line.startswith('EE_REFERENCE_QKV_QUALITY ')]
    expect = [] if wanted is None else [f'EE_REFERENCE_QKV_QUALITY {label} {wanted}']
    if markers != expect:
        raise ValueError(f'{label}: wrong or missing deterministic source identity')
    records = [json.loads(line) for line in log.splitlines() if line.startswith('{') and '"kind": "quality' in line]
    stats = [r for r in records if r.get('kind')=='quality_summary']
    rows = [r for r in records if r.get('kind')=='quality']
    cases = {r['case'] for r in rows}
    if (stats != [dict(kind='quality_summary',passed=6,total=6)] or len(rows)!=6 or len(cases)!=6
        or not all(r['passed'] is True for r in rows)):
        raise ValueError(f'{label}: incomplete application API acceptance records')
    return cases


def compare(e0, candidate, lock):
    raw_hashes = verify_candidate_artifacts(candidate)
    refs = audit(e0)
    check = audit(candidate)
    for key in ('passed','inputs','output_groups','verified_requests','verified_prefill_waves'):
        if refs[key] != check[key]:
            raise ValueError(f'Deterministic output differs in {key}')
    if not check['passed'] or (check['verified_requests'], check['verified_prefill_waves']) != (36, 24):
        raise ValueError('Deterministic source results unstable or incomplete')
    for label in ROUNDS:
        def read(p):
            return (p/f'{label}.server.log').read_text(encoding='utf-8')
        if read_semantic_gate(read(e0),label) != read_semantic_gate(read(candidate),label, lock['source_sha256']['b']):
            raise ValueError('Different built-in API quality cases')
    return dict(verified_native_requests=check['verified_requests'], verified_native_prefill_waves=check['verified_prefill_waves'],
                candidate_api_cases_passed=6*len(ROUNDS), candidate_sha256=lock['source_sha256']['b'],
                raw_evidence_manifest=raw_hashes,
                scope='Bounded diagnostic deterministic output evidence; not a production-quality threshold.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('e0',type=Path)
    p.add_argument('candidate',type=Path)
    a=p.parse_args()
    print(json.dumps(compare(a.e0,a.candidate,json.loads(LOCK.read_bytes())),indent=2,sort_keys=True))


if __name__=='__main__':
    main()
