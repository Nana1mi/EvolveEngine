"""Verify pinned offline diagnostic, then evaluate prespecified QKV decision."""
import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median

from reference_pair_audit import checked_manifest

EXPECTED = {'qkv_micro_lock.json','qkv_micro.py','QKV_MICRO_PLAN.md','spark2_5_gguf.py','run_qkv_micro_diagnostic.sh'}


def read_data(records, lock, lock_digest):
    evidence = iter(records)
    identity = next(evidence, {})
    if ((identity.get('kind'),identity.get('lock_sha256'),identity.get('source_sha256'),identity.get('capability'))
        != ('identity',lock_digest,lock['source_sha256'],[12,1]) or identity.get('gpu') != 'NVIDIA GB10'):
        raise ValueError('Wrong or missing runtime identity')
    results = {}
    observations = 0
    exact_checks = 0
    for projection,shape_order in (('qkv',lock['shape_order']),('gate_up_control',lock['negative_control_shape_order'])):
        sub = results[projection] = {}
        for m in shape_order:
            if next(evidence, {}) != dict(kind='exact', projection=projection, m=m, layers=lock['layers']):
                raise ValueError(f'{projection}/{m}: missing full exact comparisons')
            exact_checks += len(lock['layers'])
            pairs = []
            for idx,order in enumerate(lock['sample_orders']):
                row = next(evidence, {})
                if (row.get('kind'),row.get('projection'),row.get('m'),row.get('index'),row.get('order')) != ('sample',projection,m,idx,order):
                    raise ValueError(f'{projection}/{m}: wrong, missing or duplicated sample')
                sample = row['ms_per_call']
                if set(sample) != {'a','b'} or not all(type(x) in (int,float) and isfinite(x) and x>0 for x in sample.values()):
                    raise ValueError(f'{projection}/{m}: invalid timing')
                a,b = (sample[k] for k in 'ab')
                pairs.append(dict(a_ms=a,b_ms=b,saving_pct=100*(a-b)/a))
                observations += 1
            sub[m] = dict(median_saving_pct=median(r['saving_pct'] for r in pairs), favorable_pairs=sum(r['saving_pct']>0 for r in pairs),
                          split_median_ms=median(r['a_ms'] for r in pairs), joined_median_ms=median(r['b_ms'] for r in pairs), pairs=pairs)
    if next(evidence,{}) != dict(kind='complete') or next(evidence,None) is not None:
        raise ValueError('Missing completion, or extra records')
    q = results['qkv']
    decode = all(q[m]['median_saving_pct']>lock['minimum_decode_saving_pct'] and
                 q[m]['favorable_pairs']>=lock['minimum_decode_favorable_pairs'] for m in (1,2))
    prefill = all(q[m]['median_saving_pct']>=-lock['maximum_prefill_slowdown_pct'] for m in q if m not in (1,2))
    decision = 'all_tested_rows' if decode and prefill else 'decode_rows_only' if decode else 'do_not_advance'
    return dict(results=results, decision=decision, sample_pairs=observations, bf16_bit_exact_checks=exact_checks,
                checked_decode_gate=decode, checked_prefill_guard=prefill, device=identity['gpu'],
                scope='No serving speed claim. The gate selects a follow-up candidate only.')


def audit(root, bundle_digest):
    if sha256((root/'bundle.sha256').read_bytes()).hexdigest() != bundle_digest:
        raise ValueError('Bundle differs from pre-run commitment')
    files = checked_manifest(root,'bundle.sha256',EXPECTED)
    raw = checked_manifest(root,'raw-evidence.sha256',{'micro.stdout','micro.stderr'})
    if (root/'run.closed').read_text(encoding='utf-8').strip().split('\t')[-1]!='0' or (root/'post-compute.csv').read_text(encoding='utf-8').strip():
        raise ValueError('Run not cleanly closed')
    lock = json.loads((root/'qkv_micro_lock.json').read_bytes())
    if lock['schema']!=1 or lock['purpose']!='qkv_only_vs_split_operator_diagnostic_not_service_benchmark':
        raise ValueError('Invalid study lock')
    records = [json.loads(line) for line in (root/'micro.stdout').read_text(encoding='utf-8').splitlines() if line.startswith('{')]
    return dict(bundle_sha256=bundle_digest, raw_evidence=raw, **read_data(records,lock,files['qkv_micro_lock.json']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence_dir',type=Path)
    p.add_argument('--bundle-sha256',required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.evidence_dir,a.bundle_sha256),indent=2,sort_keys=True))


if __name__=='__main__':
    main()
