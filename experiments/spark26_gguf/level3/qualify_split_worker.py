"""Qualify the registered split gate through the production development Worker.

Runs one E0 observation, one legal source/config positive and one native
compute negative. Never contacts a model, opens Q/T, or retries a case.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path

from experiments.spark26_gguf.level3 import legal_variant_control, live_split_quality as split, native_controls
from experiments.spark26_gguf.level3 import node_preflight as n, worker_stdio as w
from evolveengine.v3_level3_collect import QUALITY_CASES, write
from evolveengine.v3_level3_deploy import launch_args
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import SPLIT_CASES, SPLIT_PROTOCOL, trajectory_cases

SCHEMA='spark26-live-split-worker-controls-v1'
BF16_SCHEMA='spark26-bf16-live-split-worker-controls-v1'
ORDER=('base-build','base-quality','positive-build','positive-quality','fault-build','fault-quality')
BF16_ORDER=('base-build','base-quality','positive-build','positive-quality','positive-measure',
            'fault-build','fault-quality','lowprec-build','lowprec-quality')


def run(root, baseline_path, authorization_path, authorization_pin, *, previous_acceptance):
    reg,lease,scope,controls=w.authorization(authorization_path,authorization_pin)
    ref=controls['split_quality']; cal=controls['trajectory_calibration']['sha256']
    registered=w.split_registration(controls,reg); cfg=n.checked_inputs(baseline_path,reg,split_registration=registered)
    n.need(scope==w.ENGINEERING_EXPLORATION and lease.jobs is None and lease.node_ms is None,
           'split qualification needs the production metered Worker scope')
    contract,e0,worker=w.make_worker(n.fresh(root/'worker'),cfg,reg,lease,authorization_pin,scope=scope,controls=controls)
    positive=legal_variant_control.candidate(contract); fault=native_controls.make(contract)['native_compute_negative']
    proof=split.read(Path(ref['path']).parent/'compound-plan.json',split.PROOFS['compound-plan'])
    n.need(positive.digest==proof['engine'] and previous_acceptance==proof['acceptance'] and
           worker.trajectory_e0.identity==registered['teacher'], 'control engine or independent teacher changed')
    engines={'base':e0,'positive':positive,'fault':fault}; pins={}; rows={}
    plan=write(root/'plan.json',dict(schema=SCHEMA,driver=file_sha256(Path(__file__)),baseline=reg,calibration=cal,
        previous_acceptance=previous_acceptance,split_registration=ref['sha256'],authorization=authorization_pin,
        observer=registered['observer'],teacher=registered['teacher'],order=ORDER,
        identities={k:eng.digest for k,eng in engines.items()},retry=False,holdout=False,paid_calls=0))
    for name in ORDER:
        arm,action=name.split('-'); engine=engines[arm]; op='develop_'+action
        parent=rows.get(arm+'-build') if action=='quality' else rows.get('base-build')
        cap=lease.max_per_op[op]; write(root/(name+'-reserved.json'),dict(operation=op,engine=engine.digest,
            build=parent.evidence if parent else None,cap=cap))
        print('starting',name,flush=True); result=worker(op,engine,parent,cap); verified=worker.verify(result)
        print('settled',name,result.status,result.node_ms,verified,flush=True)
        pins[name]=write(root/(name+'.json'),dict(verified=verified,result=asdict(result)))
        rows[name]=result
        check(name,result.status,result.data,verified,registered,ref['sha256'],cal)
        if name=='positive-build':
            n.need(list(launch_args(worker.service.spec,worker.deployments[result.evidence]))==proof['candidate_argv'],
                   'worker positive did not activate pinned configuration')
    return write(root/'acceptance.json',dict(schema=SCHEMA,plan=plan,receipts=pins,holdout=False,paid_calls=0))


def check(name,status,data,verified,registered,split_pin,cal):
    expected='built' if name.endswith('-build') else 'violation' if name=='fault-quality' else 'valid'
    n.need(verified is True and status==expected,'unexpected '+name+' result; no replacement')
    if name.endswith('-build'): return
    row=data['trajectory_screen']
    n.need(row['calibration']==cal and row['teacher_deployment']==registered['teacher'] and row['observer']==registered['observer'] and
           row['split_registration']==split_pin and split.pin(row['evidence_summary']) and data.get('weight_observation') and
           set(data['case_outcomes'])==set((*QUALITY_CASES[:2],*SPLIT_CASES)) and 'sampling_screen' not in data and
           all(data['case_outcomes'][k]==v for k,v in trajectory_cases(row,SPLIT_PROTOCOL).items()), 'wrong split quality binding')
    if name=='fault-quality':
        n.need(data['case_outcomes']['numeric_32']=='fail','native correctness fault was not observed')
    else:
        n.need(set(data['case_outcomes'].values())=={'pass'}, 'positive lacked complete witnesses')


def verify(root, acceptance, *, baseline, cal, previous_acceptance, split_pin, registered):
    """Controller side. Every summarized status is linked to its WorkerResult."""
    n.need(set(acceptance)=={'schema','plan','receipts','holdout','paid_calls'} and acceptance['schema']==SCHEMA and
           acceptance['holdout'] is False and acceptance['paid_calls']==0 and set(acceptance['receipts'])==set(ORDER) and
           len(set(acceptance['receipts'].values()))==len(ORDER),'unclosed split controls')
    p=split.read(root/'plan.json',acceptance['plan'])
    n.need(p['schema']==SCHEMA and p['driver']==file_sha256(Path(__file__)) and p['baseline']==baseline and p['calibration']==cal and
           p['previous_acceptance']==previous_acceptance and p['split_registration']==split_pin and
           p['observer']==registered['observer'] and p['teacher']==registered['teacher'] and p['order']==list(ORDER) and
           p['retry'] is p['holdout'] is False and p['paid_calls']==0 and set(p['identities'])=={'base','positive','fault'} and
           len(set(p['identities'].values()))==3, 'split hardware control plan changed')
    started={}; services=set()
    proof=split.read(root.parent/'split'/'compound-plan.json',split.PROOFS['compound-plan'])
    n.need(p['identities']['positive']==proof['engine'], 'different hardware control candidate')
    for name in ORDER:
        d=split.read(root/(name+'.json'),acceptance['receipts'][name]); r=d['result']; arm,action=name.split('-')
        n.need(r['operation']=='develop_'+action and r['engine']==p['identities'][arm] and 0<r['node_ms'] and
               split.pin(r['evidence']),'invalid split hardware control receipt')
        check(name,r['status'],r['data'],d['verified'],registered,split_pin,cal)
        if action=='quality':
            row=r['data']['trajectory_screen']
            n.need(r['build']==started[arm] and row['engine']==r['engine'] and row['service'] not in services and
                   all(split.pin(row[k]) for k in ('candidate_receipt','teacher_receipt')) and
                   row['candidate_receipt']!=row['teacher_receipt'], 'split quality signed the wrong build or service')
            services.add(row['service'])
        else:
            n.need(r['build'] is None, 'degenerate control build')
            started[arm]=r['evidence']
            if arm=='positive':
                n.need(r['data']['runtime']['argv']==proof['candidate_argv'], 'split control startup differed')
    return True


def run_bf16(root, baseline_path, authorization_path, authorization_pin):
    """Unified BF16 Worker qualification: E0, legal BF16 positive+measure, CUDA fault, FP8 copy audit."""
    from experiments.spark26_gguf.level3 import bf16_controls
    from evolveengine import v3_level3_weight_audit as weights

    reg, lease, scope, controls = w.authorization(authorization_path, authorization_pin)
    ref = controls['split_quality']
    cal = controls['trajectory_calibration']['sha256']
    registered = w.split_registration(controls, reg)
    n.need(registered is not None and registered.get('schema') == split.BF16_SCHEMA,
           'BF16 qualification requires a BF16 split registration')
    cfg = n.checked_inputs(baseline_path, reg, split_registration=registered)
    n.need(scope == w.ENGINEERING_EXPLORATION and lease.jobs is None and lease.node_ms is None
           and controls.get('weight_audit') == weights.BF16_POLICY,
           'BF16 qualification needs metered BF16 engineering Worker scope')
    contract, e0, worker = w.make_worker(n.fresh(root/'worker'), cfg, reg, lease, authorization_pin,
                                         scope=scope, controls=controls)
    has_long_ctx = 'anchor_long_ctx' in cfg
    engines = bf16_controls.make(contract, include_long_ctx=has_long_ctx)
    order = (
        (*BF16_ORDER, 'longctx_pos-build', 'longctx_pos-quality', 'longctx_fault-build', 'longctx_fault-quality')
        if has_long_ctx else BF16_ORDER
    )
    n.need(engines['base'] == e0 and worker.trajectory_e0.identity == registered['teacher'],
           'BF16 control engine or independent teacher changed')
    pins, rows = {}, {}
    plan = write(root/'plan.json', dict(
        schema=BF16_SCHEMA, driver=file_sha256(Path(__file__)), baseline=reg, calibration=cal,
        split_registration=ref['sha256'], authorization=authorization_pin,
        observer=registered['observer'], teacher=registered['teacher'], order=order,
        identities={k: eng.digest for k, eng in engines.items()}, retry=False, holdout=False, paid_calls=0))
    for name in order:
        arm, action = name.split('-')
        engine = engines[arm]
        op = 'develop_' + action
        parent = (rows.get(arm + '-build') if action != 'build'
                  else rows.get('positive-build') if arm == 'fault'
                  else rows.get('base-build') if arm != 'base' else None)
        cap = lease.max_per_op[op]
        write(root/(name + '-reserved.json'), dict(operation=op, engine=engine.digest,
              build=parent.evidence if parent else None, cap=cap))
        print('starting', name, flush=True)
        if action == 'measure':
            result = worker.measure(engine, parent, cap, 0)
        else:
            result = worker(op, engine, parent, cap)
        verified = worker.verify(result)
        print('settled', name, result.status, result.node_ms, verified, flush=True)
        pins[name] = write(root/(name + '.json'), dict(verified=verified, result=asdict(result)))
        rows[name] = result
        check_bf16(name, result.status, result.data, verified, registered, ref['sha256'], cal, has_long_ctx=has_long_ctx)
    scan = weights.static_scan(contract.base, engines['lowprec'].edits)
    n.need(any('conversion_near_persistent_storage' in x['hints'] for x in scan['locations']),
           'BF16 low-precision weight negative was not flagged by static scan')
    scan_pin = write(root/'lowprec-scan.json', scan)
    return write(root/'acceptance.json', dict(
        schema=BF16_SCHEMA, plan=plan, receipts=pins, lowprec_scan=scan_pin, holdout=False, paid_calls=0))


def check_bf16(name, status, data, verified, registered, split_pin, cal, *, has_long_ctx=False):
    from evolveengine.v3_level3_collect import LONG_CTX_CASE
    expected = ('built' if name.endswith('-build') else 'observed' if name.endswith('-measure')
                else 'violation' if name in ('fault-quality', 'longctx_fault-quality') else 'valid')
    n.need(verified is True and status == expected, 'unexpected ' + name + ' result; no replacement')
    if name.endswith('-build'):
        return
    if name.endswith('-measure'):
        n.need(data['output_authentication']['status'] == 'verified', 'BF16 positive measure lacked output auth')
        from types import SimpleNamespace
        from evolveengine.v3_level3_run import compare_dev
        from evolveengine.v3_level3_scoring import WORKBUDDY_V2_REPLAY_SCENES
        if set(data.get('scenes') or {}) == set(WORKBUDDY_V2_REPLAY_SCENES):
            stub = SimpleNamespace(operation='develop_measure', status='observed', evidence='0' * 64, data=data)
            comp = compare_dev(stub, stub)
            n.need(set(comp.get('guardrail_ratios') or {}) == {'w4_cold_e2e', 'g64k_ttft', 'g64k_decode', 'p95_itl'},
                   'BF16 WorkBuddy v2 public measure receipt missing verified guardrails')
        return
    row = data['trajectory_screen']
    expected_cases = set((*QUALITY_CASES[:2], *SPLIT_CASES, *((LONG_CTX_CASE,) if has_long_ctx else ())))
    n.need(row['calibration'] == cal and row['teacher_deployment'] == registered['teacher']
           and row['observer'] == registered['observer'] and row['split_registration'] == split_pin
           and split.pin(row['evidence_summary']) and data.get('weight_observation')
           and set(data['case_outcomes']) == expected_cases
           and 'sampling_screen' not in data
           and all(data['case_outcomes'][k] == v for k, v in trajectory_cases(row, SPLIT_PROTOCOL).items()),
           'wrong BF16 split quality binding')
    if name == 'fault-quality':
        n.need(data['case_outcomes']['numeric_32'] == 'fail'
               and data['case_outcomes']['autoregressive_full_vocab'] == 'fail',
               'BF16 native correctness fault was not caught')
    elif name == 'longctx_fault-quality':
        n.need(data['case_outcomes'][LONG_CTX_CASE] == 'fail',
               'BF16 long-context fault was not caught by 16K/64K logprob gate')
    else:
        n.need(set(data['case_outcomes'].values()) == {'pass'}, name + ' lacked complete witnesses')


def verify_bf16(root, acceptance, *, baseline, cal, split_pin, registered, has_long_ctx=False):
    """Controller side verification for unified BF16 split controls."""
    order = (
        (*BF16_ORDER, 'longctx_pos-build', 'longctx_pos-quality', 'longctx_fault-build', 'longctx_fault-quality')
        if has_long_ctx else BF16_ORDER
    )
    expected_ids = {'base', 'positive', 'fault', 'lowprec', *(('longctx_pos', 'longctx_fault') if has_long_ctx else ())}
    n.need(set(acceptance) == {'schema', 'plan', 'receipts', 'lowprec_scan', 'holdout', 'paid_calls'}
           and acceptance['schema'] == BF16_SCHEMA and acceptance['holdout'] is False
           and acceptance['paid_calls'] == 0 and set(acceptance['receipts']) == set(order)
           and len(set(acceptance['receipts'].values())) == len(order), 'unclosed BF16 split controls')
    p = split.read(root/'plan.json', acceptance['plan'])
    n.need(p['schema'] == BF16_SCHEMA and p['driver'] == file_sha256(Path(__file__))
           and p['baseline'] == baseline and p['calibration'] == cal and p['split_registration'] == split_pin
           and p['observer'] == registered['observer'] and p['teacher'] == registered['teacher']
           and p['order'] == list(order) and p['retry'] is p['holdout'] is False and p['paid_calls'] == 0
           and set(p['identities']) == expected_ids
           and len(set(p['identities'].values())) == len(expected_ids), 'BF16 split control plan changed')
    started, services = {}, set()
    for name in order:
        d = split.read(root/(name + '.json'), acceptance['receipts'][name], cap=2 << 20)
        r = d['result']
        arm, action = name.split('-')
        n.need(r['operation'] == 'develop_' + action and r['engine'] == p['identities'][arm]
               and 0 < r['node_ms'] and split.pin(r['evidence']), 'invalid BF16 control receipt')
        check_bf16(name, r['status'], r['data'], d['verified'], registered, split_pin, cal, has_long_ctx=has_long_ctx)
        if action == 'quality':
            row = r['data']['trajectory_screen']
            n.need(r['build'] == started[arm] and row['engine'] == r['engine'] and row['service'] not in services
                   and all(split.pin(row[k]) for k in ('candidate_receipt', 'teacher_receipt'))
                   and row['candidate_receipt'] != row['teacher_receipt'], 'BF16 quality signed wrong build/service')
            services.add(row['service'])
        elif action == 'measure':
            n.need(r['build'] == started[arm], 'BF16 measure signed wrong build')
        else:
            n.need(r['build'] is None, 'degenerate BF16 control build')
            started[arm] = r['evidence']
            if arm == 'positive':
                argv = r['data']['runtime']['argv']
                n.need('--disable-radix-cache' not in argv and '--dtype' in argv
                       and argv[argv.index('--dtype') + 1] == 'bfloat16'
                       and argv[argv.index('--kv-cache-dtype') + 1] == 'bfloat16',
                       'BF16 positive control startup differed')
    scan = split.read(root/'lowprec-scan.json', acceptance['lowprec_scan'])
    n.need(any('conversion_near_persistent_storage' in x['hints'] for x in scan['locations']),
           'BF16 low-precision weight negative scan missing flag')
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); p.add_argument('baseline',type=Path)
    p.add_argument('authorization',type=Path); p.add_argument('authorization_pin')
    p.add_argument('previous_acceptance', nargs='?')
    p.add_argument('--bf16', action='store_true')
    a=p.parse_args();os.umask(0o077);root=n.fresh(a.root)
    try:
        if a.bf16:
            print(run_bf16(root,a.baseline,a.authorization,a.authorization_pin),flush=True)
        else:
            n.need(a.previous_acceptance, 'previous_acceptance required for Q8_0 split qualification')
            print(run(root,a.baseline,a.authorization,a.authorization_pin,previous_acceptance=a.previous_acceptance),flush=True)
    except BaseException as exc:
        write(root/'incomplete.json',dict(kind=type(exc).__name__,reason=str(exc)[:360]));raise
