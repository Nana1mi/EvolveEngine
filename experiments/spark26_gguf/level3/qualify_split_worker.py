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
ORDER=('base-build','base-quality','positive-build','positive-quality','fault-build','fault-quality')


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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path); p.add_argument('baseline',type=Path)
    p.add_argument('authorization',type=Path); p.add_argument('authorization_pin');p.add_argument('previous_acceptance')
    a=p.parse_args();os.umask(0o077);root=n.fresh(a.root)
    try: print(run(root,a.baseline,a.authorization,a.authorization_pin,previous_acceptance=a.previous_acceptance),flush=True)
    except BaseException as exc:
        write(root/'incomplete.json',dict(kind=type(exc).__name__,reason=str(exc)[:360]));raise
