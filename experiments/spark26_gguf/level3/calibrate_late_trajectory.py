"""One E0-only hardware feasibility check for the cold late-decode TV probe."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
from secrets import token_hex
from time import monotonic

from experiments.spark26_gguf.level3 import calibrate_isolation_cache as cold, control_run as c
from experiments.spark26_gguf.level3 import late_trajectory as late, node_preflight as n, replay_e0
from experiments.spark26_gguf.level3.summarize_isolation_cache import summary
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_harness import canonical
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import launch_args, stage_deployment
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import candidate_jobs, scripts

SCHEMA='spark26-late-cold-full-teacher-e0-feasibility-v1'


def matrix_evidence(root, pin, primary, acceptance, helper_pin):
    matrix=c.read_pinned(root/'result.json',pin)
    plan=c.read_pinned(root/'frozen-plan.json',matrix['plan'])
    n.need(matrix['schema']==plan['schema']==cold.SCHEMA and matrix['status']=='E0_observations_only' and
        matrix['gate_active'] is False and matrix['holdout'] is plan['holdout'] is plan['retry'] is False and
        matrix['paid_calls']==plan['paid_calls']==0 and plan['acceptance']==acceptance and plan['parity_helper']==helper_pin and
        plan['order']==[list(x) for x in cold.ORDER] and len(matrix['runs'])==len(cold.ORDER) and
        plan['driver']==file_sha256(Path(cold.__file__)),'not the completed E0 cold matrix')
    rows=[]; pins=set()
    for item,(config,trial,teacher) in zip(matrix['runs'],cold.ORDER,strict=True):
        row=c.read_pinned(root/(config+'-'+trial)/'metrics.json',item['sha256'])
        n.need((item['configuration'],item['candidate'],item['teacher'])==(config,trial,teacher) and
               row['plan']==matrix['plan'] and type(row['node_ms']) is int and row['node_ms']>0 and
               c._sha(row['candidate_pin']) and c._sha(row['teacher_pin']),'incorrectly bound E0 cold observations')
        rows.append(row); pins.update((row['candidate_pin'],row['teacher_pin']))
    n.need(len(pins)==2*len(cold.ORDER),'E0 cold captures were reused')
    a=c.read_pinned(primary/'controls/acceptance.json',acceptance)
    n.need((a['registration'],a['trajectory'],a['noise'],a['graph'])==tuple(plan[k] for k in ('registration','trajectory_v2','noise','graph')),
           'E0 matrix from the wrong DEV registration')
    return a,plan,summary(rows)


def run(root, primary, acceptance, matrix, matrix_pin, helper, helper_pin):
    n.need(file_sha256(helper)==helper_pin,'initial-cache observer changed')
    a,matrix_plan,matrix_summary=matrix_evidence(matrix,matrix_pin,primary,acceptance,helper_pin)
    reg=a['registration']; cal=c.read_pinned(primary/'trajectory/calibration.json',a['trajectory'])
    noise=c.read_pinned(primary/'noise/result.json',a['noise'],2<<20)
    plan=c.read_pinned(primary/'noise-plan.json',noise['plan_sha256'])
    prior_ref=plan['build_registration']; baseline=Path(plan['baseline']); prior=c.read_pinned(Path(prior_ref['path']),prior_ref['sha256'])
    cfg=n.checked_inputs(primary/'registration.json',reg); replay_e0.check_build_compatibility(cfg,prior)
    contract,e0,backend,svc,lock,collectors,_,_=n.setup_baseline(root,cfg,reg)
    quality=collectors['develop_quality'];quality.policy=replace(quality.policy,memory_limit_bytes=None)
    lane=late.TailLane(quality,scripts(n.HERE),helper)
    n.need(lane.parent_code==cal['code'] and lane.cold_code==matrix_plan['observer'],'wrong parent controls or cold instrumentation')
    builds={}
    for arm in ('e0a','e0b'):
        artifact,rec=replay_e0.restore_build(baseline.parent/f'{arm}-develop_build.json',plan['build_quality_measure_pins'][arm]['develop_build'],
            contract=contract,e0=e0,backend=backend,registration=prior,registration_pin=prior_ref['sha256'],arm=arm)
        demo=stage_deployment(artifact,root/(arm+'-deployment'))
        q.need(demo.identity==rec['data']['deployment']==cal['deployments'][arm]==matrix_plan['deployments'][arm],
               'wrong E0 wheel or deployment')
        builds[arm]=demo
    cap=cfg['node_ms']['develop_quality']; k=replace(builds['e0a'],runtime_config=canonical(cold.KNOBS)); t=builds['e0b']
    n.need(k.identity!=t.identity and tuple(launch_args(svc.spec,k))==tuple(matrix_plan['argvs']['radix_bs6']) and
        tuple(launch_args(svc.spec,t))==tuple(matrix_plan['argvs']['registered']), 'late probe altered E0 launch conditions')
    plan_pin=write(root/'frozen-plan.json',dict(schema=SCHEMA,scope='E0_observation_only',matrix=matrix_pin,matrix_summary=matrix_summary,
       registration=reg,acceptance=acceptance,observer=lane.code,cold_observer=lane.cold_code,helper=helper_pin,
       driver=file_sha256(Path(__file__)), metric=file_sha256(Path(late.__file__)),bounds=file_sha256(Path(late.partial_tv.__file__)),
       candidate=k.identity,teacher=t.identity,candidate_argv=matrix_plan['argvs']['radix_bs6'],teacher_argv=matrix_plan['argvs']['registered'],
       top_k=late.TOP_K,target=late.TV_LIMIT,normalization_slack=late.NORMALIZATION_SLACK,retry=False,
       max_trials=1,per_service_cap_ms=cap,holdout=False,paid_calls=0,
       decision={'certified':'this E0 case supplies enough late distribution mass for the absolute TV claim',
          'inconclusive':'record coverage and report insufficient evidence, never reinterpret as pass',
          'different':'do not accept late probe on E0', 'pairwise_norm':'observation only, issue no new cancellation thresholds'}))
    start=monotonic()
    with lock(start+2*cap/1000):
        deadline=start+cap/1000
        print('E0 radix survivor starting',flush=True)
        with svc.start(k,'ee-l3-late-e0-'+token_hex(8),deadline) as agent:
            cp=lane.collect(agent,root/'trajectory-candidate',deadline)
        cand,cb=lane.read(root/'trajectory-candidate',cp,k); jobs=candidate_jobs(cand)
        for pair in cand['isolation']:
            for arm in ('complete','cancel'):
                for step in q.TAIL: late.view(pair[arm][0],step)
        deadline=min(start+2*cap/1000,monotonic()+cap/1000)
        print('independent E0 full-vocabulary teacher starting',flush=True)
        with svc.start(t,'ee-l3-late-teacher-'+token_hex(8),deadline) as teacher:
            tp=lane.collect(teacher,root/'trajectory-teacher',deadline,jobs=jobs)
    observed,tb=lane.read(root/'trajectory-teacher',tp,t,jobs=jobs)
    q.need(cb['service_id']!=tb['service_id'] and cb['run_id']!=tb['run_id'] and cb['end_ns']<=tb['start_ns'],
           'not independently served full-vocabulary evidence')
    report=late.tail_report(cand,observed)
    return write(root/'result.json',dict(schema=SCHEMA,scope='one_e0_feasibility_no_live_gate',plan=plan_pin,
        candidate_receipt=cp,teacher_receipt=tp,node_ms=round((monotonic()-start)*1000),observe=report,
        paid_calls=0,holdout=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('primary',type=Path);p.add_argument('acceptance')
    p.add_argument('matrix',type=Path);p.add_argument('matrix_pin');p.add_argument('helper',type=Path);p.add_argument('helper_pin')
    a=p.parse_args(); os.umask(0o077); root=n.fresh(a.root)
    try: print(run(root,a.primary,a.acceptance,a.matrix,a.matrix_pin,a.helper,a.helper_pin),flush=True)
    except BaseException as exc:
        write(root/'incomplete.json',dict(status='incomplete_no_live_gate',kind=type(exc).__name__,reason=str(exc)[:360]));raise
