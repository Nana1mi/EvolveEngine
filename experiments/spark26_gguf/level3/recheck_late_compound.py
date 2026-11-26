"""One real replay of the archived compound positive under the frozen split probe.

No source rebuild, paid model call, or admission. The historical v2 failure is
preserved; new evidence prospectively judges only this new run.
"""
import argparse
from dataclasses import replace
import os
from pathlib import Path
from secrets import token_hex
from time import monotonic

from experiments.spark26_gguf.level3 import control_run as c, late_trajectory as late, node_preflight as n
from experiments.spark26_gguf.level3 import reduction_control, replay_e0, split_quality
from experiments.spark26_gguf.level3.calibrate_late_trajectory import SCHEMA as E0_SCHEMA
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_harness import canonical
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import launch_args, stage_deployment
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import candidate_jobs, scripts

SCHEMA='spark26-late-quality-archived-compound-recheck-v1'


def controls_pass(replay):
    n.need(replay['scope']=='offline_e0_feasibility_and_fault_validation_not_live_admission' and
           replay['holdout'] is False and replay['paid_calls']==0 and
           replay['metric_rule']==file_sha256(Path(split_quality.__file__)) and
           replay['checks']['baseline']['status']=='valid','E0 late positive not verified')
    bad=replay['checks']['synthetic_controls']
    for label,group in (('common_eos','mixed_tail_absolute'),('paired_eos','paired_cancel_numeric')):
        n.need(set(bad[label]['groups'])==set(split_quality.GROUPS) and all(x['status']==
               ('violation' if k==group else 'valid') for k,x in bad[label]['groups'].items()),
               'E0 offline late controls did not separate the failure groups')


def run(root, primary, e0_source, e0_pin, replay_path, replay_pin, history, plan_pin, build_pin, quality_pin, helper):
    e0_obs=c.read_pinned(e0_source/'result.json',e0_pin,2<<20)
    e0_plan=c.read_pinned(e0_source/'frozen-plan.json',e0_obs['plan'])
    replay=c.read_pinned(replay_path,replay_pin)
    controls_pass(replay)
    n.need(replay['source']==e0_pin and replay['plan']==e0_obs['plan'] and e0_plan['schema']==e0_obs['schema']==E0_SCHEMA and
           e0_obs['observe']['absolute_tail_tv']=='certified' and e0_obs['observe']['live_gate'] is False and
           e0_obs['holdout'] is e0_plan['holdout'] is e0_plan['retry'] is False and
           e0_obs['paid_calls']==e0_plan['paid_calls']==0 and e0_plan['metric']==file_sha256(Path(late.__file__)) and
           e0_plan['bounds']==file_sha256(Path(late.partial_tv.__file__)) and
           e0_plan['helper']==file_sha256(helper),'late observer differs from E0 positive')
    a=c.read_pinned(primary/'controls/acceptance.json',e0_plan['acceptance'])
    reg=a['registration']; cal=c.read_pinned(primary/'trajectory/calibration.json',a['trajectory'])
    n.need(cal['limits']==q.envelope([r['metrics'] for r in cal['arms']]) and
           reg==cal['registration_sha256']==e0_plan['registration'] and a['trajectory']==replay['trajectory_v2'],
           'wrong independently bound quality calibration')
    cmp_plan=c.read_pinned(history/'plan.json',plan_pin)
    n.need(cmp_plan['acceptance']==e0_plan['acceptance'] and cmp_plan['trajectory']==a['trajectory'] and
           cmp_plan['runtime']==late.cold.KNOBS and cmp_plan['retry'] is cmp_plan['holdout'] is False,
           'wrong archived variant')
    archived,build=reduction_control.archived(history,build_pin,quality_pin,cmp_plan['engine'])
    variant=replace(archived,runtime_config=canonical(cmp_plan['runtime']))
    noise=c.read_pinned(primary/'noise/result.json',a['noise'],2<<20)
    old=c.read_pinned(primary/'noise-plan.json',noise['plan_sha256'])
    br=old['build_registration']; prior=c.read_pinned(Path(br['path']),br['sha256']); baseline=Path(old['baseline'])
    cfg=n.checked_inputs(primary/'registration.json',reg); replay_e0.check_build_compatibility(cfg,prior)
    contract,e0,backend,svc,lock,collectors,_,_=n.setup_baseline(root,cfg,reg)
    quality=collectors['develop_quality'];quality.policy=replace(quality.policy,memory_limit_bytes=None)
    lane=late.TailLane(quality,scripts(n.HERE),helper)
    n.need(lane.code==e0_plan['observer'] and lane.parent_code==cal['code'] and
           build['data']['environment_sha256']==backend.environment_sha256 and
           list(launch_args(svc.spec,variant))==build['data']['runtime']['argv']==e0_plan['candidate_argv'],
           'variant observer, original launch args, or runtime changed')
    label='e0b'; artifact,rec=replay_e0.restore_build(baseline.parent/f'{label}-develop_build.json',old['build_quality_measure_pins'][label]['develop_build'],
        contract=contract,e0=e0,backend=backend,registration=prior,registration_pin=br['sha256'],arm=label)
    teacher=stage_deployment(artifact,root/'teacher-deployment')
    n.need(teacher.identity==rec['data']['deployment']==cal['deployments'][label]==e0_plan['teacher'] and
           list(launch_args(svc.spec,teacher))==e0_plan['teacher_argv'],'teacher deployment changed')
    cap=cfg['node_ms']['develop_quality']
    p=write(root/'frozen-plan.json',dict(schema=SCHEMA,acceptance=e0_plan['acceptance'],calibration_v2=a['trajectory'],
       e0_late=e0_pin,e0_controls=replay_pin,observer=lane.code,archived_plan=plan_pin,archived_build=build_pin,archived_quality=quality_pin,
       candidate=variant.identity,teacher=teacher.identity,engine=variant.candidate,scope='archived_build_only_no_admission',
       candidate_argv=e0_plan['candidate_argv'],teacher_argv=e0_plan['teacher_argv'],metric_rule=replay['metric_rule'],
       driver=file_sha256(Path(__file__)),top_k=late.TOP_K,per_service_cap_ms=cap,retry=False,paid_calls=0,holdout=False,
       conditions=['affirm only observed late/early metrics; retain original v2 failure',
                   'report insufficient evidence separately; no automatic reruns or threshold revision']))
    start=monotonic()
    with lock(start+2*cap/1000):
        deadline=start+cap/1000
        print('archived compound variant starting',flush=True)
        with svc.start(variant,'ee-l3-late-variant-'+token_hex(8),deadline) as live:
            cp=lane.collect(live,root/'trajectory-candidate',deadline)
        candidate,cb=lane.read(root/'trajectory-candidate',cp,variant); jobs=candidate_jobs(candidate)
        for pair in candidate['isolation']:
            for arm in ('complete','cancel'):
                for step in q.TAIL: late.view(pair[arm][0],step)
        deadline=min(start+2*cap/1000,monotonic()+cap/1000)
        print('independent E0 teacher for compound starting',flush=True)
        with svc.start(teacher,'ee-l3-late-variant-teach-'+token_hex(8),deadline) as live:
            tp=lane.collect(live,root/'trajectory-teacher',deadline,jobs=jobs)
    observed,tb=lane.read(root/'trajectory-teacher',tp,teacher,jobs=jobs)
    n.need(cb['service_id']!=tb['service_id'] and cb['run_id']!=tb['run_id'] and cb['end_ns']<=tb['start_ns'],
           'candidate and teacher were not independent')
    report=late.tail_report(candidate,observed)
    assessment=split_quality.assess(report,cal['limits'])
    return write(root/'result.json',dict(schema=SCHEMA,plan=p,candidate_receipt=cp,teacher_receipt=tp,
       scope='archived_build_only_no_admission',observation=report,assessment=assessment,
       node_ms=round((monotonic()-start)*1000),paid_calls=0,holdout=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','primary','e0_source'): p.add_argument(k,type=Path)
    p.add_argument('e0_pin');p.add_argument('replay_path',type=Path);p.add_argument('replay_pin');p.add_argument('history',type=Path)
    for k in ('plan_pin','build_pin','quality_pin'): p.add_argument(k)
    p.add_argument('helper',type=Path);a=p.parse_args();args=vars(a);os.umask(0o077);args['root']=n.fresh(a.root)
    try: print(run(**args),flush=True)
    except BaseException as exc:
        write(a.root/'incomplete.json',dict(status='incomplete_no_live_admission',kind=type(exc).__name__,reason=str(exc)[:360]));raise
