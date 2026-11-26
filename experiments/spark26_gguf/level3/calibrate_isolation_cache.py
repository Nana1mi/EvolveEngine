"""E0-only calibration of cold-armed cancellation probes (no scoring or model API).

Use the archived independent E0 wheels, two launch configurations, and no
candidate metrics in the calibration. A missing witness stops the matrix.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import os
from secrets import token_hex
from time import monotonic
from types import SimpleNamespace

from experiments.spark26_gguf.level3 import control_run as c, node_preflight as n, replay_e0, repeat_e0
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import source_bytes, write
from evolveengine.v3_level3_deploy import launch_args, stage_deployment
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector, candidate_jobs, scripts

SCHEMA='spark26-e0-isolation-cold-matrix-v1'
KNOBS={'cuda-graph-bs-decode':[1,2,3,4,5,6],'max-running-requests':6,'radix-cache':True}
ORDER=(('registered','e0a','e0b'),('radix_bs6','e0b','e0a'),('radix_bs6','e0a','e0b'),('registered','e0b','e0a'))


def instrument(source):
    patches=(
        (b'from threading import Barrier\n',b'from threading import Barrier\nfrom isolation_cache_parity import ColdPairs\n'),
        (b'    decode=[]; isolation=[]\n',b'    decode=[]; isolation=[]; cold=ColdPairs(post)\n'),
        (b'            gate=Barrier(2)\n',b'            gate=Barrier(2); survivor_scope,neighbor_scope=cold.arm(trigger,arm)\n'),
        (b'gate,survivor=True)',b'gate,survivor=True,call=survivor_scope.call)'),
        (b"trigger=trigger if arm=='cancel' else None)",b"trigger=trigger if arm=='cancel' else None,call=neighbor_scope.call)"),
        (b'decode=decode,isolation=isolation)',b'decode=decode,isolation=isolation,cache_parity=cold.witness(q.TRIGGERS))'),
    )
    for anchor,new in patches:
        if source.count(anchor)!=1: raise ValueError('frozen trajectory observer anchor changed')
        source=source.replace(anchor,new,1)
    return source


def verify_cache(capture):
    rows=capture.get('cache_parity')
    ordered=[(k,a) for k in q.TRIGGERS for a in ('cancel','complete')]
    q.need(type(rows) is list and len(rows)==len(ordered),'cold parity was not captured')
    for row,(k,a) in zip(rows,ordered,strict=True):
        q.need(type(row) is dict and set(row)=={'trigger','arm','cached_tokens_initial'} and
               (row['trigger'],row['arm'])==(k,a) and type(row['cached_tokens_initial']) is list and
               len(row['cached_tokens_initial'])==2 and all(type(x) is int and x==0 for x in row['cached_tokens_initial']),
               'unverified initial cache state')
    return rows


class ColdLane(TrajectoryCollector):
    def __init__(self, quality, paths, helper):
        super().__init__(quality,paths)
        self.parent_code=self.code
        self.files['observe_trajectory.py']=instrument(self.files['observe_trajectory.py'])
        self.files['isolation_cache_parity.py']=source_bytes(helper)
        self.code=digest(canonical({k:digest(v) for k,v in self.files.items()}))

    def read(self, root, pin, deployment, *, jobs=None):
        data,binding=super().read(root,pin,deployment,jobs=jobs)
        if jobs is None: verify_cache(data)
        return data,binding

    def activate(self,*args,**kwargs):
        raise ValueError('calibration observer cannot sign a live verdict')

    def screened(self,*args,**kwargs):
        raise ValueError('calibration observer cannot sign a live verdict')


def compare(candidate, teacher, *, require_e0_ceiling=True):
    reference=q.compare(candidate,teacher)
    if require_e0_ceiling:
        q.need(reference['maxima']['decode_tv']<=.05,'E0 absolute Decode TV exceeds the existing ceiling')
    jobs=dict(zip((j['key'] for j in q.teacher_plan(candidate)),teacher['rows'],strict=True))
    planned={j['key']:j for j in q.teacher_plan(candidate)}
    pts=[]
    for pair in candidate['isolation']:
        k=pair['trigger']
        for i in q.TAIL:
            a,b=actual(pair,'complete',i,k,planned,jobs),actual(pair,'cancel',i,k,planned,jobs)
            same_draw=a.id==b.id
            pts.append(dict(trigger=k,step=i,same_prefix=a.prefix==b.prefix,same_draw=same_draw,
               abs_marker_probability=max(abs(x) for x in (*a.markers,*b.markers)),
               abs_draw_probability=max(abs(a.draw),abs(b.draw)),abs_eos_logp=max(abs(a.eos),abs(b.eos)),
               paired_marker_probability=max(abs(x-y) for x,y in zip(a.markers,b.markers,strict=True)),
               paired_draw_probability=abs(a.draw-b.draw) if same_draw else None,
               paired_eos_logp=abs(a.eos-b.eos)))
    return dict(reference=reference,points=pts,cache=verify_cache(candidate))


def actual(pair, arm, i, k, planned, jobs):
    from math import exp
    key=f'i/{k}/{arm}/{i}'; s=pair[arm][0]; j=planned[key]; draw=s['ids'][i]
    ref=dict(zip(j['wanted'],jobs[key]['values'],strict=True)); scored=s['probs'][i]
    return SimpleNamespace(prefix=j['ids'],id=draw,draw=exp(scored[-1])-exp(ref[draw]),
       markers=[exp(p)-exp(ref[t]) for t,p in zip(q.MARKERS,scored[:-1],strict=True)],
       eos=scored[q.MARKERS.index(q.EOS)]-ref[q.EOS])


def run(root, primary, acceptance, helper, helper_pin):
    a=c.read_pinned(primary/'controls/acceptance.json',acceptance)
    reg,v2,noise,graph=[a[k] for k in ('registration','trajectory','noise','graph')]
    cal=c.read_pinned(primary/'trajectory/calibration.json',v2)
    frozen=c.read_pinned(primary/'trajectory/frozen-plan.json',cal['plan_sha256'])
    summary=c.read_pinned(primary/'noise/result.json',noise,2<<20)
    old=c.read_pinned(primary/'noise-plan.json',summary['plan_sha256'])
    q.need(old['build_quality_measure_pins']==frozen['builds'] and old['baseline_sha256']==frozen['baseline_sha256'] and
        old['build_registration']['sha256']==frozen['build_registration_sha256'] and
        old['registration_sha256']==frozen['registration_sha256']==reg and
        summary['stable'] is True and file_sha256(helper)==helper_pin,'inputs differ from archived E0 registration')
    cfg=n.checked_inputs(primary/'registration.json',reg)
    baseline=Path(old['baseline']); prior_ref=old['build_registration']; prior_path=Path(prior_ref['path'])
    prior=c.read_pinned(prior_path,prior_ref['sha256']); replay_e0.check_build_compatibility(cfg,prior)
    history=repeat_e0.draft(cfg,reg,baseline,old['baseline_sha256'],build_registration=prior_ref)
    q.need(history['build_quality_measure_pins']==frozen['builds'],'E0 build pins changed')
    contract,e0,backend,svc,exclusive,collectors,_,_=n.setup_baseline(root,cfg,reg)
    quality=collectors['develop_quality'];quality.policy=replace(quality.policy,memory_limit_bytes=None)
    lane=ColdLane(quality,scripts(n.HERE),helper)
    q.need(cal['code']==lane.parent_code,'parent live trajectory observer changed')
    arms={}
    for label in ('e0a','e0b'):
        fp=history['build_quality_measure_pins'][label]['develop_build']
        native,row=replay_e0.restore_build(baseline.parent/f'{label}-develop_build.json',fp,contract=contract,e0=e0,
            backend=backend,registration=prior,registration_pin=prior_ref['sha256'],arm=label)
        d=stage_deployment(native,root/(label+'-deployment'))
        q.need(d.identity==row['data']['deployment']==cal['deployments'][label],'wrong independent E0 deployment')
        arms[label]=d
    q.need(arms['e0a'].identity!=arms['e0b'].identity,'need both independent E0 builds')
    configs={'registered':b'','radix_bs6':canonical(KNOBS)}
    argvs={k:list(launch_args(svc.spec,replace(arms['e0a'],runtime_config=v))) for k,v in configs.items()}
    cap=cfg['node_ms']['develop_quality']
    plan=write(root/'frozen-plan.json',dict(schema=SCHEMA,scope='E0_only_non_scoring_no_admission',acceptance=acceptance,
        registration=reg,trajectory_v2=v2,noise=noise,graph=graph,driver=file_sha256(Path(__file__)),observer=lane.code,
        observer_parent=lane.parent_code,parity_helper=helper_pin,order=ORDER,argvs=argvs,deployments=cal['deployments'],
        per_service_cap_ms=cap,retry=False,holdout=False,paid_calls=0,
        decision={'all_four_verified':'record descriptive E0 noise by configuration, do not issue a passing threshold',
                  'different_draws':'record missing draw residual, never impute zero or require token identity',
                  'any_incomplete':'stop without a calibration result; no substitutions'}))
    results=[]
    for mode,trial,teacher in ORDER:
        folder=root/(mode+'-'+trial);folder.mkdir(mode=0o700);start=monotonic()
        d,t=replace(arms[trial],runtime_config=configs[mode]),arms[teacher]
        print('dispatch',mode,trial,teacher,flush=True)
        write(folder/'reserved.json',dict(plan=plan,configuration=mode,candidate=trial,teacher=teacher))
        with exclusive(start+2*cap/1000):
            deadline=start+cap/1000
            with svc.start(d,'ee-l3-isolation-cold-'+token_hex(8),deadline) as live:
                cp=lane.collect(live,folder/'trajectory-candidate',deadline)
            cand,cb=lane.read(folder/'trajectory-candidate',cp,d); jobs=candidate_jobs(cand)
            deadline=min(start+2*cap/1000,monotonic()+cap/1000)
            with svc.start(t,'ee-l3-isolation-teach-'+token_hex(8),deadline) as live:
                tp=lane.collect(live,folder/'trajectory-teacher',deadline,jobs=jobs)
        teach,tb=lane.read(folder/'trajectory-teacher',tp,t,jobs=jobs)
        q.need(cb['service_id']!=tb['service_id'] and cb['run_id']!=tb['run_id'] and cb['end_ns']<=tb['start_ns'],
               'not independent sequential E0 services')
        metrics=compare(cand,teach)
        pin=write(folder/'metrics.json',dict(plan=plan,configuration=mode,candidate=trial,teacher=teacher,
                   candidate_pin=cp,teacher_pin=tp,node_ms=round((monotonic()-start)*1000),metrics=metrics))
        results.append(dict(configuration=mode,candidate=trial,teacher=teacher,sha256=pin))
        print('settled',mode,trial,pin,flush=True)
    return write(root/'result.json',dict(schema=SCHEMA,plan=plan,status='E0_observations_only',runs=results,
                                        gate_active=False,paid_calls=0,holdout=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('primary',type=Path);p.add_argument('acceptance')
    p.add_argument('helper',type=Path);p.add_argument('helper_pin');a=p.parse_args()
    os.umask(0o077); root=n.fresh(a.root)
    try: print(run(root,a.primary,a.acceptance,a.helper,a.helper_pin),flush=True)
    except BaseException as exc:
        write(root/'incomplete.json',dict(status='incomplete_no_calibration',kind=type(exc).__name__,reason=str(exc)[:320]));raise
