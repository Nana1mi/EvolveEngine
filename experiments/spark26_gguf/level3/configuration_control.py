"""One DEV-only discriminator: same E0 binary with the mixed-control launch knobs."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from experiments.spark26_gguf.level3 import accept_development as ac, control_run as c, node_preflight as n, worker_stdio as w
from evolveengine import v3_level3_eos as eos
from evolveengine.v3_harness import canonical
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_startup import PATH
from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector,scripts

KNOBS={'cuda-graph-bs-decode':[1,2,3,4,5,6],'max-running-requests':6,'radix-cache':True}


def run(root, primary, acceptance, prior_quality, prior_pin):
    a=c.read_pinned(primary/'controls/acceptance.json',acceptance)
    reg,tr,ng,g=[a[k] for k in ('registration','trajectory','noise','graph')]
    ac.verify(primary/'controls',a,reg=reg,trajectory_pin=tr,noise_pin=ng,graph_pin=g)
    failed=c.read_pinned(prior_quality,prior_pin)
    n.need(failed['verified'] is True and failed['result']['status']=='violation' and
           failed['result']['data']['trajectory_screen']['failed']==['iso_eos_logp','iso_probability'],
           'not the frozen failed compound control')
    cfg=n.checked_inputs(primary/'registration.json',reg)
    tref=primary/'trajectory/calibration.json'; graph=primary/'controls/graph/replayed.json'
    c.read_pinned(tref,tr);c.read_pinned(graph,g)
    repeat=c.read_pinned(primary/'noise/result.json',ng,2<<20)
    plan=c.read_pinned(primary/'noise-plan.json',repeat['plan_sha256'])
    contract,e0,backend,svc,lease,collectors,_,_=n.setup_baseline(root,cfg,reg)
    engine=contract.from_args({'files':{PATH:canonical(KNOBS).decode()}})
    p=write(root/'plan.json',dict(scope='one_config_only_discriminator_not_a_replacement_control',acceptance=acceptance,
         prior_compound_quality=prior_pin,registration=reg,trajectory=tr,graph=g,noise=ng,runtime=KNOBS,
         engine=engine.digest,driver=file_sha256(Path(__file__)),max_runs=1,retry=False,
         decision={'valid':'this launch-only change does not reproduce the compound failure',
                   'violation':'identify actual failing predicates; do not change tolerance from either result',
                   'insufficient_evidence':'incomplete; stop'},holdout=False,paid_calls=0))
    q=collectors['develop_quality'];q.policy=replace(q.policy,memory_limit_bytes=None)
    timing=collectors['develop_measure'];n.need(timing.enable_natural_eos(plan['chat_workload'])==repeat['timing_workload_sha256'],'workload drift')
    lane=TrajectoryCollector(q,scripts(n.HERE));lane.activate(dict(path=str(tref),sha256=tr),reg,e0.digest)
    q.enable_trajectory(tr,lane.code,lambda rows:eos.grade(rows,Path(cfg['anchor_d3']),Path(cfg['anchor_r1']),paired=True))
    worker=w.EngineeringWorker(root/'worker',contract,backend,svc,exclusive=lease,
      collect=lambda op,*args,**kwargs:collectors[op](op,*args,**kwargs),settings=cfg,replay_path=graph,profile_timing=timing)
    worker.trajectory_lane=lane
    build=None; refs={}
    for label,op,current in (('base','develop_build',e0),('config','develop_build',engine),('quality','develop_quality',engine)):
        cap=cfg['node_ms'][op];write(root/(label+'-reserved.json'),dict(operation=op,engine=current.digest,cap=cap,build=build.evidence if build else None))
        r=worker(op,current,build,cap); ok=worker.verify(r); refs[label]=write(root/(label+'.json'),dict(result=asdict(r),verified=ok))
        n.need(ok, 'unverified configuration discriminator'); print(label,r.status,r.node_ms,flush=True)
        if label=='base':
            n.need(r.status=='built','no archived E0 wheel');build=r
        if label=='config':
            n.need(r.status=='built' and r.data['build_mode']=='component_source_only' and r.data['base_wheels']==build.data['wheels'],
                   'configuration changed the E0 binary')
            parent,child=worker.deployments[build.evidence],worker.deployments[r.evidence]
            n.need(parent.files==child.files and parent.wheels==child.wheels and child.runtime_config==canonical(KNOBS),
                   'configuration is not an exact E0 artifact comparison');build=r
    return write(root/'result.json',dict(plan=p,receipts=refs,outcome=r.status,engine=engine.digest,paid_calls=0,holdout=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('report',type=Path);cli.add_argument('primary',type=Path)
    cli.add_argument('acceptance');cli.add_argument('compound_quality',type=Path);cli.add_argument('compound_pin')
    a=cli.parse_args();os.umask(0o077);root=n.fresh(a.report)
    try: print(run(root,a.primary,a.acceptance,a.compound_quality,a.compound_pin),flush=True)
    except BaseException as e:write(root/'incomplete.json',dict(kind=type(e).__name__,detail=str(e)[:350]));raise
