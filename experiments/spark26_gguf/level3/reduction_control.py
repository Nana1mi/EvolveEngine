"""One DEV discriminator: archived reduction binary with the registered E0 argv.

This is a trajectory-only experiment, not a candidate build/quality receipt, a
new admission positive, or an attempt to rescore the archived compound control.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from secrets import token_hex
import sys
from time import monotonic

from experiments.spark26_gguf.level3 import accept_development as ac, control_run as c, node_preflight as n, worker_stdio as w
from evolveengine import v3_level3_trajectory as q
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import Deployment, RuntimeFile, launch_args
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector, candidate_jobs, scripts


def archived(root, build_pin, quality_pin, compound):
    b=c.read_pinned(root/'candidate.json',build_pin)
    r=c.read_pinned(root/'quality.json',quality_pin)
    n.need(b['verified'] is True and r['verified'] is True and
           b['result']['engine']==r['result']['engine']==compound and
           r['result']['status']=='violation' and b['result']['status']=='built' and
           r['result']['build']==b['result']['evidence'] and
           r['result']['data']['trajectory_screen']['failed']==['iso_eos_logp','iso_probability'],
           'not the archived failed compound control')
    public=b['result']; pin=public['evidence']
    jobs=[x for x in (root/'worker').glob('job-*/result.json') if x.is_file() and x.stat().st_size<=65536 and file_sha256(x)==pin]
    n.need(len(jobs)==1,'no unique archived component build')
    job=jobs[0].parent; data=public['data']; deployed=data['deployment']
    n.need(c.read_pinned(jobs[0],pin)==dict({k:v for k,v in public.items() if k!='evidence'},deployment=deployed,
                                         observation_pin=None,observation_unavailable=False), 'archived build row drift')
    component=c.read_pinned(job/'component-build.json',data['component_receipt'])
    n.need(component['status']=='built' and component['candidate']==compound and component['deployment']==deployed and
           data['build_mode']=='component_native','not the registered reduction component')
    folder=job/'deployment'; mf=folder/'manifest.json'
    n.need(mf.resolve(strict=True)==mf and mf.stat().st_size<=24<<20,'unsafe archived manifest')
    raw=mf.read_bytes(); manifest=json.loads(raw)
    n.need(raw==canonical(manifest)+b'\n' and
           compound==manifest['candidate'] and job.name==manifest['build_job'] and
           data['base_wheels']==manifest['wheels'] and data['parent_deployment']==manifest['parent_deployment'],
           'archived runtime manifest mismatch')
    result=Deployment(compound,job.name,tuple(sorted(manifest['wheels'].items())),
                      tuple(RuntimeFile(**x) for x in manifest['files']),folder,manifest['parent_deployment'])
    n.need(result.identity==deployed and not result.runtime_config,'changed deployed artifact identity')
    return result.verify().verify_runtime_access(), b['result']


def run(root, primary, main_pin, history, compound_plan_pin, build_pin, quality_pin, config, config_pin):
    a=c.read_pinned(primary/'controls/acceptance.json',main_pin)
    reg,tr,noise,graph=[a[k] for k in ('registration','trajectory','noise','graph')]
    ac.verify(primary/'controls',a,reg=reg,trajectory_pin=tr,noise_pin=noise,graph_pin=graph)
    result=c.read_pinned(config/'result.json',config_pin)
    plan=c.read_pinned(config/'plan.json',result['plan'])
    row=c.read_pinned(config/'quality.json',result['receipts']['quality'])
    n.need(result['outcome']=='valid' and row['verified'] is True and row['result']['status']=='valid' and
           plan['acceptance']==main_pin and plan['trajectory']==tr and plan['prior_compound_quality']==quality_pin and
           result['engine']==row['result']['engine']==plan['engine'], 'unverified launch-only discriminator')
    compound=c.read_pinned(history/'plan.json',compound_plan_pin)
    n.need(compound['acceptance']==main_pin and compound['runtime']==plan['runtime'], 'compound plan does not match launch-only knobs')
    candidate, build=archived(history,build_pin,quality_pin,compound['engine'])
    cfg=n.checked_inputs(primary/'registration.json',reg)
    contract,e0,backend,svc,lease,collectors,_,_=n.setup_baseline(root,cfg,reg)
    n.need(build['data']['environment_sha256']==backend.environment_sha256 and
           list(launch_args(svc.spec,replace(candidate,runtime_config=canonical(compound['runtime']))))==build['data']['runtime']['argv'] and
           launch_args(svc.spec,candidate)!=tuple(build['data']['runtime']['argv']), 'not the registered build or no argv intervention')
    cal=primary/'trajectory/calibration.json'; cap=cfg['node_ms']['develop_quality']
    quality=collectors['develop_quality']; quality.policy=replace(quality.policy,memory_limit_bytes=None)
    lane=TrajectoryCollector(quality,scripts(n.HERE)); lane.activate(dict(path=str(cal),sha256=tr),reg,e0.digest)
    p=write(root/'plan.json',dict(scope='one_archived_binary_trajectory_discriminator_not_admission',registration=reg,
        acceptance=main_pin,trajectory=tr,configuration_only=config_pin,prior_compound_build=build_pin,
        prior_compound_quality=quality_pin,prior_compound_plan=compound_plan_pin,candidate_deployment=candidate.identity,candidate_argv=list(launch_args(svc.spec,candidate)),
        telemetry_only_memory=True,max_runs=1,retry=False,per_service_cap_ms=cap,
        decision={'valid':'reduction without changed argv does not reproduce the compound trajectory failure in this run',
                  'violation':'report predicates; reduction is sufficient to reproduce the reported predicates only if they match',
                  'incomplete':'preserve evidence; no rerun'},
        driver=file_sha256(Path(__file__)),holdout=False,paid_calls=0))
    worker=w.EngineeringWorker(root/'worker',contract,backend,svc,exclusive=lease,
        collect=quality,settings=cfg,replay_path=primary/'controls/graph/replayed.json',profile_timing=collectors['develop_measure'])
    worker.trajectory_lane=lane
    base=worker('develop_build',e0,None,cfg['node_ms']['develop_build'])
    n.need(base.status=='built' and worker.verify(base),'cannot verify original E0 teacher')
    teacher=worker.trajectory_e0
    n.need(teacher.identity in lane.teacher_deployments and teacher.identity!=candidate.identity,'wrong teacher deployment')
    started=monotonic()
    with lease(started+2*cap/1000):
        deadline=started+cap/1000
        print('candidate trajectory starting',flush=True)
        with svc.start(candidate,'ee-l3-reduction-'+token_hex(8),deadline) as live:
            cp=lane.collect(live,root/'trajectory-candidate',deadline)
        capture,_=lane.read(root/'trajectory-candidate',cp,candidate); tasks=candidate_jobs(capture)
        deadline=min(started+2*cap/1000,monotonic()+cap/1000)
        print('independent E0 teacher starting',flush=True)
        with svc.start(teacher,'ee-l3-reduction-teacher-'+token_hex(8),deadline) as live:
            tp=lane.collect(live,root/'trajectory-teacher',deadline,jobs=tasks)
    checked=lane.screened(root,(cp,tp),candidate,teacher)
    comparison={'reproduces_original_predicates':checked['failed']==['iso_eos_logp','iso_probability']}
    return write(root/'result.json',dict(plan=p,original_archive_candidate=compound['engine'],
        node_ms=round((monotonic()-started)*1000),verdict=checked,comparison=comparison,holdout=False,paid_calls=0))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('report','primary'): cli.add_argument(name,type=Path)
    cli.add_argument('main_pin');cli.add_argument('history',type=Path);cli.add_argument('compound_plan_pin');cli.add_argument('build_pin');cli.add_argument('quality_pin')
    cli.add_argument('config',type=Path);cli.add_argument('config_pin')
    a=cli.parse_args(); os.umask(0o077); root=n.fresh(a.report)
    try:
        print(run(root,a.primary,a.main_pin,a.history,a.compound_plan_pin,a.build_pin,a.quality_pin,a.config,a.config_pin),flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(status='incomplete_no_rerun',reason=str(exc)[:400],kind=type(exc).__name__))
        raise
