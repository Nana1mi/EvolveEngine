"""Engineering calibration only: autonomous E0 decode vs one-token E0 prefill on its prefixes.

Use two previously authenticated, independently cold-built E0 arms. No model API,
no holdout. A missing witness or unstable E0 produces no tolerance file or retry.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from secrets import token_hex
from time import monotonic

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from experiments.spark26_gguf.level3 import node_preflight as node, control_run, replay_e0, repeat_e0
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import stage_deployment
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector, scripts
from evolveengine import v3_level3_trajectory as q


def run(root,settings,registration,pair,pair_pin,prior_path,prior_pin):
    prior=control_run.read_pinned(prior_path,prior_pin)
    history=repeat_e0.draft(settings,registration,pair,pair_pin,build_registration=dict(path=str(prior_path),sha256=prior_pin))
    cap=settings['node_ms']['develop_quality']
    plan=write(root/'frozen-plan.json',dict(protocol=q.VERSION,registration_sha256=registration,
        build_registration_sha256=prior_pin,baseline_sha256=pair_pin,builds=history['build_quality_measure_pins'],
        order=[['e0a','e0b'],['e0b','e0a']],triggers=q.TRIGGERS,decode_steps=q.STEPS,per_service_cap_ms=cap,
        no_automatic_retry=True,driver=file_sha256(Path(__file__)),metric_code=file_sha256(Path(q.__file__)),
        calibration='reject observed TV above .05; otherwise cap the max(2*observed, BF16-logprob quantum) margin at .05 for TV'))
    contract,e0,backend,svc,exclusive,collectors,_,_=node.setup_baseline(root,settings,registration)
    lane=TrajectoryCollector(collectors['develop_quality'],scripts(HERE)); arms={}
    for arm in ('e0a','e0b'):
        native,receipt=replay_e0.restore_build(pair.parent/f'{arm}-develop_build.json',history['build_quality_measure_pins'][arm]['develop_build'],
            contract=contract,e0=e0,backend=backend,registration=prior,registration_pin=prior_pin,arm=arm)
        deployment=stage_deployment(native,root/(arm+'-deployment'))
        q.need(deployment.identity==receipt['data']['deployment'],'archived wheel deployment differs')
        arms[arm]=deployment
    q.need(arms['e0a'].identity!=arms['e0b'].identity,'E0 calibration requires distinct independent builds')
    reports,proofs=[],[]
    for trial,teacher in (('e0a','e0b'),('e0b','e0a')):
        folder=root/trial;folder.mkdir(mode=0o700);started=monotonic()
        with exclusive(started+2*cap/1000):
            deadline=started+cap/1000
            with svc.start(arms[trial],'ee-l3-trajectory-'+token_hex(8),deadline) as live:
                cp=lane.collect(live,folder/'trajectory-candidate',deadline)
            c,_=lane.read(folder/'trajectory-candidate',cp,arms[trial]); jobs=q.teacher_plan(c)
            deadline=min(started+2*cap/1000,monotonic()+cap/1000)
            with svc.start(arms[teacher],'ee-l3-teacher-'+token_hex(8),deadline) as live:
                tp=lane.collect(live,folder/'trajectory-teacher',deadline,jobs=jobs)
        report=lane.report(folder,(cp,tp),arms[trial],arms[teacher]);reports.append(report)
        proofs.append(dict(candidate=trial,teacher=teacher,candidate_receipt=cp,teacher_receipt=tp,
                           observed_node_ms=round((monotonic()-started)*1000),metrics=report))
        write(folder/'metrics.json',proofs[-1])
    result=dict(protocol=q.VERSION,plan_sha256=plan,registration_sha256=registration,e0=e0.digest,code=lane.code,
                deployments={k:d.identity for k,d in arms.items()},arms=proofs,limits=q.envelope(reports))
    return write(root/'calibration.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('new_directory',type=Path)
    for key in ('registration','pair','build-registration'):
        parser.add_argument('--'+key,type=Path,required=True);parser.add_argument('--'+key+'-sha256',required=True)
    a=parser.parse_args();os.umask(0o077);settings=node.checked_inputs(a.registration,a.registration_sha256)
    root=node.fresh(a.new_directory)
    try: print(run(root,settings,a.registration_sha256,a.pair,a.pair_sha256,a.build_registration,a.build_registration_sha256),flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(status='incomplete_no_tolerance',kind=type(exc).__name__,reason=str(exc)[:300])); raise


if __name__=='__main__': main()
