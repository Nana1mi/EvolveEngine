"""Replay one registered non-scoring diagnostic on independently built E0."""
import argparse
from dataclasses import asdict
import os
import signal
from pathlib import Path
import stat
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))

from experiments.spark26_gguf.level3 import node_preflight as node, control_run, replay_e0
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import stage_deployment
from evolveengine.v3_level3_diagnostic import DockerDiagnostic, MAX_SCRIPT, MAX_MS, MIN_MS

SCHEMA = 'level3-e0-diagnostic-engineering-v1'


def make_plan(registration, build, build_pin, probe, probe_pin, cap):
    node.need(MIN_MS<=cap<=MAX_MS and control_run._sha(probe_pin) and control_run._sha(build_pin),
              'bad diagnostic scope or hash')
    return dict(schema=SCHEMA,registration_sha256=registration,build=str(build),build_sha256=build_pin,
                script=str(probe),script_sha256=probe_pin,node_ms=cap,scoring=False,paid_api_calls=0,holdout=False,
                runner_sha256=digest(Path(__file__).read_bytes()),
                diagnostic_driver_sha256=digest((HERE.parents[2]/'evolveengine/v3_level3_diagnostic.py').read_bytes()))


def execute(root, settings, reg_pin, plan):
    probe, build = (Path(plan[k]) for k in ('script','build'))
    node.need(probe.resolve(strict=True)==probe and stat.S_ISREG(probe.stat().st_mode) and 0<probe.stat().st_size<=MAX_SCRIPT,
              'diagnostic source is not a registered local file')
    raw = probe.read_bytes()
    node.need(digest(raw)==plan['script_sha256'] and canonical(plan)==canonical(make_plan(reg_pin,build,plan['build_sha256'],
             probe,plan['script_sha256'],plan['node_ms'])),'diagnostic implementation or source drift')
    contract,e0,backend,svc,exclusive,*_ = node.setup_baseline(root,settings,reg_pin)
    native,receipt = replay_e0.restore_build(build,plan['build_sha256'],contract=contract,e0=e0,backend=backend,
                                         registration=settings,registration_pin=reg_pin)
    deployment = stage_deployment(native,root/'deployment')
    node.need(deployment.identity==receipt['data']['deployment'],'diagnostic deployment differs from registered build')
    diag = DockerDiagnostic(svc.spec,docker=svc.docker,exclusive=exclusive)
    result, archive = diag.launch(deployment,receipt['evidence'],raw.decode('utf-8'),root,plan['node_ms'])
    write(root/'diagnostic.json',dict(label='initial_operator_fixture_non_scoring',receipt=asdict(result),archive=archive.name))
    node.need(result.status=='completed_non_scoring','diagnostic did not complete; cannot infer a candidate score')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--registration',required=True,type=Path)
    a.add_argument('--registration-sha256',required=True)
    sp = a.add_subparsers(dest='mode',required=True)
    p = sp.add_parser('plan')
    p.add_argument('--build',required=True,type=Path)
    p.add_argument('--build-sha256',required=True)
    p.add_argument('--script',required=True,type=Path)
    p.add_argument('--script-sha256',required=True)
    p.add_argument('--node-ms',type=int,default=90000)
    r = sp.add_parser('run')
    r.add_argument('report',type=Path)
    r.add_argument('--plan',required=True,type=Path)
    r.add_argument('--plan-sha256',required=True)
    args = a.parse_args()
    os.umask(0o077)
    settings = node.checked_inputs(args.registration,args.registration_sha256)
    if args.mode=='plan':
        row = make_plan(args.registration_sha256,args.build,args.build_sha256,args.script,args.script_sha256,args.node_ms)
        sys.stdout.buffer.write(canonical(row)+b'\n')
        return
    plan = control_run.read_pinned(args.plan,args.plan_sha256)
    root = node.fresh(args.report)
    write(root/'diagnostic-plan.json',dict(sha256=args.plan_sha256,plan=plan))
    def stop(signum, frame):
        # Only the first stop interrupts this run; cleanup must be allowed to finish.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt('trusted operator stopped diagnostic')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        execute(root,settings,args.registration_sha256,plan)
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(kind=type(exc).__name__,reason=str(exc)[:300],scoring=False))
        raise


if __name__=='__main__':
    main()
