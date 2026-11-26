"""Exercise a real operator stop after a leased CUDA diagnostic allocates memory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from experiments.spark26_gguf.level3 import control_run, node_preflight as node
from evolveengine.v3_harness import digest
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_gb10 import Gb10

MARKER = b'cuda-stop-fixture-ready\n'


def observed_marker(path):
    with path.open('rb') as stream:
        return stream.read(len(MARKER)) == MARKER


def run(root, target, registration, cfg, reg_pin, plan_path, plan_pin):
    plan = control_run.read_pinned(plan_path,plan_pin)
    probe = Path(plan['script'])
    node.need(plan['registration_sha256']==reg_pin and probe==HERE/'fixtures/cuda_diagnostic_stop.py'
              and file_sha256(probe)==plan['script_sha256'] and not target.exists(),'unregistered stop fixture or reused destination')
    host = Gb10(cfg['gpu_uuid'],cfg['lock'])
    argv = [sys.executable,'-Bu',str(HERE/'diagnose_e0.py'),'--registration',str(registration),
            '--registration-sha256',reg_pin,'run',str(target),'--plan',str(plan_path),'--plan-sha256',plan_pin]
    write(root/'registration.json',dict(registered_plan=plan_pin,baseline=reg_pin,target=str(target),paid_api_calls=0))
    began = monotonic()
    observed = None
    with (root/'stdout').open('xb') as out,(root/'stderr').open('xb') as err:
        proc = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
        try:
            while proc.poll() is None and monotonic()-began<80:
                matches = list(target.glob('diag-*/observer.stdout'))
                if len(matches)==1 and observed_marker(matches[0]):
                    job = matches[0].parent
                    view = json.loads((job/'docker.inspect.json').read_bytes())
                    seen = host.inventory(monotonic()+5)
                    pids = host.pids(view['Id'],monotonic()+5)
                    identity = set(seen['compute_pids'])
                    if identity and identity <= pids and seen['boot_id']==cfg['boot_id']:
                        observed = dict(cuda_compute_observed=True,compute_pid_count=len(identity),
                                        marker_sha256=digest(MARKER),container=view['Id'],
                                        elapsed_ms=int((monotonic()-began)*1000),job=job.name)
                        break
                sleep(.2)
        finally:
            if observed is not None:
                write(root/'before-stop.json',observed)
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                write(root/'stop-sent.json',dict(signal='SIGINT',after_cuda_ready=observed is not None))
            rc = proc.wait(timeout=35)
    node.need(observed is not None and rc!=0 and (root/'stop-sent.json').exists(), 'stop was not witnessed during an active CUDA diagnostic')
    outcome = json.loads((target/'incomplete.json').read_bytes())
    unsettled = json.loads((target/observed['job']/'unsettled.json').read_bytes())
    node.need(outcome['kind']=='KeyboardInterrupt' and unsettled==dict(error='KeyboardInterrupt',reservation_retained_ms=plan['node_ms'])
              and not (target/observed['job']/'result.json').exists(),'interrupted run claimed a completed diagnostic or missed settlement')
    with host.exclusive(monotonic()+10):
        after = host.inventory(monotonic()+8)
        node.need(after['boot_id']==cfg['boot_id'] and not after['compute_pids'] and not host.leftover_jobs(monotonic()+5),
                  'verified diagnostic cleanup is missing')
    return write(root/'result.json',dict(kind='diagnostic_manual_stop_scope_only',passed=True,container_removed=True,
                 cuda_pids_after=0,stop_witness_sha256=file_sha256(root/'before-stop.json'),conservative_node_ms=plan['node_ms'],
                 wall_ms=int((monotonic()-began)*1000),scope='one_gpu_diagnostic_lane_non_scoring'))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('report',type=Path)
    a.add_argument('--target',required=True,type=Path)
    a.add_argument('--registration',required=True,type=Path)
    a.add_argument('--registration-sha256',required=True)
    a.add_argument('--plan',required=True,type=Path)
    a.add_argument('--plan-sha256',required=True)
    args=a.parse_args()
    os.umask(0o077)
    cfg = node.checked_inputs(args.registration,args.registration_sha256)
    root = node.fresh(args.report)
    try:
        print(run(root,args.target,args.registration,cfg,args.registration_sha256,args.plan,args.plan_sha256),flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(kind=type(exc).__name__,reason=str(exc)[:300],scoring=False))
        raise


if __name__=='__main__':
    main()
