"""One bounded active GB10 metrology experiment; no Engine, API, or model."""
import argparse
from contextlib import contextmanager
import json
import mmap
import os
from pathlib import Path
import selectors
import subprocess
import sys
from time import monotonic, sleep


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_cuda_meter import CudaMeter
from evolveengine.v3_level3_gb10 import Gb10, need


MIB = 1<<20
PLAN = dict(schema='gb10-cuda-meter-cal-v1',order=['idle','cpu-128','gpu-16','gpu-128','gpu-512']*3,
            cpu_control_mib=128,smallest_tested_gpu_mib=16,samples_per_phase=5,interval_seconds=.1,
            max_absolute_gpu_error_mib=4,max_cpu_or_idle_change_mib=4,max_recovery_error_mib=4,
            wall_seconds=150,measure='CUDA-visible used = total - free; not allocator or tensor residency')


@contextmanager
def cpu_pages(size):
    with mmap.mmap(-1,size) as memory:
        for off in range(0,size,mmap.PAGESIZE):
            memory[off] = 1
        yield


def child(gpu, what, size):
    if what == 'cpu':
        with cpu_pages(size):
            print('allocated',flush=True)
            need(sys.stdin.readline()=='release\n','invalid calibration control')
        print('released',flush=True)
    if what == 'gpu':
        with CudaMeter(gpu).context() as ctx:
            print('ready',flush=True)
            need(sys.stdin.readline()=='allocate\n','invalid calibration control')
            with ctx.calibration_allocation(size):
                print('allocated',flush=True)
                need(sys.stdin.readline()=='release\n','invalid calibration control')
            print('released',flush=True)
            need(sys.stdin.readline()=='exit\n','invalid calibration control')


@contextmanager
def probe_process(gpu, what, size):
    proc = subprocess.Popen([sys.executable,'-I',str(Path(__file__).resolve()),'--child',gpu,what,str(size)],
                            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,bufsize=0)
    try:
        yield proc
        need(proc.wait(timeout=7)==0,'calibration fixture exited with error')
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        proc.stdin.close()
        proc.stdout.close()


def expect(proc, line):
    with selectors.DefaultSelector() as wait:
        wait.register(proc.stdout,selectors.EVENT_READ)
        need(bool(wait.select(8)) and proc.stdout.readline(32)==line,'calibration fixture handshake failed')


def send(proc, txt):
    proc.stdin.write(txt+b'\n')


def stable_samples(meter):
    rows = []
    for _ in range(PLAN['samples_per_phase']):
        row = meter.sample()
        row['host_memavailable_kib'] = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines()
                                                if l.startswith('MemAvailable:')))
        rows.append(row)
        sleep(PLAN['interval_seconds'])
    return rows


def evaluate(rows):
    need([r['trial'] for r in rows]==PLAN['order'],'incomplete or reordered calibration')
    outcomes = []
    for row in rows:
        phases = [row[k] for k in ('before','held','after')]
        need(all(len(p)==PLAN['samples_per_phase'] for p in phases),'incomplete sampling phases')
        totals = {s['total_bytes'] for phase in phases for s in phase}
        seq = [s['at_ns'] for phase in phases for s in phase]
        need(len(totals)==1 and seq==sorted(set(seq)), 'unstable CUDA-visible total or sample chronology')
        b,h,a = [[s['used_bytes'] for s in phase] for phase in phases]
        expected = int(row['trial'].split('-')[1])*MIB if row['trial'].startswith('gpu-') else 0
        seen = (min(h)-max(b),max(h)-min(b))
        back = (min(a)-max(b),max(a)-min(b))
        bound = (PLAN['max_absolute_gpu_error_mib'] if expected else PLAN['max_cpu_or_idle_change_mib'])*MIB
        passed = all(abs(x-expected)<=bound for x in seen) and all(abs(x)<=PLAN['max_recovery_error_mib']*MIB for x in back)
        outcomes.append(dict(trial=row['trial'],expected_bytes=expected,held_delta_bytes=seen,recovery_delta_bytes=back,
                             accepted=passed))
    return dict(status='observed_within_registered_scope' if all(r['accepted'] for r in outcomes) else 'unqualified_meter',
                trials=outcomes)


def run(root, plan_pin):
    need(plan_pin==digest(canonical(PLAN)+b'\n'),'calibration plan fingerprint differs')
    root.mkdir(mode=0o700)
    write(root/'plan.json',PLAN)
    deadline = monotonic()+PLAN['wall_seconds']
    host = Gb10.discover(deadline)
    write(root/'node.json',host.inventory(deadline))
    rows = []
    with host.exclusive(deadline):
        with (root/'samples.jsonl').open('xb') as out, CudaMeter(host.gpu_uuid).context() as meter:
            for trial in PLAN['order']:
                need(monotonic()+22<deadline,'insufficient calibration cleanup margin')
                if trial=='idle':
                    before,held,after = (stable_samples(meter) for _ in range(3))
                else:
                    kind,n = trial.split('-')
                    before = stable_samples(meter) if kind=='cpu' else None
                    with probe_process(host.gpu_uuid,kind,int(n)*MIB) as proc:
                        if kind=='gpu':
                            expect(proc,b'ready\n')
                            before = stable_samples(meter)
                            send(proc,b'allocate')
                        expect(proc,b'allocated\n')
                        held = stable_samples(meter)
                        send(proc,b'release')
                        expect(proc,b'released\n')
                        after = stable_samples(meter)
                        if kind=='gpu':
                            send(proc,b'exit')
                row = dict(trial=trial,before=before,held=held,after=after)
                rows.append(row)
                out.write(canonical(row)+b'\n')
                out.flush()
    verdict = evaluate(rows)
    write(root/'result.json',dict(**verdict,plan_sha256=plan_pin))
    print(canonical(verdict).decode())


if __name__ == '__main__':
    if len(sys.argv)==5 and sys.argv[1]=='--child':
        child(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--plan',action='store_true')
        parser.add_argument('--report',type=Path)
        parser.add_argument('--sha256')
        args = parser.parse_args()
        if args.plan:
            print(canonical(dict(plan=PLAN,sha256=digest(canonical(PLAN)+b'\n'))).decode())
        else:
            if args.report is None or args.sha256 is None:
                parser.error('--report and --sha256 are required for a live calibration')
            os.umask(0o077)
            run(args.report,args.sha256)
