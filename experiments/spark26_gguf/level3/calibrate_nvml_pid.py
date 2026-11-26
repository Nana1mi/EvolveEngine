"""Bounded NVML per-compute-PID memory calibration on the registered GB10."""
import argparse
import json
from pathlib import Path
import sys
from time import monotonic, sleep

sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrate_cuda_meter import MIB, digest, canonical, Gb10, need, expect, probe_process, send, write
from evolveengine.v3_level3_evidence import parse_nvidia_smi_process_memory


PLAN = dict(schema='gb10-nvml-pid-cal-v1',order=['cpu-128','gpu-12','gpu-16','gpu-128','gpu-512']*3,
            samples_per_phase=3,max_error_mib=1,cpu_control_has_no_compute_pid=True,wall_seconds=150,
            metric='nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory for the exact external PID',
            scope='NVML-reported compute process allocations; not tensor attribution or arbitrary CUDA memory coverage')


def sample(host, pid, deadline):
    rows = []
    for _ in range(PLAN['samples_per_phase']):
        raw = host.compute_memory(deadline)
        parsed = parse_nvidia_smi_process_memory(raw,host.gpu_uuid)
        need(set(parsed)<={pid}, 'unexpected active compute PID')
        rows.append(dict(raw=raw,pid=pid,used_bytes=parsed.get(pid)))
        sleep(.08)
    return rows


def evaluate(rows):
    need([r['trial'] for r in rows]==PLAN['order'],'incomplete or reordered calibration trials')
    tolerance = PLAN['max_error_mib']*MIB
    outcomes = []
    for row in rows:
        phases = [[r['used_bytes'] for r in row[k]] for k in ('before','held','after')]
        b,h,a = phases
        need(all(len(p)==PLAN['samples_per_phase'] for p in phases),'incomplete calibration samples')
        if row['trial'].startswith('cpu-'):
            result = dict(accepted=all(x is None for p in phases for x in p),cpu_has_no_compute_pid=all(x is None for x in h))
        elif any(type(x) is not int or x <= 0 for p in phases for x in p):
            result = dict(accepted=False,reason='missing per-compute-PID sample')
        else:
            expected = int(row['trial'].split('-')[1])*MIB
            delta = (min(h)-max(b),max(h)-min(b))
            recovery = (min(a)-max(b),max(a)-min(b))
            result = dict(accepted=all(abs(x-expected)<=tolerance for x in delta) and all(abs(x)<=tolerance for x in recovery),
                          expected_bytes=expected,held_delta_bytes=delta,recovery_delta_bytes=recovery)
        outcomes.append(dict(trial=row['trial'],**result))
    return dict(status='observed_within_registered_scope' if all(r['accepted'] for r in outcomes) else 'unqualified_meter',
                trials=outcomes)


def verify_calibration(root, result_sha256, node):
    """Replay raw NVML samples, not just the recorded verdict."""
    def read(name, cap):
        path = root/name
        need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
             'missing, linked or oversized NVML calibration artifact')
        return path.read_bytes()

    raw_result = read('result.json', 16*1024)
    need(digest(raw_result)==result_sha256,'NVML calibration result hash mismatch')
    result = json.loads(raw_result)
    raw_plan = read('plan.json', 8*1024)
    raw_node = read('node.json', 8*1024)
    raw_samples = read('samples.jsonl', 128*1024)
    pinned_node = json.loads(raw_node)
    need(raw_plan==canonical(PLAN)+b'\n' and result['plan_sha256']==digest(raw_plan) and
         result['raw_sha256']==digest(raw_samples) and result['node']==pinned_node,
         'NVML calibration plan, node or raw hash mismatch')
    need(all(pinned_node[key]==node[key] for key in ('worker_id','boot_id','device_uuid','driver')) and
         pinned_node['compute_pids']==[], 'NVML calibration belongs to another node, boot, GPU or driver')
    rows = [json.loads(line) for line in raw_samples.splitlines()]
    for row in rows:
        for phase in ('before','held','after'):
            for sample in row[phase]:
                parsed = parse_nvidia_smi_process_memory(sample['raw'], node['device_uuid'])
                saved_pid = sample['pid']
                need(type(saved_pid) is int and set(parsed)<={saved_pid} and
                     parsed.get(saved_pid)==sample['used_bytes'], 'NVML calibration sample does not replay')
    replay = evaluate(rows)
    need(replay['status']=='observed_within_registered_scope' and
         all(canonical(value)==canonical(result[key]) for key,value in replay.items()),
         'NVML calibration failed to reproduce the accepted verdict')
    return dict(source='nvidia_smi_compute_pid',result_sha256=result_sha256,
                plan_sha256=result['plan_sha256'],raw_sha256=result['raw_sha256'],
                scope=PLAN['scope'],smallest_calibrated_gpu_delta_bytes=12*MIB,
                role='engineering_E0_observation_not_candidate_residency_gate')


def run(root, pin):
    need(pin==digest(canonical(PLAN)+b'\n'),'calibration plan fingerprint differs')
    root.mkdir(mode=0o700)
    write(root/'plan.json',PLAN)
    deadline = monotonic()+PLAN['wall_seconds']
    host = Gb10.discover(deadline)
    pins = host.inventory(deadline)
    write(root/'node.json',pins)
    rows = []
    with host.exclusive(deadline), (root/'samples.jsonl').open('xb') as out:
        for trial in PLAN['order']:
            need(monotonic()+25<deadline,'insufficient room for cleanup')
            kind,n = trial.split('-')
            before = sample(host,-1,deadline) if kind=='cpu' else None
            with probe_process(host.gpu_uuid,kind,int(n)*MIB) as proc:
                if kind=='gpu':
                    expect(proc,b'ready\n')
                    before = sample(host,proc.pid,deadline)
                    send(proc,b'allocate')
                expect(proc,b'allocated\n')
                held = sample(host,proc.pid,deadline)
                send(proc,b'release')
                expect(proc,b'released\n')
                after = sample(host,proc.pid,deadline)
                if kind=='gpu':
                    send(proc,b'exit')
            row = dict(trial=trial,before=before,held=held,after=after)
            rows.append(row)
            out.write(canonical(row)+b'\n')
            out.flush()
    verdict = evaluate(rows)
    write(root/'result.json',dict(**verdict,plan_sha256=pin,node=pins,raw_sha256=digest((root/'samples.jsonl').read_bytes())))
    print(canonical(verdict).decode())


if __name__=='__main__':
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
        run(args.report,args.sha256)
