"""Supplemental DEV-only positive: vary floating-point reduction by batch, not by prompt.

The historical renamed-kernel control is left untouched. This separate receipt tests
whether a numerically non-identical but legal implementation can use the new gates.
It is not a performance result, not a holdout, and cannot authorize a model call.
"""
import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from experiments.spark26_gguf.level3 import accept_development as ac, control_run, native_controls as nc, node_preflight as n, worker_stdio
from evolveengine import v3_level3_eos as eos
from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_startup import PATH
from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector, scripts

TEMPLATE = 'vec_dot_q_cuda_t vec_dot_q_cuda>\nstatic __global__ void mul_mat_vec_q('
MASK = 'for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {'
LAUNCH = ('  mul_mat_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>\n'
          '      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);')
KNOBS = {'cuda-graph-bs-decode': [1,2,3,4,5,6], 'max-running-requests': 6, 'radix-cache': True}


def candidate(contract):
    n.need(contract.provenance.get('archive_sha256') == nc.SGLANG_ARCHIVE_SHA256 and
           contract.provenance.get('revision') == nc.SGLANG_REVISION, 'wrong positive control E0')
    header = contract.base[nc.MMVQ].decode()
    header = nc.once(header, TEMPLATE, TEMPLATE.replace('vec_dot_q_cuda>', 'vec_dot_q_cuda, bool ascending = false>'))
    # Same XOR butterfly, same terms exactly once, alternate parenthesization.
    header = nc.once(header, MASK, 'for (int bit = 1; bit < WARP_SIZE; bit <<= 1) {\n'
                                  '    const int mask = ascending ? bit : (WARP_SIZE / (2 * bit));')
    header = nc.once(header, LAUNCH, '  if (nvecs == 1) {\n' + LAUNCH.replace('q8_0_q8_1>', 'q8_0_q8_1, true>')
                     + '\n  } else {\n' + LAUNCH + '\n  }')
    return contract.from_args({'files': {nc.MMVQ: header, PATH: canonical(KNOBS).decode()}})


def run(root, primary, acceptance_sha):
    a = control_run.read_pinned(primary/'controls/acceptance.json', acceptance_sha)
    reg, traj, noise, graph = [a[k] for k in ('registration','trajectory','noise','graph')]
    ac.verify(primary/'controls', a, reg=reg, trajectory_pin=traj, noise_pin=noise, graph_pin=graph)
    cfg = n.checked_inputs(primary/'registration.json', reg)
    graph_file = primary/'controls/graph/replayed.json'; control_run.read_pinned(graph_file, graph)
    trajectory = primary/'trajectory/calibration.json'; control_run.read_pinned(trajectory, traj)
    t = control_run.read_pinned(primary/'noise/result.json', noise, 2<<20)
    plan = control_run.read_pinned(primary/'noise-plan.json', t['plan_sha256'])
    contract, e0, backend, service, lock, collectors, _, _ = n.setup_baseline(root, cfg, reg)
    positive = candidate(contract)
    p = write(root/'plan.json', dict(driver=file_sha256(Path(__file__)), acceptance=acceptance_sha,
        trajectory=traj, registration=reg, graph=graph, noise=noise, engine=positive.digest, runtime=KNOBS,
        expected=['built','built','valid'], scope='quality_positive_not_performance', retry=False, holdout=False))
    q=collectors['develop_quality']; q.policy=replace(q.policy,memory_limit_bytes=None)
    tm=collectors['develop_measure']; n.need(tm.enable_natural_eos(plan['chat_workload'])==t['timing_workload_sha256'],'workload drift')
    lane=TrajectoryCollector(q,scripts(n.HERE)); lane.activate(dict(path=str(trajectory),sha256=traj),reg,e0.digest)
    q.enable_trajectory(traj,lane.code,lambda rows:eos.grade(rows,Path(cfg['anchor_d3']),Path(cfg['anchor_r1']),paired=True))
    w=worker_stdio.EngineeringWorker(root/'worker',contract,backend,service,exclusive=lock,
        collect=lambda op,*args,**kwargs: collectors[op](op,*args,**kwargs),settings=cfg,replay_path=graph_file,profile_timing=tm)
    w.trajectory_lane=lane; build=None; rows={}
    for label,op,engine,status in [('base','develop_build',e0,'built'),('candidate','develop_build',positive,'built'),('quality','develop_quality',positive,'valid')]:
        cap=cfg['node_ms'][op]; write(root/(label+'-reserved.json'),dict(engine=engine.digest,build=build.evidence if build else None,cap=cap))
        r=w(op,engine,build,cap); ok=w.verify(r); rows[label]=write(root/(label+'.json'),dict(result=asdict(r),verified=ok))
        print(label,r.status,r.node_ms,flush=True); n.need(ok and r.status==status,'unexpected '+label+' outcome; no replacement')
        if label=='candidate':
            argv=r.data['runtime']['argv']; i=argv.index('--cuda-graph-bs-decode');n.need(argv[i+1:i+7]==list(map(str,KNOBS['cuda-graph-bs-decode'])) and '--disable-radix-cache' not in argv,'declared argv not present')
        if op=='develop_build': build=r
    screen=r.data['trajectory_screen'];n.need(screen['calibration']==traj and screen['status']=='valid' and set(r.data['case_outcomes'].values())=={'pass'},'incomplete quality witness')
    return write(root/'result.json',dict(plan=p,receipts=rows,engine=positive.digest,verdict='valid',scope='candidate_quality_not_speed_or_per_shape_kernel_trace',holdout=False,paid_calls=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('new_directory',type=Path);ap.add_argument('primary',type=Path);ap.add_argument('acceptance_sha256');args=ap.parse_args()
    os.umask(0o077);root=n.fresh(args.new_directory)
    try: print(run(root,args.primary.resolve(strict=True),args.acceptance_sha256),flush=True)
    except BaseException as exc: write(root/'incomplete.json',dict(error=type(exc).__name__,detail=str(exc)[:350]));raise
