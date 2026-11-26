"""Prepare (never execute) a Level 3 chat launch from completed engineering receipts.

The local worker authorization is a draft. Copy its exact bytes to the declared
trusted host path separately. No SSH, API keys, Q/T files or candidate processes.
"""
import argparse
import json
from pathlib import Path, PurePosixPath

import launch_live as live
from worker_stdio import ENGINEERING_MS
from evolveengine.v3_level3_collect import QUALITY_CASES, write
from evolveengine.v3_level3_worker_ssh import _remote_path


EXTRA_PREREG = (live.HERE/'evidence/component-builder-engineering-20261125/extra-legal-h-prereg.json',
                '6a6b7dd485f64f8f17d87294e3d75d51f0fd079ff4e82c2c9b862df36567ae63')


def verify_extra(receipts, result_path, result_sha256, acceptance_sha256, *, prereg_ref=EXTRA_PREREG):
    """Check the separate legal drift positive, without modifying its frozen driver."""
    from experiments.spark26_gguf.level3 import control_run, legal_variant_control as legal
    read = control_run.read_pinned
    refs = receipts.resolve(strict=True)
    result_path = result_path.resolve(strict=True)
    prereg = read(*prereg_ref)
    from evolveengine.v3_level3_evidence import file_sha256
    live.need(file_sha256(refs/'bundle.tgz') == prereg['primary_bundle'], 'wrong primary source bundle')
    a = read(refs/'controls/acceptance.json', acceptance_sha256)
    result = read(result_path, result_sha256)
    engine = prereg['engine']
    labels = ('base', 'candidate', 'quality')
    live.need(set(result) == {'plan','receipts','engine','verdict','scope','holdout','paid_calls'} and
              result['engine'] == engine and result['verdict'] == 'valid' and
              result['scope'] == 'candidate_quality_not_speed_or_per_shape_kernel_trace' and
              result['holdout'] is False and type(result['paid_calls']) is int and result['paid_calls'] == 0 and
              set(result['receipts']) == set(labels) and len(set(result['receipts'].values())) == len(labels),
              'missing supplemental legal positive')
    root = result_path.parent
    plan = read(root/'plan.json', result['plan'])
    live.need(plan == dict(driver=prereg['driver'], acceptance=acceptance_sha256,
        trajectory=a['trajectory'], registration=a['registration'], graph=a['graph'], noise=a['noise'],
        engine=engine, runtime=legal.KNOBS, expected=prereg['expected'],
        scope='quality_positive_not_performance', retry=False, holdout=False) and
        all(plan[k] is False for k in ('retry','holdout')) and
        live.digest(Path(legal.__file__).read_bytes()) == prereg['driver'],
        'supplemental positive does not bind the frozen candidate and primary acceptance')
    codes = ('develop_build','develop_build','develop_quality')
    engines = (a['identities']['base'], engine, engine)
    rows = {}
    for name, operation, identity, verdict in zip(labels, codes, engines, prereg['expected'], strict=True):
        row = read(root/(name+'.json'), result['receipts'][name], 2<<20)
        r = row['result']
        live.need(row['verified'] is True and (r['engine'], r['operation'], r['status']) == (identity, operation, verdict)
            and type(r['node_ms']) is int and r['node_ms'] > 0 and control_run._sha(r['evidence']),
            'unverified supplemental positive '+name)
        rows[name] = r
    quality = rows['quality']
    data = quality['data']
    argv = rows['candidate']['data']['runtime']['argv']
    buckets = list(map(str, legal.KNOBS['cuda-graph-bs-decode']))
    # A build can claim success with the wrong service flags. Check that the
    # claimed argv retains the exact registered buckets and the stable wire settings.
    def args(flag):
        live.need(argv.count(flag) == 1, 'missing or duplicate positive runtime flag '+flag)
        start = argv.index(flag)+1
        end = next((i for i in range(start, len(argv)) if argv[i].startswith('--')), len(argv))
        return argv[start:end]
    live.need(args('--cuda-graph-bs-decode') == buckets and args('--max-running-requests') == ['6'] and
              args('--stream-interval') == ['1'] and args('--context-length') == ['8192'] and
              not {'--disable-radix-cache','--disable-prefill-cuda-graph','--disable-decode-cuda-graph'} & set(argv),
              'supplemental startup policy changed')
    live.need(quality['build'] == rows['candidate']['evidence'] and
              data['case_outcomes'] == dict.fromkeys((*QUALITY_CASES[:2], 'paired_cancel_numeric', 'autoregressive_full_vocab'), 'pass') and
              data['trajectory_screen']['status'] == 'valid' and data['trajectory_screen']['calibration'] == a['trajectory'] and
              'sampling_screen' not in data and bool(data.get('weight_observation')),
              'supplemental quality cases did not run against the approved candidate build')
    return dict(acceptance=acceptance_sha256, result=result_sha256, preregistration=prereg_ref[1],
                engine=engine, node_ms={name:r['node_ms'] for name,r in rows.items()})


def prepare(receipts, destination, *, expected_acceptance, remote, judge_deps,
            split_root, split_controls_sha256, worker_root, trajectory_root=None):
    receipts = receipts.resolve(strict=True)
    remote = PurePosixPath(_remote_path(remote)); _remote_path(judge_deps)
    worker_root = PurePosixPath(_remote_path(worker_root))
    split_root=Path(split_root).resolve(strict=True)
    trajectory_root = PurePosixPath(_remote_path(trajectory_root)) if trajectory_root else remote

    def ref(name):
        path = receipts/name
        return dict(path=str(path), sha256=live.digest(path.read_bytes()))

    def read(name):
        return json.loads((receipts/name).read_bytes())

    accept_ref=ref('controls/acceptance.json')
    live.need(accept_ref['sha256']==expected_acceptance,'unexpected engineering acceptance')
    a=read('controls/acceptance.json'); graph=ref('controls/graph/replayed.json')
    noise_ref=ref('noise/result.json'); noise=read('noise/result.json'); traj=ref('trajectory/calibration.json')
    reg=ref('registration.json')['sha256']; cfg=read('registration.json'); plan=read('noise-plan.json')
    from accept_development import verify
    verify(receipts/'controls',a,reg=reg,trajectory_pin=traj['sha256'],noise_pin=noise_ref['sha256'],graph_pin=graph['sha256'])
    live.need(ref('noise-plan.json')['sha256']==noise['plan_sha256'] and plan['registration_sha256']==reg and
              plan['trajectory_calibration']==dict(path=str(trajectory_root/'trajectory/calibration.json'),sha256=traj['sha256']),
              'wrong remote registration or noise plan')
    measured=live.checked_repeatability(noise)
    policy=live.business.natural_policy(noise_ref['sha256'],noise['timing_workload_sha256'],
        measured['descriptive_log_envelope']['scenes'],chat_corpus=plan['chat_workload']['sha256'],
        replay_floors=live.business.replay_audit_floors(noise['receipts']))
    from experiments.spark26_gguf.level3 import live_split_quality as split, qualify_split_worker as qualified, node_preflight
    sp=split_root/'split/registration.json'; cp=split_root/'qualification/acceptance.json'
    sr=dict(path=str(sp),sha256=live.digest(sp.read_bytes()))
    cr=dict(path=str(cp),sha256=split_controls_sha256)
    registered=split.read_registration(sp,sr['sha256'],reg,traj['sha256'])
    split.verify_overlay(cfg['implementation_sha256'],node_preflight.implementation_pins(),registered)
    qualified.verify(cp.parent,split.read(cp,cr['sha256']),baseline=reg,cal=traj['sha256'],
        previous_acceptance=accept_ref['sha256'],split_pin=sr['sha256'],registered=registered)
    raw=live.ENGINEERING_CONFIG.read_bytes();config,limits=live.cfg.validate_objective_config(dict(json.loads(raw),registered_scoring=policy))
    _,runtime=live.profile_and_runtime(limits,live.system_for(config),resources=config['guest_resources'])
    destination=destination.resolve(); destination.mkdir(mode=0o700)
    cap=cfg['node_ms'] | dict(develop_profile=7200000,develop_diagnostic=ENGINEERING_MS)
    lease=dict(schema=live.WORKER_SCHEMA,engine_scope=live.ENGINEERING_EXPLORATION,baseline_sha256=reg,jobs=None,node_ms=None,max_per_op=cap,
               controls=dict(replay=str(remote/'controls/graph/replayed.json'),replay_sha256=graph['sha256'],chat_workload=plan['chat_workload'],
                             natural_workload_sha256=noise['timing_workload_sha256'],weight_audit=live.weights.POLICY,
                             trajectory_calibration=plan['trajectory_calibration'],
                             split_quality=dict(path=str(worker_root/'split/registration.json'),sha256=sr['sha256'])))
    lp=destination/'authorization.json';pin=write(lp,lease)
    worker=dict(registration=str(worker_root/'registration.json'),authorization=str(worker_root/'authorization.json'),
                entry=str(worker_root/'src/experiments/spark26_gguf/level3/worker_stdio.py'),judge_deps=judge_deps,
                report=str(worker_root/'live-worker'),lease=dict(path=str(lp),sha256=pin))
    plan=dict(schema=live.SCHEMA,scope=live.ENGINEERING_SCOPE,acceptance_sha256=live.ACCEPTANCE_SHA,
        controller_sha256=live.controller_pins(),config_sha256=live.digest(raw),runtime_sha256=live.digest(live.canonical(runtime)),
        allowed_resources=dict(server='spark-c3bc',paid_deepseek=True,paid_research=config['research']==live.OPEN_RESEARCH,
                               formal_w=False,holdout_disclosure=False),
        evidence=dict(graph_e0_replay=graph,natural_repeatability=noise_ref,development_controls=accept_ref,
                      split_quality_registration=sr,split_quality_controls=cr),
        holdout=None,worker=worker,corpus_sha256=None)
    pp=destination/'launch-plan.json';approval=dict(path=str(pp),sha256=write(pp,plan))
    live.audit(approval)
    return write(destination/'audit-only.json',dict(approval=approval,authorization=pin,paid_calls=0,worker_started=False,
                remote_authorization_installed=False,holdout=False,split_controls=cr['sha256']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipts',type=Path);p.add_argument('new_directory',type=Path)
    for name in ('expected-acceptance','remote','judge-deps'): p.add_argument('--'+name,required=True)
    p.add_argument('--split-root', type=Path,required=True); p.add_argument('--split-controls-sha256',required=True)
    p.add_argument('--worker-root',required=True)
    p.add_argument('--trajectory-root', help='Explicit registered node root for a reused calibration')
    a=p.parse_args()
    print(prepare(a.receipts,a.new_directory,expected_acceptance=a.expected_acceptance,remote=a.remote,
                  judge_deps=a.judge_deps,split_root=a.split_root,split_controls_sha256=a.split_controls_sha256,
                  worker_root=a.worker_root,trajectory_root=a.trajectory_root))
