"""Registered same-node timing repeatability check; no Agent, API or Q/T."""
import argparse
from dataclasses import asdict
from math import exp, fsum, log, isfinite
import os
from pathlib import Path
from secrets import token_hex
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))

from experiments.spark26_gguf.level3 import node_preflight as node, control_run, replay_e0
from evolveengine.v3_level3_collect import write
from evolveengine.v3_level3_deploy import stage_deployment
from evolveengine.v3_level3_evidence import file_sha256, validate_evidence
from evolveengine.v3_level3_scoring import (
    GUARDRAIL_KEYS, WORKBUDDY_V2_PRIMARY_SCENES, WORKBUDDY_V2_REPLAY_SCENES, WORKBUDDY_V2_WEIGHTS,
    calibration_output_work, paired_w3_ttft, replay_agreement, workbuddy_v2_precision_target, REPLAY_AUDIT_MARGIN,
)

SCHEMA = 'spark26-e0-repeatability-v1'
# Alternate the order of two independently built, quality-accepted E0 deployments.
# These are fresh service/observer jobs, not additional independent builds.
ORDER = (('e0a','e0b'),('e0b','e0a'),('e0a','e0b'),('e0b','e0a'),('e0a','e0b'),('e0b','e0a'))
SCENES = ('W1','W2','W3','W4')
FLOOR = 0.01
MAX_GLOBAL = 0.03
MAX_SCENE = 0.05


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def draft(settings, registration_pin, pair_path, pair_pin, *, build_registration=None, natural_eos=False,
          chat=None,trajectory=None):
    prior, prior_pin = settings, registration_pin
    if build_registration is not None:
        prior = control_run.read_pinned(Path(build_registration['path']), build_registration['sha256'])
        prior_pin = build_registration['sha256']
        require(prior_pin != registration_pin and 'timing_text' in settings,
                'reusing builds requires a distinct real-text registration')
        replay_e0.check_build_compatibility(settings, prior)
    pair = control_run.read_pinned(pair_path,pair_pin)
    saved = control_run.read_pinned(pair_path.parent/'registration.json',
                                    node.digest(node.canonical(dict(sha256=prior_pin,config=prior))+b'\n'))
    require(saved['sha256']==prior_pin and pair.get('kind')==control_run.PAIR_KIND
            and control_run._sha(pair.get('engine')) and pair.get('candidate_residency_gate') is False
            and type(pair.get('receipts')) is list and len(pair['receipts'])==6
            and all(control_run._sha(x) for x in pair['receipts']) and len(set(pair['receipts']))==6,
            'not a completed independent baseline pair')
    arms = {}
    for idx, arm in enumerate(('e0a','e0b')):
        pins = {}
        for i, (op, status) in enumerate((('develop_build','built'),('develop_quality','valid'),('develop_measure','observed'))):
            f = pair_path.parent/f'{arm}-{op}.json'
            pin = file_sha256(f)
            r = control_run.read_pinned(f,pin)
            require((r['operation'],r['status'],r['evidence'],r['engine'],r['build'])==
                    (op,status,pair['receipts'][3*idx+i],pair['engine'],pair['receipts'][3*idx] if i else None),
                    'missing or unrelated E0 build/quality/timing receipt')
            pins[op]=pin
        arms[arm]=pins
    plan = dict(schema=SCHEMA, baseline=str(pair_path), baseline_sha256=pair_pin, registration_sha256=registration_pin,
                build_quality_measure_pins=arms, execution_order=ORDER, comparisons='chronological_second_over_first',
                aggregation='mean_of_equal_scene_log_ratios', accepted_max_abs_log_ratio=MAX_GLOBAL,
                accepted_max_abs_scene_log_ratio=MAX_SCENE, descriptive_envelope_floor_log=FLOOR,
                no_substitution_or_retries=True, max_total_node_ms=2*len(ORDER)*settings['node_ms']['develop_measure'],
                runner_sha256=file_sha256(Path(__file__).resolve()),wheel_restore_sha256=file_sha256(Path(replay_e0.__file__)),
                holdout=False, paid_api_calls=0)
    if build_registration is not None:
        plan['build_registration'] = build_registration
        plan['max_total_node_ms'] += 2*settings['node_ms']['develop_quality']
    if natural_eos:
        require('timing_text' in settings, 'natural EOS requires the real-text timing plan')
        plan['timing_mode'] = 'natural_eos'
    if chat is not None:
        require(natural_eos and type(chat) is dict and set(chat) == {'path','sha256'} and
                Path(chat['path']).is_absolute() and control_run._sha(chat['sha256']),
                'chat calibration requires a pinned natural-EOS request pool')
        plan['replay_audit_absolute_margin'] = REPLAY_AUDIT_MARGIN
        plan['chat_workload'] = chat
    if trajectory is not None:
        require(chat and prior is not None and set(trajectory)=={'path','sha256'} and Path(trajectory['path']).is_absolute()
                and control_run._sha(trajectory['sha256']), 'unbound trajectory registration')
        plan['trajectory_calibration']=trajectory
        plan['max_total_node_ms']+=2*settings['node_ms']['develop_quality']
    return plan


def summarize(pairs):
    require(len(pairs)==len(ORDER),'incomplete timing repeats; no envelope')
    wb_v2 = bool(pairs and set(pairs[0][0].get('scenes', {})) == set(WORKBUDDY_V2_REPLAY_SCENES))
    if wb_v2:
        score_keys = tuple(WORKBUDDY_V2_WEIGHTS)
        by_scene = {s: [] for s in score_keys}
        by_guardrail = {g: [] for g in GUARDRAIL_KEYS}
        pair_log = []
        for seq, ((a, b), (first, second)) in enumerate(zip(ORDER, pairs), 1):
            require((first['arm'], second['arm']) == (a, b)
                    and all(set(row['scenes']) == set(WORKBUDDY_V2_REPLAY_SCENES) for row in (first, second)),
                    'unexpected timing arm order or incomplete v2 scenes')
            for s in ('W1', 'W2', 'W3', 'W5'):
                x, y = (r['scenes'][s]['dev_score'] for r in (first, second))
                require(all(type(v) in (int, float) and isfinite(v) and v > 0 for v in (x, y)), 'invalid E0 v2 score')
                by_scene[s].append(log(y) - log(x))
            xh, yh = (r['scenes']['W4']['w4_hot_decode_tps'] for r in (first, second))
            xc, yc = (r['scenes']['W4']['w4_cold_ttft_ms'] for r in (first, second))
            require(all(type(v) in (int, float) and isfinite(v) and v > 0 for v in (xh, yh, xc, yc)), 'invalid E0 W4 sub-scores')
            by_scene['W4_hot'].append(log(yh) - log(xh))
            by_scene['W4_cold'].append(log(xc) - log(yc))
            # Guardrails (positive ratio = better)
            xe2e, ye2e = (r['scenes']['W4']['w4_cold_e2e_ms'] for r in (first, second))
            xgt, ygt = (r['guardrails']['G64K']['ttft_ms'] for r in (first, second))
            xgd, ygd = (r['guardrails']['G64K']['decode_tokens_per_s'] for r in (first, second))
            xitl = max(first['scenes'][s]['p95_itl_ms'] for s in ('W3', 'W4', 'W5'))
            yitl = max(second['scenes'][s]['p95_itl_ms'] for s in ('W3', 'W4', 'W5'))
            require(all(type(v) in (int, float) and isfinite(v) and v > 0
                        for v in (xe2e, ye2e, xgt, ygt, xgd, ygd, xitl, yitl)), 'invalid E0 guardrail values')
            by_guardrail['w4_cold_e2e'].append(log(xe2e) - log(ye2e))
            by_guardrail['g64k_ttft'].append(log(xgt) - log(ygt))
            by_guardrail['g64k_decode'].append(log(ygd) - log(xgd))
            by_guardrail['p95_itl'].append(log(xitl) - log(yitl))
            z = fsum(by_scene[s][-1] for s in score_keys) / len(score_keys)
            pair_log.append(dict(pair=seq, order=[a, b],
                                 scenes_log_ratio={s: by_scene[s][-1] for s in score_keys},
                                 guardrail_log_ratio={g: by_guardrail[g][-1] for g in GUARDRAIL_KEYS},
                                 equal_scene_geomean_ratio=exp(z), equal_scene_log_ratio=z))
        global_peak = max(abs(r['equal_scene_log_ratio']) for r in pair_log)
        scene_peaks = {s: max(map(abs, by_scene[s])) for s in score_keys}
        guardrail_peaks = {g: max(map(abs, by_guardrail[g])) for g in GUARDRAIL_KEYS}
        max_scene_v2 = 0.08
        stable = (global_peak <= MAX_GLOBAL and max(scene_peaks.values()) <= max_scene_v2
                  and max(guardrail_peaks.values()) <= 0.25)
        precision_targets = {s: workbuddy_v2_precision_target(by_scene[s]) for s in WORKBUDDY_V2_PRIMARY_SCENES}
        return dict(
            kind='same_node_engineering_repeatability_not_confidence_interval',
            stable=stable, pairs=pair_log,
            observed_max_abs_log_ratio=global_peak,
            observed_max_abs_scene_log_ratio=scene_peaks,
            observed_max_abs_guardrail_log_ratio=guardrail_peaks,
            descriptive_log_envelope=dict(
                equal_scene=max(FLOOR, global_peak),
                scenes={s: max(FLOOR, z) for s, z in scene_peaks.items()},
            ) if stable else None,
            guardrail_log_envelope={g: max(FLOOR, z) for g, z in guardrail_peaks.items()} if stable else None,
            scene_precision_targets=precision_targets if stable else None,
            formal_w_or_holdout=False,
        )
    by_scene = {s:[] for s in SCENES}
    pair_log = []
    for seq, ((a,b), (first,second)) in enumerate(zip(ORDER,pairs),1):
        require((first['arm'],second['arm'])==(a,b) and all(set(row['scenes'])==set(SCENES) for row in (first,second)),
                'unexpected timing arm order or incomplete scenes')
        for s in SCENES:
            x,y = (r['scenes'][s]['dev_score'] for r in (first,second))
            require(all(type(v) in (int,float) and isfinite(v) and v>0 for v in (x,y)), 'invalid E0 development score')
            if s=='W3' and ('w3_requests' in first or 'w3_requests' in second):
                ratio,_ = paired_w3_ttft(first['w3_requests'],second['w3_requests'])
                by_scene[s].append(log(ratio))
            else:
                by_scene[s].append(log(y)-log(x))
        # Match the pinned GB10 receipt on both Python 3.11 and 3.13.
        # Do not relax exact replay or round the noise envelope.
        z = fsum(by_scene[s][-1] for s in SCENES)/len(SCENES)
        pair_log.append(dict(pair=seq,order=[a,b],scenes_log_ratio={s:by_scene[s][-1] for s in SCENES},
                             equal_scene_geomean_ratio=exp(z),equal_scene_log_ratio=z))
    global_peak = max(abs(r['equal_scene_log_ratio']) for r in pair_log)
    scene_peaks = {s:max(map(abs,by_scene[s])) for s in SCENES}
    stable = global_peak<=MAX_GLOBAL and max(scene_peaks.values())<=MAX_SCENE
    return dict(kind='same_node_engineering_repeatability_not_confidence_interval',stable=stable,pairs=pair_log,
                observed_max_abs_log_ratio=global_peak,observed_max_abs_scene_log_ratio=scene_peaks,
                descriptive_log_envelope=dict(equal_scene=max(FLOOR,global_peak),
                    scenes={s:max(FLOOR,z) for s,z in scene_peaks.items()}) if stable else None,
                formal_w_or_holdout=False)


def execute(root, settings, reg_pin, plan, pin):
    pair = Path(plan['baseline'])
    prior_ref = plan.get('build_registration')
    natural_eos = plan.get('timing_mode') == 'natural_eos'
    chat = plan.get('chat_workload')
    expected = draft(settings,reg_pin,pair,plan['baseline_sha256'],build_registration=prior_ref,
                     natural_eos=natural_eos,chat=chat,trajectory=plan.get('trajectory_calibration'))
    require(node.canonical(expected)==node.canonical(plan),'repeatability plan differs; no timing dispatched')
    prior, prior_pin = settings, reg_pin
    if prior_ref is not None:
        prior_pin = prior_ref['sha256']
        prior = control_run.read_pinned(Path(prior_ref['path']),prior_pin)
    write(root/'noise-plan.json',dict(sha256=pin,plan=plan))
    contract,e0,backend,service,exclusive,collectors,_,calibration = node.setup_baseline(root,settings,reg_pin)
    timing_mode = {}
    if natural_eos:
        timing_mode = dict(timing_mode='natural_eos',
                           timing_workload_sha256=collectors['develop_measure'].enable_natural_eos(chat))
        if chat:
            from evolveengine.v3_level3_eos import PROTOCOL
            require(settings.get('eos_contract') == PROTOCOL, 'chat calibration requires registered EOS contract')
            timing_mode['eos_contract'] = PROTOCOL
            timing_mode.update(chat_corpus_sha256=chat['sha256'], enable_thinking=True)
    deployments = {}
    for arm in ('e0a','e0b'):
        native, receipt = replay_e0.restore_build(pair.parent/f'{arm}-develop_build.json',
                plan['build_quality_measure_pins'][arm]['develop_build'],contract=contract,e0=e0,backend=backend,
                registration=prior,registration_pin=prior_pin,arm=arm)
        deployment = stage_deployment(native,root/f'{arm}-deployment')
        require(deployment.identity==receipt['data']['deployment'],'replayed E0 wheel deployment differs')
        deployments[arm] = deployment
    lane=None
    if 'trajectory_calibration' in plan:
        from evolveengine.v3_level3_eos import grade
        from evolveengine.v3_level3_trajectory import teacher_plan
        from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector,scripts
        ref=plan['trajectory_calibration']; quality=collectors['develop_quality'];lane=TrajectoryCollector(quality,scripts(HERE))
        lane.activate(ref,reg_pin,e0.digest)
        require({d.identity for d in deployments.values()}==lane.teacher_deployments,'calibration and restored deployments differ')
        d3_pin = settings.get('anchor_d3_sha256')
        r1_pin = settings.get('anchor_r1_sha256')
        lc_anchor = Path(settings['anchor_long_ctx']) if 'anchor_long_ctx' in settings else None
        lc_sha = settings.get('anchor_long_ctx_sha256')
        quality.enable_trajectory(ref['sha256'],lane.code,
            lambda rows:grade(rows,Path(settings['anchor_d3']),Path(settings['anchor_r1']),paired=True,
                              d3_manifest=d3_pin,r1_manifest=r1_pin,
                              long_ctx_anchor=lc_anchor,long_ctx_sha256=lc_sha))
        timing_mode['trajectory_calibration_sha256']=ref['sha256']
    new_quality = {}
    if prior_ref is not None:
        for arm in ('e0a','e0b'):
            op = 'develop_quality'
            start = monotonic()
            cap = settings['node_ms'][op]
            until = start+cap/1000
            archive = root/f'quality-{arm}'
            with exclusive(start+cap/1000*(2 if lane else 1)):
                with service.start(deployments[arm],'ee-l3-job-noise-quality-'+token_hex(8),until) as running:
                    obs = collectors[op](op,running,archive,until)
                    if lane:
                        cp=lane.collect(running,archive/'trajectory-candidate',until)
                if lane:
                    c,_=lane.read(archive/'trajectory-candidate',cp,deployments[arm]);jobs=teacher_plan(c)
                    teacher=deployments['e0b' if arm=='e0a' else 'e0a'];until=min(start+2*cap/1000,monotonic()+cap/1000)
                    with service.start(teacher,'ee-l3-noise-teacher-'+token_hex(8),until) as live:
                        tp=lane.collect(live,archive/'trajectory-teacher',until,jobs=jobs)
                    obs=obs(lane.screened(archive,(cp,tp),deployments[arm],teacher))
            verdict = validate_evidence(archive,obs.policy,{op:obs.pin}).data['operations'][op]
            elapsed = int((monotonic()-start)*1000)
            receipt = write(root/f'quality-{arm}.json',dict(arm=arm,engine=e0.digest,deployment=deployments[arm].identity,
                            node_ms=elapsed,observation_sha256=obs.pin,result=asdict(verdict)))
            require(elapsed<=cap*(2 if lane else 1) and verdict.status=='valid','new-profile E0 quality incomplete; no timing dispatched')
            new_quality[arm] = dict(receipt=receipt,node_ms=elapsed)
        quality_baseline = write(root/'quality-baseline.json', dict(kind=control_run.REPLAYED_QUALITY_KIND,
                engine=e0.digest, registration_sha256=reg_pin, plan_sha256=pin, memory_calibration=calibration,
                receipts={arm: row['receipt'] for arm, row in new_quality.items()}, candidate_residency_gate=False))
    pairs = []
    op = 'develop_measure'
    cap = settings['node_ms'][op]
    for i, order in enumerate(ORDER,1):
        out = []
        witnesses = []
        # Exclude other work between the two service/observer windows of each pair.
        with exclusive(monotonic()+2*cap/1000):
            for arm in order:
                start = monotonic()
                until = start+cap/1000
                archive = root/f'pair-{i}-{arm}'
                with service.start(deployments[arm],'ee-l3-job-noise-'+token_hex(8),until) as running:
                    obs = collectors[op](op,running,archive,until,pair_index=i)
                verdict = validate_evidence(archive,obs.policy,{op:obs.pin}).data['operations'][op]
                elapsed = int((monotonic()-start)*1000)
                receipt = dict(arm=arm,engine=e0.digest,deployment=deployments[arm].identity,node_ms=elapsed,
                               observation_sha256=obs.pin,result=asdict(verdict))
                receipt_pin = write(root/f'pair-{i}-{arm}.json',receipt)
                require(elapsed<=cap and verdict.status=='observed','E0 timing repeat incomplete; no envelope')
                summary = dict(arm=arm,scenes=verdict.data['scenes'],receipt=receipt_pin,node_ms=elapsed)
                if 'guardrails' in verdict.data:
                    summary['guardrails'] = verdict.data['guardrails']
                if chat:
                    witnesses.append(verdict.data['_private_replay'])
                    if 'w3_requests' in verdict.data:
                        summary['w3_requests'] = verdict.data['w3_requests']
                    summary['timed_request_outputs'] = verdict.data['timed_request_outputs']
                if 'timing_text' in settings:
                    summary.update(measurement_pair=verdict.data['measurement_pair'],
                                   measurement_inputs_sha256=verdict.data['measurement_inputs_sha256'],
                                   timed_outputs_sha256=verdict.data['timed_outputs_sha256'])
                out.append(summary)
        if 'timing_text' in settings:
            require(out[0]['measurement_pair'] == out[1]['measurement_pair'] == i and
                    out[0]['measurement_inputs_sha256'] == out[1]['measurement_inputs_sha256'] and
                    (chat or out[0]['timed_outputs_sha256'] == out[1]['timed_outputs_sha256']), 'E0 arms differ in inputs or timed outputs')
        if chat:
            first,second = witnesses
            require(first['service']!=second['service'] and first['run']!=second['run'] and
                    set(first['rids']).isdisjoint(second['rids']),'calibration requires independent services and requests')
            agreement=replay_agreement(first['tokens'],second['tokens'])
            work = calibration_output_work(*out, chat['sha256'])
            require(all(r['comparable'] for r in work.values()), 'E0 arms differ in output work')
            for arm in out:
                arm['paired_replay_agreement'] = agreement
                arm['paired_output_work'] = work
                arm['replay_protocol'] = collectors[op].policy.replay_protocol
        pairs.append(out)
    quality = dict(replayed_quality=new_quality, build_registration_sha256=prior_pin,
                   quality_baseline_sha256=quality_baseline) if new_quality else {}
    return write(root/'result.json',dict(plan_sha256=pin,registration_sha256=reg_pin,receipts=pairs,
                 **(timing_mode or dict(timing_workload_sha256=settings['measure_workload_sha256'])),
                 node_ms=sum(row['node_ms'] for pair in pairs for row in pair)+sum(r['node_ms'] for r in new_quality.values()),
                 **quality,**summarize(pairs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    for mode in ('plan','run'):
        args = modes.add_parser(mode)
        args.add_argument('--registration',required=True,type=Path)
        args.add_argument('--registration-sha256',required=True)
        if mode=='plan':
            args.add_argument('--natural-eos',action='store_true')
            args.add_argument('--trajectory-calibration',type=Path)
            args.add_argument('--trajectory-calibration-sha256')
            args.add_argument('--chat-workload',type=Path)
            args.add_argument('--chat-workload-sha256')
            args.add_argument('--pair',required=True,type=Path)
            args.add_argument('--pair-sha256',required=True)
            args.add_argument('--build-registration',type=Path)
            args.add_argument('--build-registration-sha256')
        else:
            args.add_argument('report_dir',type=Path)
            args.add_argument('--plan',required=True,type=Path)
            args.add_argument('--plan-sha256',required=True)
    args = parser.parse_args()
    os.umask(0o077)
    settings = node.checked_inputs(args.registration,args.registration_sha256)
    if args.mode=='plan':
        if bool(args.build_registration)!=bool(args.build_registration_sha256):
            parser.error('prior registration and fingerprint must be supplied together')
        prior = dict(path=str(args.build_registration),sha256=args.build_registration_sha256) if args.build_registration else None
        if bool(args.chat_workload)!=bool(args.chat_workload_sha256):
            parser.error('chat workload and fingerprint must be supplied together')
        chat = dict(path=str(args.chat_workload),sha256=args.chat_workload_sha256) if args.chat_workload else None
        if bool(args.trajectory_calibration)!=bool(args.trajectory_calibration_sha256): parser.error('expected trajectory path and hash')
        ref=dict(path=str(args.trajectory_calibration),sha256=args.trajectory_calibration_sha256) if args.trajectory_calibration else None
        plan = draft(settings,args.registration_sha256,args.pair,args.pair_sha256,
                     build_registration=prior,natural_eos=args.natural_eos,chat=chat,trajectory=ref)
        sys.stdout.buffer.write(node.canonical(plan)+b'\n')
        return
    plan = control_run.read_pinned(args.plan,args.plan_sha256)
    root = node.fresh(args.report_dir)
    try:
        print(execute(root,settings,args.registration_sha256,plan,args.plan_sha256))
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'incomplete.json',dict(status='incomplete_no_noise_envelope',kind=type(exc).__name__,reason=str(exc)[:400]))
        raise


if __name__=='__main__':
    main()
