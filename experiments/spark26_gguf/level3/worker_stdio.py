"""Fixed, trusted GB10 Worker entrypoint for calibrated engineering lanes.

Use authenticated SSH without a PTY; stdout is the canonical Worker channel.
The Agent neither starts this process nor has access to the channel or file paths.
Uncalibrated engineering residency limits must not qualify candidates.
This entrypoint does not admit Q/T or confer full Level 3 launch approval.
"""
import argparse
from contextlib import contextmanager
import json
from dataclasses import asdict, replace
import os
from math import isfinite
from pathlib import Path
from secrets import token_hex
from time import monotonic
import select
import signal
import sys
from threading import Event, Thread

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))

from experiments.spark26_gguf.level3 import control_run, native_controls, replay_e0
from experiments.spark26_gguf.level3.node_preflight import (
    E0_MEMORY_GUARD_BYTES, canonical, checked_inputs, digest, fresh, need, setup_baseline, write,
)
from evolveengine.v3_level3_weight_audit import POLICIES as WEIGHT_AUDITS, POLICY as WEIGHT_AUDIT
from evolveengine.v3_level3_artifacts import DevelopmentArtifacts
from evolveengine.v3_level3_worker import NativeWorker, WorkerResult
from evolveengine.v3_level3_collect import QUALITY_CASES, stage_inputs
from evolveengine.v3_level3_deploy import GRAPH_PROFILES, stage_deployment
from evolveengine.v3_level3_evidence import file_sha256
from evolveengine.v3_level3_run import SCENES
from evolveengine.v3_level3_diagnostic import DockerDiagnostic, ENGINEERING_MS, MAX_MS, MIN_MS
from evolveengine.v3_level3_workload import EXTENDED_PAIRS, profile_inputs
from evolveengine.v3_level3_worker_channel import WorkerLease, serve

SCHEMA = 'level3-gb10-engineering-worker-v2'
E0_ONLY = 'e0_only'
NATIVE_DEVELOPMENT = 'calibrated_native_development'
ENGINEERING_EXPLORATION = 'graph_engineering_exploration'


def authorization(path, pin):
    row = control_run.read_pinned(path,pin,cap=65536)
    fields = {'schema','baseline_sha256','node_ms','jobs','max_per_op','engine_scope','controls'}
    scope = row.get('engine_scope')
    need(set(row)==fields and row['schema']==SCHEMA and scope in (E0_ONLY,NATIVE_DEVELOPMENT,ENGINEERING_EXPLORATION)
         and control_run._sha(row['baseline_sha256']), 'unregistered Worker authorization')
    need(scope == ENGINEERING_EXPLORATION or (row['jobs'] is not None and row['node_ms'] is not None),
         'only engineering authorization supports metering without cumulative caps')
    controls = row['controls']
    if scope==NATIVE_DEVELOPMENT:
        need(type(controls) is dict and set(controls)=={'plan','plan_sha256','result','result_sha256'}
             and all(type(controls[k]) is str and Path(controls[k]).is_absolute() for k in ('plan','result'))
             and all(control_run._sha(controls[k]) for k in ('plan_sha256','result_sha256')),
             'candidate scope requires pinned Native controls')
    elif scope == ENGINEERING_EXPLORATION:
        need(type(controls) is dict and {'replay','replay_sha256'} <= set(controls) <=
             {'replay','replay_sha256','profile_artifacts','natural_workload_sha256','chat_workload','weight_audit','trajectory_calibration','split_quality'}
             and type(controls['replay']) is str and Path(controls['replay']).is_absolute()
             and control_run._sha(controls['replay_sha256']), 'engineering scope requires a pinned Graph E0 replay')
        if 'natural_workload_sha256' in controls:
            need(row['jobs'] is None and control_run._sha(controls['natural_workload_sha256'])
                 and 'develop_profile' in row['max_per_op'], 'invalid natural workload registration')
        if 'weight_audit' in controls:
            need(controls['weight_audit'] in WEIGHT_AUDITS and 'chat_workload' in controls, 'unsupported weight audit policy')
        if 'trajectory_calibration' in controls:
            ref=controls['trajectory_calibration']
            need('chat_workload' in controls and type(ref) is dict and set(ref)=={'path','sha256'} and
                 Path(ref['path']).is_absolute() and control_run._sha(ref['sha256']),'invalid trajectory calibration reference')
        if 'split_quality' in controls:
            sp=controls['split_quality']
            need('trajectory_calibration' in controls and type(sp) is dict and set(sp)=={'path','sha256'} and
                 type(sp['path']) is str and Path(sp['path']).is_absolute() and control_run._sha(sp['sha256']),
                 'invalid split quality registration reference')
        if 'chat_workload' in controls:
            chat = controls['chat_workload']
            need('natural_workload_sha256' in controls and type(chat) is dict and set(chat) == {'path','sha256'}
                 and type(chat['path']) is str and Path(chat['path']).is_absolute() and control_run._sha(chat['sha256']),
                 'invalid thinking-enabled chat workload registration')
        if 'profile_artifacts' in controls:
            need('chat_workload' not in controls, 'unscoped historical traces cannot be published to a chat exploration')
            ref = controls['profile_artifacts']
            need(type(ref) is dict and set(ref)=={'manifest','sha256'} and
                 type(ref['manifest']) is str and Path(ref['manifest']).is_absolute() and control_run._sha(ref['sha256']),
                 'invalid published fixture artifact manifest')
    else:
        need(controls is None,'E0 scope cannot use candidate controls')
    return row['baseline_sha256'],WorkerLease(row['node_ms'],row['jobs'],row['max_per_op']),scope,controls


def candidate_ceiling(settings, registration, contract, e0, calibration, controls):
    """Admit only the exact gate exercised by the completed same-node controls."""
    plan_pin = controls['plan_sha256']
    plan = control_run.read_pinned(Path(controls['plan']),plan_pin)
    result_file = Path(controls['result'])
    result = control_run.read_pinned(result_file,controls['result_sha256'])
    archived_plan_pin = digest(canonical(dict(sha256=plan_pin,plan=plan))+b'\n')
    control_run.read_pinned(result_file.parent/'control-plan.json',archived_plan_pin)
    variants = native_controls.make(contract)
    expected = control_run.draft(settings,registration,Path(plan['baseline_pair']),plan['pair_sha256'],variants)
    need(canonical(expected)==canonical(plan) and variants['e0']==e0 and calibration['source']==plan['rule']['memory_source'],
         'candidate gate does not match registered inputs or implementation')
    receipts, files = result.get('receipts'),result.get('receipt_files')
    order = {f'{role}/{stage}' for role,stages in control_run.ORDER.items() for stage in stages}
    limit = plan['rule']['sample_ceiling_bytes']
    need(result.get('kind')=='engineering_controls_not_final_admission' and result.get('sample_ceiling_bytes')==limit
         and type(receipts) is dict and set(receipts)==order
         and all(control_run._sha(x) for x in receipts.values()) and len(set(receipts.values()))==len(order)
         and type(files) is dict and set(files)==order and all(control_run._sha(x) for x in files.values())
         and type(result.get('node_ms')) is int and 0 < result['node_ms']<=plan['max_total_node_ms']
         and result.get('holds_no_qt') is True and result.get('paid_api_calls')==0, 'incomplete candidate gate evidence')
    for role,stages in control_run.ORDER.items():
        for stage in stages:
            key = f'{role}/{stage}'
            row = control_run.read_pinned(result_file.parent/f'{role}-{stage}.json',files[key],cap=128*1024)
            need(row['evidence']==receipts[key] and row['engine']==variants[role].digest,
                 'control receipt identity drift')
            if stage.startswith('quality_'):
                control_run.check_quality(role,WorkerResult(**row),limit)
    return limit


def graph_replay_peak(replay, *, telemetry_only=False):
    """Check the scoped E0 replay without claiming full acceptance or repeatability."""
    required = {phase for phase in ('decode', 'prefill')
                if f'--disable-{phase}-cuda-graph' in GRAPH_PROFILES.get(replay['profile'], ())}
    captures = replay['capture_log_witnesses']
    need(replay['kind'] == 'trusted_engineering_replay_not_E0_E0_pair' and required
         and control_run._sha(replay['engine']) and all(captures.get(phase) for phase in required),
         'missing Graph E0 launch evidence')
    operations = replay['verdicts']
    quality, measure = operations['develop_quality'], operations['develop_measure']
    q, m = quality['data'], measure['data']
    peak = q['observed_peak_sample_bytes']
    from evolveengine.v3_level3_collect import LONG_CTX_CASE
    from evolveengine.v3_level3_scoring import WORKBUDDY_V2_REPLAY_SCENES
    expected_q = dict.fromkeys((*QUALITY_CASES, LONG_CTX_CASE), 'pass') if LONG_CTX_CASE in q.get('case_outcomes', {}) else dict.fromkeys(QUALITY_CASES, 'pass')
    expected_s = set(WORKBUDDY_V2_REPLAY_SCENES) if set(m.get('scenes', {})) == set(WORKBUDDY_V2_REPLAY_SCENES) else set(SCENES)
    need(quality['status'] == 'valid' and measure['status'] == 'observed'
         and q['engine'] == m['engine'] == replay['engine']
         and q['case_outcomes'] == expected_q
         and type(peak) is int and peak > 0 and (telemetry_only or peak < E0_MEMORY_GUARD_BYTES - control_run.MARGIN_BYTES)
         and m['measurement_pair'] == 0 and m['profiler_enabled'] is False
         and all(control_run._sha(m[key]) for key in ('measurement_inputs_sha256','timed_outputs_sha256'))
         and set(m['scenes']) == expected_s
         and all(type(row['dev_score']) in (int, float) and isfinite(row['dev_score']) and row['dev_score'] > 0
                 for row in m['scenes'].values()), 'Graph E0 quality or real-text timing is incomplete')
    return peak


def engineering_ceiling(settings, contract_engine, calibration, controls):
    path = Path(controls['replay'])
    replay = control_run.read_pinned(path, controls['replay_sha256'])
    peak = graph_replay_peak(replay, telemetry_only=controls.get('weight_audit') in WEIGHT_AUDITS)
    registration = (path.parent/'registration.json').read_bytes()
    prior = json.loads(registration)
    prior_config = prior['config']
    # The stdio admission and authenticated transport can change without
    # changing the build, deployment, observation or host bindings.
    transport = {'experiments/spark26_gguf/level3/worker_stdio.py', 'evolveengine/v3_level3_worker_channel.py'}
    before, after = prior_config['implementation_sha256'], settings['implementation_sha256']
    need(digest(canonical(prior_config)+b'\n') == prior['sha256']
         and prior_config == (settings | {'implementation_sha256': before})
         and before.keys() == after.keys() and transport <= before.keys()
         and all(before[key] == after[key] for key in before.keys() - transport)
         and 'timing_text' in settings and replay['profile'] == settings['launch_profile']
         and replay['engine'] == contract_engine.digest
         and replay['verdicts']['develop_quality']['data']['memory_source'] == calibration['source'],
         'Graph replay does not match the registered node, engine and observation protocol')
    for operation in ('develop_quality', 'develop_measure'):
        record = control_run.read_pinned(path.parent/f'{operation}.json', replay['receipts'][operation])
        need(record['result'] == replay['verdicts'][operation]
             and record['observation_sha256'] == record['result']['data']['receipt_id']
             and record['observation_sha256'] == digest((path.parent/operation/operation/'receipt.json').read_bytes()),
             'Graph replay receipt differs from its archived observation')
    return None if controls.get('weight_audit') in WEIGHT_AUDITS else peak + control_run.MARGIN_BYTES


class EngineeringWorker(NativeWorker):
    """Revalidate and stage the E0 wheel already exercised by this Graph replay."""
    component_builds = True

    def __init__(self, *args, settings, replay_path, profile_artifacts=None, profile_timing=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.replay_path = replay_path
        self.artifacts = DevelopmentArtifacts()
        self.published_diagnostics = set()
        self.profile_timing = profile_timing
        self.profiles = {}
        if profile_artifacts is not None:
            manifest = Path(profile_artifacts['manifest'])
            entries = control_run.read_pinned(manifest,profile_artifacts['sha256'],cap=65536)
            self.artifacts.import_profile(manifest.parent, entries)

    def artifact_query(self, **args):
        # Only diagnostic stdout/stderr reach this publication lane. Judge inputs,
        # raw quality measurements, controller files and held-out data never do.
        for receipt,(result,root) in self.diagnostics.items():
            if receipt in self.published_diagnostics:
                continue
            for name,expected in (('stdout',result.stdout_sha256),('stderr',result.stderr_sha256)):
                if expected:
                    self.artifacts.publish(root/('observer.'+name), kind='diagnostic_'+name,
                                           label='diagnostic '+receipt[:12]+' '+name, expected=expected)
            self.published_diagnostics.add(receipt)
        return self.artifacts.query(**args)

    def profile(self, engine, build, cap, workload):
        """Non-scoring device capture using a fresh isolated service and a trusted traffic client."""
        from experiments.spark26_gguf.level3.profile_control import DURATION, ProfileService

        timing = self.profile_timing
        need(timing is not None and cap >= 420000 and not self.quarantined, 'profile lane unavailable')
        need(build.operation == 'develop_build' and build.status == 'built' and build.engine == engine.digest
             and self.verify(build), 'profiling requires a verified deployment')
        need(type(workload) is dict and workload.keys() <= {'scenes','windows','inputs'}, 'invalid profile workload')
        deployment = self.deployments[build.evidence]
        began = monotonic()
        deadline = began + cap/1000
        root = self.root / ('profile-' + token_hex(12))
        root.mkdir(mode=0o700)
        try:
            plan = json.loads(timing.workload)
            scenes = workload.get('scenes', list(plan['scenes']))
            count = workload.get('windows', 1)
            need(type(scenes) is list and scenes and all(type(s) is str and s in plan['scenes'] for s in scenes)
                 and len(set(scenes)) == len(scenes) and type(count) is int and 1 <= count <= plan['timed_waves'],
                 'invalid profile scenes or windows')
            custom = workload.get('inputs')
            if custom is not None:
                need(type(custom) is dict and set(custom) == set(scenes), 'custom inputs must match profile scenes')
                for scene in scenes:
                    waves = custom[scene]
                    shape = plan['scenes'][scene]
                    need(type(waves) is list and len(waves) == count, 'invalid profile request window count')
                    for wave in waves:
                        need(type(wave) is list and len(wave) == len(shape), 'invalid profile request count')
                        for ids, (length, _) in zip(wave, shape):
                            need(type(ids) is list and len(ids) == length and
                                 all(type(t) is int and 0 <= t < 131072 for t in ids), 'invalid profile token IDs')
                inputs = custom
            else:
                inputs = profile_inputs(timing.schedule.documents, {s:plan['scenes'][s] for s in scenes},
                                        count, timing.schedule.seed, getattr(timing, 'chat_schedule', None))
            plan.update(scenes={s: plan['scenes'][s] for s in scenes}, warm_waves=0, timed_waves=count)
            plan.pop('w3_visible_answer',None)
            plan.pop('replay_protocol',None)  # no raw timing-output witnesses in publishable diagnostics
            traffic = root / 'inputs'
            stage_inputs(traffic, {**timing.files, 'workload.json':canonical(plan), 'inputs.json':canonical(inputs)})
            script = 'observe_workbuddy_timing.py' if 'observe_workbuddy_timing.py' in timing.files else 'observe_text_timing.py'
            service = ProfileService(timing.service_spec, mode='e0_graph', docker=timing.host.docker,
                                     diagnostics=root/'service-diagnostics')
            with self.exclusive(deadline), service.start(deployment, 'ee-l3-job-'+root.name, deadline) as live:
                timing.observer(live, traffic, root, min(deadline-90, began+DURATION-5),
                                script=script)
                service.collect_trace(live, root, deadline)
            manifest = root/'development-artifacts.json'
            files = json.loads(manifest.read_bytes())
            self.artifacts.import_profile(root, files)
            data = dict(non_scoring=True, artifacts=files['files'], manifest_sha256=file_sha256(manifest),
                        workload_sha256=digest(canonical(dict(plan=plan, inputs=inputs))), verified_removed=True)
            status = 'completed_non_scoring'
        except (ValueError, OSError) as exc:
            from evolveengine.v3_level3_build import CleanupUncertain
            if isinstance(exc, CleanupUncertain):
                self.quarantined = True
                raise
            data = dict(non_scoring=True, failure=type(exc).__name__, detail=str(exc)[:240])
            status = 'incomplete_non_scoring'
        elapsed = int((monotonic()-began)*1000)
        need(elapsed <= cap, 'profile lease overrun')
        fields = dict(engine=engine.digest, operation='develop_profile', status=status,
                      node_ms=elapsed, build=build.evidence, data=data)
        evidence = self._save(root/'result.json', fields)
        result = WorkerResult(**fields, evidence=evidence)
        self.profiles[evidence] = (result, root)
        return result

    def sampling(self,engine,build,cap,challenge):
        return self.sampling_lane.run(engine,build,cap,challenge)

    def verify(self, result):
        lane=getattr(self,'sampling_lane',None)
        if lane is not None and result.evidence in lane.records: return lane.verify(result)
        if result.operation != 'develop_profile':
            return super().verify(result)
        try:
            recorded, root = self.profiles[result.evidence]
            return bool(recorded == result and file_sha256(root/'result.json') == result.evidence and
                        (result.status == 'incomplete_non_scoring' or
                         file_sha256(root/'development-artifacts.json') == result.data['manifest_sha256']))
        except (KeyError, OSError, ValueError):
            return False

    def __call__(self, op, engine, build, cap, *, pair_index=None):
        if op != 'develop_build' or engine.edits or engine.digest != self.contract.from_args({'files': {}}).digest:
            return super().__call__(op, engine, build, cap, pair_index=pair_index)
        need(build is None and pair_index is None and cap > 0 and not self.quarantined, 'invalid E0 reuse request')
        started = monotonic()
        replay_root = self.replay_path.parent
        replay_plan = json.loads((replay_root/'replay-plan.json').read_bytes())
        prior_path = Path(replay_plan['prior_build'])
        prior_pin = replay_plan['prior_build_registration_sha256']
        prior = json.loads((prior_path.parent/'registration.json').read_bytes())['config']
        need(digest(canonical(prior)+b'\n') == prior_pin, 'archived build registration fingerprint mismatch')
        replay_e0.check_build_compatibility(self.settings, prior)
        native, archived = replay_e0.restore_build(prior_path, replay_plan['prior_build_sha256'],
                                     contract=self.contract, e0=engine, backend=self.native,
                                     registration=prior, registration_pin=prior_pin)
        for operation in ('develop_quality','develop_measure'):
            record = json.loads((replay_root/f'{operation}.json').read_bytes())
            need(record['build_evidence'] == archived['evidence'], 'Graph replay used another build')
        job = self.root/('job-'+token_hex(12))
        job.mkdir(mode=0o700)
        deployment = stage_deployment(native, job/'deployment')
        need(deployment.identity == archived['data']['deployment'], 'archived E0 deployment differs')
        elapsed = round((monotonic()-started)*1000)
        need(elapsed <= cap, 'E0 artifact verification exceeded its operation limit')
        data = dict(archived['data'], build_mode='verified_e0_reuse', archived_evidence=archived['evidence'],
                    archived_node_ms=archived['node_ms'])
        public = dict(engine=engine.digest, operation=op, status='built', node_ms=elapsed, build=None, data=data)
        evidence = self._save(job/'result.json', dict(public, deployment=deployment.identity, observation_pin=None,
                                                      observation_unavailable=False))
        result = WorkerResult(evidence=evidence, **public)
        self.records[evidence] = (job, result, native, deployment, None, None, False)
        self.deployments[evidence] = deployment
        if self.trajectory_lane is not None:
            reference = self.trajectory_e0 if self.trajectory_e0 is not None else deployment
            need(reference.candidate==engine.digest and reference.identity in self.trajectory_lane.teacher_deployments,
                 'E0 bootstrap differs from calibrated teacher')
            self.trajectory_e0=reference
        return result


@contextmanager
def watch_controller(source):
    """Interrupt an active job when the authenticated SSH writer closes stdin."""
    reader = select.poll()
    faults = select.POLLHUP | select.POLLERR | select.POLLNVAL | select.POLLRDHUP
    reader.register(source.fileno(), faults)
    need(not reader.poll(0), 'controller closed before Worker dispatch')
    done = Event()

    def forward_eof():
        while not done.is_set():
            if reader.poll(100):
                if not done.is_set():
                    os.kill(os.getpid(), signal.SIGUSR1)
                return

    waiter = Thread(target=forward_eof, daemon=True, name='level3-ssh-eof')
    waiter.start()
    try:
        yield
    finally:
        done.set()
        waiter.join()


def split_registration(controls, registration):
    ref=(controls or {}).get('split_quality')
    if ref is None: return None
    from experiments.spark26_gguf.level3.live_split_quality import read_registration
    return read_registration(Path(ref['path']),ref['sha256'],registration,controls['trajectory_calibration']['sha256'])


def make_worker(root, settings, registration, lease, lease_pin, *, scope, controls):
    extended = scope == ENGINEERING_EXPLORATION and lease.jobs is None
    limits = settings['node_ms'] | {'develop_diagnostic': ENGINEERING_MS if extended else MAX_MS}
    if extended and 'natural_workload_sha256' in controls:
        limits['develop_profile'] = 7200000
    need(all(cap <= limits.get(op,0) for op,cap in lease.max_per_op.items()) and
         lease.max_per_op.get('develop_diagnostic',MIN_MS) >= MIN_MS,
         'Worker lease outside registered lane limits')
    contract,e0,backend,svc,pinned,collectors,_,calibration = setup_baseline(root,settings,registration)
    ceiling = None
    if scope==NATIVE_DEVELOPMENT:
        ceiling = candidate_ceiling(settings,registration,contract,e0,calibration,controls)
    elif scope == ENGINEERING_EXPLORATION:
        ceiling = engineering_ceiling(settings, e0, calibration, controls)
    if ceiling is not None or scope == ENGINEERING_EXPLORATION:
        quality = collectors['develop_quality']
        quality.policy = replace(quality.policy,memory_limit_bytes=ceiling)
    diagnostic = DockerDiagnostic(svc.spec,docker=svc.docker,exclusive=pinned,
                                  retain_artifacts=scope==ENGINEERING_EXPLORATION, extended=extended)
    worker_type, options = NativeWorker, {}
    if scope == ENGINEERING_EXPLORATION:
        worker_type = EngineeringWorker
        options = dict(settings=settings, replay_path=Path(controls['replay']),
                       profile_artifacts=controls.get('profile_artifacts'))
        if lease.jobs is None:
            timing = collectors['develop_measure']
            need(timing.schedule is not None, 'objective timing requires a pinned real-text corpus')
            timing.schedule.maximum = EXTENDED_PAIRS
            options['timing_max'] = EXTENDED_PAIRS
            if 'natural_workload_sha256' in controls:
                if controls.get('chat_workload'):
                    from evolveengine.v3_level3_eos import PROTOCOL
                    need(settings.get('eos_contract') == PROTOCOL, 'chat exploration requires registered streaming EOS quality')
                need(timing.enable_natural_eos(controls.get('chat_workload')) == controls['natural_workload_sha256'],
                     'Worker timing workload differs')
                options['profile_timing'] = timing
                quality = collectors['develop_quality']
                quality.sampling_enabled = 'chat_workload' not in controls
                quality.policy = replace(quality.policy, sampling_required=quality.sampling_enabled)
    worker = worker_type(root/'development',contract,backend,svc,exclusive=pinned,diagnostic=diagnostic,
                         collect=lambda op,*a,**kw:collectors[op](op,*a,**kw), **options)
    if controls and controls.get('chat_workload'):
        from evolveengine.v3_level3_eos import grade
        from evolveengine.v3_level3_trajectory_lane import TrajectoryCollector, scripts
        ref=controls.get('trajectory_calibration'); need(ref is not None,'chat worker requires fresh paired trajectory calibration')
        quality=collectors['develop_quality']
        registered=split_registration(controls,registration)
        if registered is None:
            lane=TrajectoryCollector(quality,scripts(HERE))
            lane.activate(ref,registration,e0.digest)
            d3_pin = settings.get('anchor_d3_sha256')
            r1_pin = settings.get('anchor_r1_sha256')
            lc_anchor = Path(settings['anchor_long_ctx']) if 'anchor_long_ctx' in settings else None
            lc_sha = settings.get('anchor_long_ctx_sha256')
            quality.enable_trajectory(ref['sha256'],lane.code,
                lambda rows:grade(rows,Path(settings['anchor_d3']),Path(settings['anchor_r1']),paired=True,
                                  d3_manifest=d3_pin,r1_manifest=r1_pin,
                                  long_ctx_anchor=lc_anchor,long_ctx_sha256=lc_sha))
        else:
            from experiments.spark26_gguf.level3.live_split_quality import enable, restore_teacher
            lane=enable(quality,settings,ref,controls['split_quality'],registration,e0,registered)
            worker.trajectory_e0=restore_teacher(root,settings,contract,e0,backend,registered)
        from evolveengine.v3_level3_sampling_lane import SamplingLane
        worker.sampling_lane=SamplingLane(worker,quality)
        worker.trajectory_lane=lane
    write(root/'worker-authorization.json',dict(sha256=lease_pin,registration=registration,engine_scope=scope,
                                               lease=asdict(lease),controls=controls,registered_sample_ceiling_bytes=ceiling))
    return contract,e0,worker


def run(root, settings, registration, lease, lease_pin, source, sink, *, scope, controls, durable=False):
    contract,_,worker=make_worker(root,settings,registration,lease,lease_pin,scope=scope,controls=controls)
    if durable:
        serve(source,sink,contract,worker,lease_pin,lease,e0_only=scope==E0_ONLY)
    else:
        serve(source,sink,contract,worker,lease_pin,lease,e0_only=scope==E0_ONLY,
              monitor=lambda: watch_controller(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_report_dir',type=Path)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--sha256',required=True,help='controller-pinned Worker authorization SHA-256')
    parser.add_argument('--durable', choices=('bridge','server','stop'))
    args = parser.parse_args()
    os.umask(0o077)
    stop_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)

    def stop(signum, _frame):
        # A single interrupt gives the in-flight operation time to unwind.
        for code in stop_signals:
            signal.signal(code, signal.SIG_IGN)
        reason = 'controller disconnected during Worker operation' if signum == signal.SIGUSR1 else 'trusted operator interrupted Worker'
        raise KeyboardInterrupt(reason)

    for code in stop_signals:
        signal.signal(code,stop)
    registration,lease,scope,controls = authorization(args.authorization,args.sha256)
    if args.durable:
        need(scope == ENGINEERING_EXPLORATION and lease.jobs is None, 'durable Worker requires objective scope')
        from evolveengine.v3_level3_worker_spool import SpoolIO, bridge, stop as stop_spool
        if args.durable == 'stop':
            stop_spool(args.new_report_dir,args.sha256)
            return
        if args.durable == 'bridge':
            command = [sys.executable,'-Bu',str(Path(__file__).resolve()),str(args.new_report_dir),
                       '--registration',str(args.registration),'--authorization',str(args.authorization),
                       '--sha256',args.sha256,'--durable','server']
            bridge(args.new_report_dir,args.sha256,command,sys.stdin.buffer,sys.stdout.buffer)
            return
    settings = checked_inputs(args.registration,registration,split_registration=split_registration(controls,registration))
    root = fresh(args.new_report_dir)
    try:
        stream = SpoolIO(root,args.sha256) if args.durable == 'server' else None
        run(root,settings,registration,lease,args.sha256,stream or sys.stdin.buffer,
            stream or sys.stdout.buffer,scope=scope,controls=controls,durable=stream is not None)
    except (Exception,KeyboardInterrupt) as exc:
        write(root/'worker-exit.json',dict(status='stopped',kind=type(exc).__name__,reason=str(exc)[:400]))
        raise
    else:
        write(root/'worker-exit.json',dict(status='closed'))


if __name__=='__main__':
    main()
