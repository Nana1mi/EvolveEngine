"""Join the verified Native build and hardware evidence lanes for the Level 3 Runner.

Collectors are trusted, controller-supplied host functions. Candidate processes
never receive the observation directory, evidence pins, or collector handles.
"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from secrets import token_hex
from time import monotonic

from .v3_harness import canonical, digest
from .v3_level3_build import BuildRequest, CleanupUncertain, verify_artifacts
from .v3_level3_components import ComponentReceipt, build_components, changed_components
from .v3_level3_deploy import stage_deployment
from .v3_level3_evidence import EvidencePolicy, file_sha256, validate_evidence
from .v3_level3_run import OPS, WorkerResult
from .v3_level3_workload import MAX_PAIRS


@dataclass(frozen=True)
class Observation:
    policy: EvidencePolicy
    pin: str


class ObservationUnavailable(RuntimeError):
    """A workload failed without a verdict; lease cleanup remains mandatory."""


def need(ok, why):
    if not ok:
        raise ValueError(why)


def _hex(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _diagnostic_text(raw):
    return ''.join(ch if ch in '\n\t' or ord(ch) >= 32 else '?' for ch in raw.decode('utf-8','replace'))[:16*1024]


def _stdout_preview(archive):
    path = archive/'observer.stdout'
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        return _diagnostic_text(stream.read(16 << 10))


class NativeWorker:
    component_builds = False

    def __init__(self, root, contract, native, service, *, exclusive, collect, diagnostic=None,
                 native_variant='sm100', independent_cold=False, timing_max=MAX_PAIRS):
        root = Path(root).resolve()
        root.mkdir(mode=0o700,parents=False,exist_ok=False)
        need(native_variant in ('sm90','sm100'), 'unregistered Native variant')
        self.root,self.contract,self.native,self.service = root,contract,native,service
        self.exclusive,self.collect,self.variant = exclusive,collect,native_variant
        self.diagnostic = diagnostic
        self.independent_cold = independent_cold is True
        self.timing_max = timing_max
        self.records,self.deployments,self.diagnostics = {},{},{}
        self.built_engines = {}
        self.trajectory_lane = self.trajectory_e0 = None
        self.sampling_reference = None
        self.timing_replays = {}
        self.quarantined = False

    def _save(self, path, payload):
        data = canonical(payload)+b'\n'
        with path.open('xb') as out:
            out.write(data)
        return digest(data)

    def _validate(self, op, archive, observation, expected_weights, deployment, *, check_replay=True):
        need(type(observation) is Observation and type(observation.policy) is EvidencePolicy and _hex(observation.pin),
             'no controller-pinned hardware observation')
        p = observation.policy
        expected_mappings = deployment.native_mapping(self.variant)[1]
        need(p.origin == 'hardware_collection' and p.engine == deployment.candidate
             and p.device_uuid == self.service.spec.gpu_uuid and dict(p.weights)==expected_weights
             and dict(p.loaded_artifacts)==expected_mappings, 'collector policy not bound to hardware deployment')
        check = validate_evidence(archive,p,{op:observation.pin}).data['operations'][op]
        if check.status == 'insufficient_evidence' and 'case_outcomes' not in check.data:
            return check.status, {}
        need(check.status in ('observed','valid','violation','insufficient_evidence'), 'invalid hardware observation verdict')
        allowed = ('scenes','case_outcomes','observed_peak_sample_bytes','profiler_enabled','residency_limits',
                   'weight_observation','kernel_summary','trace_summary','counter_status','counters','measurement_pair',
                   'measurement_inputs_sha256','timed_outputs_sha256','timed_request_outputs',
                   'speculation','timing_mode','timing_workload_sha256','chat_corpus_sha256','replay_protocol','concurrent_requests','w3_requests')
        # This is the only data shown to Pi (never archive paths, stderr, or raw output).
        public = {key:check.data[key] for key in allowed if key in check.data}
        if op == 'develop_measure' and p.natural_eos and not check_replay and p.replay_protocol:
            # Only the recursive host-side caller receives these; never include raw IDs in a WorkerResult.
            public['_private_replay'] = check.data['_private_replay']
        if op == 'develop_measure' and p.natural_eos and check_replay:
            replay, replay_weights = self.timing_replays[archive.parent]
            status, witness = self._validate(op, archive.parent/'authenticity-replay', replay, replay_weights,
                                             deployment, check_replay=False)
            matched = (status == check.status == 'observed' and replay_weights == expected_weights and
                       replay.pin != observation.pin and p.replay_protocol == replay.policy.replay_protocol and
                       witness.get('measurement_inputs_sha256') != public['measurement_inputs_sha256'] and
                       all(witness.get(k) == public[k] for k in ('timing_workload_sha256', 'measurement_pair')))
            proof = dict(method=p.replay_protocol or 'independent_same_engine_replay', replay_receipt=replay.pin)
            if p.replay_protocol and matched:
                primary, independent = check.data['_private_replay'], witness['_private_replay']
                left, right = primary['tokens'], independent['tokens']
                matched = (primary['service'] != independent['service'] and primary['run'] != independent['run'] and
                           set(primary['rids']).isdisjoint(independent['rids']) and
                           set(primary['warm']).isdisjoint(independent['warm']) and
                           left.keys() == right.keys() and all(left[k] and left[k].keys() == right[k].keys() for k in left))
                if matched:
                    from .v3_level3_scoring import replay_agreement
                    proof['agreement'] = replay_agreement(left,right)
            elif not p.replay_protocol and matched:
                matched = witness['timed_request_outputs'] == public['timed_request_outputs']
            public['output_authentication'] = dict(proof, status='verified' if matched else 'mismatch')
            if not matched:
                return 'output_replay_mismatch', public
        if op == 'develop_quality' and 'paired_cancel_numeric' in p.quality_cases:
            need(self.trajectory_lane is not None and self.trajectory_e0 is not None,'trajectory worker not activated')
            record = json.loads((archive/op/'trajectory-verdict.json').read_bytes())
            pins = (record['candidate_receipt'],record['teacher_receipt'])
            need(self.trajectory_lane.screened(archive.parent,pins,deployment,self.trajectory_e0)==record,
                 'stored trajectory verdict failed to replay')
            from .v3_level3_trajectory_lane import trajectory_cases
            outcomes=trajectory_cases(record,self.trajectory_lane.protocol)
            need(all(public['case_outcomes'][k]==v for k,v in outcomes.items()),'trajectory result and quality witnesses differ')
            public['trajectory_screen'] = record
        if op == 'develop_quality' and p.sampling_required and check.status!='insufficient_evidence':
            from .v3_level3_sampling import checked_samples, compare

            sample_file = archive/'develop_quality'/'sampling.json'
            observed = checked_samples(json.loads(sample_file.read_bytes()))
            observed_sha = file_sha256(sample_file)
            if p.engine == self.contract.from_args({'files': {}}).digest:
                if self.sampling_reference is None and check.status == 'valid':
                    self.sampling_reference = sample_file, observed_sha
                public['sampling_screen'] = dict(status='reference' if check.status == 'valid' else 'not_admitted', sample_sha256=observed_sha)
            else:
                need(self.sampling_reference is not None, 'missing independent E0 sampling reference')
                reference, reference_sha = self.sampling_reference
                need(file_sha256(reference) == reference_sha, 'E0 sampling reference changed')
                report = compare(json.loads(reference.read_bytes()), observed)
                public['sampling_screen'] = dict(report, reference_sha256=reference_sha, sample_sha256=observed_sha)
                if report['status'] != 'pass':
                    return 'violation', public
        return check.status, public

    def measure(self, engine, build, cap, pair_index):
        return self('develop_measure', engine, build, cap, pair_index=pair_index)

    def stress(self, engine, build, cap, pair_index):
        return self._execute('develop_measure', engine, build, cap, pair_index=pair_index, stress=True)

    def __call__(self, op, engine, build, cap, *, pair_index=None):
        return self._execute(op,engine,build,cap,pair_index=pair_index)

    def independent_build(self, engine, cap):
        """Controller-only final verification. Bypass bootstrap, parent components and ccache."""
        return self._execute('develop_build',engine,None,cap,independent=True)

    def _execute(self, op, engine, build, cap, *, pair_index=None, independent=False, stress=False):
        need(op in OPS-{'develop_diagnostic'} and type(cap) is int and cap > 0 and not self.quarantined,
             'worker lease unavailable')
        need(pair_index is None or (op == 'develop_measure' and type(pair_index) is int and
                                    0 <= pair_index < self.timing_max), 'invalid worker timing pair')
        options = {} if pair_index is None else dict(pair_index=pair_index)
        if stress:
            options['stress'] = True
        if independent:
            options['independent_cold'] = True
        began = monotonic()
        deadline = began + cap/1000
        job = self.root/('job-'+token_hex(12))
        job.mkdir(mode=0o700)
        self._save(job/'reservation.json',dict(operation=op, candidate=engine.digest, cap_ms=cap,
                                               build=build.evidence if build else None, **options))
        native = deployment = observation = weights = None
        observation_unavailable = False
        try:
            with self.exclusive(deadline):
                if op == 'develop_build':
                    need(build is None or self.verify(build) and build.status == 'built' and build.operation == op,
                         'incremental build lacks its verified parent')
                    native, deployment, public = self._build(engine, build, job, deadline, independent=independent)
                    status = native.status
                    from .v3_level3_startup import PATH, VERSION
                    config = dict(engine.edits).get(PATH,b'')
                    if deployment and config:
                        from .v3_level3_deploy import launch_args
                        deployment=replace(deployment,runtime_config=config)
                        public['runtime']=dict(protocol=VERSION,source_sha256=digest(config),argv=list(launch_args(self.service.spec,deployment)))
                else:
                    need(build is not None and self.verify(build) and build.engine == engine.digest,
                         'job lacks its verified build')
                    deployment = self.deployments[build.evidence]
                    # Observer artifacts are written by a trusted host collector while the
                    # service runs. Only after its verified removal do we accept them.
                    try:
                        with self.service.start(deployment,'ee-l3-'+job.name,deadline) as service:
                            weights = service.model_hashes
                            observation = self.collect(op,service,job/'observation',deadline,**options)
                            if op=='develop_quality' and self.trajectory_lane is not None:
                                lane=self.trajectory_lane
                                need(self.trajectory_e0 is not None and callable(observation),'verified E0 teacher required')
                                cp=lane.collect(service,job/'trajectory-candidate',deadline)
                        if op=='develop_quality' and self.trajectory_lane is not None:
                            from .v3_level3_trajectory_lane import candidate_jobs
                            c,_=lane.read(job/'trajectory-candidate',cp,deployment)
                            tasks=candidate_jobs(c)
                            with self.service.start(self.trajectory_e0,'ee-l3-teacher-'+job.name,deadline) as teacher:
                                tp=lane.collect(teacher,job/'trajectory-teacher',deadline,jobs=tasks)
                            observation=observation(lane.screened(job,(cp,tp),deployment,self.trajectory_e0))
                        if op == 'develop_measure' and observation.policy.natural_eos:
                            with self.service.start(deployment,'ee-l3-replay-'+job.name,deadline) as replay_service:
                                replay = self.collect(op,replay_service,job/'authenticity-replay',deadline,
                                                      authenticity_replay=True,**options)
                                self.timing_replays[job] = replay, replay_service.model_hashes
                    except ObservationUnavailable as exc:
                        observation = None  # a deferred, unfinished collector is not signed evidence
                        self._save(job/'observer-failure.json',dict(kind=type(exc).__name__,reason=str(exc)[:400]))
                        observation_unavailable = True
                        status,public = 'insufficient_evidence', {'failure': 'observer_incomplete'}
                    else:
                        status,public = self._validate(op,job/'observation',observation,weights,deployment)
        except (Exception,KeyboardInterrupt):
            self.quarantined = True
            self._save(job/'uncertain.json',dict(status='uncertain',operation=op,charged_ms=cap))
            raise
        elapsed = int((monotonic()-began)*1000)
        if elapsed > cap:
            self.quarantined = True
            self._save(job/'uncertain.json',dict(status='lease_overrun',operation=op,observed_ms=elapsed,charged_ms=cap))
            raise TimeoutError('worker lease overrun')
        build_id = None if op == 'develop_build' else build.evidence
        fields = dict(engine=engine.digest,operation=op,status=status,node_ms=elapsed,build=build_id,data=public)
        record = dict(**fields,deployment=deployment.identity if deployment else None,
                      observation_pin=observation.pin if observation else None,
                      observation_unavailable=observation_unavailable)
        address = self._save(job/'result.json',record)
        result = WorkerResult(**fields,evidence=address)
        self.records[address] = (job,result,native,deployment,observation,weights,observation_unavailable)
        if op == 'develop_build' and status == 'built':
            self.deployments[address] = deployment
            self.built_engines[engine.digest] = engine
        return result

    def _build(self, engine, parent, job, deadline, *, independent=False):
        same = parent is not None and parent.engine == engine.digest
        mode = 'independent_cold' if independent or self.independent_cold else 'incremental' if same else 'cold'
        request = BuildRequest.from_contract(self.contract, engine, mode=mode)
        if self.component_builds and mode != 'independent_cold' and parent is not None:
            previous = self.built_engines.get(parent.engine)
            if previous is None:
                # The engineering bootstrap may have restored a registered independent E0 build.
                baseline = self.contract.from_args({'files': {}})
                need(parent.engine == baseline.digest, 'unknown parent Engine for component build')
                previous = baseline
            parent_native = self.records[parent.evidence][2]
            deployment = self.deployments[parent.evidence]
            previous = BuildRequest.from_contract(self.contract, previous)
            changes = changed_components(request, previous, self.contract, deployment)
            if changes is not None:
                native, deployment, public = build_components(self.native, request, previous, parent_native,
                                                             deployment, changes, job, deadline)
                public['parent_build'] = parent.evidence
                public['component_receipt'] = self._save(job/'component-build.json', dict(
                    candidate=engine.digest, source=native.source_sha256, status=native.status, **public))
                return native, deployment, public
        native = self.native.build(request, deadline=deadline)
        need(native.candidate_id == engine.digest, 'build bound to wrong source')
        if native.status == 'cleanup_uncertain':
            raise CleanupUncertain('build may not have released resources')
        public = dict(build_mode=native.mode, build_ms=native.elapsed_ms, environment_sha256=native.environment_sha256)
        deployment = None
        if native.status == 'built':
            started = monotonic()
            deployment = stage_deployment(native,job/'deployment')
            public.update(deployment=deployment.identity,wheels=native.artifact_hashes)
            if self.component_builds:
                public['host_phase_ms'] = dict(cold_recipe=native.elapsed_ms, deploy_and_verify=round(1000*(monotonic()-started)))
        self._save(job/'native-build.json',dict(job=native.job_id,source=native.source_sha256,
                   image=native.image,executor=native.executor_kind,recipe=native.build_args_json.decode(),
                   status=native.status,cache_in=native.cache_input_sha256,cache_out=native.cache_output_sha256,
                   limits=asdict(native.limits),artifacts=[asdict(item) for item in native.artifacts]))
        (job/'build.stderr-tail').write_bytes(native.log_tail)
        return native, deployment, public

    def diagnose(self, engine, build, source, cap):
        need(self.diagnostic is not None and type(cap) is int and cap > 0 and not self.quarantined
             and build is not None and self.verify(build) and build.engine == engine.digest
             and build.operation == 'develop_build' and build.status == 'built',
             'diagnostics require a registered lane and a verified Native build')
        started = monotonic()
        job = self.root/('job-'+token_hex(12))
        job.mkdir(mode=0o700)
        self._save(job/'reservation.json',dict(operation='develop_diagnostic',candidate=engine.digest,
                                               cap_ms=cap,build=build.evidence,scoring=False))
        deployment = self.deployments[build.evidence]
        try:
            observed,archive = self.diagnostic.launch(deployment,build.evidence,source,job,cap)
            elapsed = int((monotonic()-started)*1000)
            need(elapsed <= cap and observed.status in ('completed_non_scoring','incomplete_non_scoring'),
                 'diagnostic did not settle within its lease')
            text = _stdout_preview(archive)
            data = dict(script_sha256=observed.script,diagnostic_sha256=observed.evidence,
                        stdout_bytes=observed.stdout_bytes,stdout_sha256=observed.stdout_sha256,
                        stderr_sha256=observed.stderr_sha256,stdout=text,non_scoring=True,
                        verified_removed=True)
            fields = dict(engine=engine.digest,operation='develop_diagnostic',status=observed.status,
                          node_ms=elapsed,build=build.evidence,data=data)
            record = dict(**fields,diagnostic_archive=archive.name,diagnostic_result_sha256=observed.evidence)
            address = self._save(job/'result.json',record)
            result = WorkerResult(**fields,evidence=address)
            self.records[address] = (job,result,None,deployment,None,None,False)
            self.diagnostics[address] = observed,archive
            return result
        except (Exception,KeyboardInterrupt):
            self.quarantined = True
            self._save(job/'uncertain.json',dict(status='uncertain',operation='develop_diagnostic',charged_ms=cap))
            raise

    def verify(self, result):
        try:
            record = self.records.get(result.evidence)
            if not record:
                return False
            job, saved, native, deployment, observation, weights, observation_unavailable = record
            if saved != result or file_sha256(job/'result.json') != result.evidence:
                return False
            if deployment:
                deployment.verify()
                if getattr(deployment,'runtime_config',b''):
                    from .v3_level3_startup import PATH, VERSION
                    from .v3_level3_deploy import launch_args
                    config=dict(self.built_engines[deployment.candidate].edits)[PATH]
                    if config!=deployment.runtime_config: return False
                    if result.operation=='develop_build' and result.data.get('runtime')!=dict(protocol=VERSION,
                            source_sha256=digest(config),argv=list(launch_args(self.service.spec,deployment))): return False
            if isinstance(native, ComponentReceipt):
                if file_sha256(job/'component-build.json') != result.data.get('component_receipt'):
                    return False
                if deployment and deployment.identity != result.data.get('deployment'):
                    return False
            elif native and native.status == 'built':
                verify_artifacts(native)
            if observation:
                status, data = self._validate(result.operation,job/'observation',observation,weights,deployment)
                return (status,data)==(result.status,result.data)
            if observation_unavailable:
                return result.status == 'insufficient_evidence' and result.data == {'failure': 'observer_incomplete'}
            if result.operation == 'develop_diagnostic':
                diagnostic = self.diagnostics.get(result.evidence)
                if not diagnostic:
                    return False
                observed,archive = diagnostic
                raw = (archive/'result.json').read_bytes()
                inner = json.loads(raw)
                expected = dict(script_sha256=observed.script,diagnostic_sha256=observed.evidence,
                                stdout_bytes=observed.stdout_bytes,stdout_sha256=observed.stdout_sha256,
                                stderr_sha256=observed.stderr_sha256,stdout=_stdout_preview(archive),
                                non_scoring=True,verified_removed=True)
                return (file_sha256(archive/'result.json') == observed.evidence
                        and inner.get('verified_removed') is True and inner.get('non_scoring') is True
                        and inner.get('engine') == result.engine and inner.get('build') == result.build
                        and inner.get('script') == result.data.get('script_sha256')
                        and observed.evidence == result.data.get('diagnostic_sha256')
                        and inner.get('stdout_sha256') == observed.stdout_sha256
                        and inner.get('stdout_bytes') == observed.stdout_bytes
                        and result.status == observed.status and result.data == expected)
            return native is not None and native.status == result.status
        except (OSError,ValueError,KeyError,TypeError):
            return False
