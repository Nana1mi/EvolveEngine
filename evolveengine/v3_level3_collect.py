"""Trusted host collectors for non-profiled Native observations."""
from dataclasses import replace
import json
from pathlib import Path
import re
from secrets import token_hex
from selectors import DefaultSelector, EVENT_READ
from shutil import disk_usage
import subprocess
from time import monotonic, monotonic_ns

from .v3_harness import canonical, digest
from .v3_level3_build import CleanupUncertain
from .v3_level3_evidence import DEV_SCORE_RULE, SCHEMA, file_sha256, parse_nvidia_smi_device_memory
from .v3_level3_gb10 import docker_command, docker_env
from .v3_level3_worker import Observation, ObservationUnavailable
from .v3_level3_workload import TimingSchedule, input_fingerprints, select_inputs


def need(ok, why):
    if not ok:
        raise ValueError(why)


def artifact_space(directory):
    """Leave room for the host journal and cleanup on the same filesystem."""
    usage = disk_usage(directory)
    return max(0, usage.free - max(512 << 20, min(4 << 30, usage.total // 50)))


def write(path, doc):
    out = canonical(doc)+b'\n'
    with path.open('xb') as f:
        f.write(out)
    return digest(out)


def write_rows(path, binding, rows):
    with path.open('xb') as stream:
        stream.write(canonical(binding)+b'\n')
        for row in rows:
            stream.write(canonical(row)+b'\n')


def identity(host, service, spec, archive, op, policy, deadline, variant):
    before = dict(service.model_hashes)
    load = host.mappings(service,variant,deadline)
    after = spec.recheck_model(deadline=deadline)
    end = monotonic_ns()
    expected_hashes = service.deployment.native_mapping(variant)[1]
    policy = replace(policy,engine=service.deployment.candidate,weights=before,loaded_artifacts=expected_hashes)
    binding = dict(schema=SCHEMA,operation=op,engine=policy.engine,worker_id=policy.worker_id,boot_id=policy.boot_id,
                   device_uuid=policy.device_uuid,origin=policy.origin,run_id=token_hex(16),service_id=service.container_id,
                   start_ns=service.weight_check_started_ns,end_ns=end)
    row = dict(**binding,weights_before=before,weights_after=after,loaded_artifacts=load['loaded'],
               weight_source='host_asset_sha256',loaded_source='host_proc_map_files',
               weights_before_at_ns=service.weight_check_finished_ns,loaded_at_ns=load['end_ns'],weights_after_at_ns=end)
    write(archive/'external-identity.json',row)
    write(archive/'host-mappings.json',load)
    write(archive/'service.inspect.json',service.inspection)
    return policy,binding


def stage_inputs(path, files):
    path.mkdir(mode=0o700)
    for name,raw in files.items():
        need(re.fullmatch('[a-zA-Z0-9_.-]+',name) and name not in ('.','..'),'invalid trusted observer file name')
        (path/name).write_bytes(raw)
        (path/name).chmod(0o444)
    # The bind root must be traversable by the unprivileged observer even
    # under the trusted host's umask 077. The archive parent stays private.
    path.chmod(0o555)


def pinned_files(files, hashes, required):
    need(set(files) == set(hashes) == required,'missing trusted client sources')
    items = {name:source_bytes(path) for name,path in files.items()}
    need(all(digest(buf)==hashes[name] for name,buf in items.items()),'observer client identity mismatch')
    return items


def source_bytes(path):
    # The registered observer runs these exact LF bytes on Linux regardless of
    # the controller checkout's Git core.autocrlf setting.
    return Path(path).read_bytes().replace(b'\r\n',b'\n')


class TimingCollector:
    """No candidate-owned mount, clock, workload, or environment enters this client."""
    def __init__(self, host, service_spec, policy, observer_files, observer_hashes, *, variant='sm100',
                 input_plan=None, documents=None):
        self.host,self.service_spec,self.policy,self.variant = host,service_spec,policy,variant
        self.schedule = None
        need((input_plan is None) == (documents is None), 'timing input plan needs its verified corpus')
        files = {'observe_timing.py','timing.py'}
        if input_plan is not None:
            obs_name = 'observe_workbuddy_timing.py' if (policy.workbuddy_sla and 'observe_workbuddy_timing.py' in observer_hashes) else 'observe_text_timing.py'
            files.add(obs_name)
        self.files = pinned_files(observer_files,observer_hashes,files)
        plan = dict(scenes=policy.scenes,warm_waves=policy.warm_waves,timed_waves=policy.timed_waves,
                    observer_sha256=observer_hashes,score_rule=policy.score_rule or DEV_SCORE_RULE)
        if policy.workbuddy_sla:
            plan['workbuddy_sla'] = True
        if policy.workbuddy_v2:
            plan['workbuddy_v2'] = True
        if policy.scene_warm_waves:
            plan['scene_warm_waves'] = dict(policy.scene_warm_waves)
        if policy.scene_timed_waves:
            plan['scene_timed_waves'] = dict(policy.scene_timed_waves)
        if input_plan is not None:
            plan['input_plan'] = input_plan
            scene_waves = ({s: (policy.scene_warm_waves or {}).get(s, policy.warm_waves) +
                            (policy.scene_timed_waves or {}).get(s, policy.timed_waves)
                            for s in policy.scenes}
                           if (policy.scene_warm_waves or policy.scene_timed_waves) else None)
            self.schedule = TimingSchedule(documents, policy.scenes, policy.warm_waves + policy.timed_waves,
                                           input_plan['seed'], allow_wrap=policy.workbuddy_sla,
                                           scene_waves=scene_waves)
        self.workload = canonical(plan)+b'\n'
        need(digest(self.workload)==policy.measure_workload_sha256, 'timing workload not frozen in hardware policy')
        self.observer = DockerObserver(host,service_spec)
        self.fixed_workload = self.workload
        self.fixed_policy = self.policy

    def enable_natural_eos(self, chat=None, *, split='development'):
        need(self.schedule is not None, 'natural EOS requires the frozen real-text workload')
        plan = json.loads(self.fixed_workload)
        plan['timing_mode'] = 'natural_eos'
        self.chat_schedule = None
        w3_timed = None
        if chat is not None:
            from .v3_level3_evidence import BOUND_REPLAY
            plan['replay_protocol'] = BOUND_REPLAY
            from .v3_level3_chat_timing import ChatSchedule, SCHEMA as CHAT_SCHEMA, W3_INPUT_RANGE, load_chat

            hashes = self.policy.weights
            waves = load_chat(chat, hashes['tokenizer.json'], hashes['chat_template.jinja'],
                              self.policy.scenes, self.policy.warm_waves, self.policy.timed_waves,
                              split=split, engine=self.policy.engine, workbuddy=self.policy.workbuddy_sla,
                              scene_warm_waves=self.policy.scene_warm_waves,
                              scene_timed_waves=self.policy.scene_timed_waves)
            self.chat_schedule = ChatSchedule(waves, self.policy.warm_waves,
                                              self.policy.timed_waves, self.schedule.seed,
                                              scene_warm_waves=self.policy.scene_warm_waves,
                                              scene_timed_waves=self.policy.scene_timed_waves)
            self.chat_schedule.maximum = self.schedule.maximum
            plan['input_plan'] = dict(distribution=CHAT_SCHEMA, corpus_sha256=chat['sha256'],
                                      enable_thinking=True, split=split)
            if split=='generalization':
                from .v3_level3_chat_timing import bounds, GENERALIZATION
                plan['length_strata']=GENERALIZATION
                plan['lane_ranges']={s:[(bounds(s,n,split,0,workbuddy=self.policy.workbuddy_sla)[0],
                                         bounds(s,n,split,1,workbuddy=self.policy.workbuddy_sla)[1])
                                        for n,_ in shape] for s,shape in self.policy.scenes.items()}
            if not self.policy.workbuddy_sla:
                w3_timed = len(waves['W3']) - 2*self.policy.warm_waves
                plan['w3_input_range'] = W3_INPUT_RANGE
                plan['w3_visible_answer'] = 'spark_after_reasoning_end'
                plan['w3_timed_waves'] = w3_timed
        self.workload = canonical(plan) + b'\n'
        self.policy = replace(self.fixed_policy, natural_eos=True, measure_workload_sha256=digest(self.workload),
                              chat_corpus_sha256=chat['sha256'] if chat else None,w3_timed_waves=w3_timed,
                              replay_protocol=plan.get('replay_protocol'))
        return self.policy.measure_workload_sha256

    def __call__(self, op, service, archive, deadline, *, pair_index=0, stress=False, authenticity_replay=False):
        need(op=='develop_measure','timing collector cannot sign quality or profiling')
        archive.mkdir(mode=0o700)
        operation = archive/op
        operation.mkdir(mode=0o700)
        observer_input = archive/'client-input'
        staged = {**self.files,'workload.json':self.fixed_workload if stress else self.workload}
        script, policy, extra = 'observe_timing.py', self.fixed_policy if stress else self.policy, {}
        if self.schedule is not None:
            chat = None if stress else getattr(self, 'chat_schedule', None)
            if chat:
                inputs, hashes = chat.get(pair_index, replay=authenticity_replay)
            else:
                inputs, hashes = self.schedule.get(pair_index)
            if authenticity_replay and not chat:
                warm = policy.warm_waves
                need(policy.natural_eos and warm > 0, 'authenticity replay requires independent warm-up')
                old = {pin for waves in hashes.values() for wave in waves for pin in wave}
                distinct = select_inputs(self.schedule.documents, policy.scenes, warm,
                                         seed=digest(canonical([self.schedule.seed,'authenticity-replay'])),
                                         pair=pair_index, previously_used=old)
                inputs = {scene: distinct[scene] + list(reversed(waves[warm:])) for scene,waves in inputs.items()}
                hashes = input_fingerprints(inputs)
            if 'lane_ranges' in json.loads(staged['workload.json']):
                from .v3_level3_chat_timing import length_map
                policy=replace(policy,measure_input_lengths=length_map(inputs))
            staged['inputs.json'] = canonical(inputs) + b'\n'
            policy = replace(policy, measure_input_hashes=hashes, measure_pair=pair_index)
            script = 'observe_workbuddy_timing.py' if 'observe_workbuddy_timing.py' in self.files else 'observe_text_timing.py'
            extra = dict(pair_index=pair_index)
            if policy.natural_eos:
                extra['timing_mode'] = 'natural_eos'
                if policy.chat_corpus_sha256:
                    extra['chat_corpus_sha256'] = policy.chat_corpus_sha256
        stage_inputs(observer_input,staged)
        if extra:
            (operation/'timing-inputs.json').write_bytes(staged['inputs.json'])
        observer_options = {'model':True} if policy.chat_corpus_sha256 else {}
        if policy.replay_protocol:
            # Bound by the *registered* request shapes, not a candidate-reported output length.
            observer_options['output_cap'] = 512*1024 + sum(len(inputs[s])*sum(8*k for _,k in shape)
                                                            for s,shape in policy.scenes.items())
        raw = self.observer(service,observer_input,operation,deadline,script=script,**observer_options)
        policy,binding = identity(self.host,service,self.service_spec,operation,op,policy,deadline,self.variant)
        raw_rows = raw.splitlines()
        expected = sum((policy.scene_warm_waves or {}).get(s, policy.warm_waves) +
                       ((policy.scene_timed_waves or {}).get(s, policy.w3_timed_waves if s=='W3' and policy.w3_timed_waves
                                                             else policy.timed_waves)) for s in policy.scenes)
        need(len(raw_rows)==expected,'incomplete timing client coverage')
        write_rows(operation/'timing.jsonl',binding,map(json.loads,raw_rows))
        required = ('external-identity.json','host-mappings.json','service.inspect.json','timing.jsonl',
                    'observer.stdout','observer.stderr','observer.inspect.json')
        if extra:
            required += ('timing-inputs.json',)
        pins = {name:file_sha256(operation/name) for name in required}
        pin = write(operation/'receipt.json',dict(**binding,artifacts=pins,observer_outside_candidate=True,
                    profiler_enabled=False,clock='external_client_monotonic',workload_sha256=policy.measure_workload_sha256,
                    **extra))
        return Observation(policy,pin)

class DockerObserver:
    def __init__(self, host, service_spec):
        self.host,self.service_spec = host,service_spec

    def __call__(self, service, inputs, outputs, deadline, *, script, model=False, output_cap=512*1024,
                 on_first_line=None):
        need(',' not in str(inputs),'unsafe observer staging path')
        need(script in ('observe_timing.py','observe_text_timing.py','observe_workbuddy_timing.py','observe_quality.py','observe_trajectory.py'),'not a registered trusted observer')
        need(re.fullmatch('[0-9a-f]{64}',service.container_id),'invalid observer peer')
        namespace = 'container:'+service.container_id
        name = 'ee-l3-observer-'+token_hex(10)
        tool = self.host.docker
        spec = self.service_spec
        def run(*args, max_seconds=20):
            timeout = min(max_seconds,deadline-monotonic())
            need(timeout>0,'observer deadline exhausted')
            result = subprocess.run(docker_command(tool,*args),stdin=subprocess.DEVNULL,capture_output=True,
                                    timeout=timeout,env=docker_env())
            need(result.returncode==0 and len(result.stdout)<=1024*1024,'trusted observer command failed or evidence too large')
            return result.stdout
        mounts = {'/observer': str(inputs)}
        if model:
            mounts['/model'] = str(spec.model_dir)
        try:
            run('create','--name',name,'--pull=never','--runtime','runc','--network',namespace,'--read-only',
                *(('--interactive',) if on_first_line is not None else ()),
                '--cap-drop','ALL','--security-opt','no-new-privileges','--user','12346:12346','--pids-limit','64',
                '--memory','1024m','--no-healthcheck','--restart','no','--log-driver','none',
                '--tmpfs','/tmp:rw,nosuid,nodev,mode=1777,size=64m',
                *(part for dest,src in mounts.items() for part in ('--mount',f'type=bind,src={src},dst={dest},readonly')),
                '--env','PYTHONPATH=/opt/spark26-deps','--env','PYTHONNOUSERSITE=1','--env','NVIDIA_VISIBLE_DEVICES=void',
                '--env','HOME=/tmp','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--entrypoint','python',
                spec.image_id,'-B','-s','/observer/'+script)
            raw = json.loads(run('inspect',name))
            item, = raw
            host = item['HostConfig']
            need(item['Image']==spec.image_id and item['Config']['User']=='12346:12346' and host['NetworkMode']==namespace
                 and host['ReadonlyRootfs'] and not host['Privileged'] and not host['DeviceRequests'] and not host['Devices']
                 and host['CapDrop']==['ALL'] and not host['CapAdd'] and not host['PortBindings']
                 and host['Runtime']=='runc' and host['LogConfig']['Type']=='none'
                 and item['Config']['OpenStdin'] is (on_first_line is not None)
                 and item['Config']['Entrypoint']==['python'] and item['Config']['Cmd']==['-B','-s','/observer/'+script]
                 and {m['Destination']:(m['Source'],m['RW'],m['Type']) for m in item['Mounts']}
                     == {dest:(src,False,'bind') for dest,src in mounts.items()},'observer isolation drift')
            write(outputs/'observer.inspect.json',raw)
            attach = ('start','-a','-i',name) if on_first_line is not None else ('start','-a',name)
            self.capture(docker_command(tool,*attach),outputs,deadline-55,output_cap,env=docker_env(),
                         on_first_line=on_first_line)
            return (outputs/'observer.stdout').read_bytes()
        finally:
            try:
                subprocess.run(docker_command(tool,'rm','-f',name),stdin=subprocess.DEVNULL,capture_output=True,
                               timeout=10,env=docker_env())
                gone = subprocess.run(docker_command(tool,'inspect',name),stdin=subprocess.DEVNULL,capture_output=True,
                                      timeout=5,env=docker_env())
                need(gone.returncode==1 and b'no such' in gone.stderr.lower(),'observer removal could not be verified')
            except (ValueError,OSError,subprocess.SubprocessError) as exc:
                raise CleanupUncertain('trusted observer cleanup uncertain') from exc

    @staticmethod
    def capture(argv, outputs, stop, output_cap, *, env=None, on_first_line=None, artifact=False):
        if monotonic() >= stop:
            raise ObservationUnavailable('observer has no cleanup margin')
        with (outputs/'observer.stdout').open('xb', buffering=0) as out,(outputs/'observer.stderr').open('xb') as err:
            stdin = subprocess.PIPE if on_first_line is not None else subprocess.DEVNULL
            with subprocess.Popen(argv,stdin=stdin,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                  bufsize=0,env=env) as proc, DefaultSelector() as stream:
                try:
                    first = bytearray()
                    stream.register(proc.stdout,EVENT_READ,[out,output_cap])
                    stream.register(proc.stderr,EVENT_READ,[err,output_cap if artifact else 512*1024])
                    until_space_check = 0
                    while stream.get_map():
                        left = stop-monotonic()
                        if left <= 0:
                            raise ObservationUnavailable('observer exceeded its observation time')
                        for key,_ in stream.select(left):
                            block = key.fileobj.read(65536)
                            if not block:
                                stream.unregister(key.fileobj)
                                continue
                            target,remaining = key.data
                            if len(block) > remaining:
                                raise ObservationUnavailable('observer exceeded its retained output allowance')
                            if artifact:
                                if until_space_check < len(block):
                                    headroom = artifact_space(outputs)
                                    if headroom < len(block):
                                        raise ObservationUnavailable('artifact disk reserve reached')
                                    until_space_check = min(16 << 20, headroom)
                                until_space_check -= len(block)
                            target.write(block)
                            key.data[1] -= len(block)
                            if target is out and on_first_line is not None:
                                line, sep, _ = block.partition(b'\n')
                                first.extend(line)
                                if len(first)>1024:
                                    raise ObservationUnavailable('missing trusted observer startup witness')
                                if sep:
                                    on_first_line(bytes(first))
                                    if monotonic() >= stop:
                                        raise ObservationUnavailable('ready witness consumed observation time')
                                    try:
                                        proc.stdin.write(b'begin\n')
                                        proc.stdin.close()
                                    except OSError as exc:
                                        raise ObservationUnavailable('unable to release trusted observer start barrier') from exc
                                    on_first_line = None
                    if on_first_line is not None:
                        raise ObservationUnavailable('missing trusted observer startup witness')
                    if proc.wait(timeout=max(0,stop-monotonic())) != 0:
                        raise ObservationUnavailable('observer process ended without complete evidence')
                except subprocess.TimeoutExpired as exc:
                    raise ObservationUnavailable('observer exceeded its observation time') from exc
                finally:
                    if proc.poll() is None:
                        proc.kill()


QUALITY_CASES = ('numeric_32','natural_eos_4','cancel_mixed_3')
LONG_CTX_CASE = 'long_context_logprob_16k_64k'
MEMORY_STAGES = ('ready','post_requests')


class QualityCollector:
    """Reuse the registered dev Judge, never its historical pass outcomes."""
    def __init__(self, host, service_spec, policy, observer_files, observer_hashes, *, score, rule, variant='sm100',
                 long_ctx_cases_bytes=None):
        self.host,self.service_spec,self.policy,self.score,self.variant = host,service_spec,policy,score,variant
        self.long_ctx_cases_bytes = long_ctx_cases_bytes
        required = {'observe_quality.py','numeric.py','legacy_eos.py'}
        if 'eos_contract' in rule:
            from .v3_level3_eos import PROTOCOL
            need(rule['eos_contract'] == PROTOCOL, 'unknown EOS contract')
            required.add('observe_text_timing.py')
        self.files = pinned_files(observer_files,observer_hashes,required)
        expected_cases = (*QUALITY_CASES, LONG_CTX_CASE) if long_ctx_cases_bytes is not None else QUALITY_CASES
        need(policy.quality_cases==expected_cases and policy.memory_stages==MEMORY_STAGES,
             'unexpected development coverage')
        need(rule.get('observer_sha256')==observer_hashes and digest(canonical(rule)+b'\n')==policy.quality_rule_sha256,
             'quality rule or client is not frozen')
        self.rule = rule
        self.observer = DockerObserver(host,service_spec)
        self.sampling_enabled = False

    def enable_trajectory(self, calibration_pin, code_pin, grade, *, split_pin=None):
        from .v3_level3_trajectory import VERSION
        from .v3_level3_trajectory_lane import SPLIT_CASES, SPLIT_PROTOCOL
        need('eos_contract' in self.rule and all(re.fullmatch('[0-9a-f]{64}',x) for x in (calibration_pin,code_pin)),
             'trajectory gate needs frozen calibration and observers')
        cases=(*QUALITY_CASES[:2],'paired_cancel_numeric','autoregressive_full_vocab')
        trajectory=dict(protocol=VERSION,calibration=calibration_pin,code=code_pin)
        if split_pin is not None:
            need(re.fullmatch('[0-9a-f]{64}',split_pin),'invalid split quality registration')
            cases=(*QUALITY_CASES[:2],*SPLIT_CASES)
            trajectory.update(protocol=SPLIT_PROTOCOL,split_registration=split_pin)
        if self.long_ctx_cases_bytes is not None:
            cases = (*cases, LONG_CTX_CASE)
        self.rule=dict(self.rule,cases=cases,trajectory=trajectory)
        self.policy=replace(self.policy,quality_cases=cases,quality_rule_sha256=digest(canonical(self.rule)+b'\n'))
        self.score=grade

    def __call__(self, op, service, archive, deadline):
        need(op=='develop_quality','quality collector cannot sign timing or profiling')
        archive.mkdir(mode=0o700)
        operation = archive/op
        operation.mkdir(mode=0o700)
        observer_input = archive/'client-input'
        files = dict(self.files)
        if 'trajectory' in self.rule:
            files['paired-trajectory.json']=canonical(self.rule['trajectory'])
        if self.long_ctx_cases_bytes is not None:
            files['long-context-cases.json'] = self.long_ctx_cases_bytes
        if self.sampling_enabled:
            from .v3_level3_sampling import CASES, SAMPLES
            files['sampling-policy.json'] = canonical(dict(cases=CASES,samples=SAMPLES))
        stage_inputs(observer_input,files)
        memory_samples = []
        pid_witnesses = {}
        source = self.policy.memory_source or 'nvidia_smi'
        def memory(stage):
            try:
                if source=='nvidia_smi_compute_pid':
                    witness = self.host.service_memory(service,deadline)
                    filename = 'memory-'+stage+'.json'
                    pid_witnesses[filename] = witness
                    sample = dict(at_ns=witness['at_ns'],used_bytes=witness['used_bytes'],scope='service_compute_pids')
                else:
                    raw = self.host.memory(deadline)
                    filename = 'memory-'+stage+'.csv'
                    (operation/filename).write_bytes(raw.encode('utf-8'))
                    sample = dict(at_ns=monotonic_ns(),used_bytes=parse_nvidia_smi_device_memory(raw,self.policy.device_uuid),
                                  scope='exclusive_device')
            except (ValueError,FileNotFoundError) as exc:
                raise ObservationUnavailable('device memory sample unavailable') from exc
            memory_samples.append(dict(stage=stage,witness=filename,**sample))

        def ready(line):
            if line != b'{"kind":"level3_observer_ready"}':
                raise ObservationUnavailable('missing independent health witness')
            memory(MEMORY_STAGES[0])
        raw = self.observer(service,observer_input,operation,deadline,script='observe_quality.py',model=True,
                            output_cap=36*1024*1024,on_first_line=ready)
        need([row['stage'] for row in memory_samples]==list(MEMORY_STAGES[:1]),'missing ready memory sample')
        memory(MEMORY_STAGES[1])
        try:
            rows = [json.loads(line) for line in raw.splitlines()[1:]]
            score = self.score([row for row in rows if row.get('kind') != 'sampling'])
            if self.sampling_enabled:
                from .v3_level3_sampling import CASES, checked_samples
                sampling_rows = [r for r in rows if r.get('kind') == 'sampling']
                need([r['case'] for r in sampling_rows] == list(CASES), 'sample cases are out of order or missing')
                row = checked_samples({str(r['case']):r['sequences'] for r in sampling_rows})
                write(operation/'sampling.json', row)
        except (ValueError,KeyError,TypeError,AssertionError) as exc:
            raise ObservationUnavailable('incomplete development output witness') from exc
        policy,binding = identity(self.host,service,self.service_spec,operation,op,self.policy,deadline,self.variant)
        for name,witness in pid_witnesses.items():
            write(operation/name,dict(**binding,**witness))
        write(operation/'quality-rule.json',self.rule)
        write_rows(operation/'external-memory.jsonl',binding,memory_samples)

        def finish(trajectory=None):
            need(('trajectory' in self.rule)==(trajectory is not None),'incomplete supplemental quality observation')
            def outcome(ok): return 'pass' if ok else 'fail'
            tests={QUALITY_CASES[0]:(outcome(score['screen']['status']=='within_dev_screen'),'observer.stdout'),
                   QUALITY_CASES[1]:(outcome(score['eos_passed']==4),'observer.stdout')}
            if trajectory is None:
                tests[QUALITY_CASES[2]]=(outcome(score['mixed_passed'] is True),'observer.stdout')
            else:
                from .v3_level3_trajectory_lane import trajectory_cases
                registered=self.rule['trajectory']
                need(trajectory['engine']==policy.engine and trajectory['calibration']==registered['calibration'] and
                     trajectory.get('split_registration')==registered.get('split_registration') and
                     trajectory['service']==binding['service_id'] and score['status']=='pending_trajectory','unbound trajectory')
                write(operation/'trajectory-verdict.json',trajectory)
                tests.update({k:(v,'trajectory-verdict.json') for k,v in trajectory_cases(trajectory,registered['protocol']).items()})
            if self.long_ctx_cases_bytes is not None:
                lc = score.get('long_context') or {}
                tests[LONG_CTX_CASE] = (outcome(lc.get('status') == 'valid'), 'observer.stdout')
            outcomes={v for v,_ in tests.values()}
            write(operation/'judge.json',dict(score,status='violation' if 'fail' in outcomes else
                       'insufficient_evidence' if 'insufficient_evidence' in outcomes else 'valid'))
            need(tuple(tests)==policy.quality_cases,'missing or reordered development cases')
            write_rows(operation/'isolated-quality.jsonl',binding,
                       (dict(case=case,outcome=value,witness=witness) for case,(value,witness) in tests.items()))
            pins = {p.name:file_sha256(p) for p in operation.iterdir()}
            pin = write(operation/'receipt.json',dict(**binding,artifacts=pins,observer_outside_candidate=True,
                        profiler_enabled=False,rule_sha256=policy.quality_rule_sha256,isolated_output_probe=True,memory_source=source))
            return Observation(policy,pin)
        return finish if 'trajectory' in self.rule else finish()
