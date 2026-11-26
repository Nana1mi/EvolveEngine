from dataclasses import replace
import json
from itertools import count
import os
import stat
from pathlib import Path
from random import Random
import sys
from tempfile import TemporaryDirectory
from time import monotonic
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_collect import DockerObserver, MEMORY_STAGES, QUALITY_CASES, QualityCollector, TimingCollector, stage_inputs
from evolveengine.v3_level3_evidence import (DEV_SCORE_RULE, EvidenceError, EvidencePolicy, parse_nvidia_smi_process_memory,
                                           validate_evidence)
from evolveengine.v3_level3_worker import NativeWorker, Observation, ObservationUnavailable

GPU = 'GPU-00000000-0000-0000-0000-000000000001'


class CollectorTests(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.weights = {'model':digest(b'weights')}
        self.libs = {'native.so':digest(b'library')}
        # Stub operations can share a Windows clock tick; give fixture events
        # distinct timestamps without weakening hardware chronology checks.
        self.tick = count(1_000_000, 10_000).__next__
        clock = patch('evolveengine.v3_level3_collect.monotonic_ns', self.tick)
        clock.start()
        self.addCleanup(clock.stop)
        self.host = SimpleNamespace(mappings=lambda *a:{'end_ns':self.tick(),'loaded':self.libs},
                                    memory=lambda *a:f'{GPU}, 64\n')
        self.spec = SimpleNamespace(recheck_model=lambda **k:self.weights)
        t = self.tick()
        deployment = SimpleNamespace(candidate=digest(b'actual engine'),native_mapping=lambda *a:({}, self.libs))
        self.service = SimpleNamespace(model_hashes=self.weights,deployment=deployment,container_id='1'*64,
                                       weight_check_started_ns=t,weight_check_finished_ns=t+1,inspection={'fixture':True})
        self.p = EvidencePolicy(digest(b'E0'), 'fixture', 'fixture boot', GPU, self.weights, self.libs, 'a'*64,
                                QUALITY_CASES,'b'*64,'c'*64,{f'W{i}':((3,2),)*(4 if i==4 else 1) for i in range(1,5)},
                                1,2,MEMORY_STAGES,128*(1<<20),('not-invoked',),counter_names=('none',))

    def files(self,*names):
        paths = {name:self.root/name for name in names}
        for name,path in paths.items():
            path.write_text(name)
        return paths,{name:digest(name.encode()) for name in paths}

    def test_worker_rejects_previous_output_replayed_under_the_current_request(self):
        p = replace(self.p,origin='hardware_collection',engine=self.service.deployment.candidate,natural_eos=True)
        first,replay = Observation(p,'a'*64),Observation(p,'b'*64)
        worker = NativeWorker.__new__(NativeWorker)
        worker.service = SimpleNamespace(spec=SimpleNamespace(gpu_uuid=GPU))
        worker.variant = 'sm100'
        worker.timing_replays = {self.root:(replay,self.weights)}
        op = 'develop_measure'
        def receipt(history,output):
            data = dict(measurement_inputs_sha256=history,timing_workload_sha256='c'*64,measurement_pair=7,
                        timed_request_outputs={'W2':{'current-request':{'sha256':output,'tokens':24,'finish_reason':'stop'}}})
            return SimpleNamespace(data={'operations':{op:SimpleNamespace(status='observed',data=data)}})
        with patch('evolveengine.v3_level3_worker.validate_evidence',side_effect=[
                receipt('d'*64,'0'*64),receipt('e'*64,'1'*64)]):
            status, data = worker._validate(op,self.root/'observation',first,self.weights,self.service.deployment)
            self.assertEqual(status,'output_replay_mismatch')
            self.assertEqual(data['output_authentication']['status'],'mismatch')
        with patch('evolveengine.v3_level3_worker.validate_evidence',side_effect=[
                receipt('d'*64,'1'*64),receipt('e'*64,'1'*64)]):
            status, data = worker._validate(op,self.root/'observation',first,self.weights,self.service.deployment)
            self.assertEqual(status,'observed')
            self.assertEqual(data['output_authentication']['status'],'verified')

    def test_bound_replay_accepts_different_tokens_but_rejects_changed_request(self):
        from evolveengine.v3_level3_evidence import BOUND_REPLAY
        p = replace(self.p,origin='hardware_collection',engine=self.service.deployment.candidate,
                    natural_eos=True,replay_protocol=BOUND_REPLAY)
        first, replay = Observation(p,'a'*64), Observation(p,'b'*64)
        worker = NativeWorker.__new__(NativeWorker)
        worker.service = SimpleNamespace(spec=SimpleNamespace(gpu_uuid=GPU))
        worker.variant = 'sm100'
        worker.timing_replays = {self.root:(replay,self.weights)}
        op = 'develop_measure'
        def receipt(history, key, ids, **identity):
            data = dict(measurement_inputs_sha256=history,timing_workload_sha256='c'*64,measurement_pair=7,
                        replay_protocol=BOUND_REPLAY,_private_replay=dict(dict(service=history, run=history, warm=[history],
                        rids=[history],tokens={'W2':{key:ids}}), **identity),
                        timed_request_outputs={'W2':{key:dict(sha256=digest(canonical(ids)),tokens=len(ids),finish_reason='stop')}})
            return SimpleNamespace(data={'operations':{op:SimpleNamespace(status='observed',data=data)}})
        for key, history, identity, expected in [('q','e',{},'observed'),('other','e',{},'output_replay_mismatch'),
                    ('q','d',{},'output_replay_mismatch'), ('q','e',{'service':'d'*64},'output_replay_mismatch'),
                    ('q','e',{'rids':['d'*64]},'output_replay_mismatch'), ('q','e',{'warm':['d'*64]},'output_replay_mismatch')]:
            with self.subTest(key=key, history=history), patch('evolveengine.v3_level3_worker.validate_evidence',side_effect=[
                    receipt('d'*64,'q',[3,4,1]),receipt(history*64,key,[3,9,8,1],**identity)]):
                status, data = worker._validate(op,self.root/'observation',first,self.weights,self.service.deployment)
                self.assertEqual(status, expected)
                self.assertNotIn('_private_replay',data)
                if status == 'observed':
                    self.assertEqual(data['output_authentication']['agreement']['W2'],
                                     dict(requests=1,exact=0,same_positions=1,positions=4,rate=.25))

    def test_independent_timing_binding_and_tamper(self):
        files,hashes = self.files('observe_timing.py','timing.py')
        workload = dict(scenes=self.p.scenes,warm_waves=1,timed_waves=2,observer_sha256=hashes,score_rule=DEV_SCORE_RULE)
        collector = TimingCollector(self.host,self.spec,replace(self.p,measure_workload_sha256=digest(canonical(workload)+b'\n')),
                                    files,hashes)
        def observe(service,inputs,out,deadline,**opts):
            self.assertEqual(opts,{'script':'observe_timing.py'})
            self.assertEqual(set(f.name for f in inputs.iterdir()),set(files)|{'workload.json'})
            clock = self.tick()
            # The synthetic observer simulates sequential waves with overlap inside W4.
            waves = []
            for scene,shape in self.p.scenes.items():
                for i in range(3):
                    clock += 10
                    requests = [dict(rid=f'{scene}-{i}-{j}',input=n,output=k,sha256='e'*64,start_ns=clock,
                                     first_ns=clock+1,last_ns=clock+6,done_ns=clock+8,first_token_unambiguous=True,
                                     stream_complete=True) for j,(n,k) in enumerate(shape)]
                    waves.append(dict(scene=scene,index=i,timed=i>0,requests=requests))
            raw = b''.join(canonical(row)+b'\n' for row in waves)
            for name,payload in (('observer.stdout',raw),('observer.stderr',b''),('observer.inspect.json',b'{}')):
                (out/name).write_bytes(payload)
            return raw
        collector.observer = observe
        archive,op = self.root/'archive','develop_measure'
        receipt = collector(op,self.service,archive,monotonic()+30)
        self.assertEqual(receipt.policy.origin,'offline_fixture')
        self.assertEqual(receipt.policy.engine,self.service.deployment.candidate)
        data = validate_evidence(archive,receipt.policy,{op:receipt.pin}).data['operations'][op]
        self.assertEqual(data.status,'observed',data.reason)
        ident = json.loads((archive/op/'external-identity.json').read_bytes())
        self.assertEqual(ident['weights_before_at_ns'],self.service.weight_check_finished_ns)
        self.assertEqual(set(data.data['scenes']),set(self.p.scenes))
        (archive/op/'service.inspect.json').write_text('{}')
        data = validate_evidence(archive,receipt.policy,{op:receipt.pin}).data['operations'][op]
        self.assertEqual(data.status,'insufficient_evidence')

    def test_quality_uses_witnesses_and_changes_only_verdict(self):
        files,hashes = self.files('observe_quality.py','numeric.py','legacy_eos.py')
        rule = {'observer_sha256':hashes}
        policy = replace(self.p,quality_rule_sha256=digest(canonical(rule)+b'\n'))
        def score(rows):
            return dict(screen={'status':'within_dev_screen'},eos_passed=rows[0]['eos_passed'],mixed_passed=True)
        collector = QualityCollector(self.host,self.spec,policy,files,hashes,score=score,rule=rule)
        for n,expected in ((4,'valid'),(3,'violation')):
            def observe(service,inputs,out,deadline,**opts):
                self.assertTrue(opts['model'])
                self.assertNotIn('quality-rule.json',{p.name for p in inputs.iterdir()})
                first = b'{"kind":"level3_observer_ready"}'
                opts['on_first_line'](first)
                raw = first+b'\n'+canonical({'eos_passed':n})+b'\n'
                (out/'observer.stdout').write_bytes(raw)
                return raw
            collector.observer = observe
            archive,op = self.root/f'archive-{n}','develop_quality'
            receipt = collector(op,self.service,archive,monotonic()+30)
            data = validate_evidence(archive,receipt.policy,{op:receipt.pin}).data['operations'][op]
            self.assertEqual(data.status,expected,data.reason)
            self.assertEqual(data.data['observed_peak_sample_bytes'],64*(1<<20))

    def test_split_trajectory_separates_missing_evidence_and_established_violation(self):
        from evolveengine.v3_level3_eos import PROTOCOL
        from evolveengine.v3_level3_trajectory_lane import SPLIT_CASES, SPLIT_PROTOCOL

        files,hashes=self.files('observe_quality.py','numeric.py','legacy_eos.py','observe_text_timing.py')
        rule={'observer_sha256':hashes,'eos_contract':PROTOCOL}
        policy=replace(self.p,quality_rule_sha256=digest(canonical(rule)+b'\n'))
        collector=QualityCollector(self.host,self.spec,policy,files,hashes,score=None,rule=rule)
        screen={'status':'within_dev_screen'}
        def grade(_): return dict(screen=screen,eos_passed=4,status='pending_trajectory')
        split_pin,cal,code=(digest(x) for x in (b'split',b'cal',b'code'))
        collector.enable_trajectory(cal,code,grade,split_pin=split_pin)
        def observe(service,inputs,out,deadline,**opts):
            self.assertEqual(json.loads((inputs/'paired-trajectory.json').read_bytes())['split_registration'],split_pin)
            first=b'{"kind":"level3_observer_ready"}'; opts['on_first_line'](first)
            raw=first+b'\n{}\n'; (out/'observer.stdout').write_bytes(raw); return raw
        collector.observer=observe
        group={k:dict(status='valid',violations=[],insufficient=[]) for k in SPLIT_CASES}
        group['mixed_tail_absolute']=dict(status='insufficient_evidence',violations=[],insufficient=['8/158/cancel/tv'])
        for expected in ('insufficient_evidence','violation'):
            with self.subTest(expected=expected):
                if expected=='violation': screen={'status':'outside_dev_screen'}
                archive=self.root/expected; op='develop_quality'
                finalize=collector(op,self.service,archive,monotonic()+30)
                binding=json.loads((archive/op/'external-identity.json').read_bytes())
                signed=finalize(dict(protocol=SPLIT_PROTOCOL,live_gate=True,engine=binding['engine'],calibration=cal,
                    split_registration=split_pin,service=binding['service_id'],groups=group,status='insufficient_evidence'))
                verdict=validate_evidence(archive,signed.policy,{op:signed.pin}).data['operations'][op]
                self.assertEqual(verdict.status,expected,verdict.reason)
                self.assertEqual(verdict.data['case_outcomes']['mixed_tail_absolute'],'insufficient_evidence')
                self.assertEqual(verdict.data['case_outcomes']['numeric_32'],'pass' if expected!='violation' else 'fail')

    def test_sampling_and_device_memory_remain_separate_pinned_witnesses(self):
        from evolveengine.v3_level3_sampling import CASES, SAMPLES

        files, hashes = self.files('observe_quality.py', 'numeric.py', 'legacy_eos.py')
        rule = {'observer_sha256': hashes}
        policy = replace(self.p, quality_rule_sha256=digest(canonical(rule)+b'\n'), sampling_required=True)
        def score(rows):
            self.assertEqual(rows, [{'eos_passed': 4}])
            return dict(screen={'status': 'within_dev_screen'}, eos_passed=4, mixed_passed=True)
        collector = QualityCollector(self.host, self.spec, policy, files, hashes, score=score, rule=rule)
        collector.sampling_enabled = True
        def observe(service, inputs, out, deadline, **opts):
            self.assertTrue((inputs/'sampling-policy.json').is_file())
            first = b'{"kind":"level3_observer_ready"}'
            opts['on_first_line'](first)
            rows = [{'eos_passed': 4}, *(dict(kind='sampling', case=case, sequences=[[case]]*SAMPLES)
                                          for case in CASES)]
            raw = first+b'\n'+b''.join(canonical(row)+b'\n' for row in rows)
            (out/'observer.stdout').write_bytes(raw)
            return raw
        collector.observer = observe
        archive, op = self.root/'sampled-archive', 'develop_quality'
        receipt = collector(op, self.service, archive, monotonic()+30)
        quality = archive/op
        def verdict(pin):
            return validate_evidence(archive, receipt.policy, {op:pin}).data['operations'][op]
        result = verdict(receipt.pin)
        self.assertEqual(result.status, 'valid', result.reason)
        self.assertEqual(result.data['observed_peak_sample_bytes'], 64 << 20)
        memory = list(map(json.loads, (quality/'external-memory.jsonl').read_bytes().splitlines()[1:]))
        self.assertEqual([r['stage'] for r in memory], list(MEMORY_STAGES))
        self.assertEqual(set(json.loads((quality/'sampling.json').read_bytes())), set(map(str, CASES)))
        # Even a present sampling file cannot count when it is absent from the signed manifest.
        path = quality/'receipt.json'
        doc = json.loads(path.read_bytes())
        bad = canonical({})+b'\n'
        (quality/'sampling.json').write_bytes(bad)
        doc['artifacts']['sampling.json'] = digest(bad)
        raw = canonical(doc)+b'\n'
        path.write_bytes(raw)
        self.assertEqual(verdict(digest(raw)).status, 'insufficient_evidence')
        del doc['artifacts']['sampling.json']
        raw = canonical(doc)+b'\n'
        path.write_bytes(raw)
        self.assertEqual(verdict(digest(raw)).status, 'insufficient_evidence')

    def test_real_text_pair_binds_requests_and_rejects_a_reused_warm_prompt(self):
        files, hashes = self.files('observe_timing.py', 'timing.py', 'observe_text_timing.py')
        rng = Random(7)
        documents = [[rng.randrange(1000) for _ in range(1000)] for _ in range(2)]
        plan = {'seed': 'a' * 64}
        workload = dict(scenes=self.p.scenes, warm_waves=1, timed_waves=2,
                        observer_sha256=hashes, score_rule=DEV_SCORE_RULE, input_plan=plan)
        policy = replace(self.p, measure_workload_sha256=digest(canonical(workload) + b'\n'))
        collector = TimingCollector(self.host, self.spec, policy, files, hashes, input_plan=plan, documents=documents)
        requests = []
        def observe(service, inputs, out, deadline, **opts):
            self.assertEqual(opts, {'script': 'observe_text_timing.py'})
            prompt_plan = json.loads((inputs / 'inputs.json').read_bytes())
            requests.append(prompt_plan)
            clock = self.tick()
            waves = []
            for scene, shape in self.p.scenes.items():
                warm = prompt_plan[scene][0][0]
                for i, wave in enumerate(prompt_plan[scene]):
                    clock += 10
                    reqs = []
                    for j, (ids, (_, k)) in enumerate(zip(wave, shape)):
                        if len(requests) == 4 and scene == 'W1' and i == 1:
                            ids = warm
                        reqs.append(dict(rid=f'{scene}-{i}-{j}', input=len(ids), output=k,
                                         input_sha256=digest(canonical(ids)), sha256='e'*64,
                                         start_ns=clock, first_ns=clock+1, last_ns=clock+6, done_ns=clock+8,
                                         first_token_unambiguous=True, stream_complete=True,
                                         reported_speculation=dict(accepted=2, proposed=4, verifies=1) if scene == 'W2' else None))
                    waves.append(dict(scene=scene, index=i, timed=i>0, requests=reqs))
            raw = b''.join(canonical(row) + b'\n' for row in waves)
            for name, payload in (('observer.stdout', raw), ('observer.stderr', b''), ('observer.inspect.json', b'{}')):
                (out / name).write_bytes(payload)
            return raw
        collector.observer = observe
        op = 'develop_measure'
        checks = []
        for index, pair in enumerate((2, 2, 3, 3)):
            archive = self.root / f'paired-{index}'
            receipt = collector(op, self.service, archive, monotonic()+20, pair_index=pair)
            checks.append(validate_evidence(archive, receipt.policy, {op: receipt.pin}).data['operations'][op])
        self.assertEqual([c.status for c in checks], ['observed']*3 + ['insufficient_evidence'])
        self.assertEqual(requests[0], requests[1])
        first, second = ({tuple(ids) for waves in plan.values() for wave in waves for ids in wave}
                         for plan in (requests[0], requests[2]))
        self.assertEqual(len(first), 21)
        self.assertFalse(first & second)
        self.assertEqual(checks[0].data['measurement_inputs_sha256'], checks[1].data['measurement_inputs_sha256'])
        timed = checks[0].data['speculation']['timed']
        self.assertEqual(timed['W2'], dict(reported=2, requests=2, accepted=4, proposed=8, verifies=2, accept_rate=.5))
        self.assertIsNone(timed['W1']['accept_rate'])

    def test_bound_timing_archive_checks_raw_observer_token_hash_and_request_identity(self):
        from evolveengine.v3_level3_evidence import BOUND_REPLAY
        files, hashes = self.files('observe_timing.py', 'timing.py', 'observe_text_timing.py')
        plan = {'seed':'a'*64}
        fixed = dict(scenes=self.p.scenes,warm_waves=1,timed_waves=2,observer_sha256=hashes,
                     score_rule=DEV_SCORE_RULE,input_plan=plan)
        rng = Random(17)
        meter = TimingCollector(self.host,self.spec,replace(self.p,measure_workload_sha256=digest(canonical(fixed)+b'\n')),
                                files,hashes,input_plan=plan,documents=[[rng.randrange(1000) for _ in range(2000)]])
        meter.enable_natural_eos()
        meter.policy = replace(meter.policy,replay_protocol=BOUND_REPLAY)
        def observe(service, inputs, out, deadline, **opts):
            clock, rows = self.tick(), []
            for scene, waves in json.loads((inputs/'inputs.json').read_bytes()).items():
                for i, wave in enumerate(waves):
                    clock += 10
                    reqs = [dict(rid=f'{scene}-{i}-{j}', input=len(ids), output=2, output_cap=2,
                                 input_sha256=digest(canonical(ids)), sha256=digest(canonical([42,1])), observed_ids=[42,1],
                                 observed_finish=dict(type='stop',matched=1),finish_reason='stop',
                                 start_ns=clock,first_ns=clock+1,last_ns=clock+6,done_ns=clock+8,
                                 first_token_unambiguous=True,stream_complete=True,reported_speculation=None)
                            for j,ids in enumerate(wave)]
                    rows.append(dict(scene=scene,index=i,timed=i>0,requests=reqs))
            data = b''.join(canonical(r)+b'\n' for r in rows)
            for name, body in [('observer.stdout',data),('observer.stderr',b''),('observer.inspect.json',b'{}')]:
                (out/name).write_bytes(body)
            return data
        meter.observer = observe
        archive, op = self.root/'bound', 'develop_measure'
        obs = meter(op,self.service,archive,monotonic()+20,pair_index=1)
        def verdict(pin): return validate_evidence(archive,obs.policy,{op:pin}).data['operations'][op]
        self.assertEqual(verdict(obs.pin).status,'observed')
        root = archive/op
        originals = {k:(root/k).read_bytes() for k in ('observer.stdout','timing.jsonl','receipt.json')}
        for change, stdout_matches in [({'sha256':'0'*64},True), ({'rid':'swapped'},False),
                                       ({'observed_ids':[1,42],'sha256':digest(canonical([1,42]))},True)]:
            envelope = json.loads(originals['receipt.json'])
            for name in ('observer.stdout','timing.jsonl'):
                lines = [json.loads(x) for x in originals[name].splitlines()]
                if name == 'timing.jsonl' or stdout_matches:
                    lines[int(name=='timing.jsonl')]['requests'][0].update(change)
                data = b''.join(canonical(r)+b'\n' for r in lines)
                (root/name).write_bytes(data)
                envelope['artifacts'][name] = digest(data)
            data = canonical(envelope)+b'\n'
            (root/'receipt.json').write_bytes(data)
            self.assertEqual(verdict(digest(data)).status,'insufficient_evidence')

    def test_pid_memory_observation_replays_and_requires_registered_source(self):
        files,hashes = self.files('observe_quality.py','numeric.py','legacy_eos.py')
        rule = {'observer_sha256':hashes}
        policy = replace(self.p,quality_rule_sha256=digest(canonical(rule)+b'\n'),memory_source='nvidia_smi_compute_pid')
        collector = QualityCollector(self.host,self.spec,policy,files,hashes,
            score=lambda rows:dict(screen={'status':'within_dev_screen'},eos_passed=4,mixed_passed=True),rule=rule)
        def observe(service,inputs,out,deadline,**opts):
            ready = b'{"kind":"level3_observer_ready"}'
            opts['on_first_line'](ready)
            raw = ready+b'\n{}\n'
            (out/'observer.stdout').write_bytes(raw)
            return raw
        collector.observer = observe
        for num,proc,amount in ((1,'101',64),(2,'999',64),(3,'101',129)):
            self.host.service_memory = lambda *a:dict(at_ns=self.tick(),raw_nvml=f'{GPU}, 101, {amount}\n',
                service_processes={proc:dict(start_ticks=123,mount_ns=456)},used_bytes=amount*(1<<20))
            archive,op = self.root/f'pid-archive-{num}','develop_quality'
            receipt = collector(op,self.service,archive,monotonic()+30)
            def check(p, pin):
                return validate_evidence(archive,p,{op:pin}).data['operations'][op]
            data = check(receipt.policy,receipt.pin)
            self.assertEqual(data.status,{1:'valid',2:'insufficient_evidence',3:'violation'}[num],data.reason)
            self.assertEqual(check(replace(receipt.policy,memory_source='nvidia_smi'),receipt.pin).status,'insufficient_evidence')
            witness = archive/op/'memory-ready.json'
            witness.write_bytes(witness.read_bytes()+b'\n')
            self.assertEqual(check(receipt.policy,receipt.pin).status,'insufficient_evidence')

    def test_nvml_count_does_not_coerce_missing_and_rejects_wrong_uuid(self):
        self.assertEqual(parse_nvidia_smi_process_memory('',GPU),{})
        self.assertEqual(parse_nvidia_smi_process_memory(f'{GPU}, 101, 64\n{GPU}, 102, 12\n',GPU),
                         {101:64*(1<<20),102:12*(1<<20)})
        for raw in (f'{GPU}, 101, [N/A]\n', f'{GPU}, 101, 64\n{GPU}, 101, 12\n', 'GPU-foreign, 101, 64\n'):
            with self.subTest(raw=raw), self.assertRaises(EvidenceError):
                parse_nvidia_smi_process_memory(raw,GPU)

    @skipUnless(os.name == 'posix','POSIX bind permissions')
    def test_input_bind_is_readable_with_private_host_umask(self):
        parent = self.root/'private'
        parent.mkdir(mode=0o700)
        inp = parent/'client-input'
        original = os.umask(0o077)
        try:
            stage_inputs(inp,{'observe.py': b'pass\n'})
        finally:
            os.umask(original)
        self.assertEqual(stat.S_IMODE(parent.stat().st_mode),0o700)
        self.assertEqual(stat.S_IMODE(inp.stat().st_mode),0o555)
        self.assertEqual(stat.S_IMODE((inp/'observe.py').stat().st_mode),0o444)

    @skipUnless(sys.platform.startswith('linux'),'Docker host capture uses Linux pipes')
    def test_ready_callback_releases_observer_instead_of_racing_first_request(self):
        witnessed = []
        script = "import sys; print('ready',flush=True); assert sys.stdin.readline() == 'begin\\n'; print('requested')"
        def observe(line):
            self.assertEqual(line,b'ready')
            witnessed.append(line)
        DockerObserver.capture([sys.executable,'-c',script],self.root,monotonic()+5,128,on_first_line=observe)
        self.assertEqual(witnessed,[b'ready'])
        self.assertEqual((self.root/'observer.stdout').read_bytes(),b'ready\nrequested\n')

    @skipUnless(sys.platform.startswith('linux'),'Docker host capture uses Linux pipes')
    def test_pipe_capture_enforces_retention_before_cleanup(self):
        DockerObserver.capture([sys.executable,'-c','print(12345)'],self.root,monotonic()+5,128)
        self.assertEqual((self.root/'observer.stdout').read_bytes(),b'12345\n')
        over = self.root/'over'
        over.mkdir()
        with self.assertRaisesRegex(ObservationUnavailable,'allowance'):
            DockerObserver.capture([sys.executable,'-c','print(10000*"a")'],over,monotonic()+5,128)
        self.assertLessEqual((over/'observer.stdout').stat().st_size,128)
