from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_worker import WorkerResult
from experiments.spark26_gguf.level3 import control_run as run, worker_stdio as worker


class BaselineRegistrationTests(TestCase):
    def test_runtime_is_bound_and_legacy_opt_in_is_replay_only(self):
        node = run.node
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            registration = root/'registration.json'
            runtime_file = 'evolveengine/v3_level3_deploy.py'
            image = 'sha256:'+'e'*64
            pins = node.implementation_pins()
            d = dict(schema=node.SCHEMA,boot_id='boot',gpu_uuid='test GPU',
                     build_image=image,build_image_id=image,runtime_image=image,runtime_image_id=image,
                     limits=vars(node.BuildLimits()),node_ms=dict.fromkeys(('develop_build','develop_measure','develop_quality'),120000),
                     observer_sha256={},quality_rule_sha256='a'*64,measure_workload_sha256='a'*64,
                     memory_calibration_sha256='a'*64,implementation_sha256=pins,
                     native_backend_sha256=pins['evolveengine/v3_level3_build.py'])
            d.update(dict.fromkeys(('lock','archive','model_dir','weight','anchor_d3','anchor_r1','memory_calibration'),str(root)))

            def write():
                raw = canonical(d)+b'\n'
                registration.write_bytes(raw)
                return digest(raw)

            self.assertIn(runtime_file,pins)
            for module in ('v3_provider.py','v3_runner.py'):
                path = f'evolveengine/{module}'
                self.assertEqual(pins[path],digest((node.ROOT/path).read_bytes()))
            self.assertEqual(node.checked_inputs(registration,write()),d)
            pins[runtime_file] = '0'*64
            with self.assertRaisesRegex(ValueError,'hardware-side implementation'):
                node.checked_inputs(registration,write())
            d.pop('implementation_sha256')
            d['schema'] = node.LEGACY_SCHEMA
            pin = write()
            with self.assertRaisesRegex(ValueError,'configuration version'):
                node.checked_inputs(registration,pin)
            self.assertEqual(node.checked_inputs(registration,pin,allow_legacy=True),d)


class ControlRunTests(TestCase):
    def setUp(self):
        t = TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name).resolve()
        self.config = dict(node_ms=dict(develop_build=80,develop_quality=60,develop_measure=70),
                           memory_calibration_sha256='a'*64)
        self.config_pin = digest(canonical(self.config)+b'\n')
        self.variants = {name:SimpleNamespace(digest=digest(name.encode())) for name in ('e0',*run.ORDER)}
        self.pair = self.root/'pair.json'
        self.pair_raw = dict(kind=run.PAIR_KIND,engine=self.variants['e0'].digest, receipts=[str(k)*64 for k in range(1,7)],
                             candidate_residency_gate=False,memory_calibration={'source':'nvidia_smi_compute_pid','result_sha256':'a'*64},
                             observed_peak_compute_pid_bytes={'e0a':61*run.MIB,'e0b':65*run.MIB})
        self.encode(self.root/'registration.json',dict(sha256=self.config_pin,config=self.config))
        self.repin()

    def encode(self, path, row):
        raw = canonical(row)+b'\n'
        path.write_bytes(raw)
        return digest(raw)

    def repin(self):
        self.pair_pin = self.encode(self.pair,self.pair_raw)

    def draft(self):
        return run.draft(self.config,self.config_pin,self.pair,self.pair_pin,self.variants)

    def test_control_plan_uses_prespecified_margin_and_no_scoring_for_faults(self):
        plan = self.draft()
        self.assertEqual(plan['rule']['sample_ceiling_bytes'],73*run.MIB)
        self.assertEqual(plan['max_total_node_ms'],4*80+4*60+2*70)
        self.assertNotIn('measure_cold',plan['schedule']['residency_negative'])
        self.assertTrue(plan['rule']['profile_required_separately'])
        self.assertEqual(len(plan['implementation']),10)

    def test_bad_or_incomplete_pair_cannot_choose_threshold(self):
        self.pair_raw['observed_peak_compute_pid_bytes']['e0a'] = 45*run.MIB
        self.repin()
        with self.assertRaisesRegex(ValueError,'spread'):
            self.draft()
        self.pair_raw['observed_peak_compute_pid_bytes']['e0a'] = 65*run.MIB
        self.repin()
        self.pair.write_bytes(self.pair.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'fingerprint changed'):
            self.draft()
        self.pair_raw['receipts'].pop()
        self.repin()
        with self.assertRaisesRegex(ValueError,'not a complete'):
            self.draft()

    def test_same_profile_replayed_quality_can_anchor_controls(self):
        self.config.update(timing_text={'seed': 'f'*64}, launch_profile='graph_defaults')
        self.config_pin = self.encode(self.root/'inputs.json', self.config)
        self.encode(self.root/'registration.json', dict(sha256=self.config_pin, config=self.config))
        self.pair_raw.update(kind=run.REPLAYED_QUALITY_KIND, registration_sha256=self.config_pin, receipts={})
        peaks = self.pair_raw.pop('observed_peak_compute_pid_bytes')
        rows = {}
        for index, (arm, peak) in enumerate(peaks.items(), 1):
            rows[arm] = dict(arm=arm, engine=self.variants['e0'].digest, deployment='c'*64,
                             observation_sha256=str(index)*64, node_ms=45,
                             result=dict(status='valid', data=dict(case_outcomes=dict.fromkeys(run.QUALITY_CASES, 'pass'),
                                         memory_source='nvidia_smi_compute_pid', observed_peak_sample_bytes=peak)))
            self.pair_raw['receipts'][arm] = self.encode(self.root/f'quality-{arm}.json', rows[arm])
        self.repin()
        self.assertEqual(self.draft()['rule']['sample_ceiling_bytes'], 73*run.MIB)

        path = self.root/'quality-e0b.json'
        rows['e0b']['result']['data']['memory_source'] = 'untrusted'
        changed_pin = self.encode(path, rows['e0b'])
        with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
            self.draft()
        self.pair_raw['receipts']['e0b'] = changed_pin
        self.repin()
        with self.assertRaisesRegex(ValueError, 'memory witness'):
            self.draft()

    def test_archived_control_index_pins_prior_build_and_complete_cache_pair(self):
        config = dict(self.config, archive='/e0', build_image='native', build_image_id='image', limits={},
                      native_backend_sha256='b'*64, boot_id='boot', gpu_uuid='gpu', lock='/lock')
        config_pin = digest(canonical(config)+b'\n')
        self.encode(self.root/'registration.json', dict(sha256=config_pin, config=config))
        plan = dict(baseline_registration_sha256=config_pin, schedule=run.ORDER,
                    candidates={role: self.variants[role].digest for role in run.ORDER})
        self.encode(self.root/'control-plan.json', dict(sha256=digest(canonical(plan)+b'\n'), plan=plan))
        role = 'native_positive'
        index = {}
        for stage in ('build_cold', 'build_incremental'):
            path = self.root/f'{role}-{stage}.json'
            pin = self.encode(path, dict(engine=self.variants[role].digest, status='built', evidence=digest(stage.encode())))
            index[f'{role}/{stage}'] = dict(path=str(path), sha256=pin)
        path = self.root/'index.json'
        ref = dict(path=str(path), sha256=self.encode(path, index))
        self.assertEqual(set(run.archived_controls(ref, config, self.variants)), set(index))
        changed_negative = dict(self.variants, native_compute_negative=self.variants['residency_negative'])
        self.assertEqual(set(run.archived_controls(ref, config, changed_negative)), set(index))
        changed_positive = dict(self.variants, native_positive=self.variants['residency_negative'])
        with self.assertRaisesRegex(ValueError, 'different Native environment or source'):
            run.archived_controls(ref, config, changed_positive)
        with self.assertRaisesRegex(ValueError, 'different Native environment'):
            run.archived_controls(ref, dict(config, build_image_id='other'), self.variants)
        index.pop(f'{role}/build_cold')
        ref['sha256'] = self.encode(path, index)
        with self.assertRaisesRegex(ValueError, 'original cold and incremental pair'):
            run.archived_controls(ref, config, self.variants)

    def quality(self,role,*,status='valid',failed=(),delta=0):
        return WorkerResult(self.variants[role].digest,'develop_quality',status,7,'e'*64,'a'*64,
                            dict(case_outcomes={key:'fail' if key in failed else 'pass' for key in run.QUALITY_CASES},
                                 observed_peak_sample_bytes=73*run.MIB+delta))

    def test_faults_must_be_detected_by_expected_independent_witness(self):
        limit = 73*run.MIB
        run.check_quality('native_positive',self.quality('native_positive'),limit)
        compute='native_compute_negative'
        mirror='residency_negative'
        x=self.quality(compute,status='violation',failed=('numeric_32',))
        y=self.quality(mirror,status='violation',delta=run.MIB)
        run.check_quality(compute,x,limit)
        run.check_quality(mirror,y,limit)
        for role,result in ((compute,replace(x,status='insufficient_evidence',data={})),
                            (compute,self.quality(compute,status='violation',delta=run.MIB)),
                            (mirror,self.quality(mirror,status='violation',failed=('numeric_32',),delta=run.MIB)),
                            (mirror,replace(y,status='valid'))):
            with self.subTest(role=role,result=result),self.assertRaises(ValueError):
                run.check_quality(role,result,limit)

    def test_frozen_runner_follows_cold_incremental_and_keeps_receipts(self):
        plan=self.draft()
        report=self.root/'report'
        report.mkdir()
        stages=[]
        by_digest = {engine.digest:key for key,engine in self.variants.items()}
        checks = {
            'native_positive':self.quality('native_positive'),
            'native_compute_negative':self.quality('native_compute_negative',status='violation',failed=('numeric_32',)),
            'residency_negative':self.quality('residency_negative',status='violation',delta=run.MIB),
        }

        class FakeWorker:
            def __init__(self,*a,**k):
                self.records={}
                self.receipts={}

            def __call__(self,op,engine,build,cap):
                role=by_digest[engine.digest]
                address=digest(canonical([op,role,len(stages)]))
                parent=build.evidence if build else None
                stages.append((role,op,parent))
                if op=='develop_build':
                    r=WorkerResult(engine.digest,op,'built',9,None,address,dict(build_mode='incremental' if build else 'cold'))
                    self.records[address]=(None,r,SimpleNamespace(cache_input_sha256='a'*64 if build else None,
                                                                   cache_output_sha256='a'*64))
                elif op=='develop_quality':
                    r=replace(checks[role],build=parent,evidence=address)
                else:
                    r=WorkerResult(engine.digest,op,'observed',7,parent,address,
                                   dict(scenes=dict.fromkeys(('W1','W2','W3','W4'),{})))
                self.receipts[address]=r
                return r

            def verify(self,result):
                return self.receipts.get(result.evidence)==result

        @dataclass(frozen=True)
        class Policy:
            memory_limit_bytes: int = 1
        collector=SimpleNamespace(policy=Policy())
        setup=(None,self.variants['e0'],None,None,None,{'develop_quality':collector},{},dict(source='nvidia_smi_compute_pid'))
        with patch.object(run.node,'setup_baseline',return_value=setup),patch.object(run.native_controls,'make',return_value=self.variants),patch.object(run,'NativeWorker',FakeWorker):
            result=run.execute(report,self.config,self.config_pin,plan,'f'*64)
        raw=(report/'result.json').read_bytes()
        self.assertEqual(result,digest(raw))
        self.assertFalse(json.loads(raw)['native_path_execution_verified'])
        self.assertEqual(len(stages),10)
        self.assertIsNotNone(stages[3][2])
        self.assertIsNone(stages[6][2])
        plan_path = self.root/'plan.json'
        plan_pin = self.encode(plan_path,plan)
        summary = json.loads(raw)
        summary_path = report/'result.json'
        controls = dict(plan=str(plan_path),plan_sha256=plan_pin,result=str(summary_path),result_sha256=result)

        def admit():
            return worker.candidate_ceiling(self.config,self.config_pin,None,self.variants['e0'],setup[-1],controls)

        with patch.object(worker.native_controls,'make',return_value=self.variants):
            # Changing even the archival wrapper must fail. With exact bytes,
            # the positive and both independent negatives open the development gate.
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):
                admit()
            (report/'control-plan.json').unlink()
            self.encode(report/'control-plan.json',dict(sha256=plan_pin,plan=plan))
            self.assertEqual(admit(),73*run.MIB)
            target = report/'residency_negative-quality_cold.json'
            changed = json.loads(target.read_bytes())
            changed['status'] = 'valid'
            changed_pin = self.encode(target,changed)
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):
                admit()
            summary['receipt_files']['residency_negative/quality_cold'] = changed_pin
            controls['result_sha256'] = self.encode(summary_path,summary)
            with self.assertRaisesRegex(ValueError,'did not isolate'):
                admit()

        # A failed positive must suppress its timing, but still leave both
        # independent negative-control verdicts and an incomplete settlement.
        failed_report = self.root/'failed-report'
        failed_report.mkdir()
        stages.clear()
        checks['native_positive'] = self.quality('native_positive',status='violation',failed=('numeric_32',))
        with patch.object(run.node,'setup_baseline',return_value=setup),patch.object(run.native_controls,'make',return_value=self.variants),patch.object(run,'NativeWorker',FakeWorker):
            with self.assertRaisesRegex(ValueError,'registered Native quality controls failed'):
                run.execute(failed_report,self.config,self.config_pin,plan,'f'*64)
        expected = [(role, op) for role in checks for op in ('develop_build','develop_quality')]
        self.assertEqual([(role, op) for role, op, _ in stages], expected)
        self.assertFalse((failed_report/'result.json').exists())
        failure = json.loads((failed_report/'failed-controls.json').read_bytes())
        self.assertEqual(set(failure['failed_quality_checks']), {'native_positive/quality_cold'})
        self.assertEqual(set(failure['receipt_files']), {
            f'{role}/{stage}' for role in checks for stage in ('build_cold','quality_cold')
        })

    def test_worker_accepts_diagnostic_lease_without_expanding_baseline(self):
        settings = {'node_ms': {'develop_build': 200}}
        def start(cap):
            lease = worker.WorkerLease(200000, 1, {'develop_diagnostic': cap})
            worker.run(self.root, settings, self.config_pin, lease, 'd'*64, None, None,
                       scope=worker.E0_ONLY, controls=None)
        with patch.object(worker, 'setup_baseline', side_effect=RuntimeError('preflight reached')):
            with self.assertRaisesRegex(RuntimeError, 'preflight reached'):
                start(worker.MAX_MS)
            for cap in (worker.MIN_MS-1, worker.MAX_MS+1):
                with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'outside registered lane limits'):
                    start(cap)

    def test_worker_authorization_requires_control_pins_for_candidate_scope(self):
        p = self.root/'worker.json'
        doc = dict(schema=worker.SCHEMA,baseline_sha256=self.config_pin,node_ms=300,jobs=1,
                   max_per_op={'develop_build':200},engine_scope=worker.E0_ONLY,controls=None)
        self.assertEqual(worker.authorization(p,self.encode(p,doc))[2],worker.E0_ONLY)
        doc['engine_scope'] = worker.NATIVE_DEVELOPMENT
        with self.assertRaisesRegex(ValueError,'requires pinned Native controls'):
            worker.authorization(p,self.encode(p,doc))
        doc['controls'] = dict(plan=str(self.root/'plan.json'),result=str(self.root/'result.json'),
                               plan_sha256='f'*64,result_sha256='e'*64)
        self.assertEqual(worker.authorization(p,self.encode(p,doc))[3],doc['controls'])
