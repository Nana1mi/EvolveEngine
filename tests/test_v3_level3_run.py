from contextlib import contextmanager
from pathlib import Path
import json
import os
from dataclasses import replace
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from evolveengine.journal import Journal
from evolveengine.v3_harness import canonical, digest

from evolveengine import v3_level3_scoring as business
from evolveengine.v3_level3_run import CORE, Run, WorkerResult, compare_dev, explore, _budget_limited
from evolveengine.v3_level3_source import SourceContract
from evolveengine.v3_provider import MeteredLimits


def measure(evidence, scores):
    return WorkerResult(engine="a" * 64, operation="develop_measure", status="observed", node_ms=10,
                        build="b" * 64, evidence=evidence,
                        data={"scenes": {name: {"dev_score": score} for name, score in scores.items()}})


class DevelopmentComparisonTests(TestCase):
    def test_natural_eos_accepts_authenticated_token_divergence_and_detects_workload_drift(self):
        base = measure('a'*64, dict.fromkeys(business.WEIGHTS, 100))
        def arm(result, output, length, proof='verified'):
            return replace(result, data=dict(result.data, timing_mode='natural_eos',
                              timing_workload_sha256='c'*64,
                              output_authentication=dict(method='independent_same_engine_replay', status=proof,
                                                         replay_receipt='e'*64),
                              timed_request_outputs={scene: {'t': dict(sha256=output, tokens=length, finish_reason='stop')}
                                                     for scene in business.WEIGHTS}))
        baseline = arm(base, '1'*64, 100)
        legal = arm(replace(base, evidence='b'*64), '2'*64, 102)
        self.assertTrue(all(v['comparable'] and v['mismatch_rate'] == 1
                            for v in compare_dev(baseline,legal)['output_work'].values()))
        short = arm(legal, '2'*64, 70)
        self.assertFalse(compare_dev(baseline,short)['output_work']['W2']['comparable'])
        with self.assertRaisesRegex(ValueError,'missing independent timing-output replay'):
            compare_dev(baseline,arm(legal,'1'*64,100,'mismatch'))

    def test_e0_calibration_recomputes_work_in_fixed_arm_order_not_output_equality(self):
        def arm(name, h, length):
            rows = {s:{digest(s.encode()):dict(sha256=h*64,tokens=length,finish_reason='length')} for s in business.WEIGHTS}
            pin = digest(canonical({k:[r['sha256'],length,'length'] for scene in rows.values() for k,r in scene.items()}))
            return dict(arm=name,timed_request_outputs=rows,timed_outputs_sha256=pin)
        a,b = arm('e0a','a',100),arm('e0b','b',100)
        good = business.calibration_output_work(a,b,'c'*64)
        self.assertEqual(good,business.calibration_output_work(b,a,'c'*64))
        self.assertTrue(all(r['comparable'] and r['mismatch_rate']==1 for r in good.values()))
        short = arm('e0b','b',50)
        self.assertFalse(business.calibration_output_work(a,short,'c'*64)['W1']['comparable'])
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            business.calibration_output_work(a,dict(b,timed_outputs_sha256=a['timed_outputs_sha256']),'c'*64)

    def test_continuous_nomination_can_resume_from_another_engine(self):
        state = Run.__new__(Run)
        state.tiered_sampling = False  # archived pre-milestone scoring protocol
        state.scoring, state.continuous, state.require_measurement = business.NATURAL_VERSION, True, True
        candidate = SimpleNamespace(engine='candidate', build='build')
        state.qualified = {'candidate':candidate}
        state.builds = {'candidate':SimpleNamespace(evidence='build',data={})}
        state.measurements = {('candidate','build'):'receipt'}
        state.business_summary = lambda _: dict(eligible_for_delivery=True, weighted_score=1.01)
        state.journal = SimpleNamespace(get_event=lambda _: dict(checkpoint='current'))
        state.continuation_checkpoint = lambda current: state.segment_checkpoint or current
        status, reply = state._rpc('finish_exploration',{'engine':'candidate','checkpoint':'other'},7,'e0')
        self.assertEqual((status,reply['status'],reply['checkpoint']),(200,'proposal_then_continue','other'))
        self.assertEqual(state.pending[0],'nominate')
        self.assertEqual(state.pending[1][:2],(candidate,'other'))
    def test_equal_scene_geometric_mean_matches_registered_metric(self):
        baseline = measure("c" * 64, dict(W1=100, W2=200, W3=300, W4=400))
        candidate = measure("d" * 64, dict(W1=110, W2=180, W3=330, W4=400))
        result = compare_dev(baseline, candidate)
        self.assertAlmostEqual(result["scene_ratios"]["W1"], 1.1)
        self.assertAlmostEqual(result["scene_ratios"]["W2"], 0.9)
        self.assertAlmostEqual(result["equal_scene_geomean"], (1.1 * 0.9 * 1.1) ** 0.25)
        self.assertEqual(result["interpretation"], "descriptive_development_comparison")

    def test_incomplete_or_nonpositive_metric_fails_closed(self):
        baseline = measure("c" * 64, dict(W1=1, W2=1, W3=1, W4=1))
        with self.assertRaises(ValueError):
            compare_dev(baseline, measure("d" * 64, dict(W1=1, W2=1, W3=1)))
        with self.assertRaises(ValueError):
            compare_dev(baseline, measure("d" * 64, dict(W1=1, W2=1, W3=0, W4=1)))

    def test_interactive_weights_and_confirmed_regression_gate(self):
        self.assertAlmostEqual(business.weighted_score(dict(W1=1.01, W2=1.02, W3=4, W4=2)),
                               1.01 ** .20 * 1.02 ** .70 * 4 ** .05 * 2 ** .05)

        def observations(w2, *, count=6, uncertainty=False):
            jobs = []
            for pair in range(count):
                baseline = dict(evidence=f'e0-{pair}')
                candidate = dict(engine='e', build='b', evidence=f'e-{pair}', data=dict(
                    measurement_pair=pair, measurement_inputs_sha256=f'input-{pair}'),
                    performance=dict(reference=baseline['evidence'], scene_ratios=dict(
                        W1=1.01, W2=w2 + (.035 if uncertainty and pair % 2 else 0), W3=4, W4=2)))
                jobs.extend((candidate, baseline) if pair % 2 else (baseline, candidate))
            return business.summarize_build(jobs, 'e', 'b')

        self.assertTrue(observations(1.02)['eligible_for_delivery'])
        self.assertEqual(observations(1.02, count=5)['status'], 'needs_repeats')
        self.assertEqual(observations(.98, uncertainty=True)['status'], 'inconclusive')
        regression = observations(.97)
        self.assertGreater(regression['weighted_score'], 1)
        self.assertEqual(regression['status'], 'regression')
        self.assertEqual(regression['below_floor'], ['W2'])

    def test_finish_requires_business_gate_but_checkpoint_is_available(self):
        state = Run.__new__(Run)
        state.scoring = business.VERSION
        state.qualified = {'engine': SimpleNamespace(engine='engine', build='build')}
        state.builds = {'engine': SimpleNamespace(evidence='build')}
        state.checkpoint = lambda engine, wip: 'saved'
        state.business_summary = lambda engine: dict(eligible_for_delivery=False, status='regression')
        denied, diagnostic = state._rpc('finish_exploration', {'engine': 'engine'}, 1, 'engine')
        self.assertEqual(denied, 409)
        self.assertEqual(diagnostic['qualification']['status'], 'regression')
        self.assertEqual(state._rpc('save_checkpoint', {'engine': 'engine'}, 1, 'engine'),
                         (200, {'checkpoint': 'saved'}))

    def test_interactive_seed_retest_alternating_pairs_and_independent_delivery(self):
        self.exercise_delivery(False)

    def test_weight_review_is_async_until_explicit_export(self):
        self.exercise_delivery(True)

    def test_reviewed_cold_nomination_exports_after_clean_continuous_pause(self):
        self.exercise_delivery(True, continuous=True)

    def exercise_delivery(self, audit, continuous=False):
        import json
        from evolveengine import v3_level3_weight_audit as w
        from evolveengine.v3_level3_delivery import export_delivery
        path = 'python/sglang/srt/engine.py'
        source = SourceContract({path: b'engine = 0\n'})
        baseline = source.from_args({'files': {}})
        candidate = source.from_args({'files': {path: 'engine = 1'}})
        runtime = dict(pi_image='a'*64, package_lock_sha='b'*64, runner_sha='c'*64,
                       base_config_sha='d'*64, node_version='24.13.0', pi_version='1.5.2')
        pairs, operations = [], []

        class Worker:
            @staticmethod
            def verify(result):
                return True

            def __call__(self, operation, engine, parent, cap):
                return self.record(operation, engine, parent)

            def independent_build(self, engine, cap):
                return self.record('develop_build', engine, None, dict(build_mode='independent_cold'))

            def measure(self, engine, parent, cap, index):
                pairs.append((index, engine.digest))
                scores = {scene: dict(dev_score=103 if engine == candidate and scene == 'W2' else 100)
                          for scene in business.WEIGHTS}
                extra = dict(timing_mode='natural_eos', timing_workload_sha256='d'*64,
                             output_authentication=dict(method='independent_same_engine_replay',status='verified',replay_receipt='e'*64),
                             timed_request_outputs={k:{'t':dict(sha256='a'*64,tokens=10,finish_reason='stop')} for k in scores}) if continuous else {}
                return self.record('develop_measure', engine, parent, dict(**extra,
                    scenes=scores, measurement_pair=index, measurement_inputs_sha256=f'{index:064x}',
                    timed_outputs_sha256='c'*64))

            @staticmethod
            def record(operation, engine, parent, data=None):
                operations.append(operation)
                if operation=='develop_quality' and audit:
                    data={'weight_observation':dict(workload='a'*64, sampled_bytes={'ready':100,'post_requests':500})}
                status = dict(develop_build='built', develop_quality='valid', develop_measure='observed')[operation]
                return WorkerResult(engine.digest, operation, status, 1,
                                    parent.evidence if parent and operation != 'develop_build' else None,
                                    f'{len(operations):064x}', data or {})

        def launcher(engine, snapshot, prompt, gateway, **options):
            self.assertEqual(engine, candidate)
            choose = {'engine': candidate.digest}
            status, reply = gateway.handler('finish_exploration', choose)
            self.assertEqual((status, reply['qualification']['pairs']), (409, 1))
            self.assertEqual(gateway.handler('confirm_performance', choose)[1]['status'], 'eligible')
            if not continuous:
                self.assertEqual(gateway.handler('finish_exploration', choose)[1]['status'], 'independent_cold_build_required')
            result = gateway.handler('confirm_cold_build', choose)[1]
            self.assertEqual(result['data']['build_mode'], 'independent_cold')
            self.assertEqual(gateway.handler('develop_quality', choose)[1]['status'], 'valid')
            self.assertEqual(gateway.handler('confirm_performance', choose)[1]['pairs'], business.MIN_PAIRS)
            self.assertEqual(gateway.handler('finish_exploration', choose)[0], 200)
            if continuous: (Path(directory)/'run/pause-after-segment').touch()
            return dict(isolated_state_removed=True, socket_removed=True, unchanged_install=True, pi_exit=0)

        with TemporaryDirectory() as directory:
            result = explore(Path(directory)/'run', source, Worker(), launcher,
                             lambda *_: SimpleNamespace(rows=[], rejections=[], spent_input=0, spent_output=0),
                             runtime, 'test', MeteredLimits(None, None, None, None, 393216), node_ms=None, calls=None,
                             reservations=dict.fromkeys(CORE, 100), bootstrap_measure=True, paired_timing=True,
                             continuous=continuous, scoring=(business.natural_policy('c'*64,'d'*64,dict.fromkeys(business.WEIGHTS,.01)) if continuous else business.VERSION),
                             seed_engine=candidate, weight_audit=w.POLICY if audit else None)
            if audit:
                root=Path(directory); folder=root/'run/weight-audit'; out=root/'bundle'
                packets=[e for e in result['events'] if e['operation']=='weight_audit_requested']
                self.assertEqual({e['trigger'] for e in packets},{'historical_best','checkpoint','final_nomination'})
                selected=result['stage_proposals'][0] if continuous else result['final']
                self.assertEqual(selected['kind'],'stage_proposal' if continuous else 'weight_review_nomination')
                args={'proposal_segment':selected['segment']} if continuous else {}
                points=[dict(engine=j['engine'],build=j['build'],development=j['performance'],
                             **{k:j['data'][k] for k in ('measurement_pair','measurement_inputs_sha256','timed_outputs_sha256')})
                        for j in result['jobs'] if 'performance' in j]
                for pt in points:self.assertEqual(pt['development']['gain_status'],w.PROVISIONAL)
                (root/'registration.json').write_bytes(w.wire(dict(runtime=runtime,weight_audit=w.POLICY,business_scoring=result['business_scoring'])))
                (root/'settlement.json').write_bytes(w.wire(dict(status=result['status'],last_phase='worker_close',node_ms=result['node_ms'],worker_cost={'unsettled_node_reservation_ms':0})))
                (root/'performance-trajectory.json').write_bytes(w.wire(dict(reference=result['development_reference'],points=points,charged_node_ms=result['node_ms'])))
                with self.assertRaisesRegex(ValueError,'review pending'):export_delivery(root,out,source,**args)
                self.assertFalse(out.exists())
                key=next(e['packet'] for e in packets if e['trigger']=='final_nomination'); doc=w.packet_read(folder,key)
                self.assertEqual(doc['chosen'],selected['chosen']);self.assertEqual(doc['memory']['status'],'matched')
                blob=root/'runtime-probe';blob.write_bytes(w.wire(dict(schema='q8-runtime-witness-v1',chosen=doc['chosen'],method='offline fixture',
                    observations=[dict(stage=stage,representation='synthetic Q8',lifetime='fixture lifetime',raw_observation='fixture probe')
                                  for stage in ('weights_loaded','post_requests')])))
                report=dict(packet_sha256=key,chosen=doc['chosen'],decision='approved',reviewer='offline fixture',
                            checks=dict.fromkeys(w.CHECKS,'fixture finding'),runtime_evidence_sha256=w.file_digest(blob))
                w.record_review(folder,key,report,blob)
                self.assertEqual(export_delivery(root,out,source,**args)['gain_status'],'externally_reviewed')
        self.assertEqual(result['status'], 'incomplete' if continuous else 'pending_weight_review' if audit else 'delivered')
        summary=result['stage_proposals'][0]['score'] if continuous else result['chosen_business_score']
        self.assertEqual(summary['candidate_first'],3)
        self.assertEqual([engine for _, engine in pairs[1:5]],
                         [baseline.digest, candidate.digest, candidate.digest, baseline.digest])


class ExplorationAccountingTests(TestCase):
    def test_live_controller_opens_model_lease_after_connecting_worker(self):
        scripts = Path(__file__).resolve().parents[1]/'experiments/spark26_gguf/level3'
        with patch.object(sys, 'path', [str(scripts), *sys.path]):
            import launch_live

        config, limits = launch_live.cfg.validate_objective_config(
            json.loads(launch_live.ENGINEERING_CONFIG.read_bytes()))
        config = dict(config, research=None, scoring=None)
        plan = dict(scope=launch_live.ENGINEERING_SCOPE, corpus_sha256=None)
        lease = dict(node_ms=None, jobs=None, max_per_op={})
        admission = (plan, config, limits, {}, {}, lease, {'pin': 'a'*64}, None)
        wire = b'offline request fixture'
        upstream_reply = [b'data: offline fixture\n']

        @contextmanager
        def connect_worker(*args, **kwargs):
            yield object()

        @contextmanager
        def send_to_upstream(body, timeout):
            self.assertEqual((body, timeout), (wire, 1))
            yield upstream_reply

        class ReachedUpstream(Exception):
            pass

        def begin_exploration(output, _contract, _worker, _launch, make_lease,
                              _runtime, _prompt, active_limits, **options):
            self.assertTrue(options['continuous'])
            model = make_lease(active_limits, 1, ())
            self.assertEqual(model.model_socket_idle_seconds,
                             config['transport']['relay_socket_idle_seconds'])
            model.attempts = 1
            with model.transport(wire, 1) as reply:
                self.assertEqual(list(reply), upstream_reply)
                heartbeat = json.loads((model.directory/'stream-progress.json').read_bytes())
                self.assertEqual(heartbeat['attempt'], 1)
                self.assertEqual(heartbeat['data_frames'], 1)
                self.assertTrue((output.parent/'deepseek-started.json').is_file())
            raise ReachedUpstream

        with (TemporaryDirectory() as tmp,
              patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-fixture'}),
              patch.object(launch_live, 'audit', return_value=admission),
              patch.object(launch_live.SourceContract, 'acquire_spark26_e0', return_value=object()),
              patch.object(launch_live.cfg.narrow, 'PublicPromptCounter', return_value=object()),
              patch.object(launch_live.cfg, 'provider_transport', return_value=send_to_upstream),
              patch.object(launch_live, 'durable_ssh_worker', connect_worker),
              patch.object(launch_live, 'explore', begin_exploration)):
            root = Path(tmp)
            approval = dict(path=str(root/'plan.json'), sha256='b'*64)
            with self.assertRaises(ReachedUpstream):
                launch_live.execute(root/'receipt', approval, None, None, None, 'offline-worker')

    def test_token_reservation_refusal_is_recorded_as_resource_limit(self):
        model = SimpleNamespace(rejections=[dict(dispatched=False, reason='token_reservation')])
        self.assertTrue(_budget_limited(model))
        model.rejections = [dict(dispatched=True, reason='token_reservation')]
        self.assertFalse(_budget_limited(model))
        model.rejections = [dict(dispatched=False, reason='context_generation_window')]
        self.assertFalse(_budget_limited(model))
        model.rejections = [dict(dispatched=False, reason='api_cost_limit')]
        self.assertFalse(_budget_limited(model))

    def test_interruption_charges_the_unsettled_worker_reservation(self):
        scripts = Path(__file__).resolve().parents[1]/'experiments/spark26_gguf/level3'
        with patch.object(sys, 'path', [str(scripts), *sys.path]):
            from launch_live import worker_cost

        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'trajectory.sqlite3'
            engine = 'a'*64
            with Journal(path) as journal:
                journal.event('dispatch/000001', dict(operation='worker_reserved', engine=engine,
                              worker_operation='develop_build', reserved_ms=300))
                evidence = 'b'*64
                receipt = journal.put('level3_worker_result', dict(operation='develop_build', engine=engine,
                                      evidence=evidence, node_ms=170))
                journal.event('dispatch/000002', dict(operation='worker_settled', result_object=receipt, evidence=evidence))
                journal.event('dispatch/000003', dict(operation='worker_reserved', engine=engine,
                              worker_operation='develop_measure', reserved_ms=220))
            self.assertEqual(worker_cost(None, path), dict(source='durable_journal', charged_node_ms=390,
                             settled_node_ms=170, unsettled_node_reservation_ms=220, reserved_operations=2))


class EngineeringAdmissionTests(TestCase):
    def test_same_profile_replay_binds_nested_raw_receipts(self):
        from experiments.spark26_gguf.level3.worker_stdio import engineering_ceiling
        from evolveengine.v3_level3_collect import QUALITY_CASES

        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            def write(path, row):
                raw = canonical(row)+b'\n'
                path.write_bytes(raw)
                return digest(raw)

            engine = 'a'*64
            peak, margin = 5000 << 20, 8 << 20
            pins = {'experiments/spark26_gguf/level3/worker_stdio.py': 'b'*64,
                    'evolveengine/v3_level3_worker_channel.py': 'c'*64,
                    'evolveengine/v3_level3_evidence.py': 'd'*64}
            prior = dict(launch_profile='graph_defaults', timing_text={'seed': 'e'*64}, implementation_sha256=pins)
            settings = dict(prior, implementation_sha256=pins | {
                'experiments/spark26_gguf/level3/worker_stdio.py': 'f'*64})
            write(root/'registration.json', dict(config=prior, sha256=digest(canonical(prior)+b'\n')))
            observations = dict(
                develop_quality=dict(status='valid', data=dict(engine=engine, memory_source='nvml_fixture',
                    observed_peak_sample_bytes=peak, case_outcomes=dict.fromkeys(QUALITY_CASES,'pass'))),
                develop_measure=dict(status='observed', data=dict(engine=engine, measurement_pair=0,
                    profiler_enabled=False, scenes={s:dict(dev_score=1) for s in ('W1','W2','W3','W4')},
                    measurement_inputs_sha256='1'*64, timed_outputs_sha256='2'*64)))
            receipts = {}
            for op, result in observations.items():
                location = root/op/op
                location.mkdir(parents=True)
                pin = write(location/'receipt.json', {'operation':op})
                result['data']['receipt_id'] = pin
                receipts[op] = write(root/f'{op}.json', dict(observation_sha256=pin, result=result))
            replay = root/'replayed.json'
            replay_sha = write(replay, dict(kind='trusted_engineering_replay_not_E0_E0_pair',
                engine=engine, profile=prior['launch_profile'], verdicts=observations, receipts=receipts,
                capture_log_witnesses=dict(prefill=['prefill.log'], decode=['decode.log'])))
            controls = dict(replay=str(replay), replay_sha256=replay_sha)
            source = dict(source='nvml_fixture')
            e0 = SimpleNamespace(digest=engine)
            self.assertEqual(engineering_ceiling(settings, e0, source, controls), peak+margin)
            self.assertIsNone(engineering_ceiling(settings, e0, source, dict(controls, weight_audit='q8-external-review-v1')))
            with self.assertRaisesRegex(ValueError,'does not match'):
                engineering_ceiling(dict(settings,timing_text={'seed':'1'*64}), e0, source, controls)
            (root/'develop_measure/develop_measure/receipt.json').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'differs'):
                engineering_ceiling(settings, e0, source, controls)
