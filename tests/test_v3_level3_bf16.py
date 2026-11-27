"""Offline unit tests for Level 3 BF16 deployment, weight audit, controls, and prompts."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf/level3'))

from evolveengine.v3_harness import digest
from evolveengine.v3_level3_deploy import (
    IMMEDIATE_GRAPH_WIRE,
    RUNTIME_ENV,
    ServiceSpec,
    TARGET,
    launch_args,
    runtime_tmpfs,
    validate_container,
)
from evolveengine import v3_level3_weight_audit as weights
from evolveengine.v3_level3_source import SourceContract
from experiments.spark26_gguf.level3 import bf16_controls, launch_live


class Bf16MigrationTests(unittest.TestCase):
    def test_bf16_service_spec_launch_and_container_validation(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td).resolve() / 'model'
            model_dir.mkdir()
            pins = {}
            for name, body in (
                ('config.json', b'{"dtype":"bfloat16"}'),
                ('tokenizer.json', b'{}'),
                ('chat_template.jinja', b'{{messages}}'),
                ('model.safetensors.index.json', b'{"weight_map":{}}'),
                ('model-00001-of-00002.safetensors', b'fake-bf16-shard-1'),
            ):
                p = model_dir / name
                p.write_bytes(body)
                pins[name] = digest(body)
            spec = ServiceSpec(
                image='sha256:' + 'a' * 64,
                image_id='sha256:' + 'a' * 64,
                model_dir=model_dir,
                model_files=tuple(sorted(pins.items())),
                gguf=None,
                gguf_sha256=None,
                gpu_uuid='GPU-f35dfcc5-bbd8-f29f-d939-a9d096a20a7a',
                launch_profile=IMMEDIATE_GRAPH_WIRE,
                model_format='bf16_safetensors',
            )
            checked = spec.recheck_model()
            self.assertEqual(checked, pins)
            argv = launch_args(spec)
            self.assertIn('--load-format', argv)
            self.assertEqual(argv[argv.index('--load-format') + 1], 'safetensors')
            self.assertEqual(argv[argv.index('--dtype') + 1], 'bfloat16')
            self.assertEqual(argv[argv.index('--kv-cache-dtype') + 1], 'bfloat16')
            self.assertEqual(argv[argv.index('--served-model-name') + 1], 'spark26-bf16')
            self.assertIn('--incremental-streaming-output', argv)
            self.assertNotIn('--disable-decode-cuda-graph', argv)
            self.assertNotIn('--disable-prefill-cuda-graph', argv)

            # Validate container mounts only /model (no GGUF mount)
            class _FakeDep:
                tree = Path(td).resolve() / 'site'
                runtime_config = b''

            raw = {
                'Id': 'b' * 64,
                'Name': '/ee-l3-bf16-test',
                'Image': spec.image_id,
                'State': {'Running': False},
                'HostConfig': {
                    'NetworkMode': 'none',
                    'ReadonlyRootfs': True,
                    'Privileged': False,
                    'PidMode': '',
                    'IpcMode': 'private',
                    'PortBindings': {},
                    'Devices': [],
                    'CapDrop': ['ALL'],
                    'CapAdd': None,
                    'SecurityOpt': ['no-new-privileges'],
                    'Tmpfs': runtime_tmpfs(),
                    'PidsLimit': 512,
                    'Memory': 65536 * 1024**2,
                    'DeviceRequests': [{'DeviceIDs': [spec.gpu_uuid], 'Count': 0, 'Capabilities': [['gpu']]}],
                },
                'Config': {
                    'User': '12345:12345',
                    'Entrypoint': ['python'],
                    'Cmd': list(argv),
                    'Env': [f'{k}={v}' for k, v in RUNTIME_ENV.items()],
                },
                'Mounts': [
                    {'Destination': TARGET, 'Source': str(_FakeDep.tree), 'RW': False, 'Type': 'bind'},
                    {'Destination': '/model', 'Source': str(model_dir), 'RW': False, 'Type': 'bind'},
                ],
            }
            self.assertEqual(validate_container(raw, _FakeDep, spec, 'ee-l3-bf16-test'), 'b' * 64)

    def test_bf16_weight_audit_policy_and_static_scan(self):
        self.assertIn(weights.BF16_POLICY, weights.POLICIES)
        self.assertIn('BF16', weights.agent_notice(weights.BF16_POLICY))
        bw = weights.bandwidth_hint(
            {'scenes': {'W1': {'tpot_ms': 10.0}, 'W2': {'tpot_ms': 14.0}}},
            policy=weights.BF16_POLICY,
        )
        self.assertTrue(bw['W1']['investigate'])
        self.assertFalse(bw['W2']['investigate'])

        base = {bf16_controls.ACTIVATION_CU: (bf16_controls.GELU_ANCHOR + '\n').encode(),
                bf16_controls.SPARK_MODEL: ('class M:\n    def f(self, x):\n' + bf16_controls.MLP_ANCHOR + '        return gate_up\n').encode()}
        contract = SourceContract(
            base,
            provenance={
                'revision': bf16_controls.SGLANG_REVISION,
                'archive_sha256': bf16_controls.SGLANG_ARCHIVE_SHA256,
                'target_precision': 'bf16',
                'overlay': None,
            },
        )
        variants = bf16_controls.make(contract)
        self.assertEqual(set(variants), {'base', 'positive', 'fault', 'lowprec'})
        scan = weights.static_scan(contract.base, variants['lowprec'].edits)
        self.assertTrue(any('conversion_near_persistent_storage' in loc['hints'] for loc in scan['locations']))

    def test_bf16_system_prompt_keeps_open_research_and_bf16_notice(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] /
                          'experiments/spark26_gguf/level3/objective-exploration-bf16.json').read_text(encoding='utf-8'))
        cfg['registered_scoring'] = {
            'scene_floors': {'W1': 0.98, 'W2': 0.98, 'W3': 0.98, 'W4': 0.98},
            'scene_log_envelope': {'W1': 0.01, 'W2': 0.01, 'W3': 0.01, 'W4': 0.01},
        }
        prompt = launch_live.system_for(cfg)
        self.assertIn('Spark-X2.6-1.7B BF16', prompt)
        self.assertNotIn('GGUF Q8_0', prompt)
        self.assertIn('including current upstream code, PRs and issues', prompt)
        self.assertIn('returns page 1 directly', prompt)
        self.assertIn('After locating a hotspot', prompt)
        self.assertIn('Do not persist any FP8, INT8, FP4', prompt)

    def test_web_research_count_limit_compat_and_rpc_alignment(self):
        from evolveengine.v3_level3_web_research import WebResearch

        class _FakeResp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, n=-1): return json.dumps({'results': [{'url': 'https://example.com', 'title': 'ok', 'snippet': 's'}]}).encode()

        calls = []
        class _FakeProvider:
            def request(self, path, *, payload, timeout=60):
                calls.append((path, json.loads(payload)))
                return _FakeResp()

        with tempfile.TemporaryDirectory() as td:
            wr = WebResearch(_FakeProvider(), Path(td) / 'wr')
            r1 = wr.search('sglang triton', count=5)
            self.assertEqual(len(r1['results']), 1)
            self.assertEqual(calls[-1][1]['count'], 5)
            r2 = wr.search('sglang triton', limit=7)
            self.assertEqual(calls[-1][1]['count'], 7)
            r3 = wr.search('sglang triton', count=4, limit=4)
            self.assertEqual(calls[-1][1]['count'], 4)
            with self.assertRaises(ValueError):
                wr.search('sglang triton', count=5, limit=7)

        rpc_text = (Path(__file__).resolve().parents[1] / 'evolveengine/v3_level3_rpc.mjs').read_text(encoding='utf-8')
        self.assertIn('research_fetch', rpc_text)
        self.assertIn('research_reuse', rpc_text)
        self.assertIn('count?', rpc_text)
        self.assertIn('returns page 1 directly', rpc_text)

    def test_startup_knobs_and_jit_kernel_fast_component_kind(self):
        from evolveengine.v3_level3_startup import apply, options
        from evolveengine.v3_level3_components import _kind

        raw = json.dumps({
            'radix-cache': True,
            'enable-mixed-chunk': True,
            'disable-overlap-schedule': False,
            'swa-full-tokens-ratio': 0.75,
            'page-size': 16,
            'num-continuous-decode-steps': 4,
            'schedule-conservativeness': 0.5,
            'cuda-graph-max-bs': 32,
        }).encode('utf-8')
        opts = options(raw)
        self.assertEqual(opts['page-size'], 16)
        base = ('--dtype', 'bfloat16', '--disable-radix-cache', '--disable-overlap-schedule', '--context-length', '8192')
        argv = apply(base, raw)
        self.assertNotIn('--disable-radix-cache', argv)
        self.assertNotIn('--disable-overlap-schedule', argv)
        self.assertIn('--enable-mixed-chunk', argv)
        self.assertEqual(argv[argv.index('--swa-full-tokens-ratio') + 1], '0.75')
        self.assertEqual(argv[argv.index('--page-size') + 1], '16')
        for locked in (b'{"dtype":"float16"}', b'{"kv-cache-dtype":"fp8_e4m3"}', b'{"context-length":65536}'):
            with self.assertRaises(ValueError):
                options(locked)
        self.assertEqual(_kind('python/sglang/kernels/jit/csrc/elementwise/activation.cuh'), 'python')
        self.assertEqual(_kind('python/sglang/jit_kernel/csrc/fast_swa.cuh'), 'python')
        self.assertEqual(_kind('python/sglang/srt/layers/my_opt/__init__.py'), 'python')
        self.assertIsNone(_kind('python/sglang/__init__.py'))

    def test_workbuddy_profile_and_scoring_policy(self):
        from types import SimpleNamespace
        from evolveengine.v3_level3_deploy import BF16_WORKBUDDY_64K
        from evolveengine import v3_level3_scoring as scoring

        spec = SimpleNamespace(launch_profile=BF16_WORKBUDDY_64K, model_format='bf16_safetensors')
        argv = launch_args(spec)
        self.assertEqual(argv[argv.index('--context-length') + 1], '65536')
        self.assertEqual(argv[argv.index('--max-total-tokens') + 1], '524288')
        self.assertEqual(argv[argv.index('--max-running-requests') + 1], '24')
        self.assertEqual(argv[argv.index('--mem-fraction-static') + 1], '0.75')
        self.assertNotIn('--disable-radix-cache', argv)
        self.assertIn('--incremental-streaming-output', argv)
        with self.assertRaises(Exception):
            launch_args(SimpleNamespace(launch_profile=BF16_WORKBUDDY_64K, model_format='gguf'))

        pol = scoring.natural_policy(
            '1' * 64, '2' * 64, {'W1': 0.01, 'W2': 0.01, 'W3': 0.01, 'W4': 0.01},
            chat_corpus='3' * 64, version=scoring.WORKBUDDY_VERSION,
        )
        self.assertEqual(pol['weight_percent'], {'W1': 15, 'W2': 20, 'W3': 30, 'W4': 35})
        self.assertEqual(scoring.resolve_policy(pol), pol)
        s = scoring.weighted_score({'W1': 1.0, 'W2': 1.0, 'W3': 1.2, 'W4': 1.5}, pol['weight_percent'])
        self.assertGreater(s, 1.20)
        sla = scoring.workbuddy_sla_summary({
            'W4': {'share_ge_40_tps': 0.0, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 27.8, 'unfinished_count': 0}
        })
        self.assertEqual(sla['W4']['share_ge_40_tps'], 0.0)
        cfg = json.loads((Path(__file__).resolve().parents[1] /
                          'experiments/spark26_gguf/level3/objective-exploration-bf16-workbuddy.json').read_text(encoding='utf-8'))
        cfg['registered_scoring'] = pol
        prompt = launch_live.system_for(cfg)
        self.assertIn('0.15 ln(W1) + 0.20 ln(W2) + 0.30 ln(W3) + 0.35 ln(W4)', prompt)
        self.assertIn('W1 single-request decode 15%', prompt)
        self.assertIn('W2 long-context cold TTFT 20%', prompt)
        self.assertIn('W3 multi-session hot-prefix decode 30%', prompt)
        self.assertIn('W4 mixed hot-decode + cold-prefill serving 35%', prompt)
        self.assertNotIn('W2 ordinary interactive decode 70%', prompt)

    def test_workbuddy_delivery_replay_parity_and_evidence_sla(self):
        from evolveengine import v3_level3_scoring as scoring
        from evolveengine.v3_level3_delivery import _same_measurement
        from evolveengine.v3_level3_evidence import (
            EvidencePolicy, SCHEMA, WORKBUDDY_SCORE_RULE, _measure,
        )
        from evolveengine.v3_harness import canonical

        pol = scoring.natural_policy(
            '1' * 64, '2' * 64, {'W1': 0.01, 'W2': 0.01, 'W3': 0.01, 'W4': 0.01},
            chat_corpus='3' * 64, version=scoring.WORKBUDDY_VERSION,
        )
        ratios = {'W1': 1.0, 'W2': 1.0, 'W3': 1.2, 'W4': 1.5}
        trial_scenes = {
            'W1': {'dev_score': 50.0, 'tpot_ms': 20.0, 'ttft_ms': 100.0, 'output_toks_per_s': 45.0,
                   'share_ge_40_tps': 1.0, 'share_ge_50_tps': 1.0, 'p95_itl_ms': 21.0, 'unfinished_count': 0},
            'W2': {'dev_score': 0.65, 'tpot_ms': 22.0, 'ttft_ms': 1538.0, 'output_toks_per_s': 20.0,
                   'share_ge_40_tps': 1.0, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 23.0, 'unfinished_count': 0},
            'W3': {'dev_score': 48.0, 'tpot_ms': 20.8, 'ttft_ms': 300.0, 'output_toks_per_s': 300.0,
                   'share_ge_40_tps': 0.75, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 25.5, 'unfinished_count': 0},
            'W4': {'dev_score': 150.0, 'tpot_ms': 35.0, 'ttft_ms': 2000.0, 'output_toks_per_s': 150.0,
                   'share_ge_40_tps': 0.0, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 31.2, 'unfinished_count': 0},
        }
        dev_cmp = {
            'reference': 'a' * 64, 'receipt': 'b' * 64, 'scene_ratios': ratios,
            'equal_scene_geomean': 1.1583, 'interpretation': 'descriptive_development_comparison',
            'scoring_version': pol['version'],
            'weighted_score': scoring.weighted_score(ratios, pol['weight_percent']),
            'workbuddy_sla': scoring.workbuddy_sla_summary(trial_scenes),
        }
        # Delivery replay must produce identical weighted_score and workbuddy_sla
        replayed = dict(dev_cmp)
        replayed['weighted_score'] = scoring.weighted_score(replayed['scene_ratios'], pol.get('weight_percent'))
        replayed['workbuddy_sla'] = scoring.workbuddy_sla_summary(trial_scenes)
        self.assertTrue(_same_measurement(replayed, dev_cmp))
        self.assertAlmostEqual(replayed['weighted_score'], 1.2172683, places=5)

        # Verify _measure computes per-scene SLA metrics and uses WORKBUDDY_SCORE_RULE
        scenes = {'W1': ((128, 8),), 'W2': ((256, 8),), 'W3': ((128, 8), (128, 8)), 'W4': ((128, 8), (256, 8))}
        policy = EvidencePolicy(
            engine='e' * 64, worker_id='w', boot_id='b', device_uuid='u',
            weights={'tokenizer.json': '1' * 64}, loaded_artifacts={'a': '2' * 64},
            quality_rule_sha256='3' * 64, quality_cases=('numeric_32',),
            measure_workload_sha256='4' * 64, profile_workload_sha256='5' * 64,
            scenes=scenes, warm_waves=1, timed_waves=1,
            memory_stages=('ready', 'post_requests'), memory_limit_bytes=None,
            profile_target=('nsys',), natural_eos=True, score_rule=WORKBUDDY_SCORE_RULE, workbuddy_sla=True,
        )
        with tempfile.TemporaryDirectory() as td:
            tpath = Path(td) / 'timing.jsonl'
            binding = dict(
                schema=SCHEMA, operation='develop_measure', engine='e' * 64, worker_id='w', boot_id='b',
                device_uuid='u', origin='offline_fixture', run_id='r', service_id='s',
                start_ns=1_000_000, end_ns=999_000_000_000,
                workload_sha256='4' * 64, clock='external_client_monotonic', timing_mode='natural_eos',
            )
            t = 10_000_000
            rows = [binding]
            for scene, shape in scenes.items():
                for widx in range(2):
                    reqs = []
                    for lane, (inp, out) in enumerate(shape):
                        start = t
                        first = start + (500_000_000 if scene == 'W2' else 50_000_000)
                        # Lane 0 fast (50 tok/s -> 20ms/tok), lane 1 slow (25 tok/s -> 40ms/tok)
                        step_ns = 20_000_000 if lane == 0 else 40_000_000
                        last = first + (out - 1) * step_ns
                        done = last + 1_000_000
                        reqs.append(dict(
                            rid=f'{scene}-{widx}-{lane}', input=inp, output=out, output_cap=out,
                            finish_reason='length', sha256='c' * 64,
                            start_ns=start, first_ns=first, last_ns=last, done_ns=done,
                            first_token_unambiguous=True, stream_complete=True,
                            reported_speculation=None, p95_itl_ms=step_ns / 1e6,
                        ))
                    t += 5_000_000_000
                    rows.append(dict(scene=scene, index=widx, timed=widx >= 1, requests=reqs))
            tpath.write_bytes(b''.join(canonical(r) + b'\n' for r in rows))
            data = _measure(binding, policy, tpath)
            self.assertAlmostEqual(data['scenes']['W2']['dev_score'], 2.0, places=3)  # 1000 / 500ms TTFT
            self.assertEqual(data['scenes']['W3']['share_ge_40_tps'], 0.5)
            self.assertEqual(data['scenes']['W3']['share_ge_50_tps'], 0.5)
            self.assertEqual(data['scenes']['W3']['p95_itl_ms'], 40.0)
            self.assertEqual(data['scenes']['W3']['unfinished_count'], 0)
            self.assertNotIn('w3_requests', data)

    def test_workbuddy_admission_rejects_legacy_8k_registration(self):
        from unittest.mock import patch
        from evolveengine.v3_harness import canonical
        from experiments.spark26_gguf.level3 import live_split_quality as split
        wb_approval = Path(r'C:\Users\rog\AppData\Local\EvolveEngine\l3-bf16-20261126\approval-bf16-workbuddy.json')
        if not wb_approval.is_file():
            self.skipTest('local legacy 8K approval fixture not present')
        wb_cfg = Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf/level3/objective-exploration-bf16-workbuddy.json'
        plan = json.loads(wb_approval.read_bytes())
        plan['controller_sha256'] = launch_live.controller_pins()
        sr_doc = json.loads(Path(plan['evidence']['split_quality_registration']['path']).read_bytes())
        with tempfile.TemporaryDirectory() as td, patch.object(split, 'extra_sources', return_value=sr_doc['sources']):
            p = Path(td).resolve() / 'legacy-8k-wb.json'
            raw = canonical(plan) + b'\n'
            p.write_bytes(raw)
            ref = {'path': str(p), 'sha256': digest(raw)}
            with self.assertRaisesRegex(ValueError, 'BF16 64K WorkBuddy baseline registration'):
                launch_live.audit(ref, config_path=wb_cfg)

    def test_workbuddy_profile_and_stress_sampling_on_small_corpus(self):
        from evolveengine.v3_level3_workload import TimingSchedule, profile_inputs
        from experiments.spark26_gguf.level3.node_preflight import WORKBUDDY_SCENES

        # 121,631 tokens total (same capacity as registered timing-corpus.json), while 1 WorkBuddy wave needs 223,232 tokens
        docs = [list(range(100 + i * 20_000, 100 + i * 20_000 + 30_408)) for i in range(4)]
        seed = 'a' * 64
        prof = profile_inputs(docs, WORKBUDDY_SCENES, 1, seed)
        self.assertEqual(set(prof), {'W1', 'W2', 'W3', 'W4'})
        self.assertEqual(len(prof['W3'][0]), 8)
        self.assertEqual(len(prof['W4'][0]), 8)
        self.assertEqual(len(prof['W2'][0][0]), 16384)
        sched = TimingSchedule(docs, WORKBUDDY_SCENES, 3, seed, allow_wrap=True)
        inputs, hashes = sched.get(0)
        all_pins = [pin for waves in hashes.values() for wave in waves for pin in wave]
        self.assertEqual(len(all_pins), len(set(all_pins)))

    def test_profile_bad_inputs_return_incomplete_without_killing_worker(self):
        from types import SimpleNamespace
        from experiments.spark26_gguf.level3.worker_stdio import EngineeringWorker

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            w = EngineeringWorker.__new__(EngineeringWorker)
            w.root = root
            w.quarantined = False
            w.deployments = {'b' * 64: object()}
            w.profiles = {}
            w.verify = lambda r: True
            w._save = lambda path, fields: (path.write_bytes(json.dumps(fields).encode()), digest(path.read_bytes()))[1]
            w.profile_timing = SimpleNamespace(
                workload=json.dumps({'scenes': {'W1': [[2048, 256]]}, 'timed_waves': 2}),
                files={'observe_workbuddy_timing.py': b''},
                policy=SimpleNamespace(workbuddy_sla=True),
                schedule=SimpleNamespace(documents=[[1] * 10], seed='a' * 64),
            )
            eng = SimpleNamespace(digest='e' * 64)
            bld = SimpleNamespace(operation='develop_build', status='built', engine='e' * 64, evidence='b' * 64)
            res = w.profile(eng, bld, 500_000, {'scenes': ['UNKNOWN_SCENE'], 'windows': 1})
            self.assertEqual(res.status, 'incomplete_non_scoring')
            self.assertEqual(res.data['failure'], 'ValueError')
            self.assertIn(res.evidence, w.profiles)

    def test_delivery_cli_selects_bf16_contract_and_observatory_uses_workbuddy_weights(self):
        from unittest.mock import patch
        from evolveengine import v3_level3_delivery as delivery, observatory_data as obs, v3_level3_scoring as scoring
        from evolveengine.v3_harness import canonical

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            run_dir = root / 'wb-run'
            run_dir.mkdir()
            reg = {
                'approval_sha256': '1' * 64,
                'weight_audit': weights.BF16_POLICY,
                'business_scoring': {
                    'version': scoring.WORKBUDDY_VERSION,
                    'weight_percent': dict(scoring.WORKBUDDY_WEIGHTS),
                },
            }
            (run_dir / 'registration.json').write_bytes(canonical(reg) + b'\n')
            with patch.object(SourceContract, 'acquire_spark26_bf16_e0', return_value='BF16_CONTRACT') as m_bf16, \
                 patch.object(SourceContract, 'acquire_spark26_e0', return_value='Q8_CONTRACT') as m_q8:
                self.assertEqual(delivery.contract_for_run(run_dir, Path('sglang.tar.gz')), 'BF16_CONTRACT')
                m_bf16.assert_called_once()
                m_q8.assert_not_called()

            # Observatory operator replay must accept 1.2172683 (WorkBuddy) and reject 1.029825 (legacy 20/70/5/5)
            wb_score = scoring.weighted_score({'W1': 1.0, 'W2': 1.0, 'W3': 1.2, 'W4': 1.5}, scoring.WORKBUDDY_WEIGHTS)
            legacy_score = scoring.weighted_score({'W1': 1.0, 'W2': 1.0, 'W3': 1.2, 'W4': 1.5}, scoring.WEIGHTS)
            base_id, cand_id = 'a' * 64, 'b' * 64
            cases = {k: 'pass' for k in obs.REPLAY_CASES}
            sidecar = {
                'schema': 'ee-observatory-operator-v1',
                'run': 'wb-run',
                'approval_sha256': '1' * 64,
                'measurements': [
                    {'engine': cand_id, 'timing': '2' * 64, 'build': '3' * 64, 'quality': '4' * 64, 'reference': '5' * 64,
                     'reference_engine': base_id, 'evidence_origin': 'operator_replay', 'verdict': 'measured',
                     'weight_audit': 'pending_if_gain', 'cases': cases,
                     'ratios': {'W1': 1.0, 'W2': 1.0, 'W3': 1.2, 'W4': 1.5}, 'weighted_score': legacy_score},
                    {'engine': cand_id, 'timing': '6' * 64, 'build': '3' * 64, 'quality': '4' * 64, 'reference': '5' * 64,
                     'reference_engine': base_id, 'evidence_origin': 'operator_replay', 'verdict': 'measured',
                     'weight_audit': 'pending_if_gain', 'cases': cases,
                     'ratios': {'W1': 1.0, 'W2': 1.0, 'W3': 1.2, 'W4': 1.5}, 'weighted_score': wb_score},
                ],
            }
            (root / 'wb-run.observatory-operator.json').write_bytes(canonical(sidecar) + b'\n')
            out = {'id': 'wb-run', 'source_ref': '1' * 64, 'baseline': base_id, 'points': [], 'events': []}
            obs.collect_operator_measurements(out, run_dir, {base_id: 0, cand_id: 1})
            self.assertEqual(len(out['points']), 1)
            self.assertEqual(out['points'][0]['id'], '6' * 64)
            self.assertAlmostEqual(out['points'][0]['aggregate'], 1.2172683, places=5)


if __name__ == '__main__':
    unittest.main()
