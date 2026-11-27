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

    def test_workbuddy_v2_scoring_guardrails_and_precision_targets(self):
        from math import exp, log
        from evolveengine import v3_level3_scoring as scoring
        from experiments.spark26_gguf.level3 import repeat_e0

        # 1. Precision target rule: min(2.5%, max(1.5%, 1.2 * half))
        tight = scoring.workbuddy_v2_precision_target([0.001, -0.001, 0.002, -0.002, 0.001, -0.001])
        self.assertEqual(tight, 1.015)
        wide = scoring.workbuddy_v2_precision_target([0.04, -0.04, 0.03, -0.03, 0.04, -0.04])
        self.assertEqual(wide, 1.025)
        mid = scoring.workbuddy_v2_precision_target([0.015, -0.015, 0.016, -0.016, 0.014, -0.014])
        self.assertGreater(mid, 1.015)
        self.assertLess(mid, 1.025)

        # 2. Policy creation & roundtrip resolve_policy
        env = {'W1': 0.01, 'W2': 0.02, 'W3': 0.03, 'W4_hot': 0.025, 'W4_cold': 0.03, 'W5': 0.03}
        genv = {'w4_cold_e2e': 0.03, 'g64k_ttft': 0.03, 'g64k_decode': 0.02, 'p95_itl': 0.05}
        ptargets = {'W3': 1.022, 'W4_hot': 1.020, 'W4_cold': 1.024, 'W5': 1.021}
        rfloors = {s: 0.90 for s in scoring.WORKBUDDY_V2_REPLAY_SCENES}
        pol = scoring.natural_policy(
            '1' * 64, '2' * 64, env, chat_corpus='3' * 64, replay_floors=rfloors,
            version=scoring.WORKBUDDY_V2_VERSION, scene_precision_targets=ptargets, guardrail_envelope=genv,
        )
        self.assertEqual(scoring.resolve_policy(pol), pol)
        self.assertEqual(pol['weight_percent'], {'W1': 10, 'W2': 15, 'W3': 25, 'W4_hot': 15, 'W4_cold': 15, 'W5': 20})
        self.assertEqual(pol['guardrail_floors']['p95_itl'], 0.80)

        # 3. summarize_build: passes when healthy, rejects on w4_cold_e2e / g64k / p95_itl / unfinished regression
        def make_jobs(g_override=None, unfinished=0, w3_vals=None):
            jobs = []
            w3_seq = w3_vals or [1.10, 1.11, 1.09, 1.10, 1.11, 1.09]
            for i in range(6):
                ref_id = f'{i:064x}'
                cand_id = f'{i+100:064x}'
                g = {'w4_cold_e2e': 1.02, 'g64k_ttft': 1.01, 'g64k_decode': 1.01, 'p95_itl': 0.95}
                if g_override:
                    g.update(g_override)
                ref_row = {'evidence': ref_id, 'engine': 'e0', 'build': 'b0'}
                cand_row = {
                    'evidence': cand_id, 'engine': 'c1', 'build': 'bc1',
                    'data': {'measurement_pair': i + 1, 'measurement_inputs_sha256': f'{i+200:064x}'},
                    'performance': {
                        'reference': ref_id,
                        'scene_ratios': {'W1': 1.02, 'W2': 1.05, 'W3': w3_seq[i], 'W4_hot': 1.15, 'W4_cold': 1.08, 'W5': 1.25},
                        'guardrail_ratios': g,
                        'unfinished_count': unfinished,
                    },
                }
                if i % 2 == 0:
                    jobs.extend([ref_row, cand_row])
                else:
                    jobs.extend([cand_row, ref_row])
            return jobs

        ok = scoring.summarize_build(make_jobs(), 'c1', 'bc1', pol)
        self.assertEqual(ok['status'], 'eligible')
        self.assertTrue(ok['eligible_for_delivery'])

        reg_e2e = scoring.summarize_build(make_jobs(g_override={'w4_cold_e2e': 0.85}), 'c1', 'bc1', pol)
        self.assertEqual(reg_e2e['status'], 'regression')
        self.assertIn('w4_cold_e2e', reg_e2e['below_floor'])
        self.assertFalse(reg_e2e['eligible_for_delivery'])

        reg_64k = scoring.summarize_build(make_jobs(g_override={'g64k_ttft': 0.80}), 'c1', 'bc1', pol)
        self.assertEqual(reg_64k['status'], 'regression')
        self.assertIn('g64k_ttft', reg_64k['below_floor'])

        reg_itl = scoring.summarize_build(make_jobs(g_override={'p95_itl': 0.70}), 'c1', 'bc1', pol)
        self.assertEqual(reg_itl['status'], 'regression')
        self.assertIn('p95_itl', reg_itl['below_floor'])

        reg_unf = scoring.summarize_build(make_jobs(unfinished=1), 'c1', 'bc1', pol)
        self.assertEqual(reg_unf['status'], 'regression')
        self.assertIn('unfinished_count', reg_unf['below_floor'])

        # 4. repeat_e0.summarize on 6 v2 pairs
        v2_pairs = []
        for idx, (a, b) in enumerate(repeat_e0.ORDER):
            arm_a = {
                'arm': a,
                'scenes': {
                    'W1': {'dev_score': 54.0}, 'W2': {'dev_score': 0.65},
                    'W3': {'dev_score': 33.0, 'p95_itl_ms': 24.0},
                    'W4': {'dev_score': 10.0, 'w4_hot_decode_tps': 32.0, 'w4_cold_ttft_ms': 3000.0, 'w4_cold_e2e_ms': 8000.0, 'p95_itl_ms': 25.0},
                    'W5': {'dev_score': 28.0, 'p95_itl_ms': 34.0},
                },
                'guardrails': {'G64K': {'ttft_ms': 9900.0, 'decode_tokens_per_s': 37.0}},
            }
            arm_b = {
                'arm': b,
                'scenes': {
                    'W1': {'dev_score': 54.2}, 'W2': {'dev_score': 0.655},
                    'W3': {'dev_score': 33.3 if idx % 2 == 0 else 32.7, 'p95_itl_ms': 24.2},
                    'W4': {'dev_score': 10.1, 'w4_hot_decode_tps': 32.2, 'w4_cold_ttft_ms': 3020.0, 'w4_cold_e2e_ms': 8040.0, 'p95_itl_ms': 25.1},
                    'W5': {'dev_score': 28.2, 'p95_itl_ms': 34.3},
                },
                'guardrails': {'G64K': {'ttft_ms': 9950.0, 'decode_tokens_per_s': 37.1}},
            }
            v2_pairs.append([arm_a, arm_b])
        summary = repeat_e0.summarize(v2_pairs)
        self.assertTrue(summary['stable'])
        self.assertEqual(set(summary['descriptive_log_envelope']['scenes']), set(scoring.WORKBUDDY_V2_WEIGHTS))
        self.assertEqual(set(summary['guardrail_log_envelope']), set(scoring.GUARDRAIL_KEYS))
        self.assertEqual(set(summary['scene_precision_targets']), set(scoring.WORKBUDDY_V2_PRIMARY_SCENES))

    def test_long_context_logprob_calibration_and_grading(self):
        from math import log
        from evolveengine import v3_level3_long_ctx as lc
        from evolveengine.v3_harness import canonical, digest

        def make_dist(top1_id, p1, other_id, p2):
            return {
                'prompt_tokens': 16384, 'prompt_sha256': 'a' * 64,
                'top1_id': top1_id, 'top1_logprob': log(p1),
                'top_logprobs': {str(top1_id): log(p1), str(other_id): log(p2)},
            }

        e0a = {
            'ctx16k': make_dist(10, 0.90, 11, 0.095),
            'ctx64k': dict(make_dist(20, 0.85, 21, 0.145), prompt_tokens=64000, prompt_sha256='b' * 64),
        }
        e0b = {
            'ctx16k': make_dist(10, 0.899, 11, 0.096),
            'ctx64k': dict(make_dist(20, 0.849, 21, 0.146), prompt_tokens=64000, prompt_sha256='b' * 64),
        }
        cal = lc.calibrate_bounds(e0a, e0b)
        self.assertLess(cal['e0e0_max_tv_upper'], 0.01)
        self.assertGreaterEqual(cal['max_tv_upper'], 0.02)
        with tempfile.TemporaryDirectory() as td:
            anchor_path = Path(td) / 'anchor_long_ctx.json'
            doc = {'schema': lc.SCHEMA, 'engine': 'c' * 64, 'calibration': cal, 'cases': e0a}
            raw = canonical(doc) + b'\n'
            anchor_path.write_bytes(raw)
            sha = digest(raw)
            good_rows = [dict(case=k, **v) for k, v in e0b.items()]
            self.assertEqual(lc.grade_long_ctx(good_rows, anchor_path, sha)['status'], 'valid')
            bad_rows = [
                dict(case='ctx16k', **e0b['ctx16k']),
                dict(case='ctx64k', **dict(make_dist(99, 0.95, 98, 0.04), prompt_tokens=64000, prompt_sha256='b' * 64)),
            ]
            res = lc.grade_long_ctx(bad_rows, anchor_path, sha)
            self.assertEqual(res['status'], 'violation')
            self.assertTrue(res['cases']['ctx16k']['passed'])
            self.assertFalse(res['cases']['ctx64k']['passed'])

    def test_p0_worker_public_receipt_preserves_guardrails_and_compare_dev(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from evolveengine import v3_level3_scoring as scoring
        from evolveengine.v3_level3_worker import NativeWorker

        pol = scoring.natural_policy(
            '1' * 64, '2' * 64,
            {'W1': 0.01, 'W2': 0.01, 'W3': 0.01, 'W4_hot': 0.0127, 'W4_cold': 0.01, 'W5': 0.01},
            chat_corpus='3' * 64, version=scoring.WORKBUDDY_V2_VERSION,
            scene_precision_targets={'W3': 1.015, 'W4_hot': 1.015, 'W4_cold': 1.015, 'W5': 1.015},
            guardrail_envelope={'w4_cold_e2e': 0.01, 'g64k_ttft': 0.01, 'g64k_decode': 0.0158, 'p95_itl': 0.0209},
        )
        fake_validated = {
            'status': 'valid',
            'scenes': {
                'W1': {'dev_score': 55.0, 'tpot_ms': 18.0, 'ttft_ms': 120.0, 'output_toks_per_s': 55.0,
                       'share_ge_40_tps': 1.0, 'share_ge_50_tps': 1.0, 'p95_itl_ms': 19.0, 'unfinished_count': 0},
                'W2': {'dev_score': 0.65, 'tpot_ms': 19.0, 'ttft_ms': 1538.0, 'output_toks_per_s': 48.0,
                       'share_ge_40_tps': 1.0, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 20.0, 'unfinished_count': 0},
                'W3': {'dev_score': 35.0, 'tpot_ms': 28.0, 'ttft_ms': 300.0, 'output_toks_per_s': 280.0,
                       'share_ge_40_tps': 0.25, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 24.0, 'unfinished_count': 0},
                'W4': {'dev_score': 33.0, 'tpot_ms': 29.0, 'ttft_ms': 400.0, 'output_toks_per_s': 240.0,
                       'share_ge_40_tps': 0.0, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 26.0, 'unfinished_count': 0,
                       'w4_hot_decode_tps': 33.0, 'w4_cold_ttft_ms': 3200.0, 'w4_cold_e2e_ms': 8200.0},
                'W5': {'dev_score': 31.0, 'tpot_ms': 32.0, 'ttft_ms': 3800.0, 'output_toks_per_s': 344.0,
                       'share_ge_40_tps': 0.0, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 35.0, 'unfinished_count': 0},
            },
            'guardrails': {'G64K': {'ttft_ms': 10200.0, 'decode_tokens_per_s': 38.0, 'unfinished_count': 0}},
            'memory': {'peak_device_bytes': 123456},
            'measurement_pair': 0,
            'measurement_inputs_sha256': 'a' * 64,
            '_private_replay': {'should_not_leak': True},
        }
        from evolveengine.v3_level3_collect import Observation
        from evolveengine.v3_level3_evidence import EvidencePolicy

        w = NativeWorker.__new__(NativeWorker)
        w.variant = 'sm100'
        w.service = SimpleNamespace(spec=SimpleNamespace(gpu_uuid='u'))
        pol_ev = EvidencePolicy(
            engine='b' * 64, worker_id='w', boot_id='b', device_uuid='u',
            weights={'tokenizer.json': '1' * 64}, loaded_artifacts={'a': '2' * 64},
            quality_rule_sha256='3' * 64, quality_cases=('numeric_32',),
            measure_workload_sha256='4' * 64, profile_workload_sha256='5' * 64,
            scenes={'W1': ((64, 8),)}, warm_waves=1, timed_waves=2,
            memory_stages=('ready', 'post_requests'), memory_limit_bytes=None,
            profile_target=('nsys',), natural_eos=False, origin='hardware_collection',
        )
        dep = SimpleNamespace(candidate='b' * 64, native_mapping=lambda v: (None, {'a': '2' * 64}))
        fake_check = SimpleNamespace(data={'operations': {'develop_measure': SimpleNamespace(status='observed', data=fake_validated)}})
        with tempfile.TemporaryDirectory() as td, \
             patch('evolveengine.v3_level3_worker.validate_evidence', return_value=fake_check):
            obs = Observation(pol_ev, 'f' * 64)
            status, public = w._validate('develop_measure', Path(td) / 'measure', obs, {'tokenizer.json': '1' * 64}, dep, check_replay=False)
        self.assertEqual(status, 'observed')
        self.assertIn('guardrails', public)
        self.assertEqual(public['guardrails']['G64K']['ttft_ms'], 10200.0)
        self.assertNotIn('_private_replay', public)

        from evolveengine.v3_level3_run import compare_dev
        stub = SimpleNamespace(operation='develop_measure', status='observed', evidence='e' * 64, data=public)
        startup_cmp = compare_dev(stub, stub)
        self.assertEqual(set(startup_cmp['guardrail_ratios']), {'w4_cold_e2e', 'g64k_ttft', 'g64k_decode', 'p95_itl'})
        self.assertAlmostEqual(scoring.weighted_score(startup_cmp['scene_ratios'], pol['weight_percent']), 1.0, places=6)

        # Fail-closed: stripping guardrails must still raise ValueError
        broken = SimpleNamespace(operation='develop_measure', status='observed', evidence='e' * 64,
                                 data={k: v for k, v in public.items() if k != 'guardrails'})
        with self.assertRaisesRegex(ValueError, 'invalid workbuddy v2 guardrail metrics'):
            compare_dev(broken, broken)

        # Verify qualify_split_worker.check_bf16 executes compare_dev on the real public receipt
        from experiments.spark26_gguf.level3.qualify_split_worker import check_bf16
        pub_with_auth = dict(public, output_authentication={'status': 'verified'})
        check_bf16('positive-measure', 'observed', pub_with_auth, True, {}, '0' * 64, '0' * 64)
        with self.assertRaisesRegex(ValueError, 'invalid workbuddy v2 guardrail metrics'):
            check_bf16('positive-measure', 'observed',
                       {k: v for k, v in pub_with_auth.items() if k != 'guardrails'}, True, {}, '0' * 64, '0' * 64)

    def test_p1_develop_stress_wave_count_matches_collector_and_measure(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from evolveengine.v3_harness import canonical, digest
        from evolveengine.v3_level3_collect import TimingCollector
        from evolveengine.v3_level3_evidence import (
            EvidencePolicy, SCHEMA, WORKBUDDY_V2_SCORE_RULE, _measure,
        )
        from experiments.spark26_gguf.level3.node_preflight import (
            WORKBUDDY_V2_SCENES, WORKBUDDY_V2_SCENE_TIMED, WORKBUDDY_V2_SCENE_WARM,
        )

        # Use scaled-down token lengths with exact WORKBUDDY_V2_SCENES keys/lanes & wave config
        small_scenes = {s: tuple((64, 8) for _ in shape) for s, shape in WORKBUDDY_V2_SCENES.items()}
        docs = [list(range(100 + i * 500, 100 + (i + 1) * 500)) for i in range(4)]
        obs_files = {
            'observe_timing.py': b'# timing\n',
            'timing.py': b'# lib\n',
            'observe_workbuddy_timing.py': b'# wb\n',
        }
        obs_hashes = {k: digest(v) for k, v in obs_files.items()}
        input_plan = {'distribution': 'fixed-v1', 'seed': 'a' * 64}
        plan = dict(
            scenes=small_scenes, warm_waves=1, timed_waves=2,
            observer_sha256=obs_hashes, score_rule=WORKBUDDY_V2_SCORE_RULE,
            workbuddy_sla=True, workbuddy_v2=True,
            scene_warm_waves=dict(WORKBUDDY_V2_SCENE_WARM),
            scene_timed_waves=dict(WORKBUDDY_V2_SCENE_TIMED),
            input_plan=input_plan,
        )
        wsha = digest(canonical(plan) + b'\n')
        policy = EvidencePolicy(
            engine='e' * 64, worker_id='w', boot_id='b', device_uuid='u',
            weights={'tokenizer.json': '1' * 64}, loaded_artifacts={'a': '2' * 64},
            quality_rule_sha256='3' * 64, quality_cases=('numeric_32',),
            measure_workload_sha256=wsha, profile_workload_sha256='5' * 64,
            scenes=small_scenes, warm_waves=1, timed_waves=2,
            memory_stages=('ready', 'post_requests'), memory_limit_bytes=None,
            profile_target=('nsys',), natural_eos=False, score_rule=WORKBUDDY_V2_SCORE_RULE,
            workbuddy_sla=True, workbuddy_v2=True,
            scene_warm_waves=dict(WORKBUDDY_V2_SCENE_WARM),
            scene_timed_waves=dict(WORKBUDDY_V2_SCENE_TIMED),
        )
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            obs_dir = td_path / 'obs'
            obs_dir.mkdir()
            obs_paths = {}
            for name, raw in obs_files.items():
                p = obs_dir / name
                p.write_bytes(raw)
                obs_paths[name] = p
            collector = TimingCollector(
                None, None, policy, obs_paths, obs_hashes,
                documents=docs, input_plan=input_plan,
            )
            inputs, _ = collector.schedule.get(0)
            wave_counts = {s: len(w) for s, w in inputs.items()}
            self.assertEqual(wave_counts, {'W1': 3, 'W2': 3, 'W3': 6, 'W4': 5, 'W5': 3, 'G64K': 1})
            self.assertEqual(sum(wave_counts.values()), 21)

            def fake_observer(service, observer_input, operation, deadline, *, script, **kwargs):
                staged_inputs = json.loads((observer_input / 'inputs.json').read_bytes())
                (operation / 'observer.stdout').write_bytes(b'')
                (operation / 'observer.stderr').write_bytes(b'')
                (operation / 'observer.inspect.json').write_bytes(b'{}\n')
                t = 10_000_000
                lines = []
                for scene, waves in staged_inputs.items():
                    warm_n = WORKBUDDY_V2_SCENE_WARM.get(scene, 1)
                    for widx, wave in enumerate(waves):
                        reqs = []
                        for lane, ids in enumerate(wave):
                            out = small_scenes[scene][lane][1]
                            start = t
                            first = start + 50_000_000
                            second = first + 25_000_000
                            last = first + (out - 1) * 25_000_000
                            done = last + 1_000_000
                            reqs.append(dict(
                                rid=f'{scene}-{widx}-{lane}', input=len(ids), output=out,
                                sha256='c' * 64, input_sha256=digest(canonical(ids)),
                                start_ns=start, first_ns=first, second_ns=second, last_ns=last, done_ns=done,
                                first_token_unambiguous=True, stream_complete=True,
                                reported_speculation=None, cached_tokens=0, p95_itl_ms=25.0,
                            ))
                        t += 5_000_000_000
                        lines.append(json.dumps(dict(scene=scene, index=widx, timed=widx >= warm_n, requests=reqs)))
                return '\n'.join(lines)

            def fake_identity(host, service, service_spec, operation, op, pol_in, deadline, variant):
                for fn in ('external-identity.json', 'host-mappings.json', 'service.inspect.json'):
                    (operation / fn).write_bytes(b'{}\n')
                binding = dict(
                    schema=SCHEMA, operation=op, engine='e' * 64, worker_id='w', boot_id='b',
                    device_uuid='u', origin='offline_fixture', run_id='r', service_id='s',
                    start_ns=1_000_000, end_ns=999_000_000_000,
                )
                return pol_in, binding

            collector.observer = fake_observer
            archive = td_path / 'stress-run'
            with patch('evolveengine.v3_level3_collect.identity', side_effect=fake_identity):
                obs = collector('develop_measure', object(), archive, deadline=10**18, pair_index=0, stress=True)
            timing_path = archive / 'develop_measure' / 'timing.jsonl'
            rows = [json.loads(line) for line in timing_path.read_bytes().splitlines()]
            self.assertEqual(len(rows), 22)  # 1 binding header + 21 wave rows
            receipt = json.loads((archive / 'develop_measure' / 'receipt.json').read_bytes())
            measured = _measure(receipt, obs.policy, timing_path)
            self.assertEqual(set(measured['scenes']), {'W1', 'W2', 'W3', 'W4', 'W5'})
            self.assertIn('G64K', measured['guardrails'])
            self.assertGreater(measured['guardrails']['G64K']['decode_tokens_per_s'], 0)


if __name__ == '__main__':
    unittest.main()
