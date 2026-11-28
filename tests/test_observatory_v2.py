"""Tests for WorkBuddy v2 Observatory projection, run-scoped trace/diff APIs, and security."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from evolveengine import observatory as api
from evolveengine import observatory_data as projection
from evolveengine.journal import Journal
from evolveengine.schema import canonical
from evolveengine.v3_harness import Lineage, Snapshot
from evolveengine.v3_level3_trace import TraceRecorder

E0, E1, E2 = '0' * 64, '1' * 64, '2' * 64
B0, B1 = 'a' * 64, 'b' * 64
Q0, Q1 = 'c' * 64, 'd' * 64
M0, M1 = 'e' * 64, 'f' * 64
WB2_REG = {
    'scope': 'one_engineering_exploration_without_holdout',
    'holdout_review_sha256': None,
    'full_acceptance': False,
    'approval_sha256': '9' * 64,
    'business_scoring': {
        'version': 'workbuddy-agent-serving-v2',
        'weight_percent': {'W1': 10, 'W2': 15, 'W3': 25, 'W4_hot': 15, 'W4_cold': 15, 'W5': 20},
        'scene_floors': {'W1': 0.99, 'W2': 0.99, 'W3': 0.99, 'W4_hot': 0.987, 'W4_cold': 0.99, 'W5': 0.99},
        'guardrail_floors': {'w4_cold_e2e': 0.99, 'g64k_ttft': 0.99, 'g64k_decode': 0.984, 'p95_itl': 0.80},
        'scene_precision_targets': {'W3': 1.015, 'W4_hot': 1.015, 'W4_cold': 1.015, 'W5': 1.015},
        'minimum_pairs': 6,
        'sla_targets_tps': [40.0, 50.0],
    },
}


def _write(path: Path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding='utf-8')


def _wb2_scenes(scale=1.0):
    return {
        'W1': {'dev_score': 54.0 * scale, 'output_toks_per_s': 54.0 * scale, 'tpot_ms': 18.5 / scale, 'ttft_ms': 180.0, 'share_ge_40_tps': 1.0, 'share_ge_50_tps': 1.0, 'p95_itl_ms': 19.0, 'unfinished_count': 0},
        'W2': {'dev_score': 0.65 * scale, 'output_toks_per_s': 49.0, 'tpot_ms': 20.4, 'ttft_ms': 1540.0 / scale, 'cached_tokens_min': 4, 'unfinished_count': 0},
        'W3': {'dev_score': 43.0 * scale, 'output_toks_per_s': 330.0 * scale, 'tpot_ms': 23.2 / scale, 'ttft_ms': 530.0, 'share_ge_40_tps': 0.85, 'share_ge_50_tps': 0.10, 'p95_itl_ms': 25.0, 'cached_tokens_min': 11770, 'unfinished_count': 0},
        'W4': {'dev_score': 132.0 * scale, 'output_toks_per_s': 132.0 * scale, 'tpot_ms': 28.0, 'ttft_ms': 2350.0, 'w4_hot_decode_tps': 29.0 * scale, 'w4_cold_ttft_ms': 2350.0 / scale, 'w4_cold_e2e_ms': 8100.0 / scale, 'share_ge_40_tps': 0.35, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 26.0, 'hot_cached_tokens_min': 11770, 'unfinished_count': 0},
        'W5': {'dev_score': 31.0 * scale, 'output_toks_per_s': 344.0 * scale, 'tpot_ms': 32.2 / scale, 'ttft_ms': 3860.0 / scale, 'share_ge_40_tps': 0.25 if scale > 1 else 0.0, 'share_ge_50_tps': 0.0, 'p95_itl_ms': 34.0, 'cached_tokens_min': 11770, 'unfinished_count': 0},
    }


def _seed_wb2_run(root: Path, *, num_pairs: int = 1, holdout: bool = False):
    reg = dict(WB2_REG, holdout_review_sha256='8' * 64 if holdout else None)
    _write(root / 'registration.json', reg)
    db_path = root / 'run' / 'trajectory.sqlite3'
    order = 0

    def add_worker(db, op, eng, proof, status, data, build=B1, pair=None):
        nonlocal order
        order += 1
        db.event(f'w/{order}/r', {'operation': 'worker_reserved', 'engine': eng, 'worker_operation': op, 'reserved_ms': 1000, **({'measurement_pair': pair} if pair is not None else {})})
        row = {'operation': op, 'status': status, 'engine': eng, 'node_ms': 250, 'evidence': proof, 'build': None if op == 'develop_build' else build, 'data': data}
        obj = db.put('level3_worker_result', row)
        db.event(f'w/{order}/s', {'operation': 'worker_settled', 'evidence': proof, 'result_object': obj})
        prefix = f'{order:05}'
        _write(root / 'worker-operations' / f'{prefix}-reserved.json', {
            'engine': eng, 'operation': op, 'cap_ms': 1000, 'elapsed_ms': order * 1000,
            **({'measurement_pair': pair} if pair is not None else {}),
            'settled_input': 1000, 'settled_output': 200,
            'api_cost_reservation_ledger': {'total_charged_rmb': '0.05'},
        })
        _write(root / 'worker-operations' / f'{prefix}-transport.json', {
            'evidence': proof, 'result_sha256': projection.sha_bytes(canonical(row).encode()),
            'elapsed_ms': order * 1000 + 400,
        })

    with Journal(db_path) as db:
        s0 = db.put('level3_engine', {'files': {'ee_runtime.json': '{"max-running-requests": 24}\n', 'python/sglang/srt/sched.py': 'x = 1\n'}})
        db.event('e/0', {'operation': 'engine_registered', 'engine': E0, 'parent': None, 'source_object': s0})
        s1 = db.put('level3_engine', {'files': {'ee_runtime.json': '{"max-running-requests": 24, "page-size": 16}\n', 'python/sglang/srt/sched.py': 'x = 2\n'}})
        db.event('e/1', {'operation': 'engine_registered', 'engine': E1, 'parent': E0, 'source_object': s1})

        add_worker(db, 'develop_build', E1, B1, 'built', {'build_mode': 'incremental'})
        add_worker(db, 'develop_quality', E1, Q1, 'valid', {
            'case_outcomes': {
                'numeric_32': 'pass', 'natural_eos_4': 'pass', 'autoregressive_full_vocab': 'pass',
                'mixed_tail_absolute': 'pass', 'paired_cancel_numeric': 'pass', 'long_context_logprob_16k_64k': 'pass',
            },
            'trajectory_screen': {'status': 'valid', 'maxima': {'decode_tv': 0.0012}, 'failed': []},
            'sampling_screen': {'status': 'pass', 'max_tv': 0.004},
        })

        for i in range(1, num_pairs + 1):
            ref_id = f'{i:064x}'
            cand_id = f'{100 + i:064x}'
            add_worker(db, 'develop_measure', E0, ref_id, 'observed', {
                'measurement_pair': i, 'scenes': _wb2_scenes(1.0),
                'guardrails': {'G64K': {'ttft_ms': 10000.0, 'decode_tokens_per_s': 37.0, 'unfinished_count': 0}},
            }, build=B0, pair=i)
            if i == 1:
                db.event('ref/0', {'operation': 'development_reference', 'engine': E0, 'evidence': ref_id})
            scale = 1.08 + 0.002 * (i % 3)
            add_worker(db, 'develop_measure', E1, cand_id, 'observed', {
                'measurement_pair': i, 'scenes': _wb2_scenes(scale),
                'guardrails': {'G64K': {'ttft_ms': 9950.0, 'decode_tokens_per_s': 37.2, 'unfinished_count': 0}},
            }, build=B1, pair=i)
            db.event(f'cmp/{i}', {
                'operation': 'development_comparison', 'engine': E1, 'reference': ref_id, 'receipt': cand_id,
                'weighted_score': scale,
                'scene_ratios': {'W1': scale, 'W2': scale, 'W3': scale, 'W4_hot': scale, 'W4_cold': scale, 'W5': scale},
                'guardrail_ratios': {'w4_cold_e2e': 1.02, 'g64k_ttft': 1.005, 'g64k_decode': 1.005, 'p95_itl': 1.01},
                'unfinished_count': 0,
            })

    # Write model receipts for context growth
    mseg = root / 'model-segment-0001'
    _write(mseg / '0001-result.json', {
        'attempt': 1, 'started_utc': '2026-11-28T00:00:01.000Z', 'ended_utc': '2026-11-28T00:00:03.000Z',
        'usage': {'usage': 'settled', 'finish': 'tool_calls', 'wall_ms': 2000,
                  'input_tokens': 1200, 'output_including_reasoning': 180, 'reasoning_tokens': 100,
                  'reported_cache_usage': {'hit': 900, 'miss': 300},
                  'generation_timing': {'first_delta_ms': 350, 'last_delta_ms': 1950}},
    })

    # Write web-research files
    wr = root / 'web-research'
    _write(wr / '000001-request.json', {'operation': 'search', 'payload': {'query': 'sglang swa kv cache'}, 'utc': '2026-11-28T00:00:04.000Z'})
    _write(wr / '000001-response.json', {'results': [{'title': 'SGLang SWA', 'url': 'https://example.com/swa', 'snippet': 'SWA KV tuning'}]})
    _write(wr / '000001-settlement.json', {'operation': 'search', 'status': 200, 'elapsed_ms': 320, 'ended_utc': '2026-11-28T00:00:04.320Z'})
    page_id = '5' * 64
    _write(wr / f'page-{page_id}.json', {'url': 'https://example.com/swa', 'title': 'SGLang SWA', 'content': 'Detailed SWA notes'})
    reuse_id = '6' * 64
    _write(wr / f'reuse-{reuse_id}.json', {'kind': 'technology_transfer', 'paths': ['ee_runtime.json'], 'url': 'https://example.com/swa', 'revision': 'main', 'license': 'Apache-2.0', 'identity': page_id})

    # Write Harness lineage
    rt = {'pi_version': '1.5.2', 'pi_image': '1'*64, 'node_version': '24.13.0', 'package_lock_sha': '2'*64, 'runner_sha': '3'*64, 'base_config_sha': '4'*64}
    lin = Lineage(root / 'run' / 'harness-lineage', rt)
    empty_load = {'prompts': [], 'skills': [], 'extensions': [], 'workflows': [], 'memory': []}
    h0 = lin.initialize(Snapshot.make({}, empty_load, rt))
    h1 = lin.submit('evolving_train', {'prompts/tips.md': b'Check W5 TTFT and throughput together.\n'},
                    {**empty_load, 'prompts': ['prompts/tips.md']})
    lin.activate('evolving_train', h1, h0, actor='agent_rpc', request_id='req-1')

    # Write live TraceRecorder events linked to E1
    rec = TraceRecorder(root / 'trace', started_utc_ns=1_795_824_000_000_000_000)
    try:
        sse = (
            b'data: {"model":"deepseek-flash","choices":[{"index":0,"delta":{"reasoning_content":"Analyze W5 batching."}}]}\n\n'
            b'data: {"model":"deepseek-flash","choices":[{"index":0,"delta":{"content":"Submitting candidate E1."}}]}\n\n'
            b'data: {"model":"deepseek-flash","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tc_sub","type":"function","function":{"name":"ee_rpc","arguments":"{\\"operation\\":\\"candidate_submit\\",\\"arguments\\":{\\"paths\\":[\\"ee_runtime.json\\"]}}"}}]}}]}\n\n'
            b'data: {"model":"deepseek-flash","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1200,"completion_tokens":180,"total_tokens":1380}}\n\n'
            b'data: [DONE]\n\n'
        )
        rec.record_model_exchange(
            segment=1, attempt=1, started_utc='2026-11-28T00:00:01.000Z', ended_utc='2026-11-28T00:00:03.000Z',
            raw_request=canonical({'model': 'deepseek-flash', 'messages': [{'role': 'user', 'content': 'Tune W5'}]}),
            raw_sse=sse,
            usage={'wall_ms': 2000, 'finish': 'tool_calls', 'result': 'complete', 'usage': 'settled',
                   'input_tokens': 1200, 'output_including_reasoning': 180, 'reasoning_tokens': 100},
            rejection=None,
        )
        rec.record_pi_event(segment=1, utc_ns=1_795_824_003_100_000_000, row={
            'type': 'tool_execution_start', 'toolCallId': 'tc_sub', 'toolName': 'ee_rpc',
            'args': {'operation': 'candidate_submit', 'arguments': {'paths': ['ee_runtime.json'], 'parent': E0}},
        })
        rec.record_pi_event(segment=1, utc_ns=1_795_824_003_300_000_000, row={
            'type': 'tool_execution_end', 'toolCallId': 'tc_sub', 'toolName': 'ee_rpc', 'isError': False,
            'result': {'content': [{'type': 'text', 'text': json.dumps({'status': 200, 'result': {'engine': E1, 'parent': E0}})}]},
        })
        rec.flush(3.0)
    finally:
        rec.close()


class _Conn(http.client.HTTPConnection):
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class _JoinServer(api.Server):
    daemon_threads = False


class ObservatoryV2Tests(unittest.TestCase):
    def test_wb2_projection_ci_gating_and_detail_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            r1, r2, r_holdout = base / 'run-one', base / 'run-six', base / 'run-holdout'
            _seed_wb2_run(r1, num_pairs=1)
            _seed_wb2_run(r2, num_pairs=6)
            _seed_wb2_run(r_holdout, num_pairs=1, holdout=True)

            # Single pair: ratio present, NO confidence interval yet!
            s1 = projection.local_run(r1, 'run-one')
            self.assertTrue(s1['has_trace_content'])
            self.assertEqual(s1['scoring']['version'], 'workbuddy-agent-serving-v2')
            pt = s1['points'][0]
            self.assertEqual(set(pt['ratios']), {'W1', 'W2', 'W3', 'W4_hot', 'W4_cold', 'W5'})
            self.assertIn('G64K', pt['guardrails'])
            self.assertEqual(pt['scenes']['W3']['cached_tokens_min'], 11770)
            self.assertEqual(s1['build_summaries'][0]['pairs'], 1)
            self.assertFalse(s1['build_summaries'][0]['ci_available'])
            self.assertEqual(s1['build_summaries'][0]['intervals'], {})

            # Quality details: long_context_logprob_16k_64k present + verdict_only_cases marked
            q_ev = next(e for e in s1['events'] if e.get('kind') == 'worker_settled' and e.get('operation') == 'develop_quality')
            self.assertEqual(q_ev['cases']['long_context_logprob_16k_64k'], 'pass')
            self.assertEqual(q_ev['quality_detail']['trajectory_screen']['maxima']['decode_tv'], 0.0012)
            self.assertIn('long_context_logprob_16k_64k', q_ev['quality_detail']['verdict_only_cases'])

            # Six pairs: CI available and eligible!
            s6 = projection.local_run(r2, 'run-six')
            self.assertTrue(s6['build_summaries'][0]['ci_available'])
            self.assertEqual(s6['build_summaries'][0]['status'], 'eligible')
            self.assertIn('W5', s6['build_summaries'][0]['intervals'])

            # Spin up HTTP server and verify all new endpoints + run-scoped blob isolation + holdout block
            static = base / 'dist'
            static.mkdir()
            (static / 'index.html').write_text('ok', encoding='utf-8')
            store = api.Store({'one': r1, 'six': r2, 'holdout': r_holdout}, archive=None)
            srv = _JoinServer(('127.0.0.1', 0), store, static=static)
            th = threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.02})
            th.start()
            try:
                def get(path):
                    with _Conn('127.0.0.1', srv.server_port, timeout=3) as c:
                        c.request('GET', path)
                        with c.getresponse() as res:
                            return res.status, json.loads(res.read())

                code, tr = get('/api/runs/one/trace?candidates=1')
                self.assertEqual(code, 200)
                self.assertTrue(tr['has_trace_content'])
                self.assertEqual(tr['matched_turns'], 1)
                blob_sha = tr['turns'][0]['response_blob']

                # Blob readable inside its own run
                code, blob = get(f'/api/runs/one/blobs/{blob_sha}')
                self.assertEqual(code, 200)
                self.assertIn('Analyze W5 batching.', blob['content']['reasoning'])

                # Cross-run isolation: a random/unreferenced blob hash returns 404
                code, _ = get(f'/api/runs/one/blobs/{"7"*64}')
                self.assertEqual(code, 404)

                # Holdout run is blocked on ALL detail endpoints
                for p in ('/api/runs/holdout/trace', f'/api/runs/holdout/candidates/{E1}/detail',
                          '/api/runs/holdout/waterfall', '/api/runs/holdout/research',
                          '/api/runs/holdout/harness', '/api/runs/holdout/context'):
                    self.assertEqual(get(p)[0], 403)

                # Candidate detail has code diff, ee_runtime.json diff, quality, and ID-linked turn
                code, cand = get(f'/api/runs/one/candidates/{E1}/detail')
                self.assertEqual(code, 200)
                self.assertEqual(cand['engine'], E1)
                self.assertEqual(len(cand['diff_vs_parent']['files']), 2)
                self.assertEqual(cand['diff_vs_parent']['runtime_diff'][0]['key'], 'page-size')
                self.assertEqual(cand['conversation'][0]['correlation'], 'submitted_candidate')

                # Waterfall, research, harness, context
                self.assertEqual(get('/api/runs/one/waterfall')[0], 200)
                code, res = get('/api/runs/one/research')
                self.assertEqual(code, 200)
                self.assertEqual(res['searches'][0]['query'], 'sglang swa kv cache')
                self.assertIn(E1, res['reuses'][0]['matched_engines'])
                self.assertEqual(get(f'/api/runs/one/research/pages/{"5"*64}')[0], 200)

                code, har = get('/api/runs/one/harness')
                self.assertEqual(code, 200)
                self.assertEqual(len(har['versions']), 2)
                self.assertEqual(len(har['diff']), 1)

                code, ctx = get('/api/runs/one/context')
                self.assertEqual(code, 200)
                self.assertEqual(ctx['points'][0]['cache_hit_tokens'], 900)
            finally:
                srv.shutdown()
                srv.server_close()
                th.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
