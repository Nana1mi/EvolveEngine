"""Offline verification for Phase 1 async content-addressed Level 3 trace capture."""
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_guest import _run_pi
from evolveengine.v3_level3_trace import TraceRecorder, parse_sse_response, redact_obj


def _sse_bytes(reasoning: str, content: str, tool_calls=None, finish='stop', usage=None) -> bytes:
    chunks = []
    if reasoning:
        chunks.append({'model': 'deepseek-flash', 'choices': [{'index': 0, 'delta': {'reasoning_content': reasoning}}]})
    if content:
        chunks.append({'model': 'deepseek-flash', 'choices': [{'index': 0, 'delta': {'content': content}}]})
    if tool_calls:
        for idx, tc in enumerate(tool_calls):
            chunks.append({'model': 'deepseek-flash', 'choices': [{'index': 0, 'delta': {'tool_calls': [{
                'index': idx, 'id': tc['id'], 'type': 'function',
                'function': {'name': tc['name'], 'arguments': json.dumps(tc['arguments'])},
            }]}}]})
    chunks.append({
        'model': 'deepseek-flash',
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
        'usage': usage or {'prompt_tokens': 120, 'completion_tokens': 40, 'total_tokens': 160,
                           'completion_tokens_details': {'reasoning_tokens': 25}},
    })
    return b''.join(b'data: ' + canonical(c) + b'\n\n' for c in chunks) + b'data: [DONE]\n\n'


class TraceCaptureTests(unittest.TestCase):
    def test_redaction_and_deduplication_and_truncation(self):
        secret_key = 'sk-testSecretKey1234567890abcdef'
        bearer_tok = 'Bearer supersecrettoken9876543210'
        env_secret = 'env_deepseek_secret_9988776655'
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'trace'
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': env_secret}):
                rec = TraceRecorder(root, started_utc_ns=1_000_000_000_000, extra_secrets=('gw_token_1234567890',))
            try:
                m_sys = {'role': 'system', 'content': f'sys prompt with {secret_key} and {env_secret}'}
                m_usr = {'role': 'user', 'content': f'header {bearer_tok} gw_token_1234567890'}
                m_ast = {'role': 'assistant', 'content': 'checking code', 'reasoning_content': 'let us inspect'}
                m_usr2 = {'role': 'user', 'content': 'next step'}

                req1 = canonical({'model': 'deepseek-flash', 'messages': [m_sys, m_usr]})
                req2 = canonical({'model': 'deepseek-flash', 'messages': [m_sys, m_usr, m_ast, m_usr2]})
                sse1 = _sse_bytes('thinking about W5', 'I will edit kernel', finish='length')
                sse2 = _sse_bytes('continuing after truncation', 'Submitting candidate', finish='stop')

                rec.record_model_exchange(
                    segment=1, attempt=1, started_utc='2026-11-28T00:00:01Z', ended_utc='2026-11-28T00:00:02Z',
                    raw_request=req1, raw_sse=sse1,
                    usage={'wall_ms': 1000, 'finish': 'length', 'result': 'generation_truncated', 'usage': 'settled',
                           'input_tokens': 120, 'output_including_reasoning': 40, 'reasoning_tokens': 25},
                    rejection=None,
                )
                rec.record_model_exchange(
                    segment=1, attempt=2, started_utc='2026-11-28T00:00:03Z', ended_utc='2026-11-28T00:00:04Z',
                    raw_request=req2, raw_sse=sse2,
                    usage={'wall_ms': 900, 'finish': 'stop', 'result': 'complete', 'usage': 'settled',
                           'input_tokens': 240, 'output_including_reasoning': 30, 'reasoning_tokens': 15},
                    rejection=None,
                )
                self.assertTrue(rec.flush(3.0))

                seg_path = root / 'segment-0001.jsonl'
                rows = [json.loads(line) for line in seg_path.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(rows), 2)
                # Shared prefix messages deduplicated by content hash
                self.assertEqual(rows[0]['message_blobs'][0]['blob'], rows[1]['message_blobs'][0]['blob'])
                self.assertEqual(rows[0]['message_blobs'][1]['blob'], rows[1]['message_blobs'][1]['blob'])
                # Total unique blobs = 4 unique messages + 2 unique responses = 6 (not 8)
                blobs = list((root / 'blobs').glob('*.json'))
                self.assertEqual(len(blobs), 6)
                # Truncated length turn preserved full reasoning & content
                resp1 = json.loads((root / 'blobs' / f"{rows[0]['response_blob']}.json").read_text(encoding='utf-8'))
                self.assertEqual(resp1['reasoning'], 'thinking about W5')
                self.assertEqual(resp1['finish'], 'length')
                # No secrets leaked anywhere on disk
                all_text = seg_path.read_text(encoding='utf-8') + ''.join(b.read_text(encoding='utf-8') for b in blobs)
                for s in (secret_key, 'supersecrettoken9876543210', env_secret, 'gw_token_1234567890'):
                    self.assertNotIn(s, all_text)
            finally:
                rec.close()

    def test_live_replay_interrupt_resume_and_id_correlation(self):
        """End-to-end offline replay: model_request -> pi_tool -> candidate_submit -> Worker -> point."""
        e0 = '0' * 64
        e1 = '1' * 64
        b1 = 'b' * 64
        ev_measure = 'c' * 64
        call_edit = 'call_edit_001'
        call_submit = 'call_submit_002'

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'trace'
            rec = TraceRecorder(root, started_utc_ns=2_000_000_000_000)
            try:
                # 1. Model request emits edit + candidate_submit tool calls
                sse = _sse_bytes(
                    'Need to tune chunked prefill and max running requests for W5.',
                    'Editing ee_runtime.json and submitting candidate.',
                    tool_calls=[
                        {'id': call_edit, 'name': 'edit', 'arguments': {'path': 'ee_runtime.json', 'oldText': '{}', 'newText': '{"page-size": 16}'}},
                        {'id': call_submit, 'name': 'ee_rpc', 'arguments': {'operation': 'candidate_submit', 'arguments': {'paths': ['ee_runtime.json'], 'parent': e0}}},
                    ],
                    finish='tool_calls',
                )
                req = canonical({'model': 'deepseek-flash', 'messages': [{'role': 'user', 'content': 'Optimize W5'}]})
                rec.record_model_exchange(
                    segment=1, attempt=1, started_utc='2026-11-28T00:01:00Z', ended_utc='2026-11-28T00:01:02Z',
                    raw_request=req, raw_sse=sse,
                    usage={'wall_ms': 2000, 'finish': 'tool_calls', 'result': 'complete', 'usage': 'settled',
                           'input_tokens': 500, 'output_including_reasoning': 120, 'reasoning_tokens': 80,
                           'reported_cache_usage': {'hit': 400, 'miss': 100},
                           'generation_timing': {'first_delta_ms': 300, 'last_delta_ms': 1950, 'output_span_ms': 1650}},
                    rejection=None,
                )
                self.assertTrue(rec.flush(3.0))

                # Verify live readable BEFORE Pi finishes (simulate consumer reading mid-run)
                seg_file = root / 'segment-0001.jsonl'
                part1_bytes = seg_file.read_bytes()
                part1 = [json.loads(line) for line in part1_bytes.splitlines()]
                self.assertEqual(len(part1), 1)
                self.assertEqual(part1[0]['tool_call_ids'], [call_edit, call_submit])

                # 2. Pi executes edit + ee_rpc(candidate_submit) via _run_pi
                pi_lines = [
                    {'type': 'tool_execution_start', 'toolCallId': call_edit, 'toolName': 'edit',
                     'args': {'path': 'ee_runtime.json', 'oldText': '{}', 'newText': '{"page-size": 16}'}},
                    {'type': 'tool_execution_end', 'toolCallId': call_edit, 'toolName': 'edit',
                     'isError': False, 'result': {'content': [{'type': 'text', 'text': 'Edited ee_runtime.json'}]}},
                    {'type': 'tool_execution_start', 'toolCallId': call_submit, 'toolName': 'ee_rpc',
                     'args': {'operation': 'candidate_submit', 'arguments': {'paths': ['ee_runtime.json'], 'parent': e0}}},
                    {'type': 'tool_execution_end', 'toolCallId': call_submit, 'toolName': 'ee_rpc',
                     'isError': False, 'result': {'content': [{'type': 'text', 'text': json.dumps({'status': 200, 'result': {'engine': e1, 'parent': e0}})}]}},
                ]
                proc = Mock()
                proc.poll.side_effect = [0, 0]
                proc.stdout = io.BytesIO(b''.join(canonical(line) + b'\n' for line in pi_lines))
                proc.returncode = 0
                relay = SimpleNamespace(error=None, thread=SimpleNamespace(is_alive=lambda: True),
                                        proc=SimpleNamespace(poll=lambda: None),
                                        gateway=SimpleNamespace(stalled_tool=lambda _: False))
                report = {}
                with patch('evolveengine.v3_level3_guest.subprocess.Popen', return_value=proc):
                    _run_pi('fixture-run', 10, report, relay, trace_recorder=rec, segment=1)

                # 3. Worker events recorded live
                rec.record_worker_event(segment=1, phase='reserved', order=1, operation='develop_measure', engine=e1, pair_index=1)
                rec.record_worker_event(segment=1, phase='settled', order=1, operation='develop_measure', engine=e1,
                                        status='observed', evidence=ev_measure, build=b1, node_ms=4200, pair_index=1)
                self.assertTrue(rec.flush(3.0))

                # Simulate disconnect/resume: read only bytes appended after part1_bytes offset!
                with seg_file.open('rb') as handle:
                    handle.seek(len(part1_bytes))
                    tail_rows = [json.loads(line) for line in handle.read().splitlines()]
                self.assertEqual(len(tail_rows), 6)  # 4 pi events + 2 worker events
                all_rows = part1 + tail_rows
                self.assertEqual([r['seq'] for r in all_rows], list(range(1, 8)))

                # Verify explicit ID correlation chain: model -> call_submit -> candidate e1 -> worker receipt
                submit_end = next(r for r in all_rows if r['kind'] == 'pi_tool_end' and r['tool_call_id'] == call_submit)
                self.assertEqual(submit_end['rpc_operation'], 'candidate_submit')
                self.assertEqual(submit_end['engine'], e1)
                self.assertEqual(submit_end['parent'], e0)
                self.assertIn(submit_end['call_sha256'], part1[0]['call_sha256s'])

                worker_end = next(r for r in all_rows if r['kind'] == 'worker_settled')
                self.assertEqual(worker_end['engine'], e1)
                self.assertEqual(worker_end['evidence'], ev_measure)
                self.assertEqual(worker_end['build'], b1)
            finally:
                rec.close()


if __name__ == '__main__':
    unittest.main()
