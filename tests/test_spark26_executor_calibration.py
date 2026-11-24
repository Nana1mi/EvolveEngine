import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments/spark26_gguf'), *sys.path]):
    import executor_calibration as calibration

from evolveengine.deepseek import Provider
from evolveengine.pi import PiAgent
from evolveengine.schema import Cost


def calibration_stream(req, call=None):
    delta = {'role':'assistant', 'content':'{"applied":true}'}
    finish = 'stop'
    if call:
        name, args = call
        delta = {'role':'assistant','tool_calls':[{'index':0,'id':'id-'+name,'type':'function',
                 'function':{'name':name,'arguments':json.dumps(args)}}]}
        finish = 'tool_calls'
    base = {'id':'fixture','object':'chat.completion.chunk','model':req['model'],'created':1}
    events = [dict(base,choices=[dict(index=0,delta=delta,finish_reason=None)]),
              dict(base,choices=[dict(index=0,delta={},finish_reason=finish)]),
              dict(base,choices=[],usage=dict(prompt_tokens=100,completion_tokens=30,total_tokens=130))]
    return io.BytesIO(b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events) + b'data: [DONE]\n\n')


class CalibrationTests(unittest.TestCase):
    def provider(self):
        return Provider('http://127.0.0.1:1','mock-only','EE_MOCK_KEY','mock-only',2500)

    def test_independent_caps_and_maximum_turns(self):
        b = calibration.CalibrationBudget(self.provider(), calibration.CAP)
        b.spent = Cost(input_tokens=39_980, output_tokens=7997)
        self.assertEqual(b.turn_output_ceiling(20),3)
        with self.assertRaisesRegex(ValueError,'input reservation'):
            b.turn_output_ceiling(21)
        b.rows = [{}]*calibration.MAX_TURNS
        with self.assertRaisesRegex(ValueError,'turn cap'):
            b.turn_output_ceiling(1)

    @unittest.skipUnless(os.getenv('EE_GGUF_SOURCE') and os.getenv('EE_PI_CLI'), 'optional fixed upstream + pinned Pi CLI')
    def test_pinned_pi_nonblind_mock_and_blocked_outside_read(self):
        source = Path(os.environ['EE_GGUF_SOURCE'])
        cli = os.environ['EE_PI_CLI']
        reference = calibration.qkv.edited(source.read_bytes())
        want = calibration.sha256(reference).hexdigest()
        self.assertEqual(want,'a5c3e5157bb8cfba8a5db700c21ac7ce226146648c676f3d30680e31d57c4b0b')
        for blocked in (False,True):
            upstream_calls = []
            def model(url, *, payload=None, timeout=15):
                if url == '/models':
                    return io.BytesIO(b'{"data":[{"id":"mock-only","name":"mock-only"}]}')
                req = json.loads(payload)
                self.assertEqual({r['function']['name'] for r in req['tools']},{'read','edit'})
                turn = len(upstream_calls)
                upstream_calls.append(url)
                call = None
                if turn == 0:
                    call = ('read',{'path': '../unrelated.py' if blocked else calibration.SRC,'offset':500,'limit':32})
                if turn == 1 and not blocked:
                    call = ('edit', {'path':calibration.SRC,'edits':[{'oldText':calibration.qkv.ANCHOR,
                              'newText':calibration.qkv.GUARD.replace('in ROWS',f'in {calibration.qkv.PROFILED_ROWS!r}')} ]})
                return calibration_stream(req,call)
            agent = PiAgent(['node',cli], calibration.VERSION,self.provider())
            with tempfile.TemporaryDirectory() as folder, patch.object(Provider, 'request', side_effect=model):
                path = Path(folder)/'receipt'
                out = calibration.run_calibration(agent,source,{'candidate_sha256':want},path)
                events = [json.loads(line) for line in (path/'tool-receipts.jsonl').read_text().splitlines()]
                self.assertEqual(out['status'], 'infra_or_edit_failure' if blocked else 'pass',out)
                self.assertEqual(out['provider']['usage']['input_tokens'],100*len(upstream_calls))
                self.assertEqual(events[0]['accepted'],not blocked)
                if not blocked:
                    self.assertEqual(out['result_sha256'],want)
                    self.assertEqual([r['tool'] for r in events],['read','edit'])
                    self.assertEqual(len(upstream_calls),3)
