import email.message
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


script = Path(__file__).resolve().parents[1] / 'experiments' / 'spark26_gguf' / 'e0_probe.py'
spec = importlib.util.spec_from_file_location('spark26_e0_probe', script)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SyntheticSSE(io.BytesIO):
    status = 200

    def __init__(self, *, done=True, usage=True):
        self.headers = email.message.Message()
        self.headers['Content-Type'] = 'text/event-stream; charset=utf-8'
        data = [dict(choices=[dict(delta=dict(content='就'), finish_reason=None)]),
                dict(choices=[dict(delta=dict(content='绪'), finish_reason=None)]),
                dict(choices=[dict(delta={}, finish_reason='stop')])]
        if usage:
            data.append(dict(choices=[], usage=dict(prompt_tokens=19, completion_tokens=2)))
        lines = [b'data: ' + json.dumps(event, ensure_ascii=False).encode('utf-8') + b'\n\n' for event in data]
        if done:
            lines.append(b'data: [DONE]\n\n')
        super().__init__(b''.join(lines))


class E0ProbeTests(unittest.TestCase):
    def test_usage_is_requested_and_parsed_from_empty_choices_event(self):
        with patch.object(probe, 'build_opener') as make:
            opener = make.return_value
            opener.open.return_value = SyntheticSSE()
            result = probe.call('测试')
            request = opener.open.call_args.args[0]
        self.assertEqual(json.loads(request.data)['stream_options'], {'include_usage': True})
        self.assertEqual((result['text'], result['finish'], result['done']), ('就绪', 'stop', True))
        self.assertEqual((result['completion_tokens'], result['content_chunks']), (2, 2))
        self.assertGreaterEqual(result['e2e_ms'], result['ttft_ms'])

    def test_stream_without_done_or_usage_is_not_admitted(self):
        for options, message in [({'done': False}, 'Incomplete SSE'), ({'usage': False}, 'invalid token usage')]:
            with self.subTest(options=options), patch.object(probe, 'build_opener') as make:
                make.return_value.open.return_value = SyntheticSSE(**options)
                with self.assertRaisesRegex(ValueError, message):
                    probe.call('测试')

    def test_throughput_uses_wall_time_and_nearest_rank_p95(self):
        rows = [dict(prompt_tokens=20+i, completion_tokens=4, ttft_ms=i, tpot_ms=.5*i)
                for i in range(1, 9)]
        block = probe.summarize_block(rows, 2.0)
        self.assertEqual((block['output_tok_s'], block['requests_s']), (16., 4.))
        self.assertEqual((block['ttft_p50_ms'], block['ttft_p95_ms'], block['tpot_p95_ms']), (4.5, 8, 4.))


if __name__ == '__main__':
    unittest.main()
