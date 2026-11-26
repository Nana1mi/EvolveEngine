from email.message import Message
from io import BytesIO
import json
from threading import Barrier
from unittest import TestCase

from experiments.spark26_gguf.level3.observe_text_timing import fingerprint, merge_stream_output, request


class MockSSEClient:
    def __init__(self, frames, tail=b''):
        self.frames, self.tail = frames, tail

    def open(self, message, timeout):
        body = json.loads(message.data)
        lines = []
        for count, output, reason in self.frames:
            meta = dict(id=body['rid'], prompt_tokens=len(body['input_ids']), completion_tokens=count,
                        finish_reason=(reason if isinstance(reason,dict) else {'type': reason}) if reason else None)
            lines.append(b'data: ' + json.dumps(dict(output_ids=output, meta_info=meta)).encode() + b'\n\n')
        reply = BytesIO(b''.join(lines) + b'data: [DONE]\n\n' + self.tail)
        reply.status = 200
        reply.headers = Message()
        reply.headers['content-type'] = 'text/event-stream'
        return reply


class ObserveTextTimingStreamTests(TestCase):
    def test_reconstructs_cumulative_frames(self):
        output = []
        output = merge_stream_output(output, [11], 1)
        output = merge_stream_output(output, [11, 12], 2)
        output = merge_stream_output(output, [11, 12, 13], 3, final=True)
        self.assertEqual(output, [11, 12, 13])

    def test_reconstructs_incremental_frames(self):
        output = []
        output = merge_stream_output(output, [11], 1)
        output = merge_stream_output(output, [12], 2)
        output = merge_stream_output(output, [13], 3, final=True)
        self.assertEqual(output, [11, 12, 13])

    def test_accepts_empty_or_repeated_final_delta_without_double_count(self):
        output = merge_stream_output([], [11], 1)
        output = merge_stream_output(output, [], 1, final=True)
        self.assertEqual(merge_stream_output(output, [11], 1, final=True), [11])

    def test_rejects_gaps_and_changed_cumulative_prefixes(self):
        with self.assertRaisesRegex(ValueError, 'do not match'):
            merge_stream_output([11], [13], 3)
        with self.assertRaisesRegex(ValueError, 'changed an earlier token'):
            merge_stream_output([11, 12], [11, 99, 13], 3)
        with self.assertRaisesRegex(ValueError, 'without advancing'):
            merge_stream_output([11, 12], [13], 2)

    def test_rejects_invalid_token_ids_and_nonmonotonic_counts(self):
        with self.assertRaisesRegex(ValueError, 'invalid streamed'):
            merge_stream_output([], [True], 1)
        with self.assertRaisesRegex(ValueError, 'invalid streamed'):
            merge_stream_output([11, 12], [13], 1)

    def test_request_hashes_complete_sequence_when_terminal_frame_has_one_id(self):
        for middle, last in (([101, 102], [101, 102, 103]), ([102], [103])):
            with self.subTest(middle=middle, last=last):
                frames = [(1, [101], None), (2, middle, None), (3, last, 'length')]
                result = request([16, 17], 3, Barrier(1), client=MockSSEClient(frames),
                                 clock=iter(range(1, 100)).__next__)
                self.assertEqual(result['sha256'], fingerprint([101, 102, 103]))
                self.assertEqual((result['first_ns'], result['last_ns'], result['done_ns']), (2, 4, 5))

    def test_request_rejects_terminal_replacement_of_earlier_tokens(self):
        frames = [(1, [101], None), (2, [102], None), (3, [101, 999, 103], 'length')]
        with self.assertRaisesRegex(ValueError, 'changed an earlier token'):
            request([16, 17], 3, Barrier(1), client=MockSSEClient(frames),
                    clock=iter(range(1, 100)).__next__)

    def test_w3_first_visible_token_follows_reasoning_and_whitespace(self):
        class Codec:
            @staticmethod
            def decode(ids, *, skip_special_tokens):
                return ''.join({220:' ',104:'Answer'}[i] for i in ids)
        frames = [(1,[101],None),(2,[4],None),(3,[220],None),(4,[104],'length')]
        result = request([16,17],4,Barrier(1),client=MockSSEClient(frames),natural_eos=True,
                         clock=iter(range(1,100)).__next__,answer_codec=Codec())
        self.assertEqual(result['first_ns'],2)
        self.assertEqual(result['visible_answer_ns'],5)

    def test_quality_eos_allows_coalesced_first_frame_not_timing_and_rejects_trailing_data(self):
        stop = dict(type='stop',matched=1)
        frames = [(2,[42,1],stop)]
        def run(frames, tail=b'', quality=True):
            return request([16],48,Barrier(1), client=MockSSEClient(frames,tail), natural_eos=True,
                           quality_only=quality,retain_ids=True,clock=iter(range(1,100)).__next__)
        result = run(frames)
        self.assertEqual(result['observed_ids'], [42,1])
        self.assertFalse(result['first_token_unambiguous'])
        for capture, tail in [(frames,b'data: [DONE]\n'),(frames+[(3,[90],None)],b''),
                              ([(3,[42,1,1],stop)],b'')]:
            with self.assertRaises(ValueError): run(capture,tail)
        with self.assertRaisesRegex(ValueError,'ambiguous'): run(frames,quality=False)

    def test_quality_preview_requires_same_stream_confirmation_never_used_for_ttft(self):
        stop = dict(type='stop', matched=1)
        frames = [(1, [39119,70], None), (2, [39119,70], None), (3,[39119,70,1],stop)]
        def run(frames, *, timed=False):
            return request([16], 4, Barrier(1),client=MockSSEClient(frames),natural_eos=True,retain_ids=True,
                           quality_only=not timed,clock=iter(range(1,100)).__next__)
        good=run(frames)
        self.assertEqual(good['observed_ids'],[39119,70,1]);self.assertFalse(good['first_token_unambiguous'])
        self.assertEqual(run([frames[0],(2,[70],None),(3,[1],stop)])['sha256'], good['sha256'])
        with self.assertRaises(ValueError): run(frames,timed=True)
        for bad in ([frames[0],(2,[71],None),frames[-1]], [frames[0],(2,[39119,1],stop)],
                    [(1,[39119,70,1],None),(2,[39119,70],stop)], [(1,[39119,70],stop)],
                    [(1,[39119,70,70,70,1],None)], [(1,[39119,True],None)],
                    [frames[0],(2,[39119,71,1],None),frames[-1]], [frames[0],(3,[1],stop)]):
            with self.subTest(frames=bad), self.assertRaises(ValueError): run(bad)


class LegacyGraphDeltaAdapterTests(TestCase):
    def test_graph_only_adapter_recovers_strict_delta_ids_without_changing_legacy_fixture(self):
        from pathlib import Path
        import sys
        from unittest.mock import patch
        from experiments.spark26_gguf.level3 import observe_text_timing, replay_e0
        source = Path(replay_e0.node.DEV/'numeric.py')
        raw = source.read_bytes().replace(b'\r\n',b'\n')
        good = [(1,[17],None),(2,[18],None),(3,[19],'length')]
        bad = [(1,[17],None),(3,[19],'length')]
        with patch.dict(sys.modules,{'observe_text_timing':observe_text_timing}):
            code = {'__name__':'graph_delta_fixture'}
            exec(compile(replay_e0.graph_delta_numeric(raw),str(source),'exec'),code)
        for frames in (good,bad):
            def call(route,payload):
                from types import SimpleNamespace
                self.assertEqual(route,'generate')
                return MockSSEClient(frames).open(SimpleNamespace(data=json.dumps(payload)),3)
            code['call'] = call
            if frames == good:
                self.assertEqual(code['stream']('fixture',[16,17],3)['ids'],[17,18,19])
            else:
                with self.assertRaisesRegex(ValueError,'do not match'):
                    code['stream']('fixture',[16,17],3)
        self.assertEqual(source.read_bytes().replace(b'\r\n',b'\n'),raw)
        with self.assertRaisesRegex(ValueError,'source drift'):
            replay_e0.graph_delta_numeric(raw.replace(b"final=msg['output_ids']",b"final=[]"))
