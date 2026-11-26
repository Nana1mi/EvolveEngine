from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evolveengine.v3_harness import canonical, digest
from evolveengine.v3_level3_chat_timing import ChatSchedule, SCHEMA, load_chat
from evolveengine.v3_level3_scoring import paired_w3_ttft


class ChatTimingTests(unittest.TestCase):
    def test_chat_replay_changes_warmup_and_preserves_all_w3_timed_requests(self):
        waves = dict(W1=[[[1]],[[2]],[[3]],[[4]],[[5]]],W3=[[[6]],[[7]],[[8]],[[9]],[[10]]])
        schedule = ChatSchedule(waves,1,2,'a'*64)
        first,_ = schedule.get(1)
        replay,_ = schedule.get(1,replay=True)
        for scene in waves:
            self.assertNotEqual(first[scene][:1],replay[scene][:1])
            self.assertEqual(first[scene][1:],list(reversed(replay[scene][1:])))
        self.assertEqual(set(x[0][0] for x in first['W3'][1:]),{8,9,10})
        self.assertEqual(len(first['W1']),3)

    def test_w3_observation_uses_matched_raw_ttft(self):
        def record(n,raw,visible):
            return dict(input_tokens=n,raw_ttft_ms=raw,visible_answer_ttft_ms=visible)
        baseline = {'x':record(3990,12,100),'y':record(4202,18,None)}
        candidate = {'x':record(3990,6,None),'y':record(4202,18,150)}
        ratio, report = paired_w3_ttft(baseline,candidate)
        self.assertAlmostEqual(ratio,2**.5)
        self.assertIsNone(report['x']['candidate_visible_answer_ttft_ms'])
        with self.assertRaises(ValueError):
            paired_w3_ttft(baseline,{'x':candidate['x']})

    def test_frozen_chat_range_is_W3_only(self):
        ids = [[7]*3990,[8]*4095,[9]*4100,[10]*4202]
        def raw():
            data = dict(schema=SCHEMA,split='development',enable_thinking=True,tokenizer_sha256='1'*64,
                        template_sha256='2'*64,screening={'engine':'3'*64,'receipt_sha256':'4'*64},
                        requests={'W3':[[dict(ids=seq, source='dataset@example',source_id=str(i),
                                             task='question',license='CC-BY-4.0')]
                                             for i,seq in enumerate(ids)]})
            return canonical(data)+b'\n'
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'requests.json'
            p.write_bytes(raw())
            ref=dict(path=str(p),sha256=digest(raw()))
            self.assertEqual(load_chat(ref,'1'*64,'2'*64,dict(W3=((4096,128),)),1,2)['W3'][0][0],ids[0])
            ids[-1].append(10)
            p.write_bytes(raw())
            with self.assertRaisesRegex(ValueError,'input length'):
                load_chat(dict(ref,sha256=digest(raw())),'1'*64,'2'*64,dict(W3=((4096,128),)),1,2)


if __name__ == '__main__':
    unittest.main()
