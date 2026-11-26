from pathlib import Path
from math import fsum, log
from random import Random
from unittest import TestCase
from unittest.mock import patch

from experiments.spark26_gguf.level3 import late_trajectory as late
from evolveengine.v3_level3_collect import source_bytes


class LateTrajectoryAdapterTests(TestCase):
    def test_frozen_adapter_targets_only_survivor_and_independent_teacher(self):
        source=source_bytes(Path(late.__file__).with_name('observe_trajectory.py'))
        wired=late.instrument(late.cold.instrument(source)); compile(wired,'late observer','exec')
        self.assertIn(b'if survivor: payload.update',wired)
        self.assertIn(b'top_logprobs_num=TAIL_TOP_K',wired)
        self.assertIn(b'for step in q.TAIL:',wired)
        self.assertIn(b"'all' if tail else job['wanted']",wired)
        self.assertIn(b'cache_parity=cold.witness(q.TRIGGERS)',wired)
        self.assertEqual(wired.count(b'top_logprobs_num=TAIL_TOP_K'),1)
        with self.assertRaisesRegex(ValueError,'cannot bind'):
            late.instrument(wired)

    def test_duplicate_or_disagreeing_projection_is_not_evidence(self):
        with patch.object(late,'TOP_K',2), patch.object(late.q,'TAIL',(0,)), patch.object(late.q,'MARKERS',(1,)):
            survivor=dict(ids=[3],probs=[[-4.,-.01]],tail_top={'0':[[-.01,3],[-5.,2]]})
            view=late.view(survivor,0)
            self.assertEqual(set(view),{1,2,3})
            for invalid in ([[-.01,3],[-5.,3]],[[-.02,3],[-5.,2]],[[-.01,3]]):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    late.view(dict(survivor,tail_top={'0':invalid}),0)

    def test_paired_bounds_cover_different_e0_prefix_distributions(self):
        rng=Random(0x20261126); vocab=16
        with patch.object(late.q,'VOCAB',vocab):
            for _ in range(64):
                probs=[]
                for j in range(4):
                    weights=[rng.random()+.001 for _ in range(vocab)]
                    probs.append([x/fsum(weights) for x in weights])
                a,b,ra,rb=probs
                a_ids={0,*rng.sample(range(1,vocab),rng.randrange(vocab))}
                b_ids={0,*rng.sample(range(1,vocab),rng.randrange(vocab))}
                r=late.adjusted_norm({i:log(a[i]) for i in a_ids},{i:log(b[i]) for i in b_ids},
                                     list(map(log,ra)),list(map(log,rb)))
                actual=fsum(abs((y-x)-(v-u)) for x,y,u,v in zip(a,b,ra,rb,strict=True))/2
                self.assertLessEqual(r['lower']-1e-12,actual)
                self.assertGreaterEqual(r['upper']+1e-12,actual)
