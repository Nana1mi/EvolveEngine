from math import fsum, log
from random import Random
from unittest import TestCase

from experiments.spark26_gguf.level3.partial_tv import certify, matched_tv


def logs(rows): return {k:log(v) for k,v in rows.items()}


class PartialProbabilityBoundsTests(TestCase):
    def test_top_shapes_can_differ_but_are_neither_zero_filled_nor_waived(self):
        a=logs({0:.97,1:.01}); b=logs({0:.96,2:.01})
        r=matched_tv(a,b,5,normalization_error=1e-12)
        self.assertAlmostEqual(r['lower'],.01)
        self.assertAlmostEqual(r['upper'],.04)
        self.assertEqual(certify(r,.05),'certified')
        self.assertEqual(certify(r,.03),'inconclusive')
        self.assertEqual(certify(r,.005),'different')

    def test_missing_mass_controls_degree_of_confidence(self):
        r=matched_tv(logs({0:.15}),logs({0:.15}),10)
        self.assertAlmostEqual(r['upper'],.85)
        self.assertEqual(certify(r,.05),'inconclusive')
        r=matched_tv(logs({0:.72,1:.2}),logs({0:.73,1:.23}),3)
        self.assertAlmostEqual(r['lower'],.04)
        self.assertAlmostEqual(r['upper'],.04)
        self.assertFalse(r['complete'])

    def test_complete_and_malformed_views(self):
        r=matched_tv(logs({0:.75,1:.25}),logs({0:.70,1:.30}),2,normalization_error=1e-12)
        self.assertTrue(r['complete'])
        self.assertAlmostEqual(r['lower'],.05); self.assertEqual(r['lower'],r['upper'])
        for a,b in ((logs({0:.8,1:.4}),logs({0:.7})),(logs({0:.6}),logs({1:.7})),
                    ({0:float('nan')},logs({0:.7})),(logs({False:.6}),logs({0:.7})),
                    (logs({0:.5,1:.35}),logs({0:.7,1:.3}))):
            with self.subTest(a=a,b=b), self.assertRaises(ValueError):
                matched_tv(a,b,2,normalization_error=1e-12)

    def test_intervals_cover_dense_tv_with_different_partial_views(self):
        rng=Random(0xE0)
        for _ in range(100):
            vocab=16
            a,b=([rng.random()+.001 for _ in range(vocab)] for _ in ('a','b'))
            a=[x/fsum(a) for x in a]; b=[x/fsum(b) for x in b]
            a_ids={0,*rng.sample(range(1,vocab),rng.randrange(vocab))}
            b_ids={0,*rng.sample(range(1,vocab),rng.randrange(vocab))}
            r=matched_tv(logs({i:a[i] for i in a_ids}),logs({i:b[i] for i in b_ids}),vocab,
                         normalization_error=1e-12)
            tv=fsum(abs(x-y) for x,y in zip(a,b,strict=True))/2
            self.assertLessEqual(r['lower']-1e-12,tv)
            self.assertGreaterEqual(r['upper']+1e-12,tv)
