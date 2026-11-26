from math import log
from unittest import TestCase
from unittest.mock import patch

from experiments.spark26_gguf.level3 import same_prefix_decode_audit as audit


class SamePrefixDecodeAuditTests(TestCase):
    def test_prefill_is_excluded_and_divergence_does_not_compare_different_prefixes(self):
        with patch.multiple(audit.q,VOCAB=16,STEPS=4,CASES=(0,),prefix=lambda _: [2,0]):
            def model(rid, ids, frames):
                return dict(role='candidate',protocol=audit.q.VERSION,decode=[dict(case=0,rid=rid*32,
                    prompt_sha=audit.q.ih([2,0]),input=2,cap=4,finish='length',ids=ids,
                    vectors=[audit.q.pack([(log(p),i) for i,p in enumerate([x*(1-13e-8) for x in v]+[1e-8]*13)]) for v in frames])])
            a=model('a',[2,0,0,1],[(.9,.05,.05),(.6,.1,.3),(.2,.5,.3),(.8,.1,.1)])
            b=model('b',[2,0,2,1],[(.05,.05,.9),(.6,.1,.3),(.3,.4,.3),(.1,.8,.1)])
            report=audit.compare_decode(a,b)
            self.assertEqual([row['step'] for row in report['coordinates']],[1,2])
            self.assertAlmostEqual(report['maxima']['tv'],.1,places=6)
            self.assertEqual(audit.hints(report,{'tv':.09,'eos_log_delta':1}),
                             dict(status='review_suggested',flags=['tv'],scoring=False))
            b['decode'][0]['ids'][0]=0
            out=audit.compare_decode(a,b)
            self.assertIsNone(out['maxima'])
            self.assertEqual(audit.hints(out,{'tv':.09,'eos_log_delta':1})['status'],'insufficient_aligned_prefix')
