import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
import sys
import unittest
import numpy as np

EX=Path(__file__).resolve().parents[1]/'experiments'
sys.path[:0]=[str(EX),str(EX/'spark26_gguf')]
from g1_d2 import audit as g
from p2_cpu_wire import check as wire

class NumericTests(unittest.TestCase):
    def test_near_tie_flip_is_measured_not_rejected(self):
        a=np.log([.5001,.4999]);b=a[::-1];x=g.metrics(a,b)
        self.assertEqual(g.scope(x['ref_margin']),'near'); self.assertTrue(x['flip']); self.assertAlmostEqual(x['tv'],.0002);self.assertGreater(x['kl'],0)
    def test_topk_union_includes_candidate_entrant(self):
        a=np.log([.6,.399,.001]); b=np.log([.3,.30,.4]); x=g.metrics(a,b,k=1)
        self.assertEqual(x['top1'],2); self.assertAlmostEqual(x['top10_max'],np.log(400))
    def test_normalization_and_deterministic_ties(self):
        a=np.log([.4,.4,.2]); x=g.metrics(a,a+.00001);self.assertEqual(x['ref_top1'],0);self.assertEqual(x['margin'],0);self.assertAlmostEqual(x['tv'],0);self.assertAlmostEqual(x['kl'],0)
    def test_positions_are_unique_and_in_dev_scope(self):
        self.assertEqual({len(g.prefix(i)) for i in g.POS},{130,260});self.assertEqual(len({g.ih(g.prefix(i)) for i in g.POS}),32)

class CancelTests(unittest.TestCase):
    def test_both_streams_must_overlap_abort(self):
        tag='devkv-b-8';s=dict(rid=tag+'-s',done=True,input_sha=g.ih(g.prefix(7)),cap=160,ack=None,finish={'type':'length'},ids=[11]*160,samples=[[1,1],[5,4],[160,50]])
        t=dict(rid=tag+'-t',done=True,input_sha=g.ih(g.prefix(6)),cap=128,ack=[8,10,12,200],finish={'type':'abort'},ids=[14]*10,samples=[[1,1],[8,9],[10,15]])
        row=dict(ok=True,survivor=s,target=t); self.assertEqual(g.mixed(row,'b',8)['survivor_before'],5)
        invalid=copy.deepcopy(row);invalid['target']['finish']={'type':'length'}
        with self.assertRaises(ValueError):g.mixed(invalid,'b',8)
        invalid=copy.deepcopy(row);invalid['survivor']['samples']=[[1,18],[160,50]]
        with self.assertRaises(ValueError):g.mixed(invalid,'b',8)

class WireTests(unittest.TestCase):
    def request(self):
        return dict(model=wire.MODEL,stream=True,max_tokens=4096,thinking={'type':'enabled'},reasoning_effort='max',tools=[{'function':{'name':'read'}}],messages=[{'role':'system','content':wire.SYSTEM},{'role':'assistant','tool_calls':[{'id':'abc'}],'reasoning_content':''.join(wire.THINK)},{'role':'tool','tool_call_id':'abc','content':wire.FACT}])
    def test_max_or_reasoning_history_mismatch_fails(self):
        self.assertTrue(wire.wire(self.request(),False,2)['synthetic_reasoning_exact'])
        for key,value in [('reasoning_effort','high'),('thinking',{'type':'disabled'}),('model','other')]:
            q=self.request();q[key]=value
            with self.assertRaises(ValueError):wire.wire(q,False,2)
        q=self.request();q['messages'][1]['reasoning_content']=wire.THINK[0]
        with self.assertRaises(ValueError):wire.wire(q,False,2)
    def test_h_not_silently_shared(self):
        with self.assertRaises(ValueError):wire.wire(self.request(),True,2)


@unittest.skipUnless(os.getenv('EE_G1_D2_EVIDENCE'),'archived D2 run not supplied')
class D2RawTests(unittest.TestCase):
    root=Path(os.getenv('EE_G1_D2_EVIDENCE') or '.')

    def test_replay_archive(self):
        doc=g.audit(self.root);saved=json.loads(Path(g.__file__).with_name('receipt-d2-20261123.json').read_bytes());self.assertEqual(doc,saved)
        self.assertTrue(doc['exact_reference_reproduced']);self.assertEqual(doc['request_hit_counts']['b'],32)
        self.assertEqual(doc['mixed_verdict'],'pass_scoped');self.assertFalse(doc['opened_g1_v'])

    def test_missing_c_does_not_erase_independent_d(self):
        with tempfile.TemporaryDirectory() as t:
            dest=Path(t)/'raw';shutil.copytree(self.root,dest,ignore=shutil.ignore_patterns('c.jsonl'))
            doc=g.audit(dest);self.assertTrue(doc['issues']);self.assertEqual(doc['summaries']['c']['n'],0)
            self.assertEqual(doc['summaries']['d']['top1_flips'],9);self.assertFalse(doc['formal_g1'])

if __name__=='__main__':unittest.main()
