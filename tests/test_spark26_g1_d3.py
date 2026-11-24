from collections import Counter
import json
import os
from pathlib import Path
import sys
import unittest

BASE = Path(__file__).resolve().parents[1]/'experiments/spark26_gguf'
sys.path.insert(0, str(BASE))
from g1_d3 import audit as d3
from g1_d3.screen import COUNT, score


class ScreenTests(unittest.TestCase):
    def rows(self):
        return {i:dict(tv=0.02, ref_margin=1.0, flip=False) for i in range(COUNT)}

    def test_boundaries(self):
        rs=self.rows()
        for i in range(3):rs[i]['tv']=0.05
        self.assertEqual(score(rs)['status'],'within_dev_screen')
        rs[3]['tv']=0.02000001
        self.assertEqual(score(rs)['status'],'outside_dev_screen')
        rs=self.rows();rs[3]['tv']=0.05000001
        self.assertEqual(score(rs)['tv_over_005'],[3])

    def test_margin_and_missing(self):
        rs=self.rows();rs[0].update(flip=True,ref_margin=0.125)
        self.assertEqual(score(rs)['status'],'within_dev_screen');del rs[9]
        self.assertEqual(score(rs)['status'],'insufficient_evidence');rs[0]['ref_margin']=1.0
        self.assertEqual(score(rs)['status'],'outside_dev_screen');self.assertEqual(score(rs)['missing'],[9])

    def test_incomplete_or_malformed_inputs(self):
        rs=self.rows();rs[0]['tv']=float('nan');rs[1]['flip']=0;del rs[2]['ref_margin']
        self.assertEqual(score(rs)['missing'],[0,1,2]);rs[32]=rs[1]
        with self.assertRaises(ValueError):score(rs)

    def test_trace_checks_every_layer_and_mode(self):
        r='local0a';d='f'*64;i=0;ids=d3.prefix(i);tag=f'dev-{r}-{i:02}:'+d3.ih(ids)
        lines=[f'EE_G1D3_HIT {tag} {len(ids)} model.layers.{k}.self_attn.q_k_v_proj {"whole" if k else "split"} {d3.MODULE} {d}' for k in range(28)]
        self.assertEqual(d3.layer_coverage(lines,r,d),{0})
        for broken in (lines[:-1],lines+[lines[0]],[s.replace(' whole ',' split ') for s in lines]):
            self.assertEqual(d3.layer_coverage(broken,r,d),set())
        with self.assertRaises(ValueError):d3.layer_coverage(lines,r,'0'*64)


@unittest.skipUnless(os.getenv('EE_G1_D3_EVIDENCE') and os.getenv('EE_G1_D2_EVIDENCE'),'raw D2 and D3 archives required')
class Golden(unittest.TestCase):
    def test_replay(self):
        raw=Path(os.environ['EE_G1_D3_EVIDENCE']);old=Path(os.environ['EE_G1_D2_EVIDENCE'])
        result=d3.audit(raw,old);expected=json.loads((BASE/'g1_d3/receipt-d3-20261123.json').read_bytes());self.assertEqual(result,expected)
        self.assertFalse(result['validation_opened']);self.assertFalse(result['formal_g1'])

if __name__=='__main__':unittest.main()
