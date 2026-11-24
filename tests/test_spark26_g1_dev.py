import base64
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/spark26_gguf'))
from g1_dev import audit as g


class DecisionTests(unittest.TestCase):
    def test_missing_does_not_erase_verified_difference(self):
        self.assertEqual(g.combine({'eos':{'status':g.FAIL},'vec':{'status':g.UNKNOWN}}),g.FAIL)
        self.assertEqual(g.combine({'eos':{'status':g.PASS},'vec':{'status':g.UNKNOWN}}),g.UNKNOWN)

    def test_ack_without_abort_terminal_is_not_a_pass(self):
        row=dict(ok=True, rid='g1-abort-qkv-target', trigger=4, last_n=7, cap=512, abort_http=200, done=True, stream_error=None)
        for finish in (None,{'type':'length'},{'type':'stop','matched':1}):
            self.assertEqual(g.evaluate(g.cancelled,{'cancel':dict(row,finish=finish)},'qkv')['status'],g.UNKNOWN)

    def test_missing_matched_is_not_an_eos_counterexample(self):
        row=dict(ok=True, rid='g1-qkv-before-0', cap=48, n=2, ids=[5,1], input_sha='a'*64, finish={'type':'stop'})
        self.assertEqual(g.evaluate(g.natural,{'before0':row},'qkv','before',0)['status'],g.UNKNOWN)

    def test_vectors_require_complete_finite_normalized_distribution(self):
        prefix=sha256(json.dumps([6510+i%37 for i in range(130)]).encode()).hexdigest()
        def decode(value,count):
            raw=struct.pack('<f',value)*count
            r=dict(count=130,kind='numeric130',prefix_sha256=prefix,ok=True,vocab=g.VOCAB,
                   format='little-f32-zlib-base64',sha256=sha256(raw).hexdigest(),vector=base64.b64encode(zlib.compress(raw)))
            return g.vector({'numeric130':r},130)
        for v,n in [(-math.log(g.VOCAB),1),(math.nan,g.VOCAB),(0,g.VOCAB)]:
            with self.assertRaises(ValueError): decode(v,n)
        self.assertAlmostEqual(decode(-math.log(g.VOCAB),g.VOCAB)[1],1,places=5)


@unittest.skipUnless(os.getenv('EE_G1_R1_EVIDENCE'),'archived raw dir not supplied')
class ArchivedReplayTests(unittest.TestCase):
    root=Path(os.getenv('EE_G1_R1_EVIDENCE') or '.')

    def test_golden(self):
        result=g.audit(self.root)
        saved=Path(g.__file__).with_name('receipt-r1-20261123.json')
        self.assertEqual(result,json.loads(saved.read_bytes()))
        self.assertTrue(result['evidence_complete'])
        for n in (130,260):
            self.assertEqual(result['arms']['qkv']['checks'][f'numeric{n}']['kl_e0_to_role'],0)
            self.assertGreater(result['arms']['fault']['checks'][f'numeric{n}']['kl_e0_to_role'],0)
        self.assertEqual(result['arms']['fault']['checks']['post_cancel_stability0']['status'],g.PASS)
        self.assertEqual(result['arms']['fault']['checks']['after0']['status'],g.FAIL)

    def test_missing_arm_does_not_become_zero_or_annul_other_arm(self):
        with tempfile.TemporaryDirectory() as t:
            dst=Path(t)/'archive'; shutil.copytree(self.root,dst,ignore=shutil.ignore_patterns('qkv.jsonl'))
            result=g.audit(dst)
            self.assertEqual(result['arms']['qkv']['status'],g.UNKNOWN)
            self.assertEqual(result['arms']['fault']['status'],g.FAIL)
            self.assertFalse(result['control_pattern_observed'])
            self.assertFalse(result['evidence_complete'])

    def test_missing_numeric_is_not_a_fault_but_valid_eos_failure_remains(self):
        data={r:g.rows_for(self.root,r) for r in g.ROLES}
        del data['fault']['numeric130']
        eos=g.evaluate(g.eos_check,data,'fault','before',0)
        num=g.evaluate(g.vector,data['fault'],130)
        self.assertEqual((eos['status'],num['status'],g.combine({'eos':eos,'numeric':num})),(g.FAIL,g.UNKNOWN,g.FAIL))


if __name__=='__main__':
    unittest.main()
