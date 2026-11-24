import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

DIRECTORY = Path(__file__).resolve().parents[1]/'experiments/spark26_gguf'
with patch.object(sys,'path',[str(DIRECTORY),*sys.path]):
    from qkv_micro_audit import read_data
    import reference_qkv


def fixture(slow=None):
    p = json.loads((DIRECTORY/'qkv_micro_lock.json').read_bytes())
    rows = [dict(kind='identity',lock_sha256='a',source_sha256=p['source_sha256'],capability=[12,1],gpu='NVIDIA GB10')]
    for kind,ms in (('qkv',p['shape_order']),('gate_up_control',p['negative_control_shape_order'])):
        for m in ms:
            rows.append(dict(kind='exact',projection=kind,m=m,layers=p['layers']))
            for i,order in enumerate(p['sample_orders']):
                rows.append(dict(kind='sample',projection=kind,m=m,index=i,order=order,ms_per_call={'a':1.,'b':1.1 if m==slow else .93}))
    return rows+[dict(kind='complete')],p


class QkvMicroTest(unittest.TestCase):
    def test_all_shapes_neg_control_excluded_and_bit_evidence_count(self):
        rows,p = fixture()
        self.assertEqual(read_data(rows,p,'a')['decision'],'all_tested_rows')
        self.assertEqual(read_data(rows,p,'a')['bf16_bit_exact_checks'],48)
        self.assertEqual(read_data(rows,p,'a')['sample_pairs'],48)

    def test_scope_depends_on_decode_and_prefill(self):
        rows,p = fixture(slow=130)
        self.assertEqual(read_data(rows,p,'a')['decision'],'decode_rows_only')
        rows,p = fixture(slow=1)
        self.assertEqual(read_data(rows,p,'a')['decision'],'do_not_advance')

    def test_incomplete_and_out_of_order_fail_closed(self):
        rows,p = fixture()
        with self.assertRaisesRegex(ValueError,'completion'):
            read_data(rows[:-1],p,'a')
        rows[2],rows[3]=rows[3],rows[2]
        with self.assertRaisesRegex(ValueError,'sample'):
            read_data(rows,p,'a')

    def test_qkv_edit_dispatch_remains_narrow_without_ffn(self):
        small = (b'class GGUFLinearMethod(LinearMethodBase):\n'
                 b'    def apply(self, layer, x, bias=None):\n'
                 + reference_qkv.ANCHOR.encode() + b'            return "split"\n')
        with patch.object(reference_qkv,'BASE_SHA256',hashlib.sha256(small).hexdigest()):
            code = reference_qkv.edited(small)
        torch = SimpleNamespace(uint8='u8',bfloat16='bf16')
        calls = []

        def matmul(x,w,qtype):
            calls.append((x.shape,w.shape,qtype))
            return 'joined'

        ns = dict(LinearMethodBase=object,_is_cuda=True,torch=torch,WeightType=SimpleNamespace(Q8_0=8),
                  fused_mul_mat_gguf=matmul)
        exec(code,ns)
        method = ns['GGUFLinearMethod']()
        wt = SimpleNamespace(shape=(3072,2176),dtype='u8',device='cuda:0',is_contiguous=lambda:True,
                             shard_id=['q','k','v'],shard_offset_map={'q':(0,2048,2176),'k':(2048,2560,2176),'v':(2560,3072,2176)})
        meta = dict(qweight=wt,qweight_type=SimpleNamespace(shard_weight_type=dict.fromkeys('qkv',8)))
        qkv=type('QKVParallelLinear',(),meta)()
        ff=type('MergedColumnParallelLinear',(),meta)()
        x = SimpleNamespace(shape=(2,2048),ndim=2,is_cuda=True,is_contiguous=lambda:True,dtype='bf16',device='cuda:0')
        self.assertEqual(method.apply(qkv,x),'joined')
        self.assertEqual(method.apply(ff,x),'split')
        x.shape = (3,2048)
        self.assertEqual(method.apply(qkv,x),'split')
        x.shape=(2,2048)
        x.device='cuda:1'
        self.assertEqual(method.apply(qkv,x),'split')
        self.assertEqual(calls,[((2,2048),(3072,2176),8)])
