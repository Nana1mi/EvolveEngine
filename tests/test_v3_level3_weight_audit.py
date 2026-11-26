from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from evolveengine import v3_level3_weight_audit as w
from evolveengine.v3_level3_source import SourceContract


class WeightAuditTests(TestCase):
    def test_signals_do_not_decide_legality(self):
        base={'x.py':b'# initial\n'}
        for change in (b'self.cached = weight.half()',b'register_buffer("q", quantize_q4(weight))',
                       b'new = torch.empty_like(weight, dtype=torch.float8_e4m3fn)'):
            self.assertGreater(w.static_scan(base, [('x.py',change)])['changed_line_hits'],0)
        ref={'weight_observation':dict(workload='a',sampled_bytes={'ready':0,'post_requests':10})}
        obs={'weight_observation':dict(workload='a',sampled_bytes={'ready':0,'post_requests':500<<20})}
        hint=w.memory_hint(obs,ref);self.assertTrue(hint['investigate_growth_over_300_mib'])
        self.assertEqual(w.memory_hint({},ref)['status'],'no_matched_comparison')
        obs['weight_observation']['workload']='b';self.assertEqual(w.memory_hint(obs,ref)['status'],'no_matched_comparison')
        data=dict(scenes={'W1':dict(tpot_ms=4),'W2':dict(tpot_ms=4)},
                  speculation={'timed':{'W2':dict(requests=2,reported=2,proposed=3)}})
        bw=w.bandwidth_hint(data); self.assertEqual(bw['W1']['speculation'],'unknown');self.assertTrue(bw['W1']['investigate'])
        self.assertFalse(bw['W2']['investigate'])

    def test_input_lookup_and_multiline_tables_are_review_hints_only(self):
        changes = [('tables.json', ('[\n'+',\n'.join(map(str,range(66)))+']').encode()),
                   ('dispatch.cu', b'auto h = digest(input_ids);\nif (answers.contains(h)) return answers[h];')]
        found = w.static_scan({}, changes)['locations']
        self.assertEqual({t for x in found for t in x['hints']}, {'input_fingerprint_dispatch','embedded_integer_table'})
        self.assertFalse(w.static_scan(dict(changes), changes)['locations'])

    def test_review_cannot_reuse_another_build_or_only_a_weight_hash(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp); c=SourceContract({'python/sglang/srt/x.py':b'x=1\n'});e=c.from_args({'files':{}})
            chosen={'engine':e.digest,'build':'a'*64,'quality':'b'*64};b=dict(engine=e.digest,evidence=chosen['build']);q=dict(chosen,data={},evidence=chosen['quality'])
            packet=lambda trigger:w.packet(folder,c,e,chosen,b,q,q,[],trigger=trigger,context={})
            final=packet('final_nomination');checkpoint=packet('checkpoint')
            runtime=folder/'external.json'; probe=dict(schema='q8-runtime-witness-v1',chosen=chosen,method='simulated external probe',
                observations=[dict(stage=stage,representation='fixture',lifetime='fixture',raw_observation='fixture') for stage in ('weights_loaded','post_requests')])
            runtime.write_bytes(w.wire(dict(gguf_sha256='c'*64))); r=dict(chosen=chosen,decision='approved',reviewer='fixture',
                packet_sha256=final,checks=dict.fromkeys(w.CHECKS,'offline finding'),runtime_evidence_sha256=w.file_digest(runtime))
            with self.assertRaises(ValueError):w.record_review(folder,final,r,runtime)
            runtime.write_bytes(w.wire(probe));r['runtime_evidence_sha256']=w.file_digest(runtime)
            with self.assertRaises(ValueError):w.record_review(folder,checkpoint,r,runtime)
            w.record_review(folder,final,dict(r,decision='inconclusive'),runtime)
            with self.assertRaisesRegex(ValueError,'pending'):w.approved(folder,final,chosen)
            w.record_review(folder,final,r,runtime);w.approved(folder,final,chosen)
            with self.assertRaises(ValueError):w.approved(folder,final,dict(chosen,build='d'*64))
            archived=folder/'runtime-evidence'/r['runtime_evidence_sha256'];archived.write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'changed'):w.approved(folder,final,chosen)
            archived.write_bytes(runtime.read_bytes()); w.record_review(folder,checkpoint,dict(r,packet_sha256=checkpoint,decision='rejected'),runtime)
            with self.assertRaisesRegex(ValueError,'rejected'):w.approved(folder,final,chosen)
