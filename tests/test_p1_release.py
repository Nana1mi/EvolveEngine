import hashlib,json,tempfile,unittest
from pathlib import Path
from experiments.p1.release import NotReleased,verified_cards,balanced_schedule
from experiments.p1.trial import LIMIT
from evolveengine.schema import digest


def sha(data):return hashlib.sha256(data).hexdigest()


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.dir=Path(self.tmp.name)
        H='2'*64;self.exp={'engine_sha':'f'*40,'harness_sha':'b'*40,'worker_digest':H,'executor_protocol_sha256':'d'*64,'minimum_noise_log':.009,'cards':{}}
        self.m={**self.exp,'cards':{},'schema':'p1-independent-release-v1','status':'approved','reviewer':'test-only-peer', 'author':'test-only-builder','reviewer_did_not_build':True,'budget':LIMIT.record()}
        for k,kind in [('A','negative'),('B','negative'),('C','positive_local'),('D','positive_cross')]:
            s={'kind':kind,'card_sha256':H,'fixture_sha256':H,'inputs_sha256':H,'witness_code':'toy-specific','blocks':[str(i)*64 for i in range(3)]};self.exp['cards'][k]=s
            raw={'card':k,'source':'independent_node_replay','engine_sha':self.exp['engine_sha'],'worker_digest':H,'fixture_sha256':H,'inputs_sha256':H,'observed':'toy-specific','detection_confirmed':True,'mechanism_verified':True,'quality':dict.fromkeys(('launch','full_vocab','kl','top1','sequence','output_length','eos','kv_isolation'),True),'noise_log':.01,'pairs':[{'block_sha256':i,'e0_tok_s':10,'ref_tok_s':11} for i in s['blocks']]}
            raw['source_files']={}
            for i in range(8):
                src=f'{k}-{i}.json';data=json.dumps({'toy':i}).encode();(self.dir/src).write_bytes(data);raw['source_files'][src]=sha(data)
            name=k+'.raw.json';b=json.dumps(raw).encode();(self.dir/name).write_bytes(b);r={**s,'card_id':k,'schema':'p1-independent-replay-v1','status':'accepted','reviewer':self.m['reviewer'],'fresh_checkout':True,'engine_sha':self.exp['engine_sha'],'worker_digest':H,'raw_file':name,'raw_sha256':sha(b)};b=json.dumps(r).encode();f=k+'.json';(self.dir/f).write_bytes(b);self.m['cards'][k]={'receipt_file':f,'receipt_sha256':sha(b)}
        self.m['expected_digest']=digest(self.exp)
    def run_gate(self):
        b=json.dumps(self.m).encode();return verified_cards(b,sha(b),self.dir,self.exp)
    def test_synthetic_attestation_has_balanced_nonoverlapping_slots(self):
        seq=balanced_schedule(self.run_gate());self.assertEqual(len(set(seq)),16);self.assertEqual(seq[0][1],seq[9][1]);self.assertNotEqual(seq[0][1],seq[8][1])
    def test_builder_cannot_sign(self):
        self.m['reviewer']=self.m['author'];self.assertRaises(NotReleased,self.run_gate)
    def test_incomplete_card_set(self):
        del self.m['cards']['A'];self.assertRaises(NotReleased,self.run_gate)
    def test_mutated_raw_rejected(self):
        (self.dir/'C.raw.json').write_bytes(b'{}');self.assertRaises(NotReleased,self.run_gate)
    def test_wrong_card_binding(self):
        self.exp['cards']['D']['card_sha256']='5'*64;self.assertRaises(NotReleased,self.run_gate)
    def test_untrusted_manifest(self):
        self.assertRaises(NotReleased,verified_cards,json.dumps(self.m).encode(),'0'*64,self.dir,self.exp)
    def test_gain_must_clear_each_block(self):
        p=self.dir/'C.raw.json';v=json.loads(p.read_text());v['pairs'][1]['ref_tok_s']=10;p.write_text(json.dumps(v));q=self.dir/'C.json';r=json.loads(q.read_text());r['raw_sha256']=sha(p.read_bytes());q.write_text(json.dumps(r));self.m['cards']['C']['receipt_sha256']=sha(q.read_bytes());self.assertRaises(NotReleased,self.run_gate)
    def test_dispatch_rejects_before_model_or_slot(self):
        from unittest.mock import patch
        from experiments.p1.dispatch import dispatch, protocol_hash
        expected={**self.exp,'executor_protocol_sha256':protocol_hash()}
        with patch('experiments.p1.dispatch.PiAgent') as agent:
            with self.assertRaises(NotReleased):
                dispatch(expected,b'{}','0'*64,self.dir,{},self.dir,self.dir/'runs',1)
            agent.assert_not_called();self.assertFalse((self.dir/'runs').exists())

    def test_original_node_receipts_cannot_be_lost(self):
        (self.dir/'B-3.json').write_bytes(b'null');self.assertRaises(NotReleased,self.run_gate)
