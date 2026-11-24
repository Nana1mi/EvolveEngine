import tempfile
import unittest
from pathlib import Path

from evolveengine.v3_harness import Lineage, Snapshot, digest
from evolveengine.v3_session import DevReceipt, Session, SourceContract

EMPTY=dict(prompts=[],skills=[],extensions=[],workflows=[],memory=[])
LOCK=dict(pi_version='1.5.2',node_version='24.13.0',**{k:digest(k.encode()) for k in ('pi_image','package_lock_sha','runner_sha','base_config_sha')})
BASE={'engine.py':b'def x():\n    return 1\n'}
EDIT={'files':{'engine.py':'def x():\n    return 2\n'}}
OTHER={'files':{'engine.py':'def x():\n    return 3\n'}}

class SessionTest(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup)
        self.line=Lineage(self.dir.name,LOCK);self.line.initialize(Snapshot.make({},EMPTY,LOCK));
        self.contract=SourceContract(BASE,BASE,max_files=1,max_bytes=4096,max_text_growth=20,max_new_literal=20)

    def session(self,worker,mode='evaluation',node=200):
        return Session(mode,self.line,self.contract,{'toy':1},worker,wall_ms=20000,node_ms=node,dev_calls=2,
                       reservations={'develop_quality':80,'develop_measure':80})

    def test_explicit_quality_signature_is_bound_to_candidate_not_speed(self):
        def worker(op,artifact,remaining):return DevReceipt(artifact.digest,op,'valid' if op=='develop_quality' else 'observed',30,'trusted-1',{'score':.5})
        s=self.session(worker)
        self.assertEqual(s('deliver_engine',EDIT)[0],403);self.assertEqual(s('develop_quality',EDIT)[1]['status'],'valid')
        self.assertEqual(s('deliver_engine',OTHER)[0],403);self.assertIsNone(s.delivered)
        self.assertEqual(s('develop_measure',EDIT)[1]['data']['score'],.5);self.assertIsNone(s.delivered)
        self.assertEqual(s('deliver_engine',EDIT)[1]['result'],'signed');self.assertEqual(s.delivered,self.contract.from_args(EDIT))
        self.assertEqual(s.spent_node,60)

    def test_invalid_reply_is_not_legal_zero_and_worker_lease_poisoned(self):
        s=self.session(lambda op,e,m:DevReceipt('different',op,'valid',20,'wrong',{}))
        self.assertEqual(s('develop_quality',EDIT),(502,{'status':'insufficient_evidence'}));self.assertEqual(s.spent_node,80)
        self.assertEqual(s('deliver_engine',EDIT)[0],403)
        self.assertEqual(s('develop_measure',EDIT)[0],403);self.assertEqual(s.spent_node,80)

    def test_settled_missing_candidate_evidence_can_be_followed_by_another_attempt(self):
        bad=self.contract.from_args(EDIT).digest
        def worker(op,e,_):return DevReceipt(e.digest,op,'insufficient_evidence' if e.digest==bad else 'valid',37,'settled',{})
        s=self.session(worker)
        self.assertEqual(s('develop_quality',EDIT)[1]['status'],'insufficient_evidence')
        self.assertEqual(s('deliver_engine',EDIT)[0],403);self.assertFalse(s.worker_invalid)
        self.assertEqual(s('develop_quality',OTHER)[1]['status'],'valid')
        self.assertEqual(s('deliver_engine',OTHER)[1]['result'],'signed');self.assertEqual(s.spent_node,74)

    def test_denied_tools_and_out_of_scope_source_never_invoke_worker(self):
        def worker(*unused):self.fail('invalid request invoked worker')
        s=self.session(worker)
        for route,args in (('sealed',{}),('judge',{}),('develop_profile',EDIT),('develop_quality',{'files':{'notes.md':'keep'}}),
                           ('develop_quality',{'files':{'engine.py':'#'+'x'*200+'\nreturn 1'}}),('harness_submit',{'files':{},'load':EMPTY})):
            with self.subTest(route=route):self.assertEqual(s(route,args)[0],403)
        self.assertEqual(s.spent_node,0)

    def test_insufficient_reservation_blocks_admission(self):
        s=self.session(lambda *a:self.fail('overspent'),node=79);self.assertEqual(s('develop_quality',EDIT)[0],403)

    def test_evolving_submit_and_explicit_activate_only(self):
        s=self.session(lambda *a:None,mode='evolving_train');new={'files':{'prompts/a.md':'note'},'load':dict(EMPTY,prompts=['prompts/a.md'])}
        h0=self.line.active().identity; selected=s('harness_submit',new)[1]['version'];self.assertEqual(self.line.active().identity,h0)
        self.assertEqual(s('harness_activate',{'version':selected,'expected':h0})[0],200)
        self.assertEqual(self.line.freeze().identity,selected)

if __name__=='__main__':unittest.main()
