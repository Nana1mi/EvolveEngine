import tempfile
import unittest
from pathlib import Path

from evolveengine.v3_harness import Lineage, Snapshot, digest


H0 = dict(prompts=[], skills=[], extensions=[], workflows=[], memory=[])
H1 = dict(H0, extensions=['extensions/count.mjs'])
LOCK = dict(pi_version='1.5.2', node_version='24.13.0', **{k:digest(k.encode()) for k in ('pi_image','package_lock_sha','runner_sha','base_config_sha')})
GOOD = {'extensions/count.mjs': b'export const count=()=>0;', 'deps/helper.mjs': b'export const identity=x=>x;'}
BAD = dict(GOOD, **{'extensions/count.mjs': b'export const count=()=>{throw Error("broken at use time")};'})


class HarnessTest(unittest.TestCase):
    def test_freeze_cutoff_and_explicit_rollback(self):
        with tempfile.TemporaryDirectory() as t:
            log=Lineage(Path(t)/'ledger',LOCK)
            h0=Snapshot.make({},H0,LOCK);self.assertEqual(log.initialize(h0),h0.identity)
            good=log.submit('evolving_train',GOOD,H1);bad=log.submit('evolving_train',BAD,H1)
            self.assertEqual(log.freeze(),h0)
            trace=log.activate('evolving_train',bad,h0.identity,actor='tool_call',request_id='q1')
            self.assertEqual(trace['new'],bad); frozen=log.freeze()
            self.assertEqual(frozen.identity,bad);self.assertEqual(dict(frozen.items),BAD)
            with self.assertRaises(PermissionError):log.activate('evaluation',good,bad,actor='tool_call',request_id='q2')
            with self.assertRaises(PermissionError):log.submit('fixed_train',GOOD,H1)
            with self.assertRaises(ValueError):log.activate('evolving_train',good,h0.identity,actor='tool_call',request_id='stale')
            self.assertEqual(log.active(),frozen)
            self.assertEqual(log.activate('evolving_train',good,bad,actor='tool_call',request_id='q3')['old'],bad)
            self.assertEqual(log.active().identity,good)
            self.assertEqual(Lineage(log.root,LOCK).get(frozen.identity),frozen)
            frozen.stage(Path(t)/'evaluation',LOCK)
            self.assertEqual((Path(t)/'evaluation/extensions/count.mjs').read_bytes(),BAD['extensions/count.mjs'])

    def test_dependencies_and_pi_are_part_of_identity(self):
        a=Snapshot.make(GOOD,H1,LOCK)
        self.assertNotEqual(a.identity,Snapshot.make(dict(GOOD,**{'deps/helper.mjs':b'changed'}),H1,LOCK).identity)
        with self.assertRaises(ValueError):a.recheck(dict(LOCK,pi_version='1.5.3'))
        with tempfile.TemporaryDirectory() as t:
            stage=Path(t)/'installed';a.stage(stage,LOCK)
            (stage/'deps/helper.mjs').write_bytes(b'wrong dependency')
            with self.assertRaises(ValueError):a.verify_stage(stage,LOCK)

    def test_load_order_and_path_contract(self):
        files={'extensions/first.mjs':b'1','extensions/second.mjs':b'2'}
        order=list(files)
        self.assertNotEqual(Snapshot.make(files,dict(H0,extensions=order),LOCK).identity,
            Snapshot.make(files,dict(H0,extensions=order[::-1]),LOCK).identity)
        for candidate in ('../escape','extensions/C:/foo','extensions//a','extensions/con.mjs','extensions/extra.','extensions/a\\b'):
            with self.subTest(candidate=candidate),self.assertRaises(ValueError):Snapshot.make({candidate:b'1'},H0,LOCK)
        with self.assertRaises(ValueError):Snapshot.make({'deps/A/foo':b'1','deps/a/bar':b'2'},H0,LOCK)
        with self.assertRaises(ValueError):Snapshot.make({'deps/a':b'1','deps/a/bar':b'2'},H0,LOCK)
        with self.assertRaises(ValueError):Snapshot.make({'skills/x.md':b'1'},dict(H0,extensions=['skills/x.md']),LOCK)

    def test_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            log=Lineage(t,LOCK); h=Snapshot.make(GOOD,H1,LOCK); log.initialize(h)
            (log.versions/h.identity/'files/deps/helper.mjs').write_bytes(b'replaced')
            with self.assertRaises(ValueError):log.freeze()

if __name__=='__main__': unittest.main()
