from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evolveengine.journal import Journal
from evolveengine.v3_level3_lifecycle import Checkpoint, Lifecycle, QualifiedEngine


class Level3LifecycleTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'run.sqlite3'
        self.base = QualifiedEngine('0'*64, '1'*64, '2'*64)
        self.other = QualifiedEngine('3'*64, '4'*64, '5'*64)
        self.h0, self.h1 = '6'*64, '7'*64
        self.state = Checkpoint(self.base, self.h0, '8'*64, ('9'*64,), ('a'*64,), 'b'*64)
        self.artifacts = {
            'engine': {self.base.engine, self.other.engine}, 'build': {self.base.build, self.other.build},
            'quality': {self.base.quality, self.other.quality}, 'harness': {self.h0, self.h1},
            'trusted_facts': {'8'*64}, 'harness_memory': {'9'*64}, 'research': {'a'*64}, 'wip': {'b'*64},
        }

    def controller(self, journal, max_recoveries=1):
        return Lifecycle(journal, max_recoveries=max_recoveries,
                         verify_artifact=lambda kind, ref: ref in self.artifacts[kind],
                         verify_quality=lambda engine: engine in (self.base, self.other),
                         verify_measure=lambda engine, ref: engine == self.other and ref == 'c'*64,
                         verify_wip=lambda wip, parent: wip == 'b'*64 and parent == self.base.engine)

    def test_qualification_and_measure_do_not_finish_and_controller_does_not_select(self):
        with Journal(self.path) as journal:
            run = self.controller(journal)
            run.record_quality_pass(self.base)
            segment = run.start(run.save(self.state))
            run.record_quality_pass(self.other)
            run.record_measure(self.other, 'c'*64)
            self.assertEqual(run.entry(segment), self.state)
            with self.assertRaisesRegex(ValueError, 'invalid development measure'):
                run.record_measure(self.base, 'c'*64)
            self.assertIsNone(journal.get_event('level3/delivery'))
            result = run.finish(segment, self.base, active_harness=self.h1)
            self.assertEqual(result['chosen']['engine'], self.base.engine)
            self.assertEqual(result['active_harness'], self.h1)
            self.assertNotIn('wip', result)
            with self.assertRaisesRegex(ValueError, 'already ended'):
                run.finish(segment, self.other, active_harness=self.h1)

    def test_normal_handoff_survives_restart_and_is_not_recovery(self):
        with Journal(self.path) as journal:
            run = self.controller(journal, max_recoveries=0)
            run.record_quality_pass(self.base)
            segment = run.start(run.save(self.state))
            next_state = replace(self.state, harness=self.h1)
            handoff = run.save(next_state)
            with self.assertRaisesRegex(ValueError, 'no longer active'):
                run.handoff(segment, handoff, active_harness=self.h0)
            self.assertEqual(run.entry(segment), self.state)
            self.assertEqual(run.handoff(segment, handoff, active_harness=self.h1), 2)
        with Journal(self.path) as journal:
            run = self.controller(journal, max_recoveries=0)
            self.assertEqual(run.entry(2), next_state)
            self.assertEqual(journal.get_event('level3/segment/2/start')['kind'], 'ordinary')
            with self.assertRaisesRegex(ValueError, 'already ended'):
                run.handoff(1, handoff, active_harness=self.h1)

    def test_bound_wip_separate_from_verified_engine_and_facts(self):
        with Journal(self.path) as journal:
            run = self.controller(journal)
            run.record_quality_pass(self.base)
            run.record_quality_pass(self.other)
            with self.assertRaisesRegex(ValueError, 'WIP base engine mismatch'):
                run.save(replace(self.state, engine=self.other))
            with self.assertRaisesRegex(ValueError, 'trusted_facts'):
                run.save(replace(self.state, trusted_facts=self.state.wip))
            cp = run.save(self.state)
            run.start(cp)
            self.artifacts['wip'].clear()
            with self.assertRaisesRegex(ValueError, 'wip artifact'):
                run.entry(1)

    def test_recovery_rejects_corrupt_or_unqualified_and_never_resets_cap(self):
        with Journal(self.path) as journal:
            run = self.controller(journal)
            run.record_quality_pass(self.base)
            cp = run.save(self.state)
            run.start(cp)
            with self.assertRaisesRegex(ValueError, 'not recorded as qualified'):
                run.finish(1, self.other, active_harness=self.h0)
            with self.assertRaisesRegex(ValueError, 'Unknown level3_checkpoint'):
                run.handoff(1, 'f'*64, abnormal_cause='network_failure', active_harness=self.h0)
            with self.assertRaisesRegex(ValueError, 'not recoverable'):
                run.handoff(1, cp, abnormal_cause='usage_uncertain', active_harness=self.h0)
            self.assertEqual(run.handoff(1, cp, abnormal_cause='network_failure', active_harness=self.h0), 2)
        with Journal(self.path) as journal:
            run = self.controller(journal)
            self.assertEqual(run.entry(2), self.state)
            with self.assertRaisesRegex(ValueError, 'recovery cap reached'):
                run.handoff(2, cp, abnormal_cause='generation_truncated', active_harness=self.h0)
            run.stop(2, cause='generation_truncated')
            self.assertIsNone(journal.get_event('level3/delivery'))
            self.assertEqual(journal.get_event('level3/segment/2/end'),
                             {'kind': 'stopped', 'cause': 'generation_truncated'})
