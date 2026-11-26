import sqlite3
import tempfile
import unittest
from pathlib import Path

from evolveengine.journal import Journal
from evolveengine.v3_level3_trajectory_lane import SPLIT_CASES, SPLIT_PROTOCOL
from experiments.spark26_gguf.level3.audit_quality_stability import KIND, observed, summarize


def row(pin, *, status='valid', build='same-build', case='autoregressive_full_vocab'):
    service = 'fresh-' + pin
    groups = {k: dict(status='valid', violations=[], insufficient=[]) for k in SPLIT_CASES}
    cases = dict.fromkeys((*SPLIT_CASES, 'numeric_32', 'eos_protocol'), 'pass')
    if status == 'violation':
        groups[case].update(status=status, violations=['decode_tv'])
        cases[case] = 'fail'
    return dict(evidence=pin, engine='same-engine', build=build, operation='develop_quality', status=status,
                data=dict(case_outcomes=cases, trajectory_screen=dict(protocol=SPLIT_PROTOCOL, status=status,
                     engine='same-engine', service=service, live_gate=True, groups=groups)))


class QualityStabilityTest(unittest.TestCase):
    def test_same_build_flip_reported_but_no_retry(self):
        good, bad = row('a'), row('b', status='violation')
        out = summarize([good, bad, row('c', build='different'), good])
        self.assertFalse(out['scoring'])
        self.assertFalse(out['retry'])
        self.assertEqual(len(out['repeated_builds']), 1)
        group = out['repeated_builds'][0]
        self.assertEqual(group['status'], 'quality_flip_review')
        self.assertEqual([r['evidence'] for r in group['observations']], ['a', 'b'])

    def test_observer_failure_is_a_coverage_issue(self):
        partial = row('b') | dict(data={}, status='insufficient_evidence')
        self.assertEqual(summarize([row('a'), partial])['repeated_builds'][0]['status'], 'coverage_review')
        self.assertEqual(summarize([row('a'), row('b')])['repeated_builds'][0]['status'], 'consistent_observations')

    def test_only_ordinary_settled_observations_in_read_only_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run.sqlite3'
            with Journal(path) as journal:
                for i, result in enumerate((row('a'), row('excluded'), row('b'))):
                    extra = dict(sampling_challenge='milestone') if i == 1 else {}
                    journal.event(f'dispatch/{2*i:06}', dict(operation='worker_reserved', engine=result['engine'],
                                  worker_operation=result['operation'], **extra))
                    journal.event(f'dispatch/{2*i+1:06}', dict(operation='worker_settled', evidence=result['evidence'],
                                  result_object=journal.put(KIND, result)))
                journal.event('dispatch/000006', dict(operation='worker_reserved', engine='same-engine',
                              worker_operation='develop_quality'))
            db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
            try:
                self.assertEqual([r['evidence'] for r in observed(db)], ['a', 'b'])
            finally:
                db.close()
