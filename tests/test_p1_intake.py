import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.p1 import intake


class ReviewIntakeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / 'negative-raw').mkdir()
        (self.root / 'round2-raw').mkdir()
        self.write('run.json', {'schema': 'p1-review-run-v1', 'reviewer_supplied': 'toy',
                                'status': 'ready_for_reviewer_decision', 'started_utc': self.stamp(1),
                                'ended_utc': self.stamp(16), 'wall_seconds': 15.2})
        self.stage('negative-raw', 2, 8, 'observed_not_independently_accepted')
        self.stage('round2-raw', 9, 15, 'complete_author_only')

    def stamp(self, s):
        return f'2020-01-01T00:00:{s:02}+00:00'

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data), encoding='utf-8')

    def stage(self, name, start, end, state):
        self.write(name + '/summary.json', {'start_utc': self.stamp(start), 'end_utc': self.stamp(end),
                                            'charged_seconds': end - start, 'status': state})

    def test_parent_clock_charged_once(self):
        self.assertEqual(intake.replay_cost(self.root, 'toy'), 16)

    def test_mixing_a_different_reviewer_fails(self):
        with self.assertRaises(ValueError): intake.replay_cost(self.root, 'someone else')

    def test_overlapping_or_outside_parent_stage_fails(self):
        for start,end in [(7,15),(9,17)]:
            self.stage('round2-raw',start,end,'complete_author_only')
            with self.assertRaises(ValueError): intake.replay_cost(self.root,'toy')

    def test_missing_stage_is_not_an_acceptance(self):
        (self.root / 'round2-raw/summary.json').unlink()
        with self.assertRaises(FileNotFoundError): intake.replay_cost(self.root,'toy')

    def test_unsigned_manifest_rejected_before_accounting(self):
        import hashlib
        m=b'{"status":"pending_human_decision"}'
        with patch.object(intake,'replay_cost') as cost:
            with self.assertRaisesRegex(ValueError,'No approved release'):
                intake.inspect_review({'executor_protocol_sha256':intake.protocol_hash()},m,hashlib.sha256(m).hexdigest(),self.root)
            cost.assert_not_called()
