"""Regression checks for the read-only Level 3 protocol registration."""
import importlib.util
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'level3_audit_registration', ROOT / 'experiments/spark26_gguf/level3/audit_registration.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class Level3RegistrationTests(unittest.TestCase):
    def test_frozen_registration_reports_all_domains_but_no_acceptance(self):
        report = audit.audit()
        self.assertTrue(report['registration_valid'])
        self.assertEqual([x['id'] for x in report['domains']], list('ABCDEF'))
        self.assertEqual(sum(len(x['checks']) for x in report['domains']), 15)
        self.assertFalse(report['engineering_acceptance_evaluated'])
        self.assertFalse(report['launch_admission_evaluated'])

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        data = json.loads((ROOT / audit.MANIFEST).read_bytes())
        self.objective = data['objective']['path']
        paths = [audit.MANIFEST, self.objective, data['prior_registration']['path']]
        for row in data['domains']:
            paths.extend(row['existing_evidence'])
        for item in set(paths):
            target = self.root / item
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / item, target)

    def test_edited_criteria_are_not_silently_accepted(self):
        path = self.root / audit.MANIFEST
        path.write_bytes(path.read_bytes().replace(b'"required": true', b'"required":false', 1))
        with self.assertRaisesRegex(ValueError, 'frozen SHA-256 mismatch'):
            audit.audit(self.root)

    def test_edited_objective_is_not_silently_accepted(self):
        path = self.root / self.objective
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'frozen SHA-256 mismatch'):
            audit.audit(self.root)

    def test_missing_historical_evidence_is_reported(self):
        (self.root / 'experiments/spark26_gguf/v3_dev_rpc/EXPLORATION.md').unlink()
        with self.assertRaises(FileNotFoundError):
            audit.audit(self.root)
