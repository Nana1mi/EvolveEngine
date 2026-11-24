import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'experiments' / 'spark26_gguf' / 'batch_trace_audit.py'
spec = importlib.util.spec_from_file_location('spark26_batch_trace_audit', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BatchTraceAuditTests(unittest.TestCase):
    def test_requires_actual_joint_pick_not_just_in_flight_requests(self):
        stages = ('paired-1',)
        audit.validate_picks("picked-rids=['paired-1-1', 'paired-1-0']", stages)
        for log in ("picked-rids=['paired-1-0']\npicked-rids=['paired-1-1']",
                    "picked-rids=['paired-1-0', 'paired-1-1', 'unrelated']",
                    "picked-rids=['paired-1-0', 'paired-1-1']\npicked-rids=['paired-1-0']",
                    "request A 'paired-1-0'; request B 'paired-1-1'"):
            with self.subTest(log=log), self.assertRaisesRegex(ValueError, 'schedule pick'):
                audit.validate_picks(log, stages)


if __name__ == '__main__':
    unittest.main()
