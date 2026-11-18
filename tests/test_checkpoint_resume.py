from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evolveengine.controller import Controller, Lane
from evolveengine.experiment import MOCK_PROTOCOL, deploy_pair
from evolveengine.journal import Interrupted, Journal
from evolveengine.mock import MockBackend
from evolveengine.partitions import Partitions
from evolveengine.schema import Cost


class ResumeTests(unittest.TestCase):
    def test_resume_after_parent_timing_reuses_it(self):
        def stop(key):
            if key == "arm/step-0/screen/child":
                raise Interrupted(key)

        with TemporaryDirectory() as directory:
            database = Path(directory) / "journal.sqlite"
            backend = MockBackend()
            partition = Partitions(3, b"c" * 32)
            lane = Lane("arm", "fixed", "s1", Cost(20, 20, 1000, blocks=2))
            environment = {"simulated": True}
            with self.assertRaises(Interrupted), Journal(database, stop) as journal:
                controller = Controller(journal, backend, partition, environment)
                start = controller.checkpoint(backend.initial_engine(), "arm")
                controller.run(lane, start, 1)
            with Journal(database) as journal:
                controller = Controller(journal, backend, partition, environment)
                result = controller.run(lane, start, 1)
                self.assertEqual(result["checkpoint"]["engine_digest"], start.engine_digest)
                self.assertEqual(result["cost"]["blocks"], 2)
                timing = {row["id"].split("/")[-1]: row["receipt"]["status"]
                          for row in journal.budget_rows("arm/step-0/screen/")}
                self.assertEqual(timing, {"parent": "ok", "child": "uncertain"})
                self.assertEqual(controller.run(lane, start, 1), result)

    def test_paired_trial_accepts_and_rejects_without_leakage(self):
        nodes = ("s1", "s2", "s3", "s4")
        for scenario in ("accept", "reject"):
            with self.subTest(scenario=scenario), TemporaryDirectory() as directory:
                with Journal(Path(directory) / "run.sqlite") as journal:
                    report = deploy_pair(journal, MockBackend(scenario),
                                         Partitions(3, b"t" * 32, final_salt=b"q" * 32),
                                         MOCK_PROTOCOL, nodes, 0, {"simulated": True}, causal_points=1)
                    decision = next(row for row in report["events"] if row["id"].endswith("/decided"))
                    self.assertEqual(decision["status"], scenario + "ed")
                    self.assertEqual(len(decision["comparisons"]), 2)
                    for row in decision["comparisons"]:
                        self.assertTrue(row["checkpoint"]["lineage"].startswith("bank/D/"))
                        self.assertEqual(len(row["gates"]), 2)
                    dev_calls = [row for row in report["budget_ledger"] if "/window-0/d-" in row["id"]]
                    self.assertTrue(all(row["owner"] == "evolving" and row["node"] == "s2" for row in dev_calls))
                    self.assertTrue(any(not row["physical"] for row in dev_calls))
                    engine = report["submission"]["evolving"]["checkpoint"]["engine_digest"]
                    last = [row for row in report["events"]
                            if row["id"].startswith("replicate-0/main/evolving/step-")
                            and row["id"].endswith("/finished")][-1]
                    self.assertEqual(last["checkpoint"]["engine_digest"], engine)
                    self.assertEqual(report["sealed_evaluation"]["rq1"]["status"], "ok")
                    kinds = [event["id"] for event in report["events"]]
                    self.assertLess(kinds.index("replicate-0/sealed"), kinds.index("replicate-0/evaluation/rq1/evaluated"))
