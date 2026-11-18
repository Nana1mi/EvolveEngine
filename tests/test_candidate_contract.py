from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evolveengine.controller import Controller, Lane
from evolveengine.journal import Journal
from evolveengine.mock import MockBackend
from evolveengine.partitions import Partitions
from evolveengine.schema import Cost


class ObservedMock(MockBackend):
    def __init__(self, scenario):
        super().__init__(scenario)
        self.contracts = []
        self.stages = []

    def run(self, stage, request, cap, node):
        self.stages.append(stage)
        if stage == "implement":
            self.contracts.append(request["contract"])
        return super().run(stage, request, cap, node)


class ContractTests(unittest.TestCase):
    def test_engine_requires_faithful_patch_and_quality(self):
        for scenario, expected in (("accept", "mainline"), ("deviation", "deviation"),
                                   ("quality_failure", "dev_gate_failed")):
            with self.subTest(scenario=scenario), TemporaryDirectory() as directory:
                with Journal(Path(directory) / "run.sqlite") as journal:
                    backend = ObservedMock(scenario)
                    controller = Controller(journal, backend, Partitions(3, b"s" * 32), {"simulated": True})
                    start = controller.checkpoint(backend.initial_engine(), "root")
                    lane = Lane("root", "fixed", "s1", Cost(30, 30, 1000, blocks=4))
                    result = controller.step(lane, "step-0", start)
                    self.assertEqual(result["outcome"], expected)
                    self.assertEqual(len(backend.contracts), 1)
                    self.assertEqual(set(backend.contracts[0]), {"candidate", "observation", "memory"})
                    self.assertEqual(backend.contracts[0]["candidate"]["candidate_id"], "p0")
                    self.assertEqual(result["checkpoint"]["engine_digest"] == start.engine_digest,
                                     scenario != "accept")
                    if scenario != "accept":
                        self.assertNotIn("measure", backend.stages)
