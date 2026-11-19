import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evolveengine.controller import Controller, Lane
from evolveengine.experiment import seal_pair
from evolveengine.journal import Journal
from evolveengine.mock import MockBackend
from evolveengine.partitions import Partitions
from evolveengine.schema import Cost, Result
from evolveengine.spark import PREFLIGHT_SCHEMA, SparkBackend
from evolveengine.spark_worker import quality_prompts


class SparkProbe(SparkBackend):
    def __init__(self, config, fail_count=None):
        super().__init__(config, evaluation_only=True)
        self.calls = []
        self.mock = MockBackend()
        self.fail_count = fail_count

    def initial_engine(self):
        return self.mock.initial_engine()

    def _preflight_agent(self):
        return {"metered_probe": [{"usage": {"prompt_tokens": 15, "completion_tokens": 5}}]}

    def _remote(self, stage, request, cap, node):
        block = request["block"]
        count = len(block["prompts"])
        self.calls.append((stage, count, block["partition"], cap))
        if stage == "preflight":
            return Result({
                "receipt": {"passed": True, "checks": {"full_vocab": True}},
                "negative_controls": {"truncated_vocab_rejected": True},
                "sensor_before": {"uuid": node, "driver": "fixture"},
                "runtime": {"capability": [12, 1]}, "image": "fixture", "engine_revision": "fixture",
                "model_digest": "fixture", "profile": {"kineto_gpu_activity_count": 1},
            }, Cost(10, 10))
        if count == self.fail_count:
            return Result({}, cap, "timeout", "simulated measurement deadline")
        measured = self.mock.run("measure", request, cap, node)
        return Result(measured.data, Cost(count * 10, count * 10, blocks=1))


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("configs/spark.controller.example.json").read_text(encoding="utf-8"))
        cls.partitions = Partitions(12, b"s" * 32, final_salt=b"t" * 32)

    def test_small_real_node_pools_are_allowed_without_weakening_v1(self):
        for nodes in (["gb-a"], ["gb-a", "gb-b"]):
            with self.subTest(nodes=nodes):
                probe = SparkProbe({**self.config, "nodes": nodes})
                receipt = probe.pilot_preflight(self.partitions.development_view())
                self.assertEqual(list(receipt["nodes"]), nodes)
                with self.assertRaisesRegex(ValueError, "Four distinct"):
                    seal_pair(None, probe, None, None, nodes, 0, {})
        for bad in ([], ["same", "same"], "abcd"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SparkProbe({**self.config, "nodes": bad})

    def test_invalid_target_set_rejected_before_metered_probe(self):
        probe = SparkProbe(self.config)
        with patch.object(probe, "_preflight_agent") as agent:
            for targets in ([], ["outside"], ["spark-1", "spark-1"]):
                with self.subTest(targets=targets), self.assertRaises(ValueError):
                    probe.pilot_preflight(self.partitions.development_view(), targets)
            agent.assert_not_called()
            self.assertFalse(probe.calls)

    def test_calibrates_all_shapes_with_public_data_and_real_caps(self):
        probe = SparkProbe(self.config)
        result = probe.pilot_preflight(self.partitions.development_view(), ["spark-1"])
        self.assertEqual(result["schema"], PREFLIGHT_SCHEMA)
        timing_calls = probe.calls[1:]
        sizes = [8, 8, 16, 16, 64, 64]
        self.assertEqual([row[1] for row in timing_calls], sizes)
        self.assertEqual([row[2] for row in timing_calls], ["S"] * len(sizes))
        self.assertEqual([row[3].wall_ms for row in timing_calls], [105000] * 2 + [150000] * 2 + [360000] * 2)
        node = result["nodes"]["spark-1"]
        self.assertEqual(node["cost"]["blocks"], 6)
        self.assertTrue(node["negative_controls"]["short_response_rejected"])
        self.assertEqual(node["calibration"]["final"]["elapsed_ms"], [640, 640])
        diagnostics = result["reservation_diagnostics"]
        self.assertEqual(diagnostics["one_harness_update"]["wall_ms"], 5220000)
        self.assertTrue(diagnostics["one_step_first_screen_fits"]["development"])
        self.assertFalse(diagnostics["one_step_ceiling_fits"]["development"])

    def test_small_blocks_cannot_hide_large_block_timeout(self):
        probe = SparkProbe(self.config, fail_count=64)
        with self.assertRaisesRegex(ValueError, r"final calibration \(64 requests\).+deadline"):
            probe.pilot_preflight(self.partitions.development_view(), ["spark-1"])

    def test_final_job_debits_its_registered_cap_in_the_controller(self):
        probe = SparkProbe(self.config)
        block = self.partitions.block("final", "T", "judge/a", 64)
        with TemporaryDirectory() as directory, Journal(Path(directory) / "judge.sqlite") as journal:
            controller = Controller(journal, probe, self.partitions.final_view(), {"simulated": True})
            lane = Lane("judge", "research_overhead", "spark-1", Cost(360000, 360000, blocks=1))
            rate = controller.measure(lane, "block", probe.initial_engine(), block)
            self.assertIsNotNone(rate)
            row, = tuple(journal.budget_rows())
            self.assertEqual(row["reserved"]["wall_ms"], 360000)
            self.assertEqual(row["receipt"]["cost"]["blocks"], 1)

    def test_quality_sample_preserves_coverage_regardless_of_input_order(self):
        prompts = [[3] * 512] * 4 + [[7] * 2048] * 2 + [[11] * 128] * 2
        self.assertEqual([len(prompt) for prompt in quality_prompts(prompts, 4)], [128, 512, 2048, 512])
