from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evolveengine.experiment import MOCK_PROTOCOL, evaluate_pair, seal_pair
from evolveengine.journal import Journal
from evolveengine.mock import MockBackend
from evolveengine.partitions import Partitions
from evolveengine.schema import Cost, digest
from evolveengine.spark import SparkBackend


class ObservedBackend(MockBackend):
    def __init__(self):
        super().__init__()
        self.calls = []

    def run(self, stage, request, cap, node):
        self.calls.append((stage, request.get("block", {}).get("partition"), node))
        return super().run(stage, request, cap, node)


class SealedEvaluationTests(unittest.TestCase):
    def test_independent_key_and_detached_journal(self):
        engine = ObservedBackend()
        d_key, q_key = b"d" * 32, b"q" * 32
        commit_d = hashlib.sha256(d_key).hexdigest()
        commit_q = hashlib.sha256(q_key).hexdigest()
        development = Partitions(6, d_key, final_commitment=commit_q)
        judge = Partitions(6, final_salt=q_key, development_commitment=commit_d)
        with self.assertRaises(PermissionError):
            development.block("final", "Q", "rq1", 4)
        with self.assertRaises(PermissionError):
            judge.block("development", "D", "rq1", 4)
        with self.assertRaises(ValueError):
            Partitions(6, d_key, final_salt=d_key)
        with self.assertRaises(ValueError):
            Partitions(6, final_salt=b"z" * 32, final_commitment=commit_q)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with Journal(root / "optimization.sqlite") as search:
                bundle, report = seal_pair(search, engine, development, MOCK_PROTOCOL,
                                           ("s1", "s2", "s3", "s4"), 0, {"simulated": True},
                                           causal_points=1)
                self.assertNotIn("sealed_evaluation", report)
                self.assertTrue(all(pool not in {"Q", "T"} for _, pool, _ in engine.calls))
                remaining = bundle["payload"]["remaining_evaluation_budget"]["blocks"]
                self.assertEqual(remaining + report["causal_cost"]["blocks"], MOCK_PROTOCOL.evaluation.blocks)
                self.assertEqual(bundle["digest"], report["submission_digest"])
            detached = json.loads(json.dumps(bundle))
            proving = ObservedBackend()
            with Journal(root / "judge.sqlite") as evaluation:
                altered = deepcopy(detached)
                next(iter(altered["payload"]["engines"].values()))["edits"]["runtime.py"] = "rate = 1\n"
                with self.assertRaises(ValueError):
                    evaluate_pair(evaluation, proving, judge, MOCK_PROTOCOL, altered, report["submission_digest"])
                altered["digest"] = digest(altered["payload"])
                with self.assertRaises(ValueError):
                    evaluate_pair(evaluation, proving, judge, MOCK_PROTOCOL, altered, report["submission_digest"])
                self.assertEqual(proving.calls, [])
                scores = evaluate_pair(evaluation, proving, judge, MOCK_PROTOCOL, detached,
                                       report["submission_digest"])
                self.assertEqual(scores["sealed_evaluation"]["rq1"]["status"], "ok")
                self.assertEqual(scores["sealed_evaluation"]["rq2"][0]["status"], "ok")
                latency = scores["sealed_evaluation"]["rq1"]["engines"]["fixed"]["latency_ms"]
                self.assertEqual(latency["ttft"]["n"], 2 * MOCK_PROTOCOL.final_blocks * MOCK_PROTOCOL.final_count)
                self.assertTrue(proving.calls)
                private = [call for call in proving.calls if call[1] in {"Q", "T"}]
                public = [call for call in proving.calls if call[:2] == ("measure", "S_dev")]
                self.assertTrue(private and public)
                self.assertEqual(proving.calls, private + public)
                self.assertTrue(all(node == "s4" for _, _, node in proving.calls))
                measured = list(proving.calls)
                self.assertEqual(evaluate_pair(evaluation, proving, judge, MOCK_PROTOCOL, detached,
                                               report["submission_digest"]), scores)
                self.assertEqual(proving.calls, measured)

    def test_judge_backend_does_not_need_agent_or_engine_checkout(self):
        config = json.loads(Path("configs/spark.controller.example.json").read_text(encoding="utf-8"))
        evaluator = SparkBackend(config, evaluation_only=True)
        self.assertFalse(hasattr(evaluator, "pi"))
        self.assertFalse(hasattr(evaluator, "repository"))
        with self.assertRaises(PermissionError):
            evaluator.run("propose", {}, Cost(), config["nodes"][0])
        with self.assertRaises(PermissionError):
            evaluator.run("measure", {"block": {"partition": "S"}}, Cost(), config["nodes"][0])
