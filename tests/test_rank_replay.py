from pathlib import Path
import runpy
import unittest

from evolveengine.mock import HOST, RULE, MockBackend
from evolveengine.ranking import freeze_pool, rank


class RankTests(unittest.TestCase):
    def test_matches_the_original_replay(self):
        example = runpy.run_path(str(next(Path("研究方案").glob("04*.py"))))
        fixture = example["EXAMPLE"]
        output = example["replay"](fixture)
        mock = MockBackend()
        pool = mock._generate({"engine": mock.initial_engine()})["candidates"]
        self.assertEqual(rank(HOST, pool, {"rules": []}), [name + "0" for name in output["h0_order"]])
        self.assertEqual(rank(HOST, list(reversed(pool)), {"rules": [RULE]}),
                         [name + "0" for name in output["proposed_order"]])
        self.assertEqual(rank({**HOST, "host_launch_gap": "unknown"}, pool, {"rules": [RULE]}), ["p0", "s0"])

    def test_refuses_rule_identity_and_duplicate_candidates(self):
        backend = MockBackend()
        pool = backend._generate({"engine": backend.initial_engine()})["candidates"]
        with self.assertRaises(ValueError):
            rank(HOST, pool, {"rules": [{"when": {"candidate_id": "p0"}, "promote_action": "reduce_launch_gap"}]})
        with self.assertRaises(ValueError):
            freeze_pool([pool[0], pool[0]], backend.allowed_scope)
        self.assertEqual(freeze_pool([{**pool[0], "target_scope": ["verifier.py"]}], backend.allowed_scope), [])
