"""Offline causal invariants: real controller, journal and held-out judge, noisy mock timings."""
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from evolveengine.controller import Controller, Lane
from evolveengine.experiment import MOCK_PROTOCOL, deploy_pair, final_comparison
from evolveengine.journal import InsufficientBudget, Interrupted, Journal
from evolveengine.mock import HOST, RULE, SOURCE, MockBackend
from evolveengine.partitions import Partitions
from evolveengine.ranking import rank
from evolveengine.reports import difference, summarize
from evolveengine.schema import Checkpoint, Cost, Result


BIG = Cost(10000, 10000, 100000, blocks=1000)
NO_FIRE = {"rules": [{"when": {"phase": "prefill"}, "promote_action": "reduce_launch_gap"}]}
ROLES = ("parent", "proposed")


class Probe(MockBackend):
    def __init__(self, *, noisy=None, single_at_dev=False, late_flip=False):
        super().__init__()
        self.calls = Counter()
        self.noisy, self.single_at_dev, self.late_flip = noisy, single_at_dev, late_flip

    def run(self, stage, request, cap, node):
        partition = request.get("block", {}).get("partition")
        self.calls[stage, partition] += 1
        result = super().run(stage, request, cap, node)
        if stage == "measure" and partition == self.noisy:
            n = self.calls[stage, partition]
            if partition == "S":  # Same p0 patch with a true 1% gain, either side of 2 epsilon.
                factor = (0.98 if n <= 2 else 1.02) if self._state(request["engine"])[0] == 1 else 1
            else:
                factor = 1 + 0.01 * ((n * n) % 13)
            result.data["elapsed_seconds"] *= factor
        return result

    def _generate(self, request):
        data = super()._generate(request)
        if self.single_at_dev and self._state(request["engine"])[0] >= 10:
            data["candidates"] = data["candidates"][:1]
        return data

    def _profile(self, request):
        data = super()._profile(request)
        if self.late_flip and self._state(request["engine"])[0] == 0:
            data["observation"] = {**HOST, "host_launch_gap": "low"}
        return data


class CouplingTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = Path(tmp.name) / "journal.sqlite"
        self.partitions = Partitions(0, b"d" * 32, final_salt=b"t" * 32)
        self.lane = Lane("replicate-0/causal", "research_overhead", "node", BIG)

    def setup_pair(self, j, backend, *, epsilon=.002, rules=NO_FIRE):
        c = Controller(j, backend, self.partitions, {"simulated": True}, epsilon=epsilon)
        return c, c.checkpoint(backend.initial_engine(), "bank/C/0"), j.put("policy", rules)

    def rows(self, journal, key, role):
        prefix = f"{self.lane.path}/{key}/{role}/"
        return {row["id"][len(prefix):]: row for row in journal.budget_rows(prefix)}

    def test_registered_rule_can_flip_in_each_spark_observation(self):
        backend = Probe()
        a, b = backend._generate({"engine": backend.initial_engine()})["candidates"]
        b = {**b, "bottleneck_class": "compute"}
        for gap, primary in (("high", "host_dispatch"), ("low", "unknown"), ("unknown", "unknown")):
            with self.subTest(gap=gap):
                obs = {"primary_class": primary, "phase": "mixed", "host_launch_gap": gap}
                policy = {"rules": [{"when": obs, "promote_action": b["action_tag"]}]}
                self.assertEqual(rank(obs, [b, a], {"rules": []}), ["p0", "s0"])
                self.assertEqual(rank(obs, [a, b], policy), ["s0", "p0"])
                self.assertEqual(rank(obs, [a], policy), ["p0"])
                self.assertEqual(rank(obs, [a, b], NO_FIRE), ["p0", "s0"])

    def test_screen_noise_does_not_split_identical_choice(self):
        b = Probe(noisy="S")
        with Journal(self.db) as j:
            c, start, h = self.setup_pair(j, b, epsilon=.01)
            out = c.paired_steps(self.lane, "d-0", start, c.baseline, h, replace(BIG, blocks=2), 1)
            ends = [out[r]["checkpoint"] for r in ROLES]
            records = [j.get("experience", j.get("memory", end["memory_digest"])[-1]) for end in ends]
            self.assertEqual([x["candidate"]["candidate_id"] for x in records], ["p0", "p0"])
            self.assertEqual(records[0]["evidence"]["candidate_engine"], records[1]["evidence"]["candidate_engine"])
            self.assertEqual(ends[0]["engine_digest"], ends[1]["engine_digest"])
            self.assertEqual(records[0]["evidence"], records[1]["evidence"])
            self.assertEqual(b.calls["measure", "S"], 2)

    def test_equivalent_policies_share_all_steps_but_not_other_pairs(self):
        b = Probe()
        with Journal(self.db) as j:
            c, start, h = self.setup_pair(j, b, epsilon=.008)  # Both positive screens required.
            for fork in ("c-0", "c-1"):
                result = c.paired_steps(self.lane, fork, start, c.baseline, h, BIG, 3)
                left, right = (self.rows(j, fork, role) for role in ROLES)
                self.assertEqual(left.keys(), right.keys())
                self.assertIn("step-2/confirm/child", left)
                for key in left:
                    self.assertEqual(left[key]["input_hash"], right[key]["input_hash"], key)
                    self.assertEqual(left[key]["reserved"], right[key]["reserved"], key)
                    self.assertEqual(left[key]["receipt"], right[key]["receipt"], key)
                    self.assertEqual(left[key]["physical"] + right[key]["physical"], 1, key)
                a, z = (result[r] for r in ROLES)
                self.assertEqual(a["cost"], z["cost"])
                self.assertEqual(a["checkpoint"]["candidate_pool_digest"], z["checkpoint"]["candidate_pool_digest"])
                generator = j.get("pool", a["checkpoint"]["candidate_pool_digest"])["generator_input"]
                self.assertTrue(generator["memory"])
                self.assertTrue(all("attempt_id" not in record for record in generator["memory"]))
                for role in ROLES:  # Raw per-arm attribution is not normalized away.
                    mem = j.get("memory", result[role]["checkpoint"]["memory_digest"])
                    self.assertIn(f"/{fork}/{role}/step-2", j.get("experience", mem[-1])["attempt_id"])
            self.assertEqual(b.calls["implement", None], 6)
            self.assertEqual(b.calls["generate", None], 6)

    def test_unchanged_parent_is_profiled_afresh_in_the_next_logical_step(self):
        b = Probe()
        b.scenario = "quality_failure"
        with Journal(self.db) as j:
            c, start, h = self.setup_pair(j, b)
            c.paired_steps(self.lane, "retry", start, c.baseline, h, BIG, 2, memory_enabled=False)
            left, right = (self.rows(j, "retry", r) for r in ROLES)
            self.assertEqual(left["step-0/profile"]["input_hash"], left["step-1/profile"]["input_hash"])
            self.assertNotEqual(left["step-0/implement"]["input_hash"], left["step-1/implement"]["input_hash"])
            for n in range(2):
                for stage in ("profile", "implement"):
                    key = f"step-{n}/{stage}"
                    self.assertEqual(left[key]["physical"] + right[key]["physical"], 1)
            self.assertEqual(b.calls["implement", None], 2)

    def test_first_real_flip_splits_attempt_then_future_pools(self):
        b = Probe(late_flip=True)
        with Journal(self.db) as j:
            c, start, h = self.setup_pair(j, b, rules={"rules": [RULE]})
            c.paired_steps(self.lane, "fork", start, c.baseline, h, BIG, 3)
            a, z = (self.rows(j, "fork", r) for r in ROLES)
            for suffix, expected in (("step-0/implement", 1), ("step-1/generate", 1),
                                     ("step-1/implement", 2), ("step-2/profile", 2), ("step-2/generate", 2)):
                self.assertEqual(a[suffix]["physical"] + z[suffix]["physical"], expected, suffix)
            self.assertNotEqual(a["step-1/implement"]["input_hash"], z["step-1/implement"]["input_hash"])

    def test_known_receipt_cannot_bypass_reservation(self):
        class LargeCap(Probe):
            def cap(self, stage, *, partition="S"):
                return Cost(90, 90) if stage == "profile" else super().cap(stage, partition=partition)
        b = LargeCap()
        with Journal(self.db) as j:
            c, start, h = self.setup_pair(j, b)
            req = {"engine": b.initial_engine()}
            first = c._operate(self.lane, "source", "profile", req)
            self.assertEqual(first.cost.wall_ms, 2)
            with self.assertRaises(InsufficientBudget):
                c._operate(replace(self.lane, path="small", budget=Cost(10, 10)), "p", "profile", req)
            # Also check the receiving arm, even if the physical payer could afford the cap.
            receiver = replace(self.lane, path="test/fork/proposed", budget=Cost(10, 10))
            with self.assertRaises(InsufficientBudget):
                c._operate(replace(self.lane, path="test/fork/parent", peer=receiver), "p", "profile", req)
            self.assertFalse(list(j.budget_rows("test/")))
            # The common enclosing account cannot pre-authorize 2 x 90 ms.
            tight = replace(self.lane, path="tight", budget=Cost(100, 100, 1000))
            pair = c.paired_steps(tight, "fork", start, c.baseline, h, BIG, 1)
            self.assertTrue(all(row["cost"] == Cost().record() for row in pair.values()))
            self.assertEqual(b.calls["profile", None], 1)
            c.paired_steps(self.lane, "funded", start, c.baseline, h, BIG, 1)
            for r in ROLES:
                row = self.rows(j, "funded", r)["step-0/profile"]
                self.assertEqual((row["reserved"]["wall_ms"], row["receipt"]["cost"]["wall_ms"]), (90, 2))

    def test_coupling_cannot_cross_forks_nodes_or_partitions(self):
        left = self.lane.fork("c-0/parent", BIG)
        peer = self.lane.fork("c-0/proposed", BIG)
        for invalid in (replace(peer, path="replicate-1/causal/c-0/proposed"),
                        replace(peer, path="replicate-0/causal/c-1/proposed"),
                        replace(peer, path="replicate-0/development/d-0/proposed"),
                        replace(peer, node="another"), replace(peer, owner="another")):
            with self.assertRaises(ValueError):
                replace(left, peer=invalid)

    def test_waiting_tail_does_not_lose_its_wall_budget(self):
        now = [10**12]
        class Clock(Probe):
            def run(self, stage, req, cap, node):
                result = super().run(stage, req, cap, node)
                now[0] += result.cost.wall_ms * 10**6
                return result
        with patch("evolveengine.journal.time.time_ns", side_effect=lambda: now[0]), Journal(
                self.db, enforce_wall=True) as j:
            c, start, h = self.setup_pair(j, Clock(late_flip=True), rules={"rules": [RULE]})
            pair = c.paired_steps(self.lane, "clock", start, c.baseline, h, replace(BIG, wall_ms=40), 2)
            for r in ROLES:
                self.assertEqual(pair[r]["elapsed_wall_ms"], 40)
                self.assertEqual(j.get_event(f"{self.lane.path}/clock/{r}/step-1/finished")["outcome"], "mainline")

    def test_same_engine_after_divergence_does_not_recouple(self):
        b = Probe()
        b.scenario = "quality_failure"  # Distinct choices both fall back to the original engine.
        with Journal(self.db) as j:
            c, start, h = self.setup_pair(j, b, rules={"rules": [RULE]})
            endpoints = c.paired_steps(self.lane, "fail", start, c.baseline, h, BIG, 2)
            self.assertTrue(all(v["checkpoint"]["engine_digest"] == start.engine_digest for v in endpoints.values()))
            a, z = (self.rows(j, "fail", r) for r in ROLES)
            self.assertEqual(a["step-1/profile"]["physical"] + z["step-1/profile"]["physical"], 2)
            self.assertNotEqual(a["step-1/generate"]["input_hash"], z["step-1/generate"]["input_hash"])

    def test_interrupted_shared_operation_is_not_rerun_or_billed_again(self):
        b = Probe()
        def interrupt(key):
            if key.endswith("/step-1/implement"):
                raise Interrupted(key)
        with self.assertRaises(Interrupted), Journal(self.db, interrupt) as j:
            c, start, h = self.setup_pair(j, b)
            c.paired_steps(self.lane, "resume", start, c.baseline, h, BIG, 2)
        with Journal(self.db) as j:
            c, start, h = self.setup_pair(j, b)
            result = c.paired_steps(self.lane, "resume", start, c.baseline, h, BIG, 2)
            a, z = (self.rows(j, "resume", r)["step-1/implement"] for r in ROLES)
            self.assertEqual(a["receipt"], z["receipt"])
            self.assertEqual(a["receipt"]["status"], "uncertain")
            self.assertEqual(a["physical"] + z["physical"], 1)
            spent, calls = j.spent(self.lane.path), b.calls.copy()
            self.assertEqual(result, c.paired_steps(self.lane, "resume", start, c.baseline, h, BIG, 2))
            self.assertEqual((j.spent(self.lane.path), b.calls), (spent, calls))

    def test_same_engine_d_keeps_both_zeros_without_scoring(self):
        b = Probe(noisy="D", single_at_dev=True)
        with Journal(self.db) as j:
            c, start, _ = self.setup_pair(j, b)
            lane = replace(self.lane, path="replicate-0/main/evolving", owner="evolving")
            history = Checkpoint(**c.step(lane, "step-0", start)["checkpoint"])
            starts = [c.checkpoint({"base": b.version, "edits": {SOURCE: f"stage = {n}\nrate = 100.0\n"}},
                                   f"bank/D/{n}") for n in (10, 20)]
            decision = c.update(lane, "window-0", history, starts, MOCK_PROTOCOL.development, 1)
            self.assertTrue(any(r["first_choice_changed"] for r in decision["replay"]))
            self.assertEqual(b.calls["measure", "D"], 0)
            self.assertEqual([row["log_gain"] for row in decision["comparisons"]], [0., 0.])
            self.assertEqual(decision["status"], "rejected")
            self.assertEqual(b.calls["quality", "D"], 2)

    def test_identical_invalid_d_engine_is_not_a_legal_zero(self):
        b = Probe()
        with Journal(self.db) as j:
            c, _, _ = self.setup_pair(j, b)
            bad = c.checkpoint({"base": b.version, "edits": {SOURCE: "invalid"}}, "bank/D/0")
            score = c._score_development(self.lane, "d-0", {r: {"checkpoint": asdict(bad)} for r in ROLES})
            self.assertIsNone(score["log_gain"])
            self.assertEqual(b.calls["measure", "D"], 0)
            self.assertFalse(score["gates"]["proposed"]["receipt"]["passed"])

    def test_blind_rq2_keeps_zero_in_report_and_independent_rq1_measurements(self):
        b = Probe(noisy="T")
        p = replace(MOCK_PROTOCOL, windows_ms=(), main_steps=1, final_blocks=1, final_count=4)
        with Journal(self.db) as j:
            report = deploy_pair(j, b, self.partitions, p, ("1", "2", "3", "4"), 0,
                                 {"simulated": True}, causal_points=1, ablation=True)
            self.assertEqual(summarize([report])["rq2"]["paired_log_differences"], [0.])
            final = report["sealed_evaluation"]
            self.assertEqual(final["rq2"][0]["engines"]["parent"], final["rq2"][0]["engines"]["proposed"])
            self.assertTrue(final["rq2"][0]["engines"]["parent"]["latency_ms"]["ttft"]["n"])
            self.assertNotEqual(difference(final["rq1"], "fixed", "evolving"), 0.)
            self.assertEqual(b.calls["measure", "T"], 8 + 4 + 4)  # RQ1 unchanged; two isolated C groups.

    def test_identical_reference_still_gets_real_q_and_t(self):
        class Judge(Probe):
            fail = None
            def run(self, stage, request, cap, node):
                result = super().run(stage, request, cap, node)
                if request.get("block", {}).get("partition") == self.fail:
                    return Result({}, result.cost, "failed", "injected")
                return result
        b = Judge(noisy="T")
        p = replace(MOCK_PROTOCOL, final_blocks=2, final_count=4)
        with Journal(self.db) as j:
            c, ref, _ = self.setup_pair(j, b)
            group = {r: {"checkpoint": asdict(ref)} for r in ROLES}
            scores = final_comparison(c, self.lane, "self", ref, group, p)
            self.assertEqual(scores["engines"]["parent"]["speedup"], 1.)
            self.assertEqual(difference(scores, *ROLES), 0.)
            self.assertEqual((b.calls["quality", "Q"], b.calls["measure", "T"]), (1, p.final_blocks))
            aa = final_comparison(c, self.lane, "calibration", ref, group, p, reuse_identical=False)
            self.assertNotEqual(difference(aa, *ROLES), 0.)  # Independent A/A is still a noise probe.
            for partition, status in (("Q", "invalid_reference"), ("T", "invalid_reference_measurement")):
                b.fail = partition
                self.assertEqual(final_comparison(c, self.lane, partition, ref, group, p)["status"], status)
