"""Pre-pilot invariants: common exclusions, paid redraws and sealed process measurements."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evolveengine.controller import Controller, Lane
from evolveengine.experiment import MOCK_PROTOCOL, evaluate_pair, seal_pair
from evolveengine.journal import Interrupted, Journal
from evolveengine.mock import MockBackend, RULE
from evolveengine.partitions import Partitions
from evolveengine.process_curve import measure_process_curve, process_block
from evolveengine.schema import Checkpoint, Cost, Result, digest


BIG = Cost(1000, 1000, 50000, blocks=100)
NODES = ("s1", "s2", "s3", "s4")


class Draws(MockBackend):
    def __init__(self, script):
        super().__init__("quality_failure")  # Parent unchanged across failed attempts.
        self.script = iter(script)
        self.inputs = []

    def _generate(self, request):
        self.inputs.append(request)
        a, b = super()._generate(request)["candidates"]
        draws = {"first": [a], "second": [b], "both": [a, b], "empty": [], "malformed": [a, a],
                 "filtered": [a, {**a, "candidate_id": "z"}, {**b, "target_scope": ["outside.py"]}]}
        return {"candidates": draws[next(self.script)]}


class PreparationTests(unittest.TestCase):
    def test_exclusions_are_local_outcome_free_and_independent_of_memory(self):
        exclude = [{"action_tag": "pack_request_metadata", "target_scope": ["runtime.py"]}]
        for enabled in (True, False):
            with self.subTest(enabled=enabled), TemporaryDirectory() as root, Journal(Path(root) / "j") as j:
                b = Draws(["first", "filtered", "second", "first"])
                c = Controller(j, b, Partitions(1), {})
                lane = Lane("arm", "fixed", "s1", BIG)
                start = c.checkpoint(b.initial_engine(), "arm")
                first = c.step(lane, "step-0", start, memory_enabled=enabled)
                end = c.step(lane, "step-1", Checkpoint(**first["checkpoint"]), memory_enabled=enabled)
                c.step(Lane("other", "fixed", "s1", BIG), "step-0", start, memory_enabled=enabled)
                self.assertEqual([r["excluded"] for r in b.inputs], [[], exclude, exclude, []])
                self.assertEqual(bool(b.inputs[1]["memory"]), enabled)
                self.assertEqual(j.get("experience", end["experience"])["candidate"]["candidate_id"], "s0")
                draw = j.get_event("arm/step-1/draw-0")
                self.assertEqual(draw["counts"], {"raw": 3, "out_of_scope": 1, "pool_duplicates": 1,
                                                  "previously_tried": 1, "eligible": 0})
                self.assertEqual(sum(e["id"].endswith("/pool_ready") for e in j.events("arm/step-1/")), 1)
                self.assertEqual(j.spent("arm/step-1").tokens, 2 * b.cap("generate").tokens + b.cap("implement").tokens)

    def test_diverged_arms_on_the_same_engine_never_read_each_others_exclusions(self):
        with TemporaryDirectory() as root, Journal(Path(root) / "j") as j:
            b = Draws(["both"] * 3)
            c = Controller(j, b, Partitions(1), {})
            start = c.checkpoint(b.initial_engine(), "c")
            c.paired_steps(Lane("c", "research_overhead", "s3", BIG), "pair", start, c.baseline,
                           j.put("policy", {"rules": [RULE]}), BIG, 2, memory_enabled=False)
            self.assertEqual(len(b.inputs), 3)  # One common draw, then two private draws.
            for role, action in (("parent", "pack_request_metadata"), ("proposed", "reduce_launch_gap")):
                row = j.get_event(f"c/pair/{role}/step-1/pool_ready")
                request = j.get("pool", row["pool_digest"])["generator_input"]
                self.assertEqual(request["memory"], [])
                self.assertEqual(request["engine"], b.initial_engine())
                self.assertEqual(request["excluded"], [{"action_tag": action, "target_scope": ["runtime.py"]}])

    def test_empty_redraw_has_a_hard_limit_and_mirrored_cost(self):
        with TemporaryDirectory() as root, Journal(Path(root) / "j") as j:
            b = Draws(["empty", "empty"])
            c = Controller(j, b, Partitions(1), {})
            start = c.checkpoint(b.initial_engine(), "c")
            ends = c.paired_steps(Lane("c", "research_overhead", "s3", BIG), "pair", start,
                                  c.baseline, c.baseline, BIG, 5, memory_enabled=False)
            self.assertEqual([r["draw_index"] for r in b.inputs], [0, 1])
            for role, end in ends.items():
                self.assertEqual(end["steps"], 1)
                self.assertEqual(end["cost"]["input_tokens"], 2 * b.cap("generate").input_tokens)
                self.assertEqual(j.get_event(f"c/pair/{role}/step-0/finished")["outcome"], "empty_pool")
            for op in ("profile", "generate", "generate-retry-1"):
                rows = [r for r in j.budget_rows() if r["id"].endswith("/" + op)]
                self.assertEqual(len(rows), 2)
                self.assertEqual(sum(r["physical"] for r in rows), 1)

    def test_budget_denial_and_invalid_response_are_not_empty_pools(self):
        for script, tokens, expected in [(["empty"], 200, "budget_stopped"),
                                         (["malformed"], 1000, "invalid_candidate_pool")]:
            with self.subTest(expected=expected), TemporaryDirectory() as root, Journal(Path(root) / "j") as j:
                b = Draws(script)
                c = Controller(j, b, Partitions(1), {})
                end = c.step(Lane("arm", "fixed", "s1", Cost(40, 40, tokens, blocks=2)), "step-0",
                             c.checkpoint(b.initial_engine(), "arm"))
                self.assertEqual(end["outcome"], expected)
                self.assertEqual(len(b.inputs), 1)
                self.assertEqual(j.spent("arm").tokens, 200)
                self.assertEqual(j.get_event("arm/step-0/pool_ready") is not None, expected == "budget_stopped")

    def test_interruption_at_redraw_preserves_the_first_draw_and_bills_only_once(self):
        def interrupt(key):
            if key.endswith("/generate-retry-1"):
                raise Interrupted(key)

        b = Draws(["empty"])
        lane = Lane("arm", "fixed", "s1", BIG)
        with TemporaryDirectory() as root:
            path = Path(root) / "j"
            with Journal(path, on_reserved=interrupt) as j:
                c = Controller(j, b, Partitions(1), {})
                start = c.checkpoint(b.initial_engine(), "arm")
                with self.assertRaises(Interrupted):
                    c.step(lane, "step-0", start)
            with Journal(path) as j:
                c = Controller(j, b, Partitions(1), {})
                end = c.step(lane, "step-0", start)
                self.assertEqual(end["outcome"], "generate_uncertain")
                self.assertEqual(j.spent("arm").tokens, 2 * b.cap("generate").tokens)
                self.assertEqual(len(b.inputs), 1)
                self.assertEqual(c.step(lane, "step-0", start), end)


class BudgetPointTests(unittest.TestCase):
    def test_points_use_admission_time_not_the_end_of_an_inflight_step(self):
        for limit, expected in [(100, [0, 1, 2, 2, 2]), (80, [0, 1, 2, 2, 2]), (60, [0, 0, 1, 2, 2])]:
            with self.subTest(limit=limit), TemporaryDirectory() as root, Journal(Path(root) / "j") as j:
                b = MockBackend()
                c = Controller(j, b, Partitions(1), {})
                lane = Lane("arm", "fixed", "s1", Cost(limit, 100, 8000, blocks=12))
                start = c.checkpoint(b.initial_engine(), "arm")
                end = c.run(lane, start, 2, track_budget_points=True)
                points = end["budget_points"]
                self.assertEqual(points, [j.get_event(f"arm/run/budget-point-{i}") for i in range(5)])
                self.assertEqual([p["wall_ms"] for p in points], [q * limit // 4 for q in range(5)])
                self.assertEqual([b._state(j.get("engine", p["engine_digest"]))[0] for p in points], expected)
                self.assertTrue(all(p["available_wall_ms"] <= p["wall_ms"] for p in points))
                self.assertEqual(points[-1]["engine_digest"], end["checkpoint"]["engine_digest"])

    def test_persisted_admissions_survive_restart_and_late_decisions_do_not_backfill(self):
        for late in (False, True):
            now = [0]

            class ClockBackend(MockBackend):
                def _measure(self, req):
                    now[0] += 55 if late else 19  # Decision at 110 (late) or 38 ms.
                    return super()._measure(req)

            clock = patch("evolveengine.journal.time.time_ns", side_effect=lambda: now[0] * 10**6)
            with self.subTest(late=late), TemporaryDirectory() as root, clock:
                db = Path(root) / "j"
                lane = Lane("arm", "fixed", "s1", Cost(100, 100, 8000, blocks=12))
                b = ClockBackend()
                with Journal(db, enforce_wall=True) as j:
                    c = Controller(j, b, Partitions(1), {})
                    start = c.checkpoint(b.initial_engine(), "arm")
                    write = j.event

                    def stop_at_finish(key, data):
                        if key == "arm/step-0/finished":  # After admission; before durable step completion.
                            raise Interrupted(key)
                        write(key, data)

                    with patch.object(j, "event", side_effect=stop_at_finish), self.assertRaises(Interrupted):
                        c.run(lane, start, 1, track_budget_points=True)
                now[0] = 200
                with Journal(db, enforce_wall=True) as j:
                    c = Controller(j, b, Partitions(1), {})
                    end = c.run(lane, start, 1, track_budget_points=True)
                    points = end["budget_points"]
                    self.assertEqual(j.spent("arm").blocks, 2)
                    self.assertEqual([p["available_wall_ms"] for p in points], [0] * 5 if late else [0, 0, 38, 38, 38])
                    self.assertEqual(points[-1]["engine_digest"], end["checkpoint"]["engine_digest"])
                    self.assertEqual(end["checkpoint"]["engine_digest"] == start.engine_digest, late)

    def test_public_process_curve_is_measured_after_seal_not_inherited_from_screen(self):
        class Judge(MockBackend):
            def __init__(self):
                super().__init__()
                self.calls = []

            def _measure(self, request):
                block = request["block"]
                address = digest({"kind": "engine", "content": request["engine"]})
                self.calls.append((block["partition"], digest(block), address))
                data = super()._measure(request)
                if block["partition"] == "S_dev" and request["engine"]["edits"]:
                    data["elapsed_seconds"] *= 2  # The independent curve must be allowed to go down.
                return data

        parts = Partitions(3, b"d" * 32, final_salt=b"q" * 32)
        with TemporaryDirectory() as root:
            with Journal(Path(root) / "search") as j:
                b = Judge()
                bundle, search = seal_pair(j, b, parts, MOCK_PROTOCOL, NODES, 0, {}, causal_points=1)
                self.assertNotIn("S_dev", [x[0] for x in b.calls])
                self.assertNotIn("S_dev", search)
                self.assertEqual(j.spent("replicate-0/main/fixed").record(), search["submission"]["fixed"]["cost"])
                # The D window crosses budget boundaries; D's private engines are never mainline snapshots.
                points = search["submission"]["evolving"]["budget_points"]
                self.assertEqual(points[1]["source_event"], "replicate-0/main/evolving/step-0/admission")
                groups = bundle["payload"]["comparisons"]
                private_targets = set()
                for group in [groups["rq1"], *groups["rq2"]]:
                    private_targets.add(group["reference"]["engine_digest"])
                    private_targets.update(sub["checkpoint"]["engine_digest"] for sub in group["submissions"].values())
            b = Judge()
            with Journal(Path(root) / "judge") as j:
                result = evaluate_pair(j, b, parts, MOCK_PROTOCOL, bundle, bundle["digest"])
                replay = result["S_dev"]
                measured = [r for r in b.calls if r[0] == "S_dev"]
                self.assertEqual({r[1] for r in measured}, {bundle["payload"]["run_manifest"]["S_dev_block_digest"]})
                self.assertEqual(b.calls[-len(measured):], measured)
                self.assertTrue(all(r[2] in private_targets for r in b.calls if r[0] == "T"))
                seen = {p["engine_digest"] for arm in replay["points"].values() for p in arm}
                self.assertEqual({r[2] for r in measured}, seen)
                self.assertEqual(len(measured), len(seen))
                self.assertEqual(replay["cost"]["blocks"], len(seen))
                self.assertTrue(all(p["speedup"] < 1 for p in replay["points"]["fixed"] if p["available_wall_ms"] > 0))
                self.assertEqual(replay["points"]["fixed"][0]["speedup"], 1.0)
                self.assertEqual(result["sealed_evaluation"]["rq1"]["status"], "ok")
                n = len(b.calls)
                self.assertEqual(evaluate_pair(j, b, parts, MOCK_PROTOCOL, bundle, bundle["digest"]), result)
                self.assertEqual(len(b.calls), n)

    def test_missing_reference_and_exhausted_replay_budget_do_not_invent_a_speedup(self):
        class NoReference(MockBackend):
            def run(self, stage, request, cap, node):
                if not request["engine"]["edits"]:
                    return Result({}, cap, "failed", "no public baseline measurement")
                return super().run(stage, request, cap, node)

        for blocks, expected in [(0, "evaluation_budget_exhausted"), (2, "reference_unavailable")]:
            with self.subTest(blocks=blocks), TemporaryDirectory() as root, Journal(Path(root) / "j") as j:
                b = NoReference()
                parts = Partitions(1)
                c = Controller(j, b, parts, {})
                base = b.initial_engine()
                ref = j.put("engine", base)
                child = j.put("engine", {**base, "edits": {"runtime.py": "stage = 1\nrate = 105.0\n"}})
                short = {"budget_points": [{"engine_digest": ref}, {"engine_digest": child}]}
                lane = Lane("judge", "research_overhead", "s4", Cost(100, 100, blocks=blocks))
                with self.assertRaisesRegex(ValueError, "public S_dev"):
                    measure_process_curve(c, lane, {"fixed": short}, ref, {"partition": "T"})
                self.assertEqual(j.spent("judge"), Cost())
                result = measure_process_curve(c, lane, {"fixed": short}, ref, process_block(parts, 0))
                self.assertEqual(result["cost"]["blocks"], blocks)
                self.assertEqual(result["points"]["fixed"][-1]["status"], expected)
                self.assertTrue(all(p["speedup"] is None for p in result["points"]["fixed"]))
