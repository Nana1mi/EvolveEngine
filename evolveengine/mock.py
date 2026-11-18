"""Deterministic backend for exercising controller behavior without API or GPU."""

from dataclasses import asdict
import re

from .schema import Cost, EvaluationReceipt, PatchReceipt, Result


SOURCE = "runtime.py"
BASE_TEXT = "stage = 0\nrate = 100.0\n"
HOST = {"primary_class": "host_dispatch", "phase": "decode", "host_launch_gap": "high"}
RULE = {"when": HOST, "promote_action": "reduce_launch_gap"}
COSTS = {
    "profile": Cost(2, 2),
    "generate": Cost(3, 3, 120, 80),
    "implement": Cost(6, 6, 280, 120),
    "quality": Cost(3, 3),
    "measure": Cost(3, 3, blocks=1),
    "propose": Cost(4, 4, 210, 110),
}


class MockBackend:
    version = "mock-v1"
    allowed_scope = (SOURCE,)

    def __init__(self, scenario="accept"):
        if scenario not in {"accept", "reject", "deviation", "quality_failure"}:
            raise ValueError("Unknown mock scenario")
        self.scenario = scenario

    def initial_engine(self):
        return {"base": self.version, "edits": {}}

    def read_source(self, engine, path):
        if path == SOURCE:
            return engine["edits"].get(path, BASE_TEXT)
        return None

    def cap(self, stage):
        return COSTS[stage]

    def run(self, stage, request, cap, node):
        method = getattr(self, f"_{stage}")
        return Result(method(request), COSTS[stage])

    def _state(self, engine):
        text = self.read_source(engine, SOURCE)
        match = re.fullmatch(r"stage = (\d+)\nrate = ([\d.]+)\n", text)
        return (int(match[1]), float(match[2])) if match else (-1, 0)

    def _profile(self, request):
        return {"observation": HOST, "evidence": {"source": self.version, "launch_gap_fraction": 0.42}}

    def _generate(self, request):
        stage, _ = self._state(request["engine"])
        common = {"bottleneck_class": "host_dispatch", "target_scope": [SOURCE],
                  "infeasible_conditions": "Required dispatch path is absent"}
        cheap = {**common, "candidate_id": f"p{stage}", "action_tag": "pack_request_metadata",
                 "cost_band": "low", "hypothesis": "Reduce metadata preparation time",
                 "expected_signal": "Metadata duration"}
        large = {**common, "candidate_id": f"s{stage}", "action_tag": "reduce_launch_gap",
                 "cost_band": "medium", "hypothesis": "Reduce idle intervals between launches",
                 "expected_signal": "Decode launch gap"}
        return {"candidates": [cheap, large]}

    def _implement(self, request):
        choice = request["contract"]["candidate"]
        stage, current = self._state(request["engine"])
        launch = choice["action_tag"] == "reduce_launch_gap"
        factor = 1.01
        if launch:
            factor = 1.005 if self.scenario == "reject" else 1.065
        content = f"stage = {stage + 1}\nrate = {current * factor:.8f}\n"
        if self.scenario == "quality_failure":
            content = "invalid code from a simulated edit\n"
        path = "verifier.py" if self.scenario == "deviation" else SOURCE
        patch = PatchReceipt("implemented", choice["candidate_id"], choice["action_tag"],
                             choice["hypothesis"], {path: content})
        return {"receipt": asdict(patch), "prompt": request["contract"],
                "tool_trace": [{"tool": "edit", "path": path, "status": "ok"}],
                "verified_action": choice["action_tag"]}

    def _quality(self, request):
        stage, _ = self._state(request["engine"])
        valid = stage >= 0
        checks = dict.fromkeys(("build", "full_vocab", "eos", "kv_isolation", "output_length"), valid)
        return {"receipt": asdict(EvaluationReceipt(valid, checks))}

    def _measure(self, request):
        _, rate = self._state(request["engine"])
        block = request["block"]
        output = block["max_new_tokens"]
        seconds = output / rate
        return {
            "elapsed_seconds": seconds * len(block["prompts"]),
            "requests": [
                {"input_tokens": len(prompt), "output_tokens": output, "elapsed_seconds": seconds,
                 "finish_reason": "length", "ttft_seconds": seconds / output,
                 "tpot_seconds": seconds / output}
                for prompt in block["prompts"]
            ],
        }

    def _propose(self, request):
        evidence = request["experience"]
        rules = request["policy"]["rules"]
        return {"policy": {"rules": [RULE, *rules] if RULE not in rules else rules},
                "basis": [item["attempt_id"] for item in evidence],
                "hypothesis": "Try launch-gap work ahead of metadata work when host dispatch dominates"}
