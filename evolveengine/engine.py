"""Engine snapshots, patch scope checks and development measurement."""

import difflib
import math

from .schema import EvaluationReceipt, PatchReceipt, in_scope, positive


def check_patch(candidate, implementation, allowed_scope, source):
    receipt = PatchReceipt(**implementation["receipt"])
    if receipt.selected_id != candidate.candidate_id or receipt.action_tag != candidate.action_tag:
        return "deviation", "implementation switched the selected action", ""
    if receipt.status != "implemented":
        if receipt.edits:
            return "deviation", "non-implementation returned edits", ""
        return receipt.status, receipt.reason, ""
    if not receipt.edits or not receipt.intent.strip():
        return "uncertain", "missing patch or intent", ""
    changes = []
    for path, after in sorted(receipt.edits.items()):
        if not in_scope(path, allowed_scope) or not in_scope(path, candidate.target_scope):
            return "deviation", f"changed unselected path: {path}", ""
        before = source(path)
        if before is None:
            return "deviation", f"attempt to add unregistered file: {path}", ""
        changes.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{path}", tofile=f"b/{path}",
        ))
    diff = "".join(changes)
    if not diff:
        return "uncertain", "empty diff", ""
    verified_action = implementation.get("verified_action")
    if not verified_action:
        return "uncertain", "mechanism requires a matching fidelity audit", diff
    if verified_action != candidate.action_tag:
        return "deviation", "independent mechanism audit disagrees", diff
    return "pass", "", diff


def apply_edits(engine, edits):
    return {"base": engine["base"], "edits": {**engine["edits"], **edits}}


def gate(result):
    if result.status != "ok":
        return False
    return EvaluationReceipt(**result.data["receipt"]).passed


def throughput(data, block):
    rows = data["requests"]
    if len(rows) != len(block["prompts"]) or not positive(data["elapsed_seconds"]):
        raise ValueError("Timing block did not return one complete receipt per request")
    for row, prompt in zip(rows, block["prompts"]):
        if (row["input_tokens"] != len(prompt) or row["output_tokens"] != block["max_new_tokens"]
                or row["finish_reason"] != "length" or not positive(row["elapsed_seconds"])):
            raise ValueError("Timing receipt violates the workload")
    return sum(row["output_tokens"] for row in rows) / data["elapsed_seconds"]


def compare_rates(parent, child):
    if not positive(parent) or not positive(child):
        raise ValueError("Cannot compare invalid throughput")
    return math.log(child / parent)
