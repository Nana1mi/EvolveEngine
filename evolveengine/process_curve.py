"""Secondary, public measurements of the five precommitted mainline budget points."""

from .journal import InsufficientBudget
from .schema import digest


def process_block(partitions, replicate):
    # Public like S, but a separate name/role from search. Same requests for both arms and E0.
    return partitions.block("process", "S_dev", f"replicate-{replicate}/budget-points", 8)


def measure_process_curve(controller, evaluator, submissions, reference, block):
    if block["partition"] != "S_dev":
        raise ValueError("Intermediate budget points may only be measured on public S_dev")
    journal = controller.journal
    lane = evaluator.fork("S_dev", evaluator.budget)
    saved = journal.get_event(lane.path + "/result")
    if saved is not None:
        return saved
    journal.begin_scope(lane.path)
    snapshots = {arm: sub["budget_points"] for arm, sub in submissions.items()}
    addresses = {p["engine_digest"] for points in snapshots.values() for p in points}
    observations = {}
    # One actual public measurement per distinct version within this pair, never a Q/T receipt.
    for address in (reference, *sorted(addresses - {reference})):
        key = f"measure/{address}"
        try:
            rate = controller.measure(lane, key, journal.get("engine", address), block)
            status = "ok" if rate is not None else "measurement_failed"
            receipt_id = f"{lane.path}/{key}"
        except InsufficientBudget:
            rate, receipt_id, status = None, None, "evaluation_budget_exhausted"
        observations[address] = {"status": status, "tokens_per_second": rate, "receipt_id": receipt_id}
    base = observations[reference]["tokens_per_second"]
    points = {}
    for arm, sequence in snapshots.items():
        points[arm] = []
        for snapshot in sequence:
            row = observations[snapshot["engine_digest"]]
            rate = row["tokens_per_second"]
            status = "reference_unavailable" if rate is not None and base is None else row["status"]
            points[arm].append({**snapshot, "tokens_per_second": rate, "status": status,
                                "speedup": rate / base if rate is not None and base is not None else None})
    result = {"reference": reference, "block_digest": digest(block), "count": len(block["prompts"]),
              "observations": observations, "points": points, "cost": journal.spent(lane.path).record()}
    journal.event(lane.path + "/result", result)
    return result
