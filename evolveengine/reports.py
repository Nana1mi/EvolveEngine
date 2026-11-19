"""Compare sealed experiment replicates, with one statistical unit per lineage."""

import math
from statistics import mean, median, quantiles, stdev

from .schema import Cost


# Two-sided 95% Student-t critical values for 1 through 30 degrees of freedom.
T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
       2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
       2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def paired_interval(values):
    output = {"n": len(values), "paired_log_differences": values}
    if not values:
        return output
    average = mean(values)
    output["geometric_advantage"] = math.expm1(average)
    if len(values) < 2:
        return output
    df = len(values) - 1
    # For larger degrees of freedom, a second-order expansion of the t quantile.
    z = 1.959963984540054
    critical = T95[df - 1] if df <= len(T95) else (
        z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
    )
    half_width = critical * stdev(values) / math.sqrt(len(values))
    output["ci95"] = [math.expm1(average - half_width), math.expm1(average + half_width)]
    return output


def interval(values):
    """Legacy v1 joint decision; v2.2 RQ-A uses only the interval for its test."""
    output = paired_interval(values)
    if "ci95" in output:
        output["meets_design_threshold"] = output["ci95"][0] > 0 and output["geometric_advantage"] >= 0.03
    return output


def difference(result, left, right, *, invalid_start_is_neutral=False):
    if result["status"] != "ok":
        return 0.0 if invalid_start_is_neutral and result["status"].startswith("invalid_reference") else None
    rows = result["engines"]
    return math.log(rows[right]["speedup"]) - math.log(rows[left]["speedup"])


def process_metrics(prefix, attempts, events, ledger):
    chosen = [row for row in attempts if row["attempt_id"].startswith(prefix + "/step-")]
    chosen.sort(key=lambda row: int(row["attempt_id"].rsplit("-", 1)[1]))
    costs = [row["cost"]["wall_ms"] for row in chosen]
    outcomes = {}
    curve = []
    gain = 0.0
    for row in chosen:
        outcome = row["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == "mainline":
            gain += row["evidence"]["log_gain"]
        operations = [entry for entry in ledger if entry["id"].startswith(row["attempt_id"] + "/")]
        if not operations:
            continue
        cutoff = max(entry["settled_ns"] for entry in operations)
        spent = Cost()
        for entry in ledger:
            if entry["id"].startswith(prefix + "/") and entry["settled_ns"] <= cutoff:
                spent += Cost(**entry["receipt"]["cost"])
        curve.append({"attempt": row["attempt_id"], "outcome": outcome,
                      "cumulated_accepted_screen_speedup": math.exp(gain), "cost": spent.record()})
    fidelity = [item["evidence"]["fidelity"]["status"] for item in chosen if "fidelity" in item["evidence"]]
    updates = [item for item in events if item["id"].startswith(prefix + "/window-") and item["id"].endswith("/decided")]
    replay_rows = [item for update in updates for item in update.get("replay", [])]
    update_cost = Cost()
    for entry in ledger:
        if entry["id"].startswith(prefix + "/window-"):
            update_cost += Cost(**entry["receipt"]["cost"])
    first = next((row["cost"] for row in curve if row["outcome"] == "mainline"), None)
    draws = [row["counts"] for row in events if row["id"].startswith(prefix + "/step-")
             and "/draw-" in row["id"] and "counts" in row]
    filtered = {key: sum(row[key] for row in draws)
                for key in ("raw", "out_of_scope", "pool_duplicates", "previously_tried", "eligible")}
    return {"step_count": len(chosen), "median_step_ms": median(costs) if costs else None,
            "outcomes": outcomes, "pool_filter_counts": filtered, "empty_draws": sum(not row["eligible"] for row in draws),
            "fidelity_review_count": len(fidelity),
            "fidelity_pass_rate": fidelity.count("pass") / len(fidelity) if fidelity else None,
            "first_improvement_cost": first, "screening_trace": curve,
            "update_decisions": [item["status"] for item in updates], "update_cost": update_cost.record(),
            "replayed_pool_count": len(replay_rows),
            "first_choice_flip_rate": (mean(row["first_choice_changed"] for row in replay_rows)
                                       if replay_rows else None)}


def latency_metrics(requests):
    result = {}
    for name in ("ttft", "tpot"):
        samples = [row.get(name + "_seconds") for row in requests]
        milliseconds = [value * 1000 for value in samples
                        if type(value) in (float, int) and math.isfinite(value) and value >= 0]
        if milliseconds:
            point = milliseconds[0] if len(milliseconds) == 1 else quantiles(milliseconds, n=20, method="inclusive")[-1]
            result[name] = {"n": len(milliseconds), "p50": median(milliseconds), "p95": point}
    return result


def summarize(reports):
    protocol = None
    replicates = set()
    simulated = None
    samples = {name: [] for name in ("rq1", "rq2", "policy_memory_off", "memory_at_H0",
                                    "memory_at_H_star", "memory_interaction")}
    failures = {"fixed": 0, "evolving": 0, "invalid_causal_reference": 0}
    for report in reports:
        manifest = report["run_manifest"]
        if manifest.get("endpoint"):
            raise ValueError("Expected v1 paired reports; use summarize-rqa for the frozen-policy endpoint")
        simulated = manifest.get("simulated", simulated)
        signature = (manifest.get("harness_revision"), manifest["protocol"],
                     manifest["initial_engine"], manifest["initial_policy"],
                     manifest["environment"], manifest["causal_points"], manifest["ablation"])
        if protocol is None:
            protocol = signature
        if protocol != signature or manifest["replicate"] in replicates:
            raise ValueError("Reports need a shared harness revision, protocol/environment and unique replicate IDs")
        replicates.add(manifest["replicate"])
        evaluated = report["sealed_evaluation"]
        rq1 = difference(evaluated["rq1"], "fixed", "evolving")
        if rq1 is not None:
            samples["rq1"].append(rq1)
            for arm in ("fixed", "evolving"):
                failures[arm] += evaluated["rq1"]["engines"][arm]["status"] != "valid"
        rq2 = [difference(item, "parent", "proposed", invalid_start_is_neutral=True) for item in evaluated["rq2"]]
        failures["invalid_causal_reference"] += sum(item["status"].startswith("invalid_reference") for item in evaluated["rq2"])
        if len(rq2) == manifest["causal_points"] and None not in rq2:
            samples["rq2"].append(mean(rq2))
            empty = [difference(item, "parent", "proposed", invalid_start_is_neutral=True)
                     for item in evaluated["rq3_memory_off"]]
            if len(empty) == len(rq2) and None not in empty:
                samples["policy_memory_off"].append(mean(empty))
                samples["memory_interaction"].append(mean(rq2) - mean(empty))
                off = evaluated["rq3_memory_off"]
                on = evaluated["rq2"]
                if all(item["status"] == "ok" for item in [*on, *off]):
                    for label, key in (("parent", "memory_at_H0"), ("proposed", "memory_at_H_star")):
                        samples[key].append(mean(math.log(a["engines"][label]["speedup"] /
                                                          b["engines"][label]["speedup"])
                                                 for a, b in zip(on, off)))
    result = {name: interval(values) for name, values in samples.items()}
    result["replicate_count"] = len(replicates)
    result["simulated"] = simulated
    result["failures"] = failures
    result["both_confirmatory_endpoints_pass"] = simulated is False and all(
        result[name].get("meets_design_threshold", False) for name in ("rq1", "rq2")
    )
    return result
