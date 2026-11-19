"""Run a fixed/evolving pair with separately funded validation and evaluation."""

from dataclasses import asdict, dataclass
import math
import statistics

from .controller import HARNESS_REVISION, Controller, Lane
from .engine import gate, compare_rates
from .journal import InsufficientBudget
from .partitions import load_bank, valid_digest
from .process_curve import measure_process_curve, process_block
from .reports import latency_metrics, process_metrics
from .schema import Checkpoint, Cost, canonical, digest


@dataclass(frozen=True)
class Protocol:
    main: Cost
    development: Cost
    causal: Cost
    bank: Cost
    evaluation: Cost
    windows_ms: tuple
    main_steps: int
    bank_steps: int
    fork_steps: int
    final_blocks: int = 3
    final_count: int = 64
    validation_steps: int = 1


MOCK_PROTOCOL = Protocol(
    main=Cost(160, 160, 8000, blocks=22),
    development=Cost(20, 20, 1000, blocks=2),
    causal=Cost(40, 40, 2000, blocks=4),
    bank=Cost(145, 145, 8000, blocks=20),
    evaluation=Cost(2000, 2000, 20000, blocks=300),
    windows_ms=(20,), main_steps=4, bank_steps=4, fork_steps=2,
)


def build_bank(controller, partition, lane, engine, size, steps):
    journal = controller.journal
    initial = controller.checkpoint(engine, lane.path)
    controller.run(lane, initial, steps)
    history = [asdict(initial)] + [
        row["checkpoint"] for row in journal.events(lane.path + "/step-")
        if row["id"].endswith("/finished")
    ]
    history = [item for item in history if item["environment_digest"] == controller.environment]
    if len(history) <= size:
        selected = history
    elif size == 1:
        selected = history[len(history) // 2:len(history) // 2 + 1]
    else:
        selected = [history[index * (len(history) - 1) // (size - 1)] for index in range(size)]
    return journal.put(f"bank_{partition}", selected)


def _blind_score(controller, lane, key, label, candidate, reference, protocol, *, is_reference):
    ratios, candidate_requests, reference_requests, block_rates = [], [], [], []
    for index in range(protocol.final_blocks):
        block = controller.partitions.block("final", "T", f"{lane.path}/{key}/{index}", protocol.final_count)
        prefix = f"{key}/perf/{label}/{index}"
        if is_reference:
            # A real T receipt is still required; only the *contrast* is known without timing.
            observed = controller.measure(lane, prefix + "/0", reference, block, include_requests=True)
            if observed is None:
                return None
            rate, rows = observed
            base_rate = rate
            candidate_requests.extend(rows)
            reference_requests.extend(rows)
        else:
            order = ("reference", "candidate", "candidate", "reference")
            if int(digest([key, index, label])[-1], 16) % 2:
                order = ("candidate", "reference", "reference", "candidate")
            rates = {"reference": [], "candidate": []}
            for slot, side in enumerate(order):
                engine = reference if side == "reference" else candidate
                observed = controller.measure(lane, f"{prefix}/{slot}", engine, block, include_requests=True)
                rates[side].append(None if observed is None else observed[0])
                if observed is not None:
                    rows = reference_requests if side == "reference" else candidate_requests
                    rows.extend(observed[1])
            if None in rates["reference"]:
                return None
            if None in rates["candidate"]:
                return {"status": "fallback_performance", "speedup": 1.0}
            base_rate, rate = (statistics.harmonic_mean(rates[side]) for side in ("reference", "candidate"))
        ratios.append(compare_rates(base_rate, rate))
        block_rates.append({"reference": base_rate, "candidate": rate})
    return {"status": "valid", "speedup": math.exp(statistics.mean(ratios)), "paired_log_ratios": ratios,
            "tokens_per_second": block_rates, "latency_ms": latency_metrics(candidate_requests),
            "reference_latency_ms": latency_metrics(reference_requests)}


def final_comparison(controller, lane, key, reference, submissions, protocol, *, reuse_identical=None):
    journal = controller.journal
    name = f"{lane.path}/{key}"
    if any(row.get("checkpoint") is None for row in submissions.values()):
        raise ValueError("Formal evaluation requires sealed submissions")
    if (reference.environment_digest != controller.environment or any(
            row["checkpoint"]["environment_digest"] != controller.environment for row in submissions.values())):
        raise ValueError("Formal comparison requires one frozen environment")
    if reuse_identical is None:
        reuse_identical = set(submissions) == {"parent", "proposed"}
    addresses = {label: row["checkpoint"]["engine_digest"] for label, row in submissions.items()}
    addresses["reference"] = reference.engine_digest
    # Local to this sealed comparison: never reuse a D receipt or a different C point's T block.
    unique = {reference.engine_digest: "reference"}
    for label in sorted(submissions):
        unique.setdefault(addresses[label], label)
    source = {label: unique[address] if reuse_identical else label for label, address in addresses.items()}
    engines = {label: journal.get("engine", addresses[label]) for label in dict.fromkeys(source.values())}
    quality_block = controller.partitions.block("final", "Q", name + "/gate", 32)
    quality = {label: controller._operate(lane, f"{key}/quality/{label}", "quality",
                                          {"engine": engine, "role": "final", "block": quality_block})
               for label, engine in engines.items()}
    checks = {label: quality[origin].data for label, origin in source.items()}
    if not gate(quality["reference"]):
        return {"status": "invalid_reference", "quality": checks}
    scores = {}
    for origin in dict.fromkeys(source[label] for label in submissions):
        if not gate(quality[origin]):
            scores[origin] = {"status": "fallback_quality", "speedup": 1.0}
        else:
            scores[origin] = _blind_score(controller, lane, key, origin, engines[origin], engines["reference"],
                                         protocol, is_reference=origin == "reference")
            if scores[origin] is None:
                return {"status": "invalid_reference_measurement"}
    result = {"status": "ok", "engines": {label: scores[source[label]] for label in submissions},
              "quality": checks, "receipt_sources": source}
    journal.event(f"{name}/evaluated", result)
    return result


def seal_pair(journal, backend, partitions, protocol, nodes, replicate, environment, *, causal_points=3,
              ablation=False):
    if len(nodes) != 4 or len(set(nodes)) != 4:
        raise ValueError("Four distinct node identities are required for the paired schedule")
    if type(replicate) is not int or replicate < 0:
        raise ValueError("Replicate must be a nonnegative independent lineage number")
    partitions = partitions.development_view()
    if partitions.development_commitment is None or partitions.final_commitment is None:
        raise ValueError("Register both partition commitments before sealing a pair")
    controller = Controller(journal, backend, partitions, environment,
                            epsilon=environment.get("noise_epsilon", 0.002))
    allocation = nodes[replicate % 4:] + nodes[:replicate % 4]
    fixed_node, evolving_node, dev_node, eval_node = allocation
    run = f"replicate-{replicate}"
    engine = backend.initial_engine()
    manifest = {"harness_revision": HARNESS_REVISION, "environment": controller.environment,
                "initial_engine": journal.put("engine", engine),
                "initial_policy": controller.baseline, "replicate": replicate, "nodes": allocation,
                "simulated": environment.get("simulated", False),
                "protocol": asdict(protocol), "partitions": partitions.public_manifest(),
                "S_dev_block_digest": digest(process_block(partitions, replicate)),
                "causal_points": causal_points, "ablation": ablation}
    journal.event(f"{run}/manifest", manifest)
    d_lane = Lane(f"bank/D/{run}", "research_overhead", dev_node, protocol.bank)
    c_lane = Lane(f"bank/C/{run}", "research_overhead", eval_node, protocol.bank)
    d_bank = build_bank(controller, "D", d_lane, engine, 2 * len(protocol.windows_ms), protocol.bank_steps) if protocol.windows_ms else None
    c_bank = build_bank(controller, "C", c_lane, engine, causal_points, protocol.bank_steps)
    journal.event(f"{run}/banks", {"D": d_bank, "C": c_bank,
                                  "D_cost": journal.spent(d_lane.path).record(),
                                  "C_cost": journal.spent(c_lane.path).record()})
    d_starts = [Checkpoint(**item) for item in load_bank(journal, "development", "D", d_bank)] if d_bank else []
    fixed = Lane(f"{run}/main/fixed", "fixed", fixed_node, protocol.main)
    evolving = Lane(f"{run}/main/evolving", "evolving", evolving_node, protocol.main)
    start = controller.checkpoint(engine, run)
    sealed = {}
    run_order = (fixed, evolving) if (replicate + replicate // len(nodes)) % 2 == 0 else (evolving, fixed)
    for lane in run_order:
        if lane.owner == "evolving":
            sealed[lane.owner] = controller.run(lane, start, protocol.main_steps, windows=protocol.windows_ms,
                                                dev_starts=d_starts, dev_budget=protocol.development,
                                                dev_steps=protocol.validation_steps, track_budget_points=True)
        else:
            sealed[lane.owner] = controller.run(lane, start, protocol.main_steps, track_budget_points=True)
    journal.event(f"{run}/sealed", sealed)
    evaluator = Lane(f"{run}/causal", "research_overhead", eval_node, protocol.evaluation)
    journal.begin_scope(evaluator.path)
    h_star = sealed["evolving"]["checkpoint"]["policy_digest"]
    causal_bank = load_bank(journal, "causal", "C", c_bank)
    causal = []
    memory_factor = []
    # Prepare and seal every continuation before inspecting a final outcome.
    for index, record in enumerate(causal_bank):
        checkpoint = Checkpoint(**record)
        admission = controller._operate(
            evaluator, f"causal-{index}/admit", "quality",
            {"engine": journal.get("engine", checkpoint.engine_digest), "role": "admission",
             "block": partitions.block("admission", "Q_admit", f"{run}/{index}/admission", 32)},
        )
        if not gate(admission):
            checkpoint = start
        fork = controller.paired_steps(evaluator, f"causal-{index}", checkpoint,
                                       controller.baseline, h_star, protocol.causal, protocol.fork_steps)
        causal.append((checkpoint, fork))
        if ablation:
            empty = controller.paired_steps(evaluator, f"memory-{index}", checkpoint,
                                            controller.baseline, h_star, protocol.causal, protocol.fork_steps,
                                            memory_enabled=False)
            memory_factor.append((checkpoint, empty))
    comparisons = {
        "rq1": _sealed_comparison(start, sealed),
        "rq2": [_sealed_comparison(checkpoint, pair) for checkpoint, pair in causal],
        "rq3_memory_off": [_sealed_comparison(checkpoint, pair) for checkpoint, pair in memory_factor],
    }
    bundle = journal.get_event(f"{run}/evaluation_input")
    if bundle is None:
        sealed_groups = [comparisons["rq1"], *comparisons["rq2"], *comparisons["rq3_memory_off"]]
        engine_ids = {p["engine_digest"] for submission in sealed.values() for p in submission["budget_points"]}
        for group in sealed_groups:
            engine_ids.add(group["reference"]["engine_digest"])
            engine_ids.update(value["checkpoint"]["engine_digest"] for value in group["submissions"].values())
        consumed = journal.spent(evaluator.path)
        occupied_wall = journal.wall_elapsed(evaluator.path, consumed)
        payload = {
            "schema": "evolveengine.sealed-pair.v2",
            "run_manifest": manifest,
            "environment": environment,
            "comparisons": comparisons,
            "engines": {address: journal.get("engine", address) for address in sorted(engine_ids)},
            "remaining_evaluation_budget": _remaining(protocol.evaluation, consumed, occupied_wall).record(),
        }
        bundle = {"digest": digest(payload), "payload": payload}
        journal.event(f"{run}/evaluation_input", bundle)
    elif canonical(bundle["payload"]["comparisons"]) != canonical(comparisons):
        raise ValueError("Run attempted to change an already sealed comparison")
    return bundle, _optimization_report(journal, manifest, sealed, d_starts, causal_bank, d_lane, c_lane,
                                        evaluator, fixed, evolving, bundle["digest"])


def _sealed_comparison(reference, submissions):
    return {"reference": asdict(reference), "submissions": submissions}


def _remaining(limit, spent, wall):
    return Cost(wall_ms=max(0, limit.wall_ms - wall), node_ms=max(0, limit.node_ms - spent.node_ms),
                input_tokens=max(0, limit.tokens - spent.tokens), blocks=max(0, limit.blocks - spent.blocks))


def _optimization_report(journal, manifest, sealed, d_starts, causal_bank, d_lane, c_lane, evaluator,
                         fixed, evolving, bundle_digest):
    # Snapshot this stage independently. A local mock may reuse this journal
    # afterward, so a resume should still only export optimization receipts.
    runs = (d_lane.path + "/", c_lane.path + "/", fixed.path + "/", evolving.path + "/", evaluator.path + "/")
    ledger = [row for row in journal.budget_rows() if row["id"].startswith(runs)]
    physical = {}
    for row in ledger:
        if row["physical"]:
            physical[row["node"]] = physical.get(row["node"], Cost()) + Cost(**row["receipt"]["cost"])
    attempts = journal.objects("experience")
    run_prefix = f'replicate-{manifest["replicate"]}/'
    events = [event for event in journal.events("")
              if event["id"].startswith((*runs, run_prefix)) and "/evaluation/" not in event["id"]]
    values = {
        "run_manifest": manifest, "submission": sealed,
        "bank_counts": {"D": len(d_starts), "C": len(causal_bank)},
        "submission_digest": bundle_digest,
        "bank_cost": {"D": journal.spent(d_lane.path).record(), "C": journal.spent(c_lane.path).record()},
        "causal_cost": journal.spent(evaluator.path).record(),
        "events": events, "budget_ledger": ledger,
        "process_metrics": {lane.owner: process_metrics(lane.path, list(attempts.values()), events, ledger)
                            for lane in (fixed, evolving)},
        "physical_cost_by_node": {node: cost.record() for node, cost in physical.items()},
        "policy_registry": journal.objects("policy"), "engine_checkpoints": journal.objects("engine"),
        "memory_registry": journal.objects("memory"), "experience": attempts,
    }
    return values


def verify_submission(bundle, expected_digest, *, schema="evolveengine.sealed-pair.v2"):
    if not valid_digest(expected_digest) or bundle.get("digest") != expected_digest:
        raise ValueError("Sealed submission does not match the registered digest")
    payload = bundle["payload"]
    if digest(payload) != expected_digest or payload["schema"] != schema:
        raise ValueError("Sealed submission contents or schema are invalid")
    if payload["run_manifest"].get("harness_revision") != HARNESS_REVISION:
        raise ValueError("Sealed harness revision differs; evaluate old submissions with their original judge")
    return payload


def open_sealed_judge(journal, backend, partitions, protocol, bundle, expected_digest, *, schema):
    payload = verify_submission(bundle, expected_digest, schema=schema)
    partitions = partitions.final_view()
    manifest = payload["run_manifest"]
    if canonical(partitions.public_manifest()) != canonical(manifest["partitions"]):
        raise ValueError("Judge request configuration differs from the sealed run")
    if canonical(asdict(protocol)) != canonical(manifest["protocol"]):
        raise ValueError("Judge protocol differs from the sealed run")
    controller = Controller(journal, backend, partitions, payload["environment"])
    if controller.environment != manifest["environment"]:
        raise ValueError("Sealed environment is invalid")
    for address, engine in payload["engines"].items():
        if journal.put("engine", engine) != address:
            raise ValueError("Invalid sealed engine digest")
    return payload, controller


def evaluate_pair(journal, backend, partitions, protocol, bundle, expected_digest):
    payload, controller = open_sealed_judge(journal, backend, partitions, protocol, bundle, expected_digest,
                                            schema="evolveengine.sealed-pair.v2")
    manifest = payload["run_manifest"]
    public_block = process_block(controller.partitions, manifest["replicate"])
    if digest(public_block) != manifest["S_dev_block_digest"]:
        raise ValueError("Public budget-point block differs from the precommitted one")
    evaluator = Lane(f'replicate-{manifest["replicate"]}/evaluation', "research_overhead",
                     manifest["nodes"][-1], Cost(**payload["remaining_evaluation_budget"]))
    journal.event(f"{evaluator.path}/input", {"submission_digest": expected_digest})
    journal.begin_scope(evaluator.path)

    def compare(key, group, *, coupled=True):
        stored = journal.get_event(f"{evaluator.path}/{key}/result")
        if stored is not None:
            return stored
        try:
            result = final_comparison(controller, evaluator, key, Checkpoint(**group["reference"]),
                                      group["submissions"], protocol, reuse_identical=coupled)
        except InsufficientBudget:
            result = {"status": "evaluation_budget_exhausted"}
        journal.event(f"{evaluator.path}/{key}/result", result)
        return result

    groups = payload["comparisons"]
    evaluation = {
        "rq1": compare("rq1", groups["rq1"], coupled=False),
        "rq2": [compare(f"rq2-{index}", group) for index, group in enumerate(groups["rq2"])],
        "rq3_memory_off": [compare(f"rq3-{index}", group)
                           for index, group in enumerate(groups["rq3_memory_off"])],
    }
    # Secondary research overhead comes last; it cannot displace a confirmatory measurement.
    process = measure_process_curve(controller, evaluator, groups["rq1"]["submissions"],
                                    manifest["initial_engine"], public_block)
    return {"run_manifest": manifest, "submission_digest": expected_digest, "S_dev": process,
            "submission": groups["rq1"]["submissions"], "sealed_evaluation": evaluation,
            "evaluation_cost": journal.spent(evaluator.path).record(),
            "events": journal.events(evaluator.path + "/"),
            "budget_ledger": list(journal.budget_rows(evaluator.path + "/"))}


def deploy_pair(journal, backend, partitions, protocol, nodes, replicate, environment, **options):
    """Convenience path for a simulation with both sets of keys available."""
    bundle, report = seal_pair(journal, backend, partitions, protocol, nodes, replicate, environment, **options)
    scores = evaluate_pair(journal, backend, partitions, protocol, bundle, bundle["digest"])
    physical = {node: Cost(**cost) for node, cost in report["physical_cost_by_node"].items()}
    for row in scores["budget_ledger"]:
        if row["physical"]:
            physical[row["node"]] = physical.get(row["node"], Cost()) + Cost(**row["receipt"]["cost"])
    return {**report, **scores, "events": report["events"] + scores["events"],
            "budget_ledger": report["budget_ledger"] + scores["budget_ledger"],
            "physical_cost_by_node": {node: cost.record() for node, cost in physical.items()}}
