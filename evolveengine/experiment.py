"""Run a fixed/evolving pair with separately funded validation and evaluation."""

from dataclasses import asdict, dataclass
import math
import statistics

from .controller import Controller, Lane
from .engine import gate, compare_rates
from .partitions import load_bank
from .schema import Checkpoint, Cost, digest


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


def final_comparison(controller, lane, key, reference, submissions, protocol):
    journal = controller.journal
    name = f"{lane.path}/{key}"
    if any(row.get("checkpoint") is None for row in submissions.values()):
        raise ValueError("Formal evaluation requires sealed submissions")
    quality_block = controller.partitions.block("final", "Q", name + "/gate", 32)
    engines = {label: journal.get("engine", record["checkpoint"]["engine_digest"])
               for label, record in submissions.items()}
    engines["reference"] = journal.get("engine", reference.engine_digest)
    quality = {
        label: controller._operate(lane, f"{key}/quality/{label}", "quality",
                                   {"engine": engine, "role": "final", "block": quality_block})
        for label, engine in engines.items()
    }
    if not gate(quality["reference"]):
        return {"status": "invalid_reference", "quality": {key: val.data for key, val in quality.items()}}
    output = {}
    for label in submissions:
        candidate = engines[label]
        if not gate(quality[label]):
            output[label] = {"status": "fallback_quality", "speedup": 1.0}
            continue
        ratios = []
        for block_index in range(protocol.final_blocks):
            block = controller.partitions.block("final", "T", f"{name}/{block_index}", protocol.final_count)
            order = (["reference", "candidate", "candidate", "reference"]
                     if int(digest([key, block_index, label])[-1], 16) % 2 == 0
                     else ["candidate", "reference", "reference", "candidate"])
            rates = {"reference": [], "candidate": []}
            for index, side in enumerate(order):
                engine = engines["reference"] if side == "reference" else candidate
                rates[side].append(controller.measure(lane, f"{key}/perf/{label}/{block_index}/{index}",
                                                      engine, block))
            if any(rate is None for rate in rates["reference"]):
                return {"status": "invalid_reference_measurement"}
            if any(rate is None for rate in rates["candidate"]):
                break
            # Counts are equal, so aggregate elapsed time using the harmonic mean.
            ratios.append(compare_rates(statistics.harmonic_mean(rates["reference"]),
                                        statistics.harmonic_mean(rates["candidate"])))
        if len(ratios) == protocol.final_blocks:
            output[label] = {"status": "valid", "speedup": math.exp(statistics.mean(ratios)),
                             "paired_log_ratios": ratios}
        else:
            output[label] = {"status": "fallback_performance", "speedup": 1.0}
    result = {"status": "ok", "engines": output, "quality": {key: val.data for key, val in quality.items()}}
    journal.event(f"{name}/evaluated", result)
    return result


def deploy_pair(journal, backend, partitions, protocol, nodes, replicate, environment, *, causal_points=3,
                ablation=False):
    if len(nodes) != 4 or len(set(nodes)) != 4:
        raise ValueError("Four distinct node identities are required for the paired schedule")
    if type(replicate) is not int or replicate < 0:
        raise ValueError("Replicate must be a nonnegative independent lineage number")
    controller = Controller(journal, backend, partitions, environment,
                            epsilon=environment.get("noise_epsilon", 0.002))
    allocation = nodes[replicate % 4:] + nodes[:replicate % 4]
    fixed_node, evolving_node, dev_node, eval_node = allocation
    run = f"replicate-{replicate}"
    engine = backend.initial_engine()
    manifest = {"environment": controller.environment, "initial_engine": journal.put("engine", engine),
                "initial_policy": controller.baseline, "replicate": replicate, "nodes": allocation,
                "simulated": environment.get("simulated", False),
                "protocol": asdict(protocol), "partitions": partitions.public_manifest(),
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
                                                dev_steps=protocol.validation_steps)
        else:
            sealed[lane.owner] = controller.run(lane, start, protocol.main_steps)
    journal.event(f"{run}/sealed", sealed)
    evaluator = Lane(f"{run}/evaluation", "research_overhead", eval_node, protocol.evaluation)
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
    rq1 = final_comparison(controller, evaluator, "rq1", start, sealed, protocol)
    rq2 = [final_comparison(controller, evaluator, f"rq2-{index}", checkpoint, pair, protocol)
           for index, (checkpoint, pair) in enumerate(causal)]
    rq3 = [final_comparison(controller, evaluator, f"rq3-{index}", checkpoint, pair, protocol)
           for index, (checkpoint, pair) in enumerate(memory_factor)]
    ledger = list(journal.budget_rows())
    physical = {}
    for row in ledger:
        if row["physical"]:
            physical[row["node"]] = physical.get(row["node"], Cost()) + Cost(**row["receipt"]["cost"])
    values = {
        "run_manifest": manifest, "submission": sealed,
        "bank_counts": {"D": len(d_starts), "C": len(causal_bank)},
        "sealed_evaluation": {"rq1": rq1, "rq2": rq2, "rq3_memory_off": rq3},
        "bank_cost": {"D": journal.spent(d_lane.path).record(), "C": journal.spent(c_lane.path).record()},
        "evaluation_cost": journal.spent(evaluator.path).record(),
        "events": journal.events(""), "budget_ledger": ledger,
        "physical_cost_by_node": {node: cost.record() for node, cost in physical.items()},
        "policy_registry": journal.objects("policy"), "engine_checkpoints": journal.objects("engine"),
        "memory_registry": journal.objects("memory"), "experience": journal.objects("experience"),
    }
    return values
