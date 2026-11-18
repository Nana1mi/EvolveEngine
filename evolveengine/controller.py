"""The optimization loop and its independently budgeted validation forks."""

from dataclasses import asdict, dataclass, replace
import math

from .engine import apply_edits, check_patch, compare_rates, gate, throughput
from .journal import InsufficientBudget
from .ranking import freeze_pool, policy_rules, rank, replay
from .schema import Candidate, Checkpoint, Cost, ExecutionContract, Experience, Observation, PatchReceipt, Result, canonical


H0 = {"rules": []}


@dataclass(frozen=True)
class Lane:
    path: str
    owner: str
    node: str
    budget: Cost
    outer: tuple = ()

    @property
    def limits(self):
        return ((self.path, self.budget), *self.outer)

    def fork(self, name, budget):
        return Lane(f"{self.path}/{name}", self.owner, self.node, budget, self.limits)


class Controller:
    def __init__(self, journal, backend, partitions, environment, epsilon=0.002):
        if not 0 < epsilon < 0.1:
            raise ValueError("Invalid log-speed noise margin")
        self.journal = journal
        self.backend = backend
        self.partitions = partitions
        self.environment = journal.put("environment", environment)
        self.epsilon = epsilon
        self.baseline = journal.put("policy", H0)

    def checkpoint(self, engine, lineage, policy=None, memory=()):
        address = self.journal.put("engine", engine)
        return Checkpoint(address, policy or self.baseline, self.journal.put("memory", list(memory)),
                          self.environment, lineage)

    def _operate(self, lane, key, stage, request, reused=None):
        cap = self.backend.cap(stage)
        if reused is not None:
            result = Result(reused["data"], Cost(**reused["cost"]), reused["status"], reused["detail"])
            return self.journal.call(f"{lane.path}/{key}", request, result.cost, lane.limits,
                                     lane.owner, lane.node, lambda: result, physical=False)
        result = self.journal.call(f"{lane.path}/{key}", request, cap, lane.limits, lane.owner,
                                   lane.node, lambda: self.backend.run(stage, request, cap, lane.node))
        return result

    def _memory(self, checkpoint):
        return self.journal.get("memory", checkpoint.memory_digest)

    def _memory_view(self, checkpoint, primary, enabled):
        if not enabled:
            return ()
        records = [self.journal.get("experience", address) for address in reversed(self._memory(checkpoint))]
        records.sort(key=lambda entry: (entry["observation"] or {}).get("primary_class") != primary)
        selected = []
        for record in records:
            summary = {key: record[key] for key in ("attempt_id", "parent", "outcome", "cost")}
            candidate = record["candidate"]
            if candidate:
                summary.update(action=candidate["action_tag"], target=candidate["target_scope"])
            summary["paired_log_gain"] = record["evidence"].get("log_gain")
            # A byte limit conservatively bounds the serialized payload even for non-ASCII text.
            if len(canonical([*selected, summary]).encode("utf-8")) <= 1200:
                selected.append(summary)
            if len(selected) == 4:
                break
        return tuple(selected)

    def prepare(self, lane, key, checkpoint, memory_enabled, shared=None):
        path = f"{lane.path}/{key}"
        engine = self.journal.get("engine", checkpoint.engine_digest)
        profile_input = {"engine": engine}
        profile = self._operate(lane, f"{key}/profile", "profile", profile_input,
                                None if shared is None else shared["profile"])
        if profile.status != "ok":
            return {"failure": "profile_" + profile.status, "profile": asdict(profile)}
        observation = asdict(Observation(**profile.data["observation"]))
        self.journal.event(f"{path}/profiled", {"evidence": profile.data["evidence"], "observation": observation})
        memory = self._memory_view(checkpoint, observation["primary_class"], memory_enabled)
        generator_input = {"engine": engine,
                           "profile": {"observation": observation, "evidence": profile.data["evidence"]},
                           "memory": memory}
        generated = self._operate(lane, f"{key}/generate", "generate", generator_input,
                                  None if shared is None else shared["generate"])
        prepared = {"profile": asdict(profile), "generate": asdict(generated)}
        if generated.status != "ok":
            return {**prepared, "failure": "generate_" + generated.status}
        try:
            pool = freeze_pool(generated.data["candidates"], self.backend.allowed_scope)
        except (KeyError, TypeError, ValueError) as failure:
            return {**prepared, "failure": "invalid_candidate_pool", "reason": str(failure)}
        old_attempts = [self.journal.get("experience", address) for address in self._memory(checkpoint)]
        previously_tried = {
            (row["candidate"]["action_tag"], tuple(row["candidate"]["target_scope"]))
            for row in old_attempts
            if enabled_candidate(row, checkpoint.engine_digest)
        } if memory_enabled else set()
        pool = [item for item in pool
                if (item["action_tag"], tuple(item["target_scope"])) not in previously_tried]
        pool_digest = self.journal.put("pool", {"pool": pool, "observation": observation,
                                                "generator_input": generator_input})
        self.journal.event(f"{path}/pool_ready", {"pool_digest": pool_digest})
        return {**prepared, "pool": pool, "pool_digest": pool_digest, "observation": observation,
                "memory": memory, "failure": "empty_pool" if not pool else None}

    def measure(self, lane, key, engine, block, *, include_requests=False):
        result = self._operate(lane, key, "measure", {"engine": engine, "block": block})
        if result.status != "ok":
            return None
        try:
            rate = throughput(result.data, block)
            return (rate, result.data["requests"]) if include_requests else rate
        except (ValueError, KeyError, TypeError):
            return None

    def _compare(self, lane, key, parent, candidate, partition, block_id, reverse=False):
        block = self.partitions.block("search", partition, block_id, 8)
        order = [("parent", parent), ("child", candidate)]
        if reverse:
            order.reverse()
        rates = {name: self.measure(lane, f"{key}/{name}", engine, block) for name, engine in order}
        if None in rates.values():
            return None
        return compare_rates(rates["parent"], rates["child"])

    def step(self, lane, key, checkpoint, *, memory_enabled=True, shared=None, block_id=None):
        path = f"{lane.path}/{key}"
        saved = self.journal.get_event(f"{path}/finished")
        if saved is not None:
            return saved
        parent = self.journal.get("engine", checkpoint.engine_digest)
        policy = self.journal.get("policy", checkpoint.policy_digest)
        self.journal.event(f"{path}/ready", {"checkpoint": asdict(checkpoint)})
        prepared = {}
        choice = None
        engine_digest = checkpoint.engine_digest
        evidence = {}
        outcome = "budget_stopped"
        try:
            prepared = self.prepare(lane, key, checkpoint, memory_enabled, shared)
            outcome = prepared["failure"]
            if outcome is None:
                order = rank(prepared["observation"], prepared["pool"], policy)
                choice = Candidate(**next(item for item in prepared["pool"] if item["candidate_id"] == order[0]))
                self.journal.event(f"{path}/ranked", {"ids": order, "selected_id": choice.candidate_id,
                                                       "pool_digest": prepared["pool_digest"]})
                contract = ExecutionContract(choice, Observation(**prepared["observation"]),
                                             tuple(prepared["memory"]))
                implement = self._operate(lane, f"{key}/implement", "implement",
                                          {"engine": parent, "contract": asdict(contract)})
                if implement.status != "ok":
                    outcome = f"implement_{implement.status}"
                else:
                    try:
                        fidelity, reason, diff = check_patch(
                            choice, implement.data, self.backend.allowed_scope,
                            lambda filename: self.backend.read_source(parent, filename),
                        )
                    except (ValueError, KeyError, TypeError) as failure:
                        fidelity, reason, diff = "uncertain", str(failure), ""
                    evidence["fidelity"] = {"status": fidelity, "reason": reason, "diff": diff,
                                            "selected_id": choice.candidate_id}
                    self.journal.event(f"{path}/fidelity", evidence["fidelity"])
                    outcome = fidelity
                    if fidelity == "pass":
                        patch = PatchReceipt(**implement.data["receipt"])
                        candidate = apply_edits(parent, patch.edits)
                        address = self.journal.put("engine", candidate)
                        evidence["candidate_engine"] = address
                        name = block_id or path
                        quality = self._operate(lane, f"{key}/quality", "quality",
                                                {"engine": candidate, "role": "search",
                                                 "block": self.partitions.block("search", "S", name + "/gate", 8)})
                        evidence["quality"] = quality.data
                        outcome = "dev_gate_failed"
                        if gate(quality):
                            outcome, evidence["log_gain"] = self._confirm(
                                lane, key, name, parent, candidate)
                            if outcome == "mainline":
                                engine_digest = address
        except InsufficientBudget:
            outcome = "budget_stopped"
        attempt_cost = self.journal.spent(path)
        experience = Experience(path, checkpoint.engine_digest, asdict(choice) if choice else None,
                                prepared.get("observation"), outcome, evidence, attempt_cost.record())
        experience_digest = self.journal.put("experience", asdict(experience))
        memory = self.journal.put("memory", [*self._memory(checkpoint), experience_digest])
        after = replace(checkpoint, engine_digest=engine_digest, memory_digest=memory,
                        candidate_pool_digest=prepared.get("pool_digest"))
        result = {"checkpoint": asdict(after), "experience": experience_digest, "outcome": outcome}
        self.journal.event(f"{path}/finished", result)
        return result

    def _confirm(self, lane, key, name, parent, candidate):
        timing = self.backend.cap("measure")
        def has_pair(label):
            operations = [(f"{lane.path}/{key}/{label}/{arm}", timing) for arm in ("parent", "child")]
            return self.journal.can_finish(lane.limits, operations)

        if not has_pair("screen"):
            return "budget_stopped", None
        gain = self._compare(lane, f"{key}/screen", parent, candidate, "S", name + "/a",
                             reverse=int(self.journal.put("order", name)[-1], 16) % 2 == 1)
        if gain is None:
            return "measurement_failed", None
        if gain > 2 * self.epsilon:
            return "mainline", gain
        if gain <= 0 or not has_pair("confirm"):
            return "archived", gain
        other = self._compare(lane, f"{key}/confirm", parent, candidate, "S", name + "/b",
                              reverse=int(self.journal.put("order", name)[-1], 16) % 2 == 0)
        if other is not None and other > 0 and (gain + other) / 2 > self.epsilon:
            return "mainline", (gain + other) / 2
        return "archived", gain

    def paired_steps(self, lane, key, checkpoint, parent, proposed, budget, steps, memory_enabled=True):
        first_results = self.journal.get_event(f"{lane.path}/{key}/shared")
        groups = {"parent": parent, "proposed": proposed}
        first = list(groups)
        if int(self.journal.put("order", f"{lane.path}/{key}")[-1], 16) % 2 == 1:
            first.reverse()
        outputs = {}
        for role in first:
            policy = groups[role]
            arm = lane.fork(f"{key}/{role}", budget)
            self.journal.begin_scope(arm.path)
            start = replace(checkpoint, policy_digest=policy, lineage=arm.path)
            if first_results is None:
                try:
                    first_results = self.prepare(arm, "step-0", start, memory_enabled)
                except InsufficientBudget:
                    first_results = {"failure": "budget_stopped"}
                self.journal.event(f"{lane.path}/{key}/shared", {"first": role, "data": first_results})
                shared = None
            else:
                shared = None if first_results["first"] == role else first_results["data"]
            if "profile" in (shared or {"profile": True}):
                outputs[role] = self.run(arm, start, steps, memory_enabled=memory_enabled, shared=shared,
                                         first_block_id=f"{lane.path}/{key}")
            else:
                outputs[role] = self.run(arm, start, 0, memory_enabled=memory_enabled)
            first_results = self.journal.get_event(f"{lane.path}/{key}/shared")
        return outputs

    def update(self, lane, key, checkpoint, dev_starts, dev_budget, dev_steps):
        prefix = f"{lane.path}/{key}"
        prior = self.journal.get_event(f"{prefix}/decided")
        if prior is not None:
            return prior
        parent = self.journal.get("policy", checkpoint.policy_digest)
        scoring = self.backend.cap("measure") + self.backend.cap("quality")
        reserve = self.backend.cap("propose") + dev_budget * 4 + scoring * 4
        start = self.journal.get_event(f"{prefix}/started")
        if start is None:
            start = {"reserved": len(dev_starts) == 2 and self.journal.can_reserve(lane.limits, reserve),
                     "parent": checkpoint.policy_digest}
            self.journal.event(f"{prefix}/started", start)
        decision = {"parent": checkpoint.policy_digest, "current": checkpoint.policy_digest,
                    "status": "insufficient_budget", "comparisons": []}
        if len(dev_starts) != 2:
            decision["status"] = "unavailable_development_checkpoints"
        if start["reserved"]:
            decision = self._validate_update(lane, key, checkpoint, parent, dev_starts,
                                             dev_budget, dev_steps, decision)
        self.journal.event(f"{prefix}/decided", decision)
        return decision

    def _validate_update(self, lane, key, checkpoint, parent, dev_starts, dev_budget, dev_steps, decision):
        prefix = f"{lane.path}/{key}"
        experiences = [self.journal.get("experience", item) for item in self._memory(checkpoint)[-12:]]
        training = [
            {name: row[name] for name in ("attempt_id", "candidate", "observation", "outcome", "cost")}
            | {"log_gain": row["evidence"].get("log_gain")}
            for row in experiences if row["attempt_id"].startswith(lane.path + "/step-")
        ]
        proposal = self._operate(lane, f"{key}/propose", "propose", {"policy": parent, "experience": training})
        if proposal.status != "ok":
            return {**decision, "status": "proposal_" + proposal.status}
        try:
            proposed = proposal.data["policy"]
            policy_rules(proposed)
            identifiers = {row["attempt_id"] for row in training}
            basis = proposal.data["basis"]
            if not isinstance(basis, list) or not basis or not all(isinstance(item, str) and item in identifiers for item in basis):
                raise ValueError("proposal must cite authorized training attempts")
            step_pools = [event["pool_digest"] for event in self.journal.events(f"{lane.path}/step-")
                          if event["id"].endswith("/pool_ready")]
            report = replay(parent, proposed, [
                {"pool_digest": address, **self.journal.get("pool", address)} for address in step_pools
            ])
        except (TypeError, ValueError, KeyError) as failure:
            return {**decision, "status": "invalid_proposal", "detail": str(failure)}
        child = self.journal.put("policy", proposed)
        decision = {**decision, "proposed": child, "basis": basis, "replay": report}
        if not any(item["first_choice_changed"] for item in report):
            return {**decision, "status": "no_change"}
        for index, dev in enumerate(dev_starts):
            if not dev.lineage.startswith("bank/D/") or dev.environment_digest != self.environment:
                raise PermissionError("Development verification requires an independent D checkpoint")
            arm_key = f"{key}/d-{index}"
            continuation = self.paired_steps(lane, arm_key, dev, checkpoint.policy_digest, child,
                                              dev_budget, dev_steps)
            block = self.partitions.block("development", "D", prefix + f"/d-{index}/score", 16)
            quality_block = self.partitions.block("development", "D", prefix + f"/d-{index}/gate", 32)
            rates, checks = {}, {}
            for label, submission in continuation.items():
                engine = self.journal.get("engine", submission["checkpoint"]["engine_digest"])
                quality = self._operate(lane, f"{arm_key}/gate/{label}", "quality",
                                        {"engine": engine, "role": "development", "block": quality_block})
                checks[label] = quality.data
                rates[label] = self.measure(lane, f"{arm_key}/score/{label}", engine, block) if gate(quality) else None
            decision["comparisons"].append({"checkpoint": asdict(dev), "rates": rates,
                                             "gates": checks, "continuations": continuation})
        gains = [
            compare_rates(rates["parent"], rates["proposed"]) if None not in rates.values() else None
            for rates in (item["rates"] for item in decision["comparisons"])
        ]
        if all(value is not None and value > 0 for value in gains) and sum(gains) / 2 >= math.log(1.02):
            return {**decision, "status": "accepted", "current": child}
        return {**decision, "status": "rejected"}

    def run(self, lane, checkpoint, steps, *, memory_enabled=True, shared=None, first_block_id=None,
            windows=(), dev_starts=(), dev_budget=Cost(), dev_steps=1):
        if checkpoint.environment_digest != self.environment:
            raise ValueError("Checkpoint environment is incompatible")
        prefix = f"{lane.path}/run"
        self.journal.event(f"{prefix}/manifest", {
            "start": asdict(checkpoint), "lane": asdict(lane), "steps": steps,
            "memory_enabled": memory_enabled, "first_block_id": first_block_id,
            "windows": list(windows), "bank": [asdict(start) for start in dev_starts],
            "dev_budget": dev_budget.record(), "dev_steps": dev_steps,
        })
        sealed = self.journal.get_event(f"{prefix}/sealed")
        if sealed is not None:
            return sealed
        self.journal.begin_scope(lane.path)
        step_index, window_index, tick = 0, 0, 0
        while step_index < steps:
            tick_path = f"{prefix}/tick-{tick}"
            tick += 1
            action = self.journal.get_event(tick_path)
            if action is None:
                spent = self.journal.spent(lane.path)
                due = (window_index < len(windows)
                       and self.journal.wall_elapsed(lane.path, spent) >= windows[window_index])
                action = {"next": "update" if due else "step"}
                self.journal.event(tick_path, action)
            if action["next"] == "update":
                try:
                    update = self.update(lane, f"window-{window_index}", checkpoint,
                                         dev_starts[window_index * 2:window_index * 2 + 2], dev_budget, dev_steps)
                except InsufficientBudget:
                    # An overrun has already been billed. Seal with the last approved policy.
                    self.journal.event(f"{lane.path}/window-{window_index}/decided",
                                       {"status": "overrun", "current": checkpoint.policy_digest})
                    break
                window_index += 1
                checkpoint = replace(checkpoint, policy_digest=update["current"])
                continue
            next_step = self.step(lane, f"step-{step_index}", checkpoint,
                                  memory_enabled=memory_enabled, shared=shared if step_index == 0 else None,
                                  block_id=first_block_id if step_index == 0 else None)
            checkpoint = Checkpoint(**next_step["checkpoint"])
            step_index += 1
            if next_step["outcome"] in {"empty_pool", "budget_stopped"}:
                break
        submission = {"checkpoint": asdict(checkpoint), "cost": self.journal.spent(lane.path).record(),
                      "steps": step_index, "windows": window_index}
        submission["elapsed_wall_ms"] = self.journal.wall_elapsed(lane.path, Cost(**submission["cost"]))
        self.journal.event(f"{prefix}/sealed", submission)
        return submission


def enabled_candidate(row, engine_digest):
    return row["parent"] == engine_digest and row["candidate"] is not None
