"""The optimization loop and its independently budgeted validation forks."""

from dataclasses import asdict, dataclass, replace
import math

from .engine import apply_edits, check_patch, compare_rates, gate, throughput
from .journal import InsufficientBudget
from .ranking import filter_pool, policy_rules, rank, replay
from .schema import Candidate, Checkpoint, Cost, ExecutionContract, Experience, Observation, PatchReceipt, canonical


H0 = {"rules": []}
HARNESS_REVISION = "paired-prefix/exclusions/budget-points-v3"
EMPTY_POOL_RETRIES = 1  # Per logical step; only a well-formed but empty filtered pool is redrawn.
STOP = {"empty_pool", "budget_stopped"}


def update_reservation(backend, development):
    scoring = backend.cap("measure", partition="D") + backend.cap("quality")
    return backend.cap("propose") + development * 4 + scoring * 4


def _record_budget_points(journal, prefix, wall_limit, admissions, points, *, before=None):
    """Sign elapsed boundaries at the next tick, then carry the last incumbent through early stop.

    Tick and admission times are durable: on resume the present clock cannot fill an old boundary.
    Defer a boundary equal to the current tick until later, to include decisions in that same ms.
    """
    for quarter in range(len(points), 5):
        at = wall_limit * quarter // 4
        if quarter and before is not None and at >= before:
            break
        incumbent = (next(row for row in reversed(admissions) if row["available_wall_ms"] <= at)
                     if quarter else admissions[0])
        snapshot = {"quarter": quarter, "wall_ms": at, **incumbent}
        journal.event(f"{prefix}/budget-point-{quarter}", snapshot)
        points.append(snapshot)


@dataclass(frozen=True)
class Lane:
    path: str
    owner: str
    node: str
    budget: Cost
    outer: tuple = ()
    peer: "Lane | None" = None  # Only while executing the shared prefix of this fork.

    def __post_init__(self):
        if self.peer is None:
            return
        fork, _, role = self.path.rpartition("/")
        other_fork, _, other_role = self.peer.path.rpartition("/")
        if (not fork or fork != other_fork or {role, other_role} != {"parent", "proposed"}
                or self.peer.peer is not None or self.outer != self.peer.outer
                or (self.owner, self.node) != (self.peer.owner, self.peer.node)):
            raise ValueError("Coupled billing requires sibling policy arms on the same account and node")

    @property
    def payers(self):
        return (self, self.peer) if self.peer is not None else (self,)

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

    def _operate(self, lane, key, stage, request):
        cap = (self.backend.cap(stage, partition=request["block"]["partition"])
               if stage == "measure" else self.backend.cap(stage))
        mirror = (f"{lane.peer.path}/{key}", lane.peer.limits) if lane.peer is not None else None
        return self.journal.call(f"{lane.path}/{key}", request, cap, lane.limits, lane.owner, lane.node,
                                 lambda: self.backend.run(stage, request, cap, lane.node), mirror=mirror)

    def _event(self, lane, key, data):
        for payer in lane.payers:
            self.journal.event(f"{payer.path}/{key}", data)

    def _memory(self, checkpoint):
        return self.journal.get("memory", checkpoint.memory_digest)

    def _memory_view(self, history, primary, enabled, *, paired=False):
        if not enabled:
            return ()
        records = sorted(reversed(history), key=lambda entry: (entry["observation"] or {}).get("primary_class") != primary)
        selected = []
        for record in records:
            summary = {key: record[key] for key in ("parent", "outcome", "cost")}
            if not paired:  # Keep raw provenance in the journal and the main-chain U0 evidence.
                summary["attempt_id"] = record["attempt_id"]
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

    def prepare(self, lane, key, checkpoint, memory_enabled, *, paired=False):
        engine = self.journal.get("engine", checkpoint.engine_digest)
        profile_input = {"engine": engine}
        try:
            profile = self._operate(lane, f"{key}/profile", "profile", profile_input)
        except InsufficientBudget:
            return {"failure": "budget_stopped"}
        if profile.status != "ok":
            return {"failure": "profile_" + profile.status}
        observation = asdict(Observation(**profile.data["observation"]))
        self._event(lane, f"{key}/profiled", {"evidence": profile.data["evidence"], "observation": observation})
        history = [self.journal.get("experience", address) for address in self._memory(checkpoint)]
        memory = self._memory_view(history, observation["primary_class"], memory_enabled, paired=paired)
        tried = set()
        for row in history:
            if row["parent"] == checkpoint.engine_digest and row["candidate"] is not None:
                candidate = row["candidate"]
                tried.add((candidate["action_tag"], tuple(candidate["target_scope"])))
        # Only the local parent's action/scope pairs, never outcomes, arm IDs or another arm's history.
        generator_input = {
            "engine": engine, "profile": {"observation": observation, "evidence": profile.data["evidence"]},
            "memory": memory, "excluded": [{"action_tag": action, "target_scope": list(scope)}
                                          for action, scope in sorted(tried)],
        }
        prepared = {"observation": observation, "memory": memory}
        failure = "empty_pool"
        for draw in range(EMPTY_POOL_RETRIES + 1):
            request = {**generator_input, "draw_index": draw}
            op = "generate" if not draw else f"generate-retry-{draw}"
            try:
                generated = self._operate(lane, f"{key}/{op}", "generate", request)
            except InsufficientBudget:
                failure = "budget_stopped"
                self._event(lane, f"{key}/draw-{draw}", {"status": failure})
                break
            if generated.status != "ok":
                failure = "generate_" + generated.status
                self._event(lane, f"{key}/draw-{draw}", {"status": failure})
                break
            try:
                pool, counts = filter_pool(generated.data["candidates"], self.backend.allowed_scope, tried)
            except (KeyError, TypeError, ValueError) as exc:
                failure = "invalid_candidate_pool"
                self._event(lane, f"{key}/draw-{draw}", {"status": failure, "reason": str(exc)})
                break
            address = self.journal.put("pool", {"pool": pool, "observation": observation, "generator_input": request})
            self._event(lane, f"{key}/draw-{draw}", {"pool_digest": address, "counts": counts,
                                                      "status": "eligible" if pool else "empty"})
            prepared.update(pool=pool, pool_digest=address)
            if pool:
                failure = None
                break
        # Keep the last real pool (even an empty one followed by a denied retry), once per decision.
        if "pool_digest" in prepared:
            self._event(lane, f"{key}/pool_ready", {"pool_digest": prepared["pool_digest"]})
        return {**prepared, "failure": failure}

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

    def _choose(self, lane, key, checkpoint, prepared):
        if prepared["failure"] is not None:
            return None
        policy = self.journal.get("policy", checkpoint.policy_digest)
        order = rank(prepared["observation"], prepared["pool"], policy)
        self._event(lane, f"{key}/ranked", {"ids": order, "selected_id": order[0],
                                           "pool_digest": prepared["pool_digest"]})
        return Candidate(**next(item for item in prepared["pool"] if item["candidate_id"] == order[0]))

    def _attempt(self, lane, key, checkpoint, prepared, choice, block_id):
        """Single implementation, fidelity audit, gate, screens and resulting mainline decision."""
        engine_digest = checkpoint.engine_digest
        evidence = {}
        if choice is None:
            return engine_digest, prepared["failure"], evidence
        parent = self.journal.get("engine", engine_digest)
        contract = ExecutionContract(choice, Observation(**prepared["observation"]), tuple(prepared["memory"]))
        try:
            implement = self._operate(lane, f"{key}/implement", "implement",
                                      {"engine": parent, "contract": asdict(contract)})
            if implement.status != "ok":
                return engine_digest, "implement_" + implement.status, evidence
            try:
                fidelity, reason, diff = check_patch(choice, implement.data, self.backend.allowed_scope,
                                                     lambda filename: self.backend.read_source(parent, filename))
            except (ValueError, KeyError, TypeError) as failure:
                fidelity, reason, diff = "uncertain", str(failure), ""
            evidence["fidelity"] = {"status": fidelity, "reason": reason, "diff": diff,
                                    "selected_id": choice.candidate_id}
            self._event(lane, f"{key}/fidelity", evidence["fidelity"])
            if fidelity != "pass":
                return engine_digest, fidelity, evidence
            candidate = apply_edits(parent, PatchReceipt(**implement.data["receipt"]).edits)
            address = self.journal.put("engine", candidate)
            evidence["candidate_engine"] = address
            quality = self._operate(lane, f"{key}/quality", "quality", {
                "engine": candidate, "role": "search",
                "block": self.partitions.block("search", "S", block_id + "/gate", 8)})
            evidence["quality"] = quality.data
            if not gate(quality):
                return engine_digest, "dev_gate_failed", evidence
            outcome, evidence["log_gain"] = self._confirm(lane, key, block_id, parent, candidate)
            return (address if outcome == "mainline" else engine_digest), outcome, evidence
        except InsufficientBudget:
            return engine_digest, "budget_stopped", evidence

    def _finish(self, lane, key, checkpoint, prepared, choice, engine_digest, outcome, evidence, *, stamp=False, stamp_completion=False):
        path = f"{lane.path}/{key}"
        attempt_cost = self.journal.spent(path)
        admitted = None
        if stamp and outcome == "mainline":
            # This small durable decision precedes memory bookkeeping. A crash between the two
            # must neither erase an on-time admission nor backdate one made only after restart.
            decision_key = path + "/admission"
            decision = self.journal.get_event(decision_key)
            if decision is None:
                elapsed = self.journal.observed_wall_ms(lane.path)
                decision = {"engine_digest": engine_digest, "wall_ms": elapsed}
                self.journal.event(decision_key, decision)
            if decision["engine_digest"] != engine_digest:
                raise ValueError("Replayed mainline decision differs from its signed candidate")
            elapsed = decision["wall_ms"]
            if elapsed > lane.budget.wall_ms:
                engine_digest, outcome = checkpoint.engine_digest, "budget_stopped"
                evidence = {**evidence, "late_decision_wall_ms": elapsed}
            else:
                admitted = elapsed
        experience = Experience(path, checkpoint.engine_digest, asdict(choice) if choice else None,
                                prepared.get("observation"), outcome, evidence, attempt_cost.record())
        experience_digest = self.journal.put("experience", asdict(experience))
        memory = self.journal.put("memory", [*self._memory(checkpoint), experience_digest])
        after = replace(checkpoint, engine_digest=engine_digest, memory_digest=memory,
                        candidate_pool_digest=prepared.get("pool_digest"))
        result = {"checkpoint": asdict(after), "experience": experience_digest, "outcome": outcome}
        if admitted is not None:
            result["available_wall_ms"] = admitted
        if stamp_completion:
            # This timestamp belongs to the recoverable (E,M) event, not to a previous
            # engine admission. A crash before /finished must not backdate a new memory.
            result["completed_wall_ms"] = self.journal.observed_wall_ms(lane.path)
        self.journal.event(f"{path}/finished", result)
        return result

    def step(self, lane, key, checkpoint, *, memory_enabled=True, block_id=None, paired=False, prepared=None,
             stamp_mainline=False, stamp_completion=False):
        path = f"{lane.path}/{key}"
        saved = self.journal.get_event(f"{path}/finished")
        if saved is not None:
            return saved
        self._event(lane, f"{key}/ready", {"checkpoint": asdict(checkpoint)})
        if prepared is None:
            prepared = self.prepare(lane, key, checkpoint, memory_enabled, paired=paired)
        choice = self._choose(lane, key, checkpoint, prepared)
        attempt = self._attempt(lane, key, checkpoint, prepared, choice, block_id or path)
        return self._finish(lane, key, checkpoint, prepared, choice, *attempt,
                            stamp=stamp_mainline or stamp_completion, stamp_completion=stamp_completion)

    def _confirm(self, lane, key, name, parent, candidate):
        timing = self.backend.cap("measure")

        def has_pair(label):
            return self.journal.can_fund((f"{payer.path}/{key}/{label}/{arm}", payer.limits, timing)
                                         for payer in lane.payers for arm in ("parent", "child"))

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
        """D/C only. One common history by construction; never rejoin after the first flip."""
        if checkpoint.environment_digest != self.environment:
            raise ValueError("Checkpoint environment is incompatible")
        j = self.journal
        prefix = f"{lane.path}/{key}"
        j.event(prefix + "/manifest", {"harness": HARNESS_REVISION, "start": asdict(checkpoint),
                                      "policies": [parent, proposed], "lane": asdict(lane),
                                      "budget": budget.record(), "steps": steps, "memory": memory_enabled,
                                      "epsilon": self.epsilon})
        sealed = j.get_event(prefix + "/sealed")
        if sealed is not None:
            return sealed
        roles = ["parent", "proposed"]
        policies = dict(zip(roles, (parent, proposed)))
        if int(j.put("order", prefix)[-1], 16) % 2:
            roles.reverse()
        arms = {r: lane.fork(f"{key}/{r}", budget) for r in roles}
        states = {r: replace(checkpoint, policy_digest=policies[r], lineage=arms[r].path) for r in roles}
        lead, follow = roles
        common = replace(arms[lead], peer=arms[follow])
        j.begin_scope(common.path)
        count = 0
        for index in range(steps):
            step = f"step-{index}"
            block = f"{prefix}/{step}"
            if j.get_event(block + "/coupled") is None:
                for r in roles:
                    self._event(arms[r], step + "/ready", {"checkpoint": asdict(states[r])})
                prepared = self.prepare(common, step, states[lead], memory_enabled, paired=True)
                choices = {r: self._choose(arms[r], step, states[r], prepared) for r in roles}
                if choices[lead] != choices[follow]:  # Complete candidates from the very same frozen pool.
                    split = j.get_event(block + "/split")
                    if split is None:
                        split = {"elapsed_ms": j.wall_elapsed(common.path, j.spent(common.path)),
                                 "pool_digest": prepared["pool_digest"]}
                        j.event(block + "/split", split)
                    outputs = {r: self._split_tail(arms[r], states[r], index, steps, block, prepared,
                                                   memory_enabled, split["elapsed_ms"]) for r in roles}
                    j.event(prefix + "/sealed", outputs)
                    return outputs
                attempt = self._attempt(common, step, states[lead], prepared, choices[lead], block)
                with j.transaction():
                    for r in roles:
                        self._finish(arms[r], step, states[r], prepared, choices[r], *attempt)
                    j.event(block + "/coupled", {"pool_digest": prepared.get("pool_digest"),
                                                "selected": choices[lead].candidate_id if choices[lead] else None})
            finished = {r: j.get_event(f"{arms[r].path}/{step}/finished") for r in roles}
            states = {r: Checkpoint(**end["checkpoint"]) for r, end in finished.items()}
            count = index + 1
            if finished[lead]["outcome"] in STOP:
                break
        elapsed = j.wall_elapsed(common.path, j.spent(common.path))
        with j.transaction():
            outputs = {r: self._seal_run(arms[r], states[r], count, 0, elapsed=elapsed) for r in roles}
            j.event(prefix + "/sealed", outputs)
        return outputs

    def _split_tail(self, arm, state, index, steps, block, prepared, memory_enabled, prefix_elapsed):
        # In serial execution the waiting arm owes the shared prefix time, not the other arm's tail.
        self.journal.begin_scope(arm.path, elapsed_ms=prefix_elapsed)
        for n in range(index, steps):
            end = self.step(arm, f"step-{n}", state, memory_enabled=memory_enabled, paired=True,
                            prepared=prepared if n == index else None, block_id=block if n == index else None)
            state = Checkpoint(**end["checkpoint"])
            if end["outcome"] in STOP:
                break
        return self._seal_run(arm, state, n + 1, 0)

    def update(self, lane, key, checkpoint, dev_starts, dev_budget, dev_steps):
        prefix = f"{lane.path}/{key}"
        prior = self.journal.get_event(f"{prefix}/decided")
        if prior is not None:
            return prior
        parent = self.journal.get("policy", checkpoint.policy_digest)
        reserve = update_reservation(self.backend, dev_budget)
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
            score = self._score_development(lane, arm_key, continuation)
            decision["comparisons"].append({"checkpoint": asdict(dev), "continuations": continuation, **score})
        gains = [item["log_gain"] for item in decision["comparisons"]]
        if all(value is not None and value > 0 for value in gains) and sum(gains) / 2 >= math.log(1.02):
            return {**decision, "status": "accepted", "current": child}
        return {**decision, "status": "rejected"}

    def _score_development(self, lane, key, continuation):
        addresses = {r: end["checkpoint"]["engine_digest"] for r, end in continuation.items()}
        identical = addresses["parent"] == addresses["proposed"]
        prefix = f"{lane.path}/{key}"
        quality_block = self.partitions.block("development", "D", prefix + "/gate", 32)

        def check(label):
            engine = self.journal.get("engine", addresses[label])
            quality = self._operate(lane, f"{key}/gate/{label}", "quality",
                                    {"engine": engine, "role": "development", "block": quality_block})
            return engine, quality

        rates = dict.fromkeys(addresses)  # A structural zero is not an invented throughput.
        if identical:
            _, quality = check("parent")
            return {"rates": rates, "gates": dict.fromkeys(addresses, quality.data), "identical_engine": True,
                    "log_gain": 0.0 if gate(quality) else None}
        block = self.partitions.block("development", "D", prefix + "/score", 16)
        checks = {}
        for label in addresses:
            engine, quality = check(label)
            checks[label] = quality.data
            if gate(quality):
                rates[label] = self.measure(lane, f"{key}/score/{label}", engine, block)
        gain = compare_rates(rates["parent"], rates["proposed"]) if None not in rates.values() else None
        return {"rates": rates, "gates": checks, "identical_engine": False, "log_gain": gain}

    def run(self, lane, checkpoint, steps, *, memory_enabled=True,
            windows=(), dev_starts=(), dev_budget=Cost(), dev_steps=1, track_budget_points=False,
            track_completions=False):
        if checkpoint.environment_digest != self.environment:
            raise ValueError("Checkpoint environment is incompatible")
        prefix = f"{lane.path}/run"
        self.journal.event(f"{prefix}/manifest", {
            "start": asdict(checkpoint), "lane": asdict(lane), "steps": steps,
            "memory_enabled": memory_enabled, "harness": HARNESS_REVISION,
            "windows": list(windows), "bank": [asdict(start) for start in dev_starts],
            "dev_budget": dev_budget.record(), "dev_steps": dev_steps, "track_budget_points": track_budget_points,
            **({"track_completions": True} if track_completions else {}),
        })
        sealed = self.journal.get_event(f"{prefix}/sealed")
        if sealed is not None:
            return sealed
        self.journal.begin_scope(lane.path)
        points = [] if track_budget_points else None
        admissions = [{"source_event": prefix + "/manifest", "available_wall_ms": 0,
                       "engine_digest": checkpoint.engine_digest}]
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
                if track_budget_points:
                    action["at_wall_ms"] = self.journal.observed_wall_ms(lane.path, spent)
                self.journal.event(tick_path, action)
            if track_budget_points:
                _record_budget_points(self.journal, prefix, lane.budget.wall_ms, admissions, points,
                                      before=action["at_wall_ms"])
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
            step_key = f"step-{step_index}"
            next_step = self.step(lane, step_key, checkpoint, memory_enabled=memory_enabled,
                                  stamp_mainline=track_budget_points, stamp_completion=track_completions)
            checkpoint = Checkpoint(**next_step["checkpoint"])
            if track_budget_points and next_step["outcome"] == "mainline":
                admissions.append({"source_event": f"{lane.path}/{step_key}/admission",
                                   "available_wall_ms": next_step["available_wall_ms"],
                                   "engine_digest": checkpoint.engine_digest})
            step_index += 1
            if next_step["outcome"] in STOP:
                break
        if track_budget_points:
            _record_budget_points(self.journal, prefix, lane.budget.wall_ms, admissions, points)
        return self._seal_run(lane, checkpoint, step_index, window_index, budget_points=points)

    def _seal_run(self, lane, checkpoint, steps, windows, *, elapsed=None, budget_points=None):
        key = lane.path + "/run/sealed"
        saved = self.journal.get_event(key)
        if saved is not None:
            return saved
        cost = self.journal.spent(lane.path)
        submission = {"checkpoint": asdict(checkpoint), "cost": cost.record(), "steps": steps, "windows": windows,
                      "elapsed_wall_ms": self.journal.wall_elapsed(lane.path, cost) if elapsed is None else elapsed}
        if budget_points is not None:
            submission["budget_points"] = budget_points
        self.journal.event(key, submission)
        return submission
