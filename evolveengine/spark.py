"""CPU-side backend for pinned Pi and four SSH-addressable Spark nodes."""

import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

from .deepseek import Provider, ProviderBudget
from .controller import EMPTY_POOL_RETRIES, update_reservation
from .engine import throughput
from .experiment import Protocol
from .partitions import Partitions
from .pi import ModelStages, PiAgent
from .schema import Cost, EvaluationReceipt, Result, canonical, digest
from .workspace import Repository

MEASUREMENT_CAP = {"S": "measure", "S_dev": "measure", "D": "measure_development", "T": "measure_final"}
PREFLIGHT_SCHEMA = "evolveengine.spark-preflight.v2"


def cost_from(record):
    return Cost(**record)


def load_protocol(values):
    converted = dict(values)
    for name in ("main", "development", "causal", "bank", "evaluation"):
        converted[name] = cost_from(converted[name])
    converted["windows_ms"] = tuple(converted["windows_ms"])
    protocol = Protocol(**converted)
    if (len(protocol.windows_ms) > 2 or tuple(sorted(set(protocol.windows_ms))) != protocol.windows_ms
            or any(type(value) is not int or not 0 < value < protocol.main.wall_ms for value in protocol.windows_ms)
            or min(protocol.main_steps, protocol.bank_steps, protocol.fork_steps, protocol.validation_steps) <= 0
            or protocol.final_count <= 0 or protocol.final_count % 4 or protocol.final_blocks <= 0):
        raise ValueError("Invalid preregistered Spark protocol")
    return protocol


class SparkBackend:
    version = "spark-v1"

    def __init__(self, config, *, evaluation_only=False):
        self.config = config
        self.evaluation_only = evaluation_only
        self.nodes = tuple(config["nodes"])
        if len(self.nodes) != 4 or len(set(self.nodes)) != 4:
            raise ValueError("Configure four distinct OpenSSH host aliases")
        if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,120}", host) for host in self.nodes):
            raise ValueError("Invalid OpenSSH host alias")
        self.allowed_scope = tuple(config["editable"])
        if not evaluation_only:
            self.repository = Repository(config["engine_checkout"], config["engine_revision"], self.allowed_scope)
            self.provider = Provider(**config["provider"])
            self.pi = PiAgent(config["pi_command"], config["pi_version"], self.provider)
            self.model = ModelStages(self.pi, self.repository, config["source_hints"])
        self.protocol = load_protocol(config["protocol"])
        self.caps = {stage: cost_from(value) for stage, value in config["caps"].items()}
        required = {"generate", "implement", "propose", "profile", "quality", "measure", "preflight"}
        if not required <= self.caps.keys() <= required | set(MEASUREMENT_CAP.values()):
            raise ValueError("Declare the cost cap of every operation")
        if any(cost.wall_ms < 5000 for cost in self.caps.values()):
            raise ValueError("Live operations must allow watchdog startup and cleanup")
        if any(cost.blocks != (1 if stage in MEASUREMENT_CAP.values() else 0)
               for stage, cost in self.caps.items() if stage != "preflight"):
            raise ValueError("One timing operation must reserve one benchmark block")

    def initial_engine(self):
        return {"base": self.repository.revision, "edits": {}}

    def read_source(self, engine, path):
        return self.repository.source(engine, path)

    def cap(self, stage, *, partition="S"):
        if stage == "measure":
            return self.caps.get(MEASUREMENT_CAP[partition], self.caps["measure"])
        return self.caps[stage]

    def run(self, stage, request, cap, node):
        partition = request.get("block", {}).get("partition")
        if partition == "S_dev" and not self.evaluation_only:
            raise PermissionError("Budget point replay is a post-seal evaluation operation")
        if self.evaluation_only:
            if not ((stage == "measure" and partition in {"T", "S_dev"}) or
                    (stage == "quality" and partition == "Q" and request.get("role") == "final")):
                raise PermissionError("Judge backend only runs sealed quality and timing operations")
            return self._remote(stage, request, cap, node)
        if stage in {"generate", "implement", "propose"}:
            return self.model.run(stage, request, cap)
        return self._remote(stage, request, cap, node)

    def _remote(self, stage, request, cap, node):
        if node not in self.nodes:
            raise ValueError("Node is outside the registered Spark pool")
        timed_blocks = int(stage == "measure")
        worktree = shlex.quote(self.config["remote_project"])
        settings = shlex.quote(self.config["remote_worker_config"])
        command = f"cd {worktree} && python3 -m evolveengine.spark_worker --config {settings}"
        job = {**request, "stage": stage, "timeout_ms": cap.wall_ms - 3000,
               "configuration": self.config["worker_config_digest"]}
        start = time.monotonic()
        try:
            completed = subprocess.run(
                ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, command],
                input=canonical(job).encode("utf-8"), capture_output=True, timeout=cap.wall_ms / 1000,
            )
            if completed.returncode:
                status, detail, data = "failed", f"SSH worker exited {completed.returncode}: " + completed.stderr.decode(errors="replace")[-1500:], {}
            else:
                receipt = json.loads(completed.stdout)
                status, detail, data = receipt["status"], receipt.get("detail", ""), receipt["data"]
                if status != "ok" and completed.stderr:
                    detail += "\n" + completed.stderr.decode(errors="replace")[-1500:]
        except subprocess.TimeoutExpired:
            status, detail, data = "timeout", "SSH worker exceeded the reservation", {}
        except (OSError, ValueError, KeyError, TypeError) as failure:
            status, detail, data = "failed", str(failure), {}
        elapsed = math.ceil((time.monotonic() - start) * 1000)
        # A node is reserved for the entire remote operation, including launch and cleanup.
        return Result(data, Cost(elapsed, elapsed, blocks=timed_blocks), status, detail)

    def pilot_preflight(self, partitions, targets=None):
        targets = self.nodes if targets is None else tuple(targets)
        bounds = self.reservation_diagnostics()
        pi = self._preflight_agent()
        nodes = {}
        block = partitions.block("search", "S", "preflight/smoke", 8)
        for node in targets:
            nodes[node] = self._preflight_node(node, partitions, block)
        if len({item["sensor"]["uuid"] for item in nodes.values()}) != len(nodes):
            raise ValueError("Different node aliases resolve to the same GPU")
        return {"schema": PREFLIGHT_SCHEMA, "nodes": nodes, "agent": pi, "configuration_digest": digest(self.config),
                "partition_manifest": partitions.public_manifest(), "reservation_diagnostics": bounds}

    def reservation_diagnostics(self):
        core = sum((self.cap(stage) for stage in ("profile", "generate", "implement", "quality")), Cost())
        step = core + self.cap("measure") * 2
        ceiling = step + self.cap("generate") * EMPTY_POOL_RETRIES + self.cap("measure") * 2
        lanes = ("main", "development", "causal", "bank")
        update = update_reservation(self, self.protocol.development)
        windows = [{"at_ms": opening,
                    "can_reserve_from_window_start": (Cost(wall_ms=opening) + update).fits(self.protocol.main)}
                   for opening in self.protocol.windows_ms]
        return {"one_step_first_screen": step.record(), "one_step_reserved_ceiling": ceiling.record(),
                "one_step_first_screen_fits": {name: step.fits(getattr(self.protocol, name)) for name in lanes},
                "one_step_ceiling_fits": {name: ceiling.fits(getattr(self.protocol, name)) for name in lanes},
                "one_harness_update": update.record(), "windows": windows}

    def _preflight_agent(self):
        pi = self.pi.preflight()
        with ProviderBudget(self.provider, self.cap("generate")) as broker:
            with tempfile.TemporaryDirectory(prefix="evolve-preflight-") as work:
                response, _ = self.pi.session(Path(work), 'Return the object {"ready":true}.', broker)
            if response != {"ready": True} or not broker.rows or broker.failure:
                raise ValueError("Pi to DeepSeek preflight could not verify a metered response")
            pi["metered_probe"] = broker.rows
        return pi

    def _preflight_node(self, node, partitions, block):
        engine = self.initial_engine()
        probe = self.cap("preflight")
        result = self._remote("preflight", {"engine": engine, "block": block}, probe, node)
        if result.status != "ok" or not result.cost.fits(probe):
            raise ValueError(f"{node} failed Spark preflight: {result.detail or 'probe exceeded its reservation'}")
        receipt = result.data
        if not EvaluationReceipt(**receipt["receipt"]).passed:
            raise ValueError(f"{node} failed the E0 quality gates: {receipt['receipt']['checks']}")
        if receipt.get("negative_controls", {}).get("truncated_vocab_rejected") is not True:
            raise ValueError(f"{node} did not reject the truncated-vocabulary control")
        costs = result.cost
        calibration = {}
        controls = dict(receipt["negative_controls"])
        repeats = self.config.get("preflight_repeats", 2)
        if type(repeats) is not int or repeats < 2:
            raise ValueError("Preflight needs at least two independent timings of each block shape")
        for purpose, partition, size in (("search", "S", 8), ("development", "D", 16),
                                         ("final", "T", self.protocol.final_count)):
            # Only public prompts are used. The reservation matches the real job shape.
            sample = partitions.block("search", "S", f"preflight/calibration/{purpose}", size)
            cap = self.cap("measure", partition=partition)
            rates, durations = [], []
            for trial in range(repeats):
                timing = self._remote("measure", {"engine": engine, "block": sample}, cap, node)
                if timing.status != "ok" or not timing.cost.fits(cap):
                    detail = timing.detail or "block exceeded its reservation"
                    raise ValueError(f"{node} failed the {purpose} calibration ({size} requests): {detail}")
                rates.append(throughput(timing.data, sample))
                durations.append(timing.cost.wall_ms)
                costs += timing.cost
                if partition == "S" and trial == 0:
                    first, *rest = timing.data["requests"]
                    short = {**first, "output_tokens": first["output_tokens"] - 1}
                    try:
                        throughput({**timing.data, "requests": [short, *rest]}, sample)
                    except ValueError:
                        controls["short_response_rejected"] = True
                    else:
                        raise ValueError(f"{node} did not reject the short-response control")
            drift = math.log(max(rates) / min(rates))
            if drift > self.config["max_preflight_drift"]:
                raise ValueError(f"{node} {purpose} E0 blocks exceeded the configured stability allowance: {drift:.4f}")
            calibration[purpose] = {"request_count": size, "rates": rates, "elapsed_ms": durations,
                                    "max_abs_log_ratio": drift, "per_block_cap": cap.record()}
        return {"driver": receipt["sensor_before"]["driver"], "sensor": receipt["sensor_before"],
                "runtime": receipt["runtime"], "runtime_image": receipt["image"],
                "engine_revision": receipt["engine_revision"], "model_digest": receipt["model_digest"],
                "calibration": calibration, "cost": costs.record(),
                "quality": receipt["receipt"], "profile": receipt["profile"], "negative_controls": controls}


def data_partitions(config, seed, salt=None, **keys):
    filename = Path(config["token_ids"])
    data = json.loads(filename.read_text(encoding="utf-8"))
    if digest(data) != config["token_ids_digest"]:
        raise ValueError("Token ID manifest does not match its lock")
    return Partitions(seed, salt, data["token_ids"], data.get("special_prefix", ()), **keys)
