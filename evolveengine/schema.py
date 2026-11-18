"""Shared experiment contracts. Times are integer milliseconds."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class Cost:
    wall_ms: int = 0
    node_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    blocks: int = 0

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in self.values()):
            raise ValueError("Cost requires nonnegative integer counters")

    def values(self):
        return tuple(getattr(self, field.name) for field in fields(self))

    @property
    def tokens(self):
        return self.input_tokens + self.output_tokens

    def __add__(self, other):
        return Cost(*(a + b for a, b in zip(self.values(), other.values())))

    def __mul__(self, multiplier):
        return Cost(*(value * multiplier for value in self.values()))

    def fits(self, limit):
        return (self.wall_ms <= limit.wall_ms and self.node_ms <= limit.node_ms
                and self.tokens <= limit.tokens and self.blocks <= limit.blocks)

    def record(self):
        return asdict(self)


CLASSES = ("host_dispatch", "compute", "memory", "kv_cache")
PHASES = ("prefill", "decode", "mixed", "unknown")
BANDS = ("low", "medium", "high")
ACTIONS = ("pack_request_metadata", "reduce_launch_gap", "improve_locality",
           "fuse_compute", "manage_kv_cache", "other")
FEATURES = {"primary_class": (*CLASSES, "unknown"), "phase": PHASES,
            "host_launch_gap": ("low", "high", "unknown")}


@dataclass(frozen=True)
class Observation:
    primary_class: str
    phase: str
    host_launch_gap: str

    def __post_init__(self):
        if any(getattr(self, key) not in allowed for key, allowed in FEATURES.items()):
            raise ValueError("Observation has an unregistered feature")


def safe_path(name):
    if not isinstance(name, str) or not re.fullmatch(r"[-.A-Za-z0-9_/]+", name):
        return False
    path = PurePosixPath(name)
    return (not path.is_absolute() and bool(path.parts) and str(path) == name
            and not any(part.lower() in (".", "..", ".git") for part in path.parts))


def in_scope(path, scope):
    return any(path == target or path.startswith(target + "/") for target in scope)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    action_tag: str
    bottleneck_class: str
    target_scope: tuple
    hypothesis: str
    expected_signal: str
    cost_band: str
    infeasible_conditions: str

    def __post_init__(self):
        if not isinstance(self.candidate_id, str) or not re.fullmatch(r"[\w-]{1,48}", self.candidate_id):
            raise ValueError("Invalid candidate ID")
        if self.action_tag not in ACTIONS or self.bottleneck_class not in CLASSES or self.cost_band not in BANDS:
            raise ValueError("Candidate uses an unregistered tag or bin")
        if (not isinstance(self.target_scope, (list, tuple)) or not self.target_scope
                or not all(safe_path(path) for path in self.target_scope)):
            raise ValueError("Candidate target scope requires repository-relative paths")
        for value in (self.hypothesis, self.expected_signal, self.infeasible_conditions):
            if not isinstance(value, str) or not value.strip() or len(value) > 1000:
                raise ValueError("Candidate requires bounded hypothesis, signal and infeasibility text")
        object.__setattr__(self, "target_scope", tuple(sorted(set(self.target_scope))))


@dataclass(frozen=True)
class ExecutionContract:
    candidate: Candidate
    observation: Observation
    memory: tuple


@dataclass(frozen=True)
class PatchReceipt:
    status: str
    selected_id: str
    action_tag: str
    intent: str
    edits: dict
    reason: str = ""

    def __post_init__(self):
        if self.status not in ("implemented", "infeasible", "failed"):
            raise ValueError("Unknown implementation status")
        if not all(isinstance(item, str) for item in (self.selected_id, self.action_tag, self.intent, self.reason)):
            raise ValueError("Invalid implementation receipt")
        if (not isinstance(self.edits, dict) or
                not all(safe_path(path) and isinstance(content, str) for path, content in self.edits.items())):
            raise ValueError("Patch edits must be repository-relative text files")


@dataclass(frozen=True)
class EvaluationReceipt:
    passed: bool
    checks: dict
    reason: str = ""

    def __post_init__(self):
        if type(self.passed) is not bool or not self.checks or type(self.checks) is not dict:
            raise ValueError("Missing gate receipt")
        if not all(type(check) is bool for check in self.checks.values()):
            raise ValueError("Every gate requires a boolean result")
        if self.passed != all(self.checks.values()):
            raise ValueError("Gate receipt contradicts its checks")


@dataclass(frozen=True)
class Checkpoint:
    engine_digest: str
    policy_digest: str
    memory_digest: str
    environment_digest: str
    lineage: str
    candidate_pool_digest: str | None = None


@dataclass(frozen=True)
class Experience:
    attempt_id: str
    parent: str
    candidate: dict | None
    observation: dict | None
    outcome: str
    evidence: dict
    cost: dict


@dataclass(frozen=True)
class Result:
    data: dict
    cost: Cost
    status: str = "ok"
    detail: str = ""
