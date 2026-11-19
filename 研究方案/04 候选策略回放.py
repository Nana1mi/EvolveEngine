"""回放同一观测和候选池上的两种排序。只产生行为证据，不产生性能或晋级分数。"""

import argparse
import hashlib
import json
from pathlib import Path


CLASSES = ("host_dispatch", "compute", "memory", "kv_cache")
PHASES = ("prefill", "decode", "mixed", "unknown")
GAPS = ("low", "high", "unknown")
COSTS = ("low", "medium", "high")
ACTIONS = ("pack_request_metadata", "reduce_launch_gap", "improve_locality", "fuse_compute", "manage_kv_cache", "other")

EXAMPLE = {
    "observation": {
        "primary_class": "host_dispatch", "phase": "decode", "host_launch_gap": "high"
    },
    "pool": [
        {"id": "p", "class": "host_dispatch", "cost_band": "low", "action_tag": "pack_request_metadata"},
        {"id": "s", "class": "host_dispatch", "cost_band": "medium", "action_tag": "reduce_launch_gap"},
    ],
    "rule": {
        "when": {"primary_class": "host_dispatch", "phase": "decode", "host_launch_gap": "high"},
        "promote_action": "reduce_launch_gap",
    },
}

FEATURES = {"primary_class": (*CLASSES, "unknown"), "phase": PHASES, "host_launch_gap": GAPS}


def validate_fields(record, allowed, label):
    if not isinstance(record, dict) or set(record) != set(allowed):
        raise ValueError(f"{label} 必须包含且仅包含 {sorted(allowed)}")


def check_features(record):
    for key, value in record.items():
        if key not in FEATURES or value not in FEATURES[key]:
            raise ValueError(f"未登记的观测字段或分档 {key}={value!r}")


def validate(record):
    validate_fields(record, ("observation", "pool", "rule"), "输入")
    observation, pool, rule = record["observation"], record["pool"], record["rule"]
    validate_fields(observation, FEATURES, "观测")
    check_features(observation)
    validate_fields(rule, ("when", "promote_action"), "候选规则")
    if not isinstance(rule["when"], dict) or not rule["when"]:
        raise ValueError("候选规则需要至少一个登记的触发条件")
    check_features(rule["when"])
    if rule["promote_action"] not in ACTIONS:
        raise ValueError("候选规则引用了未登记的动作")
    if not isinstance(pool, list):
        raise ValueError("候选池必须是列表")
    ids = set()
    for candidate in pool:
        validate_fields(candidate, ("id", "class", "cost_band", "action_tag"), "候选")
        key = candidate["id"]
        if not isinstance(key, str) or not key.strip() or key in ids:
            raise ValueError("候选 ID 必须为非空且唯一的字符串")
        ids.add(key)
        if (candidate["class"] not in CLASSES or candidate["cost_band"] not in COSTS
                or candidate["action_tag"] not in ACTIONS):
            raise ValueError(f"{key} 使用了未登记的候选特征")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def rank(observation, pool, rule=None):
    matches = rule is not None and all(observation[k] == v for k, v in rule["when"].items())

    def priority(candidate):
        primary = observation["primary_class"]
        mismatch = primary != "unknown" and candidate["class"] != primary
        # 命中的登记动作可跨类提前；未命中或同档候选仍按 H0 排序。
        promoted = matches and candidate["action_tag"] == rule["promote_action"]
        return (not promoted, mismatch, COSTS.index(candidate["cost_band"]),
                CLASSES.index(candidate["class"]), candidate["id"])

    return [item["id"] for item in sorted(pool, key=priority)]


def replay(record):
    validate(record)
    observation, pool, rule = record["observation"], record["pool"], record["rule"]
    parent = rank(observation, pool)
    child = rank(observation, pool, rule)
    return {
        "input_digest": digest({"observation": observation, "pool": sorted(pool, key=lambda c: c["id"])}),
        "rule_digest": digest(rule),
        "condition_matched": all(observation[k] == v for k, v in rule["when"].items()),
        "h0_order": parent,
        "proposed_order": child,
        "first_choice_changed": parent[:1] != child[:1],
    }


def demo():
    results = {}
    for gap, expected in (("high", ["s", "p"]), ("low", ["p", "s"]), ("unknown", ["p", "s"])):
        record = {**EXAMPLE, "observation": {**EXAMPLE["observation"], "host_launch_gap": gap}}
        result = replay(record)
        reversed_result = replay({**record, "pool": list(reversed(record["pool"]))})
        if result["h0_order"] != ["p", "s"] or result["proposed_order"] != expected or result != reversed_result:
            raise RuntimeError(f"{gap} 分档的排序或置换检查失败")
        results[gap] = result
    return {"fixture": "synthetic", "replays": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--example", action="store_true", help="输出可编辑的 JSON 输入")
    group.add_argument("--input", type=Path, help="读取一份 UTF-8 JSON 输入并回放")
    args = parser.parse_args()
    try:
        if args.example:
            result = EXAMPLE
        elif args.input is not None:
            result = replay(json.loads(args.input.read_text(encoding="utf-8-sig")))
        else:
            result = demo()
    except (OSError, ValueError) as error:
        parser.exit(2, f"回放失败：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
