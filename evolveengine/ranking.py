"""Deterministic, data-only candidate policies."""

from dataclasses import asdict

from .schema import ACTIONS, BANDS, CLASSES, FEATURES, Candidate, Observation, in_scope


def policy_rules(policy):
    if type(policy) is not dict or set(policy) != {"rules"}:
        raise ValueError("Policy requires exactly a rules field")
    rules = policy["rules"]
    if type(rules) is not list or len(rules) > 8:
        raise ValueError("A policy can contain up to eight rules")
    for rule in rules:
        if type(rule) is not dict or set(rule) != {"when", "promote_action"}:
            raise ValueError("Unknown rule fields")
        when = rule["when"]
        if (type(when) is not dict or not when
                or any(key not in FEATURES or value not in FEATURES[key] for key, value in when.items())
                or rule["promote_action"] not in ACTIONS):
            raise ValueError("Rule conditions and actions must use the registered vocabulary")
    return rules


def freeze_pool(raw, allowed_scope):
    if type(raw) is not list or len(raw) > 4:
        raise ValueError("A generator response may contain at most four candidates")
    candidates = [Candidate(**item) for item in raw]
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate ID")
    kept = {}
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if all(in_scope(path, allowed_scope) for path in candidate.target_scope):
            key = (candidate.action_tag, candidate.target_scope)
            kept.setdefault(key, asdict(candidate))
    return list(kept.values())


def rank(observation, pool, policy):
    observation = asdict(Observation(**observation))
    candidates = [Candidate(**item) for item in pool]
    primary = observation["primary_class"]
    promotions = [
        rule["promote_action"] for rule in policy_rules(policy)
        if all(observation[key] == value for key, value in rule["when"].items())
    ]

    def priority(candidate):
        mismatch = primary != "unknown" and candidate.bottleneck_class != primary
        action = candidate.action_tag
        rule_order = len(promotions)
        if action in promotions:
            rule_order = promotions.index(action)
        return (rule_order, mismatch, BANDS.index(candidate.cost_band),
                CLASSES.index(candidate.bottleneck_class), candidate.candidate_id)

    return [candidate.candidate_id for candidate in sorted(candidates, key=priority)]


def replay(parent, proposed, prepared_steps):
    policy_rules(proposed)
    comparisons = []
    for step in prepared_steps:
        old = rank(step["observation"], step["pool"], parent)
        new = rank(step["observation"], step["pool"], proposed)
        comparisons.append({"pool_digest": step["pool_digest"], "parent": old, "proposed": new,
                            "first_choice_changed": old[:1] != new[:1]})
    return comparisons
