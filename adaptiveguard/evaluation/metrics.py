"""Environment-independent summaries for controller runs and rule graphs."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any

from ..core.rules import RuleBeliefGraph
from ..core.types import RuleTier


def aggregate_run(episodes: Iterable[dict[str, Any]]) -> dict[str, float]:
    records = list(episodes)
    if not records:
        return {"episodes": 0.0}
    total_steps = sum(float(item.get("steps", 0)) for item in records)
    total_tokens = sum(
        float(item.get("model_usage", {}).get("total_tokens", 0)) for item in records
    )
    total_calls = sum(float(item.get("model_usage", {}).get("calls", 0)) for item in records)
    return {
        "episodes": float(len(records)),
        "mean_steps": mean(float(item.get("steps", 0)) for item in records),
        "mean_total_reward": mean(float(item.get("total_reward", 0.0)) for item in records),
        "completion_rate": mean(float(bool(item.get("done"))) for item in records),
        "mean_failed_trials": mean(float(item.get("failed_trials", 0)) for item in records),
        "model_calls": total_calls,
        "tokens_per_executed_action": total_tokens / total_steps if total_steps else 0.0,
    }


def rule_graph_statistics(graph: RuleBeliefGraph) -> dict[str, float]:
    rules = list(graph)
    return {
        "rules": float(len(rules)),
        "tentative": float(len(graph.by_tier(RuleTier.TENTATIVE))),
        "confirmed": float(len(graph.by_tier(RuleTier.CONFIRMED))),
        "deprecated": float(len(graph.by_tier(RuleTier.DEPRECATED))),
        "mean_confidence": mean(rule.confidence for rule in rules) if rules else 0.0,
    }


def symmetric_difference_size(left: RuleBeliefGraph, right: RuleBeliefGraph) -> int:
    left_keys = {rule.symbolic_key for rule in left if rule.tier != RuleTier.DEPRECATED}
    right_keys = {rule.symbolic_key for rule in right if rule.tier != RuleTier.DEPRECATED}
    return len(left_keys.symmetric_difference(right_keys))


def conservative_bias_proxy(candidate: RuleBeliefGraph, reference: RuleBeliefGraph) -> float:
    reference_keys = {rule.symbolic_key for rule in reference}
    return sum(
        rule.confidence
        for rule in candidate
        if rule.symbolic_key not in reference_keys and rule.tier != RuleTier.DEPRECATED
    )
