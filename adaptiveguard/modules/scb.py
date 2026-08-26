"""Stratified Constraint Belief (SCB)."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.rules import ConstraintRule, RuleBeliefGraph
from ..core.types import RuleTier


@dataclass(frozen=True)
class ScbConfig:
    theta_low: float = 0.20
    theta_high: float = 0.70
    oscillation_window: int = 10

    def __post_init__(self) -> None:
        if not 0 <= self.theta_low < self.theta_high <= 1:
            raise ValueError("SCB thresholds must satisfy 0 <= low < high <= 1")


class StratifiedConstraintBelief:
    def __init__(
        self, graph: RuleBeliefGraph | None = None, config: ScbConfig | None = None
    ) -> None:
        self.graph = graph or RuleBeliefGraph()
        self.config = config or ScbConfig()

    def add_or_merge(self, rule: ConstraintRule) -> ConstraintRule:
        result = self.graph.add(rule, merge=True)
        result.refresh_tier(self.config.theta_low, self.config.theta_high)
        return result

    def confirmed_blockers(self, action: str, state: object = None) -> list[ConstraintRule]:
        return self.graph.matching(action, RuleTier.CONFIRMED, state)

    def tentative_matches(self, action: str, state: object = None) -> list[ConstraintRule]:
        return self.graph.matching(action, RuleTier.TENTATIVE, state)

    def update_rule(
        self, rule_id: str, *, support: float = 0.0, refute: float = 0.0
    ) -> ConstraintRule:
        rule = self.graph.get(rule_id)
        if rule is None:
            raise KeyError(f"unknown rule: {rule_id}")
        rule.update(support=support, refute=refute, window=self.config.oscillation_window)
        rule.refresh_tier(self.config.theta_low, self.config.theta_high)
        return rule

    def refresh(self) -> None:
        self.graph.refresh_tiers(self.config.theta_low, self.config.theta_high)

    def consistency_check(self) -> list[tuple[str, str]]:
        contradictions = self.graph.resolve_contradictions()
        self.graph.enforce_acyclicity()
        return contradictions
