"""Contrastive Causal Attribution (CCA) signal fusion and rule updates."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..core.rules import ConstraintRule, RuleBeliefGraph
from ..core.trajectory import TrajectoryBuffer
from ..core.types import EvidenceVector, HcaLabel, Judgment


@dataclass(frozen=True)
class CcaConfig:
    default_weights: tuple[float, float, float] = (0.35, 0.25, 0.40)
    environment_weights: tuple[float, float, float] = (0.55, 0.20, 0.25)
    silent_reward_weights: tuple[float, float, float] = (0.35, 0.10, 0.55)
    invalid_hca_weights: tuple[float, float, float] = (0.583, 0.417, 0.0)
    theta_minus: float = -0.20
    theta_plus: float = 0.20
    eta: float = 0.5
    score_range: float = 100.0
    sps_window: int = 8
    gamma: float = 0.9


HCA_EFFECTS: dict[HcaLabel, tuple[float, str | None]] = {
    HcaLabel.IDENTICAL: (0.0, None),
    HcaLabel.PARAM_DIFF: (-0.4, "G_env.E_1"),
    HcaLabel.MISSING_PRECOND: (-0.8, "G_beh"),
    HcaLabel.WRONG_OBJECT: (-0.6, "G_env.E_1"),
    HcaLabel.WRONG_TIMING: (-0.7, "G_env.E_3"),
    HcaLabel.BOTH_WRONG: (0.0, None),
}


class ContrastiveCausalAttribution:
    def __init__(
        self, config: CcaConfig | None = None, trajectory_buffer: TrajectoryBuffer | None = None
    ) -> None:
        self.config = config or CcaConfig()
        self.trajectory_buffer = trajectory_buffer or TrajectoryBuffer(
            sps_window=self.config.sps_window, gamma=self.config.gamma
        )

    @staticmethod
    def majority(votes: Iterable[int]) -> int:
        values = list(votes)
        if not values:
            return 0
        counts = {value: values.count(value) for value in (-1, 0, 1)}
        return max(counts, key=lambda value: (counts[value], value))

    def fuse(
        self, evidence: EvidenceVector, *, mode: str = "default", hca_valid: bool = True
    ) -> float:
        weights = self._weights_for(mode, hca_valid)
        total = sum(weights)
        if total <= 0:
            raise ValueError("CCA weights must have a positive sum")
        return (
            sum(weight * value for weight, value in zip(weights, evidence.as_list(), strict=True))
            / total
        )

    def judgment(self, q_value: float) -> Judgment:
        if q_value > self.config.theta_plus:
            return Judgment.GOOD
        if q_value < self.config.theta_minus:
            return Judgment.BAD
        return Judgment.UNCERTAIN

    def score_progression(self, score_delta: float) -> float:
        return self.trajectory_buffer.observe_score(score_delta, self.config.score_range)

    def update_rules(
        self,
        graph: RuleBeliefGraph,
        *,
        action: str,
        judgment: Judgment,
        q_value: float,
        trial_rule_id: str | None = None,
        tentative_rules: list[ConstraintRule] | None = None,
        hypothesis_factory: Callable[[str, HcaLabel | None, Any], ConstraintRule | None]
        | None = None,
        hca_label: HcaLabel | None = None,
        evidence: Any = None,
    ) -> list[dict[str, Any]]:
        if trial_rule_id is not None:
            return self._update_trial_rule(graph, trial_rule_id, judgment)
        if tentative_rules:
            return self._update_tentative_rules(tentative_rules, hca_label, judgment, q_value)
        if judgment == Judgment.BAD and hypothesis_factory is not None:
            return self._update_hypothesis(graph, action, hca_label, evidence, hypothesis_factory)
        return []

    def _weights_for(self, mode: str, hca_valid: bool) -> tuple[float, float, float]:
        if not hca_valid:
            return self.config.invalid_hca_weights
        if mode == "environment_error":
            return self.config.environment_weights
        if mode == "silent_reward":
            return self.config.silent_reward_weights
        return self.config.default_weights

    @staticmethod
    def _update_trial_rule(
        graph: RuleBeliefGraph, rule_id: str, judgment: Judgment
    ) -> list[dict[str, Any]]:
        rule = graph.get(rule_id)
        if rule is None:
            return []
        if judgment == Judgment.GOOD:
            rule.update(refute=1.0)
            return [{"rule_id": rule.rule_id, "refute": 1.0}]
        if judgment == Judgment.BAD:
            rule.update(support=1.0)
            return [{"rule_id": rule.rule_id, "support": 1.0}]
        return []

    def _update_tentative_rules(
        self,
        rules: list[ConstraintRule],
        hca_label: HcaLabel | None,
        judgment: Judgment,
        q_value: float,
    ) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        _, target = self.hca_effect(hca_label)
        for rule in rules:
            rule_target = rule.metadata.get("graph_type")
            if target is not None and rule_target is not None and target != rule_target:
                continue
            update = self._tentative_update(rule, judgment, q_value)
            if update is not None:
                updates.append(update)
        return updates

    def _tentative_update(
        self, rule: ConstraintRule, judgment: Judgment, q_value: float
    ) -> dict[str, Any] | None:
        if judgment == Judgment.GOOD:
            rule.update(refute=1.0)
            return {"rule_id": rule.rule_id, "refute": 1.0}
        if judgment == Judgment.BAD:
            rule.update(support=1.0)
            return {"rule_id": rule.rule_id, "support": 1.0}
        if q_value > 0:
            rule.update(refute=self.config.eta)
            return {"rule_id": rule.rule_id, "refute": self.config.eta}
        if q_value < 0:
            rule.update(support=self.config.eta)
            return {"rule_id": rule.rule_id, "support": self.config.eta}
        return None

    @staticmethod
    def _update_hypothesis(
        graph: RuleBeliefGraph,
        action: str,
        hca_label: HcaLabel | None,
        evidence: Any,
        hypothesis_factory: Callable[[str, HcaLabel | None, Any], ConstraintRule | None],
    ) -> list[dict[str, Any]]:
        candidate = hypothesis_factory(action, hca_label, evidence)
        if candidate is None:
            return []
        existing = graph.get_by_symbolic_key(candidate.symbolic_key)
        if existing is not None:
            existing.update(support=1.0)
            return [{"rule_id": existing.rule_id, "support": 1.0}]
        graph.add(candidate, merge=False)
        return [{"rule_id": candidate.rule_id, "created": True}]

    @staticmethod
    def hca_effect(label: HcaLabel | None) -> tuple[float, str | None]:
        return HCA_EFFECTS[label] if label is not None else (0.0, None)
