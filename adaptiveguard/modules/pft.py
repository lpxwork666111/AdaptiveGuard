"""Purposeful Falsification Trials (PFT)."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.budget import TrialBudget
from ..core.rules import ConstraintRule
from ..core.trajectory import TrajectoryBuffer
from ..core.types import ActionDecision, PftDecision


@dataclass(frozen=True)
class PftConfig:
    tau: float = 0.25
    epsilon0: float = 0.1
    budget_base: float = 5.0
    budget_max: float | None = None
    budget_refund: float = 0.5


class PurposefulFalsificationTrials:
    def __init__(
        self,
        config: PftConfig | None = None,
        trajectory_buffer: TrajectoryBuffer | None = None,
    ) -> None:
        self.config = config or PftConfig()
        self.trajectory_buffer = trajectory_buffer or TrajectoryBuffer()
        self.budget = TrialBudget(
            base=self.config.budget_base,
            maximum=self.config.budget_max,
            refund=self.config.budget_refund,
        )
        self.failed_trials = 0

    def reset_episode(self) -> None:
        self.budget.reset()
        self.failed_trials = 0

    def decide(
        self,
        action: str,
        blockers: list[ConstraintRule],
        self_confidence: float,
        criticality: float | None = None,
    ) -> PftDecision:
        criticality = (
            self.trajectory_buffer.criticality(action, self.config.epsilon0)
            if criticality is None
            else max(0.0, min(1.0, criticality))
        )
        if not blockers:
            return PftDecision(
                ActionDecision.ALLOW, None, criticality=criticality, reason="no confirmed blocker"
            )
        confidence = max(0.0, min(1.0, self_confidence))
        scores = {r.rule_id: (1 - r.confidence) * confidence * criticality for r in blockers}
        selected = max(scores, key=lambda rule_id: scores[rule_id])
        eligible = all(score > self.config.tau for score in scores.values())
        if eligible and self.budget.can_trial():
            return PftDecision(
                ActionDecision.TRIAL,
                selected,
                tuple(scores),
                scores,
                criticality,
                "all confirmed blockers pass challenge gate",
            )
        reason = (
            "budget exhausted"
            if not self.budget.can_trial()
            else "at least one blocker fails challenge gate"
        )
        return PftDecision(
            ActionDecision.BLOCK, selected, tuple(scores), scores, criticality, reason
        )

    def record_outcome(self, decision: ActionDecision, successful_or_good: bool | None) -> None:
        if decision != ActionDecision.TRIAL:
            return
        if successful_or_good is True:
            self.budget.on_success()
        else:
            self.budget.on_failure()
            self.failed_trials += 1
