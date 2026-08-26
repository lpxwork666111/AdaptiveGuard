"""Closed-loop AdaptiveGuard controller."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..environments.adapters import observation_to_text
from ..judges.base import HcaJudge, ProcessRewardJudge
from ..modules.cca import ContrastiveCausalAttribution
from ..modules.pft import PurposefulFalsificationTrials
from ..modules.scb import StratifiedConstraintBelief
from ..planners.base import Planner
from .evidence import StepAudit
from .trajectory import StepRecord
from .types import ActionDecision, EvidenceVector, Judgment

logger = logging.getLogger(__name__)


class AdaptiveGuardController:
    def __init__(
        self,
        environment: Any,
        planner: Planner,
        scb: StratifiedConstraintBelief,
        pft: PurposefulFalsificationTrials,
        cca: ContrastiveCausalAttribution,
        process_judge: ProcessRewardJudge,
        hca_judge: HcaJudge,
        rule_factory: Callable[[str, Any, Any], Any] | None = None,
        usage_provider: Callable[[], dict[str, int]] | None = None,
        max_steps: int = 100,
        tier_refresh_interval: int = 10,
    ) -> None:
        self.environment = environment
        self.planner = planner
        self.scb = scb
        self.pft = pft
        self.cca = cca
        self.process_judge = process_judge
        self.hca_judge = hca_judge
        self.rule_factory = rule_factory
        self.usage_provider = usage_provider
        self.max_steps = max_steps
        self.tier_refresh_interval = max(1, tier_refresh_interval)
        self.history: list[dict[str, Any]] = []
        self.audits: list[StepAudit] = []

    def run_episode(self, goal: str, *, episode_id: str = "episode-0") -> dict[str, Any]:
        observation = self.environment.reset()
        usage_before = self.usage_provider() if self.usage_provider else {}
        self.history = []
        self.audits = []
        self.pft.reset_episode()
        trajectory: list[StepRecord] = []
        total_reward = 0.0
        final_score: float | None = None
        done = False

        for step_index in range(self.max_steps):
            actions = list(self.environment.available_actions())
            planner_output = self.planner.plan(
                observation_to_text(observation), goal, self.history, actions or None
            )
            blockers = self.scb.confirmed_blockers(planner_output.action, observation)
            tentative = self.scb.tentative_matches(planner_output.action, observation)
            budget_before = self.pft.budget.remaining
            pft_decision = self.pft.decide(
                planner_output.action, blockers, planner_output.self_confidence
            )
            if pft_decision.decision == ActionDecision.BLOCK:
                self.history.append(
                    {"blocked_action": planner_output.action, "reason": pft_decision.reason}
                )
                self.audits.append(
                    StepAudit(
                        step_index,
                        planner_output,
                        pft_decision,
                        budget_before=budget_before,
                        budget_after=budget_before,
                    )
                )
                continue

            transition = self.environment.step(planner_output.action)
            score = transition.score if transition.score is not None else final_score
            score_delta = 0.0 if score is None or final_score is None else score - final_score
            final_score = score
            total_reward += transition.reward
            trajectory.append(
                StepRecord(
                    planner_output.action,
                    transition.observation,
                    transition.reward,
                    score,
                    transition.done,
                    dict(transition.info),
                )
            )
            prj = self.process_judge.judge(
                planner_output.action,
                transition.observation,
                transition.reward,
                score_delta,
                transition.error or transition.info,
            )
            sps = self.cca.score_progression(score_delta)
            hca = self.hca_judge.judge(
                planner_output.action,
                planner_output.expected_outcome,
                transition.observation,
                transition.reward,
                {"goal": goal, "history": self.history, "environment_feedback": transition.error},
            )
            evidence = EvidenceVector(prj.signed_value, sps, hca.signed_value if hca.valid else 0.0)
            mode = (
                "environment_error"
                if transition.error
                else ("silent_reward" if transition.reward == 0 else "default")
            )
            q_value = self.cca.fuse(evidence, mode=mode, hca_valid=hca.valid)
            judgment = self.cca.judgment(q_value)
            updates = self.cca.update_rules(
                self.scb.graph,
                action=planner_output.action,
                judgment=judgment,
                q_value=q_value,
                trial_rule_id=pft_decision.selected_rule_id
                if pft_decision.decision == ActionDecision.TRIAL
                else None,
                tentative_rules=tentative
                if pft_decision.decision == ActionDecision.ALLOW
                else None,
                hypothesis_factory=self.rule_factory,
                hca_label=hca.label,
                evidence={
                    "prj": prj,
                    "sps": sps,
                    "hca": hca,
                    "environment_feedback": transition.error,
                },
            )
            good = judgment == Judgment.GOOD
            self.pft.record_outcome(
                pft_decision.decision,
                good if pft_decision.decision == ActionDecision.TRIAL else None,
            )
            if (step_index + 1) % self.tier_refresh_interval == 0:
                self.scb.refresh()
            self.scb.consistency_check()
            budget_after = self.pft.budget.remaining
            self.audits.append(
                StepAudit(
                    step_index,
                    planner_output,
                    pft_decision,
                    transition,
                    evidence,
                    judgment,
                    hca,
                    q_value,
                    budget_before,
                    budget_after,
                    updates,
                )
            )
            self.history.append(
                {
                    "action": planner_output.action,
                    "observation": observation_to_text(transition.observation),
                    "judgment": judgment.value,
                    "q": q_value,
                }
            )
            observation = transition.observation
            if transition.done:
                done = True
                break

        self.cca.trajectory_buffer.add_trajectory(trajectory, done)
        usage_after = self.usage_provider() if self.usage_provider else {}
        episode_usage = {
            key: usage_after.get(key, 0) - usage_before.get(key, 0) for key in usage_after
        }
        return {
            "episode_id": episode_id,
            "steps": len(trajectory),
            "total_reward": total_reward,
            "final_score": final_score,
            "done": done,
            "failed_trials": self.pft.failed_trials,
            "model_usage": episode_usage,
            "audits": [audit.to_dict() for audit in self.audits],
            "rules": self.scb.graph.to_dict(),
        }
