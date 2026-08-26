"""OpenAI-compatible PRJ and HCA judges."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.types import HcaLabel, HcaResult, JudgeResult
from ..planners.llm import OpenAICompatibleClient
from .audit import audit_judge_prompt
from .base import HcaJudge, ProcessRewardJudge

PRJ_SYSTEM = """Judge an executed action using only the action and environment feedback.
Return JSON with label (+1, 0, or -1) and rationale. +1 means useful/progressing,
-1 means failed/harmful, and 0 means ambiguous. Do not infer hidden rules."""

HINT_SYSTEM = """Compare the predicted outcome with environment feedback. Return JSON
with useful (boolean) and hint (a concise fact that would improve the next action)."""

TEACHER_SYSTEM = """Given an observation, goal, failed action, and hindsight hint,
return JSON with action containing a better next action. Do not include hidden rules."""

HCA_SYSTEM = """Classify the difference between an executed action and a teacher action.
Return JSON with label, which must be IDENTICAL, PARAM_DIFF, MISSING_PRECOND,
WRONG_OBJECT, WRONG_TIMING, or BOTH_WRONG."""


class LLMProcessRewardJudge(ProcessRewardJudge):
    def __init__(self, client: OpenAICompatibleClient, votes: int = 3) -> None:
        self.client = client
        self.votes = max(1, votes)

    def judge(
        self,
        action: str,
        observation: Any,
        reward: float,
        score_delta: float,
        environment_feedback: Any = None,
    ) -> JudgeResult:
        prompt = json.dumps(
            {
                "action": action,
                "observation": observation,
                "reward": reward,
                "score_delta": score_delta,
                "environment_feedback": environment_feedback,
            },
            default=str,
        )
        audit_judge_prompt(prompt)
        values: list[int] = []
        rationales: list[str] = []
        for _ in range(self.votes):
            data, _ = self.client.chat_json(PRJ_SYSTEM, prompt)
            value = int(data.get("label", 0))
            values.append(max(-1, min(1, value)))
            rationales.append(str(data.get("rationale", "")))
        signed = sum(values) / len(values)
        majority = max((-1, 0, 1), key=lambda item: (values.count(item), item))
        return JudgeResult(str(majority), signed, values, next((r for r in rationales if r), None))


class LLMHcaJudge(HcaJudge):
    def __init__(
        self, client: OpenAICompatibleClient, hint_votes: int = 3, structured_votes: int = 3
    ) -> None:
        self.client = client
        self.hint_votes = max(1, hint_votes)
        self.structured_votes = max(1, structured_votes)

    def judge(
        self,
        action: str,
        expected_outcome: str,
        observation: Any,
        reward: float,
        context: Mapping[str, Any] | None = None,
    ) -> HcaResult:
        base = {
            "action": action,
            "expected_outcome": expected_outcome,
            "observation": observation,
            "environment_feedback": context.get("environment_feedback") if context else None,
        }
        prompt = json.dumps(base, default=str)
        audit_judge_prompt(prompt)
        hints: list[str] = []
        for _ in range(self.hint_votes):
            data, _ = self.client.chat_json(HINT_SYSTEM, prompt)
            hint = str(data.get("hint", "")).strip()
            if bool(data.get("useful")) and len(hint) > 10:
                hints.append(hint)
        selected_hint: str | None = max(hints, key=len, default=None)
        if selected_hint is None:
            return HcaResult(None, 0.0, valid=False)
        teacher_prompt = json.dumps(
            {
                "observation": observation,
                "goal": context.get("goal", "") if context else "",
                "action": action,
                "hint": selected_hint,
            },
            default=str,
        )
        audit_judge_prompt(teacher_prompt)
        teacher, _ = self.client.chat_json(TEACHER_SYSTEM, teacher_prompt)
        teacher_action = str(teacher.get("action", "")).strip()
        if not teacher_action:
            return HcaResult(None, 0.0, selected_hint, valid=False)
        structured_prompt = json.dumps(
            {"action": action, "teacher_action": teacher_action, "hint": selected_hint}
        )
        audit_judge_prompt(structured_prompt)
        labels: list[HcaLabel] = []
        for _ in range(self.structured_votes):
            data, _ = self.client.chat_json(HCA_SYSTEM, structured_prompt)
            try:
                labels.append(HcaLabel(str(data.get("label", "")).upper()))
            except ValueError:
                continue
        if not labels:
            return HcaResult(None, 0.0, selected_hint, valid=False)
        winner = max(set(labels), key=labels.count)
        if labels.count(winner) < 2 and self.structured_votes >= 3:
            return HcaResult(
                None,
                0.0,
                selected_hint,
                False,
                tuple(label.value for label in labels),
            )
        from ..modules.cca import HCA_EFFECTS

        value, _ = HCA_EFFECTS[winner]
        return HcaResult(
            winner,
            value,
            selected_hint,
            True,
            tuple(label.value for label in labels),
        )
