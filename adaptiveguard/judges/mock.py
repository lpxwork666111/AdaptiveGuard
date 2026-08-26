"""Simple judges for deterministic local execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..core.types import HcaLabel, HcaResult, JudgeResult
from .base import HcaJudge, ProcessRewardJudge


class MockProcessRewardJudge(ProcessRewardJudge):
    def judge(
        self,
        action: str,
        observation: Any,
        reward: float,
        score_delta: float,
        environment_feedback: Any = None,
    ) -> JudgeResult:
        value = 1 if reward > 0 or score_delta > 0 else (-1 if reward < 0 or score_delta < 0 else 0)
        return JudgeResult(
            "GOOD" if value > 0 else "BAD" if value < 0 else "UNCERTAIN", value, (value,)
        )


class MockHcaJudge(HcaJudge):
    def judge(
        self,
        action: str,
        expected_outcome: str,
        observation: Any,
        reward: float,
        context: Mapping[str, Any] | None = None,
    ) -> HcaResult:
        if reward > 0:
            return HcaResult(HcaLabel.IDENTICAL, 0.0, valid=True, votes=(HcaLabel.IDENTICAL.value,))
        return HcaResult(
            HcaLabel.MISSING_PRECOND,
            -0.8,
            "The action requires a precondition that was not satisfied.",
            valid=True,
            votes=(HcaLabel.MISSING_PRECOND.value,),
        )
