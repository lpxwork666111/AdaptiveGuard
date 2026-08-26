"""Judge protocols used by CCA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ..core.types import HcaResult, JudgeResult


class ProcessRewardJudge(ABC):
    @abstractmethod
    def judge(
        self,
        action: str,
        observation: Any,
        reward: float,
        score_delta: float,
        environment_feedback: Any = None,
    ) -> JudgeResult:
        raise NotImplementedError


class HcaJudge(ABC):
    @abstractmethod
    def judge(
        self,
        action: str,
        expected_outcome: str,
        observation: Any,
        reward: float,
        context: Mapping[str, Any] | None = None,
    ) -> HcaResult:
        raise NotImplementedError
