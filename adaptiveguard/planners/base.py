"""Planner protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ..core.types import PlannerOutput


class Planner(ABC):
    @abstractmethod
    def plan(
        self,
        observation: Any,
        goal: str,
        history: Sequence[Any],
        available_actions: Sequence[str] | None = None,
    ) -> PlannerOutput:
        raise NotImplementedError
