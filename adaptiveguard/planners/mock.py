"""Deterministic planner useful for smoke tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.types import ConfidenceLabel, PlannerOutput
from .base import Planner


class MockPlanner(Planner):
    def __init__(self, actions: Sequence[str] | None = None) -> None:
        self.actions = list(actions or ["look around", "wait"])
        self._index = 0

    def plan(
        self,
        observation: Any,
        goal: str,
        history: Sequence[Any],
        available_actions: Sequence[str] | None = None,
    ) -> PlannerOutput:
        candidates = list(available_actions or self.actions)
        if not candidates:
            raise ValueError("planner received no candidate actions")
        action = candidates[self._index % len(candidates)]
        self._index += 1
        return PlannerOutput(
            action, "the environment transitions to the next state", ConfidenceLabel.MEDIUM
        )
