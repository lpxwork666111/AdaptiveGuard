"""Minimal custom planner implementation."""

from collections.abc import Sequence
from typing import Any

from adaptiveguard.core.types import ConfidenceLabel, PlannerOutput
from adaptiveguard.planners.base import Planner


class FirstActionPlanner(Planner):
    def plan(
        self,
        observation: Any,
        goal: str,
        history: Sequence[Any],
        available_actions: Sequence[str] | None = None,
    ) -> PlannerOutput:
        if not available_actions:
            raise ValueError("FirstActionPlanner needs an action list")
        return PlannerOutput(
            action=available_actions[0],
            expected_outcome="the selected action changes the current state",
            confidence_label=ConfidenceLabel.MEDIUM,
        )
