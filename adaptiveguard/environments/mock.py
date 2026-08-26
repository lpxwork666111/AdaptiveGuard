"""Small deterministic environment for local end-to-end checks."""

from __future__ import annotations

from ..core.types import EnvironmentTransition
from .base import InteractiveEnvironment


class MockEnvironment(InteractiveEnvironment):
    score_range = 1.0

    def __init__(self, required_actions: list[str] | None = None) -> None:
        self.required_actions = required_actions or ["inspect", "finish"]
        self.index = 0

    def reset(self) -> str:
        self.index = 0
        return f"Need action: {self.required_actions[0]}"

    def available_actions(self) -> list[str]:
        return self.required_actions

    def step(self, action: str) -> EnvironmentTransition:
        expected = self.required_actions[self.index]
        if action != expected:
            return EnvironmentTransition(
                f"Expected {expected}",
                -1.0,
                False,
                {"valid": False},
                float(self.index),
                "unexpected action",
            )
        self.index += 1
        done = self.index >= len(self.required_actions)
        observation = (
            "Goal complete" if done else f"Need action: {self.required_actions[self.index]}"
        )
        return EnvironmentTransition(
            observation, 1.0, done, {"valid": True, "success": done}, float(self.index)
        )
