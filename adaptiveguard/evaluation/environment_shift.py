"""Episode-indexed environment intervention hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ScheduledIntervention:
    def __init__(self, episode: int, apply: Callable[[Any], None]) -> None:
        self.episode = episode
        self.apply = apply
        self.applied = False

    def maybe_apply(self, episode: int, environment: Any) -> bool:
        if not self.applied and episode >= self.episode:
            self.apply(environment)
            self.applied = True
            return True
        return False
