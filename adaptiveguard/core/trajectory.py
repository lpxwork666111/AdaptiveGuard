"""Trajectory and successful-action statistics used by PFT criticality."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ._trajectory_stats import (
    action_frequency,
    bottleneck,
    canonical_action,
    criticality,
    discounted_mean,
    normalize_score_delta,
    path_relevance,
)


@dataclass(frozen=True)
class StepRecord:
    action: str
    observation: Any
    reward: float
    score: float | None = None
    done: bool = False
    info: dict[str, Any] | None = None


class TrajectoryBuffer:
    def __init__(
        self, max_trajectories: int = 100, sps_window: int = 8, gamma: float = 0.9
    ) -> None:
        self.max_trajectories = max_trajectories
        self.sps_window = sps_window
        self.gamma = gamma
        self._successful: deque[list[StepRecord]] = deque(maxlen=max_trajectories)
        self._score_deltas: deque[float] = deque(maxlen=sps_window)
        self._visit_counts: Counter[str] = Counter()
        self._goal_counts: Counter[str] = Counter()

    @property
    def successful_trajectories(self) -> list[list[StepRecord]]:
        return list(self._successful)

    def add_trajectory(self, trajectory: Iterable[StepRecord], success: bool) -> None:
        trajectory_list = list(trajectory)
        if success and trajectory_list:
            self._successful.append(trajectory_list)
            for step in trajectory_list:
                self._visit_counts[self._template(step.action)] += 1

    def observe_score(self, score_delta: float, score_range: float = 100.0) -> float:
        self._score_deltas.append(normalize_score_delta(score_delta, score_range))
        return discounted_mean(self._score_deltas, self.gamma)

    def action_frequency(self, action: str) -> float:
        return action_frequency(self._successful, self._visit_counts, action)

    def path_relevance(self, action: str) -> float:
        """Approximate shortest-path relevance using position in successful trajectories."""
        return path_relevance(self._successful, action)

    def bottleneck(self, action: str) -> float:
        return bottleneck(self._successful, self._visit_counts, action)

    def criticality(self, action: str, epsilon0: float = 0.1) -> float:
        return criticality(self._successful, self._visit_counts, action, epsilon0)

    @staticmethod
    def _template(action: str) -> str:
        return canonical_action(action)
