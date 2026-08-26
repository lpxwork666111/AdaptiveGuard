"""Trajectory and successful-action statistics used by PFT criticality."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


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
        normalized = max(-1.0, min(1.0, score_delta / max(abs(score_range), 1e-8)))
        self._score_deltas.append(normalized)
        weights = [self.gamma**i for i in range(len(self._score_deltas))]
        value = sum(
            w * x for w, x in zip(reversed(weights), self._score_deltas, strict=True)
        ) / max(sum(weights), 1e-8)
        return max(-1.0, min(1.0, value))

    def action_frequency(self, action: str) -> float:
        if not self._successful:
            return 0.0
        return min(1.0, self._visit_counts[self._template(action)] / len(self._successful))

    def path_relevance(self, action: str) -> float:
        """Approximate shortest-path relevance using position in successful trajectories."""
        positions: list[float] = []
        template = self._template(action)
        for trajectory in self._successful:
            for index, step in enumerate(trajectory):
                if self._template(step.action) == template:
                    positions.append(1.0 - index / max(len(trajectory), 1))
        return max(positions, default=0.0)

    def bottleneck(self, action: str) -> float:
        if not self._successful:
            return 0.0
        template = self._template(action)
        return (
            1.0
            if self._visit_counts[template] >= len(self._successful)
            else self.action_frequency(action)
        )

    def criticality(self, action: str, epsilon0: float = 0.1) -> float:
        if not self._successful:
            return 0.5
        return min(
            1.0,
            max(self.action_frequency(action), self.path_relevance(action), self.bottleneck(action))
            + epsilon0,
        )

    @staticmethod
    def _template(action: str) -> str:
        return " ".join(str(action).strip().lower().split())
