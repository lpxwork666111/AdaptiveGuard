"""Internal statistics used by the trajectory buffer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def canonical_action(action: Any) -> str:
    return " ".join(str(action).strip().lower().split())


def normalize_score_delta(score_delta: float, score_range: float) -> float:
    return max(-1.0, min(1.0, score_delta / max(abs(score_range), 1e-8)))


def discounted_mean(values: Sequence[float], gamma: float) -> float:
    if not values:
        return 0.0
    weights = [gamma**index for index in range(len(values))]
    weighted = sum(weight * value for weight, value in zip(reversed(weights), values, strict=True))
    return max(-1.0, min(1.0, weighted / max(sum(weights), 1e-8)))


def action_frequency(
    successful: Iterable[Iterable[Any]], visit_counts: Mapping[str, int], action: Any
) -> float:
    trajectories = list(successful)
    if not trajectories:
        return 0.0
    template = canonical_action(action)
    return min(1.0, visit_counts.get(template, 0) / len(trajectories))


def path_relevance(successful: Iterable[Iterable[Any]], action: Any) -> float:
    template = canonical_action(action)
    positions: list[float] = []
    for trajectory in successful:
        steps = list(trajectory)
        positions.extend(
            1.0 - index / max(len(steps), 1)
            for index, step in enumerate(steps)
            if canonical_action(step.action) == template
        )
    return max(positions, default=0.0)


def bottleneck(
    successful: Iterable[Iterable[Any]], visit_counts: Mapping[str, int], action: Any
) -> float:
    trajectories = list(successful)
    if not trajectories:
        return 0.0
    template = canonical_action(action)
    if visit_counts.get(template, 0) >= len(trajectories):
        return 1.0
    return action_frequency(trajectories, visit_counts, action)


def criticality(
    successful: Iterable[Iterable[Any]],
    visit_counts: Mapping[str, int],
    action: Any,
    epsilon0: float,
) -> float:
    trajectories = list(successful)
    if not trajectories:
        return 0.5
    value = max(
        action_frequency(trajectories, visit_counts, action),
        path_relevance(trajectories, action),
        bottleneck(trajectories, visit_counts, action),
    )
    return min(1.0, value + epsilon0)
