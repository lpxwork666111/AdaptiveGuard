"""Deterministic goal-criticality calculation and caching."""

from ..core.trajectory import TrajectoryBuffer


def goal_criticality(buffer: TrajectoryBuffer, action: str, epsilon0: float = 0.1) -> float:
    return buffer.criticality(action, epsilon0)
