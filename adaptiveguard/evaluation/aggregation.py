"""Seed-aware aggregation utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import mean, stdev


def summarize(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    if not data:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "sem": 0.0}
    deviation = stdev(data) if len(data) > 1 else 0.0
    return {
        "count": float(len(data)),
        "mean": mean(data),
        "std": deviation,
        "sem": deviation / math.sqrt(len(data)),
    }
