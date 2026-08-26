"""Deterministic source subsampling for sparse-supervision diagnostics."""

from __future__ import annotations

import random

from ..core.rules import RuleBeliefGraph


def subsample_rules(graph: RuleBeliefGraph, fraction: float, seed: int) -> RuleBeliefGraph:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    rules = list(graph)
    rng = random.Random(seed)
    count = max(1, round(len(rules) * fraction)) if rules else 0
    return RuleBeliefGraph(rng.sample(rules, count))
