"""Merge independently sourced belief graphs while retaining provenance."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.rules import RuleBeliefGraph


def merge_rule_sources(graphs: Iterable[RuleBeliefGraph]) -> RuleBeliefGraph:
    merged = RuleBeliefGraph()
    for graph in graphs:
        for rule in graph:
            existing = merged.get_by_symbolic_key(rule.symbolic_key)
            if existing is not None:
                sources = set(existing.metadata.get("sources", [existing.source]))
                sources.add(rule.source)
                existing.metadata["sources"] = sorted(sources)
            merged.add(rule, merge=True)
    return merged
