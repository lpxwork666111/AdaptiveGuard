"""Consistency safeguards for belief graph edits."""

from ..core.rules import RuleBeliefGraph


def check_and_resolve(graph: RuleBeliefGraph) -> list[tuple[str, str]]:
    return graph.resolve_contradictions()
