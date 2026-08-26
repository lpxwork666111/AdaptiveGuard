"""Rule-graph warm-start filtering across domains."""

from ..core.rules import RuleBeliefGraph


def transferable_rules(
    graph: RuleBeliefGraph, target_environment: str, include_unscoped: bool = True
) -> RuleBeliefGraph:
    return RuleBeliefGraph(
        rule
        for rule in graph
        if rule.environment == target_environment or (include_unscoped and rule.environment is None)
    )
