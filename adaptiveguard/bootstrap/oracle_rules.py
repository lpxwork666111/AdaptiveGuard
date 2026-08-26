"""Oracle-trajectory helpers."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.rules import ConstraintRule, RuleBeliefGraph


def precedence_rules(actions: Iterable[str], environment: str | None = None) -> RuleBeliefGraph:
    sequence = [str(action).strip() for action in actions if str(action).strip()]
    rules = []
    for before, after in zip(sequence, sequence[1:], strict=False):
        rules.append(
            ConstraintRule(
                head=after,
                relation="requires_previous",
                tail=before,
                alpha=1.0,
                beta=1.0,
                source="oracle",
                environment=environment,
                metadata={"action_template": after.split()[0]},
            )
        )
    return RuleBeliefGraph(rules)
