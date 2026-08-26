"""Build initial rule beliefs from structured demonstrations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from ..core.rules import ConstraintRule, RuleBeliefGraph
from ..core.types import RuleTier


def rules_from_demonstrations(
    records: Iterable[Mapping[str, Any]], environment: str | None = None
) -> RuleBeliefGraph:
    """Aggregate records containing head/relation/tail into supported rules.

    Each symbolic occurrence contributes one support count, yielding Beta(n, 1).
    Records may carry an explicit ``rule`` object or the triplet at the top level.
    """

    counts: Counter[tuple[str, str, str]] = Counter()
    metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        source = record.get("rule", record)
        key = (str(source["head"]), str(source["relation"]), str(source["tail"]))
        counts[key] += 1
        metadata[key] = dict(source.get("metadata", {}))
    rules = []
    for (head, relation, tail), count in counts.items():
        rules.append(
            ConstraintRule(
                head,
                relation,
                tail,
                alpha=float(count),
                beta=1.0,
                tier=RuleTier.CONFIRMED if count > 3 else RuleTier.TENTATIVE,
                source="demonstration",
                environment=environment,
                metadata=metadata[(head, relation, tail)],
            )
        )
    return RuleBeliefGraph(rules)
