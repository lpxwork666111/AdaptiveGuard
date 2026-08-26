"""Internal consistency queries for constraint-rule collections."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from .types import RuleTier


class _RuleView(Protocol):
    rule_id: str
    head: str
    relation: str
    tail: str
    tier: RuleTier
    metadata: dict[str, Any]


def find_contradictions(rules: Iterable[_RuleView]) -> list[tuple[str, str]]:
    grouped: dict[tuple[str, str], list[_RuleView]] = {}
    for rule in rules:
        grouped.setdefault((rule.head.lower(), rule.relation.lower()), []).append(rule)

    contradictions: list[tuple[str, str]] = []
    for candidates in grouped.values():
        for left in candidates:
            for right in candidates:
                if left.rule_id >= right.rule_id:
                    continue
                if _are_contradictory(left, right):
                    contradictions.append((left.rule_id, right.rule_id))
    return contradictions


def _are_contradictory(left: _RuleView, right: _RuleView) -> bool:
    if left.tier == RuleTier.DEPRECATED or right.tier == RuleTier.DEPRECATED:
        return False

    explicitly_linked = right.rule_id in left.metadata.get(
        "contradicts", []
    ) or left.rule_id in right.metadata.get("contradicts", [])
    left_tail = left.tail.strip().lower()
    right_tail = right.tail.strip().lower()
    left_negated = left_tail.startswith("not ")
    right_negated = right_tail.startswith("not ")
    negated_pair = left_negated != right_negated and (
        left_tail.removeprefix("not ") == right_tail.removeprefix("not ")
    )
    exclusive_alternatives = (
        bool(left.metadata.get("exclusive") or right.metadata.get("exclusive"))
        and left_tail != right_tail
    )
    return explicitly_linked or negated_pair or exclusive_alternatives


def find_behavioral_cycles(rules: Iterable[_RuleView]) -> list[tuple[str, ...]]:
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for rule in rules:
        if rule.tier == RuleTier.DEPRECATED or "require" not in rule.relation.lower():
            continue
        adjacency.setdefault(rule.head.lower(), []).append((rule.tail.lower(), rule.rule_id))

    visiting: set[str] = set()
    visited: set[str] = set()
    stack_nodes: list[str] = []
    stack_rules: list[str] = []
    found: set[tuple[str, ...]] = set()

    def visit(node: str) -> None:
        visiting.add(node)
        stack_nodes.append(node)
        for target, rule_id in adjacency.get(node, []):
            if target in visiting:
                index = stack_nodes.index(target)
                found.add(tuple(sorted([*stack_rules[index:], rule_id])))
            elif target not in visited:
                stack_rules.append(rule_id)
                visit(target)
                stack_rules.pop()
        stack_nodes.pop()
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        if node not in visited:
            visit(node)
    return sorted(found)
