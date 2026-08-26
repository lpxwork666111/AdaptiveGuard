"""Probabilistic constraint rules and the mutable belief graph."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._rule_consistency import find_behavioral_cycles, find_contradictions
from .types import RuleTier


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConstraintRule:
    """A symbolic rule with a Beta-distributed validity belief."""

    head: str
    relation: str
    tail: str
    alpha: float = 1.0
    beta: float = 1.0
    tier: RuleTier = RuleTier.TENTATIVE
    rule_id: str = ""
    source: str = "runtime"
    environment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    update_count: int = 0
    flip_history: list[int] = field(default_factory=list)
    frozen: bool = False
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta parameters must be positive")
        self.rule_id = self.rule_id or self.make_id(self.head, self.relation, self.tail)
        self.tier = RuleTier(self.tier)

    @staticmethod
    def make_id(head: str, relation: str, tail: str) -> str:
        payload = "\x1f".join(part.strip().lower() for part in (head, relation, tail))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def symbolic_key(self) -> tuple[str, str, str]:
        return (self.head.strip().lower(), self.relation.strip().lower(), self.tail.strip().lower())

    def update(self, *, support: float = 0.0, refute: float = 0.0, window: int = 10) -> None:
        if support < 0 or refute < 0:
            raise ValueError("support/refute updates must be non-negative")
        if self.frozen or (support == 0 and refute == 0):
            return
        before = self.confidence
        self.alpha += support
        self.beta += refute
        self.update_count += 1
        after = self.confidence
        if (before >= 0.5) != (after >= 0.5):
            self.flip_history.append(self.update_count)
            self.flip_history = self.flip_history[-window:]
            if len(self.flip_history) > 2:
                self.frozen = True
        self.updated_at = _utc_now()

    def refresh_tier(self, theta_low: float, theta_high: float) -> RuleTier:
        if not 0 <= theta_low < theta_high <= 1:
            raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
        old = self.tier
        if self.confidence < theta_low:
            self.tier = RuleTier.DEPRECATED
        elif self.confidence <= theta_high:
            self.tier = RuleTier.TENTATIVE
        else:
            self.tier = RuleTier.CONFIRMED
        if old != self.tier:
            self.updated_at = _utc_now()
        return self.tier

    def matches(self, action: str, state: Any = None, *, strict: bool = False) -> bool:
        """Match a rule to a state-action pair using serializable metadata predicates."""

        query = action.strip().lower()
        if not query:
            return False
        explicit_action = str(self.metadata.get("action", "")).strip().lower()
        if explicit_action and (
            explicit_action != query if strict else explicit_action not in query
        ):
            return False
        if self.metadata.get("action_template"):
            template = str(self.metadata["action_template"]).lower()
            if template not in query:
                return False
        elif not explicit_action:
            head = " ".join(self.head.strip().lower().replace("_", " ").split())
            if strict and head != query:
                return False
            if not strict and head not in query:
                return False
        conditions = self.metadata.get("when", self.metadata.get("block_when", {}))
        if not conditions:
            return True
        state_text = str(state).lower()
        required = conditions.get("observation_contains", [])
        forbidden = conditions.get("observation_not_contains", [])
        required_values = [required] if isinstance(required, str) else list(required)
        forbidden_values = [forbidden] if isinstance(forbidden, str) else list(forbidden)
        return all(str(value).lower() in state_text for value in required_values) and all(
            str(value).lower() not in state_text for value in forbidden_values
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "alpha": self.alpha,
            "beta": self.beta,
            "confidence": self.confidence,
            "tier": self.tier.value,
            "source": self.source,
            "environment": self.environment,
            "metadata": self.metadata,
            "update_count": self.update_count,
            "flip_history": self.flip_history,
            "frozen": self.frozen,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConstraintRule:
        values = dict(data)
        values.pop("confidence", None)
        return cls(**values)


class RuleBeliefGraph:
    """Mutable collection of rules with merge and consistency operations."""

    def __init__(self, rules: Iterable[ConstraintRule] = ()) -> None:
        self._rules: dict[str, ConstraintRule] = {}
        for rule in rules:
            self.add(rule)

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self) -> Iterator[ConstraintRule]:
        return iter(self._rules.values())

    def add(self, rule: ConstraintRule, *, merge: bool = True) -> ConstraintRule:
        existing = self.get_by_symbolic_key(rule.symbolic_key)
        if existing is not None and merge:
            existing.alpha += rule.alpha
            existing.beta += rule.beta
            existing.update_count += rule.update_count
            existing.updated_at = _utc_now()
            return existing
        if rule.rule_id in self._rules:
            raise ValueError(f"duplicate rule_id: {rule.rule_id}")
        self._rules[rule.rule_id] = rule
        return rule

    def remove(self, rule_id: str) -> None:
        self._rules.pop(rule_id)

    def get(self, rule_id: str) -> ConstraintRule | None:
        return self._rules.get(rule_id)

    def get_by_symbolic_key(self, key: tuple[str, str, str]) -> ConstraintRule | None:
        return next((r for r in self._rules.values() if r.symbolic_key == key), None)

    def by_tier(self, tier: RuleTier) -> list[ConstraintRule]:
        return [r for r in self if r.tier == tier]

    def matching(
        self, action: str, tier: RuleTier | None = None, state: Any = None
    ) -> list[ConstraintRule]:
        rules = self if tier is None else self.by_tier(tier)
        return [r for r in rules if r.matches(action, state)]

    def refresh_tiers(self, theta_low: float, theta_high: float) -> None:
        for rule in self:
            rule.refresh_tier(theta_low, theta_high)

    def consistency_check(self) -> list[tuple[str, str]]:
        """Return contradictory symbolic rules; callers may resolve by confidence."""
        return find_contradictions(self)

    def behavioral_cycles(self) -> list[tuple[str, ...]]:
        """Find cycles among active behavioral-precondition edges."""
        return find_behavioral_cycles(self)

    def enforce_acyclicity(self) -> list[tuple[str, ...]]:
        cycles = self.behavioral_cycles()
        for cycle in cycles:
            loser = min(
                (self._rules[rule_id] for rule_id in cycle), key=lambda rule: rule.confidence
            )
            loser.tier = RuleTier.DEPRECATED
            loser.metadata["deprecated_reason"] = "behavioral_cycle"
            loser.updated_at = _utc_now()
        return cycles

    def resolve_contradictions(self) -> list[tuple[str, str]]:
        resolved = self.consistency_check()
        for left_id, right_id in resolved:
            left, right = self._rules[left_id], self._rules[right_id]
            loser = right if left.confidence >= right.confidence else left
            loser.tier = RuleTier.TENTATIVE
            loser.updated_at = _utc_now()
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "rules": [r.to_dict() for r in self]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuleBeliefGraph:
        return cls(ConstraintRule.from_dict(item) for item in data.get("rules", []))
