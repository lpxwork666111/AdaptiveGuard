"""Evidence and audit records emitted by the controller."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

from .types import EvidenceVector, HcaResult, Judgment, PftDecision, PlannerOutput


@dataclass
class StepAudit:
    step: int
    planner: PlannerOutput
    pft: PftDecision
    transition: Any | None = None
    evidence: EvidenceVector | None = None
    judgment: Judgment | None = None
    hca: HcaResult | None = None
    q_value: float | None = None
    budget_before: float | None = None
    budget_after: float | None = None
    rule_updates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value) and not isinstance(value, type):
                return {item.name: encode(getattr(value, item.name)) for item in fields(value)}
            if isinstance(value, Mapping):
                return {str(k): encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            if hasattr(value, "to_dict"):
                return encode(value.to_dict())
            if hasattr(value, "__dict__"):
                return {k: encode(v) for k, v in value.__dict__.items()}
            return value

        return encode(self)
