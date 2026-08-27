"""Evidence and audit records emitted by the controller."""

from dataclasses import dataclass, field
from typing import Any

from ._encoding import encode_value
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
        return encode_value(self)
