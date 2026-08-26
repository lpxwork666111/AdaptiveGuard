"""Typed contracts shared by planners, environments, judges, and modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuleTier(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"


class ConfidenceLabel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def value_numeric(self) -> float:
        return {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}[self.value]


class ActionDecision(str, Enum):
    ALLOW = "ALLOW"
    TRIAL = "TRIAL"
    BLOCK = "BLOCK"


class Judgment(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    UNCERTAIN = "UNCERTAIN"


class HcaLabel(str, Enum):
    IDENTICAL = "IDENTICAL"
    PARAM_DIFF = "PARAM_DIFF"
    MISSING_PRECOND = "MISSING_PRECOND"
    WRONG_OBJECT = "WRONG_OBJECT"
    WRONG_TIMING = "WRONG_TIMING"
    BOTH_WRONG = "BOTH_WRONG"


@dataclass(frozen=True)
class PlannerOutput:
    """Planner output produced in one forward pass."""

    action: str
    expected_outcome: str
    confidence_label: ConfidenceLabel
    raw_response: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def self_confidence(self) -> float:
        return self.confidence_label.value_numeric


@dataclass(frozen=True)
class EnvironmentTransition:
    observation: Any
    reward: float
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    score: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvidenceVector:
    """Three CCA signals, each clipped to [-1, 1]."""

    prj: float = 0.0
    sps: float = 0.0
    hca: float = 0.0

    def __post_init__(self) -> None:
        for name in ("prj", "sps", "hca"):
            value = float(getattr(self, name))
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1], got {value}")

    def as_list(self) -> list[float]:
        return [self.prj, self.sps, self.hca]


@dataclass(frozen=True)
class JudgeResult:
    label: str
    signed_value: float
    raw_votes: Sequence[float] = ()
    rationale: str | None = None


@dataclass(frozen=True)
class HcaResult:
    label: HcaLabel | None
    signed_value: float
    hint: str | None = None
    valid: bool = True
    votes: Sequence[str] = ()


@dataclass(frozen=True)
class PftDecision:
    decision: ActionDecision
    selected_rule_id: str | None
    blocker_ids: tuple[str, ...] = ()
    scores: Mapping[str, float] = field(default_factory=dict)
    criticality: float = 0.5
    reason: str = ""


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    steps: int
    total_reward: float
    final_score: float | None
    done: bool
    failed_trials: int
    llm_calls: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
