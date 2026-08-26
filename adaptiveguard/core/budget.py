"""Bounded asymmetric exploration budget for falsification trials."""

from dataclasses import dataclass


@dataclass
class TrialBudget:
    base: float = 5.0
    maximum: float | None = None
    refund: float = 0.5
    remaining: float | None = None

    def __post_init__(self) -> None:
        if self.base < 0 or self.refund < 0:
            raise ValueError("budget values must be non-negative")
        if self.maximum is None:
            self.maximum = 2 * self.base
        if self.maximum < self.base:
            raise ValueError("maximum budget must be >= base budget")
        if self.remaining is None:
            self.remaining = self.base
        self.remaining = min(max(0.0, self.remaining), self.maximum)

    def reset(self) -> None:
        self.remaining = self.base

    def on_success(self) -> None:
        assert self.maximum is not None and self.remaining is not None
        self.remaining = min(self.maximum, self.remaining + self.refund)

    def on_failure(self) -> None:
        assert self.remaining is not None
        self.remaining = max(0.0, self.remaining - 1.0)

    def can_trial(self) -> bool:
        assert self.remaining is not None
        return self.remaining > 0

    def to_dict(self) -> dict[str, float]:
        assert self.maximum is not None and self.remaining is not None
        return {
            "base": self.base,
            "maximum": self.maximum,
            "refund": self.refund,
            "remaining": self.remaining,
        }
