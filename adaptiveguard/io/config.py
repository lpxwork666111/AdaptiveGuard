"""YAML configuration loading with environment-variable expansion."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._config_loader import expand_environment, load_yaml_mapping


class ScbSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theta_low: float = Field(0.20, ge=0, le=1)
    theta_high: float = Field(0.70, ge=0, le=1)
    oscillation_window: int = Field(10, ge=1)

    @model_validator(mode="after")
    def ordered_thresholds(self) -> ScbSettings:
        if self.theta_low >= self.theta_high:
            raise ValueError("theta_low must be smaller than theta_high")
        return self


class PftSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tau: float = Field(0.25, ge=0, le=1)
    epsilon0: float = Field(0.10, ge=0, le=1)
    budget_base: float = Field(5.0, ge=0)
    budget_max: float | None = Field(None, ge=0)
    budget_refund: float = Field(0.5, ge=0)


class CcaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theta_minus: float = Field(-0.20, ge=-1, le=1)
    theta_plus: float = Field(0.20, ge=-1, le=1)
    eta: float = Field(0.5, gt=0)
    score_range: float = Field(100.0, gt=0)
    sps_window: int = Field(8, ge=1)
    gamma: float = Field(0.9, ge=0, le=1)
    default_weights: tuple[float, float, float] = (0.35, 0.25, 0.40)
    environment_weights: tuple[float, float, float] = (0.55, 0.20, 0.25)
    silent_reward_weights: tuple[float, float, float] = (0.35, 0.10, 0.55)
    invalid_hca_weights: tuple[float, float, float] = (0.583, 0.417, 0.0)

    @model_validator(mode="after")
    def valid_thresholds_and_weights(self) -> CcaSettings:
        if self.theta_minus >= self.theta_plus:
            raise ValueError("theta_minus must be smaller than theta_plus")
        for weights in (
            self.default_weights,
            self.environment_weights,
            self.silent_reward_weights,
            self.invalid_hca_weights,
        ):
            if any(weight < 0 for weight in weights) or sum(weights) <= 0:
                raise ValueError("CCA weights must be non-negative with a positive sum")
        return self


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    seed: int = 0
    goal: str = "complete the current task"
    episodes: int = Field(1, ge=1)
    max_steps: int = Field(100, ge=1)
    tier_refresh_interval: int = Field(10, ge=1)
    environment: dict[str, Any] = Field(default_factory=lambda: {"name": "mock"})
    planner: dict[str, Any] = Field(default_factory=lambda: {"provider": "mock"})
    judges: dict[str, Any] = Field(default_factory=dict)
    scb: ScbSettings = Field(default_factory=lambda: ScbSettings())
    pft: PftSettings = Field(default_factory=lambda: PftSettings())
    cca: CcaSettings = Field(default_factory=lambda: CcaSettings())
    rules: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=lambda: {"directory": "outputs"})


def _expand(value: Any) -> Any:
    return expand_environment(value)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    data = load_yaml_mapping(path)
    return RunSettings.model_validate(_expand(data)).model_dump(mode="python")
