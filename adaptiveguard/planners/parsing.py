"""Strict parsing for planner JSON responses."""

from __future__ import annotations

import json
from typing import Any

from ..core.types import ConfidenceLabel, PlannerOutput


def parse_planner_response(response: str | dict[str, Any]) -> PlannerOutput:
    data = json.loads(response) if isinstance(response, str) else response
    if not isinstance(data, dict):
        raise ValueError("planner response must be a JSON object")
    action = data.get("action")
    expected = data.get("expected_outcome", data.get("expectedOutcome"))
    confidence = data.get("confidence", data.get("confidence_label"))
    if not all(
        isinstance(value, str) and value.strip() for value in (action, expected, confidence)
    ):
        raise ValueError("planner response requires non-empty action, expected_outcome, confidence")
    action_text = str(action)
    expected_text = str(expected)
    confidence_text = str(confidence)
    try:
        label = ConfidenceLabel(confidence_text.upper())
    except ValueError as exc:
        raise ValueError(f"invalid confidence label: {confidence_text}") from exc
    return PlannerOutput(
        action_text.strip(),
        expected_text.strip(),
        label,
        raw_response=response if isinstance(response, str) else None,
    )
