"""Typed runtime hypothesis generation."""

from __future__ import annotations

from typing import Any

from ..core.rules import ConstraintRule
from ..core.types import HcaLabel
from ..modules.cca import HCA_EFFECTS


def typed_rule_factory(
    action: str, hca_label: HcaLabel | None, evidence: Any
) -> ConstraintRule | None:
    _, graph_type = HCA_EFFECTS[hca_label] if hca_label is not None else (0.0, None)
    if graph_type is None and isinstance(evidence, dict):
        prj = evidence.get("prj")
        if float(getattr(prj, "signed_value", 0.0)) < 0:
            graph_type = "G_env.E_1"
    if graph_type is None:
        return None
    hint = ""
    if isinstance(evidence, dict):
        hca = evidence.get("hca")
        hint = str(getattr(hca, "hint", "") or "")
    action_template = action.strip().split()[0].lower() if action.strip() else "unknown"
    feedback = evidence.get("environment_feedback") if isinstance(evidence, dict) else None
    tail = hint or str(feedback or f"unsatisfied condition for {action}")
    return ConstraintRule(
        head=action_template,
        relation=f"requires:{graph_type}",
        tail=tail,
        source="runtime",
        metadata={
            "action": action,
            "action_template": action_template,
            "hca_label": hca_label.value if hca_label else None,
            "graph_type": graph_type,
        },
    )
