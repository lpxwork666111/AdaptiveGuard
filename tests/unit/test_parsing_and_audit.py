import pytest

from adaptiveguard.core.evidence import StepAudit
from adaptiveguard.core.types import (
    ActionDecision,
    ConfidenceLabel,
    PftDecision,
    PlannerOutput,
)
from adaptiveguard.judges.audit import PromptLeakageError, audit_judge_prompt
from adaptiveguard.planners.parsing import parse_planner_response


def test_planner_json_parser() -> None:
    output = parse_planner_response(
        '{"action":"open", "expected_outcome":"opened", "confidence":"high"}'
    )
    assert output.confidence_label == ConfidenceLabel.HIGH


def test_prompt_leakage_audit() -> None:
    audit_judge_prompt("action=open, observation=closed")
    with pytest.raises(PromptLeakageError):
        audit_judge_prompt("serialized belief graph follows")


def test_step_audit_encoding_handles_nested_contracts() -> None:
    audit = StepAudit(
        step=2,
        planner=PlannerOutput("open", "opened", ConfidenceLabel.HIGH),
        pft=PftDecision(ActionDecision.ALLOW, None, scores={"rule": 0.2}),
        transition={"nested": ("value", ActionDecision.TRIAL)},
        rule_updates=[{"rule_id": "rule", "created": True}],
    )

    encoded = audit.to_dict()

    assert encoded["step"] == 2
    assert encoded["planner"]["confidence_label"] == "HIGH"
    assert encoded["pft"]["decision"] == "ALLOW"
    assert encoded["transition"] == {"nested": ["value", "TRIAL"]}
