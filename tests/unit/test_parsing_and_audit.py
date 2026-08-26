import pytest

from adaptiveguard.core.types import ConfidenceLabel
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
