from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from adaptiveguard.cli import _environment_builders
from adaptiveguard.cli.factory import build_controller, build_environment, build_rule_graph
from adaptiveguard.core.rules import ConstraintRule, RuleBeliefGraph
from adaptiveguard.environments.mock import MockEnvironment
from adaptiveguard.io.serialization import write_json
from adaptiveguard.judges.llm import LLMHcaJudge, LLMProcessRewardJudge
from adaptiveguard.judges.mock import MockHcaJudge, MockProcessRewardJudge
from adaptiveguard.planners.llm import LLMPlanner
from adaptiveguard.planners.mock import MockPlanner


def test_mock_environment_builder() -> None:
    environment = build_environment(
        {"environment": {"name": "mock", "required_actions": ["inspect"]}}
    )

    assert isinstance(environment, MockEnvironment)
    assert environment.required_actions == ["inspect"]


def test_scienceworld_environment_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    expected = MockEnvironment()

    def make_adapter(**kwargs: Any) -> MockEnvironment:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(_environment_builders, "ScienceWorldAdapter", make_adapter)
    environment = build_environment(
        {
            "max_steps": 12,
            "environment": {
                "name": "scienceworld",
                "task_name": "task",
                "variation": "2",
                "simplifications": "easy",
                "root": "environment-root",
                "jar_path": "environment.jar",
            },
        }
    )

    assert environment is expected
    assert captured == {
        "task_name": "task",
        "variation": 2,
        "simplifications": "easy",
        "step_limit": 12,
        "root": "environment-root",
        "jar_path": "environment.jar",
    }


def test_virtualhome_environment_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = object()
    predicate = object()
    expected = MockEnvironment()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        _environment_builders,
        "load_virtualhome_manifest",
        lambda path, root: {
            "initial_graph": graph,
            "name_equivalence": {"cup": ["mug"]},
            "goal_conditions": {"node_states": []},
            "candidate_actions": ["walk"],
            "char_index": "1",
        },
    )
    monkeypatch.setattr(_environment_builders, "make_goal_predicate", lambda spec: predicate)

    def make_adapter(**kwargs: Any) -> MockEnvironment:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(_environment_builders, "VirtualHomeEvolvingGraphAdapter", make_adapter)
    environment = build_environment(
        {
            "environment": {
                "name": "virtualhome",
                "task_manifest": "task.json",
                "root": "environment-root",
            }
        }
    )

    assert environment is expected
    assert captured == {
        "graph": graph,
        "name_equivalence": {"cup": ["mug"]},
        "goal_predicate": predicate,
        "candidate_actions": ["walk"],
        "root": "environment-root",
        "char_index": 1,
    }


def test_virtualhome_unity_environment_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    predicate = object()
    expected = MockEnvironment()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        _environment_builders,
        "read_json",
        lambda path: {"candidate_actions": ["walk"], "goal_conditions": {"edges": []}},
    )
    monkeypatch.setattr(_environment_builders, "make_goal_predicate", lambda spec: predicate)

    def make_adapter(**kwargs: Any) -> MockEnvironment:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(_environment_builders, "VirtualHomeUnityAdapter", make_adapter)
    environment = build_environment(
        {
            "environment": {
                "name": "virtualhome_unity",
                "task_manifest": "task.json",
                "environment_id": 4,
                "executable": "simulator",
                "root": "environment-root",
                "port": 9000,
                "no_graphics": True,
            }
        }
    )

    assert environment is expected
    assert captured == {
        "environment_id": 4,
        "executable": "simulator",
        "root": "environment-root",
        "port": "9000",
        "no_graphics": True,
        "candidate_actions": ["walk"],
        "goal_predicate": predicate,
    }


def test_openai_planner_can_use_mock_judges() -> None:
    controller = build_controller(
        {
            "environment": {"name": "mock"},
            "planner": {"provider": "openai_compatible", "model": "model"},
            "judges": {"provider": "mock"},
        }
    )

    assert isinstance(controller.planner, LLMPlanner)
    assert isinstance(controller.process_judge, MockProcessRewardJudge)
    assert isinstance(controller.hca_judge, MockHcaJudge)
    assert getattr(controller.usage_provider, "__self__", None) is controller.planner.client


def test_mock_planner_can_use_shared_openai_judges() -> None:
    controller = build_controller(
        {
            "environment": {"name": "mock"},
            "planner": {"provider": "mock"},
            "judges": {"provider": "openai_compatible", "model": "model"},
        }
    )

    assert isinstance(controller.planner, MockPlanner)
    assert isinstance(controller.process_judge, LLMProcessRewardJudge)
    assert isinstance(controller.hca_judge, LLMHcaJudge)
    assert controller.process_judge.client is controller.hca_judge.client
    assert getattr(controller.usage_provider, "__self__", None) is controller.process_judge.client


def test_openai_components_share_one_client() -> None:
    controller = build_controller(
        {
            "environment": {"name": "mock"},
            "planner": {"provider": "openai_compatible", "model": "model"},
            "judges": {"provider": "openai_compatible"},
        }
    )

    assert isinstance(controller.planner, LLMPlanner)
    assert isinstance(controller.process_judge, LLMProcessRewardJudge)
    assert isinstance(controller.hca_judge, LLMHcaJudge)
    assert controller.planner.client is controller.process_judge.client
    assert controller.planner.client is controller.hca_judge.client
    assert getattr(controller.usage_provider, "__self__", None) is controller.planner.client


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"environment": {"name": "unknown"}}, "unknown environment: unknown"),
        (
            {"environment": {"name": "mock"}, "planner": {"provider": "unknown"}},
            "unknown planner provider: unknown",
        ),
        (
            {
                "environment": {"name": "mock"},
                "planner": {"provider": "mock"},
                "judges": {"provider": "unknown"},
            },
            "unknown judge provider: unknown",
        ),
    ],
)
def test_unknown_component_errors(config: dict[str, Any], message: str) -> None:
    builder = build_environment if "unknown environment" in message else build_controller
    with pytest.raises(ValueError, match=f"^{message}$"):
        builder(config)


def test_rule_graph_loading_preserves_serialized_rules(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    source = RuleBeliefGraph([ConstraintRule("heater", "requires", "charge")])
    write_json(path, source.to_dict())

    loaded = build_rule_graph({"rules": {"path": str(path)}})

    assert loaded.to_dict() == source.to_dict()
    assert len(build_rule_graph({"rules": {"path": str(tmp_path / "missing.json")}})) == 0
