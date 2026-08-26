from adaptiveguard.cli.factory import build_controller
from adaptiveguard.core.rules import ConstraintRule
from adaptiveguard.core.types import RuleTier


def test_complete_mock_control_loop() -> None:
    config = {
        "goal": "inspect and finish",
        "max_steps": 4,
        "environment": {"name": "mock", "required_actions": ["inspect", "finish"]},
        "planner": {"provider": "mock", "actions": ["inspect", "finish"]},
        "cca": {"score_range": 1.0},
    }
    controller = build_controller(config)
    record = controller.run_episode(config["goal"], episode_id="mock-episode")
    assert set(record) == {
        "episode_id",
        "steps",
        "total_reward",
        "final_score",
        "done",
        "failed_trials",
        "model_usage",
        "audits",
        "rules",
    }
    assert record["episode_id"] == "mock-episode"
    assert record["done"] is True
    assert record["steps"] == 2
    assert record["total_reward"] == 2.0
    assert record["final_score"] == 2.0
    assert record["failed_trials"] == 0
    assert record["model_usage"] == {}
    assert len(record["audits"]) == 2
    assert [audit["step"] for audit in record["audits"]] == [0, 1]
    assert [audit["planner"]["action"] for audit in record["audits"]] == [
        "inspect",
        "finish",
    ]
    assert record["rules"]["schema_version"] == 1


def test_blocked_action_is_audited_without_environment_transition() -> None:
    config = {
        "max_steps": 1,
        "environment": {"name": "mock", "required_actions": ["inspect"]},
        "planner": {"provider": "mock", "actions": ["inspect"]},
    }
    controller = build_controller(config)
    blocker = ConstraintRule(
        "inspect",
        "requires",
        "permission",
        alpha=9,
        beta=1,
        tier=RuleTier.CONFIRMED,
    )
    controller.scb.graph.add(blocker)

    record = controller.run_episode("inspect")

    assert record["steps"] == 0
    assert record["total_reward"] == 0.0
    assert record["final_score"] is None
    assert record["done"] is False
    assert controller.environment.index == 0
    assert controller.history == [
        {
            "blocked_action": "inspect",
            "reason": "at least one blocker fails challenge gate",
        }
    ]
    assert len(record["audits"]) == 1
    audit = record["audits"][0]
    assert audit["pft"]["decision"] == "BLOCK"
    assert audit["transition"] is None
    assert audit["budget_before"] == audit["budget_after"]
