from adaptiveguard.cli.factory import build_controller


def test_complete_mock_control_loop() -> None:
    config = {
        "goal": "inspect and finish",
        "max_steps": 4,
        "environment": {"name": "mock", "required_actions": ["inspect", "finish"]},
        "planner": {"provider": "mock", "actions": ["inspect", "finish"]},
        "cca": {"score_range": 1.0},
    }
    controller = build_controller(config)
    record = controller.run_episode(config["goal"])
    assert record["done"] is True
    assert record["steps"] == 2
    assert len(record["audits"]) == 2
