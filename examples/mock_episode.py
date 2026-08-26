"""Run the complete controller without an external simulator or API."""

from adaptiveguard.cli.factory import build_controller


def main() -> None:
    config = {
        "goal": "inspect and finish",
        "max_steps": 5,
        "environment": {"name": "mock", "required_actions": ["inspect", "finish"]},
        "planner": {"provider": "mock", "actions": ["inspect", "finish"]},
        "scb": {},
        "pft": {},
        "cca": {"score_range": 1.0},
    }
    controller = build_controller(config)
    try:
        record = controller.run_episode(config["goal"])
        print({key: record[key] for key in ("steps", "total_reward", "done")})
    finally:
        controller.environment.close()


if __name__ == "__main__":
    main()
