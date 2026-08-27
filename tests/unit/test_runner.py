from __future__ import annotations

from pathlib import Path
from typing import Any

from adaptiveguard.evaluation.runner import run_episodes
from adaptiveguard.io.serialization import read_jsonl


class _Environment:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("close")


class _Controller:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.environment = _Environment(events)

    def run_episode(self, goal: str, *, episode_id: str) -> dict[str, Any]:
        self.events.append(f"run:{goal}:{episode_id}")
        return {"episode_id": episode_id, "goal": goal}


def test_run_episodes_persists_records_and_closes_each_environment(tmp_path: Path) -> None:
    events: list[str] = []
    output = tmp_path / "episodes.jsonl"

    records = run_episodes(lambda: _Controller(events), "goal", episodes=2, output_path=output)

    assert records == [
        {"episode_id": "episode-0", "goal": "goal"},
        {"episode_id": "episode-1", "goal": "goal"},
    ]
    assert events == [
        "run:goal:episode-0",
        "close",
        "run:goal:episode-1",
        "close",
    ]
    assert read_jsonl(output) == records


def test_run_episodes_closes_environment_when_execution_fails() -> None:
    events: list[str] = []

    class FailingController(_Controller):
        def run_episode(self, goal: str, *, episode_id: str) -> dict[str, Any]:
            self.events.append(f"run:{goal}:{episode_id}")
            raise RuntimeError("episode failed")

    try:
        run_episodes(lambda: FailingController(events), "goal", episodes=1)
    except RuntimeError as exc:
        assert str(exc) == "episode failed"
    else:
        raise AssertionError("run_episodes should propagate controller failures")

    assert events == ["run:goal:episode-0", "close"]
