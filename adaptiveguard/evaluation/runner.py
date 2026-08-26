"""Multi-episode runner with isolated JSONL records."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..io.serialization import append_jsonl


def run_episodes(
    controller_factory: Callable[[], Any],
    goal: str,
    episodes: int,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(episodes):
        controller = controller_factory()
        try:
            record = controller.run_episode(goal, episode_id=f"episode-{index}")
            records.append(record)
            if output_path is not None:
                append_jsonl(output_path, record)
        finally:
            controller.environment.close()
    return records
