"""Multi-episode runner with isolated JSONL records."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..io.serialization import append_jsonl


def _episode_id(index: int) -> str:
    return f"episode-{index}"


def _execute_episode(
    controller_factory: Callable[[], Any],
    goal: str,
    index: int,
    output_path: str | Path | None,
) -> dict[str, Any]:
    controller = controller_factory()
    try:
        record = controller.run_episode(goal, episode_id=_episode_id(index))
        if output_path is not None:
            append_jsonl(output_path, record)
        return record
    finally:
        controller.environment.close()


def run_episodes(
    controller_factory: Callable[[], Any],
    goal: str,
    episodes: int,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(episodes):
        records.append(_execute_episode(controller_factory, goal, index, output_path))
    return records
