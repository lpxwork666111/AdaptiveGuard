"""ScienceWorld adapter using the upstream Python API."""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

from ..core.types import EnvironmentTransition
from .base import InteractiveEnvironment


class ScienceWorldAdapter(InteractiveEnvironment):
    score_range = 100.0

    def __init__(
        self,
        task_name: str,
        variation: int = 0,
        simplifications: str = "",
        step_limit: int = 100,
        root: str | None = None,
        jar_path: str | None = None,
    ) -> None:
        root_path = Path(root or os.getenv("SCIENCEWORLD_ROOT") or "")
        if str(root_path) and root_path.exists() and str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))
        version_file = root_path / "scienceworld" / "version.py"
        if not version_file.exists() and "scienceworld.version" not in sys.modules:
            version_module = types.ModuleType("scienceworld.version")
            version_module.__version__ = "source"  # type: ignore[attr-defined]
            sys.modules["scienceworld.version"] = version_module
        try:
            java_check = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("ScienceWorld requires an available Java runtime") from exc
        if java_check.returncode != 0:
            raise RuntimeError("ScienceWorld requires an available Java runtime")
        try:
            from scienceworld import ScienceWorldEnv
        except ImportError as exc:
            raise ImportError("ScienceWorld is unavailable; set SCIENCEWORLD_ROOT") from exc
        self._env = ScienceWorldEnv(
            serverPath=jar_path or os.getenv("SCIENCEWORLD_JAR"), envStepLimit=step_limit
        )
        self._task_name = task_name
        self._variation = variation
        self._simplifications = simplifications
        self._env.load(task_name, variation, simplifications)
        self._last_score = 0.0

    def reset(self) -> Any:
        observation, _ = self._env.reset()
        self._last_score = 0.0
        return observation

    def step(self, action: str) -> EnvironmentTransition:
        observation, reward, done, info = self._env.step(action)
        score = float(info.get("score", self._last_score))
        self._last_score = score
        error = None if info.get("valid", True) else str(info.get("msg", "invalid action"))
        return EnvironmentTransition(observation, float(reward), bool(done), info, score, error)

    def available_actions(self) -> list[str]:
        return self._env.get_possible_actions()

    def close(self) -> None:
        if getattr(self, "_env", None) is not None:
            self._env.close()
