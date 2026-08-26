"""Optional VirtualHome Unity communication adapter."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from ..core.types import EnvironmentTransition
from .base import InteractiveEnvironment
from .virtualhome_tasks import prepare_virtualhome_import


class VirtualHomeUnityAdapter(InteractiveEnvironment):
    score_range = 1.0

    def __init__(
        self,
        environment_id: int | None = None,
        executable: str | None = None,
        root: str | None = None,
        port: str = "8080",
        no_graphics: bool = False,
        add_character: bool = True,
        candidate_actions: Sequence[str] = (),
        goal_predicate: Callable[[dict[str, Any]], bool] | None = None,
        communication: Any | None = None,
    ) -> None:
        self.environment_id = environment_id
        self.add_character_on_reset = add_character
        self._candidate_actions = list(candidate_actions)
        self._goal_predicate = goal_predicate
        if communication is None:
            prepare_virtualhome_import(root)
            try:
                from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
            except ImportError as exc:
                raise ImportError(
                    "VirtualHome Unity dependencies are unavailable; install adaptiveguard[unity]"
                ) from exc
            communication = UnityCommunication(
                port=port,
                file_name=executable or os.getenv("VIRTUALHOME_SIMULATOR") or None,
                no_graphics=no_graphics,
            )
        self._communication = communication
        self._graph: dict[str, Any] | None = None

    def reset(self) -> dict[str, Any]:
        if not self._communication.reset(self.environment_id):
            raise RuntimeError("VirtualHome Unity scene reset failed")
        if self.add_character_on_reset and not self._communication.add_character():
            raise RuntimeError("VirtualHome Unity character creation failed")
        valid, graph = self._communication.environment_graph()
        if not valid:
            raise RuntimeError("VirtualHome Unity graph request failed")
        self._graph = graph
        return graph

    def step(self, action: str) -> EnvironmentTransition:
        script_line = action if action.lstrip().startswith("<char") else f"<char0> {action}"
        valid, message = self._communication.render_script(
            [script_line],
            recording=False,
            skip_animation=True,
        )
        graph_valid, graph = self._communication.environment_graph()
        if graph_valid:
            self._graph = graph
        observation = self._graph or {}
        done = bool(valid and self._goal_predicate and self._goal_predicate(observation))
        return EnvironmentTransition(
            observation,
            1.0 if done else (-1.0 if not valid else 0.0),
            done,
            {"valid": bool(valid), "message": message, "success": done},
            error=None if valid else str(message),
        )

    def available_actions(self) -> list[str]:
        return self._candidate_actions

    def close(self) -> None:
        if self._communication is not None:
            self._communication.close()
            self._communication = None
