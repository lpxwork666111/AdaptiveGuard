"""VirtualHome Evolving Graph adapter.

The adapter accepts an already constructed graph and executable Script. This
keeps dataset loading separate from the controller and works without Unity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..core.types import EnvironmentTransition
from .base import InteractiveEnvironment
from .virtualhome_tasks import prepare_virtualhome_import


class VirtualHomeEvolvingGraphAdapter(InteractiveEnvironment):
    score_range = 1.0

    def __init__(
        self,
        graph: Any,
        name_equivalence: dict[str, list[str]] | None = None,
        script_factory: Callable[[str], Any] | None = None,
        goal_predicate: Callable[[Any], bool] | None = None,
        candidate_actions: Sequence[str] = (),
        root: str | None = None,
        char_index: int = 0,
    ) -> None:
        prepare_virtualhome_import(root)
        try:
            from virtualhome.simulation.evolving_graph.execution import ScriptExecutor
        except ImportError as exc:
            raise ImportError("VirtualHome is unavailable; set VIRTUALHOME_ROOT") from exc
        self._executor_cls = ScriptExecutor
        self._graph = graph
        self._name_equivalence = name_equivalence or {}
        self._script_factory = script_factory
        self._goal_predicate = goal_predicate
        self._candidate_actions = list(candidate_actions)
        self._char_index = char_index
        self._state = None
        self._last_observation: Any = graph

    def reset(self) -> Any:
        from virtualhome.simulation.evolving_graph.environment import EnvironmentState

        self._state = EnvironmentState(self._graph, self._name_equivalence, instance_selection=True)
        self._last_observation = self._state
        return self._state

    def step(self, action: str) -> EnvironmentTransition:
        if self._state is None:
            self.reset()
        if self._script_factory is None:
            from virtualhome.simulation.evolving_graph.scripts import Script, parse_script_line

            script = Script([parse_script_line(action, 1)])
        else:
            script = self._script_factory(action)
        executor = self._executor_cls(
            self._graph, self._name_equivalence, char_index=self._char_index
        )
        executable, next_state = executor.execute_one_step(script, self._state)
        if not executable:
            return EnvironmentTransition(
                self._state, -1.0, False, {"valid": False}, error="action not executable"
            )
        self._state = next_state
        self._last_observation = self._state
        done = bool(self._goal_predicate and self._goal_predicate(self._state))
        return EnvironmentTransition(
            self._state, 1.0 if done else 0.0, done, {"valid": True, "success": done}
        )

    def available_actions(self) -> list[str]:
        return self._candidate_actions

    def close(self) -> None:
        return None
