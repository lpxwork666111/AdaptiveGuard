"""Environment builders selected by configuration name."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..environments.base import InteractiveEnvironment
from ..environments.mock import MockEnvironment
from ..environments.scienceworld import ScienceWorldAdapter
from ..environments.virtualhome import VirtualHomeEvolvingGraphAdapter
from ..environments.virtualhome_tasks import load_virtualhome_manifest, make_goal_predicate
from ..environments.virtualhome_unity import VirtualHomeUnityAdapter
from ..io.serialization import read_json

ConfigMapping = Mapping[str, Any]
EnvironmentBuilder = Callable[[ConfigMapping, ConfigMapping], InteractiveEnvironment]


def _build_mock(data: ConfigMapping, config: ConfigMapping) -> InteractiveEnvironment:
    del config
    return MockEnvironment(list(data.get("required_actions", ["inspect", "finish"])))


def _build_scienceworld(data: ConfigMapping, config: ConfigMapping) -> InteractiveEnvironment:
    return ScienceWorldAdapter(
        task_name=data["task_name"],
        variation=int(data.get("variation", 0)),
        simplifications=str(data.get("simplifications", "")),
        step_limit=int(data.get("step_limit", config.get("max_steps", 100))),
        root=data.get("root"),
        jar_path=data.get("jar_path"),
    )


def _build_virtualhome(data: ConfigMapping, config: ConfigMapping) -> InteractiveEnvironment:
    del config
    manifest = load_virtualhome_manifest(data["task_manifest"], data.get("root"))
    return VirtualHomeEvolvingGraphAdapter(
        graph=manifest["initial_graph"],
        name_equivalence=manifest.get("name_equivalence", {}),
        goal_predicate=make_goal_predicate(manifest.get("goal_conditions")),
        candidate_actions=manifest.get("candidate_actions", []),
        root=data.get("root"),
        char_index=int(manifest.get("char_index", 0)),
    )


def _build_virtualhome_unity(data: ConfigMapping, config: ConfigMapping) -> InteractiveEnvironment:
    del config
    manifest = read_json(data["task_manifest"])
    return VirtualHomeUnityAdapter(
        environment_id=data.get("environment_id"),
        executable=data.get("executable"),
        root=data.get("root"),
        port=str(data.get("port", "8080")),
        no_graphics=bool(data.get("no_graphics", False)),
        candidate_actions=manifest.get("candidate_actions", []),
        goal_predicate=make_goal_predicate(manifest.get("goal_conditions")),
    )


_ENVIRONMENT_BUILDERS: dict[str, EnvironmentBuilder] = {
    "mock": _build_mock,
    "scienceworld": _build_scienceworld,
    "virtualhome": _build_virtualhome,
    "virtualhome_unity": _build_virtualhome_unity,
}


def build_configured_environment(config: ConfigMapping) -> InteractiveEnvironment:
    data = config.get("environment", {})
    name = data.get("name", "mock")
    builder = _ENVIRONMENT_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"unknown environment: {name}")
    return builder(data, config)
