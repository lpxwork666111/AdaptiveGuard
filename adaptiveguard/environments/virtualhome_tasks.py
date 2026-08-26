"""Load VirtualHome graph tasks from portable JSON manifests."""

from __future__ import annotations

import json
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any


def prepare_virtualhome_import(root: str | None) -> Path:
    root_path = Path(root or os.getenv("VIRTUALHOME_ROOT") or "")
    package_path = (
        root_path / "virtualhome"
        if (root_path / "virtualhome" / "__init__.py").exists()
        else root_path
    )
    import_root = package_path.parent
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
    existing = sys.modules.get("virtualhome")
    if existing is None or not hasattr(existing, "__path__"):
        package = types.ModuleType("virtualhome")
        package.__path__ = [str(package_path)]
        package.__package__ = "virtualhome"
        sys.modules["virtualhome"] = package
    simulation_path = package_path / "simulation"
    simulation = sys.modules.get("virtualhome.simulation")
    if simulation is None or not hasattr(simulation, "__path__"):
        simulation_package = types.ModuleType("virtualhome.simulation")
        simulation_package.__path__ = [str(simulation_path)]
        simulation_package.__package__ = "virtualhome.simulation"
        sys.modules["virtualhome.simulation"] = simulation_package
    return package_path


def _load_json_value(value: Any, base: Path) -> Any:
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return value


def load_virtualhome_manifest(
    path: str | os.PathLike[str], root: str | None = None
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    prepare_virtualhome_import(root)
    from virtualhome.simulation.evolving_graph.environment import EnvironmentGraph

    graph_dict = _load_json_value(manifest["initial_graph"], manifest_path.parent)
    manifest["initial_graph"] = EnvironmentGraph(graph_dict)
    manifest["name_equivalence"] = _load_json_value(
        manifest.get("name_equivalence", {}), manifest_path.parent
    )
    return manifest


def make_goal_predicate(spec: dict[str, Any] | None) -> Callable[[Any], bool]:
    spec = spec or {}

    def predicate(state: Any) -> bool:
        graph = state.to_dict() if hasattr(state, "to_dict") else state
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        for condition in spec.get("node_states", []):
            matching = [
                node
                for node in nodes
                if ("id" not in condition or node.get("id") == condition["id"])
                and (
                    "class_name" not in condition
                    or node.get("class_name") == condition["class_name"]
                )
            ]
            if not matching or not any(
                str(condition["state"]).upper()
                in {str(value).upper() for value in node.get("states", [])}
                for node in matching
            ):
                return False
        for required in spec.get("edges", []):
            if not any(
                all(edge.get(key) == value for key, value in required.items()) for edge in edges
            ):
                return False
        return bool(spec) and True

    return predicate
