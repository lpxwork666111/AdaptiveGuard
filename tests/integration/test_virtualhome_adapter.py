from pathlib import Path

import pytest

from adaptiveguard.environments.virtualhome import VirtualHomeEvolvingGraphAdapter
from adaptiveguard.environments.virtualhome_tasks import (
    load_virtualhome_manifest,
    make_goal_predicate,
)


def test_virtualhome_manifest_step() -> None:
    project = Path(__file__).resolve().parents[2]
    root = project.parent / "virtualhome"
    if not root.exists():
        pytest.skip("VirtualHome checkout is unavailable")
    manifest = load_virtualhome_manifest(project / "examples/data/virtualhome_task.json", str(root))
    env = VirtualHomeEvolvingGraphAdapter(
        manifest["initial_graph"],
        manifest["name_equivalence"],
        goal_predicate=make_goal_predicate(manifest["goal_conditions"]),
        candidate_actions=manifest["candidate_actions"],
        root=str(root),
    )
    env.reset()
    transition = env.step("[SWITCHON] <lamp> (3)")
    assert transition.info["valid"] is True
    assert transition.done is True
