import subprocess
from pathlib import Path

import pytest

from adaptiveguard.environments.scienceworld import ScienceWorldAdapter


def test_scienceworld_smoke() -> None:
    project = Path(__file__).resolve().parents[2]
    root = project.parent / "ScienceWorld"
    jar = root / "scienceworld/scienceworld.jar"
    if not jar.exists():
        pytest.skip("ScienceWorld JAR is unavailable")
    java = subprocess.run(["java", "-version"], capture_output=True, check=False)
    if java.returncode != 0:
        pytest.skip("Java runtime is unavailable")
    env = ScienceWorldAdapter("boil", variation=0, step_limit=5, root=str(root), jar_path=str(jar))
    try:
        observation = env.reset()
        transition = env.step("look around")
        assert isinstance(observation, str)
        assert isinstance(transition.observation, str)
    finally:
        env.close()
