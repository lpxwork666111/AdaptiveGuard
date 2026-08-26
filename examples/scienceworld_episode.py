"""Run one configured ScienceWorld episode."""

from adaptiveguard.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--config", "configs/scienceworld.yaml", "--episodes", "1"]))
