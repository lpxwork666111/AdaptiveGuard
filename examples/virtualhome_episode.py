"""Run one configured VirtualHome Evolving Graph episode."""

from adaptiveguard.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--config", "configs/virtualhome.yaml", "--episodes", "1"]))
