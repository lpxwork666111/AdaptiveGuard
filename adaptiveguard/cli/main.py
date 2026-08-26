"""AdaptiveGuard command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..bootstrap.demonstrations import rules_from_demonstrations
from ..evaluation.metrics import aggregate_run, rule_graph_statistics
from ..io.config import load_config
from ..io.logging import configure_logging
from ..io.manifests import make_manifest, seed_everything
from ..io.serialization import append_jsonl, read_json, read_jsonl, write_json
from .factory import build_controller


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.episodes is not None:
        config["episodes"] = args.episodes
    if args.seed is not None:
        config["seed"] = args.seed
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    controller = build_controller(config)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output or config.get("output", {}).get("directory", "outputs")) / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.json", config)
    write_json(output_root / "manifest.json", make_manifest(config, seed))
    episodes: list[dict[str, Any]] = []
    try:
        for index in range(int(config.get("episodes", 1))):
            record = controller.run_episode(
                str(config.get("goal", "complete the task")), episode_id=f"episode-{index}"
            )
            append_jsonl(output_root / "episodes.jsonl", record)
            episodes.append(record)
        write_json(output_root / "summary.json", aggregate_run(episodes))
        write_json(output_root / "rules.json", controller.scb.graph.to_dict())
    finally:
        controller.environment.close()
    print(output_root.resolve())
    return 0


def command_bootstrap(args: argparse.Namespace) -> int:
    records = read_jsonl(args.input)
    graph = rules_from_demonstrations(records, args.environment)
    write_json(args.output, graph.to_dict())
    print(Path(args.output).resolve())
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    from ..core.rules import RuleBeliefGraph

    graph = RuleBeliefGraph.from_dict(read_json(args.rules))
    print(json.dumps(rule_graph_statistics(graph), indent=2))
    return 0


def command_check_env(args: argparse.Namespace) -> int:
    checks: dict[str, Any] = {"python": sys.version.split()[0], "cwd": os.getcwd()}
    science_root = Path(os.getenv("SCIENCEWORLD_ROOT", "../ScienceWorld"))
    science_jar = Path(
        os.getenv("SCIENCEWORLD_JAR", str(science_root / "scienceworld/scienceworld.jar"))
    )
    virtualhome_root = Path(os.getenv("VIRTUALHOME_ROOT", "../virtualhome"))
    checks["scienceworld_root"] = science_root.resolve().is_dir()
    checks["scienceworld_jar"] = science_jar.resolve().is_file()
    checks["virtualhome_root"] = virtualhome_root.resolve().is_dir()
    try:
        java = subprocess.run(["java", "-version"], capture_output=True, check=False, timeout=10)
        checks["java_runtime"] = java.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks["java_runtime"] = False
    print(json.dumps(checks, indent=2))
    return (
        0
        if all(
            checks[key]
            for key in (
                "scienceworld_root",
                "scienceworld_jar",
                "virtualhome_root",
                "java_runtime",
            )
        )
        else 1
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptiveguard", description="AdaptiveGuard interactive-agent controller"
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one or more adaptive guarded episodes")
    run.add_argument("--config", required=True)
    run.add_argument("--episodes", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--output")
    run.add_argument("--run-id")
    run.set_defaults(handler=command_run)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="create an initial rule graph from JSONL demonstrations"
    )
    bootstrap.add_argument("--input", required=True)
    bootstrap.add_argument("--output", required=True)
    bootstrap.add_argument("--environment")
    bootstrap.set_defaults(handler=command_bootstrap)

    inspect = subparsers.add_parser("inspect-rules", help="inspect a serialized rule graph")
    inspect.add_argument("--rules", required=True)
    inspect.set_defaults(handler=command_inspect)

    check_env = subparsers.add_parser(
        "check-env", help="check local environment paths and runtime prerequisites"
    )
    check_env.set_defaults(handler=command_check_env)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
