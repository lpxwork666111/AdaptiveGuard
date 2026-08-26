"""Construct configured AdaptiveGuard components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..bootstrap.hypotheses import typed_rule_factory
from ..core.controller import AdaptiveGuardController
from ..core.rules import RuleBeliefGraph
from ..environments.mock import MockEnvironment
from ..environments.scienceworld import ScienceWorldAdapter
from ..environments.virtualhome import VirtualHomeEvolvingGraphAdapter
from ..environments.virtualhome_tasks import load_virtualhome_manifest, make_goal_predicate
from ..environments.virtualhome_unity import VirtualHomeUnityAdapter
from ..io.serialization import read_json
from ..judges.base import HcaJudge, ProcessRewardJudge
from ..judges.llm import LLMHcaJudge, LLMProcessRewardJudge
from ..judges.mock import MockHcaJudge, MockProcessRewardJudge
from ..modules.cca import CcaConfig, ContrastiveCausalAttribution
from ..modules.pft import PftConfig, PurposefulFalsificationTrials
from ..modules.scb import ScbConfig, StratifiedConstraintBelief
from ..planners.base import Planner
from ..planners.llm import LLMPlanner, OpenAICompatibleClient, OpenAICompatibleConfig
from ..planners.mock import MockPlanner


def _llm_config(data: dict[str, Any]) -> OpenAICompatibleConfig:
    keys = {"model", "base_url", "api_key", "temperature", "timeout", "max_retries", "max_tokens"}
    values = {key: value for key, value in data.items() if key in keys}
    if not values.get("model") or str(values["model"]).startswith("${"):
        raise ValueError("an LLM model must be configured")
    return OpenAICompatibleConfig(**values)


def build_environment(config: dict[str, Any]) -> Any:
    data = config.get("environment", {})
    name = data.get("name", "mock")
    if name == "mock":
        return MockEnvironment(list(data.get("required_actions", ["inspect", "finish"])))
    if name == "scienceworld":
        return ScienceWorldAdapter(
            task_name=data["task_name"],
            variation=int(data.get("variation", 0)),
            simplifications=str(data.get("simplifications", "")),
            step_limit=int(data.get("step_limit", config.get("max_steps", 100))),
            root=data.get("root"),
            jar_path=data.get("jar_path"),
        )
    if name == "virtualhome":
        manifest = load_virtualhome_manifest(data["task_manifest"], data.get("root"))
        return VirtualHomeEvolvingGraphAdapter(
            graph=manifest["initial_graph"],
            name_equivalence=manifest.get("name_equivalence", {}),
            goal_predicate=make_goal_predicate(manifest.get("goal_conditions")),
            candidate_actions=manifest.get("candidate_actions", []),
            root=data.get("root"),
            char_index=int(manifest.get("char_index", 0)),
        )
    if name == "virtualhome_unity":
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
    raise ValueError(f"unknown environment: {name}")


def build_rule_graph(config: dict[str, Any]) -> RuleBeliefGraph:
    path = config.get("rules", {}).get("path")
    if not path:
        return RuleBeliefGraph()
    target = Path(path)
    return RuleBeliefGraph.from_dict(read_json(target)) if target.exists() else RuleBeliefGraph()


def build_controller(config: dict[str, Any]) -> AdaptiveGuardController:
    environment = build_environment(config)
    planner_config = config.get("planner", {})
    planner_provider = planner_config.get("provider", "mock")
    judge_config = config.get("judges", {})
    judge_provider = judge_config.get("provider", planner_provider)
    planner: Planner
    process_judge: ProcessRewardJudge
    hca_judge: HcaJudge
    client: OpenAICompatibleClient | None = None
    if planner_provider == "mock":
        planner = MockPlanner(planner_config.get("actions"))
    elif planner_provider == "openai_compatible":
        client = OpenAICompatibleClient(_llm_config(planner_config))
        planner = LLMPlanner(client)
    else:
        raise ValueError(f"unknown planner provider: {planner_provider}")
    if judge_provider == "mock":
        process_judge = MockProcessRewardJudge()
        hca_judge = MockHcaJudge()
    elif judge_provider == "openai_compatible":
        client = client or OpenAICompatibleClient(_llm_config({**planner_config, **judge_config}))
        process_judge = LLMProcessRewardJudge(client, int(judge_config.get("prj_votes", 3)))
        hca_judge = LLMHcaJudge(
            client,
            int(judge_config.get("hca_hint_votes", 3)),
            int(judge_config.get("hca_structured_votes", 3)),
        )
    else:
        raise ValueError(f"unknown judge provider: {judge_provider}")
    scb = StratifiedConstraintBelief(build_rule_graph(config), ScbConfig(**config.get("scb", {})))
    pft = PurposefulFalsificationTrials(PftConfig(**config.get("pft", {})))
    cca_data = dict(config.get("cca", {}))
    for key in (
        "default_weights",
        "environment_weights",
        "silent_reward_weights",
        "invalid_hca_weights",
    ):
        if key in cca_data:
            cca_data[key] = tuple(cca_data[key])
    cca = ContrastiveCausalAttribution(CcaConfig(**cca_data), pft.trajectory_buffer)
    return AdaptiveGuardController(
        environment,
        planner,
        scb,
        pft,
        cca,
        process_judge,
        hca_judge,
        typed_rule_factory,
        client.usage_snapshot if client else None,
        int(config.get("max_steps", 100)),
        int(config.get("tier_refresh_interval", 10)),
    )
