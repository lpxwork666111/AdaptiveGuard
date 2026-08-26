"""Construct configured AdaptiveGuard components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..bootstrap.hypotheses import typed_rule_factory
from ..core.controller import AdaptiveGuardController
from ..core.rules import RuleBeliefGraph
from ..io.serialization import read_json
from ..modules.cca import CcaConfig, ContrastiveCausalAttribution
from ..modules.pft import PftConfig, PurposefulFalsificationTrials
from ..modules.scb import ScbConfig, StratifiedConstraintBelief
from ._environment_builders import build_configured_environment
from ._provider_builders import build_provider_components


def build_environment(config: dict[str, Any]) -> Any:
    return build_configured_environment(config)


def build_rule_graph(config: dict[str, Any]) -> RuleBeliefGraph:
    path = config.get("rules", {}).get("path")
    if not path:
        return RuleBeliefGraph()
    target = Path(path)
    return RuleBeliefGraph.from_dict(read_json(target)) if target.exists() else RuleBeliefGraph()


def build_controller(config: dict[str, Any]) -> AdaptiveGuardController:
    environment = build_environment(config)
    providers = build_provider_components(config)
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
        providers.planner,
        scb,
        pft,
        cca,
        providers.process_judge,
        providers.hca_judge,
        typed_rule_factory,
        providers.client.usage_snapshot if providers.client else None,
        int(config.get("max_steps", 100)),
        int(config.get("tier_refresh_interval", 10)),
    )
