"""Planner and judge builders selected by provider name."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..judges.base import HcaJudge, ProcessRewardJudge
from ..judges.llm import LLMHcaJudge, LLMProcessRewardJudge
from ..judges.mock import MockHcaJudge, MockProcessRewardJudge
from ..planners.base import Planner
from ..planners.llm import LLMPlanner, OpenAICompatibleClient, OpenAICompatibleConfig
from ..planners.mock import MockPlanner

ConfigMapping = Mapping[str, Any]
PlannerBuilder = Callable[[ConfigMapping], tuple[Planner, OpenAICompatibleClient | None]]
JudgeBuilder = Callable[
    [ConfigMapping, ConfigMapping, OpenAICompatibleClient | None],
    tuple[ProcessRewardJudge, HcaJudge, OpenAICompatibleClient | None],
]


@dataclass(frozen=True)
class ProviderComponents:
    planner: Planner
    process_judge: ProcessRewardJudge
    hca_judge: HcaJudge
    client: OpenAICompatibleClient | None


def _llm_config(data: ConfigMapping) -> OpenAICompatibleConfig:
    keys = {"model", "base_url", "api_key", "temperature", "timeout", "max_retries", "max_tokens"}
    values = {key: value for key, value in data.items() if key in keys}
    if not values.get("model") or str(values["model"]).startswith("${"):
        raise ValueError("an LLM model must be configured")
    return OpenAICompatibleConfig(**values)


def _build_mock_planner(
    config: ConfigMapping,
) -> tuple[Planner, OpenAICompatibleClient | None]:
    return MockPlanner(config.get("actions")), None


def _build_openai_planner(
    config: ConfigMapping,
) -> tuple[Planner, OpenAICompatibleClient | None]:
    client = OpenAICompatibleClient(_llm_config(config))
    return LLMPlanner(client), client


def _build_mock_judges(
    planner_config: ConfigMapping,
    judge_config: ConfigMapping,
    client: OpenAICompatibleClient | None,
) -> tuple[ProcessRewardJudge, HcaJudge, OpenAICompatibleClient | None]:
    del planner_config, judge_config
    return MockProcessRewardJudge(), MockHcaJudge(), client


def _build_openai_judges(
    planner_config: ConfigMapping,
    judge_config: ConfigMapping,
    client: OpenAICompatibleClient | None,
) -> tuple[ProcessRewardJudge, HcaJudge, OpenAICompatibleClient]:
    if client is None:
        client = OpenAICompatibleClient(_llm_config({**planner_config, **judge_config}))
    process_judge = LLMProcessRewardJudge(client, int(judge_config.get("prj_votes", 3)))
    hca_judge = LLMHcaJudge(
        client,
        int(judge_config.get("hca_hint_votes", 3)),
        int(judge_config.get("hca_structured_votes", 3)),
    )
    return process_judge, hca_judge, client


_PLANNER_BUILDERS: dict[str, PlannerBuilder] = {
    "mock": _build_mock_planner,
    "openai_compatible": _build_openai_planner,
}

_JUDGE_BUILDERS: dict[str, JudgeBuilder] = {
    "mock": _build_mock_judges,
    "openai_compatible": _build_openai_judges,
}


def build_provider_components(config: ConfigMapping) -> ProviderComponents:
    planner_config = config.get("planner", {})
    planner_provider = planner_config.get("provider", "mock")
    planner_builder = _PLANNER_BUILDERS.get(planner_provider)
    if planner_builder is None:
        raise ValueError(f"unknown planner provider: {planner_provider}")
    planner, client = planner_builder(planner_config)

    judge_config = config.get("judges", {})
    judge_provider = judge_config.get("provider", planner_provider)
    judge_builder = _JUDGE_BUILDERS.get(judge_provider)
    if judge_builder is None:
        raise ValueError(f"unknown judge provider: {judge_provider}")
    process_judge, hca_judge, client = judge_builder(planner_config, judge_config, client)
    return ProviderComponents(planner, process_judge, hca_judge, client)
