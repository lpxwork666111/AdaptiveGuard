"""OpenAI-compatible JSON planner implemented with the Python standard library."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.types import PlannerOutput
from .base import Planner
from .parsing import parse_planner_response
from .prompts import PLANNER_SYSTEM_PROMPT, planner_user_prompt


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 3
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> OpenAICompatibleConfig:
        model = os.getenv("ADAPTIVEGUARD_LLM_MODEL")
        if not model:
            raise ValueError("ADAPTIVEGUARD_LLM_MODEL is required")
        return cls(
            model=model,
            base_url=os.getenv("ADAPTIVEGUARD_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("ADAPTIVEGUARD_LLM_API_KEY"),
        )


class OpenAICompatibleClient:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self._usage: dict[str, int] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self._usage)

    def chat_json(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_key}"}
                    if self.config.api_key
                    else {}
                ),
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                content = response_data["choices"][0]["message"]["content"]
                usage = dict(response_data.get("usage", {}))
                self._usage["calls"] += 1
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    self._usage[key] += int(usage.get(key, 0) or 0)
                return json.loads(content), usage
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError("LLM request failed after retries") from last_error


class LLMPlanner(Planner):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def plan(
        self,
        observation: Any,
        goal: str,
        history: Sequence[Any],
        available_actions: Sequence[str] | None = None,
    ) -> PlannerOutput:
        data, usage = self.client.chat_json(
            PLANNER_SYSTEM_PROMPT,
            planner_user_prompt(
                str(observation),
                goal,
                json.dumps(list(history), default=str),
                list(available_actions) if available_actions else None,
            ),
        )
        output = parse_planner_response(data)
        return PlannerOutput(
            output.action, output.expected_outcome, output.confidence_label, json.dumps(data), usage
        )
