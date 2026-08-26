"""Static prompt audit preventing belief-graph leakage into judgment calls."""

from __future__ import annotations

import re


class PromptLeakageError(ValueError):
    pass


def audit_judge_prompt(prompt: str, *, rule_text: list[str] | None = None) -> None:
    lowered = prompt.lower()
    forbidden = ("rule graph", "constraint graph", "belief graph", "alpha", "beta")
    if any(token in lowered for token in forbidden):
        raise PromptLeakageError("judgment prompt contains serialized belief-graph terminology")
    for rule in rule_text or []:
        if rule and re.search(re.escape(rule.lower()), lowered):
            raise PromptLeakageError("judgment prompt contains a serialized rule")
