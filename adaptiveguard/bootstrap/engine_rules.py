"""Extract typed action-interface rules from environment schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..core.rules import ConstraintRule, RuleBeliefGraph


def virtualhome_action_rules(action_members: Iterable[Any]) -> RuleBeliefGraph:
    rules: list[ConstraintRule] = []
    for action in action_members:
        human_name, arity, requirements = action.value
        for index, alternatives in enumerate(requirements):
            for required_property in alternatives:
                rules.append(
                    ConstraintRule(
                        head=action.name,
                        relation=f"argument_{index}_requires",
                        tail=required_property,
                        alpha=2.0,
                        beta=1.0,
                        source="engine",
                        environment="virtualhome",
                        metadata={
                            "action_template": action.name.lower(),
                            "arity": arity,
                            "human_name": human_name,
                        },
                    )
                )
    return RuleBeliefGraph(rules)


def scienceworld_action_rules(actions: Iterable[Mapping[str, Any]]) -> RuleBeliefGraph:
    rules: list[ConstraintRule] = []
    for item in actions:
        action = str(item.get("action", item.get("action_example", "")))
        template = str(item.get("template_id", "unknown"))
        if not action:
            continue
        rules.append(
            ConstraintRule(
                head=template,
                relation="admits",
                tail=action,
                alpha=2.0,
                beta=1.0,
                source="engine",
                environment="scienceworld",
                metadata={"action": action, "action_template": action.split()[0]},
            )
        )
    return RuleBeliefGraph(rules)
