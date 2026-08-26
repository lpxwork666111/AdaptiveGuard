import pytest

from adaptiveguard.core.rules import ConstraintRule, RuleBeliefGraph
from adaptiveguard.core.types import RuleTier


def test_beta_confidence_and_tiers() -> None:
    rule = ConstraintRule("heater", "requires", "charge", alpha=3, beta=1)
    assert rule.confidence == pytest.approx(0.75)
    assert rule.refresh_tier(0.2, 0.7) == RuleTier.CONFIRMED
    rule.update(refute=20)
    assert rule.refresh_tier(0.2, 0.7) == RuleTier.DEPRECATED


def test_tier_boundaries_are_inclusive_for_tentative() -> None:
    low = ConstraintRule("a", "r", "b", alpha=1, beta=4)
    high = ConstraintRule("c", "r", "d", alpha=7, beta=3)
    assert low.refresh_tier(0.2, 0.7) == RuleTier.TENTATIVE
    assert high.refresh_tier(0.2, 0.7) == RuleTier.TENTATIVE


def test_graph_merges_equivalent_rules() -> None:
    graph = RuleBeliefGraph([ConstraintRule("A", "requires", "B", alpha=2, beta=1)])
    merged = graph.add(ConstraintRule("a", "REQUIRES", "b", alpha=3, beta=1))
    assert len(graph) == 1
    assert merged.alpha == 5
    assert merged.beta == 2


def test_rule_matches_explicit_action_template() -> None:
    rule = ConstraintRule("heater", "requires", "charge", metadata={"action_template": "turn on"})
    assert rule.matches("turn on heater")
    assert not rule.matches("look around")


def test_rule_state_predicate() -> None:
    rule = ConstraintRule(
        "turn on heater",
        "requires",
        "charged",
        metadata={"when": {"observation_not_contains": "charged"}},
    )
    assert rule.matches("turn on heater", "heater is empty")
    assert not rule.matches("turn on heater", "heater is charged")


def test_multiple_preconditions_are_not_treated_as_contradictions() -> None:
    graph = RuleBeliefGraph(
        [
            ConstraintRule("use heater", "requires", "charged"),
            ConstraintRule("use heater", "requires", "plugged in"),
        ]
    )
    assert graph.consistency_check() == []


def test_explicitly_linked_rules_are_contradictory() -> None:
    left = ConstraintRule(
        "heater",
        "mode",
        "warm",
        rule_id="rule-a",
        metadata={"contradicts": ["rule-b"]},
    )
    right = ConstraintRule("heater", "mode", "cool", rule_id="rule-b")
    graph = RuleBeliefGraph([left, right])

    assert graph.consistency_check() == [("rule-a", "rule-b")]


def test_negated_tails_are_contradictory() -> None:
    positive = ConstraintRule("heater", "state", "active", rule_id="rule-a")
    negative = ConstraintRule("heater", "state", "not active", rule_id="rule-b")
    graph = RuleBeliefGraph([positive, negative])

    assert graph.consistency_check() == [("rule-a", "rule-b")]


def test_exclusive_alternatives_are_contradictory() -> None:
    left = ConstraintRule("heater", "mode", "warm", rule_id="rule-a", metadata={"exclusive": True})
    right = ConstraintRule("heater", "mode", "cool", rule_id="rule-b")
    graph = RuleBeliefGraph([left, right])

    assert graph.consistency_check() == [("rule-a", "rule-b")]


def test_contradiction_resolution_uses_deterministic_tie_break() -> None:
    left = ConstraintRule(
        "heater",
        "mode",
        "warm",
        alpha=3,
        beta=1,
        tier=RuleTier.CONFIRMED,
        rule_id="rule-a",
        metadata={"exclusive": True},
    )
    right = ConstraintRule(
        "heater",
        "mode",
        "cool",
        alpha=3,
        beta=1,
        tier=RuleTier.CONFIRMED,
        rule_id="rule-b",
    )
    graph = RuleBeliefGraph([right, left])

    assert graph.resolve_contradictions() == [("rule-a", "rule-b")]
    assert left.tier == RuleTier.CONFIRMED
    assert right.tier == RuleTier.TENTATIVE


def test_behavioral_cycle_deprecates_lowest_confidence_rule() -> None:
    strong = ConstraintRule("a", "requires", "b", alpha=4, beta=1)
    weak = ConstraintRule("b", "requires", "a", alpha=1, beta=1)
    graph = RuleBeliefGraph([strong, weak])
    assert graph.behavioral_cycles()
    graph.enforce_acyclicity()
    assert weak.tier == RuleTier.DEPRECATED
    assert weak.metadata["deprecated_reason"] == "behavioral_cycle"


def test_multi_node_behavioral_cycle_has_stable_rule_order() -> None:
    rules = [
        ConstraintRule("a", "requires", "b", rule_id="rule-c"),
        ConstraintRule("b", "requires", "c", rule_id="rule-a"),
        ConstraintRule("c", "requires", "a", rule_id="rule-b"),
    ]
    graph = RuleBeliefGraph(rules)

    assert graph.behavioral_cycles() == [("rule-a", "rule-b", "rule-c")]


def test_cycle_resolution_uses_deterministic_tie_break() -> None:
    left = ConstraintRule("a", "requires", "b", rule_id="rule-a")
    right = ConstraintRule("b", "requires", "a", rule_id="rule-b")
    graph = RuleBeliefGraph([right, left])

    assert graph.enforce_acyclicity() == [("rule-a", "rule-b")]
    assert left.tier == RuleTier.DEPRECATED
    assert right.tier == RuleTier.TENTATIVE
