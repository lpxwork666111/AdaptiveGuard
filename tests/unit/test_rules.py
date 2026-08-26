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


def test_behavioral_cycle_deprecates_lowest_confidence_rule() -> None:
    strong = ConstraintRule("a", "requires", "b", alpha=4, beta=1)
    weak = ConstraintRule("b", "requires", "a", alpha=1, beta=1)
    graph = RuleBeliefGraph([strong, weak])
    assert graph.behavioral_cycles()
    graph.enforce_acyclicity()
    assert weak.tier == RuleTier.DEPRECATED
