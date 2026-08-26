from adaptiveguard.core.rules import ConstraintRule, RuleBeliefGraph
from adaptiveguard.modules.scb import ScbConfig, StratifiedConstraintBelief


def test_scb_retrieves_only_confirmed_blockers() -> None:
    confirmed = ConstraintRule(
        "heater", "requires", "charge", alpha=4, beta=1, metadata={"action_template": "turn on"}
    )
    tentative = ConstraintRule("heater", "near", "pot", metadata={"action_template": "turn on"})
    scb = StratifiedConstraintBelief(RuleBeliefGraph([confirmed, tentative]), ScbConfig())
    scb.refresh()
    assert scb.confirmed_blockers("turn on heater") == [confirmed]
    assert scb.tentative_matches("turn on heater") == [tentative]


def test_refutation_decreases_confidence() -> None:
    rule = ConstraintRule("x", "r", "y", alpha=4, beta=1)
    scb = StratifiedConstraintBelief(RuleBeliefGraph([rule]))
    before = rule.confidence
    scb.update_rule(rule.rule_id, refute=1)
    assert rule.confidence < before
