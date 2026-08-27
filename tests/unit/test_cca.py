import pytest

from adaptiveguard.bootstrap.hypotheses import typed_rule_factory
from adaptiveguard.core.rules import ConstraintRule, RuleBeliefGraph
from adaptiveguard.core.types import EvidenceVector, HcaLabel, Judgment
from adaptiveguard.modules.cca import ContrastiveCausalAttribution


def test_default_fusion() -> None:
    cca = ContrastiveCausalAttribution()
    q = cca.fuse(EvidenceVector(1.0, 0.0, -0.5))
    assert q == pytest.approx(0.15)
    assert cca.judgment(q) == Judgment.UNCERTAIN


def test_invalid_hca_is_removed_from_fusion() -> None:
    cca = ContrastiveCausalAttribution()
    q = cca.fuse(EvidenceVector(1.0, -1.0, -1.0), hca_valid=False)
    assert q == pytest.approx(0.166, abs=0.002)


def test_context_weights_take_precedence_over_mode() -> None:
    cca = ContrastiveCausalAttribution()
    evidence = EvidenceVector(1.0, 0.0, -1.0)

    assert cca.fuse(evidence, mode="environment_error", hca_valid=False) == pytest.approx(0.583)


def test_tentative_rule_updates_are_filtered_by_hca_graph_type() -> None:
    matching = ConstraintRule("a", "r", "b", metadata={"graph_type": "G_beh"})
    excluded = ConstraintRule("c", "r", "d", metadata={"graph_type": "G_env.E_1"})
    graph = RuleBeliefGraph([matching, excluded])
    cca = ContrastiveCausalAttribution()

    updates = cca.update_rules(
        graph,
        action="a",
        judgment=Judgment.BAD,
        q_value=-1,
        tentative_rules=[matching, excluded],
        hca_label=HcaLabel.MISSING_PRECOND,
    )

    assert updates == [{"rule_id": matching.rule_id, "support": 1.0}]
    assert matching.alpha == 2.0
    assert excluded.alpha == 1.0


def test_uncertain_tentative_rule_uses_q_value_sign() -> None:
    rule = ConstraintRule("a", "r", "b")
    cca = ContrastiveCausalAttribution()

    positive = cca.update_rules(
        RuleBeliefGraph([rule]),
        action="a",
        judgment=Judgment.UNCERTAIN,
        q_value=0.5,
        tentative_rules=[rule],
    )

    assert positive == [{"rule_id": rule.rule_id, "refute": cca.config.eta}]


def test_good_trial_refutes_rule() -> None:
    rule = ConstraintRule("a", "r", "b", alpha=3, beta=1)
    graph = RuleBeliefGraph([rule])
    cca = ContrastiveCausalAttribution()
    cca.update_rules(
        graph, action="a", judgment=Judgment.GOOD, q_value=1, trial_rule_id=rule.rule_id
    )
    assert rule.beta == 2


def test_bad_unmatched_action_creates_typed_hypothesis() -> None:
    graph = RuleBeliefGraph()
    cca = ContrastiveCausalAttribution()
    updates = cca.update_rules(
        graph,
        action="open fridge",
        judgment=Judgment.BAD,
        q_value=-1,
        hypothesis_factory=typed_rule_factory,
        hca_label=HcaLabel.MISSING_PRECOND,
        evidence={},
    )
    assert len(graph) == 1
    assert updates[0]["created"] is True
