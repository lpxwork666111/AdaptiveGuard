from adaptiveguard.core.rules import ConstraintRule
from adaptiveguard.core.types import ActionDecision, RuleTier
from adaptiveguard.modules.pft import PftConfig, PurposefulFalsificationTrials


def _blocker(alpha: float, beta: float, name: str) -> ConstraintRule:
    return ConstraintRule(name, "requires", "x", alpha=alpha, beta=beta, tier=RuleTier.CONFIRMED)


def test_allow_without_blocker() -> None:
    pft = PurposefulFalsificationTrials()
    assert pft.decide("act", [], 0.9).decision == ActionDecision.ALLOW


def test_trial_when_gate_and_budget_pass() -> None:
    pft = PurposefulFalsificationTrials(PftConfig(tau=0.25))
    decision = pft.decide("act", [_blocker(1, 1, "a")], 0.9, criticality=1.0)
    assert decision.decision == ActionDecision.TRIAL
    assert next(iter(decision.scores.values())) == 0.45


def test_every_blocker_must_pass() -> None:
    pft = PurposefulFalsificationTrials(PftConfig(tau=0.25))
    decision = pft.decide("act", [_blocker(1, 1, "a"), _blocker(9, 1, "b")], 0.9, criticality=1.0)
    assert decision.decision == ActionDecision.BLOCK


def test_failed_trial_consumes_budget() -> None:
    pft = PurposefulFalsificationTrials(PftConfig(budget_base=1, budget_max=2))
    pft.record_outcome(ActionDecision.TRIAL, False)
    assert pft.budget.remaining == 0
    assert pft.failed_trials == 1
