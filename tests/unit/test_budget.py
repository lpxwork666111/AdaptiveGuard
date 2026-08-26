from adaptiveguard.core.budget import TrialBudget


def test_budget_updates_and_caps() -> None:
    budget = TrialBudget(base=2, maximum=4, refund=0.5)
    budget.on_success()
    assert budget.remaining == 2.5
    for _ in range(10):
        budget.on_success()
    assert budget.remaining == 4
    for _ in range(10):
        budget.on_failure()
    assert budget.remaining == 0
    assert not budget.can_trial()
    budget.reset()
    assert budget.remaining == 2
