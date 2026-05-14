from argus_live.optimization.learning_apply_gate import should_apply_learning_update

def test_learning_apply_gate_requires_all_conditions() -> None:
    denied = should_apply_learning_update(
        replay_passed=False,
        promotion_passed=True,
        operator_approved=True,
    )
    assert denied.apply is False

    approved = should_apply_learning_update(
        replay_passed=True,
        promotion_passed=True,
        operator_approved=True,
    )
    assert approved.apply is True
