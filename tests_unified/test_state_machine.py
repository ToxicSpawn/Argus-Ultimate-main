from argus_live.execution.state_machine import validate_transition


def test_proposed_can_go_to_target_approved() -> None:
    assert validate_transition("PROPOSED", "TARGET_APPROVED").ok is True


def test_proposed_cannot_jump_to_filled() -> None:
    assert validate_transition("PROPOSED", "FILLED").ok is False
