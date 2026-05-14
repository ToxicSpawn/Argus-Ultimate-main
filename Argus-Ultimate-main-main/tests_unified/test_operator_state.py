from argus_live.state.operator_state import OperatorMode, set_mode, load_state

def test_operator_mode_roundtrip() -> None:
    set_mode(OperatorMode.HALTED)
    state = load_state()
    assert state.mode == OperatorMode.HALTED
