import pytest
from argus_live.state.degraded_state import SystemState, set_state, assert_can_trade

def test_degraded_recon_blocks_trading() -> None:
    set_state(SystemState.DEGRADED_RECON)
    with pytest.raises(RuntimeError):
        assert_can_trade()
