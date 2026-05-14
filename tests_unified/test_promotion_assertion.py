import pytest
from argus_live.promotion.promotion_gate import assert_strategy_allowed

def test_unapproved_strategy_raises() -> None:
    with pytest.raises(RuntimeError):
        assert_strategy_allowed("bad_strategy", "ok")
