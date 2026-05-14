import pytest
from argus_live.constitution.scaling_rules import assert_aum_safe

def test_aum_breach_raises() -> None:
    with pytest.raises(RuntimeError):
        assert_aum_safe(current_aum=20000, max_safe_aum=10000)
