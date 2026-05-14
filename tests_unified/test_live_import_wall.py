import pytest

from argus_live.control_plane.import_wall import assert_live_import_allowed


@pytest.mark.parametrize(
    "module_name",
    [
        "quantum",
        "quantum.optimizer",
        "research",
        "research.alpha_lab",
        "core.ARGUS.core.runtime",
        "enhanced_trading_launcher",
    ],
)
def test_blocked_live_imports(module_name: str) -> None:
    with pytest.raises(RuntimeError):
        assert_live_import_allowed(module_name)
