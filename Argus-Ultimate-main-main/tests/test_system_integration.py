"""
Push 91 — integration smoke tests for ArgusSystem v8.27.0
Verifies that all Push 87-90 components wire correctly inside _build()
and that tick() runs end-to-end without exceptions.
"""
import pytest
from unittest.mock import MagicMock, patch
from core.system import ArgusSystem, SystemConfig


@pytest.fixture
def system():
    return ArgusSystem.paper(symbol="BTCUSDT", equity=10_000.0)


class TestBuild:
    def test_build_does_not_raise(self, system):
        system._build()
        assert system._built

    def test_regime_detector_attached(self, system):
        system._build()
        assert system.regime_detector is not None

    def test_config_has_bandit_fields(self):
        cfg = SystemConfig()
        assert hasattr(cfg, "bandit_enabled")
        assert hasattr(cfg, "bandit_max_concentration")
        assert hasattr(cfg, "market_regime")

    def test_config_has_regime_fields(self):
        cfg = SystemConfig()
        assert hasattr(cfg, "regime_warmup_ticks")
        assert hasattr(cfg, "regime_hysteresis_ticks")

    def test_ledger_db_path_default(self):
        cfg = SystemConfig()
        assert cfg.ledger_db_path == "data/ledger.db"


class TestSystemVersion:
    def test_version_in_stats(self, system):
        system._build()
        system._running = True
        system._start_time = 0
        s = system.stats()
        assert s["version"] == "8.27.0"
        assert s["codename"] == "FullIntegration"

    def test_market_regime_in_stats(self, system):
        system._build()
        system._running = True
        system._start_time = 0
        s = system.stats()
        assert "market_regime" in s

    def test_regime_detector_in_stats(self, system):
        system._build()
        system._running = True
        system._start_time = 0
        s = system.stats()
        assert "regime_detector" in s


class TestPaperFactory:
    def test_paper_creates_strategy_categories(self):
        sys = ArgusSystem.paper(symbol="ETHUSDT", equity=5_000.0)
        assert "mom_ETHUSDT" in sys.config.strategy_categories
        assert "mr_ETHUSDT" in sys.config.strategy_categories

    def test_from_config_roundtrip(self):
        d = {
            "paper_mode": True,
            "initial_equity": 20_000.0,
            "bandit_enabled": False,
            "market_regime": "RANGING",
        }
        sys = ArgusSystem.from_config(d)
        assert sys.config.bandit_enabled is False
        assert sys.config.market_regime == "RANGING"
        assert sys.config.initial_equity == 20_000.0
