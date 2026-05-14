"""
Push 89 — Tests: ArgusSystem wiring of BanditRouter + LedgerFillObserver

These tests validate the integration seams without importing the full
exchange adapter stack — they mock out the heavy core sub-systems and
focus on the Push 87+88 wiring.
"""
from __future__ import annotations

import time
import types
from unittest.mock import MagicMock, patch

import pytest

from core.system import ArgusSystem, SystemConfig
from strategies.bandit_router import StrategyStatus
from strategies.regime_consensus import MarketRegime


# ---------------------------------------------------------------------------
# Minimal stubs so we can build ArgusSystem without a live exchange
# ---------------------------------------------------------------------------

def _make_system(equity: float = 5_000.0, bandit: bool = True) -> ArgusSystem:
    cfg = SystemConfig(
        paper_mode=True,
        initial_equity=equity,
        initial_balance=equity * 10,
        strategies=[
            {"name": "momentum",      "strategy_id": "mom_BTC", "symbol": "BTCUSDT"},
            {"name": "mean_reversion","strategy_id": "mr_BTC",  "symbol": "BTCUSDT"},
        ],
        bandit_enabled=bandit,
        ledger_db_path=":memory:",   # SQLite in-memory for tests
        fills_db_path=":memory:",
    )
    sys = ArgusSystem(cfg)
    sys._build()
    return sys


# ---------------------------------------------------------------------------
# Build / init tests
# ---------------------------------------------------------------------------

class TestSystemBuild:

    def test_ledger_created(self):
        sys = _make_system()
        assert sys.ledger is not None

    def test_fill_observer_created(self):
        sys = _make_system()
        assert sys.fill_observer is not None
        assert sys.fill_observer.ledger is sys.ledger

    def test_bandit_router_created_when_enabled(self):
        sys = _make_system(bandit=True)
        assert sys.bandit_router is not None

    def test_bandit_router_none_when_disabled(self):
        sys = _make_system(bandit=False)
        assert sys.bandit_router is None

    def test_bandit_router_has_correct_strategies(self):
        sys = _make_system()
        allocs = sys.bandit_router.allocations()
        assert "momentum" in allocs
        assert "mean_reversion" in allocs

    def test_capital_equals_initial_equity(self):
        sys = _make_system(equity=8_000.0)
        allocs = sys.bandit_router.allocations()
        assert sum(allocs.values()) == pytest.approx(8_000.0, rel=1e-4)


# ---------------------------------------------------------------------------
# record_fill_outcome integration
# ---------------------------------------------------------------------------

class TestFillOutcomeFeedback:

    def test_fill_posts_to_ledger(self):
        sys = _make_system()
        sys.record_fill_outcome(
            strategy_name="momentum",
            symbol="BTC/USD",
            side="buy",
            expected_price=60_000.0,
            actual_price=60_100.0,
            quantity_usd=500.0,
        )
        pos = sys.ledger.get_position("momentum", "BTC/USD")
        assert pos.net_qty_usd == pytest.approx(500.0)

    def test_fill_updates_bandit_router(self):
        sys = _make_system()
        for _ in range(20):
            sys.record_fill_outcome(
                strategy_name="momentum",
                symbol="BTC/USD",
                side="buy",
                expected_price=60_000.0,
                actual_price=59_000.0,
                quantity_usd=100.0,
            )
        snap = sys.bandit_router.bandit_snapshot()
        mom = next((s for s in snap if s["name"] == "momentum"), None)
        assert mom is not None
        assert mom["pulls"] == 20

    def test_unknown_strategy_fill_does_not_raise(self):
        sys = _make_system()
        sys.record_fill_outcome(
            strategy_name="ghost",
            symbol="ETH/USD",
            side="sell",
            expected_price=3_000.0,
            actual_price=2_990.0,
            quantity_usd=200.0,
        )


# ---------------------------------------------------------------------------
# Regime routing
# ---------------------------------------------------------------------------

class TestRegimeRouting:

    def test_regime_updated_from_config(self):
        sys = _make_system()
        sys.config.market_regime = "RANGE"
        allocs_range = sys.bandit_router.allocations(regime=MarketRegime.RANGE)
        assert sum(allocs_range.values()) == pytest.approx(
            sys.config.initial_equity, rel=1e-4
        )

    def test_high_vol_zeroes_scalping(self):
        """In HIGH_VOL regime scalping strategies receive 0 allocation."""
        from strategies.bandit_router import StrategySpec
        from strategies.bandit_router import BanditRouter
        router = BanditRouter(
            total_capital_usd=10_000.0,
            strategies=[
                StrategySpec(name="micro_capital_mm", category="scalping"),
                StrategySpec(name="funding_rate_arb", category="arb"),
            ],
        )
        allocs = router.allocations(regime=MarketRegime.HIGH_VOL)
        assert allocs["micro_capital_mm"] == pytest.approx(0.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Stats snapshot
# ---------------------------------------------------------------------------

class TestStatsSnapshot:

    def test_stats_includes_bandit_router(self):
        sys = _make_system()
        sys._running = True
        sys._start_time = time.time()
        s = sys.stats()
        assert "bandit_router" in s
        assert isinstance(s["bandit_router"], list)

    def test_stats_version_updated(self):
        sys = _make_system()
        sys._running = True
        sys._start_time = time.time()
        assert sys.stats()["version"] == "8.25.0"
        assert sys.stats()["codename"] == "SystemWiring"

    def test_stats_regime_present(self):
        sys = _make_system()
        sys._running = True
        sys._start_time = time.time()
        assert "regime" in sys.stats()
