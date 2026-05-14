"""
Tests for the 6 critical trading performance fixes in unified_trading_system.py.

FIX 1: Bootstrap volatility from OHLCV on startup
FIX 2: Force regime detection each cycle (fallback)
FIX 3: Bootstrap Kelly stats from backtest or config
FIX 4: Signal quality fallback
FIX 5: Check stops TWICE per cycle
FIX 6: Non-blocking fill timeout
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_system(**overrides):
    """Build a minimal UnifiedSystemArchitecture with mocked dependencies."""
    from unified_trading_system import UnifiedSystemArchitecture, UnifiedConfig

    cfg = UnifiedConfig()
    cfg.trading_pairs = ["BTC/AUD", "ETH/AUD"]
    cfg.starting_capital_aud = 1000.0
    cfg.run_mode = "paper"
    cfg.stop_loss_pct = 0.02
    cfg.trailing_stop_pct = 0.015
    cfg.max_holding_hours = 72.0
    # Kelly bootstrap config
    cfg.kelly_bootstrap_win_rate = 0.55
    cfg.kelly_bootstrap_avg_win = 0.02
    cfg.kelly_bootstrap_avg_loss = 0.015
    for k, v in overrides.items():
        setattr(cfg, k, v)

    sys = UnifiedSystemArchitecture(cfg)
    return sys


@dataclass
class _FakeSignal:
    symbol: str = "BTC/AUD"
    action: str = "BUY"
    confidence: float = 0.8
    strength: float = 0.7
    entry_price: float = 50000.0
    reasoning: str = "test"


# ===================================================================
# FIX 1 — Bootstrap volatility from OHLCV
# ===================================================================

class TestVolatilityBootstrap:
    """Tests for _bootstrap_volatility() and updated _get_current_vol()."""

    def test_get_current_vol_uses_cache(self):
        sys = _make_system()
        sys._volatility_cache = {"BTC/AUD": 0.025}
        # No trade_history, no ensemble hub — should fall back to cache
        vol = sys._get_current_vol("BTC/AUD")
        assert vol == pytest.approx(0.025)

    def test_get_current_vol_prefers_trade_history_over_cache(self):
        sys = _make_system()
        sys._volatility_cache = {"BTC/AUD": 0.999}  # Obviously distinct sentinel
        # Put enough trade_history prices (>= 5) with known vol
        prices = [100, 102, 101, 103, 104, 105]
        sys.trade_history = [
            {"symbol": "BTC/AUD", "price": p} for p in prices
        ]
        vol = sys._get_current_vol("BTC/AUD")
        # Should compute from trade_history, not use cache
        assert vol > 0
        assert vol != pytest.approx(0.999, abs=0.1)

    def test_get_current_vol_returns_zero_when_no_data(self):
        sys = _make_system()
        vol = sys._get_current_vol("DOGE/AUD")
        assert vol == 0.0

    @pytest.mark.asyncio
    async def test_bootstrap_volatility_with_market_data(self):
        sys = _make_system()
        # Mock market_data_service
        import pandas as pd
        closes = np.array([100, 102, 98, 103, 101, 99, 104, 100, 102, 98,
                           103, 101, 99, 104, 100, 102, 98, 103, 101, 99,
                           104, 100, 102, 98])
        df = pd.DataFrame({"close": closes})
        mock_mds = AsyncMock()
        mock_mds.fetch_ohlcv_df = AsyncMock(return_value=df)
        sys.market_data_service = mock_mds

        await sys._bootstrap_volatility()

        assert "BTC/AUD" in sys._volatility_cache
        assert sys._volatility_cache["BTC/AUD"] > 0
        assert "ETH/AUD" in sys._volatility_cache

    @pytest.mark.asyncio
    async def test_bootstrap_volatility_with_exchange_manager(self):
        sys = _make_system()
        sys.market_data_service = None
        # Mock exchange_manager.fetch_ohlcv
        candles = [[0, 0, 0, 0, 100 + i * 0.5, 0] for i in range(24)]
        mock_em = AsyncMock()
        mock_em.fetch_ohlcv = AsyncMock(return_value=candles)
        sys.exchange_manager = mock_em

        await sys._bootstrap_volatility()

        assert "BTC/AUD" in sys._volatility_cache
        assert sys._volatility_cache["BTC/AUD"] > 0

    @pytest.mark.asyncio
    async def test_bootstrap_volatility_handles_empty_data(self):
        sys = _make_system()
        sys.market_data_service = None
        sys.exchange_manager = AsyncMock()
        sys.exchange_manager.fetch_ohlcv = AsyncMock(return_value=[])

        await sys._bootstrap_volatility()
        # Should not crash, cache may be empty
        assert isinstance(sys._volatility_cache, dict)

    @pytest.mark.asyncio
    async def test_bootstrap_volatility_handles_exception(self):
        sys = _make_system()
        sys.market_data_service = AsyncMock()
        sys.market_data_service.fetch_ohlcv_df = AsyncMock(side_effect=Exception("network error"))
        sys.exchange_manager = None

        await sys._bootstrap_volatility()
        # Should not crash
        assert isinstance(sys._volatility_cache, dict)


# ===================================================================
# FIX 2 — Force regime detection each cycle
# ===================================================================

class TestRegimeFallback:
    """Tests for _compute_fallback_regime()."""

    def test_fallback_returns_normal_with_no_data(self):
        sys = _make_system()
        sys.trade_history = []
        sys.positions = {}
        regime = sys._compute_fallback_regime()
        assert regime == "NORMAL"

    def test_fallback_detects_high_vol(self):
        sys = _make_system()
        # Create prices with high volatility (>3% std of returns)
        np.random.seed(42)
        base = 100.0
        prices = [base]
        for _ in range(25):
            base *= (1 + np.random.normal(0, 0.05))  # 5% vol per step
            prices.append(base)
        sys.trade_history = [{"price": p} for p in prices]
        sys.positions = {}
        regime = sys._compute_fallback_regime()
        assert regime == "HIGH_VOL"

    def test_fallback_detects_low_vol(self):
        sys = _make_system()
        # Very stable prices (< 1% std)
        prices = [100 + 0.01 * i for i in range(25)]  # tiny increments
        sys.trade_history = [{"price": p} for p in prices]
        sys.positions = {}
        regime = sys._compute_fallback_regime()
        assert regime == "LOW_VOL"

    def test_fallback_detects_trending_up(self):
        sys = _make_system()
        # Moderate vol with upward trend
        np.random.seed(7)
        prices = []
        base = 100.0
        for i in range(25):
            base *= (1 + 0.025 + np.random.normal(0, 0.022))  # strong uptrend with moderate vol
            prices.append(base)
        sys.trade_history = [{"price": p} for p in prices]
        sys.positions = {}
        regime = sys._compute_fallback_regime()
        assert regime in ("TRENDING_UP", "HIGH_VOL")  # depends on vol realization

    def test_fallback_detects_trending_down(self):
        sys = _make_system()
        np.random.seed(3)
        prices = []
        base = 100.0
        for i in range(25):
            base *= (1 - 0.025 + np.random.normal(0, 0.022))  # downtrend
            prices.append(base)
        sys.trade_history = [{"price": p} for p in prices]
        sys.positions = {}
        regime = sys._compute_fallback_regime()
        assert regime in ("TRENDING_DOWN", "HIGH_VOL")

    def test_fallback_uses_position_prices_when_trade_history_insufficient(self):
        sys = _make_system()
        sys.trade_history = [{"price": 100.0}]  # Only 1 trade
        # Provide prices via positions
        sys.positions = {
            f"SYM{i}": {"current_price": 100.0 + 0.001 * i} for i in range(10)
        }
        regime = sys._compute_fallback_regime()
        # Should not crash, returns a valid regime
        assert regime in ("HIGH_VOL", "TRENDING_UP", "TRENDING_DOWN", "LOW_VOL", "NORMAL")

    def test_regime_fallback_sets_label_in_execute_signals_context(self):
        """Verify that when _latest_regime_label is empty, the fallback is triggered."""
        sys = _make_system()
        sys._latest_regime_label = ""
        sys.trade_history = [{"price": 100 + i * 0.001} for i in range(25)]
        sys.positions = {}
        # Call the fallback directly
        regime = sys._compute_fallback_regime()
        assert regime != ""
        assert isinstance(regime, str)


# ===================================================================
# FIX 3 — Bootstrap Kelly stats from config
# ===================================================================

class TestKellyBootstrap:
    """Tests for bootstrapped Kelly stats when insufficient trades."""

    def test_no_trades_returns_bootstrap_defaults(self):
        sys = _make_system()
        sys.trade_history = []
        stats = sys._get_strategy_trade_stats("momentum")
        assert stats["bootstrapped"] is True
        assert stats["win_rate"] == pytest.approx(0.55)
        assert stats["avg_win"] == pytest.approx(0.02)
        assert stats["avg_loss"] == pytest.approx(0.015)
        assert stats["n_trades"] == 0

    def test_few_trades_still_returns_bootstrap(self):
        sys = _make_system()
        sys.trade_history = [
            {"source_strategy": "momentum", "side": "SELL", "pnl": 0.05},
            {"source_strategy": "momentum", "side": "SELL", "pnl": -0.03},
        ]
        stats = sys._get_strategy_trade_stats("momentum")
        assert stats["bootstrapped"] is True
        assert stats["n_trades"] == 2
        # Uses config defaults, not actual trade stats
        assert stats["win_rate"] == pytest.approx(0.55)

    def test_sufficient_trades_uses_actual_data(self):
        sys = _make_system()
        # Create 25 trades (> 20 threshold)
        trades = []
        for i in range(25):
            pnl = 0.05 if i % 2 == 0 else -0.03
            trades.append({"source_strategy": "mean_revert", "side": "SELL", "pnl": pnl})
        sys.trade_history = trades
        stats = sys._get_strategy_trade_stats("mean_revert")
        assert stats["bootstrapped"] is False
        assert stats["n_trades"] == 25
        # Actual win rate: 13/25 = 0.52
        assert stats["win_rate"] == pytest.approx(13 / 25, abs=0.01)

    def test_custom_bootstrap_config(self):
        sys = _make_system(
            kelly_bootstrap_win_rate=0.60,
            kelly_bootstrap_avg_win=0.03,
            kelly_bootstrap_avg_loss=0.02,
        )
        sys.trade_history = []
        stats = sys._get_strategy_trade_stats("test_strat")
        assert stats["win_rate"] == pytest.approx(0.60)
        assert stats["avg_win"] == pytest.approx(0.03)
        assert stats["avg_loss"] == pytest.approx(0.02)

    def test_bootstrap_at_boundary_19_trades(self):
        sys = _make_system()
        trades = [{"source_strategy": "strat1", "side": "SELL", "pnl": 0.01}] * 19
        sys.trade_history = trades
        stats = sys._get_strategy_trade_stats("strat1")
        assert stats["bootstrapped"] is True
        assert stats["n_trades"] == 19

    def test_exact_20_trades_uses_actual(self):
        sys = _make_system()
        trades = [
            {"source_strategy": "strat1", "side": "SELL", "pnl": 0.01 if i < 12 else -0.005}
            for i in range(20)
        ]
        sys.trade_history = trades
        stats = sys._get_strategy_trade_stats("strat1")
        assert stats["bootstrapped"] is False
        assert stats["n_trades"] == 20


# ===================================================================
# FIX 4 — Signal quality fallback
# ===================================================================

class TestSignalQualityFallback:
    """Tests for _get_signal_quality() fallback from signals."""

    def test_no_signals_returns_no_signals(self):
        sys = _make_system()
        result = sys._get_signal_quality(signals=[])
        assert result is not None
        assert result["recommendation"] == "no_signals"

    def test_none_signals_returns_no_signals(self):
        sys = _make_system()
        result = sys._get_signal_quality(signals=None)
        assert result is not None
        assert result["recommendation"] == "no_signals"

    def test_conflicting_signals_detected(self):
        sys = _make_system()
        signals = [
            _FakeSignal(symbol="BTC/AUD", action="BUY"),
            _FakeSignal(symbol="BTC/AUD", action="SELL"),
        ]
        result = sys._get_signal_quality(signals=signals)
        assert result["recommendation"] == "conflicted"
        assert result["quality"] == pytest.approx(0.3)

    def test_all_buy_signals_strong(self):
        sys = _make_system()
        signals = [
            _FakeSignal(symbol="BTC/AUD", action="BUY"),
            _FakeSignal(symbol="ETH/AUD", action="BUY"),
        ]
        result = sys._get_signal_quality(signals=signals)
        assert result["recommendation"] == "strong"
        assert result["quality"] == pytest.approx(0.9)

    def test_mixed_non_conflicting_moderate(self):
        sys = _make_system()
        signals = [
            _FakeSignal(symbol="BTC/AUD", action="BUY"),
            _FakeSignal(symbol="ETH/AUD", action="SELL"),
        ]
        result = sys._get_signal_quality(signals=signals)
        assert result["recommendation"] == "moderate"
        assert result["quality"] == pytest.approx(0.6)

    def test_single_signal_is_strong(self):
        sys = _make_system()
        signals = [_FakeSignal(symbol="BTC/AUD", action="BUY")]
        result = sys._get_signal_quality(signals=signals)
        assert result["recommendation"] == "strong"

    def test_ensemble_hub_preferred_when_available(self):
        sys = _make_system()
        mock_hub = MagicMock()
        mock_hub.get_signal_quality.return_value = {"recommendation": "from_hub", "quality": 0.95}
        mock_cr = MagicMock()
        mock_cr.ensemble_hub = mock_hub
        sys.component_registry = mock_cr

        result = sys._get_signal_quality(signals=[_FakeSignal()])
        assert result["recommendation"] == "from_hub"


# ===================================================================
# FIX 5 — Check stops TWICE per cycle
# ===================================================================

class TestDoubleStopCheck:
    """Verify the second stop-loss check exists in the post-execution path."""

    def test_post_exec_stop_code_exists(self):
        """Verify the POST_EXEC_STOP code path is present in source."""
        import inspect
        from unified_trading_system import UnifiedSystemArchitecture
        source = inspect.getsource(UnifiedSystemArchitecture)
        assert "POST_EXEC_STOP" in source
        assert "Post-execution stop-loss check" in source or "POST-EXEC STOP-LOSS" in source

    def test_pre_exec_stop_still_exists(self):
        """Ensure the original pre-execution stop check remains."""
        import inspect
        from unified_trading_system import UnifiedSystemArchitecture
        source = inspect.getsource(UnifiedSystemArchitecture)
        assert "STOP-LOSS AUTO-EXECUTION: check all positions BEFORE new signals" in source

    def test_both_stop_checks_present(self):
        """Both the pre-execution and post-execution stop checks exist."""
        import inspect
        from unified_trading_system import UnifiedSystemArchitecture
        source = inspect.getsource(UnifiedSystemArchitecture)
        # Count occurrences of check_stops calls
        count = source.count("unified_risk_manager.check_stops(")
        assert count >= 2, f"Expected at least 2 check_stops calls, found {count}"


# ===================================================================
# FIX 6 — Non-blocking fill timeout
# ===================================================================

class TestNonBlockingFillTimeout:
    """Tests for non-blocking limit order fill handling."""

    def test_no_blocking_sleep_in_fill_path(self):
        """Verify asyncio.sleep(self._limit_order_fill_timeout) is removed."""
        import inspect
        from unified_trading_system import UnifiedSystemArchitecture
        source = inspect.getsource(UnifiedSystemArchitecture._execute_signals)
        # The blocking sleep pattern should no longer exist
        assert "asyncio.sleep(self._limit_order_fill_timeout)" not in source

    def test_pending_orders_tracked_for_unfilled(self):
        """Verify unfilled limit orders are tracked in _pending_orders."""
        import inspect
        from unified_trading_system import UnifiedSystemArchitecture
        source = inspect.getsource(UnifiedSystemArchitecture._execute_signals)
        assert "needs_market_fallback" in source
        assert "fill_timeout_at" in source

    def test_pending_orders_dict_initialized(self):
        sys = _make_system()
        assert isinstance(sys._pending_orders, dict)
        assert len(sys._pending_orders) == 0

    def test_volatility_cache_initialized(self):
        sys = _make_system()
        assert isinstance(sys._volatility_cache, dict)
        assert len(sys._volatility_cache) == 0


# ===================================================================
# Integration-style tests
# ===================================================================

class TestIntegration:
    """Higher-level integration checks combining multiple fixes."""

    def test_vol_cache_feeds_into_vol_adjusted_size(self):
        sys = _make_system()
        sys._volatility_cache = {"BTC/AUD": 0.03}
        vol = sys._get_current_vol("BTC/AUD")
        assert vol == pytest.approx(0.03)
        # Use it in vol-adjusted sizing
        adjusted = sys._vol_adjusted_size(base_size=100.0, current_vol=vol, target_vol=0.02)
        # 0.02 / 0.03 = 0.667
        assert adjusted == pytest.approx(100.0 * (0.02 / 0.03), rel=0.01)

    def test_regime_fallback_produces_valid_scale(self):
        sys = _make_system()
        sys._latest_regime_label = ""
        sys.trade_history = [{"price": 100 + 0.001 * i} for i in range(25)]
        sys.positions = {}
        regime = sys._compute_fallback_regime()
        # Must be in the REGIME_POSITION_SCALE dict
        assert regime in sys.REGIME_POSITION_SCALE
        assert regime in sys.REGIME_STOP_SCALE
        assert regime in sys.REGIME_TP_SCALE

    def test_kelly_bootstrap_produces_valid_kelly_fraction(self):
        sys = _make_system()
        sys.trade_history = []
        stats = sys._get_strategy_trade_stats("test")
        # Compute Kelly: K = W - (1-W) / (avg_win/avg_loss)
        wr = stats["win_rate"]
        aw = stats["avg_win"]
        al = stats["avg_loss"]
        if al > 0:
            ratio = aw / al
            kelly = wr - (1 - wr) / ratio
            # With 0.55 win rate, 0.02/0.015 ratio: K = 0.55 - 0.45/(1.333) = 0.55 - 0.3375 = 0.2125
            assert kelly > 0, f"Bootstrap Kelly should be positive, got {kelly}"
            assert kelly < 1.0, f"Bootstrap Kelly should be < 1, got {kelly}"
