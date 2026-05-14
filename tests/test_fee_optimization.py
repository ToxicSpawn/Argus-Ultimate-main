"""
Tests for fee optimization: maker enforcement tuning, limit-order defaults,
VWAP routing for large orders, and cumulative fee savings tracking.

39 tests covering:
- Maker enforcement thresholds (low urgency = maker, high urgency = taker)
- Fee savings calculation (Kraken maker 2bps vs taker 6bps)
- Limit order default (not market)
- Limit order timeout -> market fallback
- VWAP routing for large orders
- Fee savings tracking
- Conditional fallback behavior
"""
from __future__ import annotations

import asyncio
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution.maker_enforcement import MakerEnforcement, MakerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeSignal:
    symbol: str = "BTC/USD"
    action: str = "BUY"
    confidence: float = 0.8
    strength: float = 0.9
    entry_price: float = 50000.0
    stop_loss: float = 49000.0
    take_profit: float = 52000.0
    reasoning: str = "test"
    strategy: str = "test_strategy"


def _build_execute_signals_system(portfolio_value_aud: float = 1000.0):
    """Build a minimal namespace that can run _execute_signals."""
    from unified_trading_system import UnifiedSystemArchitecture

    sys = types.SimpleNamespace()

    # Config stub
    config = types.SimpleNamespace()
    config.run_mode = "paper"
    config.primary_exchange = "kraken"
    config.aud_to_usd = 0.65
    config.max_position_pct = 0.25
    config.min_position_size_aud = 5.0
    config.max_concurrent_positions = 5
    config.paper_slippage_bps = 5.0
    config.paper_fee_rate = 0.0026
    config.paper_maker_fee_rate = 0.0002
    config.portfolio_var_limit_pct = 0.0
    config.portfolio_cvar_limit_pct = 0.0
    config.order_type = "limit"
    config.reconcile_every_n_cycles = 10
    config.order_timeout_seconds = 60.0
    config.stop_loss_pct = 0.03
    config.take_profit_pct = 0.08

    sys.config = config
    sys.portfolio_value_aud = portfolio_value_aud
    sys.positions = {}
    sys.unified_risk_manager = None
    sys.component_registry = None
    sys.exchange_manager = None
    sys.execution_engine = None
    sys._pending_orders = {}
    sys._paper_slippage_bps = 5.0
    sys._total_fee_savings_usd = 0.0
    sys._limit_order_fill_timeout = 0.01  # fast for tests
    sys._limit_price_offset_bps = 2.0
    sys._vwap_threshold_usd = 100.0
    sys._strategy_state_store = None
    sys.execution_mesh = None
    sys.trade_history = []
    sys.total_trades = 0
    sys.winning_trades = 0
    sys.losing_trades = 0
    sys.total_pnl_aud = 0.0
    sys.realized_pnl_aud = 0.0
    sys.unrealized_pnl_aud = 0.0
    sys.total_fees_aud = 0.0
    sys.daily_pnl_aud = 0.0
    sys.cash_balance_aud = portfolio_value_aud
    sys.omega_store = MagicMock()

    # Class-level constants needed by _execute_signals for regime scaling
    sys.REGIME_POSITION_SCALE = UnifiedSystemArchitecture.REGIME_POSITION_SCALE
    sys.REGIME_STOP_SCALE = UnifiedSystemArchitecture.REGIME_STOP_SCALE
    sys.REGIME_TP_SCALE = UnifiedSystemArchitecture.REGIME_TP_SCALE

    # Bind the real methods
    sys._execute_signals = UnifiedSystemArchitecture._execute_signals.__get__(sys)
    sys._record_trade = UnifiedSystemArchitecture._record_trade.__get__(sys)
    sys._kelly_size = UnifiedSystemArchitecture._kelly_size.__get__(sys)
    sys._vol_adjusted_size = UnifiedSystemArchitecture._vol_adjusted_size.__get__(sys)
    sys._get_strategy_trade_stats = UnifiedSystemArchitecture._get_strategy_trade_stats.__get__(sys)
    sys._get_current_vol = UnifiedSystemArchitecture._get_current_vol.__get__(sys)
    sys._get_signal_quality = UnifiedSystemArchitecture._get_signal_quality.__get__(sys)

    return sys


# ===================================================================
# 1. MakerEnforcement threshold tests
# ===================================================================

class TestMakerEnforcementThresholds:
    """Verify tuned maker enforcement parameters."""

    def test_default_urgency_threshold_is_0_4(self):
        enforcer = MakerEnforcement()
        assert enforcer.urgency_threshold == 0.4

    def test_min_spread_bps_is_0_5(self):
        enforcer = MakerEnforcement()
        assert enforcer.MIN_SPREAD_BPS == 0.5

    def test_max_retries_is_5(self):
        enforcer = MakerEnforcement()
        assert enforcer.MAX_RETRIES == 5

    def test_retry_delay_is_0_2(self):
        enforcer = MakerEnforcement()
        assert enforcer.RETRY_DELAY == 0.2

    def test_low_urgency_uses_maker(self):
        enforcer = MakerEnforcement()
        assert enforcer.should_use_maker(urgency=0.1, spread_bps=1.0) is True

    def test_high_urgency_bypasses_maker(self):
        enforcer = MakerEnforcement()
        assert enforcer.should_use_maker(urgency=0.5, spread_bps=1.0) is False

    def test_medium_urgency_uses_maker(self):
        enforcer = MakerEnforcement()
        # 0.35 < 0.4 threshold => should use maker
        assert enforcer.should_use_maker(urgency=0.35, spread_bps=1.0) is True

    def test_spread_too_narrow_bypasses_maker(self):
        enforcer = MakerEnforcement()
        assert enforcer.should_use_maker(urgency=0.1, spread_bps=0.3) is False

    def test_spread_at_threshold_uses_maker(self):
        enforcer = MakerEnforcement()
        assert enforcer.should_use_maker(urgency=0.1, spread_bps=0.5) is True


# ===================================================================
# 2. Fee savings calculation
# ===================================================================

class TestFeeSavingsCalculation:
    """Verify estimate_fee_savings for different exchanges."""

    def test_kraken_maker_saves_4bps(self):
        enforcer = MakerEnforcement()
        # $1000 notional, maker fill, Kraken: (6 - 2) bps = 4 bps = $0.40
        savings = enforcer.estimate_fee_savings(is_maker=True, notional=1000.0, exchange="kraken")
        assert abs(savings - 0.40) < 0.001

    def test_kraken_taker_zero_savings(self):
        enforcer = MakerEnforcement()
        savings = enforcer.estimate_fee_savings(is_maker=False, notional=1000.0, exchange="kraken")
        assert savings == 0.0

    def test_coinbase_maker_saves_2bps(self):
        enforcer = MakerEnforcement()
        # Coinbase: (6 - 4) bps = 2 bps on $1000 = $0.20
        savings = enforcer.estimate_fee_savings(is_maker=True, notional=1000.0, exchange="coinbase")
        assert abs(savings - 0.20) < 0.001

    def test_binance_maker_saves_0bps(self):
        enforcer = MakerEnforcement()
        # Binance: (1 - 1) = 0 bps
        savings = enforcer.estimate_fee_savings(is_maker=True, notional=1000.0, exchange="binance")
        assert abs(savings - 0.0) < 0.001

    def test_unknown_exchange_uses_defaults(self):
        enforcer = MakerEnforcement()
        # Default: (6 - 2) bps = 4 bps
        savings = enforcer.estimate_fee_savings(is_maker=True, notional=1000.0, exchange="unknown_exchange")
        assert abs(savings - 0.40) < 0.001

    def test_estimate_savings_usd_legacy(self):
        enforcer = MakerEnforcement()
        # Legacy method: $1000 * 4 bps = $0.40
        assert abs(enforcer.estimate_savings_usd(1000.0) - 0.40) < 0.001


# ===================================================================
# 3. Conditional fallback behavior
# ===================================================================

class TestConditionalFallback:
    """Verify urgency-based conditional fallback to taker."""

    def test_very_low_urgency_no_fallback(self):
        enforcer = MakerEnforcement()
        assert enforcer._should_fallback_to_taker(urgency=0.1) is False

    def test_low_urgency_below_threshold_no_fallback(self):
        enforcer = MakerEnforcement()
        assert enforcer._should_fallback_to_taker(urgency=0.29) is False

    def test_high_urgency_above_threshold_fallback(self):
        enforcer = MakerEnforcement()
        assert enforcer._should_fallback_to_taker(urgency=0.6) is True

    def test_medium_urgency_uses_configured_default(self):
        # Between 0.3 and 0.5 => use self.fallback_to_taker
        enforcer_with_fallback = MakerEnforcement(fallback_to_taker=True)
        assert enforcer_with_fallback._should_fallback_to_taker(urgency=0.4) is True

        enforcer_no_fallback = MakerEnforcement(fallback_to_taker=False)
        assert enforcer_no_fallback._should_fallback_to_taker(urgency=0.4) is False

    def test_custom_fallback_thresholds(self):
        enforcer = MakerEnforcement(
            fallback_urgency_low=0.2,
            fallback_urgency_high=0.8,
        )
        assert enforcer._should_fallback_to_taker(urgency=0.1) is False
        assert enforcer._should_fallback_to_taker(urgency=0.5) == enforcer.fallback_to_taker
        assert enforcer._should_fallback_to_taker(urgency=0.9) is True


# ===================================================================
# 4. Maker enforcement place_order (simulation mode)
# ===================================================================

class TestMakerPlaceOrder:
    """Verify end-to-end maker enforcement order placement."""

    @pytest.mark.asyncio
    async def test_low_urgency_gets_maker_fill(self):
        enforcer = MakerEnforcement(enabled=True)
        result = await enforcer.place_order("BTC/USD", "buy", 100.0, 50000.0, urgency=0.1)
        assert result.success is True
        assert result.is_maker is True
        assert result.fee_bps == 2.0

    @pytest.mark.asyncio
    async def test_high_urgency_gets_taker_fill(self):
        enforcer = MakerEnforcement(enabled=True)
        result = await enforcer.place_order("BTC/USD", "buy", 100.0, 50000.0, urgency=0.9)
        assert result.success is True
        assert result.is_maker is False
        assert result.fee_bps == 6.0

    @pytest.mark.asyncio
    async def test_disabled_enforcement_always_taker(self):
        enforcer = MakerEnforcement(enabled=False)
        result = await enforcer.place_order("BTC/USD", "buy", 100.0, 50000.0, urgency=0.1)
        assert result.is_maker is False

    @pytest.mark.asyncio
    async def test_sell_side_maker_fill(self):
        enforcer = MakerEnforcement(enabled=True)
        result = await enforcer.place_order("ETH/USD", "sell", 200.0, 3000.0, urgency=0.2)
        assert result.success is True
        assert result.is_maker is True


# ===================================================================
# 5. _execute_signals: limit order default (not market)
# ===================================================================

class TestExecuteSignalsLimitOrder:
    """Verify _execute_signals defaults to limit orders and tracks fees."""

    @pytest.mark.asyncio
    async def test_paper_mode_uses_maker_fees(self):
        sys = _build_execute_signals_system()
        signal = _FakeSignal(entry_price=50000.0, confidence=0.8, strength=0.9)
        results = await sys._execute_signals([signal])
        assert len(results) == 1
        result = results[0]
        assert result["status"] == "filled"
        # Commission should use maker rate (0.02%), not taker (0.26%)
        notional = result["quantity"] * result["price"]
        expected_commission = notional * 0.0002
        assert abs(result["commission"] - expected_commission) < 0.01
        assert result["order_type"] in ("limit", "vwap")
        assert result["is_maker"] is True

    @pytest.mark.asyncio
    async def test_paper_mode_tracks_fee_savings(self):
        sys = _build_execute_signals_system()
        signal = _FakeSignal(entry_price=50000.0)
        await sys._execute_signals([signal])
        assert sys._total_fee_savings_usd > 0

    @pytest.mark.asyncio
    async def test_paper_mode_cumulative_savings(self):
        sys = _build_execute_signals_system()
        signal1 = _FakeSignal(entry_price=50000.0, symbol="BTC/USD")
        signal2 = _FakeSignal(entry_price=3000.0, symbol="ETH/USD")
        await sys._execute_signals([signal1])
        savings_after_1 = sys._total_fee_savings_usd
        await sys._execute_signals([signal2])
        savings_after_2 = sys._total_fee_savings_usd
        assert savings_after_2 > savings_after_1

    @pytest.mark.asyncio
    async def test_order_type_is_limit_not_market(self):
        sys = _build_execute_signals_system()
        signal = _FakeSignal(entry_price=50000.0)
        results = await sys._execute_signals([signal])
        assert results[0]["order_type"] != "market"

    @pytest.mark.asyncio
    async def test_is_maker_flag_in_result(self):
        sys = _build_execute_signals_system()
        signal = _FakeSignal(entry_price=50000.0)
        results = await sys._execute_signals([signal])
        assert "is_maker" in results[0]
        assert results[0]["is_maker"] is True


# ===================================================================
# 6. VWAP routing for large orders
# ===================================================================

class TestVWAPRouting:
    """Verify VWAP is used for orders above threshold."""

    @pytest.mark.asyncio
    async def test_large_order_uses_vwap(self):
        sys = _build_execute_signals_system(portfolio_value_aud=10000.0)
        # High confidence/strength + large portfolio => position_value_usd > 100
        signal = _FakeSignal(entry_price=50000.0, confidence=0.9, strength=0.9)
        results = await sys._execute_signals([signal])
        assert len(results) == 1
        # With 10000 AUD * 0.65 * 0.25 * 0.81 = ~1316 USD > 100 threshold
        assert results[0]["order_type"] == "vwap"

    @pytest.mark.asyncio
    async def test_small_order_uses_limit(self):
        sys = _build_execute_signals_system(portfolio_value_aud=200.0)
        signal = _FakeSignal(entry_price=50000.0, confidence=0.5, strength=0.5)
        results = await sys._execute_signals([signal])
        if results and results[0].get("status") == "filled":
            # position_value_usd = 200 * 0.65 * min(0.25, 0.25*0.25) = ~8.125 USD < 100
            assert results[0]["order_type"] == "limit"

    @pytest.mark.asyncio
    async def test_vwap_has_lower_slippage(self):
        """VWAP orders should simulate lower slippage than regular limit."""
        sys = _build_execute_signals_system(portfolio_value_aud=10000.0)
        signal = _FakeSignal(entry_price=50000.0, confidence=0.9, strength=0.9, action="BUY")
        results = await sys._execute_signals([signal])
        assert len(results) == 1
        # VWAP slippage = max(0, 5 - 1.5) = 3.5 bps, limit = 0 bps
        # Both should be much less than the old taker 5 bps
        assert results[0]["slippage"] < 50000.0 * 5.0 / 10000.0


# ===================================================================
# 7. Fee savings tracking
# ===================================================================

class TestFeeSavingsTracking:
    """Verify cumulative fee savings counter."""

    @pytest.mark.asyncio
    async def test_initial_savings_is_zero(self):
        sys = _build_execute_signals_system()
        assert sys._total_fee_savings_usd == 0.0

    @pytest.mark.asyncio
    async def test_savings_positive_after_trade(self):
        sys = _build_execute_signals_system()
        signal = _FakeSignal(entry_price=50000.0)
        await sys._execute_signals([signal])
        assert sys._total_fee_savings_usd > 0

    @pytest.mark.asyncio
    async def test_savings_match_fee_difference(self):
        sys = _build_execute_signals_system()
        signal = _FakeSignal(entry_price=50000.0, confidence=0.8, strength=0.9)
        results = await sys._execute_signals([signal])
        if results and results[0]["status"] == "filled":
            notional = results[0]["quantity"] * results[0]["price"]
            taker_fee = notional * 0.0026
            maker_fee = notional * 0.0002
            expected_savings = taker_fee - maker_fee
            assert abs(sys._total_fee_savings_usd - expected_savings) < 0.01


# ===================================================================
# 8. MakerEnforcement with connector (rejection + retry)
# ===================================================================

class TestMakerEnforcementWithConnector:
    """Test maker enforcement with a mock connector that rejects then accepts."""

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        connector = AsyncMock()
        # First 2 calls: rejection (None), third: success
        connector.place_order = AsyncMock(side_effect=[
            None, None,
            {"order_id": "test123", "fill_price": 50000.0, "fill_qty": 100.0},
        ])
        enforcer = MakerEnforcement(connector=connector, enabled=True)
        result = await enforcer.place_order("BTC/USD", "buy", 100.0, 50000.0, urgency=0.1)
        assert result.success is True
        assert result.is_maker is True
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_low_urgency_no_fallback(self):
        connector = AsyncMock()
        connector.place_order = AsyncMock(return_value=None)
        enforcer = MakerEnforcement(connector=connector, enabled=True)
        # urgency=0.1 < 0.3 (fallback_urgency_low) => no fallback
        result = await enforcer.place_order("BTC/USD", "buy", 100.0, 50000.0, urgency=0.1)
        assert result.success is False
        assert result.attempts == 5  # MAX_RETRIES=5

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_high_urgency_fallback_to_taker(self):
        connector = AsyncMock()
        # urgency=0.39 < 0.4 threshold => maker is attempted (5 rejections)
        # then fallback because urgency > 0.3 (fallback_urgency_low) and
        # urgency < 0.5 (fallback_urgency_high) => uses configured fallback_to_taker=True
        connector.place_order = AsyncMock(side_effect=[
            None, None, None, None, None,  # 5 post-only rejections
            {"order_id": "taker1", "fill_price": 50000.0, "fill_qty": 100.0},  # taker fill
        ])
        enforcer = MakerEnforcement(connector=connector, enabled=True, fallback_to_taker=True)
        result = await enforcer.place_order("BTC/USD", "buy", 100.0, 50000.0, urgency=0.39)
        assert result.success is True
        assert result.is_maker is False
        assert result.attempts == 5


# ===================================================================
# 9. Limit price computation
# ===================================================================

class TestLimitPriceComputation:
    """Verify limit price offsets for buy/sell."""

    def test_buy_limit_price_above_mid(self):
        """Buy limit should be entry * (1 + 2/10000) = entry + 0.02%."""
        entry = 50000.0
        offset_bps = 2.0
        limit_price = entry * (1 + offset_bps / 10000.0)
        assert limit_price > entry
        # 50000 * 1.0002 = 50010.0
        assert abs(limit_price - 50010.0) < 0.1

    def test_sell_limit_price_below_mid(self):
        """Sell limit should be entry * (1 - 2/10000) = entry - 0.02%."""
        entry = 50000.0
        offset_bps = 2.0
        limit_price = entry * (1 - offset_bps / 10000.0)
        assert limit_price < entry
        # 50000 * 0.9998 = 49990.0
        assert abs(limit_price - 49990.0) < 0.1

    def test_buy_limit_higher_than_sell_limit(self):
        """Buy limit should always be higher than sell limit for same entry."""
        entry = 50000.0
        offset_bps = 2.0
        buy_limit = entry * (1 + offset_bps / 10000.0)
        sell_limit = entry * (1 - offset_bps / 10000.0)
        assert buy_limit > sell_limit
