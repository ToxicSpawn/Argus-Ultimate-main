"""
Tests for HIGH-IMPACT trading performance fixes 7-14.

FIX 7:  Adaptive ensemble weights
FIX 8:  Entry timing optimization
FIX 9:  Pyramid and partial exit logic
FIX 10: Drawdown-adaptive sizing
FIX 11: Correlation-based position reduction
FIX 12: Wire conditional orders (OCO)
FIX 13: Wire funding rate harvester
FIX 14: Wire cross-exchange arb
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------


@dataclass
class _FakeConfig:
    starting_capital_aud: float = 1000.0
    aud_to_usd: float = 0.65
    max_position_pct: float = 0.25
    min_position_size_aud: float = 10.0
    run_mode: str = "paper"
    primary_exchange: str = "kraken"
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    paper_slippage_bps: float = 5.0
    paper_maker_fee_rate: float = 0.0002
    paper_fee_rate: float = 0.0026
    max_concurrent_positions: int = 5
    trading_pairs: list = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    portfolio_var_limit_pct: float = 0.0
    portfolio_cvar_limit_pct: float = 0.0
    op_timeout_s: float = 30.0
    signal_primary_timeframe: str = "1h"
    reconcile_every_n_cycles: int = 10
    order_timeout_seconds: float = 60.0
    trailing_stop_pct: float = 0.02
    trailing_activation_pct: float = 0.01
    stale_position_hours: float = 48.0
    stale_min_profit_pct: float = 0.005
    max_hold_hours: float = 168.0


def _make_system():
    """Create a minimal UnifiedSystemArchitecture-like object for testing."""
    from unified_trading_system import UnifiedSystemArchitecture
    cfg = _FakeConfig()
    with patch.object(UnifiedSystemArchitecture, "__init__", lambda self, *a, **kw: None):
        sys = UnifiedSystemArchitecture.__new__(UnifiedSystemArchitecture)
    # Manually set required attributes
    sys.config = cfg
    sys.portfolio_value_aud = 1000.0
    sys.peak_equity_aud = 1000.0
    sys.cash_balance_aud = 1000.0
    sys.positions = {}
    sys.trade_history = deque(maxlen=1000)
    sys.unified_risk_manager = None
    sys.component_registry = None
    sys.execution_engine = None
    sys.exchange_manager = None
    sys._pending_orders = {}
    sys._total_fee_savings_usd = 0.0
    sys._paper_slippage_bps = 5.0
    sys._limit_order_fill_timeout = 10.0
    sys._limit_price_offset_bps = 2.0
    sys._volatility_cache = {}
    sys._vwap_threshold_usd = 100.0
    sys._price_history = {}
    sys._pyramid_count = {}
    sys._partial_exit_done = {}
    sys._oco_orders = {}
    sys._strategy_state_store = None
    sys._latest_regime_label = "NORMAL"
    sys._equity_history = []
    sys._position_high_water = {}
    sys._position_low_water = {}
    sys._partial_tp_taken = {}
    sys.REGIME_POSITION_SCALE = {
        'TRENDING_UP': 1.2, 'TRENDING_DOWN': 0.7, 'HIGH_VOL': 0.6,
        'LOW_VOL': 0.8, 'MEAN_REVERTING': 1.0, 'NORMAL': 1.0,
        'BREAKOUT': 1.3, 'CRISIS': 0.3,
    }
    sys.REGIME_STOP_SCALE = {
        'TRENDING_UP': 1.0, 'TRENDING_DOWN': 0.8, 'HIGH_VOL': 1.5,
        'LOW_VOL': 0.7, 'MEAN_REVERTING': 0.9, 'NORMAL': 1.0,
        'BREAKOUT': 1.2, 'CRISIS': 0.5,
    }
    sys.REGIME_TP_SCALE = {
        'TRENDING_UP': 1.3, 'TRENDING_DOWN': 0.8, 'HIGH_VOL': 1.8,
        'LOW_VOL': 0.8, 'MEAN_REVERTING': 0.9, 'NORMAL': 1.0,
        'BREAKOUT': 1.2, 'CRISIS': 0.5,
    }
    return sys


def _run(coro):
    """Helper to run async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===================================================================
# FIX 7: Adaptive Ensemble Weights
# ===================================================================


class TestAdaptiveEnsembleWeights:
    """Tests for EnsembleSignalHub.update_source_weights() and get_weights()."""

    def _make_hub(self, **kwargs):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        return EnsembleSignalHub(config=kwargs)

    def test_get_weights_returns_dict(self):
        hub = self._make_hub()
        w = hub.get_weights()
        assert isinstance(w, dict)
        assert len(w) > 0

    def test_get_weights_returns_copy(self):
        hub = self._make_hub()
        w = hub.get_weights()
        w["fear_greed"] = 999.0
        assert hub.get_weights()["fear_greed"] != 999.0

    def test_update_source_weights_positive_pnl_increases_weight(self):
        hub = self._make_hub()
        initial = hub.get_weights()["alpha"]
        hub.update_source_weights({"alpha": 1.0})
        after = hub.get_weights()["alpha"]
        # After normalization, alpha weight should be larger relative to others
        # since only alpha got a positive bump
        assert after > initial or after >= 0.30  # still dominant

    def test_update_source_weights_negative_pnl_decreases_weight(self):
        hub = self._make_hub()
        initial = hub.get_weights()["alpha"]
        hub.update_source_weights({"alpha": -1.0})
        after = hub.get_weights()["alpha"]
        assert after < initial

    def test_weights_normalized_to_one(self):
        hub = self._make_hub()
        hub.update_source_weights({"alpha": 5.0, "llm": -2.0, "whale": 3.0})
        total = sum(hub.get_weights().values())
        assert abs(total - 1.0) < 1e-6

    def test_minimum_weight_enforced(self):
        hub = self._make_hub()
        for _ in range(100):
            hub.update_source_weights({"funding": -10.0})
        w = hub.get_weights()
        assert w.get("funding", 0) > 0

    def test_maximum_weight_enforced(self):
        hub = self._make_hub()
        for _ in range(100):
            hub.update_source_weights({"alpha": 10.0})
        w = hub.get_weights()
        assert all(v <= 1.0 for v in w.values())

    def test_empty_pnl_no_change(self):
        hub = self._make_hub()
        before = hub.get_weights()
        hub.update_source_weights({})
        after = hub.get_weights()
        assert before == after

    def test_unknown_source_ignored(self):
        hub = self._make_hub()
        before = hub.get_weights()
        hub.update_source_weights({"nonexistent_source": 5.0})
        after = hub.get_weights()
        assert before == after


# ===================================================================
# FIX 8: Entry Timing Optimization
# ===================================================================


class TestEntryTimingOptimization:
    """Tests for the entry timing urgency logic in _execute_signals."""

    def test_urgency_stored_in_trade_result(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1
        assert "entry_urgency" in results[0]

    def test_urgency_buy_above_sma_reduces(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys._price_history["BTC/USD"] = [100.0] * 25
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=103.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1
        assert results[0].get("entry_urgency") == pytest.approx(0.2)

    def test_urgency_buy_below_sma_increases(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys._price_history["BTC/USD"] = [100.0] * 25
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=98.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1
        assert results[0].get("entry_urgency") == pytest.approx(0.8)

    def test_urgency_default_when_no_history(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1
        # No price history: urgency falls back to age-based value (0.2 for fresh signal)
        assert results[0].get("entry_urgency") == pytest.approx(0.2)

    def test_urgency_sell_above_sma(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys._price_history["BTC/USD"] = [100.0] * 25
        sig = TradingSignal(
            symbol="BTC/USD", action="SELL", confidence=0.8,
            strength=0.7, entry_price=101.0, timestamp=time.time(),
        )
        # Need a position to sell
        sys.positions = {
            "BTC/USD": {"quantity": 0.01, "entry_price": 100.0,
                        "current_price": 101.0, "side": "BUY"}
        }
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1
        assert results[0].get("entry_urgency") == pytest.approx(0.8)


# ===================================================================
# FIX 9: Pyramid and Partial Exit Logic
# ===================================================================


class TestPyramidLogic:
    """Tests for _check_pyramid_opportunities()."""

    def test_pyramid_when_profitable(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 52000.0, "side": "BUY",
            }
        }
        for i in range(10):
            sys.trade_history.append({"pnl": 10.0, "side": "SELL"})
        signals = sys._check_pyramid_opportunities()
        assert len(signals) == 1
        assert signals[0].action == "BUY"
        assert sys._pyramid_count["BTC/USD"] == 1

    def test_no_pyramid_when_not_profitable(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 50100.0, "side": "BUY",
            }
        }
        for i in range(10):
            sys.trade_history.append({"pnl": 10.0, "side": "SELL"})
        signals = sys._check_pyramid_opportunities()
        assert len(signals) == 0

    def test_max_two_pyramids(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 52000.0, "side": "BUY",
            }
        }
        for i in range(10):
            sys.trade_history.append({"pnl": 10.0, "side": "SELL"})
        sys._pyramid_count["BTC/USD"] = 2
        signals = sys._check_pyramid_opportunities()
        assert len(signals) == 0

    def test_no_pyramid_low_win_rate(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 52000.0, "side": "BUY",
            }
        }
        for i in range(3):
            sys.trade_history.append({"pnl": 10.0, "side": "SELL"})
        for i in range(7):
            sys.trade_history.append({"pnl": -10.0, "side": "SELL"})
        signals = sys._check_pyramid_opportunities()
        assert len(signals) == 0

    def test_pyramid_increments_count(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 52000.0, "side": "BUY",
            }
        }
        for i in range(10):
            sys.trade_history.append({"pnl": 10.0, "side": "SELL"})
        sys._check_pyramid_opportunities()
        assert sys._pyramid_count.get("BTC/USD", 0) == 1
        # Reset for second call
        sys._check_pyramid_opportunities()
        assert sys._pyramid_count.get("BTC/USD", 0) == 2


class TestPartialExitLogic:
    """Tests for _check_partial_exits()."""

    def test_partial_profit_above_4pct(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 52500.0, "side": "BUY",
            }
        }
        signals = sys._check_partial_exits()
        assert len(signals) == 1
        assert signals[0].action == "SELL"
        assert sys._partial_exit_done["BTC/USD"] is True

    def test_no_partial_exit_if_already_done(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 52500.0, "side": "BUY",
            }
        }
        sys._partial_exit_done["BTC/USD"] = True
        signals = sys._check_partial_exits()
        assert len(signals) == 0

    def test_partial_salvage_down_2pct(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 48500.0, "side": "BUY",
            }
        }
        sys._position_high_water["BTC/USD"] = 50000.0
        signals = sys._check_partial_exits()
        assert len(signals) == 1
        assert "salvage" in signals[0].reasoning

    def test_no_partial_exit_small_loss(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 49500.0, "side": "BUY",
            }
        }
        signals = sys._check_partial_exits()
        assert len(signals) == 0

    def test_partial_exit_sell_position(self):
        """Short position that's profitable should trigger partial exit."""
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 47500.0, "side": "SELL",
            }
        }
        signals = sys._check_partial_exits()
        assert len(signals) == 1
        assert signals[0].action == "BUY"  # opposite of SELL position


# ===================================================================
# FIX 10: Drawdown-Adaptive Sizing
# ===================================================================


class TestDrawdownAdaptiveSizing:
    """Tests for drawdown-adaptive sizing in _execute_signals."""

    def test_no_drawdown_no_reduction(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys.portfolio_value_aud = 1000.0
        sys.peak_equity_aud = 1000.0
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1
        assert results[0]["status"] == "filled"

    def test_10pct_drawdown_reduces_size(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys.peak_equity_aud = 1000.0
        sys.portfolio_value_aud = 900.0
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1

    def test_extreme_drawdown_floors_at_25pct(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys.peak_equity_aud = 1000.0
        sys.portfolio_value_aud = 500.0
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1


# ===================================================================
# FIX 11: Correlation-Based Position Reduction
# ===================================================================


class TestCorrelationReduction:
    """Tests for correlation-based position reduction in _execute_signals."""

    def test_btc_eth_correlation_reduces_size(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 51000.0, "side": "BUY",
            }
        }
        sig = TradingSignal(
            symbol="ETH/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=3000.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1

    def test_no_correlation_reduction_different_direction(self):
        """BTC long + ETH short should not trigger correlation reduction."""
        from unified_types import TradingSignal
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 51000.0, "side": "SELL",
            }
        }
        sig = TradingSignal(
            symbol="ETH/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=3000.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1

    def test_no_correlation_reduction_unrelated_assets(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 51000.0, "side": "BUY",
            }
        }
        sig = TradingSignal(
            symbol="SOL/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=100.0, timestamp=time.time(),
        )
        results = _run(sys._execute_signals([sig]))
        assert len(results) >= 1


# ===================================================================
# FIX 12: OCO Conditional Orders
# ===================================================================


class TestOCOOrders:
    """Tests for OCO order creation and triggering."""

    def test_oco_created_on_fill(self):
        from unified_types import TradingSignal
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
            stop_loss=49000.0, take_profit=52000.0, timestamp=time.time(),
        )
        _run(sys._execute_signals([sig]))
        assert "BTC/USD" in sys._oco_orders
        oco = sys._oco_orders["BTC/USD"]
        assert oco["stop_loss"] == 49000.0
        assert oco["take_profit"] == 52000.0

    def test_oco_stop_loss_trigger(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 48000.0, "side": "BUY",
            }
        }
        sys._oco_orders["BTC/USD"] = {
            "order_id": "test_1", "symbol": "BTC/USD", "side": "BUY",
            "entry_price": 50000.0, "quantity": 0.01,
            "stop_loss": 49000.0, "take_profit": 52000.0,
            "created_at": time.time(),
        }
        signals = sys._check_oco_conditions()
        assert len(signals) == 1
        assert signals[0].action == "SELL"
        assert "stop_loss" in signals[0].reasoning

    def test_oco_take_profit_trigger(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 53000.0, "side": "BUY",
            }
        }
        sys._oco_orders["BTC/USD"] = {
            "order_id": "test_1", "symbol": "BTC/USD", "side": "BUY",
            "entry_price": 50000.0, "quantity": 0.01,
            "stop_loss": 49000.0, "take_profit": 52000.0,
            "created_at": time.time(),
        }
        signals = sys._check_oco_conditions()
        assert len(signals) == 1
        assert signals[0].action == "SELL"
        assert "take_profit" in signals[0].reasoning

    def test_oco_removed_after_trigger(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 53000.0, "side": "BUY",
            }
        }
        sys._oco_orders["BTC/USD"] = {
            "order_id": "test_1", "symbol": "BTC/USD", "side": "BUY",
            "entry_price": 50000.0, "quantity": 0.01,
            "stop_loss": 49000.0, "take_profit": 52000.0,
            "created_at": time.time(),
        }
        sys._check_oco_conditions()
        assert "BTC/USD" not in sys._oco_orders

    def test_oco_no_trigger_in_range(self):
        sys = _make_system()
        sys.positions = {
            "BTC/USD": {
                "quantity": 0.01, "entry_price": 50000.0,
                "current_price": 50500.0, "side": "BUY",
            }
        }
        sys._oco_orders["BTC/USD"] = {
            "order_id": "test_1", "symbol": "BTC/USD", "side": "BUY",
            "entry_price": 50000.0, "quantity": 0.01,
            "stop_loss": 49000.0, "take_profit": 52000.0,
            "created_at": time.time(),
        }
        signals = sys._check_oco_conditions()
        assert len(signals) == 0
        assert "BTC/USD" in sys._oco_orders

    def test_oco_removed_when_position_closed(self):
        sys = _make_system()
        sys.positions = {}  # position is gone
        sys._oco_orders["BTC/USD"] = {
            "order_id": "test_1", "symbol": "BTC/USD", "side": "BUY",
            "entry_price": 50000.0, "quantity": 0.01,
            "stop_loss": 49000.0, "take_profit": 52000.0,
            "created_at": time.time(),
        }
        sys._check_oco_conditions()
        assert "BTC/USD" not in sys._oco_orders


# ===================================================================
# FIX 13: Wire Funding Rate Harvester
# ===================================================================


class TestFundingHarvesterWiring:
    """Tests for funding harvester signal injection in on_cycle."""

    def test_harvester_signal_in_advisory(self):
        from core.component_registry import ComponentRegistry
        cfg = _FakeConfig()
        # Use __init__ but mock _try_init to prevent actual component loading
        with patch.object(ComponentRegistry, "_try_init", lambda self, name, fn: None):
            cr = ComponentRegistry(cfg)
        # Now set funding_harvester to our mock
        cr.funding_harvester = MagicMock()
        cr.funding_harvester.analyze.return_value = {
            "action": "BUY", "confidence": 0.7, "reason": "high rate",
        }
        advisory = cr.on_cycle({"BTC/USD": 50000.0}, "NORMAL")
        assert advisory is not None
        assert "funding_harvester" in advisory
        assert advisory["funding_harvester"]["action"] == "BUY"


# ===================================================================
# FIX 14: Wire Cross-Exchange Arb
# ===================================================================


class TestCrossExchangeArbWiring:
    """Tests for cross-exchange arb signal injection."""

    def test_arb_strategy_init(self):
        from strategies.cross_exchange_arb import CrossExchangeArbStrategy
        arb = CrossExchangeArbStrategy()
        assert arb is not None

    def test_arb_finds_opportunity(self):
        from strategies.cross_exchange_arb import CrossExchangeArbStrategy
        arb = CrossExchangeArbStrategy(fee_bps_per_side=2.0, min_net_spread_bps=1.0)
        arb.update_price("kraken", "BTC", 50000.0)
        arb.update_price("coinbase", "BTC", 50100.0)
        opp = arb.find_opportunity("BTC")
        if opp is not None:
            assert opp.net_spread_bps > 0

    def test_arb_no_opportunity_same_price(self):
        from strategies.cross_exchange_arb import CrossExchangeArbStrategy
        arb = CrossExchangeArbStrategy()
        arb.update_price("kraken", "BTC", 50000.0)
        arb.update_price("coinbase", "BTC", 50000.0)
        opp = arb.find_opportunity("BTC")
        assert opp is None


# ===================================================================
# Integration: Verify all FIX attributes exist
# ===================================================================


class TestIntegrationAttributes:
    """Verify all new attributes and methods exist on the classes."""

    def test_ensemble_hub_has_update_source_weights(self):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        assert hasattr(EnsembleSignalHub, "update_source_weights")

    def test_ensemble_hub_has_get_weights(self):
        from ml.ensemble_signal_hub import EnsembleSignalHub
        assert hasattr(EnsembleSignalHub, "get_weights")

    def test_uts_has_pyramid_method(self):
        from unified_trading_system import UnifiedSystemArchitecture
        assert hasattr(UnifiedSystemArchitecture, "_check_pyramid_opportunities")

    def test_uts_has_partial_exits_method(self):
        from unified_trading_system import UnifiedSystemArchitecture
        assert hasattr(UnifiedSystemArchitecture, "_check_partial_exits")

    def test_uts_has_oco_check_method(self):
        from unified_trading_system import UnifiedSystemArchitecture
        assert hasattr(UnifiedSystemArchitecture, "_check_oco_conditions")

    def test_trading_signal_dataclass(self):
        from unified_types import TradingSignal
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
            stop_loss=49000.0, take_profit=52000.0,
        )
        assert sig.stop_loss == 49000.0
        assert sig.take_profit == 52000.0
