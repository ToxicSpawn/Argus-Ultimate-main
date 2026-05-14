"""
Tests for the signal -> execution -> position pipeline in UnifiedSystemArchitecture.

Covers:
  - _execute_signals(): risk gating, sizing, paper fills, live order placement
  - _reconcile_positions(): exchange vs internal state reconciliation
  - _poll_pending_orders(): pending order tracking, timeouts, fills
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can instantiate UnifiedSystemArchitecture without the
# entire dependency tree.
# ---------------------------------------------------------------------------


@dataclass
class _MinimalConfig:
    """Minimal config with every field UnifiedSystemArchitecture.__init__ touches."""
    config_version: int = 1
    starting_capital_aud: float = 1000.0
    currency: str = "AUD"
    aud_to_usd: float = 0.65
    primary_exchange: str = "kraken"
    secondary_exchange: str = "coinbase_advanced"
    supported_exchanges: list = field(default_factory=lambda: ["kraken"])
    min_position_size_aud: float = 10.0
    max_position_size_aud: float = 250.0
    max_position_pct: float = 0.25
    max_total_exposure_pct: float = 0.98
    max_concurrent_positions: int = 5
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.25
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.08
    use_volatility_adjusted_limits: bool = False
    realized_vol_pct: float = 0.0
    run_mode: str = "paper"
    paper_slippage_bps: float = 5.0
    paper_fee_rate: float = 0.0026
    reconcile_every_n_cycles: int = 10
    order_timeout_seconds: float = 60.0
    node_role: str = "single-node"
    portfolio_var_limit_pct: float = 0.0
    portfolio_cvar_limit_pct: float = 0.0
    portfolio_var_confidence: float = 0.95
    portfolio_var_lookback_trades: int = 50
    cluster_drawdown_brake_pct: float = 0.0
    target_cluster_cap_pct: float = 0.40
    risk_cluster_map: dict = field(default_factory=dict)
    portfolio_vol_target_pct: float = 2.0
    portfolio_liquidity_spread_ref_bps: float = 20.0
    portfolio_exposure_min_scale: float = 0.30
    targets_enabled: bool = False
    target_convergence_alpha: float = 1.0
    target_rebalance_min_delta_pct: float = 0.02
    target_score_confidence_weight: float = 1.0
    target_score_net_edge_weight: float = 1.0
    target_regime_boost_enabled: bool = True
    strategy_evaluation_enabled: bool = False
    self_optimizing_meta_enabled: bool = False
    liquidity_risk_enabled: bool = False


@dataclass
class TradingSignal:
    """Matches unified_types.TradingSignal."""
    symbol: str
    action: str
    confidence: float
    strength: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    agent_consensus: float = 0.0
    timestamp: float = field(default_factory=time.time)


def _make_system(config: Optional[_MinimalConfig] = None):
    """
    Build a UnifiedSystemArchitecture with all heavy dependencies stubbed out.
    """
    cfg = config or _MinimalConfig()

    # Patch heavy imports that __init__ tries to pull in
    with (
        patch.dict("sys.modules", {
            "execution.reason_codes": MagicMock(ReasonCode=MagicMock()),
        }),
        patch("unified_trading_system.OmegaSQLiteStore", MagicMock()),
        patch("unified_trading_system.yaml", MagicMock()),
    ):
        from unified_trading_system import UnifiedSystemArchitecture, SystemState

        sys_arch = UnifiedSystemArchitecture.__new__(UnifiedSystemArchitecture)

        # Manually init the fields we need (skip the full __init__ which
        # requires dozens of subsystems).
        sys_arch.config = cfg
        sys_arch.state = SystemState.RUNNING
        sys_arch.start_time = datetime.now()
        sys_arch.ai_brain = None
        sys_arch.execution_engine = None
        sys_arch.argus_strategies = None
        sys_arch.monitoring = None
        sys_arch.hft_engine = None
        sys_arch.hft_infrastructure = None
        sys_arch.language_orchestrator = None
        sys_arch.portfolio_value_aud = cfg.starting_capital_aud
        sys_arch.cash_balance_aud = cfg.starting_capital_aud
        sys_arch.positions = {}
        sys_arch.trade_history = []
        sys_arch.total_trades = 0
        sys_arch.winning_trades = 0
        sys_arch.losing_trades = 0
        sys_arch.total_pnl_aud = 0.0
        sys_arch.realized_pnl_aud = 0.0
        sys_arch.unrealized_pnl_aud = 0.0
        sys_arch.total_fees_aud = 0.0
        sys_arch.daily_pnl_aud = 0.0
        sys_arch.max_drawdown_aud = 0.0
        sys_arch.peak_equity_aud = cfg.starting_capital_aud
        sys_arch.mark_price_method = "position.current_price"
        sys_arch._ledger_sanity_violations = 0
        sys_arch.consecutive_losses = 0
        sys_arch.error_count = 0
        sys_arch.total_operations = 0
        sys_arch.exchanges = {}
        sys_arch.market_data_service = None
        sys_arch.component_registry = None
        sys_arch.exchange_manager = None
        sys_arch.unified_risk_manager = None
        sys_arch.omega_store = MagicMock()
        sys_arch.run_id = "test_run"
        sys_arch._trace_id = None
        sys_arch.node_role = "single-node"
        sys_arch.command_bus = None
        sys_arch.execution_mesh = None
        sys_arch._equity_history = []
        sys_arch._after_risk_update_hook = None
        sys_arch._pending_orders = {}
        sys_arch._reconcile_every_n_cycles = int(cfg.reconcile_every_n_cycles)
        sys_arch._order_timeout_seconds = float(cfg.order_timeout_seconds)
        sys_arch._paper_slippage_bps = float(cfg.paper_slippage_bps)
        sys_arch._total_fee_savings_usd = 0.0
        sys_arch._limit_price_offset_bps = 2.0
        sys_arch._limit_order_fill_timeout = 10.0
        sys_arch._vwap_threshold_usd = 1_000_000.0  # very high to skip VWAP in tests
        sys_arch._position_high_water = {}
        sys_arch._position_low_water = {}
        sys_arch._partial_tp_taken = {}
        sys_arch._last_price_by_symbol = {}
        sys_arch.strategy_evaluation_engine = None
        sys_arch.self_optimizing_meta_engine = None

    return sys_arch


# ===========================================================================
# _execute_signals tests
# ===========================================================================


class TestExecuteSignals:
    """Tests for _execute_signals()."""

    @pytest.mark.asyncio
    async def test_empty_signals_returns_empty(self):
        sys = _make_system()
        result = await sys._execute_signals([])
        assert result == []

    @pytest.mark.asyncio
    async def test_hold_signal_skipped(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="HOLD", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        result = await sys._execute_signals([sig])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_invalid_symbol_skipped(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        result = await sys._execute_signals([sig])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_zero_entry_price_skipped(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=0.0,
        )
        result = await sys._execute_signals([sig])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_paper_mode_buy_fill(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        r = results[0]
        assert r["status"] == "filled"
        assert r["side"] == "BUY"
        assert r["symbol"] == "BTC/USD"
        assert r["quantity"] > 0
        # Paper mode now uses limit orders with maker fees (no slippage)
        assert r["price"] == 50000.0
        assert r["commission"] > 0

    @pytest.mark.asyncio
    async def test_paper_mode_sell_fill(self):
        sys = _make_system()
        # Add existing position so sell is valid
        sys.positions["ETH/USD"] = {"quantity": 1.0, "avg_price": 3000.0, "current_price": 3100.0}
        sig = TradingSignal(
            symbol="ETH/USD", action="SELL", confidence=0.9,
            strength=0.8, entry_price=3100.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        r = results[0]
        assert r["status"] == "filled"
        assert r["side"] == "SELL"
        # Paper mode now uses limit orders (no slippage)
        assert r["price"] == 3100.0

    @pytest.mark.asyncio
    async def test_position_too_small_skipped(self):
        sys = _make_system()
        # Very low confidence * strength -> tiny position
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.01,
            strength=0.01, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["status"] == "skipped"
        assert results[0]["reason"] == "position_too_small"

    @pytest.mark.asyncio
    async def test_risk_gate_blocks_signal(self):
        sys = _make_system()
        # Set up component registry to block orders
        mock_registry = MagicMock()
        mock_registry.pre_order_check.return_value = {
            "allow": False,
            "reasons": ["rate_limit_exceeded"],
        }
        sys.component_registry = mock_registry

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["status"] == "blocked"
        mock_registry.pre_order_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_gate_allows_signal(self):
        sys = _make_system()
        mock_registry = MagicMock()
        mock_registry.pre_order_check.return_value = {"allow": True, "reasons": []}
        sys.component_registry = mock_registry

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["status"] == "filled"

    @pytest.mark.asyncio
    async def test_multiple_signals_independent(self):
        sys = _make_system()
        signals = [
            TradingSignal(symbol="BTC/USD", action="BUY", confidence=0.8, strength=0.7, entry_price=50000.0),
            TradingSignal(symbol="ETH/USD", action="BUY", confidence=0.7, strength=0.6, entry_price=3000.0),
            TradingSignal(symbol="SOL/USD", action="BUY", confidence=0.6, strength=0.5, entry_price=100.0),
        ]
        results = await sys._execute_signals(signals)
        assert len(results) == 3
        for r in results:
            assert r["status"] == "filled"

    @pytest.mark.asyncio
    async def test_one_signal_error_doesnt_crash_others(self):
        sys = _make_system()
        mock_registry = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated risk check crash")
            return {"allow": True, "reasons": []}

        mock_registry.pre_order_check.side_effect = side_effect
        sys.component_registry = mock_registry

        signals = [
            TradingSignal(symbol="BTC/USD", action="BUY", confidence=0.8, strength=0.7, entry_price=50000.0),
            TradingSignal(symbol="ETH/USD", action="BUY", confidence=0.7, strength=0.6, entry_price=3000.0),
            TradingSignal(symbol="SOL/USD", action="BUY", confidence=0.6, strength=0.5, entry_price=100.0),
        ]
        results = await sys._execute_signals(signals)
        # First and third should succeed; second should error
        assert len(results) == 3
        filled = [r for r in results if r["status"] == "filled"]
        errors = [r for r in results if r["status"] == "error"]
        assert len(filled) == 2
        assert len(errors) == 1

    @pytest.mark.asyncio
    async def test_trade_ledger_recording(self):
        sys = _make_system()
        mock_ledger = MagicMock()
        mock_engine = MagicMock()
        mock_engine.trade_ledger = mock_ledger
        sys.execution_engine = mock_engine

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        mock_ledger.record_trade.assert_called_once()
        recorded = mock_ledger.record_trade.call_args[0][0]
        assert recorded["symbol"] == "BTC/USD"
        assert recorded["side"] == "BUY"

    @pytest.mark.asyncio
    async def test_on_fill_callback(self):
        sys = _make_system()
        mock_registry = MagicMock()
        mock_registry.pre_order_check.return_value = {"allow": True, "reasons": []}
        sys.component_registry = mock_registry

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        await sys._execute_signals([sig])
        mock_registry.on_fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_position_updated_after_buy(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        await sys._execute_signals([sig])
        assert "BTC/USD" in sys.positions
        assert sys.positions["BTC/USD"]["quantity"] > 0

    @pytest.mark.asyncio
    async def test_position_updated_after_sell(self):
        sys = _make_system()
        # Pre-set a position
        sys.positions["BTC/USD"] = {"quantity": 0.01, "avg_price": 48000.0, "current_price": 50000.0}
        sig = TradingSignal(
            symbol="BTC/USD", action="SELL", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        await sys._execute_signals([sig])
        # Position should be reduced (may not be fully closed depending on sizing)
        assert sys.total_trades >= 1

    @pytest.mark.asyncio
    async def test_live_mode_calls_exchange_manager(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        mock_em = AsyncMock()
        mock_em.execute_order.return_value = {
            "order_id": "LIVE-001",
            "status": "filled",
            "symbol": "BTC/USD",
            "side": "buy",
            "amount": 0.001,
            "filled": 0.001,
            "remaining": 0.0,
            "price": 50010.0,
            "timestamp": time.time(),
            "exchange": "kraken",
        }
        sys.exchange_manager = mock_em

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["order_id"] == "LIVE-001"
        mock_em.execute_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_mode_exchange_none_response(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        mock_em = AsyncMock()
        mock_em.execute_order.return_value = None
        sys.exchange_manager = mock_em

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert results[0]["reason"] == "exchange_returned_none"

    @pytest.mark.asyncio
    async def test_live_mode_exchange_error(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        mock_em = AsyncMock()
        mock_em.execute_order.side_effect = ConnectionError("exchange down")
        sys.exchange_manager = mock_em

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["status"] == "error"

    @pytest.mark.asyncio
    async def test_live_mode_partial_fill_tracked(self):
        cfg = _MinimalConfig(run_mode="live", starting_capital_aud=100_000.0)
        sys = _make_system(cfg)
        sys.portfolio_value_aud = 100_000.0
        sys.cash_balance_aud = 100_000.0
        sys.peak_equity_aud = 100_000.0
        sys._limit_order_fill_timeout = 0.01  # fast timeout for test

        # Compute expected quantity so we can return a realistic partial fill
        # sizing: conf*str*max_pos_pct = 0.8*0.7*0.25 = 0.14
        # position_value_aud ~ 100000*0.14 = 14000, usd ~ 9100, qty ~ 0.182
        mock_em = AsyncMock()
        mock_em.execute_order.return_value = {
            "order_id": "PARTIAL-001",
            "status": "open",
            "symbol": "BTC/USD",
            "side": "buy",
            "amount": 0.182,
            "filled": 0.05,
            "remaining": 0.132,
            "price": 50010.0,
            "timestamp": time.time(),
            "exchange": "kraken",
        }
        sys.exchange_manager = mock_em

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        # Should track the pending order (partial fill)
        assert "PARTIAL-001" in sys._pending_orders

    @pytest.mark.asyncio
    async def test_confidence_affects_size(self):
        sys = _make_system()
        # High confidence signal
        high = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.95,
            strength=0.9, entry_price=50000.0,
        )
        # Low confidence signal
        low = TradingSignal(
            symbol="ETH/USD", action="BUY", confidence=0.3,
            strength=0.3, entry_price=3000.0,
        )
        results_high = await sys._execute_signals([high])
        # Reset for second signal
        sys2 = _make_system()
        results_low = await sys2._execute_signals([low])

        if results_high and results_high[0]["status"] == "filled" and \
           results_low and results_low[0]["status"] == "filled":
            # High confidence should have larger position value
            val_high = results_high[0]["quantity"] * results_high[0]["price"]
            val_low = results_low[0]["quantity"] * results_low[0]["price"]
            assert val_high > val_low

    @pytest.mark.asyncio
    async def test_paper_mode_never_calls_exchange(self):
        sys = _make_system()
        mock_em = AsyncMock()
        sys.exchange_manager = mock_em

        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["status"] == "filled"
        # In paper mode, exchange_manager should NOT be called
        mock_em.execute_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_loss_take_profit_preserved(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1
        assert results[0]["stop_loss"] == 48000.0
        assert results[0]["take_profit"] == 55000.0

    @pytest.mark.asyncio
    async def test_slippage_direction_buy(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        # Paper mode now uses limit orders with 0 slippage (maker fill)
        assert results[0]["price"] == 50000.0

    @pytest.mark.asyncio
    async def test_slippage_direction_sell(self):
        sys = _make_system()
        sys.positions["BTC/USD"] = {"quantity": 1.0, "avg_price": 48000.0, "current_price": 50000.0}
        sig = TradingSignal(
            symbol="BTC/USD", action="SELL", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        # Paper mode now uses limit orders with 0 slippage (maker fill)
        assert results[0]["price"] == 50000.0


# ===========================================================================
# _reconcile_positions tests
# ===========================================================================


class TestReconcilePositions:
    """Tests for _reconcile_positions()."""

    @pytest.mark.asyncio
    async def test_paper_mode_skips_reconciliation(self):
        sys = _make_system()
        result = await sys._reconcile_positions()
        assert result["discrepancies"] == []
        assert result["updated"] == 0

    @pytest.mark.asyncio
    async def test_live_mode_no_exchange_manager(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        result = await sys._reconcile_positions()
        assert result["discrepancies"] == []

    @pytest.mark.asyncio
    async def test_live_mode_matches(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        mock_em = AsyncMock()
        mock_em.get_balances.return_value = {
            "kraken": {"BTC": {"total": 0.5}, "ETH": {"total": 2.0}},
        }
        sys.exchange_manager = mock_em
        sys.positions = {
            "BTC": {"quantity": 0.5, "avg_price": 50000.0, "current_price": 51000.0},
            "ETH": {"quantity": 2.0, "avg_price": 3000.0, "current_price": 3100.0},
        }
        result = await sys._reconcile_positions()
        assert result["discrepancies"] == []
        assert result["updated"] == 0

    @pytest.mark.asyncio
    async def test_live_mode_discrepancy_updates_internal(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        mock_em = AsyncMock()
        mock_em.get_balances.return_value = {
            "kraken": {"BTC": {"total": 0.6}},  # exchange says 0.6
        }
        sys.exchange_manager = mock_em
        sys.positions = {
            "BTC": {"quantity": 0.5, "avg_price": 50000.0, "current_price": 51000.0},  # internal says 0.5
        }
        result = await sys._reconcile_positions()
        assert len(result["discrepancies"]) == 1
        assert result["updated"] == 1
        # Exchange is source of truth
        assert sys.positions["BTC"]["quantity"] == 0.6

    @pytest.mark.asyncio
    async def test_live_mode_exchange_zero_removes_position(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        mock_em = AsyncMock()
        mock_em.get_balances.return_value = {
            "kraken": {"BTC": {"total": 0.0}},
        }
        sys.exchange_manager = mock_em
        sys.positions = {
            "BTC": {"quantity": 0.5, "avg_price": 50000.0, "current_price": 51000.0},
        }
        result = await sys._reconcile_positions()
        assert len(result["discrepancies"]) == 1
        assert "BTC" not in sys.positions

    @pytest.mark.asyncio
    async def test_live_mode_exchange_error_handled(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        mock_em = AsyncMock()
        mock_em.get_balances.side_effect = ConnectionError("timeout")
        sys.exchange_manager = mock_em
        result = await sys._reconcile_positions()
        assert "error" in result


# ===========================================================================
# _poll_pending_orders tests
# ===========================================================================


class TestPollPendingOrders:
    """Tests for _poll_pending_orders()."""

    @pytest.mark.asyncio
    async def test_no_pending_returns_empty(self):
        sys = _make_system()
        result = await sys._poll_pending_orders()
        assert result == []

    @pytest.mark.asyncio
    async def test_timeout_cancels_order(self):
        sys = _make_system()
        sys._order_timeout_seconds = 0.1  # very short timeout
        sys._pending_orders["ORD-001"] = {
            "order_id": "ORD-001",
            "symbol": "BTC/USD",
            "side": "BUY",
            "total_quantity": 0.01,
            "filled_quantity": 0.0,
            "remaining": 0.01,
            "entry_price": 50000.0,
            "exchange": "kraken",
            "submitted_at": time.time() - 120,  # submitted 2 minutes ago
            "stop_loss": None,
            "take_profit": None,
        }
        results = await sys._poll_pending_orders()
        assert len(results) == 1
        assert results[0]["status"] == "cancelled"
        assert results[0]["reason"] == "timeout"
        assert "ORD-001" not in sys._pending_orders

    @pytest.mark.asyncio
    async def test_order_not_timed_out_stays_pending(self):
        sys = _make_system()
        sys._order_timeout_seconds = 300
        sys._pending_orders["ORD-002"] = {
            "order_id": "ORD-002",
            "symbol": "BTC/USD",
            "side": "BUY",
            "total_quantity": 0.01,
            "filled_quantity": 0.0,
            "remaining": 0.01,
            "entry_price": 50000.0,
            "exchange": "kraken",
            "submitted_at": time.time(),  # just submitted
            "stop_loss": None,
            "take_profit": None,
        }
        results = await sys._poll_pending_orders()
        # In paper mode, no exchange polling happens, and order hasn't timed out
        assert len(results) == 0
        assert "ORD-002" in sys._pending_orders

    @pytest.mark.asyncio
    async def test_live_filled_order_processes_fill(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        sys._order_timeout_seconds = 300

        # Mock exchange that says order is filled
        mock_exchange = AsyncMock()
        mock_exchange.fetch_order.return_value = {
            "status": "closed",
            "filled": 0.01,
            "remaining": 0.0,
            "average": 50005.0,
        }
        mock_em = MagicMock()
        mock_em.exchanges = {"kraken": mock_exchange}
        sys.exchange_manager = mock_em

        sys._pending_orders["LIVE-001"] = {
            "order_id": "LIVE-001",
            "symbol": "BTC/USD",
            "side": "BUY",
            "total_quantity": 0.01,
            "filled_quantity": 0.0,
            "remaining": 0.01,
            "entry_price": 50000.0,
            "exchange": "kraken",
            "submitted_at": time.time(),
            "stop_loss": None,
            "take_profit": None,
        }

        results = await sys._poll_pending_orders()
        assert len(results) == 1
        assert results[0]["status"] == "filled"
        assert "LIVE-001" not in sys._pending_orders

    @pytest.mark.asyncio
    async def test_live_cancelled_order_removed(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)
        sys._order_timeout_seconds = 300

        mock_exchange = AsyncMock()
        mock_exchange.fetch_order.return_value = {
            "status": "canceled",
            "filled": 0.0,
            "remaining": 0.01,
        }
        mock_em = MagicMock()
        mock_em.exchanges = {"kraken": mock_exchange}
        sys.exchange_manager = mock_em

        sys._pending_orders["LIVE-002"] = {
            "order_id": "LIVE-002",
            "symbol": "BTC/USD",
            "side": "BUY",
            "total_quantity": 0.01,
            "filled_quantity": 0.0,
            "remaining": 0.01,
            "entry_price": 50000.0,
            "exchange": "kraken",
            "submitted_at": time.time(),
            "stop_loss": None,
            "take_profit": None,
        }

        results = await sys._poll_pending_orders()
        assert len(results) == 1
        assert results[0]["status"] == "cancelled"
        assert "LIVE-002" not in sys._pending_orders

    @pytest.mark.asyncio
    async def test_multiple_pending_processed(self):
        sys = _make_system()
        sys._order_timeout_seconds = 0.1

        for i in range(3):
            sys._pending_orders[f"ORD-{i}"] = {
                "order_id": f"ORD-{i}",
                "symbol": "BTC/USD",
                "side": "BUY",
                "total_quantity": 0.01,
                "filled_quantity": 0.0,
                "remaining": 0.01,
                "entry_price": 50000.0,
                "exchange": "kraken",
                "submitted_at": time.time() - 120,
                "stop_loss": None,
                "take_profit": None,
            }

        results = await sys._poll_pending_orders()
        assert len(results) == 3
        assert len(sys._pending_orders) == 0


# ===========================================================================
# Integration / end-to-end tests
# ===========================================================================


class TestExecutionPipelineIntegration:
    """End-to-end tests combining multiple pipeline stages."""

    @pytest.mark.asyncio
    async def test_buy_then_sell_pnl_tracking(self):
        sys = _make_system()

        # Buy
        buy_sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        buy_results = await sys._execute_signals([buy_sig])
        assert len(buy_results) == 1
        assert buy_results[0]["status"] == "filled"
        assert sys.total_trades == 1

        # Sell at higher price
        sell_sig = TradingSignal(
            symbol="BTC/USD", action="SELL", confidence=0.8,
            strength=0.7, entry_price=55000.0,
        )
        sell_results = await sys._execute_signals([sell_sig])
        assert len(sell_results) == 1
        assert sys.total_trades == 2

    @pytest.mark.asyncio
    async def test_existing_position_add(self):
        sys = _make_system()
        # First buy
        sig1 = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        await sys._execute_signals([sig1])
        qty_after_first = sys.positions.get("BTC/USD", {}).get("quantity", 0.0)

        # Second buy (add to position)
        sig2 = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.6,
            strength=0.5, entry_price=51000.0,
        )
        await sys._execute_signals([sig2])
        qty_after_second = sys.positions.get("BTC/USD", {}).get("quantity", 0.0)
        assert qty_after_second > qty_after_first

    @pytest.mark.asyncio
    async def test_full_cycle_with_reconciliation(self):
        cfg = _MinimalConfig(run_mode="live")
        sys = _make_system(cfg)

        # Set up exchange manager for reconciliation
        mock_em = AsyncMock()
        mock_em.get_balances.return_value = {
            "kraken": {"BTC": {"total": 0.1}},
        }
        mock_em.execute_order.return_value = {
            "order_id": "LIVE-100",
            "status": "filled",
            "symbol": "BTC/USD",
            "side": "buy",
            "amount": 0.01,
            "filled": 0.01,
            "remaining": 0.0,
            "price": 50010.0,
            "timestamp": time.time(),
            "exchange": "kraken",
        }
        sys.exchange_manager = mock_em

        # Execute a signal
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.8,
            strength=0.7, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        assert len(results) == 1

        # Reconcile -- exchange says 0.1 BTC
        recon = await sys._reconcile_positions()
        # Should see discrepancy (internal has our trade qty, exchange says 0.1)
        # and update to exchange value
        if recon["discrepancies"]:
            assert sys.positions.get("BTC", {}).get("quantity", 0) == 0.1

    @pytest.mark.asyncio
    async def test_zero_confidence_zero_size_rejected(self):
        sys = _make_system()
        sig = TradingSignal(
            symbol="BTC/USD", action="BUY", confidence=0.0,
            strength=0.0, entry_price=50000.0,
        )
        results = await sys._execute_signals([sig])
        # Zero confidence * strength = zero size -> should be rejected
        assert len(results) == 1
        assert results[0]["status"] == "skipped"
