"""
Tests for ARGUS structural improvements:
- PortfolioManager (thread-safe state management)
- Protocol interfaces
- AsyncWriteQueue (batched DB writes)
- PerformanceScorecard (auto-disable underperformers)
- PositionReconciler (exchange drift detection)
- DomainConfig (validated config dataclasses)
"""

import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
#  PortfolioManager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioManager(unittest.TestCase):
    """Thread-safe portfolio state management."""

    def setUp(self):
        from core.portfolio_manager import PortfolioManager
        self.pm = PortfolioManager(starting_capital_aud=1000.0, aud_to_usd=0.65)

    def test_initial_state(self):
        snap = self.pm.snapshot()
        self.assertAlmostEqual(snap.portfolio_value_aud, 1000.0)
        self.assertAlmostEqual(snap.cash_balance_aud, 1000.0)
        self.assertEqual(snap.total_trades, 0)
        self.assertEqual(snap.open_position_count, 0)

    def test_record_buy(self):
        result = self.pm.record_buy("BTC/USD", 0.01, 50000.0, commission_usd=1.0)
        self.assertEqual(result["symbol"], "BTC/USD")
        self.assertEqual(result["side"], "buy")
        snap = self.pm.snapshot()
        self.assertEqual(snap.total_trades, 1)
        self.assertEqual(snap.open_position_count, 1)
        self.assertAlmostEqual(snap.positions["BTC/USD"]["quantity"], 0.01)

    def test_record_sell_with_pnl(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        result = self.pm.record_sell("BTC/USD", 0.01, 55000.0)
        self.assertGreater(result["pnl"], 0)  # Profitable
        snap = self.pm.snapshot()
        self.assertEqual(snap.total_trades, 2)
        self.assertEqual(snap.winning_trades, 1)
        self.assertEqual(snap.open_position_count, 0)

    def test_record_sell_loss(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        result = self.pm.record_sell("BTC/USD", 0.01, 45000.0)
        self.assertLess(result["pnl"], 0)  # Loss
        snap = self.pm.snapshot()
        self.assertEqual(snap.losing_trades, 1)

    def test_sell_no_position(self):
        result = self.pm.record_sell("BTC/USD", 0.01, 50000.0)
        self.assertEqual(result.get("status"), "no_position")

    def test_update_price(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        self.pm.update_price("BTC/USD", 55000.0)
        pos = self.pm.get_position("BTC/USD")
        self.assertAlmostEqual(pos["current_price"], 55000.0)

    def test_update_portfolio_value(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        self.pm.update_price("BTC/USD", 60000.0)
        val = self.pm.update_portfolio_value()
        self.assertGreater(val, 0)

    def test_peak_and_drawdown(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        self.pm.update_price("BTC/USD", 60000.0)
        self.pm.update_portfolio_value()  # Peak
        self.pm.update_price("BTC/USD", 40000.0)
        self.pm.update_portfolio_value()  # Drawdown
        snap = self.pm.snapshot()
        self.assertGreater(snap.max_drawdown_aud, 0)

    def test_reset_daily(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        self.pm.record_sell("BTC/USD", 0.01, 55000.0)
        snap_before = self.pm.snapshot()
        self.assertNotEqual(snap_before.daily_pnl_aud, 0)
        self.pm.reset_daily()
        snap_after = self.pm.snapshot()
        self.assertAlmostEqual(snap_after.daily_pnl_aud, 0.0)

    def test_reconcile_position(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        self.pm.reconcile_position("BTC/USD", 0.02, 51000.0)
        pos = self.pm.get_position("BTC/USD")
        self.assertAlmostEqual(pos["quantity"], 0.02)

    def test_reconcile_zero_removes(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        self.pm.reconcile_position("BTC/USD", 0.0, 50000.0)
        pos = self.pm.get_position("BTC/USD")
        self.assertIsNone(pos)

    def test_snapshot_is_immutable(self):
        snap = self.pm.snapshot()
        snap.positions["FAKE"] = {"quantity": 999}
        # Original should not be affected
        snap2 = self.pm.snapshot()
        self.assertNotIn("FAKE", snap2.positions)

    def test_thread_safety(self):
        """Multiple threads buying/selling concurrently should not corrupt state."""
        errors = []

        def buy_thread():
            try:
                for _ in range(50):
                    self.pm.record_buy("BTC/USD", 0.001, 50000.0)
            except Exception as e:
                errors.append(e)

        def sell_thread():
            try:
                for _ in range(50):
                    self.pm.record_sell("BTC/USD", 0.001, 50000.0)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=buy_thread)
        t2 = threading.Thread(target=sell_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")
        snap = self.pm.snapshot()
        self.assertEqual(snap.total_trades, 100)

    def test_win_rate(self):
        self.pm.record_buy("BTC/USD", 0.01, 50000.0)
        self.pm.record_sell("BTC/USD", 0.01, 55000.0)  # win
        self.pm.record_buy("ETH/USD", 1.0, 3000.0)
        self.pm.record_sell("ETH/USD", 1.0, 2500.0)  # loss
        snap = self.pm.snapshot()
        self.assertAlmostEqual(snap.win_rate, 0.25)  # 1 win out of 4 total trades


# ═══════════════════════════════════════════════════════════════════════════════
#  Protocol Interface Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProtocols(unittest.TestCase):
    """Verify Protocol interfaces are importable and usable."""

    def test_import_all_protocols(self):
        from core.protocols import (
            FillTracker, RiskGate, SignalSource, PositionTracker,
            ComplianceRecorder, AlertChannel, ModelManager, EnsembleHub,
            StrategyRouter, ExchangeConnector, StateStore, WriteQueue,
        )
        # All imported successfully
        self.assertTrue(True)

    def test_fill_tracker_protocol(self):
        from core.protocols import FillTracker
        mock = MagicMock()
        mock.record_fill = MagicMock()
        self.assertTrue(isinstance(mock, FillTracker))

    def test_strategy_router_protocol(self):
        from core.protocols import StrategyRouter
        mock = MagicMock()
        mock.get_active_strategies = MagicMock(return_value=["momentum"])
        mock.get_strategy_stats = MagicMock(return_value={})
        mock.disable = MagicMock()
        mock.enable = MagicMock()
        self.assertTrue(isinstance(mock, StrategyRouter))


# ═══════════════════════════════════════════════════════════════════════════════
#  AsyncWriteQueue Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncWriteQueue(unittest.TestCase):
    """Batched async write queue for SQLite."""

    def test_enqueue_increases_pending(self):
        from core.async_write_queue import AsyncWriteQueue
        q = AsyncWriteQueue(":memory:")
        self.assertEqual(q.pending, 0)
        q.enqueue("trades", {"symbol": "BTC/USD", "side": "buy"})
        self.assertEqual(q.pending, 1)
        q.enqueue("trades", {"symbol": "ETH/USD", "side": "sell"})
        self.assertEqual(q.pending, 2)

    def test_get_stats(self):
        from core.async_write_queue import AsyncWriteQueue
        q = AsyncWriteQueue(":memory:")
        q.enqueue("trades", {"symbol": "BTC/USD"})
        stats = q.get_stats()
        self.assertEqual(stats["total_enqueued"], 1)
        self.assertEqual(stats["pending"], 1)

    def test_thread_safe_enqueue(self):
        from core.async_write_queue import AsyncWriteQueue
        q = AsyncWriteQueue(":memory:")

        def enqueue_thread():
            for i in range(100):
                q.enqueue("trades", {"id": i})

        threads = [threading.Thread(target=enqueue_thread) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(q.pending, 400)


# ═══════════════════════════════════════════════════════════════════════════════
#  PerformanceScorecard Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceScorecard(unittest.TestCase):
    """Rolling performance scorecard with auto-disable."""

    def setUp(self):
        from core.performance_scorecard import PerformanceScorecard
        self.sc = PerformanceScorecard(
            disable_sharpe=-0.5,
            disable_consec_losses=5,
            min_trades_for_eval=10,
        )

    def test_record_trade(self):
        self.sc.record_trade("momentum", 10.0)
        result = self.sc.evaluate()
        self.assertIn("momentum", result["strategy_metrics"])
        self.assertEqual(result["strategy_metrics"]["momentum"]["trades"], 1)

    def test_consecutive_losses_disable(self):
        for _ in range(6):
            self.sc.record_trade("bad_strategy", -5.0)
        result = self.sc.evaluate()
        self.assertIn("bad_strategy", result["disabled_strategies"])

    def test_good_strategy_not_disabled(self):
        for i in range(20):
            self.sc.record_trade("good_strategy", 5.0 if i % 2 == 0 else -2.0)
        result = self.sc.evaluate()
        self.assertNotIn("good_strategy", result["disabled_strategies"])

    def test_is_disabled(self):
        for _ in range(6):
            self.sc.record_trade("loser", -10.0)
        self.sc.evaluate()
        self.assertTrue(self.sc.is_disabled("loser"))
        self.assertFalse(self.sc.is_disabled("winner"))

    def test_ranking(self):
        for _ in range(10):
            self.sc.record_trade("alpha", 5.0)
            self.sc.record_trade("beta", -2.0)
        ranking = self.sc.get_ranking()
        self.assertEqual(ranking[0]["strategy"], "alpha")

    def test_router_integration(self):
        """Scorecard calls router.disable() when strategy fails."""
        router = MagicMock()
        for _ in range(6):
            self.sc.record_trade("failing", -10.0)
        self.sc.evaluate(strategy_router=router)
        router.disable.assert_called_with("failing")

    def test_no_disable_below_min_trades(self):
        """Don't disable on Sharpe with < min_trades."""
        for _ in range(5):  # Less than min_trades_for_eval=10
            self.sc.record_trade("new_strategy", -100.0)
        self.sc._strategies["new_strategy"].consecutive_losses = 0  # Reset so only Sharpe applies
        result = self.sc.evaluate()
        self.assertNotIn("new_strategy", result["disabled_strategies"])


# ═══════════════════════════════════════════════════════════════════════════════
#  DomainConfig Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDomainConfig(unittest.TestCase):
    """Domain-specific validated config dataclasses."""

    def test_capital_config_defaults_valid(self):
        from core.domain_config import CapitalConfig
        cfg = CapitalConfig()
        errors = cfg.validate()
        self.assertEqual(len(errors), 0)

    def test_capital_config_cross_field(self):
        from core.domain_config import CapitalConfig
        cfg = CapitalConfig(max_position_pct=0.90, max_total_exposure_pct=0.10)
        errors = cfg.validate()
        self.assertTrue(any("max_position_pct" in e for e in errors))

    def test_risk_config_rr_validation(self):
        from core.domain_config import RiskConfig
        cfg = RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.03)
        errors = cfg.validate()
        self.assertTrue(any("stop_loss_pct" in e for e in errors))

    def test_risk_config_defaults_valid(self):
        from core.domain_config import RiskConfig
        cfg = RiskConfig()
        errors = cfg.validate()
        self.assertEqual(len(errors), 0)

    def test_execution_config_defaults_valid(self):
        from core.domain_config import ExecutionConfig
        cfg = ExecutionConfig()
        errors = cfg.validate()
        self.assertEqual(len(errors), 0)

    def test_execution_config_bad_order_type(self):
        from core.domain_config import ExecutionConfig
        cfg = ExecutionConfig(order_type="invalid")
        errors = cfg.validate()
        self.assertTrue(any("order_type" in e for e in errors))

    def test_exchange_config_empty_pairs(self):
        from core.domain_config import ExchangeConfig
        cfg = ExchangeConfig(trading_pairs=[])
        errors = cfg.validate()
        self.assertTrue(any("trading_pairs" in e for e in errors))

    def test_validate_all(self):
        from core.domain_config import CapitalConfig, RiskConfig, validate_all
        errors = validate_all(CapitalConfig(), RiskConfig())
        self.assertEqual(len(errors), 0)

    def test_frozen_config(self):
        from core.domain_config import CapitalConfig
        cfg = CapitalConfig()
        with self.assertRaises(AttributeError):
            cfg.starting_capital_aud = 999  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
#  PositionReconciler Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionReconciler(unittest.TestCase):
    """Exchange position reconciliation."""

    def test_import(self):
        from core.position_reconciler import PositionReconciler
        self.assertTrue(True)

    def test_get_stats(self):
        from core.position_reconciler import PositionReconciler
        pm = MagicMock()
        em = MagicMock()
        rec = PositionReconciler(pm, em)
        stats = rec.get_stats()
        self.assertEqual(stats["reconcile_count"], 0)
        self.assertEqual(stats["drift_count"], 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PortfolioSnapshot Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPortfolioSnapshot(unittest.TestCase):
    """Immutable portfolio snapshot."""

    def test_win_rate_zero_trades(self):
        from core.portfolio_manager import PortfolioSnapshot
        snap = PortfolioSnapshot(
            portfolio_value_aud=1000, cash_balance_aud=1000,
            positions={}, total_trades=0, winning_trades=0,
            losing_trades=0, total_pnl_aud=0, realized_pnl_aud=0,
            daily_pnl_aud=0, total_fees_aud=0, max_drawdown_aud=0,
            peak_equity_aud=1000,
        )
        self.assertAlmostEqual(snap.win_rate, 0.0)

    def test_open_position_count(self):
        from core.portfolio_manager import PortfolioSnapshot
        snap = PortfolioSnapshot(
            portfolio_value_aud=1000, cash_balance_aud=500,
            positions={"BTC/USD": {"quantity": 0.01}, "ETH/USD": {"quantity": 0}},
            total_trades=2, winning_trades=1, losing_trades=0,
            total_pnl_aud=10, realized_pnl_aud=10, daily_pnl_aud=10,
            total_fees_aud=1, max_drawdown_aud=0, peak_equity_aud=1010,
        )
        self.assertEqual(snap.open_position_count, 1)


if __name__ == "__main__":
    unittest.main()
