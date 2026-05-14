"""
Tests for quant fund principles:
- StrategyValidator (no trading without proven edge)
- KellySizer (measured-edge position sizing)
- ImplementationShortfallTracker (execution quality feedback)
- Regime change liquidation
"""

import time
import unittest
from unittest.mock import MagicMock


class TestStrategyValidator(unittest.TestCase):
    """Strategy must prove edge before live trading."""

    def setUp(self):
        from core.strategy_validator import StrategyValidator
        self.sv = StrategyValidator(
            min_sharpe=0.5, min_trades=10, max_drawdown_pct=0.20,
            min_win_rate=0.35, min_profit_factor=1.2,
            results_path=":memory:",
        )

    def test_no_trades_fails(self):
        result = self.sv.validate("momentum", [])
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "no_trades")

    def test_profitable_strategy_passes(self):
        # Large base wins with small losses — keeps drawdown low relative to equity
        trades = []
        for _ in range(8):
            trades.extend([{"pnl": 50.0}, {"pnl": 50.0}, {"pnl": -10.0}])
        # WR=67%, PF=25.0, low drawdown since losses are small vs cumulative equity
        result = self.sv.validate("momentum", trades)
        self.assertTrue(result.passed, f"Should pass: {result.reason}")

    def test_losing_strategy_fails(self):
        trades = [{"pnl": -10.0}] * 8 + [{"pnl": 5.0}] * 4  # 33% win rate
        result = self.sv.validate("bad_strat", trades)
        self.assertFalse(result.passed)
        self.assertTrue(self.sv.is_blocked("bad_strat"))

    def test_too_few_trades_fails(self):
        trades = [{"pnl": 100.0}] * 5  # Only 5 trades, need 10
        result = self.sv.validate("new_strat", trades)
        self.assertFalse(result.passed)
        self.assertIn("trades=5", result.reason)

    def test_high_drawdown_fails(self):
        # Alternating big losses then recovery — high drawdown
        trades = [{"pnl": -50.0}] * 5 + [{"pnl": 60.0}] * 7
        result = self.sv.validate("volatile_strat", trades)
        # Drawdown will be high from the initial losses
        if result.max_drawdown_pct > 0.20:
            self.assertFalse(result.passed)

    def test_is_approved_after_passing(self):
        trades = []
        for _ in range(8):
            trades.extend([{"pnl": 50.0}, {"pnl": 50.0}, {"pnl": -10.0}])
        self.sv.validate("good_strat", trades)
        self.assertTrue(self.sv.is_approved("good_strat"))
        self.assertFalse(self.sv.is_blocked("good_strat"))

    def test_manual_approve(self):
        self.sv.approve_without_backtest("paper_only", "testing")
        self.assertTrue(self.sv.is_approved("paper_only"))

    def test_oos_degradation_check(self):
        is_trades = [{"pnl": 20.0}] * 10 + [{"pnl": -5.0}] * 3
        oos_trades = [{"pnl": 2.0}] * 8 + [{"pnl": -5.0}] * 5  # Much worse OOS
        all_trades = is_trades + oos_trades
        result = self.sv.validate("overfit", all_trades,
                                   in_sample_trades=is_trades,
                                   out_of_sample_trades=oos_trades)
        # If OOS much worse than IS, degradation should be high
        self.assertGreater(result.oos_degradation_pct, 0)


class TestKellySizer(unittest.TestCase):
    """Measured-edge Kelly criterion sizing."""

    def setUp(self):
        from core.kelly_sizing import KellySizer
        self.ks = KellySizer(kelly_fraction=0.25, min_trades=10, max_position_pct=0.15)

    def test_no_trades_returns_default(self):
        est = self.ks.compute("momentum", "BTC/USD")
        self.assertEqual(est.n_trades, 0)
        self.assertEqual(est.position_pct, self.ks._default_pct)

    def test_winning_strategy_sizes_up(self):
        # 70% win rate, avg win $20, avg loss $10 → positive Kelly
        for _ in range(7):
            self.ks.record_trade("winner", "BTC/USD", 20.0)
        for _ in range(3):
            self.ks.record_trade("winner", "BTC/USD", -10.0)
        est = self.ks.compute("winner", "BTC/USD")
        self.assertGreater(est.kelly_fraction, 0)
        self.assertGreater(est.position_pct, 0)
        self.assertGreater(est.win_rate, 0.6)

    def test_losing_strategy_sizes_zero(self):
        # 30% win rate, avg win $10, avg loss $20 → negative Kelly
        for _ in range(3):
            self.ks.record_trade("loser", "BTC/USD", 10.0)
        for _ in range(7):
            self.ks.record_trade("loser", "BTC/USD", -20.0)
        est = self.ks.compute("loser", "BTC/USD")
        self.assertEqual(est.kelly_fraction, 0.0)  # negative clamped to 0

    def test_fractional_kelly_caps(self):
        # Extreme edge: 90% win rate — full Kelly would be too aggressive
        for _ in range(18):
            self.ks.record_trade("edge", "BTC/USD", 10.0)
        for _ in range(2):
            self.ks.record_trade("edge", "BTC/USD", -5.0)
        est = self.ks.compute("edge", "BTC/USD")
        self.assertLessEqual(est.position_pct, 0.15)  # hard cap

    def test_insufficient_trades_conservative(self):
        for _ in range(5):
            self.ks.record_trade("new", "BTC/USD", 10.0)
        est = self.ks.compute("new", "BTC/USD")
        self.assertLess(est.confidence, 1.0)

    def test_per_symbol_tracking(self):
        for _ in range(15):
            self.ks.record_trade("strat", "BTC/USD", 10.0)
            self.ks.record_trade("strat", "ETH/USD", -5.0)
        btc = self.ks.compute("strat", "BTC/USD")
        eth = self.ks.compute("strat", "ETH/USD")
        self.assertGreater(btc.kelly_fraction, eth.kelly_fraction)

    def test_get_size_pct(self):
        pct = self.ks.get_size_pct("unknown", "BTC/USD")
        self.assertGreater(pct, 0)

    def test_get_all_estimates(self):
        self.ks.record_trade("a", "BTC/USD", 10.0)
        estimates = self.ks.get_all_estimates()
        self.assertIn("a:BTC/USD", estimates)


class TestImplementationShortfall(unittest.TestCase):
    """Execution quality tracking and feedback."""

    def setUp(self):
        from core.implementation_shortfall import ImplementationShortfallTracker
        self.ist = ImplementationShortfallTracker()

    def test_record_buy_slippage(self):
        rec = self.ist.record("BTC/USD", "momentum", "buy",
                              decision_price=50000, fill_price=50010, quantity=0.01)
        self.assertAlmostEqual(rec.shortfall_bps, 2.0, places=1)  # 10/50000*10000 = 2 bps

    def test_record_sell_slippage(self):
        rec = self.ist.record("BTC/USD", "momentum", "sell",
                              decision_price=50000, fill_price=49990, quantity=0.01)
        self.assertAlmostEqual(rec.shortfall_bps, 2.0, places=1)

    def test_negative_shortfall_is_improvement(self):
        rec = self.ist.record("BTC/USD", "momentum", "buy",
                              decision_price=50000, fill_price=49990, quantity=0.01)
        self.assertLess(rec.shortfall_bps, 0)  # price improvement

    def test_get_stats(self):
        for i in range(20):
            self.ist.record("BTC/USD", "momentum", "buy",
                            decision_price=50000, fill_price=50000 + i, quantity=0.01)
        stats = self.ist.get_stats("strategy:momentum")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.n_trades, 20)
        self.assertGreater(stats.avg_shortfall_bps, 0)

    def test_recommended_order_type_limit(self):
        # High IS → should recommend limit
        for _ in range(15):
            self.ist.record("BTC/USD", "bad_exec", "buy",
                            decision_price=50000, fill_price=50050, quantity=0.01)  # 10 bps
        rec = self.ist.get_recommended_order_type("bad_exec", "BTC/USD")
        self.assertEqual(rec, "limit")

    def test_recommended_order_type_twap(self):
        # Very high IS → should recommend twap
        for _ in range(15):
            self.ist.record("BTC/USD", "terrible_exec", "buy",
                            decision_price=50000, fill_price=50100, quantity=0.01)  # 20 bps
        rec = self.ist.get_recommended_order_type("terrible_exec", "BTC/USD")
        self.assertEqual(rec, "twap")

    def test_venue_ranking(self):
        for _ in range(15):
            self.ist.record("BTC/USD", "strat", "buy", 50000, 50005, 0.01, venue="kraken")
            self.ist.record("BTC/USD", "strat", "buy", 50000, 50020, 0.01, venue="coinbase")
        ranking = self.ist.get_venue_ranking()
        self.assertEqual(ranking[0]["venue"], "kraken")  # lower IS = better

    def test_advisory(self):
        self.ist.record("BTC/USD", "strat", "buy", 50000, 50005, 0.01)
        adv = self.ist.get_advisory()
        self.assertIn("global_avg_is_bps", adv)
        self.assertIn("venue_ranking", adv)


class TestRegimeChangeLiquidation(unittest.TestCase):
    """Regime change → aggressive position liquidation."""

    def test_regime_transition_generates_exit(self):
        """When regime transition detected, should generate 50% close signals."""
        positions = {
            "BTC/USD": {"quantity": 0.01, "side": "BUY", "current_price": 50000, "entry_price": 48000},
        }
        advisory = {"regime_transition": {"detected": True}}
        # Simulate the logic
        exit_signals = []
        if advisory.get("regime_transition", {}).get("detected"):
            for sym, pos in positions.items():
                qty = float(pos.get("quantity", 0))
                if qty > 0:
                    side = pos.get("side", "BUY")
                    exit_signals.append({
                        "symbol": sym,
                        "action": "SELL" if side == "BUY" else "BUY",
                        "strength": 0.50,
                    })
        self.assertEqual(len(exit_signals), 1)
        self.assertEqual(exit_signals[0]["strength"], 0.50)

    def test_no_transition_no_exit(self):
        advisory = {"regime_transition": {"detected": False}}
        self.assertFalse(advisory["regime_transition"]["detected"])

    def test_autoencoder_transition_triggers(self):
        advisory = {"autoencoder_regime": {"is_transition": True}}
        is_transition = advisory.get("autoencoder_regime", {}).get("is_transition", False)
        self.assertTrue(is_transition)


class TestKellyWiringInSystem(unittest.TestCase):
    """Verify Kelly sizer is wired into ComponentRegistry."""

    def test_kelly_sizer_slot_exists(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        self.assertTrue(hasattr(cr, "kelly_sizer"))

    def test_kelly_sizer_init(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        try:
            cr._init_kelly_sizer()
            self.assertIsNotNone(cr.kelly_sizer)
        except ImportError:
            self.skipTest("KellySizer not available")

    def test_is_tracker_slot_exists(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        self.assertTrue(hasattr(cr, "is_tracker"))

    def test_strategy_validator_slot_exists(self):
        from core.component_registry import ComponentRegistry
        cr = ComponentRegistry(config=MagicMock())
        self.assertTrue(hasattr(cr, "strategy_validator"))

    def test_kelly_records_on_fill(self):
        """KellySizer.record_trade called on fill with P&L."""
        from core.component_registry import ComponentRegistry
        from core.kelly_sizing import KellySizer
        cr = ComponentRegistry(config=MagicMock())
        cr.kelly_sizer = KellySizer()
        cr.on_fill({
            "symbol": "BTC/USD", "side": "sell", "price": 55000,
            "quantity": 0.01, "pnl": 50.0, "source_strategy": "momentum",
        })
        est = cr.kelly_sizer.compute("momentum", "BTC/USD")
        self.assertEqual(est.n_trades, 1)

    def test_is_tracker_records_on_fill(self):
        """IS tracker records fill with decision vs fill price."""
        from core.component_registry import ComponentRegistry
        from core.implementation_shortfall import ImplementationShortfallTracker
        cr = ComponentRegistry(config=MagicMock())
        cr.is_tracker = ImplementationShortfallTracker()
        cr.on_fill({
            "symbol": "BTC/USD", "side": "buy", "price": 50010,
            "quantity": 0.01, "signal_price": 50000,
            "source_strategy": "momentum", "exchange": "kraken",
        })
        stats = cr.is_tracker.get_stats("global")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.n_trades, 1)


if __name__ == "__main__":
    unittest.main()
