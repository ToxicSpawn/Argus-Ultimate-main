"""Tests for Tier 2 institutional modules — loss prevention, execution quality, audit."""
import time
import unittest


class TestStaleDataBreaker(unittest.TestCase):
    def test_fresh_data_passes(self):
        from core.stale_data_breaker import StaleDataBreaker
        b = StaleDataBreaker(threshold_seconds=30.0)
        b.update("BTC/USD", time.time())
        result = b.check("BTC/USD")
        self.assertFalse(result.is_stale)

    def test_stale_data_fails(self):
        from core.stale_data_breaker import StaleDataBreaker
        b = StaleDataBreaker(threshold_seconds=5.0)
        b.update("BTC/USD", time.time() - 10)
        result = b.check("BTC/USD")
        self.assertTrue(result.is_stale)

    def test_assert_fresh_raises(self):
        from core.stale_data_breaker import StaleDataBreaker
        b = StaleDataBreaker(threshold_seconds=1.0)
        b.update("BTC/USD", time.time() - 5)
        with self.assertRaises(RuntimeError):
            b.assert_fresh("BTC/USD")

    def test_unknown_symbol_stale(self):
        from core.stale_data_breaker import StaleDataBreaker
        b = StaleDataBreaker()
        result = b.check("UNKNOWN/USD")
        self.assertTrue(result.is_stale)


class TestOrderRateLimiter(unittest.TestCase):
    def test_under_limit_allowed(self):
        from core.order_rate_limiter import OrderRateLimiter
        limiter = OrderRateLimiter(max_orders_per_minute=10)
        for _ in range(5):
            limiter.record_order("kraken")
        result = limiter.check("kraken")
        self.assertTrue(result.allowed)

    def test_over_limit_blocked(self):
        from core.order_rate_limiter import OrderRateLimiter
        limiter = OrderRateLimiter(max_orders_per_minute=3)
        for _ in range(5):
            limiter.record_order("kraken")
        result = limiter.check("kraken")
        self.assertFalse(result.allowed)

    def test_assert_raises(self):
        from core.order_rate_limiter import OrderRateLimiter
        limiter = OrderRateLimiter(max_orders_per_minute=1)
        limiter.record_order("kraken")
        limiter.record_order("kraken")
        with self.assertRaises(RuntimeError):
            limiter.assert_allowed("kraken")


class TestDailyLossKill(unittest.TestCase):
    def test_within_limit_ok(self):
        from core.daily_loss_kill import DailyLossKill
        kill = DailyLossKill(max_daily_loss_pct=0.03, initial_capital=1000.0)
        kill.update_pnl(-20.0)  # -2% < -3%
        status = kill.check()
        self.assertFalse(status.triggered)

    def test_breach_triggers(self):
        from core.daily_loss_kill import DailyLossKill
        kill = DailyLossKill(max_daily_loss_pct=0.03, initial_capital=1000.0)
        kill.update_pnl(-40.0)  # -4% > -3%
        status = kill.check()
        self.assertTrue(status.triggered)

    def test_assert_raises_after_trigger(self):
        from core.daily_loss_kill import DailyLossKill
        kill = DailyLossKill(max_daily_loss_pct=0.01, initial_capital=1000.0)
        kill.update_pnl(-20.0)
        with self.assertRaises(RuntimeError):
            kill.assert_trading_allowed()

    def test_reset_clears(self):
        from core.daily_loss_kill import DailyLossKill
        kill = DailyLossKill(max_daily_loss_pct=0.01, initial_capital=1000.0)
        kill.update_pnl(-20.0)
        self.assertTrue(kill.check().triggered)
        kill.reset()
        self.assertFalse(kill.check().triggered)


class TestFatFingerGuard(unittest.TestCase):
    def test_small_order_allowed(self):
        from core.fat_finger_guard import FatFingerGuard
        guard = FatFingerGuard(max_order_pct=0.25)
        result = guard.check(quantity=0.001, price=50000, portfolio_value=1000)
        self.assertTrue(result.allowed)  # $50 = 5% < 25%

    def test_large_order_blocked(self):
        from core.fat_finger_guard import FatFingerGuard
        guard = FatFingerGuard(max_order_pct=0.10)
        result = guard.check(quantity=0.01, price=50000, portfolio_value=1000)
        self.assertFalse(result.allowed)  # $500 = 50% > 10%

    def test_assert_raises(self):
        from core.fat_finger_guard import FatFingerGuard
        guard = FatFingerGuard(max_order_pct=0.05)
        with self.assertRaises(RuntimeError):
            guard.assert_allowed(quantity=1.0, price=50000, portfolio_value=1000)


class TestFillRateMonitor(unittest.TestCase):
    def test_good_fill_rate(self):
        from core.fill_rate_monitor import FillRateMonitor
        monitor = FillRateMonitor(min_fill_rate=0.30)
        for _ in range(8):
            monitor.record_order("kraken", "BTC/USD", was_filled=True)
        for _ in range(2):
            monitor.record_order("kraken", "BTC/USD", was_filled=False)
        snap = monitor.get_fill_rate("kraken", "BTC/USD")
        self.assertGreater(snap.fill_rate, 0.30)
        self.assertFalse(snap.is_degraded)

    def test_bad_fill_rate(self):
        from core.fill_rate_monitor import FillRateMonitor
        monitor = FillRateMonitor(min_fill_rate=0.50)
        for _ in range(2):
            monitor.record_order("kraken", "BTC/USD", was_filled=True)
        for _ in range(8):
            monitor.record_order("kraken", "BTC/USD", was_filled=False)
        self.assertTrue(monitor.is_degraded("kraken", "BTC/USD"))


class TestVenueLatencyMonitor(unittest.TestCase):
    def test_normal_latency(self):
        from core.venue_latency_monitor import VenueLatencyMonitor
        monitor = VenueLatencyMonitor(spike_multiplier=2.0)
        for _ in range(20):
            monitor.record_latency("kraken", 100.0)
        self.assertFalse(monitor.is_elevated("kraken"))

    def test_spike_detected(self):
        from core.venue_latency_monitor import VenueLatencyMonitor
        monitor = VenueLatencyMonitor(spike_multiplier=1.5, baseline_window=10)
        for _ in range(10):
            monitor.record_latency("kraken", 100.0)  # baseline
        for _ in range(20):
            monitor.record_latency("kraken", 500.0)  # heavy spike
        snap = monitor.check("kraken")
        # avg should be well above 150 (1.5x baseline of 100)
        self.assertGreater(snap.avg_ms, 150)

    def test_recommended_venue(self):
        from core.venue_latency_monitor import VenueLatencyMonitor
        monitor = VenueLatencyMonitor()
        for _ in range(10):
            monitor.record_latency("kraken", 50.0)
            monitor.record_latency("coinbase", 150.0)
        best = monitor.get_recommended_venue(["kraken", "coinbase"])
        self.assertEqual(best, "kraken")


class TestSpreadMonitor(unittest.TestCase):
    def test_normal_spread(self):
        from core.spread_monitor import SpreadMonitor
        monitor = SpreadMonitor(wide_multiplier=2.0)
        for _ in range(20):
            monitor.record_spread("BTC/USD", 3.0)
        self.assertFalse(monitor.is_wide("BTC/USD"))
        self.assertAlmostEqual(monitor.get_size_multiplier("BTC/USD"), 1.0)

    def test_wide_spread(self):
        from core.spread_monitor import SpreadMonitor
        monitor = SpreadMonitor(wide_multiplier=1.5, baseline_window=10)
        for _ in range(10):
            monitor.record_spread("BTC/USD", 3.0)  # baseline
        for _ in range(20):
            monitor.record_spread("BTC/USD", 15.0)  # very wide
        snap = monitor.check("BTC/USD")
        # current avg should be well above 4.5 (1.5x baseline of 3)
        self.assertGreater(snap.current_bps, 4.5)


class TestDecisionAudit(unittest.TestCase):
    def test_record_and_query(self):
        import os, tempfile
        from core.decision_audit import DecisionAuditTrail, DecisionRecord
        db = os.path.join(tempfile.gettempdir(), f"audit_{os.getpid()}.db")
        try:
            audit = DecisionAuditTrail(db_path=db)
            audit.record(DecisionRecord(
                order_id="o1", symbol="BTC/USD", side="buy", strategy="momentum",
                initial_size_pct=0.15, final_size_pct=0.08,
                gates_applied=["vol_forecast", "correlation_penalty", "system_status"],
                advisory_keys_used=["ensemble", "fear_greed"],
                reason="3 gates reduced size", timestamp=time.time(),
            ))
            results = audit.query(symbol="BTC/USD")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].order_id, "o1")
        finally:
            try: os.unlink(db)
            except: pass

    def test_summary(self):
        import os, tempfile
        from core.decision_audit import DecisionAuditTrail, DecisionRecord
        db = os.path.join(tempfile.gettempdir(), f"audit2_{os.getpid()}.db")
        try:
            audit = DecisionAuditTrail(db_path=db)
            for i in range(5):
                audit.record(DecisionRecord(
                    order_id=f"o{i}", symbol="BTC/USD", side="buy", strategy="test",
                    initial_size_pct=0.10, final_size_pct=0.05,
                    gates_applied=["gate_a", "gate_b"], advisory_keys_used=["key1"],
                    reason="test", timestamp=time.time(),
                ))
            s = audit.summary()
            self.assertEqual(s["total_decisions"], 5)
        finally:
            try: os.unlink(db)
            except: pass


class TestSessionReport(unittest.TestCase):
    def test_generate_report(self):
        from core.session_report import SessionReportGenerator
        gen = SessionReportGenerator()
        gen.record_cycle()
        gen.record_cycle()
        gen.record_signal("BTC/USD", "momentum")
        gen.record_trade("BTC/USD", "buy", pnl=10.0, slippage_bps=2.0, strategy="momentum")
        gen.record_trade("ETH/USD", "sell", pnl=-5.0, slippage_bps=3.0, strategy="mean_reversion")
        gen.record_regime_change("ranging", "trending")
        report = gen.generate()
        self.assertEqual(report.cycles, 2)
        self.assertEqual(report.trades_executed, 2)
        self.assertEqual(report.signals_generated, 1)
        self.assertAlmostEqual(report.total_pnl, 5.0)
        self.assertEqual(report.regime_changes, 1)
        self.assertIn("momentum", report.strategies_used)

    def test_empty_report(self):
        from core.session_report import SessionReportGenerator
        gen = SessionReportGenerator()
        report = gen.generate()
        self.assertEqual(report.trades_executed, 0)
        self.assertAlmostEqual(report.win_rate, 0.0)

    def test_export_text(self):
        from core.session_report import SessionReportGenerator
        gen = SessionReportGenerator()
        gen.record_trade("BTC/USD", "buy", 10.0, 1.0, "test")
        text = gen.export_text()
        self.assertIn("BTC/USD", text)


if __name__ == "__main__":
    unittest.main()
