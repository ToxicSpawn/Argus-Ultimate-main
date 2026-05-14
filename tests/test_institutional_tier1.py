"""Tests for Tier 1 institutional features."""
import time
import unittest


class TestWashTradeDetector(unittest.TestCase):

    def test_opposite_side_within_window_blocked(self):
        from core.wash_trade_detector import WashTradeDetector, FillRecord
        detector = WashTradeDetector(window_seconds=300, block_mode=True)
        now = time.time()
        # Buy BTC
        detector.check(FillRecord("BTC/USD", "kraken", "buy", 0.01, 50000, now - 60))
        # Sell BTC 60 seconds later — wash trade
        alert = detector.check(FillRecord("BTC/USD", "kraken", "sell", 0.01, 50010, now))
        self.assertIsNotNone(alert)
        self.assertTrue(alert.blocked)

    def test_same_side_not_flagged(self):
        from core.wash_trade_detector import WashTradeDetector, FillRecord
        detector = WashTradeDetector()
        now = time.time()
        detector.check(FillRecord("BTC/USD", "kraken", "buy", 0.01, 50000, now - 30))
        alert = detector.check(FillRecord("BTC/USD", "kraken", "buy", 0.01, 50010, now))
        self.assertIsNone(alert)

    def test_outside_window_not_flagged(self):
        from core.wash_trade_detector import WashTradeDetector, FillRecord
        detector = WashTradeDetector(window_seconds=60)
        now = time.time()
        detector.check(FillRecord("BTC/USD", "kraken", "buy", 0.01, 50000, now - 120))
        alert = detector.check(FillRecord("BTC/USD", "kraken", "sell", 0.01, 50010, now))
        self.assertIsNone(alert)

    def test_different_venue_not_flagged(self):
        from core.wash_trade_detector import WashTradeDetector, FillRecord
        detector = WashTradeDetector()
        now = time.time()
        detector.check(FillRecord("BTC/USD", "kraken", "buy", 0.01, 50000, now - 30))
        alert = detector.check(FillRecord("BTC/USD", "coinbase", "sell", 0.01, 50010, now))
        self.assertIsNone(alert)

    def test_low_overlap_not_flagged(self):
        from core.wash_trade_detector import WashTradeDetector, FillRecord
        detector = WashTradeDetector(min_overlap_pct=0.80)
        now = time.time()
        detector.check(FillRecord("BTC/USD", "kraken", "buy", 1.0, 50000, now - 30))
        alert = detector.check(FillRecord("BTC/USD", "kraken", "sell", 0.1, 50010, now))
        self.assertIsNone(alert)  # 10% overlap < 80% threshold

    def test_warn_mode_doesnt_block(self):
        from core.wash_trade_detector import WashTradeDetector, FillRecord
        detector = WashTradeDetector(block_mode=False)
        now = time.time()
        detector.check(FillRecord("BTC/USD", "kraken", "buy", 0.01, 50000, now - 30))
        alert = detector.check(FillRecord("BTC/USD", "kraken", "sell", 0.01, 50010, now))
        self.assertIsNotNone(alert)
        self.assertFalse(alert.blocked)


class TestCrossVenueValidator(unittest.TestCase):

    def test_prices_within_tolerance(self):
        from core.cross_venue_validator import CrossVenueValidator
        v = CrossVenueValidator(max_divergence_pct=1.0)
        v.update("kraken", "BTC/USD", bid=50000, ask=50010)
        v.update("coinbase", "BTC/USD", bid=50005, ask=50015)
        result = v.validate("BTC/USD")
        self.assertTrue(result.valid)

    def test_divergent_venue_flagged(self):
        from core.cross_venue_validator import CrossVenueValidator
        v = CrossVenueValidator(max_divergence_pct=0.5)
        v.update("kraken", "BTC/USD", bid=50000, ask=50010)
        v.update("coinbase", "BTC/USD", bid=50000, ask=50010)
        v.update("bad_venue", "BTC/USD", bid=51000, ask=51010)  # 2% off
        result = v.validate("BTC/USD")
        self.assertFalse(result.valid)
        self.assertEqual(result.divergent_venue, "bad_venue")

    def test_insufficient_venues_ok(self):
        from core.cross_venue_validator import CrossVenueValidator
        v = CrossVenueValidator(min_venues=2)
        v.update("kraken", "BTC/USD", bid=50000, ask=50010)
        result = v.validate("BTC/USD")
        self.assertTrue(result.valid)  # only 1 venue, can't validate


class TestBestExecutionReport(unittest.TestCase):

    def test_record_fill(self):
        from core.best_execution_report import BestExecutionReporter
        reporter = BestExecutionReporter(report_dir="data/test_ber")
        rec = reporter.record_fill(
            order_id="o1", symbol="BTC/USD", side="buy",
            strategy="momentum", venue_used="kraken",
            decision_price=50000, arrival_price=50005,
            fill_price=50010, quantity=0.01, spread_bps=2.0,
            commission=0.13,
        )
        self.assertGreater(rec.slippage_bps, 0)
        self.assertGreater(rec.implementation_shortfall_bps, 0)

    def test_summary(self):
        from core.best_execution_report import BestExecutionReporter
        reporter = BestExecutionReporter(report_dir="data/test_ber2")
        reporter.record_fill(
            order_id="o1", symbol="BTC/USD", side="buy",
            strategy="test", venue_used="kraken",
            decision_price=50000, fill_price=50010, quantity=0.01,
        )
        reporter.record_fill(
            order_id="o2", symbol="ETH/USD", side="sell",
            strategy="test", venue_used="kraken",
            decision_price=2000, fill_price=1995, quantity=1.0,
        )
        s = reporter.summary()
        self.assertEqual(s["total_fills"], 2)

    def test_empty_summary(self):
        from core.best_execution_report import BestExecutionReporter
        reporter = BestExecutionReporter(report_dir="data/test_ber3")
        self.assertEqual(reporter.summary()["total_fills"], 0)


class TestOrderBlotter(unittest.TestCase):

    def test_record_and_query(self):
        import os, tempfile
        from core.order_blotter import OrderBlotter, BlotterEntry
        db = os.path.join(tempfile.gettempdir(), f"blotter_test_{os.getpid()}.db")
        try:
            blotter = OrderBlotter(db_path=db)
            blotter.record(BlotterEntry(
                order_id="o1", symbol="BTC/USD", side="buy", strategy="momentum",
                venue="kraken", status="filled", quantity=0.01, filled_qty=0.01,
                price=50000, fill_price=50010, commission=0.13, slippage_bps=2.0,
                created_at=time.time(), updated_at=time.time(),
            ))
            results = blotter.query(symbol="BTC/USD")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "filled")
        finally:
            try: os.unlink(db)
            except: pass

    def test_filter_by_status(self):
        import os, tempfile
        from core.order_blotter import OrderBlotter, BlotterEntry
        db = os.path.join(tempfile.gettempdir(), f"blotter_test2_{os.getpid()}.db")
        try:
            blotter = OrderBlotter(db_path=db)
            now = time.time()
            blotter.record(BlotterEntry("o1", "BTC/USD", "buy", "s1", "k", "filled", 0.01, 0.01, 50000, 50010, 0, 0, now, now))
            blotter.record(BlotterEntry("o2", "BTC/USD", "buy", "s1", "k", "open", 0.01, 0, 50000, 0, 0, 0, now, now))
            filled = blotter.query(status="filled")
            self.assertEqual(len(filled), 1)
            open_orders = blotter.query(status="open")
            self.assertEqual(len(open_orders), 1)
        finally:
            try: os.unlink(db)
            except: pass


class TestAPIKeyRotation(unittest.TestCase):

    def test_fresh_key_ok(self):
        import tempfile
        from core.api_key_rotation import APIKeyRotationPolicy
        path = tempfile.mktemp(suffix=".json")
        policy = APIKeyRotationPolicy(state_path=path, warn_days=90, block_days=180)
        policy.register_key("kraken", "ABCDEF1234567890")
        status = policy.check("kraken")
        self.assertEqual(status.status, "ok")
        self.assertLess(status.age_days, 1)

    def test_unregistered_key_expired(self):
        import tempfile
        from core.api_key_rotation import APIKeyRotationPolicy
        path = tempfile.mktemp(suffix=".json")
        policy = APIKeyRotationPolicy(state_path=path)
        status = policy.check("unknown_exchange")
        self.assertEqual(status.status, "expired")

    def test_assert_live_allowed_fresh(self):
        import tempfile
        from core.api_key_rotation import APIKeyRotationPolicy
        path = tempfile.mktemp(suffix=".json")
        policy = APIKeyRotationPolicy(state_path=path)
        policy.register_key("kraken", "ABCDEF1234567890")
        policy.assert_live_allowed("kraken")  # should not raise


if __name__ == "__main__":
    unittest.main()
