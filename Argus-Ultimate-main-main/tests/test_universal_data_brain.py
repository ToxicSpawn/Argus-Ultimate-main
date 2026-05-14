"""Tests for Universal Data Brain — omniscient market intelligence."""
import unittest
import numpy as np
from core.universal_data_brain import UniversalDataBrain, DataSignal, MarketIntelligence


class TestSignalInjection(unittest.TestCase):
    def test_inject_raw(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "test_source", "price_action", 50000, 0.5, 0.9)
        intel = brain.compute("BTC/USD")
        self.assertEqual(intel.signal_count, 1)

    def test_inject_multiple_sources(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "source1", "price_action", 100, 0.8, 0.9)
        brain.inject_raw("BTC/USD", "source2", "derivatives", 200, 0.6, 0.8)
        brain.inject_raw("BTC/USD", "source3", "onchain", 300, 0.7, 0.85)
        intel = brain.compute("BTC/USD")
        self.assertEqual(intel.signal_count, 3)

    def test_signal_ttl_expiry(self):
        brain = UniversalDataBrain(signal_ttl_seconds=1.0)
        import time
        sig = DataSignal("old", "price_action", 100, 0.5, 0.9, time.time() - 10)
        brain.inject_signal("BTC/USD", sig)
        intel = brain.compute("BTC/USD")
        self.assertEqual(intel.signal_count, 0)


class TestCompositeScoring(unittest.TestCase):
    def test_all_bullish(self):
        brain = UniversalDataBrain()
        for i, cat in enumerate(["price_action", "derivatives", "onchain", "macro", "sentiment"]):
            brain.inject_raw("BTC/USD", f"src_{i}", cat, 0, 0.8, 0.9)
        intel = brain.compute("BTC/USD")
        self.assertGreater(intel.composite_score, 0.3)
        self.assertEqual(intel.regime_hint, "STRONG_BULL")

    def test_all_bearish(self):
        brain = UniversalDataBrain()
        for i, cat in enumerate(["price_action", "derivatives", "onchain", "macro", "sentiment"]):
            brain.inject_raw("BTC/USD", f"src_{i}", cat, 0, -0.8, 0.9)
        intel = brain.compute("BTC/USD")
        self.assertLess(intel.composite_score, -0.3)
        self.assertIn("BEAR", intel.regime_hint)

    def test_mixed_signals(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "bull1", "price_action", 0, 0.9, 0.9)
        brain.inject_raw("BTC/USD", "bear1", "derivatives", 0, -0.7, 0.8)
        intel = brain.compute("BTC/USD")
        self.assertLess(abs(intel.composite_score), 0.5)


class TestBuiltInProcessors(unittest.TestCase):
    def test_process_price_data(self):
        brain = UniversalDataBrain()
        close = np.linspace(100, 120, 60)
        high = close + 2
        low = close - 2
        volume = np.ones(60) * 1e6
        brain.process_price_data("BTC/USD", close, high, low, volume)
        intel = brain.compute("BTC/USD")
        self.assertGreater(intel.signal_count, 0)

    def test_process_funding_rate(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "init", "price_action", 0, 0, 0.5)
        brain.process_funding_rate("BTC/USD", 0.01)  # positive = bearish contrarian
        intel = brain.compute("BTC/USD")
        # Should have derivatives score
        self.assertGreater(intel.signal_count, 1)

    def test_process_whale_flow(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "init", "price_action", 0, 0, 0.5)
        brain.process_whale_flow("BTC/USD", exchange_inflow_usd=1e6, exchange_outflow_usd=5e6)
        intel = brain.compute("BTC/USD")
        # Net outflow = bullish (whales withdrawing to cold storage)
        self.assertGreater(intel.signal_count, 1)

    def test_process_fear_greed(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "init", "price_action", 0, 0, 0.5)
        brain.process_fear_greed(15)  # extreme fear = contrarian bullish
        intel = brain.compute("BTC/USD")
        self.assertGreater(intel.signal_count, 1)

    def test_process_liquidations(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "init", "price_action", 0, 0, 0.5)
        brain.process_liquidations("BTC/USD", long_liq_usd=1e6, short_liq_usd=5e6)
        intel = brain.compute("BTC/USD")
        # More shorts liquidated = bullish squeeze
        self.assertGreater(intel.signal_count, 1)

    def test_process_macro(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "init", "price_action", 0, 0, 0.5)
        brain.process_macro(dxy=104.5, dxy_prev=105.0, sp500_ret=0.01)  # DXY down = bullish
        intel = brain.compute("BTC/USD")
        self.assertGreater(intel.signal_count, 1)

    def test_process_options_flow(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "init", "price_action", 0, 0, 0.5)
        brain.process_options_flow("BTC/USD", put_call_ratio=0.5, iv_skew=-0.1)
        intel = brain.compute("BTC/USD")
        self.assertGreater(intel.signal_count, 1)

    def test_process_stablecoin_flow(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "init", "price_action", 0, 0, 0.5)
        brain.process_stablecoin_flow(mint_usd=1e9, burn_usd=1e8)  # massive minting = bullish
        intel = brain.compute("BTC/USD")
        self.assertGreater(intel.signal_count, 1)


class TestConvictionLevel(unittest.TestCase):
    def test_extreme_conviction(self):
        brain = UniversalDataBrain(min_signals_for_confidence=3)
        for i in range(8):
            brain.inject_raw("BTC/USD", f"src_{i}", "price_action", 0, 0.9, 0.95)
        intel = brain.compute("BTC/USD")
        self.assertEqual(intel.conviction_level, "EXTREME")

    def test_low_conviction(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "bull", "price_action", 0, 0.5, 0.5)
        brain.inject_raw("BTC/USD", "bear", "derivatives", 0, -0.5, 0.5)
        intel = brain.compute("BTC/USD")
        self.assertIn(intel.conviction_level, ("LOW", "MEDIUM"))


class TestSourceReliability(unittest.TestCase):
    def test_record_outcome_adjusts_weight(self):
        brain = UniversalDataBrain()
        for _ in range(10):
            brain.record_outcome("good_source", True)
        for _ in range(10):
            brain.record_outcome("bad_source", False)
        self.assertGreater(brain._source_weights.get("good_source", 0),
                           brain._source_weights.get("bad_source", 1))


class TestQueries(unittest.TestCase):
    def test_get_all_intelligence(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "s1", "price_action", 0, 0.5, 0.9)
        brain.inject_raw("ETH/USD", "s2", "price_action", 0, 0.3, 0.8)
        result = brain.get_all_intelligence()
        self.assertIn("BTC/USD", result)
        self.assertIn("ETH/USD", result)

    def test_get_strongest_signal(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "weak", "price_action", 0, 0.1, 0.5)
        brain.inject_raw("BTC/USD", "strong", "derivatives", 0, 0.9, 0.95)
        strongest = brain.get_strongest_signal("BTC/USD")
        self.assertIsNotNone(strongest)
        self.assertEqual(strongest.source, "strong")

    def test_get_stats(self):
        brain = UniversalDataBrain()
        brain.inject_raw("BTC/USD", "s1", "price_action", 0, 0.5, 0.9)
        stats = brain.get_stats()
        self.assertEqual(stats["symbols_tracked"], 1)
        self.assertEqual(stats["total_signals"], 1)


if __name__ == "__main__":
    unittest.main()
