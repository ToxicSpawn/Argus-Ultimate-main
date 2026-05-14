"""Tests for price predictor and indicator cache."""
import unittest
import numpy as np
from core.price_predictor import (
    PricePredictor, PricePrediction, KalmanPriceFilter,
    MomentumPredictor, MicrostructurePredictor,
)
from core.indicator_cache import IndicatorCache


class TestKalmanFilter(unittest.TestCase):
    def test_converges_to_price(self):
        kf = KalmanPriceFilter()
        for i in range(50):
            kf.update(100.0)
        pred, std = kf.predict()
        self.assertAlmostEqual(pred, 100.0, places=0)

    def test_tracks_trend(self):
        kf = KalmanPriceFilter()
        for i in range(50):
            kf.update(100.0 + i * 0.5)
        pred, std = kf.predict()
        self.assertGreater(pred, 124)  # should predict continuation

    def test_uncertainty_decreases(self):
        kf = KalmanPriceFilter()
        kf.update(100.0)
        _, std1 = kf.predict()
        for i in range(20):
            kf.update(100.0 + i * 0.01)
        _, std2 = kf.predict()
        self.assertLess(std2, std1)


class TestMomentumPredictor(unittest.TestCase):
    def test_uptrend_predicts_up(self):
        mp = MomentumPredictor()
        for i in range(30):
            mp.update(100.0 + i * 0.5)
        pred = mp.predict()
        self.assertGreater(pred, 0)

    def test_flat_predicts_near_zero(self):
        mp = MomentumPredictor()
        for _ in range(30):
            mp.update(100.0)
        pred = mp.predict()
        self.assertAlmostEqual(pred, 0, places=1)

    def test_insufficient_data(self):
        mp = MomentumPredictor()
        mp.update(100.0)
        self.assertAlmostEqual(mp.predict(), 0.0)


class TestMicrostructurePredictor(unittest.TestCase):
    def test_accumulation_signal(self):
        mp = MicrostructurePredictor()
        # Normal then high volume + narrow spread
        for _ in range(15):
            mp.update(1000, 3.0, 0.1)
        for _ in range(10):
            mp.update(5000, 1.0, 0.2)  # high vol, narrow spread, positive
        pred = mp.predict()
        self.assertGreater(pred, 0)

    def test_insufficient_data(self):
        mp = MicrostructurePredictor()
        self.assertAlmostEqual(mp.predict(), 0.0)


class TestPricePredictor(unittest.TestCase):
    def test_predict_after_updates(self):
        pp = PricePredictor()
        for i in range(30):
            pp.update("BTC/USD", 50000 + i * 10, volume=1000, spread_bps=2.0)
        pred = pp.predict("BTC/USD")
        self.assertIsInstance(pred, PricePrediction)
        self.assertEqual(pred.symbol, "BTC/USD")
        self.assertIn(pred.direction, ("UP", "DOWN", "FLAT"))
        self.assertGreater(pred.predicted_price, 0)

    def test_unknown_symbol(self):
        pp = PricePredictor()
        pred = pp.predict("UNKNOWN/USD")
        self.assertEqual(pred.confidence, 0)
        self.assertEqual(pred.direction, "FLAT")

    def test_uptrend_predicts_up(self):
        pp = PricePredictor()
        for i in range(50):
            pp.update("BTC/USD", 50000 + i * 50, volume=1000, spread_bps=2.0)
        pred = pp.predict("BTC/USD")
        self.assertEqual(pred.direction, "UP")
        self.assertGreater(pred.predicted_return_pct, 0)

    def test_get_all_predictions(self):
        pp = PricePredictor()
        for i in range(30):
            pp.update("BTC/USD", 50000 + i)
            pp.update("ETH/USD", 3000 + i)
        preds = pp.get_all_predictions()
        self.assertIn("BTC/USD", preds)
        self.assertIn("ETH/USD", preds)

    def test_models_agree_count(self):
        pp = PricePredictor()
        for i in range(50):
            pp.update("BTC/USD", 50000 + i * 100, volume=5000, spread_bps=1.0)
        pred = pp.predict("BTC/USD")
        self.assertGreaterEqual(pred.models_agree, 0)
        self.assertLessEqual(pred.models_agree, 3)

    def test_get_stats(self):
        pp = PricePredictor()
        pp.update("BTC/USD", 50000)
        stats = pp.get_stats()
        self.assertEqual(stats["symbols_tracked"], 1)


class TestIndicatorCache(unittest.TestCase):
    def setUp(self):
        self.T = 200
        t = np.arange(self.T, dtype=float)
        self.close = 100 + 10 * np.sin(t * 0.1)
        self.high = self.close + 2
        self.low = self.close - 2
        self.volume = np.random.RandomState(42).uniform(1e5, 1e6, self.T)

    def test_build(self):
        cache = IndicatorCache()
        cache.build(self.close, self.high, self.low, self.volume)
        self.assertTrue(cache.built)
        self.assertGreater(cache.size, 0)

    def test_sma_cached(self):
        cache = IndicatorCache()
        cache.build(self.close)
        sma = cache.get("SMA", 20)
        self.assertIsNotNone(sma)
        self.assertEqual(len(sma), self.T)

    def test_ema_cached(self):
        cache = IndicatorCache()
        cache.build(self.close)
        ema = cache.get("EMA", 14)
        self.assertIsNotNone(ema)

    def test_rsi_bounded(self):
        cache = IndicatorCache()
        cache.build(self.close)
        rsi = cache.get("RSI", 14)
        self.assertIsNotNone(rsi)
        self.assertTrue(np.all(rsi >= 0))
        self.assertTrue(np.all(rsi <= 100))

    def test_macd_hist(self):
        cache = IndicatorCache()
        cache.build(self.close)
        macd = cache.get("MACD_HIST", 0)
        self.assertIsNotNone(macd)
        self.assertEqual(len(macd), self.T)

    def test_atr(self):
        cache = IndicatorCache()
        cache.build(self.close, self.high, self.low)
        atr = cache.get("ATR", 14)
        self.assertIsNotNone(atr)
        self.assertTrue(np.all(atr >= 0))

    def test_bb_bands(self):
        cache = IndicatorCache()
        cache.build(self.close)
        upper = cache.get("BB_UPPER", 20)
        lower = cache.get("BB_LOWER", 20)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(lower)
        # Upper should be above lower where both are valid
        valid = (~np.isnan(upper)) & (~np.isnan(lower)) & (upper > 0) & (lower > 0)
        self.assertTrue(np.all(upper[valid] >= lower[valid]))

    def test_get_or_compute(self):
        cache = IndicatorCache()
        cache.build(self.close)
        # Get one that wasn't pre-computed (period 7)
        result = cache.get_or_compute("SMA", 7, self.close)
        self.assertEqual(len(result), self.T)

    def test_vol_ratio(self):
        cache = IndicatorCache()
        cache.build(self.close, volume=self.volume)
        vr = cache.get("VOL_RATIO", 20)
        self.assertIsNotNone(vr)
        self.assertTrue(np.all(np.isfinite(vr)))

    def test_multiple_periods(self):
        cache = IndicatorCache()
        cache.build(self.close, periods=[5, 10, 20, 50])
        self.assertIsNotNone(cache.get("SMA", 5))
        self.assertIsNotNone(cache.get("SMA", 10))
        self.assertIsNotNone(cache.get("SMA", 20))
        self.assertIsNotNone(cache.get("SMA", 50))


if __name__ == "__main__":
    unittest.main()
