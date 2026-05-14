"""Tests for R740 engine — 192GB RAM optimizations."""
import unittest
import numpy as np
from core.r740_engine import (
    R740Engine, R740Config, InMemoryTickStore, ParallelBacktester,
    is_r740, get_ram_gb, _run_single_backtest,
)


class TestR740Config(unittest.TestCase):
    def test_full_r740(self):
        cfg = R740Config.for_hardware(192)
        self.assertEqual(cfg.evolver_population, 10000)
        self.assertEqual(cfg.market_memory_capacity, 500000)
        self.assertEqual(cfg.parallel_backtests, 100)

    def test_128gb(self):
        cfg = R740Config.for_hardware(128)
        self.assertEqual(cfg.evolver_population, 5000)

    def test_64gb(self):
        cfg = R740Config.for_hardware(64)
        self.assertEqual(cfg.evolver_population, 2000)

    def test_workstation(self):
        cfg = R740Config.for_hardware(16)
        self.assertEqual(cfg.evolver_population, 200)
        self.assertEqual(cfg.generator_population, 30)

    def test_scaling_is_monotonic(self):
        c16 = R740Config.for_hardware(16)
        c64 = R740Config.for_hardware(64)
        c128 = R740Config.for_hardware(128)
        c192 = R740Config.for_hardware(192)
        self.assertLess(c16.evolver_population, c64.evolver_population)
        self.assertLess(c64.evolver_population, c128.evolver_population)
        self.assertLessEqual(c128.evolver_population, c192.evolver_population)


class TestInMemoryTickStore(unittest.TestCase):
    def test_record_and_retrieve(self):
        ts = InMemoryTickStore(capacity_per_symbol=100)
        ts.record("BTC/USD", 50000, 0.1, "buy")
        ts.record("BTC/USD", 50010, 0.2, "sell")
        recent = ts.get_recent("BTC/USD", 10)
        self.assertEqual(len(recent), 2)
        self.assertAlmostEqual(recent[0].price, 50000)

    def test_vwap(self):
        ts = InMemoryTickStore()
        import time
        now = time.time()
        ts.record("BTC/USD", 50000, 1.0, "buy", now - 10)
        ts.record("BTC/USD", 50100, 2.0, "buy", now - 5)
        ts.record("BTC/USD", 50200, 1.0, "sell", now)
        vwap = ts.get_vwap("BTC/USD", 300)
        self.assertGreater(vwap, 50000)
        self.assertLess(vwap, 50200)

    def test_buy_sell_ratio(self):
        ts = InMemoryTickStore()
        import time
        now = time.time()
        for _ in range(8):
            ts.record("BTC/USD", 50000, 1.0, "buy", now)
        for _ in range(2):
            ts.record("BTC/USD", 50000, 1.0, "sell", now)
        ratio = ts.get_buy_sell_ratio("BTC/USD", 60)
        self.assertGreater(ratio, 3.0)

    def test_trade_velocity(self):
        ts = InMemoryTickStore()
        import time
        now = time.time()
        for i in range(10):
            ts.record("BTC/USD", 50000, 1.0, "buy", now - i)
        vel = ts.get_trade_velocity("BTC/USD", 60)
        self.assertGreater(vel, 0)

    def test_large_block_ratio(self):
        ts = InMemoryTickStore()
        import time
        now = time.time()
        # 9 small trades + 1 whale
        for _ in range(9):
            ts.record("BTC/USD", 50000, 0.01, "buy", now)
        ts.record("BTC/USD", 50000, 10.0, "buy", now)  # whale
        ratio = ts.get_large_block_ratio("BTC/USD", 300)
        self.assertGreater(ratio, 0.5)

    def test_replay(self):
        ts = InMemoryTickStore()
        import time
        now = time.time()
        ts.record("BTC/USD", 50000, 1.0, "buy", now - 100)
        ts.record("BTC/USD", 50100, 1.0, "sell", now - 50)
        ts.record("BTC/USD", 50200, 1.0, "buy", now)
        replay = ts.replay("BTC/USD", now - 75, now - 25)
        self.assertEqual(len(replay), 1)

    def test_capacity_limit(self):
        ts = InMemoryTickStore(capacity_per_symbol=10)
        for i in range(20):
            ts.record("BTC/USD", 50000 + i, 1.0, "buy")
        self.assertEqual(len(ts.get_recent("BTC/USD", 100)), 10)

    def test_empty_symbol(self):
        ts = InMemoryTickStore()
        self.assertEqual(ts.get_vwap("NONE/USD"), 0.0)
        self.assertEqual(ts.get_buy_sell_ratio("NONE/USD"), 1.0)

    def test_get_stats(self):
        ts = InMemoryTickStore()
        ts.record("BTC/USD", 50000, 1.0, "buy")
        stats = ts.get_stats()
        self.assertEqual(stats["symbols"], 1)
        self.assertEqual(stats["total_ticks"], 1)


class TestParallelBacktester(unittest.TestCase):
    def test_batch_backtest(self):
        bt = ParallelBacktester(max_workers=2)
        T = 300
        close = np.linspace(100, 150, T)
        high = close + 2
        low = close - 2
        volume = np.ones(T) * 1e6
        strategies = [
            {"id": "s1", "params": {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5}},
            {"id": "s2", "params": {"lookback": 30, "tp_pct": 3.0, "sl_pct": 2.0}},
            {"id": "s3", "params": {"lookback": 10, "tp_pct": 1.5, "sl_pct": 1.0}},
        ]
        results = bt.batch_backtest(strategies, close, high, low, volume)
        self.assertEqual(len(results), 3)
        self.assertIn("sharpe", results[0])

    def test_monte_carlo(self):
        bt = ParallelBacktester()
        trades = [2.0, -1.0, 1.5, -0.5, 3.0, -2.0, 1.0, 0.5] * 10
        mc = bt.monte_carlo(trades, n_simulations=500)
        self.assertIn("median_return", mc)
        self.assertIn("prob_profit", mc)
        self.assertGreater(mc["p95_return"], mc["p5_return"])
        self.assertEqual(mc["simulations"], 500)

    def test_monte_carlo_empty(self):
        bt = ParallelBacktester()
        mc = bt.monte_carlo([])
        self.assertAlmostEqual(mc["prob_profit"], 0)

    def test_single_backtest_worker(self):
        T = 200
        close = np.linspace(100, 130, T)
        high = close + 1
        low = close - 1
        volume = np.ones(T) * 1e6
        result = _run_single_backtest(
            ("test", close, high, low, volume, {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5})
        )
        self.assertEqual(result["strategy_id"], "test")
        self.assertIn("sharpe", result)


class TestR740Engine(unittest.TestCase):
    def test_engine_creates(self):
        engine = R740Engine()
        self.assertIsNotNone(engine.config)
        self.assertIsNotNone(engine.tick_store)
        self.assertIsNotNone(engine.backtester)

    def test_scale_component(self):
        engine = R740Engine()
        scaled = engine.scale_component("evolver_population", 200)
        self.assertGreaterEqual(scaled, 200)

    def test_get_stats(self):
        engine = R740Engine()
        stats = engine.get_stats()
        self.assertIn("active", stats)
        self.assertIn("ram_gb", stats)
        self.assertIn("config", stats)


if __name__ == "__main__":
    unittest.main()
