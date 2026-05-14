"""Tests for GPU-accelerated evolution engine."""
import unittest

try:
    import torch
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _HAS_CUDA = False

from core.gpu_evolution import (
    GPUBacktester, GPUEvolutionEngine, NeuralSurrogate,
    gpu_bootstrap_reality_check, gpu_mccv_sharpes, gpu_available,
)


@unittest.skipUnless(_HAS_TORCH, "PyTorch not installed")
class TestGPUAvailability(unittest.TestCase):
    def test_gpu_available(self):
        self.assertEqual(gpu_available(), _HAS_CUDA)


@unittest.skipUnless(_HAS_CUDA, "CUDA not available")
class TestGPUBatchBacktestBreakout(unittest.TestCase):
    def test_batch_breakout_basic(self):
        device = torch.device("cuda")
        bt = GPUBacktester()

        # Trending price → breakout should find trades
        T = 300
        close = torch.linspace(100, 150, T, device=device)
        close += torch.randn(T, device=device) * 0.5  # add noise
        high = close * 1.002

        N = 10
        lookbacks = torch.tensor([10, 15, 20, 25, 30, 10, 15, 20, 25, 30],
                                 dtype=torch.float32, device=device)
        tp_pcts = torch.full((N,), 2.0, device=device)
        sl_pcts = torch.full((N,), 1.5, device=device)

        metrics = bt.batch_backtest_breakout(close, high, lookbacks, tp_pcts, sl_pcts)
        self.assertEqual(metrics["sharpe"].shape[0], N)
        self.assertEqual(metrics["trade_count"].shape[0], N)
        # At least some genomes should find trades
        self.assertGreater(metrics["trade_count"].sum().item(), 0)

    def test_batch_breakout_empty_market(self):
        device = torch.device("cuda")
        bt = GPUBacktester()
        # Flat price → no breakouts
        T = 100
        close = torch.full((T,), 100.0, device=device)
        high = close
        N = 5
        lookbacks = torch.full((N,), 20.0, device=device)
        tp_pcts = torch.full((N,), 2.0, device=device)
        sl_pcts = torch.full((N,), 1.5, device=device)

        metrics = bt.batch_backtest_breakout(close, high, lookbacks, tp_pcts, sl_pcts)
        self.assertEqual(metrics["trade_count"].sum().item(), 0)

    def test_batch_size_scalability(self):
        """Test with 500 genomes — should not OOM on 16GB GPU."""
        device = torch.device("cuda")
        bt = GPUBacktester()
        T = 500
        close = torch.linspace(100, 200, T, device=device) + torch.randn(T, device=device) * 0.3
        high = close * 1.001
        N = 500
        lookbacks = torch.randint(5, 60, (N,), device=device).float()
        tp_pcts = torch.rand(N, device=device) * 4.5 + 0.5
        sl_pcts = torch.rand(N, device=device) * 2.5 + 0.5

        metrics = bt.batch_backtest_breakout(close, high, lookbacks, tp_pcts, sl_pcts)
        self.assertEqual(metrics["sharpe"].shape[0], 500)


@unittest.skipUnless(_HAS_CUDA, "CUDA not available")
class TestGPUBatchMeanReversion(unittest.TestCase):
    def test_batch_mean_reversion(self):
        device = torch.device("cuda")
        bt = GPUBacktester()
        T = 300
        # Oscillating price → mean reversion should work
        t = torch.arange(T, dtype=torch.float32, device=device)
        close = 100 + 5 * torch.sin(t * 0.1) + torch.randn(T, device=device) * 0.5
        N = 8
        bb_stds = torch.tensor([1.5, 2.0, 2.0, 2.5, 1.5, 2.0, 2.5, 3.0],
                               dtype=torch.float32, device=device)
        sl_pcts = torch.full((N,), 2.0, device=device)

        metrics = bt.batch_backtest_mean_reversion(close, bb_stds, sl_pcts)
        self.assertEqual(metrics["sharpe"].shape[0], N)


@unittest.skipUnless(_HAS_CUDA, "CUDA not available")
class TestGPUBatchMomentum(unittest.TestCase):
    def test_batch_momentum(self):
        device = torch.device("cuda")
        bt = GPUBacktester()
        T = 300
        close = torch.linspace(100, 180, T, device=device) + torch.randn(T, device=device) * 0.3
        N = 6
        fast_periods = torch.tensor([5, 10, 15, 20, 25, 10], dtype=torch.float32, device=device)
        trail_pcts = torch.full((N,), 2.0, device=device)

        metrics = bt.batch_backtest_momentum(close, fast_periods, trail_pcts)
        self.assertEqual(metrics["sharpe"].shape[0], N)


@unittest.skipUnless(_HAS_CUDA, "CUDA not available")
class TestGPUBootstrap(unittest.TestCase):
    def test_strong_signal(self):
        trades = torch.tensor([2.0, 1.5, 1.8, 2.2, 1.9] * 10,
                              dtype=torch.float32, device=torch.device("cuda"))
        pval = gpu_bootstrap_reality_check(trades, n_bootstrap=2000)
        self.assertLess(pval, 0.60)

    def test_empty(self):
        trades = torch.tensor([], dtype=torch.float32, device=torch.device("cuda"))
        self.assertAlmostEqual(gpu_bootstrap_reality_check(trades), 1.0)

    def test_speed_vs_volume(self):
        """2000 bootstrap samples on 1000 trades should be fast on GPU."""
        import time
        trades = torch.randn(1000, device=torch.device("cuda"))
        t0 = time.time()
        gpu_bootstrap_reality_check(trades, n_bootstrap=5000)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 2.0)  # should be <0.1s but generous limit


@unittest.skipUnless(_HAS_CUDA, "CUDA not available")
class TestGPUMCCV(unittest.TestCase):
    def test_mccv_returns_sharpes(self):
        trades = torch.tensor([1.0, -0.5, 0.8, -0.3, 1.2, -0.7, 0.5, 0.3] * 5,
                              dtype=torch.float32, device=torch.device("cuda"))
        sharpes = gpu_mccv_sharpes(trades, n_folds=5)
        self.assertGreater(len(sharpes), 0)
        self.assertTrue(all(isinstance(s, float) for s in sharpes))


@unittest.skipUnless(_HAS_TORCH, "PyTorch not installed")
class TestNeuralSurrogate(unittest.TestCase):
    def test_not_ready_initially(self):
        ns = NeuralSurrogate(device=torch.device("cpu"))
        self.assertFalse(ns.ready)

    def test_train_and_predict(self):
        ns = NeuralSurrogate(input_dim=3, hidden=16, retrain_interval=20,
                             device=torch.device("cpu"))
        # Record linear relationship: higher params → higher fitness
        for i in range(30):
            ns.record([float(i), float(i) * 0.5, 1.0], float(i) / 30.0)
        self.assertTrue(ns.ready)

        # Prediction should roughly follow trend
        pred_low = ns.predict([0.0, 0.0, 1.0])
        pred_high = ns.predict([20.0, 10.0, 1.0])
        self.assertIsNotNone(pred_low)
        self.assertIsNotNone(pred_high)
        self.assertGreater(pred_high, pred_low)

    def test_predict_none_when_untrained(self):
        ns = NeuralSurrogate(device=torch.device("cpu"))
        self.assertIsNone(ns.predict([1.0, 2.0, 3.0]))


@unittest.skipUnless(_HAS_CUDA, "CUDA not available")
class TestGPUEvolutionEngine(unittest.TestCase):
    def test_engine_available(self):
        engine = GPUEvolutionEngine()
        self.assertTrue(engine.available)

    def test_batch_evaluate_breakout(self):
        import numpy as np
        engine = GPUEvolutionEngine()
        close = np.linspace(100, 150, 300) + np.random.randn(300) * 0.5
        high = close * 1.002
        params = [
            {"lookback": 20, "tp_pct": 2.0, "sl_pct": 1.5},
            {"lookback": 30, "tp_pct": 3.0, "sl_pct": 2.0},
            {"lookback": 15, "tp_pct": 1.5, "sl_pct": 1.0},
        ]
        results = engine.batch_evaluate_breakout(close, high, params)
        self.assertEqual(len(results), 3)
        self.assertIn("sharpe", results[0])
        self.assertIn("trade_count", results[0])

    def test_batch_evaluate_mean_reversion(self):
        import numpy as np
        engine = GPUEvolutionEngine()
        t = np.arange(300)
        close = 100 + 5 * np.sin(t * 0.1) + np.random.randn(300) * 0.5
        params = [
            {"bb_std": 2.0, "sl_pct": 1.5},
            {"bb_std": 2.5, "sl_pct": 2.0},
        ]
        results = engine.batch_evaluate_mean_reversion(close, params)
        self.assertEqual(len(results), 2)

    def test_gpu_bootstrap_via_engine(self):
        engine = GPUEvolutionEngine()
        pval = engine.gpu_bootstrap([2.0, 1.5, 1.8, 2.2, 1.9] * 10, n_bootstrap=1000)
        self.assertIsInstance(pval, float)
        self.assertGreaterEqual(pval, 0.0)
        self.assertLessEqual(pval, 1.0)

    def test_gpu_mccv_via_engine(self):
        engine = GPUEvolutionEngine()
        sharpes = engine.gpu_mccv([1.0, -0.5, 0.8, -0.3, 1.2] * 10, n_folds=5)
        self.assertIsInstance(sharpes, list)


if __name__ == "__main__":
    unittest.main()
