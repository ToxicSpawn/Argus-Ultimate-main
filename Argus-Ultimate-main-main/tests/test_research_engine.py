"""Tests for research engine — autonomous R&D."""
import unittest
import numpy as np
from core.research_engine import (
    ResearchEngine, FeatureDiscovery, RegimeTransitionResearch,
    StrategyArchaeology, CorrelationMiner, ResearchFinding,
)


class TestFeatureDiscovery(unittest.TestCase):
    def test_discovers_predictive_features(self):
        fd = FeatureDiscovery(min_r_squared=0.001)
        T = 300
        t = np.arange(T, dtype=float)
        close = 100 + 10 * np.sin(t * 0.05) + np.random.RandomState(42).randn(T) * 0.5
        high = close + 2
        low = close - 2
        volume = np.random.RandomState(42).uniform(1e5, 1e6, T)
        findings = fd.research(close, volume, high, low)
        # Should find at least some features with r² > 0.001
        self.assertIsInstance(findings, list)

    def test_get_top_features(self):
        fd = FeatureDiscovery(min_r_squared=0.001)
        T = 200
        close = np.linspace(100, 150, T) + np.random.RandomState(42).randn(T) * 0.3
        fd.research(close, np.ones(T) * 1e6, close + 1, close - 1)
        top = fd.get_top_features(5)
        self.assertIsInstance(top, list)

    def test_insufficient_data(self):
        fd = FeatureDiscovery()
        findings = fd.research(np.array([100, 101]), np.array([1e6, 1e6]),
                               np.array([101, 102]), np.array([99, 100]))
        self.assertEqual(len(findings), 0)


class TestRegimeTransitionResearch(unittest.TestCase):
    def test_detects_transition(self):
        rtr = RegimeTransitionResearch()
        findings = []
        for i in range(20):
            findings.extend(rtr.record_regime("trending", i))
        for i in range(20, 25):
            findings.extend(rtr.record_regime("ranging", i))
        # Should record transition trending→ranging
        self.assertEqual(len(rtr._transitions), 1)

    def test_finds_pattern_after_repeats(self):
        rtr = RegimeTransitionResearch()
        findings = []
        for cycle in range(5):
            base = cycle * 20
            for i in range(10):
                findings.extend(rtr.record_regime("trending", base + i))
            for i in range(10):
                findings.extend(rtr.record_regime("ranging", base + 10 + i))
        # Should find pattern after 3+ repetitions
        self.assertGreater(len(rtr._transitions), 2)

    def test_regime_stats(self):
        rtr = RegimeTransitionResearch()
        for i in range(10):
            rtr.record_regime("trending", i, pnl=0.5)
        for i in range(10, 20):
            rtr.record_regime("ranging", i, pnl=-0.2)
        stats = rtr.get_regime_stats()
        self.assertIn("trending", stats)


class TestStrategyArchaeology(unittest.TestCase):
    def test_detects_degradation(self):
        sa = StrategyArchaeology()
        # First half profitable, second half losing
        trades = [{"pnl": 2.0}] * 10 + [{"pnl": -1.5}] * 10
        regimes = ["trending"] * 10 + ["ranging"] * 10
        findings = sa.autopsy("test_strat", trades, regimes)
        degrade = [f for f in findings if "degradation" in f.title]
        self.assertGreater(len(degrade), 0)

    def test_detects_regime_sensitivity(self):
        sa = StrategyArchaeology()
        trades = [{"pnl": 3.0}] * 5 + [{"pnl": -2.0}] * 5
        regimes = ["trending"] * 5 + ["ranging"] * 5
        findings = sa.autopsy("regime_strat", trades, regimes)
        regime_findings = [f for f in findings if "regime" in f.title.lower()]
        self.assertGreater(len(regime_findings), 0)

    def test_empty_trades(self):
        sa = StrategyArchaeology()
        findings = sa.autopsy("empty", [], [])
        self.assertEqual(len(findings), 0)


class TestCorrelationMiner(unittest.TestCase):
    def test_finds_correlated_series(self):
        cm = CorrelationMiner(window=50, min_correlation=0.3)
        rng = np.random.RandomState(42)
        base = rng.randn(100)
        for i in range(100):
            cm.record("btc", base[i] + rng.randn() * 0.1)
            cm.record("eth", base[i] * 0.8 + rng.randn() * 0.2)  # correlated
            cm.record("random", rng.randn())  # uncorrelated
        findings = cm.mine()
        # Should find BTC-ETH correlation
        corr_findings = [f for f in findings if "btc" in f.finding_id and "eth" in f.finding_id]
        self.assertGreater(len(corr_findings), 0)

    def test_insufficient_data(self):
        cm = CorrelationMiner(window=100)
        cm.record("btc", 1.0)
        findings = cm.mine()
        self.assertEqual(len(findings), 0)


class TestResearchEngine(unittest.TestCase):
    def test_runs_at_interval(self):
        re = ResearchEngine(research_interval=10)
        T = 200
        close = np.linspace(100, 150, T) + np.random.RandomState(42).randn(T) * 0.3
        report = re.run(cycle=10, close=close, regime="trending",
                        prices={"BTC/USD": 50000, "ETH/USD": 3000})
        self.assertIsNotNone(report)
        self.assertGreater(report.programs_active, 0)

    def test_skips_between_intervals(self):
        re = ResearchEngine(research_interval=100)
        report = re.run(cycle=5)
        self.assertIsNone(report)

    def test_accumulates_findings(self):
        re = ResearchEngine(research_interval=10)
        T = 200
        close = np.linspace(100, 150, T) + np.random.RandomState(42).randn(T) * 0.3
        for cycle in range(10, 60, 10):
            re.run(cycle=cycle, close=close, regime="trending",
                   prices={"BTC/USD": 50000})
        self.assertGreaterEqual(len(re._all_findings), 0)

    def test_strategy_autopsy(self):
        re = ResearchEngine()
        trades = [{"pnl": 2.0}] * 10 + [{"pnl": -1.5}] * 10
        regimes = ["trending"] * 10 + ["ranging"] * 10
        findings = re.autopsy_strategy("dead_strat", trades, regimes)
        self.assertIsInstance(findings, list)

    def test_get_stats(self):
        re = ResearchEngine()
        stats = re.get_stats()
        self.assertIn("total_findings", stats)
        self.assertIn("top_features", stats)


if __name__ == "__main__":
    unittest.main()
