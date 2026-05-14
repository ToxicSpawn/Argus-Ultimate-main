"""Tests for meta-learner, market memory, and entropy filter."""
import math
import unittest
from core.meta_learner import (
    MetaLearner, HyperparamConfig, MetaObservation,
    MarketMemory, MarketState, EntropyFilter,
)


class TestMetaLearner(unittest.TestCase):
    def test_initial_state(self):
        ml = MetaLearner(adjustment_interval=100)
        self.assertEqual(ml._cycle_count, 0)
        self.assertEqual(len(ml._history), 0)

    def test_no_adjustment_before_interval(self):
        ml = MetaLearner(adjustment_interval=100)
        result = ml.step(50)
        self.assertIsNone(result)

    def test_adjustment_at_interval(self):
        ml = MetaLearner(adjustment_interval=10)
        ml.record_evolver_result(0.5)
        ml.record_generator_result(0.3)
        result = ml.step(10)
        self.assertIsNotNone(result)
        self.assertIn("action", result)

    def test_history_accumulates(self):
        ml = MetaLearner(adjustment_interval=10)
        for i in range(5):
            ml.record_evolver_result(float(i) * 0.1)
            ml.step((i + 1) * 10)
        self.assertGreater(len(ml._history), 0)

    def test_explore_vs_exploit(self):
        ml = MetaLearner(adjustment_interval=10, explore_rate=1.0)  # always explore
        for i in range(4):
            ml.record_evolver_result(0.5)
            ml.step((i + 1) * 10)
        result = ml.step(50)
        if result:
            self.assertEqual(result["action"], "explore")

    def test_config_perturb_stays_valid(self):
        ml = MetaLearner()
        base = HyperparamConfig()
        for _ in range(50):
            perturbed = ml._perturb_config(base)
            self.assertGreater(perturbed.evolver_mutation_rate, 0)
            self.assertLess(perturbed.evolver_mutation_rate, 1)
            self.assertGreater(perturbed.evolver_population_size, 0)

    def test_get_stats(self):
        ml = MetaLearner()
        stats = ml.get_stats()
        self.assertIn("history_size", stats)
        self.assertIn("current_config", stats)


class TestMarketMemory(unittest.TestCase):
    def test_remember_and_recall(self):
        mm = MarketMemory(capacity=100, k_nearest=3)
        # Store 20 memories with positive outcomes in trending regime
        for i in range(20):
            state = MarketState(
                timestamp=float(i), regime="trending",
                volatility=0.02, trend_strength=0.5,
                volume_ratio=1.2, rsi=60.0, spread_bps=2.0,
                correlation_btc_eth=0.8, best_strategy="breakout",
                outcome_pnl=1.5,
            )
            mm.remember(state)

        # Recall with similar conditions
        current = MarketState(
            timestamp=100.0, regime="trending",
            volatility=0.02, trend_strength=0.4,
            volume_ratio=1.1, rsi=58.0, spread_bps=2.5,
            correlation_btc_eth=0.78, best_strategy="",
        )
        result = mm.recall(current)
        self.assertGreater(result["similar_count"], 0)
        self.assertEqual(result["recommended_strategy"], "breakout")
        self.assertGreater(result["expected_pnl"], 0)

    def test_empty_memory(self):
        mm = MarketMemory()
        state = MarketState(0, "ranging", 0.01, 0.0, 1.0, 50.0, 2.0, 0.8, "")
        result = mm.recall(state)
        self.assertEqual(result["similar_count"], 0)

    def test_update_outcome(self):
        mm = MarketMemory()
        state = MarketState(100.0, "trending", 0.02, 0.5, 1.2, 60.0, 2.0, 0.8, "breakout")
        mm.remember(state)
        mm.update_outcome(100.0, pnl=2.5)
        self.assertAlmostEqual(mm._memories[0].outcome_pnl, 2.5)

    def test_capacity_limit(self):
        mm = MarketMemory(capacity=10)
        for i in range(20):
            mm.remember(MarketState(float(i), "ranging", 0.01, 0.0, 1.0, 50.0, 2.0, 0.8, ""))
        self.assertEqual(mm.size(), 10)

    def test_feature_vector(self):
        state = MarketState(0, "trending", 0.02, 0.5, 1.5, 60.0, 3.0, 0.85, "")
        vec = state.feature_vector()
        self.assertEqual(len(vec), 7)
        self.assertAlmostEqual(vec[0], 1.0)  # trending = 1.0

    def test_different_regimes_different_recommendations(self):
        mm = MarketMemory(capacity=100, k_nearest=5)
        # Trending markets: breakout works
        for i in range(15):
            mm.remember(MarketState(
                float(i), "trending", 0.02, 0.7, 1.3, 65.0, 2.0, 0.8,
                "breakout", outcome_pnl=2.0))
        # Ranging markets: mean_reversion works
        for i in range(15, 30):
            mm.remember(MarketState(
                float(i), "ranging", 0.01, 0.0, 0.9, 50.0, 3.0, 0.6,
                "mean_reversion", outcome_pnl=1.5))

        # Query trending → should recommend breakout
        trending_q = MarketState(100, "trending", 0.02, 0.6, 1.2, 62.0, 2.0, 0.8, "")
        t_result = mm.recall(trending_q)

        # Query ranging → should recommend mean_reversion
        ranging_q = MarketState(101, "ranging", 0.01, 0.1, 0.8, 48.0, 3.5, 0.6, "")
        r_result = mm.recall(ranging_q)

        # At least one should match its regime
        self.assertTrue(
            t_result["recommended_strategy"] == "breakout" or
            r_result["recommended_strategy"] == "mean_reversion",
        )


class TestEntropyFilter(unittest.TestCase):
    def test_insufficient_data_allows_trade(self):
        ef = EntropyFilter(window=50)
        should, entropy = ef.should_trade()
        self.assertTrue(should)

    def test_random_returns_high_entropy(self):
        import random
        rng = random.Random(42)
        ef = EntropyFilter(window=50, entropy_threshold=0.7)
        for _ in range(60):
            ef.record_return(rng.gauss(0, 0.01))
        should, entropy = ef.should_trade()
        # Random returns should have high entropy
        self.assertGreater(entropy, 0.3)

    def test_structured_returns_lower_entropy(self):
        ef = EntropyFilter(window=50, entropy_threshold=0.9)
        # Very structured: all positive
        for _ in range(60):
            ef.record_return(0.01)
        should, entropy = ef.should_trade()
        # All same value → very low entropy
        self.assertLess(entropy, 0.3)

    def test_signal_redundancy_independent(self):
        import random
        rng = random.Random(42)
        ef = EntropyFilter()
        for _ in range(30):
            ef.record_signal("source_a", rng.random())
            ef.record_signal("source_b", rng.random())
        redundancy = ef.signal_redundancy("source_a", "source_b")
        self.assertLess(redundancy, 0.5)  # independent signals

    def test_signal_redundancy_correlated(self):
        ef = EntropyFilter()
        for i in range(30):
            val = float(i) / 30
            ef.record_signal("source_a", val)
            ef.record_signal("source_b", val + 0.001)  # nearly identical
        redundancy = ef.signal_redundancy("source_a", "source_b")
        self.assertGreater(redundancy, 0.3)  # highly correlated

    def test_get_stats(self):
        ef = EntropyFilter()
        ef.record_return(0.01)
        stats = ef.get_stats()
        self.assertIn("should_trade", stats)
        self.assertIn("entropy", stats)


class TestMetaObservation(unittest.TestCase):
    def test_score_computation(self):
        config = HyperparamConfig()
        obs = MetaObservation(
            config=config,
            evolver_best_fitness=0.8,
            generator_best_fitness=0.6,
            live_avg_sharpe=0.5,
            discovery_rate=0.3,
            promotion_success_rate=0.4,
        )
        self.assertGreater(obs.score, 0)

    def test_zero_score_for_empty(self):
        obs = MetaObservation(config=HyperparamConfig())
        self.assertAlmostEqual(obs.score, 0.0)


if __name__ == "__main__":
    unittest.main()
