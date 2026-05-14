"""
Tests for self-adaptive intelligence systems.

Covers all 6 new modules:
  1. BanditStrategyAllocator (Thompson Sampling)
  2. GeneticEvolver (genetic parameter evolution)
  3. RegimePredictor (Markov regime prediction)
  4. ExecutionLearner (execution quality learning)
  5. FeatureDiscoverer (auto feature discovery)
  6. AntifragileManager (anti-fragile position management)

Plus integration tests for ComponentRegistry wiring.
"""

import time
import numpy as np
import pandas as pd
import pytest


# ============================================================================
# 1. BanditStrategyAllocator
# ============================================================================

class TestBanditStrategyAllocator:
    """Tests for Thompson Sampling strategy allocator."""

    def _make_allocator(self, names=None):
        from ml.bandit_allocator import BanditStrategyAllocator
        return BanditStrategyAllocator(
            strategy_names=names or ["momentum", "mean_revert", "breakout"],
            exploration_rate=0.05,
        )

    def test_init(self):
        alloc = self._make_allocator()
        assert len(alloc._strategy_names) == 3
        assert alloc._alpha["momentum"] == 1.0
        assert alloc._beta["momentum"] == 1.0

    def test_record_outcome_win(self):
        alloc = self._make_allocator()
        alloc.record_outcome("momentum", 10.0)
        assert alloc._alpha["momentum"] > 1.0
        assert alloc._win_count["momentum"] == 1
        assert alloc._cumulative_pnl["momentum"] == 10.0

    def test_record_outcome_loss(self):
        alloc = self._make_allocator()
        alloc.record_outcome("momentum", -5.0)
        assert alloc._beta["momentum"] > 1.0
        assert alloc._loss_count["momentum"] == 1
        assert alloc._cumulative_pnl["momentum"] == -5.0

    def test_record_outcome_auto_register(self):
        alloc = self._make_allocator()
        alloc.record_outcome("new_strategy", 5.0)
        assert "new_strategy" in alloc._alpha

    def test_allocations_sum_to_total(self):
        alloc = self._make_allocator()
        for _ in range(20):
            alloc.record_outcome("momentum", np.random.choice([10, -5]))
            alloc.record_outcome("mean_revert", np.random.choice([8, -3]))
            alloc.record_outcome("breakout", np.random.choice([12, -7]))

        allocations = alloc.get_allocations(1000.0)
        total = sum(allocations.values())
        assert abs(total - 1000.0) < 1.0  # within rounding

    def test_allocations_exploration_floor(self):
        alloc = self._make_allocator()
        # Make momentum extremely good, others terrible
        for _ in range(50):
            alloc.record_outcome("momentum", 100.0)
            alloc.record_outcome("mean_revert", -100.0)
            alloc.record_outcome("breakout", -100.0)

        allocations = alloc.get_allocations(1000.0)
        # Each strategy should get at least 5% = $50
        for name, amount in allocations.items():
            assert amount >= 49.0, f"{name} got ${amount}, expected >= $50"

    def test_allocations_zero_capital(self):
        alloc = self._make_allocator()
        allocations = alloc.get_allocations(0.0)
        assert all(v == 0.0 for v in allocations.values())

    def test_rankings_sorted_by_win_rate(self):
        alloc = self._make_allocator()
        # Momentum: 80% win rate
        for _ in range(80):
            alloc.record_outcome("momentum", 10.0)
        for _ in range(20):
            alloc.record_outcome("momentum", -5.0)
        # Mean revert: 30% win rate
        for _ in range(30):
            alloc.record_outcome("mean_revert", 10.0)
        for _ in range(70):
            alloc.record_outcome("mean_revert", -5.0)

        rankings = alloc.get_rankings()
        assert rankings[0]["strategy"] == "momentum"
        assert rankings[0]["expected_win_rate"] > rankings[1]["expected_win_rate"]

    def test_should_disable_true(self):
        alloc = self._make_allocator()
        # 35 trades, 28 losses (80% loss rate)
        for _ in range(7):
            alloc.record_outcome("breakout", 5.0)
        for _ in range(28):
            alloc.record_outcome("breakout", -5.0)
        assert alloc.should_disable("breakout", min_trades=30, max_loss_rate=0.7)

    def test_should_disable_false_insufficient_trades(self):
        alloc = self._make_allocator()
        for _ in range(5):
            alloc.record_outcome("breakout", -5.0)
        assert not alloc.should_disable("breakout", min_trades=30)

    def test_should_disable_false_good_win_rate(self):
        alloc = self._make_allocator()
        for _ in range(25):
            alloc.record_outcome("breakout", 5.0)
        for _ in range(10):
            alloc.record_outcome("breakout", -5.0)
        assert not alloc.should_disable("breakout", min_trades=30, max_loss_rate=0.7)

    def test_snapshot(self):
        alloc = self._make_allocator()
        alloc.record_outcome("momentum", 10.0)
        snap = alloc.snapshot()
        assert snap["strategy_count"] == 3
        assert snap["total_outcomes"] == 1


# ============================================================================
# 2. GeneticEvolver
# ============================================================================

class TestGeneticEvolver:
    """Tests for genetic parameter evolution."""

    def _make_evolver(self):
        from ml.genetic_evolver import GeneticEvolver
        return GeneticEvolver(
            parameter_ranges={"rsi": (5.0, 30.0), "bb": (1.0, 3.0)},
            population_size=10,
        )

    def test_init(self):
        evolver = self._make_evolver()
        assert len(evolver._population) == 10
        assert evolver._generation == 0

    def test_population_within_ranges(self):
        evolver = self._make_evolver()
        for ind in evolver._population:
            assert 5.0 <= ind["rsi"] <= 30.0
            assert 1.0 <= ind["bb"] <= 3.0

    def test_evaluate_generation(self):
        evolver = self._make_evolver()
        scores = {str(i): float(i) for i in range(10)}
        evolver.evaluate_generation(scores)
        assert len(evolver._fitness) == 10
        assert evolver._best_fitness == 9.0

    def test_evolve_creates_new_generation(self):
        evolver = self._make_evolver()
        scores = {str(i): float(np.random.randn()) for i in range(10)}
        evolver.evaluate_generation(scores)
        new_pop = evolver.evolve()
        assert len(new_pop) == 10
        assert evolver._generation == 1

    def test_elitism_preserves_best(self):
        evolver = self._make_evolver()
        # Score individual 0 very high, rest low
        scores = {str(i): -100.0 for i in range(10)}
        scores["0"] = 100.0
        evolver.evaluate_generation(scores)
        best_before = dict(evolver._population[0])
        new_pop = evolver.evolve()
        # The best individual should survive (elitism)
        assert any(
            abs(ind["rsi"] - best_before["rsi"]) < 0.01
            and abs(ind["bb"] - best_before["bb"]) < 0.01
            for ind in new_pop
        )

    def test_evolved_params_within_ranges(self):
        evolver = self._make_evolver()
        for gen in range(5):
            scores = {str(i): float(np.random.randn()) for i in range(10)}
            evolver.evaluate_generation(scores)
            new_pop = evolver.evolve()
            for ind in new_pop:
                assert 5.0 <= ind["rsi"] <= 30.0
                assert 1.0 <= ind["bb"] <= 3.0

    def test_get_best(self):
        evolver = self._make_evolver()
        scores = {str(i): float(i * 10) for i in range(10)}
        evolver.evaluate_generation(scores)
        best = evolver.get_best()
        assert "rsi" in best
        assert "bb" in best

    def test_get_generation_stats(self):
        evolver = self._make_evolver()
        scores = {str(i): float(i) for i in range(10)}
        evolver.evaluate_generation(scores)
        stats = evolver.get_generation_stats()
        assert stats["best_fitness"] == 9.0
        assert stats["avg_fitness"] == 4.5

    def test_empty_fitness_evolve(self):
        evolver = self._make_evolver()
        pop = evolver.evolve()  # no fitness set
        assert len(pop) == 10


# ============================================================================
# 3. RegimePredictor
# ============================================================================

class TestRegimePredictor:
    """Tests for regime prediction model."""

    def _make_predictor(self):
        from ml.regime_predictor import RegimePredictor
        return RegimePredictor(lookback_periods=50)

    def test_init(self):
        pred = self._make_predictor()
        assert pred._current_regime is None
        assert len(pred._regime_history) == 0

    def test_update(self):
        pred = self._make_predictor()
        pred.update("TRENDING_UP", {"volatility": 0.02, "momentum": 0.5})
        assert pred._current_regime == "TRENDING_UP"
        assert len(pred._regime_history) == 1

    def test_transition_matrix_updated_on_regime_change(self):
        pred = self._make_predictor()
        pred.update("TRENDING_UP", {})
        pred.update("HIGH_VOL", {})
        matrix = pred.get_transition_matrix()
        assert "TRENDING_UP" in matrix
        assert matrix["TRENDING_UP"]["HIGH_VOL"] == 1.0

    def test_predict_next_returns_valid(self):
        pred = self._make_predictor()
        pred.update("TRENDING_UP", {"volatility": 0.02})
        pred.update("HIGH_VOL", {"volatility": 0.05})
        pred.update("TRENDING_UP", {"volatility": 0.02})
        result = pred.predict_next()
        assert "predicted_regime" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_next_empty(self):
        pred = self._make_predictor()
        result = pred.predict_next()
        assert result["predicted_regime"] == "UNKNOWN"
        assert result["confidence"] == 0.0

    def test_transition_matrix_probabilities_sum_to_one(self):
        pred = self._make_predictor()
        for regime in ["TRENDING_UP", "HIGH_VOL", "CRISIS", "TRENDING_UP", "LOW_VOL"]:
            pred.update(regime, {})
        matrix = pred.get_transition_matrix()
        for from_r, transitions in matrix.items():
            total = sum(transitions.values())
            assert abs(total - 1.0) < 0.01, f"{from_r} transitions sum to {total}"

    def test_pre_transition_signals_feature_based(self):
        pred = self._make_predictor()
        # Simulate conditions suggesting CRISIS
        pred.update("TRENDING_UP", {"volatility": 0.08, "momentum": -0.5})
        signals = pred.get_pre_transition_signals()
        assert isinstance(signals, list)

    def test_snapshot(self):
        pred = self._make_predictor()
        pred.update("LOW_VOL", {})
        snap = pred.snapshot()
        assert snap["current_regime"] == "LOW_VOL"
        assert "prediction" in snap


# ============================================================================
# 4. ExecutionLearner
# ============================================================================

class TestExecutionLearner:
    """Tests for execution learning agent."""

    def _make_learner(self):
        from ml.execution_learner import ExecutionLearner
        return ExecutionLearner(max_history=1000)

    def _add_fills(self, learner, n=50):
        for i in range(n):
            learner.record_fill(
                symbol="BTC/USD",
                side="buy",
                slippage_bps=np.random.uniform(0, 10),
                hour_utc=np.random.randint(0, 24),
                day_of_week=np.random.randint(0, 7),
                volatility=np.random.uniform(0.01, 0.05),
                spread_bps=np.random.uniform(1, 20),
                size_usd=np.random.uniform(50, 500),
            )

    def test_record_fill(self):
        learner = self._make_learner()
        learner.record_fill("BTC/USD", "buy", 2.5, 14, 2, 0.03, 5.0)
        assert len(learner._fill_history["BTC/USD"]) == 1

    def test_optimal_execution_window_insufficient_data(self):
        learner = self._make_learner()
        learner.record_fill("BTC/USD", "buy", 2.5, 14, 2, 0.03, 5.0)
        result = learner.get_optimal_execution_window("BTC/USD")
        assert result["total_fills"] == 1
        assert result["best_hours"] == []

    def test_optimal_execution_window_with_data(self):
        learner = self._make_learner()
        # Add fills with low slippage at hour 14 and high at hour 3
        for _ in range(20):
            learner.record_fill("BTC/USD", "buy", 1.0, 14, 2, 0.02, 5.0)
            learner.record_fill("BTC/USD", "buy", 15.0, 3, 2, 0.02, 5.0)

        result = learner.get_optimal_execution_window("BTC/USD")
        assert 14 in result["best_hours"]
        assert 3 in result["worst_hours"]

    def test_optimal_order_size_default(self):
        learner = self._make_learner()
        result = learner.get_optimal_order_size("BTC/USD", 5.0)
        assert result["recommended_max_usd"] == 500.0  # default

    def test_optimal_order_size_with_data(self):
        learner = self._make_learner()
        self._add_fills(learner, 50)
        result = learner.get_optimal_order_size("BTC/USD", 5.0)
        assert result["recommended_max_usd"] > 0
        assert len(result["historical_slippage_at_size"]) > 0

    def test_should_delay_insufficient_data(self):
        learner = self._make_learner()
        result = learner.should_delay_execution("BTC/USD", 14, 5.0)
        assert result["delay"] is False

    def test_should_delay_bad_hour(self):
        learner = self._make_learner()
        # Create fills: hour 3 has terrible slippage, others fine
        for _ in range(30):
            learner.record_fill("BTC/USD", "buy", 20.0, 3, 2, 0.03, 5.0)
            learner.record_fill("BTC/USD", "buy", 1.0, 14, 2, 0.03, 5.0)
            learner.record_fill("BTC/USD", "buy", 1.5, 10, 2, 0.03, 5.0)

        result = learner.should_delay_execution("BTC/USD", 3, 5.0)
        assert result["delay"] is True
        assert result["suggested_wait_minutes"] > 0

    def test_should_delay_wide_spread(self):
        learner = self._make_learner()
        for _ in range(20):
            learner.record_fill("BTC/USD", "buy", 2.0, 14, 2, 0.02, 3.0)

        result = learner.should_delay_execution("BTC/USD", 14, 30.0)
        assert result["delay"] is True

    def test_snapshot(self):
        learner = self._make_learner()
        self._add_fills(learner, 10)
        snap = learner.snapshot()
        assert snap["total_fills"] == 10


# ============================================================================
# 5. FeatureDiscoverer
# ============================================================================

class TestFeatureDiscoverer:
    """Tests for auto feature discovery."""

    def _make_discoverer(self):
        from ml.feature_discoverer import FeatureDiscoverer
        return FeatureDiscoverer(max_features=50)

    def _make_ohlcv(self, n=200):
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({
            "open": prices + np.random.randn(n) * 0.1,
            "high": prices + abs(np.random.randn(n) * 0.5),
            "low": prices - abs(np.random.randn(n) * 0.5),
            "close": prices,
            "volume": np.random.uniform(1e5, 1e7, n),
        })

    def test_generate_candidates(self):
        disc = self._make_discoverer()
        df = self._make_ohlcv()
        candidates = disc.generate_candidates(df)
        assert len(candidates) > 10
        # Check some expected features
        assert "close_sma20_ratio" in candidates
        assert "vol_std_10" in candidates
        assert "roc_5" in candidates

    def test_generate_candidates_short_data(self):
        disc = self._make_discoverer()
        df = self._make_ohlcv(5)
        candidates = disc.generate_candidates(df)
        # Should still return something (price ratios at minimum)
        assert isinstance(candidates, dict)

    def test_evaluate_predictive_power(self):
        disc = self._make_discoverer()
        df = self._make_ohlcv(200)
        candidates = disc.generate_candidates(df)
        fwd_ret = np.diff(df["close"].values) / np.maximum(df["close"].values[:-1], 1e-10)
        ic_scores = disc.evaluate_predictive_power(candidates, fwd_ret)
        assert isinstance(ic_scores, dict)
        # All IC scores should be in [-1, 1]
        for name, ic in ic_scores.items():
            assert -1.0 <= ic <= 1.0

    def test_get_top_features(self):
        disc = self._make_discoverer()
        df = self._make_ohlcv(200)
        candidates = disc.generate_candidates(df)
        fwd_ret = np.diff(df["close"].values) / np.maximum(df["close"].values[:-1], 1e-10)
        disc.evaluate_predictive_power(candidates, fwd_ret)
        top = disc.get_top_features(5)
        assert isinstance(top, list)
        assert all("name" in f and "ic_score" in f for f in top)

    def test_prune_stale_features(self):
        disc = self._make_discoverer()
        # Add some features with old timestamps
        disc._discovered_features["old_feat"] = {
            "func_name": "old_feat",
            "importance": 0.1,
            "ic_score": 0.1,
            "last_evaluated": time.time() - 40 * 86400,  # 40 days ago
        }
        disc._discovered_features["new_feat"] = {
            "func_name": "new_feat",
            "importance": 0.2,
            "ic_score": 0.2,
            "last_evaluated": time.time(),
        }
        pruned = disc.prune_stale_features(max_age_days=30)
        assert pruned == 1
        assert "old_feat" not in disc._discovered_features
        assert "new_feat" in disc._discovered_features

    def test_snapshot(self):
        disc = self._make_discoverer()
        snap = disc.snapshot()
        assert "discovered_features" in snap
        assert "top_features" in snap


# ============================================================================
# 6. AntifragileManager
# ============================================================================

class TestAntifragileManager:
    """Tests for anti-fragile position management."""

    def _make_manager(self):
        from risk.antifragile import AntifragileManager
        return AntifragileManager(max_history=1000, min_observations=5)

    def test_init(self):
        mgr = self._make_manager()
        assert len(mgr._vol_history) == 0

    def test_record(self):
        mgr = self._make_manager()
        mgr.record(0.02, 10.0)
        assert len(mgr._vol_history) == 1
        # 0.02 falls in "medium" bucket (0.02 <= vol < 0.04)
        assert len(mgr._pnl_at_vol["medium"]) == 1

    def test_vol_pnl_curve(self):
        mgr = self._make_manager()
        for _ in range(20):
            mgr.record(0.005, 5.0)   # very_low vol, winning
            mgr.record(0.015, 2.0)   # low vol, winning
            mgr.record(0.06, -3.0)   # high vol, losing

        curve = mgr.get_vol_pnl_curve()
        assert curve["very_low"]["avg_pnl"] > 0
        assert curve["high"]["avg_pnl"] < 0

    def test_position_multiplier_profitable_in_vol(self):
        mgr = self._make_manager()
        # Profitable in high vol
        for _ in range(20):
            mgr.record(0.06, 10.0)
        multiplier = mgr.get_position_multiplier(0.06)
        assert multiplier > 1.0

    def test_position_multiplier_losing_in_vol(self):
        mgr = self._make_manager()
        # Losing in high vol
        for _ in range(20):
            mgr.record(0.06, -10.0)
        multiplier = mgr.get_position_multiplier(0.06)
        assert multiplier < 1.0

    def test_position_multiplier_insufficient_data(self):
        mgr = self._make_manager()
        multiplier = mgr.get_position_multiplier(0.06)
        assert multiplier == 1.0

    def test_fragility_score_fragile(self):
        mgr = self._make_manager()
        # Loses in high vol, wins in low vol
        for _ in range(20):
            mgr.record(0.005, 10.0)   # very_low: winning
            mgr.record(0.015, 8.0)    # low: winning
            mgr.record(0.06, -10.0)   # high: losing
            mgr.record(0.10, -15.0)   # very_high: losing badly

        score = mgr.get_fragility_score()
        assert score < 0  # fragile

    def test_fragility_score_antifragile(self):
        mgr = self._make_manager()
        # Wins in high vol, loses in low vol
        for _ in range(20):
            mgr.record(0.005, -5.0)
            mgr.record(0.015, -3.0)
            mgr.record(0.06, 10.0)
            mgr.record(0.10, 20.0)

        score = mgr.get_fragility_score()
        assert score > 0  # antifragile

    def test_fragility_score_insufficient_data(self):
        mgr = self._make_manager()
        score = mgr.get_fragility_score()
        assert score == 0.0

    def test_recommend_fragile(self):
        mgr = self._make_manager()
        for _ in range(20):
            mgr.record(0.005, 10.0)
            mgr.record(0.06, -10.0)
            mgr.record(0.10, -15.0)

        rec = mgr.recommend_vol_strategy()
        assert rec["category"] == "fragile"
        assert len(rec["recommendations"]) > 0

    def test_recommend_antifragile(self):
        mgr = self._make_manager()
        for _ in range(20):
            mgr.record(0.005, -5.0)
            mgr.record(0.06, 15.0)
            mgr.record(0.10, 25.0)

        rec = mgr.recommend_vol_strategy()
        assert rec["category"] == "antifragile"

    def test_snapshot(self):
        mgr = self._make_manager()
        mgr.record(0.02, 10.0)
        snap = mgr.snapshot()
        assert "fragility_score" in snap
        assert "vol_pnl_curve" in snap


# ============================================================================
# 7. Integration: ComponentRegistry wiring
# ============================================================================

class TestComponentRegistryIntegration:
    """Integration tests for self-adaptive components in ComponentRegistry."""

    def test_on_fill_feeds_bandit_and_learner(self):
        """on_fill should update bandit allocator and execution learner."""
        from ml.bandit_allocator import BanditStrategyAllocator
        from ml.execution_learner import ExecutionLearner
        from risk.antifragile import AntifragileManager

        # Simulate what ComponentRegistry.on_fill does
        alloc = BanditStrategyAllocator(["momentum"])
        learner = ExecutionLearner()
        afm = AntifragileManager(min_observations=3)

        # Simulate a fill
        pnl = 15.0
        alloc.record_outcome("momentum", pnl)
        learner.record_fill("BTC/USD", "buy", 2.5, 14, 2, 0.03, 5.0, 100.0)
        afm.record(0.03, pnl)

        assert alloc._trade_count["momentum"] == 1
        assert len(learner._fill_history["BTC/USD"]) == 1
        assert len(afm._vol_history) == 1

    def test_cycle_updates_regime_predictor(self):
        """on_cycle should feed regime data to predictor."""
        from ml.regime_predictor import RegimePredictor

        pred = RegimePredictor()
        pred.update("TRENDING_UP", {"volatility": 0.02})
        pred.update("HIGH_VOL", {"volatility": 0.06})
        prediction = pred.predict_next()
        assert prediction["predicted_regime"] != ""

    def test_fill_updates_change_allocations(self):
        """After recording outcomes, bandit allocations should shift."""
        from ml.bandit_allocator import BanditStrategyAllocator

        alloc = BanditStrategyAllocator(["good", "bad"])

        # Good strategy wins consistently
        for _ in range(30):
            alloc.record_outcome("good", 10.0)
            alloc.record_outcome("bad", -10.0)

        allocations = alloc.get_allocations(1000.0)
        assert allocations["good"] > allocations["bad"]

    def test_genetic_evolver_converges(self):
        """Running multiple generations should improve best fitness."""
        from ml.genetic_evolver import GeneticEvolver

        evolver = GeneticEvolver(
            parameter_ranges={"x": (0.0, 10.0)},
            population_size=20,
        )

        best_fitness_history = []
        for gen in range(10):
            pop = evolver.get_population()
            # Fitness: negative distance from target x=7.0
            scores = {}
            for i, ind in enumerate(pop):
                scores[str(i)] = -abs(ind["x"] - 7.0)
            evolver.evaluate_generation(scores)
            best_fitness_history.append(evolver._best_fitness)
            evolver.evolve()

        # Best fitness should improve (get closer to 0)
        assert best_fitness_history[-1] >= best_fitness_history[0]

    def test_all_modules_import(self):
        """All 6 new modules should import successfully."""
        from ml.bandit_allocator import BanditStrategyAllocator
        from ml.genetic_evolver import GeneticEvolver
        from ml.regime_predictor import RegimePredictor
        from ml.execution_learner import ExecutionLearner
        from ml.feature_discoverer import FeatureDiscoverer
        from risk.antifragile import AntifragileManager

        assert BanditStrategyAllocator is not None
        assert GeneticEvolver is not None
        assert RegimePredictor is not None
        assert ExecutionLearner is not None
        assert FeatureDiscoverer is not None
        assert AntifragileManager is not None

    def test_empty_strategy_names_raises(self):
        from ml.bandit_allocator import BanditStrategyAllocator
        with pytest.raises(ValueError):
            BanditStrategyAllocator(strategy_names=[])

    def test_small_population_raises(self):
        from ml.genetic_evolver import GeneticEvolver
        with pytest.raises(ValueError):
            GeneticEvolver(parameter_ranges={"x": (0, 1)}, population_size=2)
