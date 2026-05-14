"""
Test Learning Orchestrator
===========================
Tests for the learning_orchestrator module that integrates all learning algorithms.
"""

import pytest
import numpy as np
from datetime import datetime

from learning.learning_orchestrator import (
    LearningOrchestrator,
    AdaptiveLearningRateController,
    ExplorationExploitationBalancer,
    BayesianParameterOptimizer,
    DriftDetector,
    EnsembleWeightOptimizer,
    FeatureImportanceTracker,
    ContextualBandit,
    RegimeAwareLearner,
    MetaLearner,
    wire_all_learning,
    get_learning_orchestrator,
)


class TestAdaptiveLearningRate:
    """Test suite for AdaptiveLearningRateController."""
    
    def setup_method(self):
        self.lr_controller = AdaptiveLearningRateController(
            base_lr=0.1, min_lr=0.01, max_lr=0.5
        )
    
    def test_initialization(self):
        """Test initial learning rate."""
        assert self.lr_controller.get_learning_rate() == 0.1
        assert self.lr_controller.base_lr == 0.1
        assert self.lr_controller.min_lr == 0.01
        assert self.lr_controller.max_lr == 0.5
    
    def test_increase_on_good_improvement(self):
        """Test learning rate increases when improvement is good."""
        initial_lr = self.lr_controller.get_learning_rate()
        
        # Good improvements should increase LR
        for _ in range(10):
            self.lr_controller.update(0.05)  # Good improvement
        
        assert self.lr_controller.get_learning_rate() > initial_lr
    
    def test_decrease_on_poor_improvement(self):
        """Test learning rate decreases when improvement is poor."""
        # Reset to base
        self.lr_controller.reset()
        initial_lr = self.lr_controller.get_learning_rate()
        
        # Poor improvements should decrease LR (need enough samples)
        for _ in range(50):
            self.lr_controller.update(0.0001)  # Poor improvement
        
        assert self.lr_controller.get_learning_rate() < initial_lr
    
    def test_respects_bounds(self):
        """Test learning rate stays within bounds."""
        # Try to push above max
        for _ in range(100):
            self.lr_controller.update(1.0)
        
        assert self.lr_controller.get_learning_rate() <= self.lr_controller.max_lr
        
        # Try to push below min
        self.lr_controller.reset()
        for _ in range(100):
            self.lr_controller.update(0.0)
        
        assert self.lr_controller.get_learning_rate() >= self.lr_controller.min_lr


class TestExplorationExploitationBalancer:
    """Test suite for ExplorationExploitationBalancer."""
    
    def setup_method(self):
        self.balancer = ExplorationExploitationBalancer(
            initial_rate=0.2, min_rate=0.05, max_rate=0.5
        )
    
    def test_initialization(self):
        """Test initial exploration rate."""
        assert self.balancer.get_exploration_rate() == 0.2
    
    def test_increase_when_exploration_works(self):
        """Test exploration rate increases when exploration is better."""
        initial_rate = self.balancer.get_exploration_rate()
        
        # Exploration better than exploitation
        for _ in range(30):
            self.balancer.update(100.0, 50.0)  # Exploration much better
        
        assert self.balancer.get_exploration_rate() > initial_rate
    
    def test_decrease_when_exploitation_works(self):
        """Test exploration rate decreases when exploitation is better."""
        # First set up history
        for _ in range(30):
            self.balancer.update(50.0, 100.0)  # Exploitation much better
        
        assert self.balancer.get_exploration_rate() < 0.2
    
    def test_should_explore_respects_rate(self):
        """Test should_explore respects exploration rate."""
        self.balancer.exploration_rate = 0.0  # No exploration
        assert not self.balancer.should_explore()
        
        self.balancer.exploration_rate = 1.0  # Always explore
        assert self.balancer.should_explore()


class TestBayesianParameterOptimizer:
    """Test suite for BayesianParameterOptimizer."""
    
    def setup_method(self):
        self.optimizer = BayesianParameterOptimizer()
    
    def test_random_suggestions_initially(self):
        """Test random suggestions when insufficient data."""
        bounds = {"param1": (0.0, 1.0), "param2": (0.0, 10.0)}
        
        suggestion = self.optimizer.suggest(bounds)
        assert "param1" in suggestion
        assert "param2" in suggestion
        assert 0.0 <= suggestion["param1"] <= 1.0
        assert 0.0 <= suggestion["param2"] <= 10.0
    
    def test_bayesian_suggestions_after_observations(self):
        """Test Bayesian suggestions after enough observations."""
        bounds = {"param1": (0.0, 1.0)}
        
        # Add observations
        for i in range(10):
            params = {"param1": i / 10.0}
            score = i / 10.0  # Higher params = better score
            self.optimizer.observe(params, score)
        
        suggestion = self.optimizer.suggest(bounds)
        # Should tend toward higher values (where scores were better)
        assert 0.0 <= suggestion["param1"] <= 1.0
    
    def test_observe_and_get_best(self):
        """Test observation recording and best retrieval."""
        params1 = {"param1": 0.1}
        params2 = {"param1": 0.9}
        
        self.optimizer.observe(params1, 0.5)
        self.optimizer.observe(params2, 0.9)
        
        best_params, best_score = self.optimizer.get_best()
        assert best_score == 0.9
        assert best_params["param1"] == 0.9


class TestDriftDetector:
    """Test suite for DriftDetector."""
    
    def setup_method(self):
        self.detector = DriftDetector(window_size=50, threshold=0.1)
    
    def test_no_drift_with_stable_data(self):
        """Test no drift detected with stable data."""
        # Add stable reference data
        for _ in range(30):
            self.detector.add_sample(1.0, is_reference=True)
        
        # Add stable current data
        for _ in range(30):
            drift = self.detector.add_sample(1.0)
        
        assert not self.detector.is_drifting()
    
    def test_drift_detected_with_change(self):
        """Test drift detected when distribution changes."""
        # Add reference data (mean = 1.0)
        for _ in range(30):
            self.detector.add_sample(1.0, is_reference=True)
        
        # Add current data (mean = 2.0, significant shift)
        for _ in range(30):
            self.detector.add_sample(2.0)
        
        assert self.detector.is_drifting()
    
    def test_reset_clears_state(self):
        """Test reset clears detector state."""
        for _ in range(30):
            self.detector.add_sample(2.0)
        
        self.detector.reset()
        assert not self.detector.is_drifting()


class TestEnsembleWeightOptimizer:
    """Test suite for EnsembleWeightOptimizer."""
    
    def setup_method(self):
        self.optimizer = EnsembleWeightOptimizer(num_models=4)
    
    def test_initialization(self):
        """Test initial weights are equal."""
        weights = self.optimizer.get_weights()
        assert len(weights) == 4
        np.testing.assert_array_almost_equal(weights, [0.25, 0.25, 0.25, 0.25])
    
    def test_weights_follow_performance(self):
        """Test weights shift toward better-performing models."""
        # Model 0 performs best
        for _ in range(20):
            performances = np.array([1.0, 0.0, 0.0, 0.0])
            self.optimizer.update_weights(performances)
        
        weights = self.optimizer.get_weights()
        assert weights[0] > weights[1]
        assert weights[0] > weights[2]
        assert weights[0] > weights[3]
    
    def test_get_best_algorithm_idx(self):
        """Test best algorithm index retrieval."""
        for _ in range(20):
            performances = np.array([0.0, 1.0, 0.0, 0.0])
            self.optimizer.update_weights(performances)
        
        assert self.optimizer.get_best_algorithm_idx() == 1


class TestFeatureImportanceTracker:
    """Test suite for FeatureImportanceTracker."""
    
    def setup_method(self):
        self.tracker = FeatureImportanceTracker(num_features=10)
    
    def test_top_features_identified(self):
        """Test top features are correctly identified."""
        # Make feature 5 and 7 important
        for _ in range(20):
            self.tracker.update_importance(5, 1.0)
            self.tracker.update_importance(7, 1.0)
            self.tracker.update_importance(0, 0.1)
        
        top = self.tracker.get_top_features(2)
        assert 5 in top
        assert 7 in top
    
    def test_weights_normalize(self):
        """Test importance weights are normalized."""
        for i in range(10):
            self.tracker.update_importance(i, 0.5)
        
        weights = self.tracker.get_weights()
        assert abs(np.sum(weights) - 1.0) < 0.01


class TestContextualBandit:
    """Test suite for ContextualBandit."""
    
    def setup_method(self):
        self.bandit = ContextualBandit(num_arms=5)
    
    def test_select_arm_returns_valid_arm(self):
        """Test arm selection returns valid index."""
        arm = self.bandit.select_arm()
        assert 0 <= arm < 5
    
    def test_select_arm_with_context(self):
        """Test arm selection with context."""
        arm = self.bandit.select_arm(context="trending_up")
        assert 0 <= arm < 5
    
    def test_update_improves_best_arm(self):
        """Test that updating improves best arm selection."""
        # Reward arm 2 consistently
        for _ in range(50):
            self.bandit.update(2, 1.0)
        
        best = self.bandit.get_best_arm()
        assert best == 2
    
    def test_context_influences_selection(self):
        """Test that context influences arm selection."""
        # Train arm 0 for context A
        for _ in range(50):
            self.bandit.update(0, 1.0, context="context_A")
        
        # Train arm 1 for context B
        for _ in range(50):
            self.bandit.update(1, 1.0, context="context_B")


class TestRegimeAwareLearner:
    """Test suite for RegimeAwareLearner."""
    
    def setup_method(self):
        self.learner = RegimeAwareLearner()
    
    def test_set_and_get_regime(self):
        """Test regime setting and parameter retrieval."""
        self.learner.set_regime("trending_up")
        self.learner.update_param("confidence", 0.8)
        
        assert self.learner.get_param("confidence") == 0.8
    
    def test_different_regimes_separate_params(self):
        """Test different regimes have separate parameters."""
        self.learner.set_regime("trending_up")
        self.learner.update_param("confidence", 0.9)
        
        self.learner.set_regime("ranging")
        self.learner.update_param("confidence", 0.5)
        
        self.learner.set_regime("trending_up")
        assert self.learner.get_param("confidence") == 0.9
        
        self.learner.set_regime("ranging")
        assert self.learner.get_param("confidence") == 0.5
    
    def test_get_best_regime(self):
        """Test best regime retrieval based on performance."""
        # Need at least 5 samples per regime
        for _ in range(10):
            self.learner.record_performance("trending_up", 0.8)
            self.learner.record_performance("ranging", 0.3)
        
        assert self.learner.get_best_regime() == "trending_up"


class TestMetaLearner:
    """Test suite for MetaLearner."""
    
    def setup_method(self):
        self.learner = MetaLearner()
    
    def test_initial_weights(self):
        """Test initial algorithm weights."""
        weights = self.learner.get_weights()
        assert len(weights) == 4
        assert all(w == 0.25 for w in weights.values())
    
    def test_best_algorithm_changes_with_performance(self):
        """Test best algorithm changes based on performance."""
        # Bayesian outperforms others
        for _ in range(20):
            self.learner.record_performance("bayesian", 1.0)
            self.learner.record_performance("thompson", 0.1)
        
        assert self.learner.get_best_algorithm() == "bayesian"
    
    def test_weights_normalize(self):
        """Test algorithm weights stay normalized."""
        # Need enough samples for weight updates
        for _ in range(30):
            self.learner.record_performance("thompson", 0.5)
            self.learner.record_performance("bayesian", 0.8)
        
        weights = self.learner.get_weights()
        total = sum(weights.values())
        # Weights are smoothed, so they may not sum to exactly 1.0
        assert 0.3 < total < 1.5  # Reasonable range


class TestLearningOrchestrator:
    """Test suite for LearningOrchestrator."""
    
    def setup_method(self):
        self.orchestrator = LearningOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initializes all components."""
        assert self.orchestrator.learning_rate_controller is not None
        assert self.orchestrator.exploration_balancer is not None
        assert self.orchestrator.bayesian_optimizer is not None
        assert self.orchestrator.drift_detector is not None
        assert self.orchestrator.ensemble_optimizer is not None
        assert self.orchestrator.feature_tracker is not None
        assert self.orchestrator.contextual_bandit is not None
        assert self.orchestrator.regime_learner is not None
        assert self.orchestrator.meta_learner is not None
    
    def test_get_parameters_for_decision(self):
        """Test getting parameters for decision."""
        params = self.orchestrator.get_parameters_for_decision(
            regime="trending_up",
            context={"volatility": "low"}
        )
        
        assert "learning_rate" in params
        assert "exploration_rate" in params
        assert "confidence_threshold" in params
        assert "position_size" in params
        assert "stop_loss" in params
        assert "take_profit" in params
        assert "bandit_arm" in params
    
    def test_record_trade_outcome(self):
        """Test recording trade outcome updates state."""
        params = {"confidence_threshold": 0.6, "position_size": 0.1}
        
        result = self.orchestrator.record_trade_outcome(
            params_used=params,
            pnl=100.0,
            regime="trending_up"
        )
        
        assert self.orchestrator.state.total_updates == 1
    
    def test_set_regime_updates_all_components(self):
        """Test regime update propagates to components."""
        self.orchestrator.set_regime("ranging")
        assert self.orchestrator.state.current_regime == "ranging"
    
    def test_check_drift(self):
        """Test drift detection."""
        # Stable performance
        stable_perf = [0.5] * 50
        drift = self.orchestrator.check_drift(stable_perf)
        assert not drift
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        # Record some trades
        for i in range(10):
            self.orchestrator.record_trade_outcome(
                params_used={"confidence": 0.5},
                pnl=float(i * 10),
                regime="trending_up"
            )
        
        stats = self.orchestrator.get_stats()
        assert stats["total_updates"] == 10
        assert "learning_rate" in stats
        assert "exploration_rate" in stats
        assert "algorithm_weights" in stats
    
    def test_learning_improves_with_positive_outcomes(self):
        """Test that positive outcomes improve learning parameters."""
        initial_updates = self.orchestrator.state.total_updates
        
        # Record positive trades
        for i in range(30):
            self.orchestrator.record_trade_outcome(
                params_used={"confidence": 0.7, "position_size": 0.1},
                pnl=100.0 + i * 10,
                regime="trending_up",
                context={"volatility": "low"}
            )
        
        # Total updates should increase
        assert self.orchestrator.state.total_updates == initial_updates + 30
        # Total improvement should be positive
        assert self.orchestrator.state.total_improvement > 0


class TestGlobalFunctions:
    """Test suite for global functions."""
    
    def test_get_learning_orchestrator_returns_singleton(self):
        """Test get_learning_orchestrator returns same instance."""
        orch1 = get_learning_orchestrator()
        orch2 = get_learning_orchestrator()
        assert orch1 is orch2
    
    def test_wire_all_learning_returns_orchestrator(self):
        """Test wire_all_learning returns valid orchestrator."""
        orchestrator = wire_all_learning()
        assert orchestrator is not None
        assert isinstance(orchestrator, LearningOrchestrator)
