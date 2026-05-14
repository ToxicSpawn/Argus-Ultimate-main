"""ML Integration tests — validates regime router, ensemble, position sizing, signal fusion.

Tests:
  - RegimeStrategyRouter (weights, performance tracking, adaptive)
  - EnsemblePredictor (multi-model combination, dynamic weights)
  - UncertaintyPositionSizer (Kelly, uncertainty adjustment)
  - SignalFusion (multi-source fusion, conflict detection)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# RegimeStrategyRouter (5 tests)
# ---------------------------------------------------------------------------

class TestRegimeStrategyRouter:
    def test_get_weights_for_known_regime(self):
        from ml.regime_strategy_router import RegimeStrategyRouter
        router = RegimeStrategyRouter()
        
        weights = router.get_strategy_weights("TREND_UP", confidence=0.8)
        
        assert weights.regime == "TREND_UP"
        assert weights.confidence == 0.8
        assert "trend_following" in weights.weights
        assert abs(sum(weights.weights.values()) - 1.0) < 0.01  # Normalized
    
    def test_unknown_regime_fallback(self):
        from ml.regime_strategy_router import RegimeStrategyRouter
        router = RegimeStrategyRouter()
        
        weights = router.get_strategy_weights("UNKNOWN_REGIME")
        
        assert weights.method == "fallback"
        assert sum(weights.weights.values()) > 0
    
    def test_position_multiplier_crisis(self):
        from ml.regime_strategy_router import RegimeStrategyRouter
        router = RegimeStrategyRouter()
        
        multiplier = router.get_position_multiplier("CRISIS", confidence=0.9)
        
        # Crisis should reduce position size
        assert multiplier < 0.5
    
    def test_performance_tracking(self):
        from ml.regime_strategy_router import RegimeStrategyRouter
        router = RegimeStrategyRouter()
        
        router.update_performance("trend_following", "TREND_UP", pnl=0.02)
        router.update_performance("trend_following", "TREND_UP", pnl=-0.01)
        
        perf = router.get_regime_performance("TREND_UP")
        assert len(perf) == 1
        assert perf[0].n_trades == 2
    
    def test_adaptive_weights(self):
        from ml.regime_strategy_router import RegimeStrategyRouter
        router = RegimeStrategyRouter(adaptive=True, min_trades_for_adaptation=5)
        
        # Build up performance data
        for _ in range(10):
            router.update_performance("trend_following", "TREND_UP", pnl=0.01)
            router.update_performance("momentum", "TREND_UP", pnl=-0.005)
        
        weights = router.get_strategy_weights("TREND_UP")
        
        # trend_following should have higher weight due to better performance
        assert weights.method == "adaptive"


class TestMLStrategyBridgeGating:
    def test_low_confidence_decision_is_gated_to_hold(self):
        from ml.regime_strategy_router import MLStrategyBridge

        bridge = MLStrategyBridge(min_trade_confidence=0.7)
        decision = bridge.make_decision(
            regime="TREND_UP",
            regime_confidence=0.2,
            price_prediction={"direction": "up", "confidence": 0.2},
        )

        assert decision["action"] == "hold"
        assert decision["position_multiplier"] == 0.0
        assert decision["gate"]["reason"] == "confidence_below_threshold"

    def test_high_confidence_decision_keeps_trade_action(self):
        from ml.regime_strategy_router import MLStrategyBridge

        bridge = MLStrategyBridge(min_trade_confidence=0.5)
        decision = bridge.make_decision(
            regime="TREND_UP",
            regime_confidence=0.95,
            price_prediction={"direction": "up", "confidence": 0.95},
        )

        assert decision["action"] == "buy"
        assert decision["position_multiplier"] > 0.0
        assert decision["gate"]["should_trade"] is True


# ---------------------------------------------------------------------------
# EnsemblePredictor (5 tests)
# ---------------------------------------------------------------------------

class TestEnsemblePredictor:
    def test_add_and_predict(self):
        from ml.ensemble_predictor import EnsemblePredictor
        
        ensemble = EnsemblePredictor()
        
        # Add simple models
        ensemble.add_model("model_a", lambda x: x * 1.0, weight=0.5)
        ensemble.add_model("model_b", lambda x: x * 1.1, weight=0.5)
        
        features = np.array([1.0, 2.0, 3.0])
        result = ensemble.predict(features)
        
        assert len(result.predictions) == 3
        assert result.n_models_used == 2
    
    def test_dynamic_weights(self):
        from ml.ensemble_predictor import EnsemblePredictor, CombinationMethod
        
        ensemble = EnsemblePredictor(method=CombinationMethod.DYNAMIC)
        ensemble.add_model("good", lambda x: x, weight=0.5)
        ensemble.add_model("bad", lambda x: x * 10, weight=0.5)
        
        features = np.array([1.0])
        target = np.array([1.0])
        
        # Update feedback multiple times
        for _ in range(20):
            ensemble.update_feedback(features, target)
        
        stats = ensemble.get_model_stats()
        # "good" should have lower error
        assert stats["good"]["avg_error"] < stats["bad"]["avg_error"]
    
    def test_model_failure_handling(self):
        from ml.ensemble_predictor import EnsemblePredictor
        
        ensemble = EnsemblePredictor(min_models=1)
        
        def failing_model(x):
            raise ValueError("Model failed")
        
        ensemble.add_model("failing", failing_model, weight=0.5)
        ensemble.add_model("working", lambda x: x, weight=0.5)
        
        features = np.array([1.0])
        result = ensemble.predict(features)
        
        # Should succeed with working model only
        assert result.n_models_used == 1
    
    def test_get_best_model(self):
        from ml.ensemble_predictor import EnsemblePredictor
        
        ensemble = EnsemblePredictor()
        ensemble.add_model("model_a", lambda x: x, weight=0.5)
        ensemble.add_model("model_b", lambda x: x, weight=0.5)
        
        # Run predictions to build history
        features = np.array([1.0])
        for _ in range(15):
            ensemble.predict(features)
        
        best = ensemble.get_best_model()
        assert best is not None
    
    def test_median_combination(self):
        from ml.ensemble_predictor import EnsemblePredictor, CombinationMethod
        
        ensemble = EnsemblePredictor(method=CombinationMethod.MEDIAN)
        ensemble.add_model("a", lambda x: np.array([1.0]), weight=1.0)
        ensemble.add_model("b", lambda x: np.array([3.0]), weight=1.0)
        ensemble.add_model("c", lambda x: np.array([5.0]), weight=1.0)
        
        result = ensemble.predict(np.array([0.0]))
        
        # Median of [1, 3, 5] = 3
        assert result.predictions[0] == 3.0


# ---------------------------------------------------------------------------
# UncertaintyPositionSizer (4 tests)
# ---------------------------------------------------------------------------

class TestUncertaintyPositionSizer:
    def test_high_confidence_low_uncertainty(self):
        from ml.position_sizing import UncertaintyPositionSizer
        
        sizer = UncertaintyPositionSizer()
        result = sizer.compute(
            prediction_confidence=0.9,
            uncertainty=0.1,
            base_equity=10000,
        )
        
        # High confidence + low uncertainty = larger position
        assert result.position_pct > 0.05
    
    def test_low_confidence_high_uncertainty(self):
        from ml.position_sizing import UncertaintyPositionSizer
        
        sizer = UncertaintyPositionSizer()
        result = sizer.compute(
            prediction_confidence=0.3,
            uncertainty=0.8,
            base_equity=10000,
        )
        
        # Low confidence + high uncertainty = smaller position
        assert result.position_pct < 0.05
    
    def test_regime_adjustment(self):
        from ml.position_sizing import UncertaintyPositionSizer
        
        sizer = UncertaintyPositionSizer()
        
        trend_result = sizer.compute_for_regime("TREND_UP", 0.8, 0.2, 10000)
        crisis_result = sizer.compute_for_regime("CRISIS", 0.8, 0.2, 10000)
        
        # Crisis should have smaller position than trend
        assert crisis_result.position_pct < trend_result.position_pct
    
    def test_kelly_computation(self):
        from ml.position_sizing import UncertaintyPositionSizer
        
        sizer = UncertaintyPositionSizer()
        result = sizer.compute(
            prediction_confidence=0.8,
            uncertainty=0.2,
            base_equity=10000,
            win_rate=0.6,
            win_loss_ratio=2.0,
        )
        
        assert result.kelly_fraction > 0
        assert result.position_usd > 0


# ---------------------------------------------------------------------------
# SignalFusion (5 tests)
# ---------------------------------------------------------------------------

class TestSignalFusion:
    def test_combine_all_sources(self):
        from ml.signal_fusion import SignalFusion
        
        fusion = SignalFusion()
        signal = fusion.combine(
            technical={"direction": 0.8, "strength": 0.7, "confidence": 0.9},
            sentiment={"score": 0.6, "confidence": 0.7},
            regime={"type": "TREND_UP", "confidence": 0.8},
            orderbook={"imbalance": 0.3, "depth_ratio": 1.2},
        )
        
        assert signal.direction > 0  # Bullish
        assert signal.action == "buy"
        assert signal.sources_used == 4
    
    def test_conflict_detection(self):
        from ml.signal_fusion import SignalFusion
        
        fusion = SignalFusion()
        signal = fusion.combine(
            technical={"direction": 0.9, "strength": 0.8, "confidence": 0.9},
            sentiment={"score": -0.8, "confidence": 0.9},  # Conflicting
        )
        
        assert signal.metadata.get("has_conflict") is True
        # Confidence should be reduced due to conflict
        assert signal.confidence < 0.8
    
    def test_hold_action(self):
        from ml.signal_fusion import SignalFusion
        
        fusion = SignalFusion()
        signal = fusion.combine(
            technical={"direction": 0.05, "strength": 0.1, "confidence": 0.3},
        )
        
        assert signal.action == "hold"
    
    def test_regime_direction(self):
        from ml.signal_fusion import SignalFusion
        
        fusion = SignalFusion()
        
        bull_signal = fusion.combine(regime={"type": "TREND_UP", "confidence": 0.9})
        bear_signal = fusion.combine(regime={"type": "TREND_DOWN", "confidence": 0.9})
        
        assert bull_signal.direction > 0
        assert bear_signal.direction < 0
    
    def test_signal_quality_scoring(self):
        from ml.signal_fusion import SignalFusion, SignalQualityScorer
        
        fusion = SignalFusion()
        signal = fusion.combine(
            technical={"direction": 0.8, "strength": 0.8, "confidence": 0.9},
            sentiment={"score": 0.6, "confidence": 0.8},
            regime={"type": "TREND_UP", "confidence": 0.9},
        )
        
        quality = SignalQualityScorer.score(signal)
        
        assert quality["overall_score"] > 50
        assert quality["recommendation"] in ("strong", "moderate")
