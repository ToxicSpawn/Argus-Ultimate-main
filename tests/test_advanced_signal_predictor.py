"""
test_advanced_signal_predictor.py
==================================
Tests for the Advanced Signal Predictor module.
"""

import unittest
import random
from ml.advanced_signal_predictor import (
    AdvancedSignalPredictor,
    FeatureEngineer,
    TransformerPredictor,
    GradientBoostedPredictor,
    MetaLabeler,
    OnlineLearner,
    MultiHorizonFusion,
    SignalType,
)


class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering."""
    
    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        prices = [50000.0 + i * 10 for i in range(100)]
        features = FeatureEngineer.extract_features(prices)
        
        self.assertIn('returns_1', features)
        self.assertIn('returns_5', features)
        self.assertIn('returns_20', features)
        self.assertIn('price_vs_ma5', features)
        self.assertIn('price_vs_ma20', features)
        self.assertIn('volatility_20', features)
    
    def test_extract_features_with_volumes(self):
        """Test feature extraction with volume data."""
        prices = [50000.0 + i * 10 for i in range(100)]
        volumes = [1000.0 + random.random() * 500 for _ in range(100)]
        
        features = FeatureEngineer.extract_features(prices, volumes=volumes)
        
        self.assertIn('volume_ratio', features)
        self.assertIn('volume_trend', features)
    
    def test_extract_features_insufficient_data(self):
        """Test with insufficient data."""
        prices = [50000.0, 50100.0, 50200.0]
        features = FeatureEngineer.extract_features(prices)
        
        self.assertEqual(features, {})


class TestTransformerPredictor(unittest.TestCase):
    """Test transformer predictor."""
    
    def test_predict_returns_prediction(self):
        """Test that predictor returns a valid prediction."""
        predictor = TransformerPredictor()
        features = {
            'returns_1': 0.001,
            'returns_5': 0.005,
            'returns_20': 0.02,
            'momentum_10': 0.015,
            'z_score_20': 0.5,
        }
        
        prediction = predictor.predict(features, horizon=5)
        
        self.assertEqual(prediction.model_name, "Transformer")
        self.assertEqual(prediction.horizon, 5)
        self.assertGreaterEqual(prediction.direction, -1.0)
        self.assertLessEqual(prediction.direction, 1.0)
        self.assertGreaterEqual(prediction.confidence, 0.0)
        self.assertLessEqual(prediction.confidence, 1.0)


class TestGradientBoostedPredictor(unittest.TestCase):
    """Test gradient boosted predictor."""
    
    def test_predict_returns_prediction(self):
        """Test that predictor returns a valid prediction."""
        predictor = GradientBoostedPredictor()
        features = {
            'returns_1': 0.001,
            'returns_5': 0.005,
            'returns_20': 0.02,
            'ma5_vs_ma20': 0.01,
            'rsi_14': 45.0,
            'z_score_20': 0.5,
            'volatility_20': 0.02,
            'momentum_10': 0.015,
            'momentum_strength': 0.6,
            'volume_ratio': 1.2,
            'vol_ratio': 1.0,
            'macd': 0.001,
        }
        
        prediction = predictor.predict(features, horizon=5)
        
        self.assertEqual(prediction.model_name, "GradientBoosted")
        self.assertGreaterEqual(prediction.direction, -1.0)
        self.assertLessEqual(prediction.direction, 1.0)


class TestMetaLabeler(unittest.TestCase):
    """Test meta-labeling system."""
    
    def test_calibrate_confidence(self):
        """Test confidence calibration."""
        labeler = MetaLabeler()
        
        from ml.advanced_signal_predictor import Prediction
        primary_pred = Prediction(
            direction=0.5,
            confidence=0.5,  # Lower base confidence to see increase
            horizon=5,
            model_name="Test",
            features_used=[],
        )
        
        features = {
            'vol_ratio': 0.4,  # Low volatility (< 0.5 triggers +0.1)
            'volume_ratio': 1.6,  # High volume (> 1.5 triggers +0.1)
            'momentum_strength': 0.8,  # Strong trend (> 0.8 triggers +0.05)
            'z_score_20': 2.6,  # Extreme z-score (> 2.5 triggers +0.1)
        }
        
        calibrated = labeler.calibrate_confidence(primary_pred, features)
        
        # Should increase confidence due to favorable conditions
        # Base 0.5 + adjustments (0.1 + 0.1 + 0.05 + 0.1) = 0.85
        self.assertGreater(calibrated, primary_pred.confidence)
        self.assertLessEqual(calibrated, 0.95)


class TestOnlineLearner(unittest.TestCase):
    """Test online learning system."""
    
    def test_update_weights(self):
        """Test that weights update correctly."""
        learner = OnlineLearner(["ModelA", "ModelB"])
        
        initial_weights = learner.get_weights()
        
        # Update ModelA with correct predictions
        for _ in range(10):
            learner.update("ModelA", was_correct=True, pnl=100.0)
        
        # Update ModelB with incorrect predictions
        for _ in range(10):
            learner.update("ModelB", was_correct=False, pnl=-50.0)
        
        final_weights = learner.get_weights()
        
        # ModelA should have higher weight
        self.assertGreater(final_weights["ModelA"], final_weights["ModelB"])
    
    def test_get_best_model(self):
        """Test getting best model."""
        learner = OnlineLearner(["ModelA", "ModelB"])
        
        # Make ModelA better
        for _ in range(20):
            learner.update("ModelA", was_correct=True, pnl=100.0)
            learner.update("ModelB", was_correct=False, pnl=-50.0)
        
        best = learner.get_best_model()
        self.assertEqual(best, "ModelA")


class TestMultiHorizonFusion(unittest.TestCase):
    """Test multi-horizon fusion."""
    
    def test_fuse_predictions(self):
        """Test prediction fusion."""
        from ml.advanced_signal_predictor import Prediction
        
        fusion = MultiHorizonFusion()
        
        predictions = {
            1: Prediction(0.5, 0.7, 1, "Test", []),
            5: Prediction(0.3, 0.6, 5, "Test", []),
            15: Prediction(0.2, 0.5, 15, "Test", []),
        }
        
        direction, confidence = fusion.fuse_predictions(predictions)
        
        self.assertGreater(direction, 0.0)
        self.assertGreater(confidence, 0.0)


class TestAdvancedSignalPredictor(unittest.TestCase):
    """Test the main advanced signal predictor."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = AdvancedSignalPredictor()
        
        self.assertIsNotNone(predictor.transformer)
        self.assertIsNotNone(predictor.gradient_boosted)
        self.assertIsNotNone(predictor.meta_labeler)
        self.assertIsNotNone(predictor.online_learner)
    
    def test_predict_generates_signal(self):
        """Test that predict generates a valid signal."""
        predictor = AdvancedSignalPredictor()
        
        # Generate synthetic price data
        random.seed(42)
        prices = [50000.0]
        for _ in range(100):
            change = random.gauss(0.0001, 0.002)
            prices.append(prices[-1] * (1 + change))
        
        signal = predictor.predict(prices)
        
        self.assertIn(signal.signal_type, SignalType)
        self.assertGreaterEqual(signal.direction, -1.0)
        self.assertLessEqual(signal.direction, 1.0)
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)
        self.assertGreaterEqual(signal.risk_score, 0.0)
        self.assertLessEqual(signal.risk_score, 100.0)
    
    def test_predict_with_volumes(self):
        """Test prediction with volume data."""
        predictor = AdvancedSignalPredictor()
        
        random.seed(42)
        prices = [50000.0]
        volumes = [1000.0]
        for _ in range(100):
            change = random.gauss(0.0001, 0.002)
            prices.append(prices[-1] * (1 + change))
            volumes.append(1000.0 + random.random() * 500)
        
        signal = predictor.predict(prices, volumes=volumes)
        
        self.assertIsNotNone(signal)
    
    def test_update_model_weights(self):
        """Test model weight updates."""
        predictor = AdvancedSignalPredictor()
        
        # Update weights
        predictor.update_model_weights("Transformer", was_correct=True, pnl=100.0)
        predictor.update_model_weights("GradientBoosted", was_correct=False, pnl=-50.0)
        
        stats = predictor.get_performance_stats()
        self.assertIn("model_weights", stats)
    
    def test_get_performance_stats(self):
        """Test performance statistics."""
        predictor = AdvancedSignalPredictor()
        
        random.seed(42)
        prices = [50000.0]
        for _ in range(100):
            change = random.gauss(0.0001, 0.002)
            prices.append(prices[-1] * (1 + change))
        
        # Make several predictions
        for _ in range(5):
            predictor.predict(prices)
        
        stats = predictor.get_performance_stats()
        
        self.assertEqual(stats["total_predictions"], 5)
        self.assertIn("signal_distribution", stats)
        self.assertIn("avg_confidence", stats)
        self.assertIn("model_weights", stats)


if __name__ == "__main__":
    unittest.main()
