"""
Test ML Learning Integration
==============================
Tests for the ml_learning_integration module.
"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import MagicMock

from ml.ml_learning_integration import (
    MLLearningConfig,
    MLDetector,
    MLMetaLearner,
    MLSignalStacker,
    MLTransferLearner,
    MLLearningManager,
    wire_ml_learning,
    reset_ml_learning,
)


def _get_test_config() -> MLLearningConfig:
    """Get config with temporary database for testing."""
    # Create a unique temp db for this test
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return MLLearningConfig(
        meta_db_path=path,
        meta_min_records=1,  # Allow selection with just 1 record for testing
    )


class TestMLLearningConfig:
    """Test suite for MLLearningConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MLLearningConfig()
        assert config.drift_window_size == 250
        assert config.drift_delta == 0.002
        assert config.drift_threshold == 0.1
        assert config.auto_reset_on_drift is True
        assert config.transfer_similarity_threshold == 0.7
        assert config.enable_drift_detector is True
        assert config.enable_meta_learner is True


class TestMLDetector:
    """Test suite for MLDetector (Drift Detection)."""
    
    def setup_method(self):
        self.detector = MLDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector is not None
        assert len(self.detector._drift_history) == 0
    
    def test_check_drift_no_drift(self):
        """Test drift detection with stable data."""
        features = {"volatility": 0.02, "momentum": 0.01}
        predictions = [0.5] * 10
        actuals = [0.5] * 10
        
        result = self.detector.check_drift(features, predictions, actuals)
        
        assert result["drift_detected"] is False
        assert result["drift_type"] == "none"
    
    def test_check_drift_with_change(self):
        """Test drift detection with changing data."""
        features = {"volatility": 0.02, "momentum": 0.01}
        
        # First, stable data
        for _ in range(30):
            self.detector.check_drift(
                features,
                predictions=[0.5],
                actuals=[0.5]
            )
        
        # Then, sudden change
        result = self.detector.check_drift(
            features,
            predictions=[0.5],
            actuals=[1.0]  # Big error
        )
        
        # Should process without error
        assert "drift_detected" in result
    
    def test_stats(self):
        """Test statistics retrieval."""
        stats = self.detector.get_stats()
        
        assert "detector_available" in stats
        assert "drift_count" in stats
        assert "window_size" in stats


class TestMLMetaLearner:
    """Test suite for MLMetaLearner (Best Model Selection)."""
    
    def setup_method(self):
        self.meta = MLMetaLearner(_get_test_config())
    
    def test_initialization(self):
        """Test meta learner initialization."""
        assert self.meta is not None
        assert len(self.meta._performance_history) == 0
    
    def test_record_performance(self):
        """Test recording performance."""
        self.meta.record_performance(
            model_name="momentum",
            regime="trending_up",
            features={"vol": 0.02},
            performance=0.72
        )
        
        assert "momentum" in self.meta._performance_history
        assert len(self.meta._performance_history["momentum"]) == 1
    
    def test_select_best_with_data(self):
        """Test selecting best model with data."""
        # Record multiple performances
        self.meta.record_performance("momentum", "trending_up", {"vol": 0.02}, 0.72)
        self.meta.record_performance("momentum", "trending_up", {"vol": 0.02}, 0.68)
        self.meta.record_performance("mean_reversion", "trending_up", {"vol": 0.02}, 0.65)
        
        result = self.meta.select_best("trending_up", {"vol": 0.02})
        
        assert result["best_model"] == "momentum"
        assert result["confidence"] > 0
        assert len(result["rankings"]) >= 1  # At least one model ranked
    
    def test_select_best_empty(self):
        """Test selecting best with no data."""
        result = self.meta.select_best("ranging", {"vol": 0.01})
        
        assert result["best_model"] == ""
        assert result["confidence"] == 0.0
    
    def test_get_model_stats(self):
        """Test getting model statistics."""
        self.meta.record_performance("xgboost", "trending_up", {"vol": 0.02}, 0.80)
        self.meta.record_performance("xgboost", "ranging", {"vol": 0.01}, 0.65)
        
        stats = self.meta.get_model_stats("xgboost")
        
        assert stats["records"] == 2
        assert abs(stats["avg_performance"] - 0.725) < 0.001  # Floating point comparison
        assert "trending_up" in stats["regimes"]
    
    def test_stats(self):
        """Test overall statistics."""
        self.meta.record_performance("model_a", "trending_up", {}, 0.7)
        
        stats = self.meta.get_stats()
        
        assert "tracked_models" in stats
        assert stats["tracked_models"] == 1


class TestMLSignalStacker:
    """Test suite for MLSignalStacker (Ensemble Fusion)."""
    
    def setup_method(self):
        self.stacker = MLSignalStacker()
    
    def test_initialization(self):
        """Test stacker initialization."""
        assert self.stacker is not None
        assert len(self.stacker._strategies) == 0
    
    def test_add_strategy(self):
        """Test adding strategy."""
        mock_strategy = MagicMock()
        self.stacker.add_strategy("momentum", mock_strategy, initial_weight=1.5)
        
        assert "momentum" in self.stacker._strategies
        assert self.stacker._strategy_weights["momentum"] == 1.5
    
    def test_stack_signals_buy(self):
        """Test stacking with buy signals."""
        self.stacker.add_strategy("momentum", MagicMock())
        self.stacker.add_strategy("mean_reversion", MagicMock())
        
        signals = {
            "momentum": {"action": "buy", "confidence": 0.8},
            "mean_reversion": {"action": "buy", "confidence": 0.6},
        }
        
        result = self.stacker.stack_signals(signals, regime="trending_up")
        
        assert result["action"] == "buy"
        assert result["confidence"] > 0
        assert result["n_strategies"] == 2
    
    def test_stack_signals_conflict(self):
        """Test stacking with conflicting signals."""
        signals = {
            "momentum": {"action": "buy", "confidence": 0.6},
            "mean_reversion": {"action": "sell", "confidence": 0.6},
        }
        
        result = self.stacker.stack_signals(signals)
        
        # Should still produce a signal (whichever has higher score)
        assert result["action"] in ["buy", "sell", "hold"]
    
    def test_stack_signals_hold(self):
        """Test stacking with weak signals."""
        signals = {
            "momentum": {"action": "buy", "confidence": 0.1},
            "mean_reversion": {"action": "sell", "confidence": 0.1},
        }
        
        result = self.stacker.stack_signals(signals)
        
        assert result["action"] == "hold"
    
    def test_update_weights(self):
        """Test updating strategy weights."""
        self.stacker.add_strategy("momentum", MagicMock(), initial_weight=1.0)
        self.stacker.add_strategy("mean_reversion", MagicMock(), initial_weight=1.0)
        
        self.stacker.update_weights({"momentum": 0.5, "mean_reversion": -0.2})  # momentum better
        
        # After normalization, momentum should have higher weight than mean_reversion
        assert self.stacker._strategy_weights["momentum"] > self.stacker._strategy_weights["mean_reversion"]
    
    def test_stats(self):
        """Test statistics retrieval."""
        self.stacker.add_strategy("momentum", MagicMock())
        
        stats = self.stacker.get_stats()
        
        assert "n_strategies" in stats
        assert stats["n_strategies"] == 1


class TestMLTransferLearner:
    """Test suite for MLTransferLearner (Cross-Asset Knowledge)."""
    
    def setup_method(self):
        self.transfer = MLTransferLearner()
    
    def test_initialization(self):
        """Test transfer learner initialization."""
        assert self.transfer is not None
        assert len(self.transfer._asset_profiles) == 0
    
    def test_register_asset(self):
        """Test registering an asset."""
        features = {"volatility": 0.02, "trend": 0.01, "momentum": 0.005}
        
        self.transfer.register_asset("BTCUSDT", features, asset_type="crypto")
        
        assert "BTCUSDT" in self.transfer._asset_profiles
    
    def test_compute_similarity_same(self):
        """Test similarity with same asset."""
        features = {"volatility": 0.02, "trend": 0.01}
        
        self.transfer.register_asset("BTCUSDT", features)
        self.transfer.register_asset("BTCUSDT_2", features)  # Same features
        
        similarity = self.transfer.compute_similarity("BTCUSDT", "BTCUSDT_2")
        
        assert similarity == 1.0
    
    def test_compute_similarity_different(self):
        """Test similarity with different assets."""
        features_btc = {"volatility": 0.02, "trend": 0.01}
        features_eth = {"volatility": 0.04, "trend": -0.02}
        
        self.transfer.register_asset("BTCUSDT", features_btc)
        self.transfer.register_asset("ETHUSDT", features_eth)
        
        similarity = self.transfer.compute_similarity("BTCUSDT", "ETHUSDT")
        
        assert 0.0 <= similarity < 1.0
    
    def test_transfer_knowledge_similar(self):
        """Test knowledge transfer with similar assets."""
        features_btc = {"volatility": 0.02, "trend": 0.01}
        features_eth = {"volatility": 0.021, "trend": 0.011}
        
        # Register both assets first
        self.transfer.register_asset("BTCUSDT", features_btc)
        self.transfer.register_asset("ETHUSDT", features_eth)
        
        result = self.transfer.transfer_knowledge(
            "BTCUSDT", "ETHUSDT", features_eth
        )
        
        # With high similarity, should transfer
        assert result["similarity"] > 0.9
        assert result["should_transfer"] is True
        assert len(result["transferred_params"]) > 0
    
    def test_transfer_knowledge_different(self):
        """Test knowledge transfer with very different assets."""
        features_btc = {"volatility": 0.02, "trend": 0.01}
        features_stock = {"volatility": 0.01, "trend": 0.0}
        
        self.transfer.register_asset("BTCUSDT", features_btc)
        
        result = self.transfer.transfer_knowledge(
            "BTCUSDT", "SPY", features_stock
        )
        
        # With low similarity, might not transfer
        assert result["should_transfer"] == (result["similarity"] >= 0.7)
    
    def test_stats(self):
        """Test statistics retrieval."""
        self.transfer.register_asset("BTCUSDT", {"vol": 0.02})
        
        stats = self.transfer.get_stats()
        
        assert "registered_assets" in stats
        assert stats["registered_assets"] == 1


class TestMLLearningManager:
    """Test suite for MLLearningManager (Main Manager)."""
    
    def setup_method(self):
        reset_ml_learning()  # Reset global state
        self.manager = MLLearningManager(_get_test_config())
    
    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager is not None
        assert self.manager.drift_detector is not None
        assert self.manager.meta_learner is not None
        assert self.manager.signal_stacker is not None
        assert self.manager.transfer_learner is not None
    
    def test_check_drift(self):
        """Test drift checking."""
        result = self.manager.check_drift(
            features={"vol": 0.02},
            predictions=[0.5],
            actuals=[0.5]
        )
        
        assert "drift_detected" in result
        assert self.manager._learning_cycle_count == 1
    
    def test_select_best_strategy(self):
        """Test selecting best strategy."""
        self.manager.record_strategy_performance(
            "momentum", "trending_up", {"vol": 0.02}, 0.72
        )
        
        result = self.manager.select_best_strategy("trending_up", {"vol": 0.02})
        
        assert result["best_model"] == "momentum"
    
    def test_stack_signals(self):
        """Test signal stacking."""
        signals = {
            "momentum": {"action": "buy", "confidence": 0.8},
            "mean_reversion": {"action": "buy", "confidence": 0.6},
        }
        
        result = self.manager.stack_signals(signals, regime="trending_up")
        
        assert result["action"] == "buy"
    
    def test_transfer_knowledge(self):
        """Test knowledge transfer."""
        self.manager.register_asset("BTCUSDT", {"vol": 0.02})
        
        result = self.manager.transfer_knowledge(
            "BTCUSDT", "ETHUSDT", {"vol": 0.021}
        )
        
        assert "should_transfer" in result
    
    def test_stats(self):
        """Test comprehensive statistics."""
        self.manager.check_drift({"vol": 0.02}, [0.5], [0.5])
        self.manager.record_strategy_performance("momentum", "trending_up", {}, 0.7)
        
        stats = self.manager.get_stats()
        
        assert "learning_cycles" in stats
        assert "drift_detector" in stats
        assert "meta_learner" in stats
        assert "signal_stacker" in stats
        assert "transfer_learner" in stats


class TestWireMLLearning:
    """Test suite for wire_ml_learning."""
    
    def test_wire_returns_manager(self):
        """Test that wiring returns a manager."""
        reset_ml_learning()  # Reset global state
        manager = wire_ml_learning(_get_test_config())
        
        assert manager is not None
        assert isinstance(manager, MLLearningManager)


class TestMLLearningIntegration:
    """Integration tests for ML learning system."""
    
    def test_full_workflow(self):
        """Test complete workflow: detect drift → select best → stack → learn."""
        reset_ml_learning()  # Reset global state
        manager = MLLearningManager(_get_test_config())
        
        # Register strategies
        manager.signal_stacker.add_strategy("momentum", MagicMock())
        manager.signal_stacker.add_strategy("mean_reversion", MagicMock())
        
        # Register assets
        manager.register_asset("BTCUSDT", {"vol": 0.02, "trend": 0.01})
        
        # Record performance
        manager.record_strategy_performance("momentum", "trending_up", {"vol": 0.02}, 0.75)
        manager.record_strategy_performance("mean_reversion", "trending_up", {"vol": 0.02}, 0.65)
        
        # Check drift
        drift = manager.check_drift(
            {"vol": 0.02},
            predictions=[0.6, 0.7],
            actuals=[0.6, 0.7]
        )
        
        # Select best strategy
        best = manager.select_best_strategy("trending_up", {"vol": 0.02})
        
        # Stack signals
        signals = {
            "momentum": {"action": "buy", "confidence": 0.8},
            "mean_reversion": {"action": "buy", "confidence": 0.6},
        }
        stacked = manager.stack_signals(signals, "trending_up")
        
        # Register target asset for transfer
        manager.register_asset("ETHUSDT", {"vol": 0.021, "trend": 0.011})
        
        # Transfer knowledge
        transfer = manager.transfer_knowledge("BTCUSDT", "ETHUSDT", {"vol": 0.021, "trend": 0.011})
        
        # Verify everything worked
        assert drift is not None
        assert best["best_model"] == "momentum"
        assert stacked["action"] == "buy"
        assert transfer["similarity"] > 0.9  # Very similar features
        
        # Get stats
        stats = manager.get_stats()
        assert stats["learning_cycles"] == 1
        assert stats["meta_learner"]["tracked_models"] == 2
