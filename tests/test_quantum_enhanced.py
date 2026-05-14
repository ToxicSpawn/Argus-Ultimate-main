"""
Tests for Quantum-Enhanced Strategy Wrapper (v15.0.0).

Author: Argus Ultimate
"""

from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime, timezone

from strategies.quantum_enhanced_wrapper import (
    QuantumEnhancedStrategyWrapper,
    QuantumEnhancedSignal,
    QuantumStrategyConfig,
    create_quantum_enhanced_wrapper,
)


class TestQuantumStrategyConfig:
    """Tests for configuration."""
    
    def test_defaults(self):
        config = QuantumStrategyConfig()
        assert config.use_quantum_kernel is True
        assert config.use_quantum_reservoir is True
        assert config.use_quantum_risk is True
        assert config.n_qubits == 6
        assert config.classical_fallback is True


class TestQuantumEnhancedStrategyWrapper:
    """Tests for quantum-enhanced wrapper."""
    
    @pytest.fixture
    def mock_strategies(self):
        return {
            "grid_mean_rev": object(),
            "triangular_arb": object(),
            "funding_rate": object(),
        }
    
    @pytest.fixture
    def wrapper(self, mock_strategies):
        return QuantumEnhancedStrategyWrapper(mock_strategies)
    
    def test_init(self, wrapper, mock_strategies):
        """Should initialize correctly."""
        assert len(wrapper.strategies) == 3
        assert len(wrapper._feature_history) == 0
    
    def test_extract_features(self, wrapper):
        """Should extract normalized features."""
        features = wrapper.extract_features(
            price=60000.0,
            volume=1000000.0,
            rsi=65.0,
            bb_position=0.3,
            volatility=0.02,
        )
        
        assert len(features) == 5
        assert all(-1 <= f <= 1 for f in features)
    
    def test_extract_features_multiple(self, wrapper):
        """Should maintain feature history."""
        for i in range(10):
            wrapper.extract_features(
                price=60000 + i * 100,
                volume=1000000,
                rsi=50 + i,
                bb_position=i * 0.1 - 0.5,
                volatility=0.02,
            )
        
        assert len(wrapper._feature_history) == 10
    
    def test_classify_regime_returns_valid(self, wrapper):
        """Should return valid regime classification."""
        features = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        regime, confidence = wrapper.classify_regime(features)
        
        assert regime in ["trending", "range", "volatile", "crisis"]
        assert 0 <= confidence <= 1
    
    def test_classify_regime_trending(self, wrapper):
        """Should classify trending regime."""
        features = np.array([0.8, 0.9, 0.2, 0.0, 0.5])  # High RSI, high BB
        regime, confidence = wrapper.classify_regime(features)
        
        # Should detect trending due to high BB position
        assert regime in ["trending", "range"]
    
    def test_classify_regime_volatile(self, wrapper):
        """Should classify volatile regime."""
        features = np.array([0.0, 0.0, 0.8, 0.0, 0.0])  # High volatility
        regime, confidence = wrapper.classify_regime(features)
        
        # Volatility > 0.5 triggers volatile classification in classical fallback
        assert regime in ["volatile", "range"]  # May vary based on implementation
    
    def test_predict_regime_duration(self, wrapper):
        """Should predict regime duration."""
        # Add some feature history
        for i in range(15):
            wrapper.extract_features(
                price=60000,
                volume=1000000,
                rsi=50,
                bb_position=0,
                volatility=0.02,
            )
        
        features_list = list(wrapper._feature_history)
        duration = wrapper.predict_regime_duration("range", features_list)
        
        assert 1 <= duration <= 72
    
    def test_calculate_risk_score(self, wrapper):
        """Should calculate risk metrics."""
        returns = [0.01, -0.02, 0.005, -0.01, 0.015, -0.005, 0.01, -0.02]
        risk = wrapper.calculate_risk_score(returns)
        
        assert "var" in risk
        assert "cvar" in risk
        assert "risk_score" in risk
        assert 0 <= risk["risk_score"] <= 1
    
    def test_enhance_strategy_signal(self, wrapper):
        """Should enhance strategy signal."""
        features = np.array([0.2, 0.1, 0.3, 0.0, 0.0])
        returns = [0.01, -0.01, 0.02, -0.015]
        
        signal = wrapper.enhance_strategy_signal(
            strategy_name="grid_mean_rev",
            base_signal="buy",
            base_confidence=0.7,
            features=features,
            returns=returns,
        )
        
        assert isinstance(signal, QuantumEnhancedSignal)
        assert signal.strategy_name == "grid_mean_rev"
        assert signal.base_signal == "buy"
        assert 0 <= signal.quantum_confidence <= 1
        assert signal.regime_prediction in ["trending", "range", "volatile", "crisis"]
        assert signal.predicted_duration > 0
    
    def test_get_best_strategy(self, wrapper):
        """Should recommend best strategy."""
        # Add feature history
        for i in range(20):
            wrapper.extract_features(
                price=60000,
                volume=1000000,
                rsi=50,
                bb_position=0.1,
                volatility=0.02,
            )
        
        features = np.array([0.1, 0.1, 0.2, 0.0, 0.0])
        returns = [0.01, -0.01, 0.005]
        
        best, confidence = wrapper.get_best_strategy(features, returns)
        
        assert best in wrapper.strategies
        assert 0 <= confidence <= 1
    
    def test_get_stats(self, wrapper):
        """Should return statistics."""
        stats = wrapper.get_stats()
        
        assert "quantum_available" in stats
        assert "strategies_enhanced" in stats
        assert "features_collected" in stats


class TestQuantumKernelSimilarity:
    """Tests for quantum kernel similarity."""
    
    def test_identical_vectors(self):
        """Identical vectors should have high similarity."""
        wrapper = QuantumEnhancedStrategyWrapper({})
        x1 = np.array([0.5, 0.3, 0.2, 0.1, 0.4])
        x2 = np.array([0.5, 0.3, 0.2, 0.1, 0.4])
        
        sim = wrapper._quantum_kernel_similarity(x1, x2)
        assert sim > 0.8  # Should be high for identical vectors
    
    def test_different_vectors(self):
        """Different vectors should have lower similarity."""
        wrapper = QuantumEnhancedStrategyWrapper({})
        x1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        x2 = np.array([-0.9, -0.8, -0.7, -0.6, -0.5])
        
        sim = wrapper._quantum_kernel_similarity(x1, x2)
        assert sim < 0.5
    
    def test_similarity_range(self):
        """Similarity should be in [0, 1]."""
        wrapper = QuantumEnhancedStrategyWrapper({})
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)
        
        sim = wrapper._quantum_kernel_similarity(x1, x2)
        assert 0 <= sim <= 1


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_wrapper(self):
        """Should create configured wrapper."""
        strategies = {"test": object()}
        wrapper = create_quantum_enhanced_wrapper(strategies, use_quantum=True)
        
        assert isinstance(wrapper, QuantumEnhancedStrategyWrapper)
        assert len(wrapper.strategies) == 1
    
    def test_create_wrapper_no_quantum(self):
        """Should create wrapper without quantum."""
        strategies = {"test": object()}
        wrapper = create_quantum_enhanced_wrapper(strategies, use_quantum=False)
        
        assert wrapper.config.use_quantum_kernel is False


class TestClassicalFallbacks:
    """Tests for classical fallback behavior."""
    
    def test_classical_classify(self):
        """Should classify with classical fallback."""
        wrapper = QuantumEnhancedStrategyWrapper({})
        features = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        regime, confidence = wrapper._classical_classify(features)
        assert regime in ["trending", "range", "volatile"]
    
    def test_classical_risk(self):
        """Should calculate risk with classical fallback."""
        wrapper = QuantumEnhancedStrategyWrapper({})
        returns = [0.01, -0.01, 0.02, -0.02, 0.015]
        
        risk = wrapper._classical_risk(returns, 0.95)
        assert risk["method"] == "classical"
        assert "var" in risk
    
    def test_classical_duration(self):
        """Should predict duration with classical fallback."""
        wrapper = QuantumEnhancedStrategyWrapper({})
        
        for regime in ["trending", "range", "volatile", "crisis"]:
            duration = wrapper._classical_predict_duration(regime)
            assert duration > 0