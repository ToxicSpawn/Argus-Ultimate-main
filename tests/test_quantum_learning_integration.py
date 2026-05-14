"""
Test Quantum Learning Integration
===================================
Tests for the quantum_learning_integration module.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from quantum.quantum_learning_integration import (
    QuantumLearningConfig,
    QuantumRiskCalculator,
    QuantumRegimeDetector,
    HybridQLearner,
    QuantumLearningManager,
    wire_quantum_learning,
)


class TestQuantumLearningConfig:
    """Test suite for QuantumLearningConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = QuantumLearningConfig()
        assert config.reservoir_qubits == 6
        assert config.reservoir_layers == 3
        assert config.hybrid_qubits == 4
        assert config.qmc_samples == 2000
        assert config.enable_reservoir is True
        assert config.enable_hybrid_rl is True
        assert config.enable_qmc is True


class TestQuantumRiskCalculator:
    """Test suite for QuantumRiskCalculator."""
    
    def setup_method(self):
        self.calculator = QuantumRiskCalculator()
    
    def test_initialization(self):
        """Test calculator initialization."""
        assert self.calculator is not None
        assert len(self.calculator.risk_history) == 0
    
    def test_calculate_var_insufficient_data(self):
        """Test VaR calculation with insufficient data."""
        result = self.calculator.calculate_var(returns=[0.01])
        
        assert result["var"] == 0.0
        assert result["method"] == "insufficient_data"
    
    def test_calculate_var_with_returns(self):
        """Test VaR calculation with returns data."""
        returns = np.random.randn(100) * 0.02  # 100 daily returns, 2% vol
        
        result = self.calculator.calculate_var(
            returns=returns.tolist(),
            confidence=0.95,
            portfolio_value=10000.0
        )
        
        assert result["var"] > 0
        assert result["cvar"] >= result["var"]  # CVaR >= VaR
        assert result["method"] in ["sobol_qmc", "classical", "classical_fallback"]
        assert result["samples_used"] > 0
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        result = self.calculator.calculate_position_size(
            capital=10000.0,
            volatility=0.02,
            confidence=0.95
        )
        
        assert result["position_size"] > 0
        assert result["position_size"] <= 10000.0 * 0.25  # Max 25%
        assert result["method"] in ["sobol_qmc", "classical", "classical_fallback"]
    
    def test_stats(self):
        """Test statistics retrieval."""
        # Run a calculation first
        self.calculator.calculate_var(returns=np.random.randn(50).tolist())
        
        stats = self.calculator.get_stats()
        
        assert "qmc_available" in stats
        assert "samples_per_calc" in stats
        assert "risk_calculations" in stats


class TestQuantumRegimeDetector:
    """Test suite for QuantumRegimeDetector."""
    
    def setup_method(self):
        self.detector = QuantumRegimeDetector()
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector is not None
        assert self.detector._fitted is False
    
    def test_fit(self):
        """Test fitting on price data."""
        prices = [50000 + np.random.randn() * 100 for _ in range(100)]
        
        self.detector.fit(prices)
        
        # Should not raise, fitted may or may not be True depending on availability
    
    def test_detect_regime(self):
        """Test regime detection."""
        prices = [50000 + np.random.randn() * 100 for _ in range(50)]
        
        result = self.detector.detect_regime(prices)
        
        assert "regime" in result
        assert result["regime"] in ["trending_up", "trending_down", "ranging", "high_vol"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert "quantum_features" in result
    
    def test_classical_fallback_features(self):
        """Test classical feature fallback."""
        prices = [50000 + np.random.randn() * 100 for _ in range(50)]
        
        features = self.detector.get_quantum_features(prices)
        
        assert len(features) == 64  # Expected dimension
    
    def test_stats(self):
        """Test statistics retrieval."""
        self.detector.detect_regime([50000] * 50)
        
        stats = self.detector.get_stats()
        
        assert "reservoir_available" in stats
        assert "regime_detections" in stats


class TestHybridQLearner:
    """Test suite for HybridQLearner."""
    
    def setup_method(self):
        self.learner = HybridQLearner()
    
    def test_initialization(self):
        """Test learner initialization."""
        assert self.learner is not None
        assert self.learner.q_table.shape == (100, 10)
        assert self.learner.update_count == 0
    
    def test_encode_state(self):
        """Test state encoding."""
        market_features = {
            "volatility": 0.02,
            "trend": 0.01,
            "momentum": 0.005,
            "regime": "ranging"
        }
        
        state = self.learner.encode_state(market_features)
        
        assert 0 <= state < 100
    
    def test_select_action(self):
        """Test action selection."""
        market_features = {
            "volatility": 0.02,
            "trend": 0.01,
        }
        
        action, source = self.learner.select_action(state=5, market_features=market_features)
        
        assert 0 <= action < 10
        assert source in ["quantum", "classical", "classical_explore"]
    
    def test_update(self):
        """Test Q-table update."""
        self.learner.update(
            state=5,
            action=3,
            reward=1.0,
            next_state=10,
            source="classical"
        )
        
        assert self.learner.update_count == 1
        assert len(self.learner.classical_rewards) == 1
    
    def test_quantum_reward_tracking(self):
        """Test quantum reward tracking."""
        self.learner.update(
            state=5,
            action=3,
            reward=1.0,
            next_state=10,
            source="quantum"
        )
        
        assert len(self.learner.quantum_rewards) == 1
    
    def test_stats(self):
        """Test statistics retrieval."""
        self.learner.update(state=5, action=3, reward=1.0, next_state=10)
        
        stats = self.learner.get_stats()
        
        assert "quantum_available" in stats
        assert "quantum_weight" in stats
        assert "update_count" in stats
        assert "q_table_filled" in stats


class TestQuantumLearningManager:
    """Test suite for QuantumLearningManager."""
    
    def setup_method(self):
        self.manager = QuantumLearningManager()
    
    def test_initialization(self):
        """Test manager initialization."""
        assert self.manager is not None
        assert self.manager.risk_calculator is not None
        assert self.manager.regime_detector is not None
        assert self.manager.hybrid_learner is not None
    
    def test_fit(self):
        """Test fitting on price data."""
        prices = [50000 + np.random.randn() * 100 for _ in range(100)]
        
        self.manager.fit(prices)
        
        assert self.manager._fitted is True
    
    def test_get_position_size(self):
        """Test quantum-enhanced position sizing."""
        result = self.manager.get_position_size(
            capital=10000.0,
            volatility=0.02
        )
        
        assert result["position_size"] > 0
        assert result["method"] == "quantum_qmc"
    
    def test_detect_regime(self):
        """Test quantum-enhanced regime detection."""
        prices = [50000 + np.random.randn() * 100 for _ in range(50)]
        
        result = self.manager.detect_regime(prices)
        
        assert "regime" in result
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_select_action(self):
        """Test quantum-enhanced action selection."""
        market_features = {"volatility": 0.02, "trend": 0.01}
        
        action, source = self.manager.select_action(
            state=5,
            market_features=market_features
        )
        
        assert 0 <= action < 10
        assert source in ["quantum", "classical", "classical_explore"]
        assert len(self.manager._quantum_decisions) == 1
    
    def test_record_trade_outcome(self):
        """Test trade outcome recording."""
        market_features = {"volatility": 0.02, "trend": 0.01}
        
        self.manager.record_trade_outcome(
            pnl=100.0,
            state=5,
            action=3,
            source="classical",
            market_features=market_features
        )
        
        assert self.manager._trade_count == 1
        assert self.manager.hybrid_learner.update_count == 1
    
    def test_stats(self):
        """Test comprehensive statistics."""
        stats = self.manager.get_stats()
        
        assert "fitted" in stats
        assert "trade_count" in stats
        assert "risk" in stats
        assert "regime" in stats
        assert "hybrid_learner" in stats


class TestWireQuantumLearning:
    """Test suite for wire_quantum_learning."""
    
    def test_wire_returns_manager(self):
        """Test that wiring returns a manager."""
        manager = wire_quantum_learning()
        
        assert manager is not None
        assert isinstance(manager, QuantumLearningManager)


class TestQuantumLearningIntegration:
    """Integration tests for quantum learning."""
    
    def test_full_workflow(self):
        """Test complete workflow: fit → detect → trade → learn."""
        manager = QuantumLearningManager()
        
        # Generate historical data
        prices = [50000 + np.random.randn() * 100 for _ in range(100)]
        
        # Fit
        manager.fit(prices)
        
        # Detect regime
        regime = manager.detect_regime(prices[-30:])
        assert "regime" in regime
        
        # Get position size
        position = manager.get_position_size(capital=10000.0, volatility=0.02)
        assert position["position_size"] > 0
        
        # Select action
        market_features = {
            "volatility": regime.get("volatility", 0.02),
            "trend": regime.get("trend", 0.0),
            "regime": regime["regime"]
        }
        action, source = manager.select_action(state=5, market_features=market_features)
        
        # Record outcome
        manager.record_trade_outcome(
            pnl=50.0,
            state=5,
            action=action,
            source=source,
            market_features=market_features
        )
        
        # Verify learning happened
        stats = manager.get_stats()
        assert stats["trade_count"] == 1
        assert stats["hybrid_learner"]["update_count"] == 1
