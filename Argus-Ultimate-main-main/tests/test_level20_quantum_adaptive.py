"""
tests/test_level20_quantum_adaptive.py — Tests for Level 20 Quantum-Adaptive Bridge
"""

import pytest
import numpy as np

from evolution.level20_quantum_adaptive_bridge import (
    Level20QuantumBridge,
    Level20AdaptiveBridge,
    Level20QuantumAdaptiveOrchestrator,
    QuantumLevel,
    AdaptiveLevel,
    QuantumCapabilities,
    AdaptiveCapabilities,
    create_level20_quantum_adaptive,
)


class TestLevel20QuantumBridge:
    """Tests for Level 20 Quantum Bridge."""
    
    def test_init(self):
        """Should initialize correctly."""
        bridge = Level20QuantumBridge()
        
        assert bridge.capabilities.level == QuantumLevel.SIMULATED
        assert bridge.capabilities.n_qubits == 24
    
    def test_upgrade_to_level20(self):
        """Should upgrade to Level 20."""
        bridge = Level20QuantumBridge()
        
        result = bridge.upgrade_to_level20()
        
        assert result["new_level"] == "nisq"
        assert len(result["upgrades"]) > 0
    
    def test_run_quantum_ml_qnn(self):
        """Should run QNN."""
        bridge = Level20QuantumBridge()
        
        data = np.random.randn(100, 5)
        result = bridge.run_quantum_ml(data, model_type="qnn")
        
        assert result["model_type"] == "qnn"
        assert "predictions" in result
        assert "quantum_advantage" in result
    
    def test_run_quantum_ml_qgan(self):
        """Should run QGAN."""
        bridge = Level20QuantumBridge()
        
        data = np.random.randn(100)
        result = bridge.run_quantum_ml(data, model_type="qgan")
        
        assert result["model_type"] == "qgan"
        assert len(result["predictions"]) == 100
    
    def test_optimize_portfolio_quantum(self):
        """Should optimize portfolio using quantum."""
        bridge = Level20QuantumBridge()
        
        returns = np.random.randn(252, 5) * 0.02
        result = bridge.optimize_portfolio_quantum(returns, n_assets=5)
        
        assert "weights" in result
        assert "sharpe_ratio" in result
        assert len(result["weights"]) == 5
        assert abs(sum(result["weights"]) - 1.0) < 0.01  # Weights sum to 1


class TestLevel20AdaptiveBridge:
    """Tests for Level 20 Adaptive Bridge."""
    
    def test_init(self):
        """Should initialize correctly."""
        bridge = Level20AdaptiveBridge()
        
        assert bridge.capabilities.level == AdaptiveLevel.PREDICTIVE
    
    def test_upgrade_to_level20(self):
        """Should upgrade to Level 20."""
        bridge = Level20AdaptiveBridge()
        
        result = bridge.upgrade_to_level20()
        
        assert result["new_level"] == "autonomous"
        assert bridge.capabilities.self_awareness_score >= 0.8
    
    def test_adapt_with_causality(self):
        """Should adapt based on causal relationships."""
        bridge = Level20AdaptiveBridge()
        
        market_data = {"volatility": 0.03}
        causal_graph = {
            "fed_announcement": [
                {"target": "volatility", "strength": 0.8, "lag": 1},
            ],
        }
        
        result = bridge.adapt_with_causality(market_data, causal_graph)
        
        assert "causal_drivers" in result
        assert "adaptations" in result
    
    def test_predict_regime_change(self):
        """Should predict regime changes."""
        bridge = Level20AdaptiveBridge()
        
        result = bridge.predict_regime_change(
            "trending",
            {"volatility": 0.04, "trend_strength": 0.1, "volume_ratio": 0.7},
        )
        
        assert "predicted_change" in result
        assert "confidence" in result
    
    def test_self_improve(self):
        """Should self-improve based on performance."""
        bridge = Level20AdaptiveBridge()
        
        performance = {"win_rate": 0.4, "sharpe": 0.5, "max_drawdown": 0.25}
        mistakes = [{"type": "overtrading"}, {"type": "late_entry"}]
        
        result = bridge.self_improve(performance, mistakes)
        
        assert "weaknesses_identified" in result
        assert "improvements_planned" in result
        assert len(result["weaknesses_identified"]) > 0


class TestLevel20QuantumAdaptiveOrchestrator:
    """Tests for Level 20 Quantum-Adaptive Orchestrator."""
    
    def test_init(self):
        """Should initialize correctly."""
        orchestrator = Level20QuantumAdaptiveOrchestrator()
        
        assert orchestrator.quantum is not None
        assert orchestrator.adaptive is not None
    
    def test_initialize_level20(self):
        """Should initialize both systems to Level 20."""
        orchestrator = Level20QuantumAdaptiveOrchestrator()
        
        result = orchestrator.initialize_level20()
        
        assert result["quantum"]["new_level"] == "nisq"
        assert result["adaptive"]["new_level"] == "autonomous"
    
    def test_analyze_and_adapt(self):
        """Should analyze and adapt."""
        orchestrator = Level20QuantumAdaptiveOrchestrator()
        orchestrator.initialize_level20()
        
        market_data = {
            "price_data": np.random.randn(100).tolist(),
            "current_regime": "trending",
            "volatility": 0.03,
            "trend_strength": 0.5,
        }
        
        result = orchestrator.analyze_and_adapt(market_data)
        
        assert "quantum_ml" in result
        assert "regime_prediction" in result
        assert "combined_confidence" in result
    
    def test_get_system_report(self):
        """Should get system report."""
        orchestrator = Level20QuantumAdaptiveOrchestrator()
        
        report = orchestrator.get_system_report()
        
        assert "quantum" in report
        assert "adaptive" in report
        assert report["combined_level"] == "20 Singularity"


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_level20_quantum_adaptive(self):
        """Should create Level 20 Quantum-Adaptive system."""
        system = create_level20_quantum_adaptive()
        
        assert isinstance(system, Level20QuantumAdaptiveOrchestrator)
