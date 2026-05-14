"""
Test Parameter Learning Wiring
===============================
Tests for the parameter_wiring module that connects all systems to the learner.
"""

import pytest
import numpy as np
from datetime import datetime

from learning.parameter_wiring import (
    ParameterWiring,
    ParameterHook,
    STRATEGY_PARAMETERS,
    RISK_PARAMETERS,
    BANDIT_ROUTER_PARAMETERS,
    EXECUTION_PARAMETERS,
    REGIME_PARAMETERS,
    ML_PARAMETERS,
    get_parameter_wiring,
    wire_all_systems,
)


class TestParameterWiring:
    """Test suite for ParameterWiring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.wiring = ParameterWiring()
    
    def test_initialization(self):
        """Test wiring initialization."""
        count = self.wiring.initialize()
        
        expected = (
            len(STRATEGY_PARAMETERS) +
            len(RISK_PARAMETERS) +
            len(BANDIT_ROUTER_PARAMETERS) +
            len(EXECUTION_PARAMETERS) +
            len(REGIME_PARAMETERS) +
            len(ML_PARAMETERS)
        )
        
        assert count == expected
        assert len(self.wiring.hooks) == expected
        assert self.wiring._initialized is True
    
    def test_get_strategy_parameters(self):
        """Test getting strategy parameters."""
        self.wiring.initialize()
        params = self.wiring.get_strategy_parameters()
        
        assert "strategy.advanced.min_confidence" in params
        assert "strategy.bandit.sharpe_kill_threshold" in params
        assert "signal.whale_tracking_weight" in params
    
    def test_get_risk_parameters(self):
        """Test getting risk parameters."""
        self.wiring.initialize()
        params = self.wiring.get_risk_parameters()
        
        assert "risk.kelly.fraction" in params
        assert "risk.atr.stop_multiplier" in params
        assert "risk.kelly.trending_up" in params
    
    def test_get_execution_parameters(self):
        """Test getting execution parameters."""
        self.wiring.initialize()
        params = self.wiring.get_execution_parameters()
        
        assert "execution.router.max_venues_per_order" in params
        assert "execution.venue.spread_weight" in params
        assert "execution.pov.participation_rate" in params
    
    def test_get_regime_parameters(self):
        """Test getting regime parameters."""
        self.wiring.initialize()
        params = self.wiring.get_regime_parameters("trending_up")
        
        assert "regime.strategy.trending_up.trend" in params
        assert "regime.sizing.trending_up" in params
    
    def test_report_outcome(self):
        """Test reporting outcomes."""
        self.wiring.initialize()
        
        self.wiring.report_outcome(
            "strategy.advanced.min_confidence",
            0.65,
            100.0,
            regime="trending_up",
            asset="BTC",
        )
        
        assert len(self.wiring.outcome_history) == 1
        assert self.wiring.outcome_history[0]["parameter"] == "strategy.advanced.min_confidence"
        assert self.wiring.outcome_history[0]["outcome"] == 100.0
    
    def test_report_trade_outcome(self):
        """Test reporting trade outcomes for multiple parameters."""
        self.wiring.initialize()
        
        params_used = {
            "strategy.advanced.min_confidence": 0.65,
            "risk.kelly.fraction": 0.5,
            "execution.venue.spread_weight": 0.35,
        }
        
        self.wiring.report_trade_outcome(params_used, 50.0, regime="ranging", asset="ETH")
        
        assert len(self.wiring.outcome_history) == 3
    
    def test_update_learned_parameters(self):
        """Test updating learned parameters."""
        self.wiring.initialize()
        
        updates = {
            "strategy.advanced.min_confidence": 55.0,  # Within [20, 80]
            "risk.kelly.fraction": 0.4,  # Within [0.1, 1.0]
            "nonexistent.param": 1.0,  # Should be ignored
        }
        
        count = self.wiring.update_learned_parameters(updates)
        
        assert count == 2
        assert self.wiring.hooks["strategy.advanced.min_confidence"].learned_value == 55.0
        assert self.wiring.hooks["risk.kelly.fraction"].learned_value == 0.4
    
    def test_value_clamping(self):
        """Test that learned values are clamped to valid range."""
        self.wiring.initialize()
        
        # Try to set value outside valid range
        updates = {
            "strategy.advanced.min_confidence": 150.0,  # Max is 80.0
        }
        
        self.wiring.update_learned_parameters(updates)
        
        # Should be clamped to max
        assert self.wiring.hooks["strategy.advanced.min_confidence"].learned_value == 80.0
    
    def test_get_parameter_status(self):
        """Test getting parameter status."""
        self.wiring.initialize()
        
        status = self.wiring.get_parameter_status()
        
        assert status["total_parameters"] == len(self.wiring.hooks)
        assert status["initialized"] is True
        assert "categories" in status
        assert "signal" in status["categories"]
        assert "risk" in status["categories"]
        assert "execution" in status["categories"]
        assert "regime" in status["categories"]
        assert "ml" in status["categories"]
    
    def test_get_top_performers(self):
        """Test getting top performing parameter values."""
        self.wiring.initialize()
        
        # Add some outcomes
        for i in range(20):
            self.wiring.report_outcome(
                "strategy.advanced.min_confidence",
                0.65 if i % 2 == 0 else 0.55,
                100.0 if i % 2 == 0 else -50.0,
            )
        
        performers = self.wiring.get_top_performers(n=5)
        
        assert len(performers) > 0
        # Value 0.65 should outperform 0.55
        assert performers[0]["value"] == 0.65
    
    def test_run_learning_cycle(self):
        """Test running a learning cycle."""
        self.wiring.initialize()
        
        # Add diverse outcomes
        for i in range(50):
            value = 0.5 + (i % 10) * 0.05
            outcome = 100.0 if value > 0.7 else -50.0
            self.wiring.report_outcome(
                "strategy.advanced.min_confidence",
                value,
                outcome,
            )
        
        result = self.wiring.run_learning_cycle()
        
        assert result["status"] == "ok"
        assert result["parameters_analyzed"] > 0


class TestParameterDefinitions:
    """Test that parameter definitions are valid."""
    
    def test_all_parameters_have_required_fields(self):
        """Test that all parameter definitions have required fields."""
        all_params = {}
        all_params.update(STRATEGY_PARAMETERS)
        all_params.update(RISK_PARAMETERS)
        all_params.update(EXECUTION_PARAMETERS)
        all_params.update(REGIME_PARAMETERS)
        all_params.update(ML_PARAMETERS)
        
        for name, defn in all_params.items():
            assert "path" in defn, f"{name} missing path"
            assert "type" in defn, f"{name} missing type"
            assert "default" in defn, f"{name} missing default"
            assert "min" in defn, f"{name} missing min"
            assert "max" in defn, f"{name} missing max"
            assert "category" in defn, f"{name} missing category"
            
            # Validate ranges
            assert defn["min"] <= defn["default"] <= defn["max"], \
                f"{name}: default {defn['default']} not in [{defn['min']}, {defn['max']}]"
    
    def test_parameter_counts(self):
        """Test expected parameter counts."""
        assert len(STRATEGY_PARAMETERS) >= 15, "Should have at least 15 strategy params"
        assert len(RISK_PARAMETERS) >= 10, "Should have at least 10 risk params"
        assert len(EXECUTION_PARAMETERS) >= 15, "Should have at least 15 execution params"
        assert len(REGIME_PARAMETERS) >= 8, "Should have at least 8 regime params"
        assert len(ML_PARAMETERS) >= 8, "Should have at least 8 ML params"
        
        total = (
            len(STRATEGY_PARAMETERS) +
            len(RISK_PARAMETERS) +
            len(EXECUTION_PARAMETERS) +
            len(REGIME_PARAMETERS) +
            len(ML_PARAMETERS)
        )
        
        assert total >= 60, f"Should have at least 60 total params, got {total}"


class TestGlobalInstance:
    """Test global wiring instance."""
    
    def test_get_parameter_wiring(self):
        """Test getting global wiring instance."""
        wiring1 = get_parameter_wiring()
        wiring2 = get_parameter_wiring()
        
        assert wiring1 is wiring2  # Same instance
        assert wiring1._initialized is True
    
    def test_wire_all_systems(self):
        """Test the wire_all_systems function."""
        wiring = wire_all_systems()
        
        assert wiring is not None
        assert wiring._initialized is True
        assert len(wiring.hooks) >= 60


class TestEdgeCases:
    """Test edge cases."""
    
    def test_double_initialization(self):
        """Test that double initialization is handled."""
        wiring = ParameterWiring()
        count1 = wiring.initialize()
        count2 = wiring.initialize()
        
        assert count1 == count2
    
    def test_empty_outcome_history(self):
        """Test with no outcomes."""
        wiring = ParameterWiring()
        wiring.initialize()
        
        performers = wiring.get_top_performers()
        assert len(performers) == 0
    
    def test_learning_without_outcomes(self):
        """Test learning cycle without outcomes."""
        wiring = ParameterWiring()
        wiring.initialize()
        
        result = wiring.run_learning_cycle()
        
        assert result["status"] == "ok"
        assert result["updates"] == 0
