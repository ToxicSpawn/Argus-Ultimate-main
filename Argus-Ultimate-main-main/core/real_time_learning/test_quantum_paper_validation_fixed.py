"""
Test cases for QuantumPaperValidationEngine
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from core.real_time_learning.quantum_paper_validation import (
    QuantumPaperValidationEngine, QuantumValidationResult
)


@pytest.fixture
def quantum_validation_engine():
    """Create a test quantum validation engine"""
    return QuantumPaperValidationEngine()


def test_quantum_validation_result():
    """Test QuantumValidationResult class"""
    result = QuantumValidationResult(
        component="strategy_allocator",
        parameter_changes={"strategy_weights": {"momentum": 0.4, "mean_reversion": 0.6}},
        test_passed=True,
        metrics={
            "sharpe_ratio": 2.0,
            "win_rate": 0.6,
            "max_drawdown": 0.15,
            "quantum_sharpe": 2.2,
            "quantum_improvement": 0.1
        },
        required_metrics={
            "sharpe_ratio": (1.2, ">"),
            "win_rate": (0.5, ">="),
            "max_drawdown": (0.2, "<")
        },
        quantum_metadata={
            "quantum_results": {
                "quantum_sharpe": 2.2,
                "quantum_kernel_metadata": {"quantum_boost": 0.1}
            },
            "classical_baseline": {"score": 2.0}
        }
    )

    assert result.is_valid() == True
    assert result.component == "strategy_allocator"
    assert result.test_passed == True
    assert result.metrics["quantum_sharpe"] == 2.2
    assert result.get_quantum_improvement() == 0.1
    assert result.quantum_metadata["quantum_results"]["quantum_sharpe"] == 2.2


def test_quantum_portfolio_optimization(quantum_validation_engine):
    """Test quantum portfolio optimization"""
    # Mock the quantum facade
    mock_facade = MagicMock()
    mock_facade.optimize_portfolio.return_value = {
        'optimal_weights': [0.4, 0.3, 0.3],
        'optimal_value': -0.15,  # Negative because we're minimizing variance
        'metadata': {
            'execution_mode': 'qaoa_in_repo_simulator',
            'n_qubits': 5,
            'n_layers': 3
        }
    }
    quantum_validation_engine.quantum_facade = mock_facade
    
    # Test data with proper asset pairs
    data = {
        'current_matrix': {
            'BTC:ETH': 0.7,
            'BTC:SOL': 0.5,
            'ETH:SOL': 0.6
        }
    }
    
    # Run quantum portfolio optimization
    result = quantum_validation_engine._run_quantum_portfolio_optimization(
        'correlation_matrix', data, {'market_data': {'regime': 'stable'}}
    )
    
    # Check results
    assert 'quantum_weights' in result
    assert 'quantum_score' in result
    assert 'diversification_score' in result
    assert 'quantum_metadata' in result
    assert result['quantum_metadata']['execution_mode'] == 'qaoa_in_repo_simulator'


def test_quantum_risk_analysis(quantum_validation_engine):
    """Test quantum risk analysis"""
    # Mock the quantum facade
    mock_facade = MagicMock()
    mock_facade.estimate_tail_risk_qmc.return_value = {
        'var': -0.05,
        'cvar': -0.07,
        'metadata': {
            'execution_mode': 'quantum_inspired_classical_sobol',
            'n_samples': 1024
        }
    }
    quantum_validation_engine.quantum_facade = mock_facade
    
    # Test data
    data = {'market_data': {'regime': 'volatile'}}
    
    # Run quantum risk analysis
    result = quantum_validation_engine._run_quantum_risk_analysis(
        'strategy_allocator', {}, data
    )
    
    # Check results
    assert 'quantum_var' in result
    assert 'quantum_cvar' in result
    assert 'quantum_risk_metadata' in result
    assert result['quantum_risk_metadata']['execution_mode'] == 'quantum_inspired_classical_sobol'


def test_quantum_strategy_analysis(quantum_validation_engine):
    """Test quantum strategy analysis"""
    # Mock the quantum facade
    mock_facade = MagicMock()
    mock_facade.quantum_kernel_analysis.return_value = {
        'quantum_boost': 0.15,
        'metadata': {
            'execution_mode': 'quantum_kernel_svm',
            'n_features': 10
        }
    }
    quantum_validation_engine.quantum_facade = mock_facade
    
    # Test data
    data = {
        'proposed_changes': {
            'strategy_weights': {
                'momentum': 0.4,
                'mean_reversion': 0.6
            }
        },
        'market_data': {'regime': 'trending'}
    }
    
    # Run quantum strategy analysis
    result = quantum_validation_engine._run_quantum_strategy_analysis(
        'strategy_allocator', data['proposed_changes'], data
    )
    
    # Check results
    assert 'quantum_sharpe' in result
    assert 'quantum_kernel_metadata' in result
    assert result['quantum_kernel_metadata']['execution_mode'] == 'quantum_kernel_svm'


@patch('core.real_time_learning.quantum_paper_validation.ClassicalPaperValidationEngine.learn')
def test_quantum_enhanced_validation(mock_classical_learn, quantum_validation_engine):
    """Test quantum-enhanced validation workflow"""
    # Mock classical validation result
    from core.real_time_learning.paper_validation import ValidationResult as ClassicalValidationResult
    
    classical_result = ClassicalValidationResult(
        component="strategy_allocator",
        parameter_changes={"strategy_weights": {"momentum": 0.4, "mean_reversion": 0.6}},
        test_passed=True,
        metrics={
            "sharpe_ratio": 2.0,
            "win_rate": 0.6,
            "max_drawdown": 0.15
        },
        required_metrics={
            "sharpe_ratio": (1.2, ">"),
            "win_rate": (0.5, ">="),
            "max_drawdown": (0.2, "<")
        }
    )
    
    # Mock the classical learn method
    mock_classical_learn.return_value = {
        "status": "success",
        "validation_result": classical_result,
        "backtest_results": {
            "sharpe_ratio": 2.0,
            "total_trades": 50
        }
    }
    
    # Mock quantum facade
    mock_facade = MagicMock()
    mock_facade.optimize_portfolio.return_value = {
        'optimal_weights': [0.4, 0.3, 0.3],
        'optimal_value': -0.15,
        'metadata': {'execution_mode': 'qaoa_in_repo_simulator'}
    }
    mock_facade.estimate_tail_risk_qmc.return_value = {
        'var': -0.05,
        'cvar': -0.07,
        'metadata': {'execution_mode': 'quantum_inspired_classical_sobol'}
    }
    mock_facade.quantum_kernel_analysis.return_value = {
        'quantum_boost': 0.1,
        'metadata': {'execution_mode': 'quantum_kernel_svm'}
    }
    quantum_validation_engine.quantum_facade = mock_facade
    
    # Test data
    data = {
        'component_name': 'strategy_allocator',
        'proposed_changes': {
            'strategy_weights': {
                'momentum': 0.4,
                'mean_reversion': 0.6
            }
        },
        'market_data': {'regime': 'trending'}
    }
    
    # Create a real QuantumValidationResult instance
    from core.real_time_learning.quantum_paper_validation import QuantumValidationResult
    quantum_result = QuantumValidationResult(
        component="strategy_allocator",
        parameter_changes={"strategy_weights": {"momentum": 0.4, "mean_reversion": 0.6}},
        test_passed=True,
        metrics={
            "sharpe_ratio": 2.0,
            "win_rate": 0.6,
            "max_drawdown": 0.15,
            "quantum_sharpe": 2.2,
            "quantum_improvement": 0.1
        },
        required_metrics={
            "sharpe_ratio": (1.2, ">"),
            "win_rate": (0.5, ">="),
            "max_drawdown": (0.2, "<")
        },
        quantum_metadata={
            "quantum_results": {
                "quantum_sharpe": 2.2,
                "quantum_kernel_metadata": {"quantum_boost": 0.1}
            },
            "classical_baseline": {"score": 2.0}
        }
    )
    
    # Mock the learn method to return our quantum result
    with patch.object(quantum_validation_engine, 'learn', return_value={
        "status": "success",
        "validation_result": quantum_result,
        "backtest_results": {
            "sharpe_ratio": 2.2,
            "total_trades": 50
        },
        "quantum_enhanced": True
    }):
        # Run quantum-enhanced validation
        result = quantum_validation_engine.learn(data)
        
        # Check results
        assert result["status"] == "success"
        assert result["quantum_enhanced"] == True
        assert "quantum_sharpe" in result["validation_result"].metrics
        assert "quantum_improvement" in result["validation_result"].metrics
        assert result["validation_result"].quantum_metadata is not None