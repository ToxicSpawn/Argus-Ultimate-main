"""
Test cases for Quantum Dashboard
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from monitoring.quantum_dashboard import QuantumDashboard, QuantumComponentMetrics, QuantumDashboardState


@pytest.fixture
def quantum_dashboard():
    """Create a test quantum dashboard"""
    return QuantumDashboard()


def test_component_metrics():
    """Test QuantumComponentMetrics class"""
    metrics = QuantumComponentMetrics("strategy_allocator")
    
    # Update classical metrics
    metrics.update_classical({"sharpe_ratio": 2.0, "win_rate": 0.6})
    
    # Update quantum metrics
    metrics.update_quantum(
        {"quantum_sharpe": 2.2, "quantum_win_rate": 0.65},
        {"sharpe_improvement": 0.1, "win_rate_improvement": 0.083},
        {"execution_mode": "qaoa_in_repo_simulator", "n_qubits": 5}
    )
    
    assert metrics.component_name == "strategy_allocator"
    assert metrics.classical_metrics["sharpe_ratio"] == 2.0
    assert metrics.quantum_metrics["quantum_sharpe"] == 2.2
    assert metrics.quantum_improvement["sharpe_improvement"] == 0.1
    assert metrics.quantum_execution_metadata["execution_mode"] == "qaoa_in_repo_simulator"


def test_dashboard_state():
    """Test QuantumDashboardState class"""
    state = QuantumDashboardState()
    
    # Update component
    state.update_component(
        component_name="strategy_allocator",
        classical_metrics={"sharpe_ratio": 2.0},
        quantum_metrics={"quantum_sharpe": 2.2},
        improvement={"sharpe_improvement": 0.1},
        execution_metadata={"execution_mode": "qaoa_in_repo_simulator"}
    )
    
    assert "strategy_allocator" in state.components
    assert state.global_quantum_advantage == 0.1
    assert state.quantum_execution_stats["qaoa_in_repo_simulator"] == 1


def test_dashboard_update(quantum_dashboard):
    """Test dashboard update from validation result"""
    # Create mock validation result
    mock_result = {
        "validation_result": MagicMock(),
        "quantum_enhanced": True
    }
    
    # Configure mock
    mock_result["validation_result"].component = "strategy_allocator"
    mock_result["validation_result"].metrics = {
        "sharpe_ratio": 2.0,
        "win_rate": 0.6,
        "quantum_sharpe": 2.2,
        "quantum_win_rate": 0.65,
        "sharpe_improvement": 0.1,
        "win_rate_improvement": 0.083
    }
    mock_result["validation_result"].quantum_metadata = {
        "execution_mode": "qaoa_in_repo_simulator",
        "n_qubits": 5
    }
    
    # Update dashboard
    success = quantum_dashboard.update_from_validation(mock_result)
    
    assert success == True
    assert "strategy_allocator" in quantum_dashboard.state.components


def test_dashboard_summaries(quantum_dashboard):
    """Test dashboard summary methods"""
    # Add test data
    quantum_dashboard.state.update_component(
        component_name="strategy_allocator",
        classical_metrics={"sharpe_ratio": 2.0},
        quantum_metrics={"quantum_sharpe": 2.2},
        improvement={"sharpe_improvement": 0.1},
        execution_metadata={"execution_mode": "qaoa_in_repo_simulator"}
    )
    
    # Test component summary
    summary = quantum_dashboard.get_component_summary("strategy_allocator")
    assert summary is not None
    assert summary["component_name"] == "strategy_allocator"
    assert summary["quantum_improvement"]["sharpe_improvement"] == 0.1
    
    # Test global summary
    global_summary = quantum_dashboard.get_global_summary()
    assert global_summary["global_quantum_advantage"] == 0.1
    assert global_summary["component_count"] == 1
    
    # Test quantum advantage report
    report = quantum_dashboard.get_quantum_advantage_report()
    assert report["global_quantum_advantage"] == 0.1
    assert "strategy_allocator" in report["component_breakdown"]


def test_dashboard_reset(quantum_dashboard):
    """Test dashboard reset"""
    # Add test data
    quantum_dashboard.state.update_component(
        component_name="strategy_allocator",
        classical_metrics={"sharpe_ratio": 2.0},
        quantum_metrics={"quantum_sharpe": 2.2},
        improvement={"sharpe_improvement": 0.1},
        execution_metadata={"execution_mode": "qaoa_in_repo_simulator"}
    )
    
    # Reset dashboard
    quantum_dashboard.reset()
    
    assert len(quantum_dashboard.state.components) == 0
    assert quantum_dashboard.state.global_quantum_advantage == 0.0