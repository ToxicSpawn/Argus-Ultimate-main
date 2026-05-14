"""
Test cases for Advanced Quantum Circuit Optimizer
"""

import pytest
from quantum.advanced.quantum_circuit_optimizer import (
    AdvancedQuantumCircuitOptimizer,
    QuantumCircuitMetrics,
    QuantumCircuitProfile,
    QuantumHardwareType,
    OptimizationObjective
)


@pytest.fixture
def circuit_optimizer():
    """Create a test circuit optimizer"""
    return AdvancedQuantumCircuitOptimizer()


def test_circuit_metrics():
    """Test QuantumCircuitMetrics class"""
    metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=20,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    assert metrics.depth == 100
    assert metrics.gate_count == 500
    assert metrics.qubit_count == 20
    assert metrics.fidelity == 0.95
    assert metrics.estimated_latency_ms == 50
    
    # Test score calculation
    score = metrics.calculate_score(OptimizationObjective.BALANCED)
    assert 0 <= score <= 1


def test_circuit_profile():
    """Test QuantumCircuitProfile class"""
    original_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=20,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    profile = QuantumCircuitProfile(
        circuit_id="test_circuit",
        original_metrics=original_metrics,
        hardware_type=QuantumHardwareType.IBM_QISKIT
    )
    
    assert profile.circuit_id == "test_circuit"
    assert profile.hardware_type == QuantumHardwareType.IBM_QISKIT
    assert profile.improvement_ratio() == 0.0  # No optimization yet


def test_optimization_strategy_selection(circuit_optimizer):
    """Test strategy selection logic"""
    original_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=20,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    profile = QuantumCircuitProfile(
        circuit_id="test_circuit",
        original_metrics=original_metrics,
        hardware_type=QuantumHardwareType.IBM_QISKIT
    )
    
    # First iteration should use gate decomposition
    strategy = circuit_optimizer._select_optimization_strategy(profile, 0)
    assert strategy == 'gate_decomposition'
    
    # For IBM hardware with low fidelity, should use noise-aware
    profile.optimized_metrics = QuantumCircuitMetrics(
        depth=80,
        gate_count=400,
        qubit_count=20,
        fidelity=0.85,  # Low fidelity
        estimated_latency_ms=40
    )
    profile.hardware_type = QuantumHardwareType.IBM_QISKIT
    strategy = circuit_optimizer._select_optimization_strategy(profile, 1)
    assert strategy == 'noise_aware'


def test_circuit_optimization(circuit_optimizer):
    """Test full circuit optimization workflow"""
    original_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=20,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    profile = QuantumCircuitProfile(
        circuit_id="test_circuit",
        original_metrics=original_metrics,
        hardware_type=QuantumHardwareType.IBM_QISKIT
    )
    
    # Optimize the circuit
    optimized_profile = circuit_optimizer.optimize_circuit(profile)
    
    # Verify optimization was applied
    assert len(optimized_profile.optimization_history) > 0
    assert optimized_profile.improvement_ratio() > 0
    
    # Verify metrics improved
    assert optimized_profile.optimized_metrics.depth <= original_metrics.depth
    assert optimized_profile.optimized_metrics.gate_count <= original_metrics.gate_count
    assert optimized_profile.optimized_metrics.fidelity >= original_metrics.fidelity


def test_optimization_analysis(circuit_optimizer):
    """Test optimization analysis"""
    original_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=20,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    profile = QuantumCircuitProfile(
        circuit_id="test_circuit",
        original_metrics=original_metrics,
        hardware_type=QuantumHardwareType.IBM_QISKIT
    )
    
    # Optimize the circuit
    optimized_profile = circuit_optimizer.optimize_circuit(profile)
    
    # Analyze the optimization
    analysis = circuit_optimizer.analyze_optimization(optimized_profile)
    
    assert analysis["status"] == "optimization_completed"
    assert analysis["improvement_ratio"] > 0
    assert len(analysis["optimization_history"]) > 0


def test_hardware_efficiency_calculation(circuit_optimizer):
    """Test hardware efficiency calculation"""
    metrics = QuantumCircuitMetrics(
        depth=50,
        gate_count=200,
        qubit_count=10,
        fidelity=0.98,
        estimated_latency_ms=25
    )
    
    # Test efficiency for different hardware
    ibm_efficiency = circuit_optimizer._calculate_hardware_efficiency(metrics, QuantumHardwareType.IBM_QISKIT)
    simulator_efficiency = circuit_optimizer._calculate_hardware_efficiency(metrics, QuantumHardwareType.SIMULATOR)
    
    assert 0 <= ibm_efficiency <= 1
    assert 0 <= simulator_efficiency <= 1
    assert simulator_efficiency > ibm_efficiency  # Simulator should be more efficient


def test_circuit_profile_creation(circuit_optimizer):
    """Test circuit profile creation"""
    original_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=20,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    profile = circuit_optimizer.create_circuit_profile(
        circuit_id="test_circuit",
        hardware_type=QuantumHardwareType.DWAVE,
        initial_metrics=original_metrics
    )
    
    assert profile.circuit_id == "test_circuit"
    assert profile.hardware_type == QuantumHardwareType.DWAVE
    assert profile.original_metrics.depth == 100