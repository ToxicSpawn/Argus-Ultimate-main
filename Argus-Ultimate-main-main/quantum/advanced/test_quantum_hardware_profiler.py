"""
Test cases for Quantum Hardware Profiler
"""

import pytest
import numpy as np
from datetime import datetime
from quantum.advanced.quantum_hardware_profiler import (
    QuantumHardwareProfiler,
    QuantumBackendProfile,
    QuantumCircuitProfile,
    QuantumBackendType,
    HardwareOptimizationResult,
    QuantumHardwareSelector,
    QuantumExecutionManager,
    QuantumResourceManager,
    QuantumHardwareMonitor
)


@pytest.fixture
def hardware_profiler():
    """Create a test hardware profiler"""
    return QuantumHardwareProfiler()


def test_backend_profile():
    """Test QuantumBackendProfile class"""
    profile = QuantumBackendProfile(
        backend_type=QuantumBackendType.IBM_QISKIT,
        name='test_backend',
        version='1.0',
        qubits=7,
        topology='heavy-hex',
        gate_error_rate=0.001,
        readout_error_rate=0.05,
        qubit_coherence_time_us=100,
        gate_time_ns=50,
        max_circuit_depth=1000,
        queue_depth=50,
        queue_time_ms=1000,
        supported_gates=['rx', 'ry', 'rz', 'cx']
    )
    
    assert profile.calculate_quantum_volume() > 0
    assert profile.to_dict()['backend_type'] == 'IBM_QISKIT'


def test_circuit_profile():
    """Test QuantumCircuitProfile class"""
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    assert circuit.calculate_circuit_complexity() > 0


def test_hardware_optimization_result():
    """Test HardwareOptimizationResult class"""
    profiler = QuantumHardwareProfiler()
    backend = profiler.get_backend_profile('simulator_statevector')
    
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    result = HardwareOptimizationResult(
        backend=backend,
        circuit=circuit,
        optimized_circuit=circuit,  # Same for test
        optimization_metrics={
            'qubit_reduction': 0.0,
            'depth_reduction': 0.0,
            'gate_reduction': 0.0,
            'fidelity_improvement': 0.0,
            'time_reduction': 0.0
        },
        execution_time_ms=100
    )
    
    assert result.to_dict()['backend'] == 'simulator_statevector'


def test_profiler_initialization(hardware_profiler):
    """Test profiler initialization"""
    assert len(hardware_profiler.backend_profiles) >= 4  # Should have at least 4 predefined backends
    assert 'simulator_statevector' in hardware_profiler.backend_profiles
    assert 'ibm_lagos' in hardware_profiler.backend_profiles


def test_get_backend_profile(hardware_profiler):
    """Test getting backend profiles"""
    # Test getting existing backend
    backend = hardware_profiler.get_backend_profile('simulator_statevector')
    assert backend.name == 'simulator_statevector'
    assert backend.backend_type == QuantumBackendType.SIMULATOR
    
    # Test getting non-existent backend
    with pytest.raises(ValueError):
        hardware_profiler.get_backend_profile('nonexistent')


def test_add_custom_backend(hardware_profiler):
    """Test adding custom backend"""
    custom_profile = QuantumBackendProfile(
        backend_type=QuantumBackendType.CUSTOM,
        name='custom_backend',
        version='1.0',
        qubits=10,
        topology='linear',
        gate_error_rate=0.002,
        readout_error_rate=0.03,
        qubit_coherence_time_us=80,
        gate_time_ns=60,
        max_circuit_depth=500,
        queue_depth=20,
        queue_time_ms=500,
        supported_gates=['rx', 'ry', 'cx']
    )
    
    hardware_profiler.add_custom_backend(custom_profile)
    assert 'custom_backend' in hardware_profiler.backend_profiles


def test_profile_backend(hardware_profiler):
    """Test profiling a backend"""
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    profile_result = hardware_profiler.profile_backend('simulator_statevector', circuit)
    
    assert profile_result['backend']['name'] == 'simulator_statevector'
    assert profile_result['circuit']['id'] == 'test_circuit'
    assert 'compatibility' in profile_result
    assert 'performance' in profile_result
    assert 'optimization_potential' in profile_result


def test_optimize_for_backend(hardware_profiler):
    """Test optimizing circuit for backend"""
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=10,
        depth=100,
        gate_count=500,
        gate_types={'rx': 100, 'ry': 100, 'cx': 300},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), 
                     (5, 6), (6, 7), (7, 8), (8, 9), (0, 5)]
    )
    
    result = hardware_profiler.optimize_for_backend('ibm_lagos', circuit)
    
    assert result.backend.name == 'ibm_lagos'
    assert result.circuit.circuit_id == 'test_circuit'
    assert result.optimized_circuit.num_qubits <= circuit.num_qubits
    assert result.optimized_circuit.depth <= circuit.depth
    assert result.optimized_circuit.gate_count <= circuit.gate_count
    assert 'optimization_metrics' in result.to_dict()


def test_compare_backends(hardware_profiler):
    """Test comparing backends"""
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    comparison = hardware_profiler.compare_backends(circuit, ['simulator_statevector', 'ibm_lagos'])
    
    assert 'circuit' in comparison
    assert 'backends' in comparison
    assert len(comparison['backends']) == 2
    assert 'best_backend' in comparison


def test_hardware_selector(hardware_profiler):
    """Test quantum hardware selector"""
    selector = QuantumHardwareSelector(hardware_profiler)
    
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    backend_name, selection_info = selector.select_best_backend(circuit, 'balanced')
    
    assert backend_name in hardware_profiler.backend_profiles
    assert 'reasoning' in selection_info
    assert 'expected_fidelity' in selection_info


def test_execution_manager(hardware_profiler):
    """Test quantum execution manager"""
    selector = QuantumHardwareSelector(hardware_profiler)
    manager = QuantumExecutionManager(hardware_profiler, selector)
    
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    result = manager.execute_circuit(circuit, 'balanced')
    
    assert result['status'] in ['success', 'fallback_success']
    assert 'backend' in result
    assert 'execution_time_ms' in result
    assert 'quantum_results' in result


def test_resource_manager(hardware_profiler):
    """Test quantum resource manager"""
    manager = QuantumResourceManager(hardware_profiler)
    
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    # Test allocation
    allocation = manager.allocate_resources(circuit, priority=1)
    assert allocation['status'] == 'allocated'
    assert allocation['backend'] in hardware_profiler.backend_profiles
    
    # Test release
    release = manager.release_resources('test_circuit')
    assert release['status'] == 'released'


def test_hardware_monitor(hardware_profiler):
    """Test quantum hardware monitor"""
    monitor = QuantumHardwareMonitor(hardware_profiler)
    
    # Test updating availability
    update_result = monitor.update_backend_availability(
        'simulator_statevector',
        available=True,
        queue_depth=0,
        estimated_wait_time_ms=0
    )
    assert update_result['status'] == 'updated'
    
    # Test getting system status
    status = monitor.get_system_status()
    assert 'total_backends' in status
    assert 'available_backends' in status


def test_connectivity_optimization(hardware_profiler):
    """Test connectivity optimization"""
    # Test with heavy-hex topology
    backend = hardware_profiler.get_backend_profile('ibm_lagos')
    
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=7,
        depth=20,
        gate_count=100,
        gate_types={'rx': 30, 'ry': 30, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (0, 3)]  # Some non-local connections
    )
    
    optimized_connectivity = hardware_profiler._optimize_connectivity(backend, circuit)
    
    # Should have fewer connections for non-all-to-all topology
    assert len(optimized_connectivity) < len(circuit.connectivity)
    
    # All connections should be local for heavy-hex
    for (q1, q2) in optimized_connectivity:
        assert abs(q1 - q2) <= 1, f"Non-local connection found: {(q1, q2)}"


def test_optimization_potential(hardware_profiler):
    """Test optimization potential calculation"""
    backend = hardware_profiler.get_backend_profile('ibm_lagos')
    
    # Circuit that exceeds backend capabilities
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=10,  # More than backend's 7 qubits
        depth=1500,   # More than backend's max depth
        gate_count=1000,
        gate_types={'rx': 200, 'ry': 200, 'cx': 600},
        connectivity=[(i, i+1) for i in range(9)]
    )
    
    potential = hardware_profiler._calculate_optimization_potential(backend, circuit)
    
    assert potential['qubit_reduction'] > 0
    assert potential['depth_reduction'] > 0
    assert potential['overall'] > 0


def test_fidelity_calculation(hardware_profiler):
    """Test fidelity calculation"""
    backend = hardware_profiler.get_backend_profile('ibm_lagos')
    
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=50,
        gate_count=200,
        gate_types={'rx': 50, 'ry': 50, 'cx': 100},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    fidelity = hardware_profiler._calculate_expected_fidelity(backend, circuit)
    
    # Should be less than 1.0 for real hardware
    assert 0 < fidelity < 1.0
    
    # Simulator should have perfect fidelity
    simulator = hardware_profiler.get_backend_profile('simulator_statevector')
    simulator_fidelity = hardware_profiler._calculate_expected_fidelity(simulator, circuit)
    assert simulator_fidelity == 1.0


def test_execution_time_calculation(hardware_profiler):
    """Test execution time calculation"""
    backend = hardware_profiler.get_backend_profile('ibm_lagos')
    
    circuit = QuantumCircuitProfile(
        circuit_id='test_circuit',
        num_qubits=5,
        depth=50,
        gate_count=200,
        gate_types={'rx': 50, 'ry': 50, 'cx': 100},
        connectivity=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    
    exec_time = hardware_profiler._calculate_expected_execution_time(backend, circuit)
    
    # Should be reasonable time for the circuit
    assert exec_time > 0
    assert exec_time < 1000  # Less than 1 second


def test_resource_allocation_optimization(hardware_profiler):
    """Test resource allocation optimization"""
    manager = QuantumResourceManager(hardware_profiler)
    
    # Allocate multiple circuits to create load imbalance
    circuit1 = QuantumCircuitProfile(
        circuit_id='circuit1',
        num_qubits=3,
        depth=10,
        gate_count=50,
        gate_types={'rx': 10, 'ry': 10, 'cx': 30},
        connectivity=[(0, 1), (1, 2)]
    )
    
    circuit2 = QuantumCircuitProfile(
        circuit_id='circuit2',
        num_qubits=4,
        depth=15,
        gate_count=70,
        gate_types={'rx': 15, 'ry': 15, 'cx': 40},
        connectivity=[(0, 1), (1, 2), (2, 3)]
    )
    
    # Allocate both to same backend to create imbalance
    manager.allocate_resources(circuit1, priority=2)
    manager.allocate_resources(circuit2, priority=3)
    
    # Get optimization suggestions
    optimization = manager.optimize_resource_allocation()
    
    assert 'suggested_actions' in optimization
    
    # Clean up
    manager.release_resources('circuit1')
    manager.release_resources('circuit2')