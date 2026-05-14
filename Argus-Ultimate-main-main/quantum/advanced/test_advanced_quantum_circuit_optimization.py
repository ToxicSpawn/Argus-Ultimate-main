# pyright: reportMissingImports=false
"""Tests for advanced quantum circuit optimization."""

import pytest

from quantum.advanced.advanced_quantum_circuit_optimization import (
    AdvancedQuantumCircuitOptimizer,
    OptimizationObjective,
    QuantumCircuitMetrics,
    QuantumGateOperation,
    QuantumHardwareType,
)


@pytest.fixture
def optimizer():
    """Create an optimizer instance."""
    return AdvancedQuantumCircuitOptimizer()


@pytest.fixture
def rich_metrics():
    """Create baseline metrics with profiling detail."""
    return QuantumCircuitMetrics(
        depth=120,
        gate_count=640,
        qubit_count=12,
        fidelity=0.91,
        estimated_latency_ms=62.0,
        two_qubit_gate_count=240,
        measurement_count=12,
        swap_count=36,
        parallelism_factor=1.7,
        estimated_error_rate=0.055,
        connectivity_overhead=0.18,
        noise_resilience=0.42,
    )


def test_profile_circuit_with_explicit_operations(optimizer, rich_metrics):
    """Profiling should expose circuit structure and recommendations."""
    operations = [
        QuantumGateOperation("h", (0,)),
        QuantumGateOperation("cx", (0, 1), native=False),
        QuantumGateOperation("rz", (1,), parameters=(0.5,)),
        QuantumGateOperation("cx", (1, 2), native=False),
        QuantumGateOperation("measure", (0,)),
        QuantumGateOperation("measure", (1,)),
        QuantumGateOperation("measure", (2,)),
    ]

    profile = optimizer.create_circuit_profile(
        circuit_id="explicit_ops",
        hardware_type=QuantumHardwareType.IBM_QISKIT,
        initial_metrics=rich_metrics,
        operations=operations,
    )

    report = optimizer.profile_circuit(profile)

    assert report.operation_count >= rich_metrics.gate_count
    assert report.gate_histogram["CX"] == 2
    assert report.critical_path_depth > 0
    assert 0 <= report.entanglement_density <= 1
    assert isinstance(report.recommendations, list)
    assert report.recommendations


def test_metrics_tracking_and_visualization_payload(optimizer, rich_metrics):
    """Metrics tracking should include timeline, deltas, and plot-ready payload."""
    profile = optimizer.create_circuit_profile(
        circuit_id="tracking_case",
        hardware_type=QuantumHardwareType.IBM_QISKIT,
        initial_metrics=rich_metrics,
    )

    optimized = optimizer.optimize_circuit(profile, objective=OptimizationObjective.BALANCED, max_iterations=4)
    tracked = optimizer.track_metrics(optimized)
    payload = optimizer.generate_visualization_payload(optimized)

    assert tracked["improvement_ratio"] > 0
    assert len(tracked["timeline"]) >= 2
    assert tracked["delta"]["depth_reduction"] >= 0
    assert "plot_series" in payload
    assert len(payload["plot_series"]["labels"]) == len(tracked["timeline"])
    assert "Optimization Summary" in payload["summary"]
    assert "Optimization Timeline" in payload["timeline"]
    assert "Qubit Allocation" in payload["allocation"]


def test_noise_aware_strategy_selection_uses_error_signals(optimizer, rich_metrics):
    """Low-fidelity or noisy circuits should trigger noise-aware optimization."""
    profile = optimizer.create_circuit_profile(
        circuit_id="noise_selection",
        hardware_type=QuantumHardwareType.IBM_QISKIT,
        initial_metrics=rich_metrics,
    )
    profile.optimized_metrics = QuantumCircuitMetrics(
        depth=95,
        gate_count=540,
        qubit_count=12,
        fidelity=0.86,
        estimated_latency_ms=58.0,
        two_qubit_gate_count=200,
        measurement_count=12,
        swap_count=24,
        parallelism_factor=1.8,
        estimated_error_rate=0.061,
        connectivity_overhead=0.14,
        noise_resilience=0.35,
        hardware_efficiency=0.50,
    )

    strategy = optimizer._select_optimization_strategy(profile, 1)
    assert strategy == "noise_aware"


def test_visualization_helpers_are_human_readable(optimizer, rich_metrics):
    """Visualization helpers should return stable readable strings."""
    profile = optimizer.create_circuit_profile(
        circuit_id="viz_case",
        hardware_type=QuantumHardwareType.DWAVE,
        initial_metrics=rich_metrics,
    )
    optimized = optimizer.optimize_circuit(profile, max_iterations=3)

    summary = optimizer.visualize_optimization_summary(optimized)
    timeline = optimizer.visualize_metrics_timeline(optimized)
    allocation = optimizer.visualize_qubit_allocation(optimized)

    assert "viz_case" in summary
    assert "Depth" in summary
    assert "iteration_1" in timeline
    assert "q0 -> p0" in allocation


def test_analyze_optimization_includes_profiling_and_visualizations(optimizer, rich_metrics):
    """Analysis output should include the new profiling and visualization surfaces."""
    profile = optimizer.create_circuit_profile(
        circuit_id="analysis_case",
        hardware_type=QuantumHardwareType.IONQ,
        initial_metrics=rich_metrics,
    )
    optimized = optimizer.optimize_circuit(profile, objective=OptimizationObjective.FIDELITY, max_iterations=4)
    analysis = optimizer.analyze_optimization(optimized)

    assert analysis["status"] == "optimization_completed"
    assert analysis["profiling_report"]["estimated_success_probability"] > 0
    assert analysis["metrics_tracking"]["strategy_impact"]
    assert "plot_series" in analysis["visualizations"]


def test_invalid_metrics_raise_clear_errors(optimizer):
    """Invalid metrics should fail fast with a clear ValueError."""
    bad_metrics = QuantumCircuitMetrics(
        depth=0,
        gate_count=10,
        qubit_count=2,
        fidelity=0.9,
        estimated_latency_ms=1.0,
    )

    with pytest.raises(ValueError):
        optimizer.create_circuit_profile(
            circuit_id="bad_case",
            hardware_type=QuantumHardwareType.SIMULATOR,
            initial_metrics=bad_metrics,
        )


def test_hardware_specific_efficiency_prefers_simulator_for_same_metrics(optimizer, rich_metrics):
    """Simulator efficiency should exceed constrained hardware for identical metrics."""
    ibm_efficiency = optimizer._calculate_hardware_efficiency(rich_metrics, QuantumHardwareType.IBM_QISKIT)
    sim_efficiency = optimizer._calculate_hardware_efficiency(rich_metrics, QuantumHardwareType.SIMULATOR)

    assert 0 <= ibm_efficiency <= 1
    assert 0 <= sim_efficiency <= 1
    assert sim_efficiency > ibm_efficiency
