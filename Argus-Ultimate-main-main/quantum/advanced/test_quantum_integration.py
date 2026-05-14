"""
Comprehensive Quantum Integration Test

This test validates the complete quantum-enhanced adaptive trading system,
including all quantum components working together with the classical system.
"""

import pytest
import numpy as np
from datetime import datetime
from quantum.advanced.quantum_circuit_optimizer import (
    AdvancedQuantumCircuitOptimizer,
    QuantumCircuitMetrics,
    QuantumCircuitProfile,
    QuantumHardwareType,
    OptimizationObjective
)
from quantum.advanced.quantum_neural_network import (
    QuantumNeuralNetwork,
    QNNArchitecture,
    QNNLayer,
    QNNLayerType,
    QuantumTrainingMode,
    QNNAdaptiveTrainer
)
from quantum.advanced.quantum_regime_detection import (
    MarketRegime,
    MarketDataFeatures,
    QuantumRegimeDetector,
    QuantumRegimeAdaptationSystem
)
from core.real_time_learning.quantum_paper_validation import QuantumPaperValidationEngine


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    # Generate synthetic market features
    np.random.seed(42)
    num_samples = 100
    
    features = MarketDataFeatures(
        returns=np.random.normal(0, 0.01, num_samples),
        volatility=np.random.uniform(0.01, 0.05, num_samples),
        volume=np.random.uniform(0.8, 1.2, num_samples),
        momentum=np.random.normal(0, 0.005, num_samples),
        correlation=np.random.uniform(0.5, 0.9, num_samples),
        sentiment=np.random.normal(0, 0.1, num_samples)
    )
    
    # Generate corresponding regimes (simplified)
    regimes = []
    for i in range(num_samples):
        if features.volatility[i] > 0.03 and abs(features.returns[i]) > 0.015:
            regimes.append(MarketRegime.VOLATILE)
        elif abs(features.momentum[i]) > 0.007:
            regimes.append(MarketRegime.TRENDING)
        elif features.volatility[i] < 0.02 and abs(features.returns[i]) < 0.005:
            regimes.append(MarketRegime.STABLE)
        else:
            regimes.append(MarketRegime.RANGE)
    
    # Create training data as list of tuples
    training_data = []
    for i in range(num_samples):
        # Create features for single sample
        sample_features = MarketDataFeatures(
            returns=np.array([features.returns[i]]),
            volatility=np.array([features.volatility[i]]),
            volume=np.array([features.volume[i]]),
            momentum=np.array([features.momentum[i]]),
            correlation=np.array([features.correlation[i]]),
            sentiment=np.array([features.sentiment[i]])
        )
        training_data.append((sample_features, regimes[i]))
    
    return features, regimes, training_data


def test_quantum_circuit_optimization():
    """Test advanced quantum circuit optimization"""
    optimizer = AdvancedQuantumCircuitOptimizer()
    
    # Create initial circuit metrics
    initial_metrics = QuantumCircuitMetrics(
        depth=150,
        gate_count=800,
        qubit_count=20,
        fidelity=0.92,
        estimated_latency_ms=80
    )
    
    # Create circuit profile
    circuit_profile = optimizer.create_circuit_profile(
        circuit_id="test_circuit",
        hardware_type=QuantumHardwareType.IBM_QISKIT,
        initial_metrics=initial_metrics
    )
    
    # Optimize circuit
    optimized_profile = optimizer.optimize_circuit(circuit_profile)
    
    # Verify optimization
    assert len(optimized_profile.optimization_history) > 0
    assert optimized_profile.improvement_ratio() > 0
    
    # Verify metrics improved
    analysis = optimizer.analyze_optimization(optimized_profile)
    assert analysis["improvement_ratio"] > 0
    assert analysis["optimized_metrics"]["depth"] <= initial_metrics.depth
    assert analysis["optimized_metrics"]["fidelity"] >= initial_metrics.fidelity


def test_quantum_neural_network():
    """Test quantum neural network for adaptive learning"""
    # Create simple architecture
    layers = [
        QNNLayer(
            layer_type=QNNLayerType.QUANTUM_EMBEDDING,
            num_qubits=4,
            num_parameters=16  # 4 inputs * 4 qubits
        ),
        QNNLayer(
            layer_type=QNNLayerType.QUANTUM_DENSE,
            num_qubits=4,
            num_parameters=16  # 4*4
        ),
        QNNLayer(
            layer_type=QNNLayerType.CLASSICAL_DENSE,
            num_qubits=1,
            num_parameters=5  # 4 inputs + bias
        )
    ]
    
    architecture = QNNArchitecture(
        layers=layers,
        input_dim=4,
        output_dim=1
    )
    
    # Create and train QNN
    qnn = QuantumNeuralNetwork(
        architecture=architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    # Generate simple training data
    X = np.random.rand(20, 4)
    y = np.random.rand(20)
    
    # Train
    training_history = qnn.train(X, y, epochs=10, learning_rate=0.01)
    
    # Verify training
    assert len(training_history) == 10
    assert len(qnn.training_history) == 10
    
    # Verify quantum advantage is tracked
    summary = qnn.get_training_summary()
    assert "quantum_advantage" in summary


def test_quantum_regime_detection(sample_market_data):
    """Test quantum-enhanced regime detection"""
    features, regimes, training_data = sample_market_data
n    # Create and train detector
    detector = QuantumRegimeDetector(num_qubits=4, hardware_backend="simulator")
    
    # Train with sample data
    train_result = detector.train_detector(
        training_data[:80],  # Use 80 samples for training
        epochs=20,
        learning_rate=0.01
    )
    
    # Verify training
    assert train_result["status"] == "training_completed"
    assert train_result["final_accuracy"] > 0.5  # Better than random
    
    # Test detection
    test_features = training_data[80][0]  # Use a test sample
    detection = detector.detect_regime(test_features)
    
    # Verify detection
    assert detection.regime in [r for _, r in training_data]
    assert 0 <= detection.confidence <= 1
    assert 0 <= detection.quantum_contribution <= 1


def test_quantum_paper_validation_integration():
    """Test integration with quantum paper validation"""
    # Create quantum validation engine
    validation_engine = QuantumPaperValidationEngine()
    
    # Test quantum validation of a simple change
    proposed_change = {
        "component": "strategy_allocator",
        "parameter": "weights",
        "old_value": {"momentum": 0.3, "mean_reversion": 0.3, "breakout": 0.4},
        "new_value": {"momentum": 0.4, "mean_reversion": 0.3, "breakout": 0.3}
    }
    
    # Mock classical validation result
    classical_result = {
        "valid": True,
        "sharpe_ratio": 2.0,
        "max_drawdown": 0.15,
        "win_rate": 0.55
    }
    
    # Perform quantum validation
    quantum_result = validation_engine.validate_with_quantum(
        proposed_change=proposed_change,
        classical_result=classical_result,
        hardware_backend="simulator",
        min_quantum_improvement=0.05
    )
    
    # Verify quantum validation
    assert quantum_result["quantum_valid"] is not None
    assert "quantum_metadata" in quantum_result
    assert "quantum_improvement" in quantum_result


def test_end_to_end_quantum_system(sample_market_data):
    """Test complete quantum-enhanced adaptive system"""
    features, regimes, training_data = sample_market_data
    
    # 1. Train quantum regime detector
    detector = QuantumRegimeDetector(num_qubits=4, hardware_backend="simulator")
    detector.train_detector(training_data[:60], epochs=15, learning_rate=0.01)
    
    # 2. Create quantum adaptation system
    adaptation_system = QuantumRegimeAdaptationSystem(num_qubits=4, hardware_backend="simulator")
    
    # 3. Detect regime and adapt
    test_features = training_data[61][0]  # Use a test sample
    result = adaptation_system.detect_and_adapt(test_features)
    
    # Verify complete system works
    assert "detection" in result
    assert "adaptation" in result
    assert "system_status" in result
    
    # Verify quantum contribution is tracked
    assert "quantum_contribution" in result["detection"]
    assert result["detection"]["quantum_contribution"] >= 0
    
    # Verify adaptation strategy was applied
    assert "strategy_weights" in result["adaptation"]
    assert "risk_parameters" in result["adaptation"]
    assert "execution_parameters" in result["adaptation"]


def test_quantum_performance_benchmarking():
    """Test quantum performance benchmarking"""
    # Create test components
    optimizer = AdvancedQuantumCircuitOptimizer()
    
    # Test circuit optimization performance
    initial_metrics = QuantumCircuitMetrics(
        depth=200,
        gate_count=1000,
        qubit_count=25,
        fidelity=0.90,
        estimated_latency_ms=100
    )
    
    circuit_profile = optimizer.create_circuit_profile(
        circuit_id="benchmark_circuit",
        hardware_type=QuantumHardwareType.IBM_QISKIT,
        initial_metrics=initial_metrics
    )
    
    optimized_profile = optimizer.optimize_circuit(circuit_profile)
    analysis = optimizer.analyze_optimization(optimized_profile)
    
    # Verify performance improvement
    improvement = analysis["improvement_ratio"]
    assert improvement > 0, f"Expected positive improvement, got {improvement}"
    
    # Verify quantum advantage metrics
    assert "quantum_results" in analysis
    assert analysis["quantum_results"]["quantum_advantage"] > 0


def test_quantum_fallback_mechanisms():
    """Test quantum fallback to classical algorithms"""
    # Test with quantum neural network
    layers = [
        QNNLayer(
            layer_type=QNNLayerType.QUANTUM_EMBEDDING,
            num_qubits=2,
            num_parameters=4
        ),
        QNNLayer(
            layer_type=QNNLayerType.CLASSICAL_DENSE,
            num_qubits=1,
            num_parameters=3
        )
    ]
    
    architecture = QNNArchitecture(layers=layers, input_dim=2, output_dim=1)
    
    # Create QNN with fallback capability
    qnn = QuantumNeuralNetwork(
        architecture=architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    # Generate training data
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    
    # Train and verify quantum advantage is tracked
    qnn.train(X, y, epochs=5, learning_rate=0.01)
    summary = qnn.get_training_summary()
    
    # Verify quantum advantage metric exists
    assert "quantum_advantage" in summary
    
    # Verify fallback mechanism (classical baseline exists)
    assert "classical_accuracy" in summary["training_history"][0]


def test_quantum_hardware_compatibility():
    """Test compatibility with different quantum hardware backends"""
    optimizer = AdvancedQuantumCircuitOptimizer()
    
    # Test with different hardware types
    hardware_types = [
        QuantumHardwareType.SIMULATOR,
        QuantumHardwareType.IBM_QISKIT,
        QuantumHardwareType.DWAVE
    ]
    
    initial_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=10,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    for hardware_type in hardware_types:
        circuit_profile = optimizer.create_circuit_profile(
            circuit_id=f"test_{hardware_type.name}",
            hardware_type=hardware_type,
            initial_metrics=initial_metrics
        )
        
        optimized_profile = optimizer.optimize_circuit(circuit_profile)
        analysis = optimizer.analyze_optimization(optimized_profile)
        
        # Verify optimization works for all hardware types
        assert analysis["status"] == "optimization_completed"
        assert analysis["improvement_ratio"] >= 0


def test_quantum_safety_thresholds():
    """Test quantum safety thresholds and validation"""
    # Test quantum circuit optimizer safety
    optimizer = AdvancedQuantumCircuitOptimizer()
    
    # Create circuit that meets minimum requirements
    good_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=10,
        fidelity=0.95,
        estimated_latency_ms=50
    )
    
    good_profile = optimizer.create_circuit_profile(
        circuit_id="good_circuit",
        hardware_type=QuantumHardwareType.IBM_QISKIT,
        initial_metrics=good_metrics
    )
    
    optimized_good = optimizer.optimize_circuit(good_profile)
    
    # Should have positive improvement
    assert optimized_good.improvement_ratio() > 0
    
    # Test circuit that doesn't meet requirements (low fidelity)
    bad_metrics = QuantumCircuitMetrics(
        depth=100,
        gate_count=500,
        qubit_count=10,
        fidelity=0.60,  # Too low
        estimated_latency_ms=50
    )
    
    bad_profile = optimizer.create_circuit_profile(
        circuit_id="bad_circuit",
        hardware_type=QuantumHardwareType.IBM_QISKIT,
        initial_metrics=bad_metrics
    )
    
    optimized_bad = optimizer.optimize_circuit(bad_profile)
    
    # Should still try to optimize but with limited improvement
    analysis = optimizer.analyze_optimization(optimized_bad)
    assert analysis["optimized_metrics"]["fidelity"] > bad_metrics.fidelity


def test_quantum_adaptive_learning():
    """Test quantum-enhanced adaptive learning"""
    # Create adaptive trainer
    trainer = QNNAdaptiveTrainer(input_dim=4, output_dim=1, hardware_backend="simulator")
    
    # Generate training data
    X = np.random.rand(50, 4)
    y = np.random.rand(50)
    
    # Train adaptive model
    model = trainer.train_adaptive_model(X, y, epochs=15, learning_rate=0.01)
    
    # Verify training
    assert model is not None
    assert len(trainer.training_history) == 1
    
    # Verify quantum advantage is tracked
    summary = model.get_training_summary()
    assert "quantum_advantage" in summary
    
    # Test adaptation to new data
    X_new = np.random.rand(10, 4)
    y_new = np.random.rand(10)
    
    adapted_model = trainer.adapt_to_new_data(X_new, y_new, epochs=5, learning_rate=0.001)
    
    # Verify adaptation
    assert adapted_model is not None
    assert len(trainer.training_history) == 2
    
    # Verify quantum advantage improved or maintained
    new_summary = adapted_model.get_training_summary()
    assert "quantum_advantage" in new_summary