"""
Test cases for Quantum Neural Network
"""

import pytest
import numpy as np
from quantum.advanced.quantum_neural_network import (
    QuantumNeuralNetwork,
    QNNArchitecture,
    QNNLayer,
    QNNLayerType,
    QuantumTrainingMode,
    QNNTrainingMetrics,
    QNNAdaptiveTrainer
)


@pytest.fixture
def simple_qnn_architecture():
    """Create a simple QNN architecture for testing"""
    return QNNArchitecture(
        layers=[
            QNNLayer(
                layer_type=QNNLayerType.QUANTUM_EMBEDDING,
                num_qubits=2,
                num_parameters=4  # 2 inputs * 2 qubits
            ),
            QNNLayer(
                layer_type=QNNLayerType.CLASSICAL_DENSE,
                num_qubits=1,
                num_parameters=3  # 2 inputs + bias
            )
        ],
        input_dim=2,
        output_dim=1
    )


def test_qnn_layer_initialization():
    """Test QNN layer initialization"""
    layer = QNNLayer(
        layer_type=QNNLayerType.QUANTUM_EMBEDDING,
        num_qubits=2,
        num_parameters=4
    )
    
    # Test initialization
    layer.initialize_parameters(seed=42)
    assert layer.parameters is not None
    assert layer.parameters.shape == (4,)
    
    # Test parameter range
    assert np.all(layer.parameters >= -np.pi)
    assert np.all(layer.parameters <= np.pi)


def test_qnn_architecture_validation():
    """Test QNN architecture validation"""
    # Valid architecture
    valid_arch = QNNArchitecture(
        layers=[
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
        ],
        input_dim=2,
        output_dim=1
    )
    
    # Should not raise exception
    valid_arch.__post_init__()
    
    # Invalid architecture (input dimension mismatch)
    with pytest.raises(ValueError):
        invalid_arch = QNNArchitecture(
            layers=[
                QNNLayer(
                    layer_type=QNNLayerType.QUANTUM_EMBEDDING,
                    num_qubits=3,  # Doesn't match input_dim
                    num_parameters=6
                ),
                QNNLayer(
                    layer_type=QNNLayerType.CLASSICAL_DENSE,
                    num_qubits=1,
                    num_parameters=3
                )
            ],
            input_dim=2,
            output_dim=1
        )
        invalid_arch.__post_init__()


def test_qnn_initialization(simple_qnn_architecture):
    """Test QNN initialization"""
    qnn = QuantumNeuralNetwork(
        architecture=simple_qnn_architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    assert len(qnn.architecture.layers) == 2
    assert qnn.training_mode == QuantumTrainingMode.ADAPTIVE
    assert qnn.hardware_backend == "simulator"
    assert len(qnn.training_history) == 0


def test_qnn_forward_pass(simple_qnn_architecture):
    """Test QNN forward pass"""
    qnn = QuantumNeuralNetwork(
        architecture=simple_qnn_architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    # Test input
    test_input = np.array([0.5, -0.5])
    
    # Forward pass should not raise exception
    output = qnn.forward(test_input)
    
    assert output.shape == (1,)
    assert isinstance(output[0], (int, float))


def test_qnn_training_metrics():
    """Test QNN training metrics"""
    metrics = QNNTrainingMetrics(
        epoch=1,
        loss=0.1,
        accuracy=0.9,
        quantum_fidelity=0.95,
        classical_accuracy=0.85,
        quantum_advantage=0.05,
        training_time_ms=100
    )
    
    metrics_dict = metrics.to_dict()
    assert metrics_dict['epoch'] == 1
    assert metrics_dict['quantum_advantage'] == 0.05


def test_qnn_training(simple_qnn_architecture):
    """Test QNN training workflow"""
    qnn = QuantumNeuralNetwork(
        architecture=simple_qnn_architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    # Simple training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR-like pattern
    
    # Train for a few epochs
    training_history = qnn.train(X, y, epochs=5, learning_rate=0.1)
    
    assert len(training_history) == 5
    assert len(qnn.training_history) == 5
    
    # Verify metrics were recorded
    for metrics in training_history:
        assert isinstance(metrics.loss, float)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.quantum_fidelity <= 1


def test_qnn_prediction(simple_qnn_architecture):
    """Test QNN prediction"""
    qnn = QuantumNeuralNetwork(
        architecture=simple_qnn_architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    # Test input
    test_input = np.array([[0, 0], [1, 1]])
    
    # Should not raise exception
    predictions = qnn.predict(test_input)
    
    assert predictions.shape == (2,)


def test_qnn_quantum_advantage(simple_qnn_architecture):
    """Test quantum advantage calculation"""
    qnn = QuantumNeuralNetwork(
        architecture=simple_qnn_architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    # Before training, advantage should be 0
    assert qnn.get_quantum_advantage() == 0.0
    
    # After training, should have some advantage
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    qnn.train(X, y, epochs=3)
    
    advantage = qnn.get_quantum_advantage()
    assert isinstance(advantage, float)


def test_qnn_model_save_load(simple_qnn_architecture, tmp_path):
    """Test QNN model save/load"""
    qnn = QuantumNeuralNetwork(
        architecture=simple_qnn_architecture,
        training_mode=QuantumTrainingMode.ADAPTIVE,
        hardware_backend="simulator"
    )
    
    # Train briefly
    X = np.random.rand(5, 2)
    y = np.random.rand(5)
    qnn.train(X, y, epochs=2)
    
    # Save model
    model_file = tmp_path / "test_qnn.json"
    qnn.save_model(str(model_file))
    
    # Load model
    loaded_qnn = QuantumNeuralNetwork.load_model(str(model_file))
    
    assert len(loaded_qnn.architecture.layers) == 2
    assert loaded_qnn.training_mode == QuantumTrainingMode.ADAPTIVE
    assert len(loaded_qnn.training_history) == 2


def test_adaptive_trainer():
    """Test adaptive QNN trainer"""
    trainer = QNNAdaptiveTrainer(input_dim=4, output_dim=1, hardware_backend="simulator")
    
    # Create architecture
    architecture = trainer.create_adaptive_architecture(num_quantum_layers=2, qubits_per_layer=2)
    assert len(architecture.layers) == 3  # 2 quantum + 1 classical
    
    # Test training
    X = np.random.rand(20, 4)
    y = np.random.rand(20)
    
    model = trainer.train_adaptive_model(X, y, epochs=3)
    assert trainer.current_model is not None
    assert len(trainer.training_history) == 1
    
    # Test prediction
    predictions = trainer.adaptive_predict(X[:5])
    assert predictions.shape == (5,)
    
    # Test adaptation metrics
    metrics = trainer.get_adaptation_metrics()
    assert metrics["status"] == "trained"
    assert "current_quantum_advantage" in metrics