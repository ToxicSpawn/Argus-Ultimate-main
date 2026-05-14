"""
Hybrid Quantum-Classical Forecaster for Argus Ultimate
=======================================================
Combines:
1. Quantum Neural Network (QNN) for macro trends (1-24hr)
2. LSTM for micro noise (1-60min)
3. Dynamic weighting based on volatility

Dependencies:
- pennylane (quantum ML)
- torch (classical ML)
- numpy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
import pennylane as qml
from collections import deque
import logging

logger = logging.getLogger(__name__)


class QuantumNeuralNetwork:
    """
    Quantum Neural Network (QNN) for macro trend prediction.
    Uses PennyLane for quantum circuit simulation.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = torch.randn(n_layers, n_qubits, requires_grad=True)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> List[torch.Tensor]:
            # Encode classical data into quantum state
            for i in range(n_qubits):
                if i < len(inputs):
                    qml.RY(inputs[i] * np.pi, wires=i)
                else:
                    qml.RY(0.0, wires=i)

            # Variational quantum circuit
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through QNN."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        outputs = torch.stack([self.circuit(x[i], self.weights) for i in range(x.shape[0])])
        return torch.mean(outputs, dim=1)  # Aggregate qubit outputs


class LSTMNoiseFilter(nn.Module):
    """
    LSTM for micro noise filtering (short-term predictions).
    """
    def __init__(self, input_size: int = 10, hidden_size: int = 64, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, seq_len, input_size)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class HybridForecaster:
    """
    Hybrid Quantum-Classical Forecaster.
    - QNN for macro trends (1-24hr)
    - LSTM for micro noise (1-60min)
    - Dynamic weighting based on volatility
    """
    def __init__(
        self,
        n_qubits: int = 4,
        lstm_input_size: int = 10,
        lstm_hidden_size: int = 64,
        volatility_threshold: float = 0.2,
    ):
        self.qnn = QuantumNeuralNetwork(n_qubits=n_qubits)
        self.lstm = LSTMNoiseFilter(input_size=lstm_input_size, hidden_size=lstm_hidden_size)
        self.volatility_threshold = volatility_threshold

        # Optimizers
        self.qnn_optimizer = torch.optim.Adam([self.qnn.weights], lr=0.01)
        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.001)

        # Training history
        self.qnn_loss_history = deque(maxlen=100)
        self.lstm_loss_history = deque(maxlen=100)

        logger.info(f"HybridForecaster initialized: {n_qubits} qubits, LSTM({lstm_input_size}->{lstm_hidden_size})")

    def predict(
        self,
        macro_features: np.ndarray,
        micro_features: np.ndarray,
        volatility: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Predict market direction and confidence.
        Args:
            macro_features: Array of macro features (e.g., trend, momentum)
            micro_features: Array of micro features (e.g., recent price changes)
            volatility: Current market volatility (if None, calculated from micro_features)
        Returns:
            Tuple[direction, confidence] where direction is -1 (sell), 0 (hold), 1 (buy)
        """
        # Convert to tensors
        macro_tensor = torch.tensor(macro_features, dtype=torch.float32)
        micro_tensor = torch.tensor(micro_features, dtype=torch.float32)

        # QNN prediction (macro)
        with torch.no_grad():
            qnn_pred = self.qnn.forward(macro_tensor)
        qnn_direction = torch.sign(qnn_pred).item()
        qnn_confidence = torch.sigmoid(torch.abs(qnn_pred)).item()

        # LSTM prediction (micro)
        with torch.no_grad():
            lstm_pred = self.lstm(micro_tensor)
        lstm_direction = torch.sign(lstm_pred).item()
        lstm_confidence = torch.sigmoid(torch.abs(lstm_pred)).item()

        # Calculate volatility if not provided
        if volatility is None:
            volatility = np.std(micro_features) if len(micro_features) > 1 else 0.1

        # Dynamic weighting
        qnn_weight = 0.7 if volatility > self.volatility_threshold else 0.4
        lstm_weight = 1 - qnn_weight

        # Weighted fusion
        fused_direction = qnn_weight * qnn_direction + lstm_weight * lstm_direction
        fused_confidence = qnn_weight * qnn_confidence + lstm_weight * lstm_confidence

        # Discretize direction
        if fused_direction > 0.5:
            direction = 1
        elif fused_direction < -0.5:
            direction = -1
        else:
            direction = 0

        return direction, min(fused_confidence, 1.0)

    def train(
        self,
        macro_data: np.ndarray,
        micro_data: np.ndarray,
        targets: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Train the hybrid model.
        Args:
            macro_data: Array of macro features (n_samples, n_macro_features)
            micro_data: Array of micro features (n_samples, n_micro_features)
            targets: Array of target directions (-1, 0, 1)
            epochs: Number of training epochs
            batch_size: Batch size for LSTM
        Returns:
            Dict with training losses
        """
        macro_tensor = torch.tensor(macro_data, dtype=torch.float32)
        micro_tensor = torch.tensor(micro_data, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        for epoch in range(epochs):
            # Train QNN
            self.qnn_optimizer.zero_grad()
            qnn_pred = self.qnn.forward(macro_tensor)
            qnn_loss = nn.MSELoss()(qnn_pred, target_tensor)
            qnn_loss.backward()
            self.qnn_optimizer.step()
            self.qnn_loss_history.append(qnn_loss.item())

            # Train LSTM (batched)
            self.lstm_optimizer.zero_grad()
            for i in range(0, len(micro_data), batch_size):
                batch_micro = micro_tensor[i:i + batch_size]
                batch_target = target_tensor[i:i + batch_size]

                if len(batch_micro) == 0:
                    continue

                lstm_pred = self.lstm(batch_micro)
                lstm_loss = nn.MSELoss()(lstm_pred.squeeze(), batch_target)
                lstm_loss.backward()
            self.lstm_optimizer.step()
            self.lstm_loss_history.append(lstm_loss.item())

        return {
            "qnn_loss": qnn_loss.item(),
            "lstm_loss": lstm_loss.item(),
            "avg_qnn_loss": float(np.mean(self.qnn_loss_history)) if self.qnn_loss_history else 0.0,
            "avg_lstm_loss": float(np.mean(self.lstm_loss_history)) if self.lstm_loss_history else 0.0,
        }

    def get_weights(self) -> Dict[str, Any]:
        """Get current model weights for debugging."""
        return {
            "qnn_weight": self.qnn.weights.detach().numpy().tolist(),
            "lstm_weights": [p.detach().numpy().tolist() for p in self.lstm.parameters()],
        }
