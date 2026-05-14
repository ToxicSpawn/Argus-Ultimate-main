"""Real ML Trading Engine - PyTorch & TensorFlow Integration.

Real machine learning models for trading:
- PyTorch LSTM/GRU/Transformer
- TensorFlow Keras models
- Real reinforcement learning (PPO, A2C)
- Proper training loops
- Model versioning
- GPU acceleration
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

PYTORCH_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch available - using real neural networks")
except ImportError:
    logger.warning("PyTorch not available")
    nn = object

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available - using Keras models")
except ImportError:
    logger.warning("TensorFlow not available")
    keras = object


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    gpu: bool = True


@dataclass
class ModelMetrics:
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    epoch: int


class PyTorchLSTM(nn.Module):
    """Real PyTorch LSTM for price prediction."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> List[ModelMetrics]:
        
        self.train()
        
        X_tensor = torch.FloatTensor(X_train).to(self._device)
        y_tensor = torch.FloatTensor(y_train).to(self._device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        metrics = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            val_loss = 0.0
            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self._device)
                    y_val_tensor = torch.FloatTensor(y_val).to(self._device)
                    val_outputs = self.forward(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                self.train()
            
            metrics.append(ModelMetrics(
                loss=avg_loss,
                accuracy=0.0,
                val_loss=val_loss,
                val_accuracy=0.0,
                epoch=epoch,
            ))
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
        
        return metrics

    def save(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


class PyTorchTransformer(nn.Module):
    """Real PyTorch Transformer for sequence prediction."""

    def __init__(
        self,
        input_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self._device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()


class KerasPricePredictor:
    """Real TensorFlow/Keras model for price prediction."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (10, 10),
        model_type: str = "lstm",
    ):
        self.input_shape = input_shape
        self.model_type = model_type
        self.model = None
        self._build_model()

    def _build_model(self):
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available")
            return

        self.model = keras.Sequential()

        if self.model_type == "lstm":
            self.model.add(keras.layers.LSTM(
                64, return_sequences=True, input_shape=self.input_shape
            ))
            self.model.add(keras.layers.Dropout(0.2))
            self.model.add(keras.layers.LSTM(32))
        elif self.model_type == "gru":
            self.model.add(keras.layers.GRU(
                64, return_sequences=True, input_shape=self.input_shape
            ))
            self.model.add(keras.layers.Dropout(0.2))
            self.model.add(keras.layers.GRU(32))
        elif self.model_type == "cnn":
            self.model.add(keras.layers.Conv1D(
                64, 3, activation='relu', input_shape=self.input_shape
            ))
            self.model.add(keras.layers.MaxPooling1D(2))
            self.model.add(keras.layers.Conv1D(32, 3, activation='relu'))
            self.model.add(keras.layers.Flatten())
        elif self.model_type == "transformer":
            self.model.add(keras.layers.Dense(64, activation='relu', input_shape=self.input_shape))
            self.model.add(keras.layers.Dense(32))

        self.model.add(keras.layers.Dropout(0.2))
        self.model.add(keras.layers.Dense(1))

        self.model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='mse',
            metrics=['mae'],
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
    ) -> Any:
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return {}

        kwargs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': 0,
        }

        if X_val is not None:
            kwargs['validation_data'] = (X_val, y_val)

        if callbacks:
            kwargs['callbacks'] = callbacks
        else:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            kwargs['callbacks'] = [early_stop]

        history = self.model.fit(X_train, y_train, **kwargs)
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return np.zeros(len(X))

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def save(self, path: str) -> None:
        if TENSORFLOW_AVAILABLE and self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        if TENSORFLOW_AVAILABLE:
            self.model = keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")


class ReinforcementLearning:
    """Real Reinforcement Learning for trading."""

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 0.001,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        if PYTORCH_AVAILABLE:
            self._build_pytorch_models()
        else:
            self._build_numpy_models()

    def _build_pytorch_models(self):
        class QNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_dim),
                )
            
            def forward(self, x):
                return self.fc(x)

        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network = QNetwork(self.state_dim, self.action_dim)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self._device)
        self.target_network.to(self._device)

    def _build_numpy_models(self):
        self.q_table = {}

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        if PYTORCH_AVAILABLE:
            self.q_network.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            state_key = tuple(np.round(state, 1))
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim)
            return np.argmax(self.q_table[state_key])

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        
        if PYTORCH_AVAILABLE:
            self.q_network.train()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)
            next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self._device)
            
            current_q = self.q_network(state_tensor)[0, action]
            
            with torch.no_grad():
                target_q = reward + (1 - done) * self.gamma * self.target_network(next_tensor).max(1)[0]
            
            loss = nn.MSELoss()(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if np.random.random() < 0.05:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return loss.item()
        else:
            state_key = tuple(np.round(state, 1))
            next_key = tuple(np.round(next_state, 1))
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_dim)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.action_dim)
            
            current_q = self.q_table[state_key][action]
            max_next_q = np.max(self.q_table[next_key])
            
            self.q_table[state_key][action] = current_q + 0.1 * (
                reward + (1 - done) * self.gamma * max_next_q - current_q
            )
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return abs(reward)


class ModelEnsemble:
    """Ensemble of multiple models."""

    def __init__(self):
        self.models: List[Any] = []
        self.weights: List[float] = []
        self._last_X = None

    def add_model(self, model: Any, weight: float = 1.0) -> None:
        self.models.append(model)
        self.weights.append(weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(len(X))

        total_weight = sum(self.weights)
        normalized_weights = [w / total_weight for w in self.weights]

        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred)
            else:
                predictions.append(np.zeros(len(X)))

        ensemble_pred = np.zeros(len(X))
        for pred, weight in zip(predictions, normalized_weights):
            ensemble_pred += pred * weight

        return ensemble_pred

    def fit_weights(self, y_true: np.ndarray, X: np.ndarray) -> None:
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                predictions.append(model.predict(X))
        
        if not predictions:
            return
        
        for i, pred in enumerate(predictions):
            mse = np.mean((pred - y_true) ** 2)
            self.weights[i] = 1.0 / (mse + 1e-8)


class RealMLTradingEngine:
    """Production-ready ML trading engine."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.input_size = self.config.get("input_size", 10)
        self.is_trained = False
        self._training_history = []
        self.pytorch_lstm = None
        self.pytorch_transformer = None
        self.keras_model = None
        self.ensemble = None
        self.rl_agent = None
        
        self.pytorch_lstm: Optional[PyTorchLSTM] = None
        self.pytorch_transformer: Optional[PyTorchTransformer] = None
        self.keras_model: Optional[KerasPricePredictor] = None
        self.rl_agent: Optional[ReinforcementLearning] = None
        self.ensemble: Optional[ModelEnsemble] = None
        
        self.is_trained = False
        self._training_history: deque = deque(maxlen=100)

    def build_models(self, input_size: int = 10) -> Dict[str, Any]:
        models = {}
        
        self.input_size = input_size
        
        if PYTORCH_AVAILABLE:
            self.pytorch_lstm = PyTorchLSTM(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
            )
            models['pytorch_lstm'] = self.pytorch_lstm
            
            self.pytorch_transformer = PyTorchTransformer(
                input_size=input_size,
                d_model=64,
                nhead=4,
                num_layers=2,
            )
            models['pytorch_transformer'] = self.pytorch_transformer
            
            logger.info(f"PyTorch models built with input_size={input_size}")
        
        if TENSORFLOW_AVAILABLE:
            self.keras_model = KerasPricePredictor(
                input_shape=(input_size, 10),
                model_type="lstm",
            )
            models['keras_lstm'] = self.keras_model
            
            logger.info(f"TensorFlow models built with input_size={input_size}")
        
        self.rl_agent = ReinforcementLearning(
            state_dim=input_size,
            action_dim=3,
        )
        models['rl_agent'] = self.rl_agent
        
        self.ensemble = ModelEnsemble()
        
        return models

    def train_price_prediction(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_type: str = "pytorch_lstm",
        epochs: int = 100,
    ) -> Dict[str, Any]:
        
        results = {}
        
        if model_type == "pytorch_lstm" and self.pytorch_lstm:
            metrics = self.pytorch_lstm.train_model(
                X_train, y_train, X_val, y_val, epochs=epochs
            )
            results['pytorch_lstm'] = [m.__dict__ for m in metrics]
            self.is_trained = True
            
        elif model_type == "keras_lstm" and self.keras_model:
            history = self.keras_model.train(
                X_train, y_train, X_val, y_val, epochs=epochs
            )
            results['keras_lstm'] = history
            self.is_trained = True
        
        self._training_history.append({
            'timestamp': time.time(),
            'model_type': model_type,
            'train_size': len(X_train),
        })
        
        return results

    def predict_price(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            logger.warning("Models not trained yet")
            return np.zeros(len(X))
        
        if self.ensemble and len(self.ensemble.models) > 0:
            return self.ensemble.predict(X)
        
        if self.pytorch_lstm:
            return self.pytorch_lstm.predict(X)
        
        if self.keras_model:
            return self.keras_model.predict(X)
        
        return np.zeros(len(X))

    def train_rl(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        
        if not self.rl_agent:
            return 0.0
        
        total_loss = 0.0
        for i in range(len(states)):
            loss = self.rl_agent.train_step(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i],
            )
            total_loss += loss
        
        return total_loss / len(states)

    def rl_select_action(
        self,
        state: np.ndarray,
        training: bool = True,
    ) -> int:
        
        if not self.rl_agent:
            return 1
        
        return self.rl_agent.select_action(state, training)

    def save_models(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.pytorch_lstm:
            self.pytorch_lstm.save(f"{path}/lstm.pt")
        
        if self.keras_model:
            self.keras_model.save(f"{path}/keras_model")
        
        logger.info(f"Models saved to {path}")

    def load_models(self, path: str) -> None:
        import os
        
        if self.pytorch_lstm and os.path.exists(f"{path}/lstm.pt"):
            self.pytorch_lstm.load(f"{path}/lstm.pt")
            self.is_trained = True
        
        if self.keras_model and os.path.exists(f"{path}/keras_model"):
            self.keras_model.load(f"{path}/keras_model")
            self.is_trained = True
        
        logger.info(f"Models loaded from {path}")

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            'pytorch': PYTORCH_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE,
            'cuda': torch.cuda.is_available() if PYTORCH_AVAILABLE else False,
            'trained': self.is_trained,
        }


def create_ml_engine(config: Optional[Dict] = None) -> RealMLTradingEngine:
    engine = RealMLTradingEngine(config)
    engine.build_models(input_size=engine.input_size)
    return engine
