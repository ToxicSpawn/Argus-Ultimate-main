"""LSTM-style market learner with a safe NumPy fallback.

The module exposes one API regardless of whether PyTorch is installed:
`fit(features, labels)`, `predict(features)`, and `predict_proba(features)`.
When torch is available it uses a compact LSTM classifier; otherwise it uses
an online softmax model so paper tests and offline Sydney workflows still run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ImportError:  # optional dependency
    torch = None
    nn = None


LABELS = ("sell", "hold", "buy")


@dataclass
class Prediction:
    action: str
    confidence: float
    probabilities: dict[str, float]


class _TorchLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 3))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class LSTMLearner:
    """Small time-series classifier for buy/hold/sell decisions."""

    def __init__(self, input_size: int = 9, lookback: int = 32, learning_rate: float = 0.01):
        self.input_size = input_size
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.uses_torch = torch is not None and nn is not None
        self.mean = np.zeros(input_size)
        self.std = np.ones(input_size)
        self.weights = np.zeros((input_size, 3))
        self.bias = np.zeros(3)
        self.model = _TorchLSTM(input_size) if self.uses_torch else None
        self.loss_history: list[float] = []

    def _normalise(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / (self.std + 1e-9)

    def _prepare(self, features: Iterable[Iterable[float]]) -> np.ndarray:
        data = np.asarray(list(features), dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[-1] != self.input_size:
            raise ValueError(f"expected {self.input_size} features, got {data.shape[-1]}")
        return data

    def fit(self, features: Iterable[Iterable[float]], labels: Iterable[int], epochs: int = 20) -> dict[str, float]:
        x = self._prepare(features)
        y = np.asarray(list(labels), dtype=int)
        if len(x) != len(y):
            raise ValueError("features and labels must have the same length")
        if len(x) == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-6)
        x_norm = self._normalise(x)

        if self.uses_torch:
            return self._fit_torch(x_norm, y, epochs)
        return self._fit_numpy(x_norm, y, epochs)

    def _fit_numpy(self, x: np.ndarray, y: np.ndarray, epochs: int) -> dict[str, float]:
        target = np.eye(3)[np.clip(y, 0, 2)]
        for _ in range(epochs):
            logits = x @ self.weights + self.bias
            probs = self._softmax(logits)
            grad = (probs - target) / len(x)
            self.weights -= self.learning_rate * x.T @ grad
            self.bias -= self.learning_rate * grad.sum(axis=0)
            loss = -np.mean(np.log(probs[np.arange(len(y)), np.clip(y, 0, 2)] + 1e-9))
            self.loss_history.append(float(loss))
        preds = np.argmax(self._softmax(x @ self.weights + self.bias), axis=1)
        return {"loss": self.loss_history[-1], "accuracy": float(np.mean(preds == y))}

    def _fit_torch(self, x: np.ndarray, y: np.ndarray, epochs: int) -> dict[str, float]:
        assert self.model is not None
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        seq = self._sequence(x)
        tx = torch.tensor(seq, dtype=torch.float32)
        ty = torch.tensor(y[-len(seq):], dtype=torch.long)
        for _ in range(epochs):
            optimiser.zero_grad()
            logits = self.model(tx)
            loss = criterion(logits, ty)
            loss.backward()
            optimiser.step()
            self.loss_history.append(float(loss.detach().cpu()))
        with torch.no_grad():
            preds = torch.argmax(self.model(tx), dim=1)
        return {"loss": self.loss_history[-1], "accuracy": float((preds == ty).float().mean())}

    def _sequence(self, x: np.ndarray) -> np.ndarray:
        if len(x) < self.lookback:
            padding = np.repeat(x[:1], self.lookback - len(x), axis=0)
            x = np.vstack([padding, x])
        return np.array([x[i - self.lookback:i] for i in range(self.lookback, len(x) + 1)])

    def predict_proba(self, features: Iterable[Iterable[float]]) -> np.ndarray:
        x = self._normalise(self._prepare(features))
        if self.uses_torch and self.model is not None:
            with torch.no_grad():
                probs = torch.softmax(self.model(torch.tensor(self._sequence(x), dtype=torch.float32)), dim=1)
            values = probs.cpu().numpy()
            if len(values) < len(x):
                pad = np.repeat(values[:1], len(x) - len(values), axis=0)
                values = np.vstack([pad, values])
            return values
        return self._softmax(x @ self.weights + self.bias)

    def predict(self, features: Iterable[Iterable[float]]) -> Prediction:
        probs = self.predict_proba(features)[-1]
        idx = int(np.argmax(probs))
        return Prediction(LABELS[idx], float(probs[idx]), dict(zip(LABELS, map(float, probs))))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


def _demo() -> None:
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(7)
    features = rng.normal(0, 0.02, size=(160, 9))
    labels = np.where(features[:, 0] + features[:, 2] > 0.015, 2, np.where(features[:, 0] < -0.015, 0, 1))
    learner = LSTMLearner(input_size=9, lookback=16)
    metrics = learner.fit(features, labels, epochs=12)
    pred = learner.predict(features[-16:])
    print("LSTM learner ready")
    print(f"backend={'torch' if learner.uses_torch else 'numpy'} loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.1%}")
    print(f"prediction={pred.action} confidence={pred.confidence:.1%}")


if __name__ == "__main__":
    _demo()
