"""
Quantum ML models (restored, dependency-light).

The historical `quantum/qml/*` modules were stubbed to satisfy compileall.
This module provides **usable** implementations that:
- do NOT require Qiskit/PennyLane
- can optionally use scikit-learn if present

These are **quantum-inspired** fallbacks, not true quantum computation.
They exist to make the QML package practical for paper/backtest experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import numpy as np


def _as_2d(x: Any) -> np.ndarray:
    a = cast(np.ndarray, np.asarray(x, dtype=float))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return cast(np.ndarray, a)


def _as_1d(y: Any) -> np.ndarray:
    a = cast(np.ndarray, np.asarray(y, dtype=float)).reshape(-1)
    return cast(np.ndarray, a)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -50.0, 50.0)
    out = 1.0 / (1.0 + np.exp(-zc))
    return cast(np.ndarray, out)


@dataclass
class QuantumKernel:
    """
    Quantum-inspired kernel: RBF by default.
    """

    gamma: float = 1.0

    def compute(self, X: Any, Y: Any) -> np.ndarray:
        X = _as_2d(X)
        Y = _as_2d(Y)
        # ||x-y||^2 = x^2 + y^2 - 2xy
        x2 = np.sum(X * X, axis=1, keepdims=True)
        y2 = np.sum(Y * Y, axis=1, keepdims=True).T
        d2 = x2 + y2 - 2.0 * (X @ Y.T)
        out = np.exp(-float(self.gamma) * np.clip(d2, 0.0, None))
        return cast(np.ndarray, out)


class VariationalQuantumClassifier:
    """
    Quantum-inspired binary classifier.

    Implementation:
    - If sklearn is available: LogisticRegression
    - Else: simple logistic regression via gradient descent
    """

    def __init__(self, *, lr: float = 0.1, steps: int = 500, l2: float = 0.0, seed: Optional[int] = 1337) -> None:
        self.lr = float(lr)
        self.steps = int(steps)
        self.l2 = float(l2)
        self.seed = seed
        self._sk_model: Any = None
        self._w: Optional[np.ndarray] = None
        self._b: float = 0.0

    def fit(self, X: Any, y: Any) -> "VariationalQuantumClassifier":
        X = _as_2d(X)
        yv = _as_1d(y)
        yb = (yv > 0).astype(float)

        # Try sklearn if present
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore

            self._sk_model = LogisticRegression(max_iter=2000, n_jobs=None)
            self._sk_model.fit(X, yb)
            return self
        except Exception:
            self._sk_model = None

        rng = np.random.default_rng(self.seed)
        self._w = rng.normal(scale=0.01, size=(X.shape[1],)).astype(float)
        self._b = 0.0

        w = cast(np.ndarray, self._w)
        for _ in range(max(1, self.steps)):
            z = X @ w + self._b
            p = _sigmoid(z)
            grad_w = (X.T @ (p - yb)) / float(len(X))
            grad_b = float(np.mean(p - yb))
            if self.l2 > 0:
                grad_w = grad_w + self.l2 * w
            w = w - self.lr * grad_w
            self._b = self._b - self.lr * grad_b
        self._w = w
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = _as_2d(X)
        if self._sk_model is not None:
            p1 = cast(np.ndarray, self._sk_model.predict_proba(X))[:, 1]
            return cast(np.ndarray, np.stack([1.0 - p1, p1], axis=1))
        if self._w is None:
            raise RuntimeError("Model not fit")
        p1 = _sigmoid(X @ self._w + self._b)
        return cast(np.ndarray, np.stack([1.0 - p1, p1], axis=1))

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(threshold)).astype(int)


class QuantumSVM:
    """
    Quantum-inspired SVM wrapper.
    """

    def __init__(self, *, C: float = 1.0, gamma: float = 1.0) -> None:
        self.C = float(C)
        self.gamma = float(gamma)
        self._sk_model: Any = None
        self._fallback: Optional[VariationalQuantumClassifier] = None

    def fit(self, X: Any, y: Any) -> "QuantumSVM":
        X = _as_2d(X)
        yv = _as_1d(y)
        yb = (yv > 0).astype(int)
        try:
            from sklearn.svm import SVC  # type: ignore

            self._sk_model = SVC(C=self.C, gamma=self.gamma, probability=True)
            self._sk_model.fit(X, yb)
            return self
        except Exception:
            self._sk_model = None
            # fallback: use VQC
            self._fallback = VariationalQuantumClassifier(lr=0.1, steps=400).fit(X, yb)
            return self

    def predict(self, X: Any) -> np.ndarray:
        X = _as_2d(X)
        if self._sk_model is not None:
            return cast(np.ndarray, self._sk_model.predict(X))
        if self._fallback is None:
            raise RuntimeError("Model not fit")
        return cast(np.ndarray, self._fallback.predict(X))


class QuantumNeuralNetwork:
    """
    Tiny neural net (quantum-inspired) for regression/classification.

    This is a minimal 1-hidden-layer MLP trained with SGD (no torch/tf required).
    """

    def __init__(self, *, hidden: int = 16, lr: float = 0.01, steps: int = 800, seed: Optional[int] = 1337) -> None:
        self.hidden = int(hidden)
        self.lr = float(lr)
        self.steps = int(steps)
        self.seed = seed
        self._params: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = None

    def fit(self, X: Any, y: Any) -> "QuantumNeuralNetwork":
        X = _as_2d(X)
        yv = _as_1d(y)
        rng = np.random.default_rng(self.seed)
        W1 = rng.normal(scale=0.1, size=(X.shape[1], self.hidden)).astype(float)
        b1 = np.zeros(self.hidden, dtype=float)
        W2 = rng.normal(scale=0.1, size=(self.hidden,)).astype(float)
        b2 = 0.0

        lr = self.lr
        for _ in range(max(1, self.steps)):
            # forward
            h = np.tanh(X @ W1 + b1)
            yhat = h @ W2 + b2
            # mse
            err = (yhat - yv)
            # backward
            dW2 = (h.T @ err) / float(len(X))
            db2 = float(np.mean(err))
            dh = np.outer(err, W2) * (1.0 - h * h)
            dW1 = (X.T @ dh) / float(len(X))
            db1 = np.mean(dh, axis=0)
            # update
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1

        self._params = (W1, b1, W2, b2)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X = _as_2d(X)
        if self._params is None:
            raise RuntimeError("Model not fit")
        W1, b1, W2, b2 = self._params
        h = np.tanh(X @ W1 + b1)
        out = (h @ W2 + b2).reshape(-1)
        return cast(np.ndarray, out)


class QuantumBoltzmannMachine:
    """
    Minimal (toy) Boltzmann-style model: learns mean feature vector and uses it as a score.
    """

    def __init__(self) -> None:
        self._mu: Optional[np.ndarray] = None

    def fit(self, X: Any) -> "QuantumBoltzmannMachine":
        X = _as_2d(X)
        self._mu = np.mean(X, axis=0)
        return self

    def score(self, X: Any) -> np.ndarray:
        X = _as_2d(X)
        if self._mu is None:
            raise RuntimeError("Model not fit")
        out = (X @ self._mu).reshape(-1)
        return cast(np.ndarray, out)

