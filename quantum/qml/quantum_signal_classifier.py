"""
Quantum Kernel Machine for Trading Signal Classification.

Implements a quantum kernel SVM that uses angle-encoded feature maps to
compute kernel values that are provably hard to simulate classically for
large qubit counts.  On classical hardware this is a simulation -- the
quantum kernel matrix is computed exactly via statevector inner products.

The quantum feature map encodes classical feature vectors into quantum
states using parameterized rotation gates.  The kernel is:
    K(x1, x2) = |<phi(x1)|phi(x2)>|^2

This kernel captures nonlinear feature interactions through quantum
entanglement structure, providing an expressive kernel for classification.

For trading: classifies signal quality (good/bad/neutral) based on
technical features like spread, volume ratio, volatility, momentum, etc.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SVM availability
_HAS_SVM = False
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    _HAS_SVM = True
except ImportError:
    pass


class QuantumSignalClassifier:
    """
    Quantum kernel machine for classifying trading signals.

    Uses angle encoding to map classical features into quantum states,
    then computes the quantum kernel matrix for SVM classification.

    Attributes:
        n_features: Number of input features.
        n_qubits: Number of qubits in the feature map circuit.
        kernel_type: "angle" or "amplitude" encoding.
    """

    def __init__(
        self,
        n_features: int = 8,
        n_qubits: Optional[int] = None,
        kernel_type: str = "angle",
        n_layers: int = 2,
    ) -> None:
        """
        Args:
            n_features: Number of classical input features.
            n_qubits: Number of qubits. Defaults to n_features for angle encoding.
            kernel_type: "angle" (RY encoding) or "amplitude" (normalized state).
            n_layers: Number of entangling layers in the feature map.
        """
        self.n_features = n_features
        self.n_qubits = n_qubits if n_qubits is not None else n_features
        self.kernel_type = kernel_type
        self.n_layers = max(1, n_layers)

        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._fitted = False
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._classes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Feature map: classical -> quantum state
    # ------------------------------------------------------------------

    def _build_feature_map(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a classical feature vector into a quantum statevector.

        For angle encoding:
            |phi(x)> = prod_layers [ CNOT_layer . prod_i RY(x_i * pi) ] |0...0>

        For amplitude encoding:
            |phi(x)> = normalize(x) as state amplitudes (padded to 2^n)

        Args:
            x: 1D feature vector of length n_features.

        Returns:
            Complex statevector of length 2^n_qubits.
        """
        n = self.n_qubits
        N = 2 ** n

        if self.kernel_type == "amplitude":
            return self._amplitude_encoding(x, N)

        # Angle encoding with entangling layers
        # Start with |0...0>
        state = np.zeros(N, dtype=np.complex128)
        state[0] = 1.0

        for layer in range(self.n_layers):
            # Single-qubit RY rotations: RY(x_i * pi * (layer+1))
            for q in range(min(n, len(x))):
                angle = x[q] * np.pi * (layer + 1)
                state = self._apply_ry(state, n, q, angle)

            # Entangling layer: CNOT cascade
            for q in range(n - 1):
                state = self._apply_cnot(state, n, q, q + 1)

            # Second rotation layer with cross-feature terms
            for q in range(min(n, len(x))):
                q2 = (q + 1) % len(x)
                angle = x[q] * x[q2] * np.pi
                state = self._apply_ry(state, n, q, angle)

        return state

    def _amplitude_encoding(self, x: np.ndarray, N: int) -> np.ndarray:
        """Encode features as state amplitudes."""
        padded = np.zeros(N)
        padded[: min(len(x), N)] = x[: min(len(x), N)]
        norm = np.linalg.norm(padded)
        if norm > 1e-12:
            padded = padded / norm
        else:
            padded[0] = 1.0
        return padded.astype(np.complex128)

    @staticmethod
    def _apply_ry(
        state: np.ndarray, n_qubits: int, qubit: int, angle: float
    ) -> np.ndarray:
        """Apply RY(angle) to a single qubit in the statevector."""
        N = len(state)
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        new_state = state.copy()
        mask = 1 << (n_qubits - 1 - qubit)

        for i in range(N):
            if i & mask:
                continue  # will be handled by partner
            j = i | mask
            a0 = state[i]
            a1 = state[j]
            new_state[i] = c * a0 - s * a1
            new_state[j] = s * a0 + c * a1

        return new_state

    @staticmethod
    def _apply_cnot(
        state: np.ndarray, n_qubits: int, control: int, target: int
    ) -> np.ndarray:
        """Apply CNOT gate."""
        N = len(state)
        c_mask = 1 << (n_qubits - 1 - control)
        t_mask = 1 << (n_qubits - 1 - target)
        new_state = state.copy()

        for i in range(N):
            if i & c_mask:  # control is |1>
                j = i ^ t_mask  # flip target
                new_state[i] = state[j]
                new_state[j] = state[i]

        return new_state

    # ------------------------------------------------------------------
    # Quantum kernel
    # ------------------------------------------------------------------

    def _quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum kernel: K(x1, x2) = |<phi(x1)|phi(x2)>|^2.

        This fidelity-based kernel measures overlap between quantum states
        created by encoding x1 and x2 through the same feature map.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Kernel value in [0, 1].
        """
        state1 = self._build_feature_map(x1)
        state2 = self._build_feature_map(x2)
        overlap = np.abs(np.dot(np.conj(state1), state2)) ** 2
        return float(overlap)

    def compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the full quantum kernel matrix for a dataset.

        Args:
            X: 2D array of shape (n_samples, n_features).

        Returns:
            Symmetric positive-semidefinite kernel matrix (n_samples, n_samples).
        """
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 1.0  # K(x, x) = 1
            for j in range(i + 1, n):
                k_val = self._quantum_kernel(X[i], X[j])
                K[i, j] = k_val
                K[j, i] = k_val
        return K

    def compute_kernel_matrix_train_test(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Compute kernel matrix between train and test sets."""
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        K = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K[i, j] = self._quantum_kernel(X_test[i], X_train[j])
        return K

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any, C: float = 1.0) -> "QuantumSignalClassifier":
        """
        Train quantum kernel SVM.

        Computes the quantum kernel matrix for training data, then fits
        an SVM with precomputed kernel.

        Args:
            X: Training features (n_samples, n_features).
            y: Training labels.
            C: SVM regularization parameter.

        Returns:
            self
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Scale features
        if _HAS_SVM:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_arr)
        else:
            X_scaled = X_arr

        self._X_train = X_scaled
        self._y_train = y_arr
        self._classes = np.unique(y_arr)

        if not _HAS_SVM:
            logger.warning("sklearn not available; using nearest-centroid fallback")
            self._fitted = True
            return self

        # Compute training kernel matrix
        K_train = self.compute_kernel_matrix(X_scaled)

        # Fit SVM with precomputed kernel
        self._model = SVC(kernel="precomputed", C=C, probability=True)
        self._model.fit(K_train, y_arr)
        self._fitted = True

        logger.info(
            "QuantumSignalClassifier fitted: %d samples, %d features, "
            "%d qubits, %d classes",
            len(y_arr), X_scaled.shape[1], self.n_qubits, len(self._classes),
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict using trained quantum kernel SVM.

        Args:
            X: Test features (n_samples, n_features).

        Returns:
            Array of predicted labels.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X_arr)
        else:
            X_scaled = X_arr

        if self._model is not None and _HAS_SVM:
            K_test = self.compute_kernel_matrix_train_test(self._X_train, X_scaled)
            return self._model.predict(K_test)
        else:
            return self._nearest_centroid_predict(X_scaled)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict_proba()")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X_arr)
        else:
            X_scaled = X_arr

        if self._model is not None and _HAS_SVM:
            K_test = self.compute_kernel_matrix_train_test(self._X_train, X_scaled)
            return self._model.predict_proba(K_test)
        else:
            # Fallback: return uniform probabilities
            n_classes = len(self._classes) if self._classes is not None else 2
            return np.ones((len(X_scaled), n_classes)) / n_classes

    def predict_signal_quality(self, features: Any) -> Dict[str, Any]:
        """
        Given signal features, return quality assessment.

        Args:
            features: 1D or 2D array of signal features.

        Returns:
            dict with quality (float 0-1), confidence (float 0-1),
            regime (str), prediction (int/str).
        """
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._fitted:
            return {
                "quality": 0.5,
                "confidence": 0.0,
                "regime": "unknown",
                "prediction": 0,
                "method": "not_fitted",
            }

        preds = self.predict(X)
        probas = self.predict_proba(X)

        quality = float(np.max(probas[0]))
        prediction = int(preds[0]) if np.issubdtype(preds.dtype, np.integer) else preds[0]
        confidence = quality

        # Regime estimation from prediction
        if isinstance(prediction, (int, np.integer)):
            regime_map = {0: "bearish", 1: "bullish", 2: "neutral"}
            regime = regime_map.get(int(prediction), "unknown")
        else:
            regime = str(prediction)

        return {
            "quality": round(quality, 4),
            "confidence": round(confidence, 4),
            "regime": regime,
            "prediction": prediction,
            "method": "quantum_kernel_svm",
        }

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _nearest_centroid_predict(self, X: np.ndarray) -> np.ndarray:
        """Simple nearest-centroid classifier as fallback."""
        if self._X_train is None or self._y_train is None:
            return np.zeros(len(X), dtype=int)

        centroids = {}
        for c in self._classes:
            mask = self._y_train == c
            centroids[c] = np.mean(self._X_train[mask], axis=0)

        predictions = []
        for x in X:
            best_c = self._classes[0]
            best_dist = float("inf")
            for c, centroid in centroids.items():
                dist = np.linalg.norm(x - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_c = c
            predictions.append(best_c)

        return np.array(predictions)
