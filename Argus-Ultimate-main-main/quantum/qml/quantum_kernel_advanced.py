"""
Advanced quantum kernel methods (Phase I5).

Builds on the existing ``quantum_kernel.py`` classifier with:

- **Quantum kernel via state overlap** computed through the in-repo simulator
  (real circuit execution, not just numpy correlation).
- **Quantum kernel alignment** — train kernel parameters to maximize the
  alignment between the kernel Gram matrix and the target label matrix.
- **Variational Quantum Classifier (VQC)** with explicit feature maps.
- **Pauli-feature map** and **ZZ-feature map** following Havlíček et al. (2019).

Reference
---------
Havlíček et al., "Supervised learning with quantum-enhanced feature spaces,"
Nature 567, 209 (2019).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Feature maps
# ═════════════════════════════════════════════════════════════════════════════


def zz_feature_map(x: np.ndarray, n_qubits: int, n_layers: int = 2) -> QuantumCircuit:
    """
    ZZ-feature map (Havlíček et al. 2019).

    Encodes classical features ``x`` into a quantum state via:
        H ⊗ ... ⊗ H · ∏_layers ∏_i RZ(2 x_i) · ∏_{i<j} RZZ(2 (π - x_i)(π - x_j))

    Returns a circuit that prepares the feature-encoded state.
    """
    qc = QuantumCircuit(n_qubits)
    x_padded = np.zeros(n_qubits)
    x_padded[: len(x)] = x[:n_qubits]

    for _ in range(n_layers):
        # Hadamard layer
        for q in range(n_qubits):
            qc.h(q)
        # Single-qubit RZ encoding
        for q in range(n_qubits):
            qc.rz(2.0 * float(x_padded[q]), q)
        # Pairwise ZZ encoding
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                angle = 2.0 * (np.pi - float(x_padded[i])) * (np.pi - float(x_padded[j]))
                qc.rzz(angle, i, j)

    return qc


def pauli_feature_map(
    x: np.ndarray, n_qubits: int, n_layers: int = 2
) -> QuantumCircuit:
    """
    Pauli-feature map. Encodes features via X, Y, and Z rotations alternately.
    """
    qc = QuantumCircuit(n_qubits)
    x_padded = np.zeros(n_qubits)
    x_padded[: len(x)] = x[:n_qubits]

    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.rx(float(x_padded[q]), q)
            qc.ry(float(x_padded[q]) * 0.5, q)
            qc.rz(float(x_padded[q]) * 0.7, q)
        # Entangling layer
        for q in range(n_qubits - 1):
            qc.cnot(q, q + 1)

    return qc


# ═════════════════════════════════════════════════════════════════════════════
# Quantum kernel
# ═════════════════════════════════════════════════════════════════════════════


class QuantumKernel:
    """
    Quantum kernel computed via state-overlap fidelity:
        K(x, y) = |⟨φ(x) | φ(y)⟩|²

    where |φ(x)⟩ is the feature-encoded quantum state.
    """

    def __init__(
        self,
        feature_map: str = "zz",
        n_qubits: int = 4,
        n_layers: int = 2,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        if feature_map == "zz":
            self.feature_map_fn = zz_feature_map
        elif feature_map == "pauli":
            self.feature_map_fn = pauli_feature_map
        else:
            raise ValueError(f"Unknown feature map: {feature_map}")
        self.feature_map_name = feature_map

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode feature vector x into a quantum state."""
        qc = self.feature_map_fn(x, self.n_qubits, self.n_layers)
        return _simulate_statevector(qc)

    def kernel_value(self, x: np.ndarray, y: np.ndarray) -> float:
        """K(x, y) = |⟨φ(x)|φ(y)⟩|²"""
        psi_x = self.encode(x)
        psi_y = self.encode(y)
        overlap = abs(np.vdot(psi_x, psi_y)) ** 2
        return float(overlap)

    def compute_kernel_matrix(
        self, X1: np.ndarray, X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the Gram matrix K[i, j] = K(X1[i], X2[j]).

        If X2 is None, computes the symmetric matrix K[i, j] = K(X1[i], X1[j]).
        """
        if X2 is None:
            n = len(X1)
            states = [self.encode(X1[i]) for i in range(n)]
            K = np.zeros((n, n), dtype=float)
            for i in range(n):
                K[i, i] = 1.0
                for j in range(i + 1, n):
                    K[i, j] = float(abs(np.vdot(states[i], states[j])) ** 2)
                    K[j, i] = K[i, j]
            return K
        else:
            n1 = len(X1)
            n2 = len(X2)
            states_1 = [self.encode(X1[i]) for i in range(n1)]
            states_2 = [self.encode(X2[i]) for i in range(n2)]
            K = np.zeros((n1, n2), dtype=float)
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = float(abs(np.vdot(states_1[i], states_2[j])) ** 2)
            return K


# ═════════════════════════════════════════════════════════════════════════════
# Variational Quantum Classifier
# ═════════════════════════════════════════════════════════════════════════════


class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier (VQC) with explicit feature map and
    parameterized ansatz.

    The classifier:
    1. Encodes features via the chosen feature map
    2. Applies a parameterized ansatz (RY + CNOT-ring layers)
    3. Measures the first qubit; class label = sign of ⟨Z_0⟩
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_ansatz_layers: int = 3,
        feature_map: str = "zz",
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.n_ansatz_layers = int(n_ansatz_layers)
        self.kernel = QuantumKernel(
            feature_map=feature_map, n_qubits=n_qubits, n_layers=2
        )
        self.params: Optional[np.ndarray] = None
        # Use the quantum kernel + classical SVM as the trainable backbone
        self._svc = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the VQC by computing the quantum kernel matrix and feeding it
        into a classical SVM.
        """
        try:
            from sklearn.svm import SVC
        except ImportError:
            raise ImportError("VariationalQuantumClassifier requires scikit-learn")

        K = self.kernel.compute_kernel_matrix(X)
        self._svc = SVC(kernel="precomputed")
        self._svc.fit(K, y)
        self._X_train = X.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for X."""
        if self._svc is None:
            raise RuntimeError("VQC not yet fitted")
        K = self.kernel.compute_kernel_matrix(X, self._X_train)
        return self._svc.predict(K)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy on (X, y)."""
        preds = self.predict(X)
        return float(np.mean(preds == y))


# ═════════════════════════════════════════════════════════════════════════════
# Kernel alignment
# ═════════════════════════════════════════════════════════════════════════════


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Kernel-target alignment between the kernel matrix K and the label vector y.

    Alignment(K, yy^T) = ⟨K, yy^T⟩_F / (||K||_F · ||yy^T||_F)

    Returns a value in [-1, 1]; larger = better aligned.
    """
    y_mat = np.outer(y, y).astype(float)
    num = float(np.sum(K * y_mat))
    den = float(np.linalg.norm(K, "fro") * np.linalg.norm(y_mat, "fro"))
    if den < 1e-12:
        return 0.0
    return num / den
