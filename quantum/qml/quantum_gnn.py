"""
Quantum Graph Neural Network (QGNN).

Encodes a graph G = (V, E) into a quantum state where each node is a
qubit, and edges correspond to entangling gates between those qubits.
Messages propagate via layers of parameterized unitaries.

Reference
---------
Verdon, McCourt, Luzhnica, Singh, Leichenauer, Hidary,
"Quantum graph neural networks," arXiv:1909.12264 (2019)

Trading use
-----------
Asset correlation graphs can be encoded as quantum graphs; the QGNN
learns an embedding used as a feature vector for downstream classifiers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quantum_simulator import QuantumCircuit, _simulate_statevector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# QGNN
# ═════════════════════════════════════════════════════════════════════════════


class QuantumGNN:
    """
    Parameterized quantum graph neural network.

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (= qubits).
    n_layers : int
        Number of QGNN message-passing rounds.
    """

    def __init__(self, n_nodes: int, n_layers: int = 2) -> None:
        if n_nodes < 2 or n_nodes > 10:
            raise ValueError(f"n_nodes must be 2..10, got {n_nodes}")
        self.n_nodes = int(n_nodes)
        self.n_layers = int(n_layers)
        # Parameters: RY for each node per layer + RZZ coupling per edge per layer
        # (we assume full connectivity; unused edges are just masked at forward time)
        self.n_edge_params = (n_nodes * (n_nodes - 1) // 2) * n_layers
        self.n_node_params = n_nodes * n_layers
        self.n_params = self.n_edge_params + self.n_node_params
        rng = np.random.default_rng(42)
        self.params = rng.uniform(-np.pi, np.pi, self.n_params)

    def forward(
        self,
        adjacency: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run one forward pass on a graph.

        Parameters
        ----------
        adjacency : np.ndarray
            (n_nodes, n_nodes) adjacency matrix (binary or weighted).
        node_features : np.ndarray, optional
            (n_nodes,) per-node features to encode into initial rotations.
        params : np.ndarray, optional
            Parameter vector; defaults to ``self.params``.

        Returns
        -------
        np.ndarray
            (n_nodes,) per-node ⟨Z⟩ expectation = graph embedding.
        """
        if params is None:
            params = self.params
        adjacency = np.asarray(adjacency, dtype=float)

        qc = QuantumCircuit(self.n_nodes)

        # Feature encoding
        if node_features is not None:
            for q in range(self.n_nodes):
                angle = float(np.clip(node_features[q], -np.pi, np.pi))
                qc.ry(angle, q)

        # Message-passing layers
        idx = 0
        for layer in range(self.n_layers):
            # Node-local rotations
            for q in range(self.n_nodes):
                qc.ry(float(params[idx]), q)
                idx += 1
            # Edge-based entangling rotations
            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    weight = float(adjacency[i, j])
                    if weight > 1e-9:
                        qc.rzz(float(params[idx]) * weight, i, j)
                    idx += 1

        # Read out per-node ⟨Z⟩ embeddings
        state = _simulate_statevector(qc)
        probs = np.abs(state) ** 2
        embeddings = np.zeros(self.n_nodes)
        for q in range(self.n_nodes):
            z = 0.0
            for k in range(len(probs)):
                bit = (k >> q) & 1
                z += probs[k] * (1 if bit == 0 else -1)
            embeddings[q] = float(z)
        return embeddings

    def fit_node_classification(
        self,
        adjacency: np.ndarray,
        node_features: np.ndarray,
        labels: np.ndarray,
        *,
        n_iter: int = 50,
    ) -> Dict[str, Any]:
        """
        Train a QGNN for node classification via a linear readout.
        """
        from scipy.optimize import minimize as sp_minimize

        t0 = time.perf_counter()
        readout_w = np.zeros(self.n_nodes)

        def cost_fn(params):
            emb = self.forward(adjacency, node_features, params)
            preds = np.tanh(emb)  # bounded output
            loss = float(np.mean((preds - labels) ** 2))
            return loss

        opt = sp_minimize(
            cost_fn,
            self.params.copy(),
            method="COBYLA",
            options={"maxiter": n_iter, "rhobeg": 0.3},
        )
        self.params = np.asarray(opt.x, dtype=float)

        return {
            "final_loss": float(opt.fun),
            "method": "qgnn_node_classification",
            "n_params": self.n_params,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }
