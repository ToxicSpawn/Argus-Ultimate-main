"""
Graph Convolutional Network (GCN) — classical Kipf & Welling formulation.

Reference
---------
Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional
Networks," ICLR 2017 (arXiv:1609.02907).

Core operation
--------------
    H^(l+1) = σ( D̂^(-1/2) Â D̂^(-1/2) H^(l) W^(l) )

where
    Â = A + I (add self-loops)
    D̂ = diag(Â row sums)
    W^(l) = trainable layer weight
    σ = nonlinearity (default ReLU)

Trading use
-----------
Asset correlation graph as adjacency; features are per-asset return
statistics. Output embeddings aggregated into ensemble signals in
``ml/ensemble_signal_hub.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization of an adjacency matrix with self-loops.

        Â = A + I
        D̂ = diag(Â row sums)
        return D̂^(-1/2) Â D̂^(-1/2)
    """
    n = adjacency.shape[0]
    a_hat = adjacency + np.eye(n, dtype=adjacency.dtype)
    d_hat = a_hat.sum(axis=1)
    d_inv_sqrt = np.power(np.clip(d_hat, 1e-9, None), -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ a_hat @ d_mat_inv_sqrt


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# GCN
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class GCN:
    """
    2-layer Graph Convolutional Network (pure numpy).

    Parameters
    ----------
    n_features : int
        Input feature dimension per node.
    n_hidden : int
        Hidden layer dimension.
    n_out : int
        Output embedding dimension.
    seed : int
    """

    n_features: int
    n_hidden: int = 32
    n_out: int = 16
    seed: int = 42

    W0: np.ndarray = field(init=False)
    b0: np.ndarray = field(init=False)
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Xavier init
        bound0 = np.sqrt(6.0 / (self.n_features + self.n_hidden))
        bound1 = np.sqrt(6.0 / (self.n_hidden + self.n_out))
        self.W0 = rng.uniform(-bound0, bound0, (self.n_features, self.n_hidden))
        self.b0 = np.zeros(self.n_hidden, dtype=np.float64)
        self.W1 = rng.uniform(-bound1, bound1, (self.n_hidden, self.n_out))
        self.b1 = np.zeros(self.n_out, dtype=np.float64)

    def forward(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        features : (n_nodes, n_features)
        adjacency : (n_nodes, n_nodes)

        Returns
        -------
        np.ndarray
            Node embeddings of shape (n_nodes, n_out).
        """
        features = np.asarray(features, dtype=np.float64)
        adjacency = np.asarray(adjacency, dtype=np.float64)
        a_norm = _normalize_adjacency(adjacency)

        h0 = _relu(a_norm @ features @ self.W0 + self.b0)
        h1 = a_norm @ h0 @ self.W1 + self.b1
        return h1

    def train_step(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 1e-2,
    ) -> float:
        """
        Single SGD step against a regression target.

        Returns the mean squared error as the loss.
        """
        features = np.asarray(features, dtype=np.float64)
        adjacency = np.asarray(adjacency, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        a_norm = _normalize_adjacency(adjacency)

        # Forward pass with caches
        z0 = a_norm @ features @ self.W0 + self.b0
        h0 = _relu(z0)
        z1 = a_norm @ h0 @ self.W1 + self.b1  # (n_nodes, n_out)

        err = z1 - targets
        loss = float(np.mean(err * err))

        # Backprop
        grad_z1 = 2.0 * err / targets.size
        grad_W1 = (a_norm @ h0).T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)

        grad_h0 = a_norm.T @ grad_z1 @ self.W1.T  # propagate back through a_norm
        grad_z0 = grad_h0 * (z0 > 0).astype(np.float64)  # ReLU derivative
        grad_W0 = (a_norm @ features).T @ grad_z0
        grad_b0 = grad_z0.sum(axis=0)

        # SGD update
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        self.W0 -= learning_rate * grad_W0
        self.b0 -= learning_rate * grad_b0

        return loss


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def gcn_forward(
    features: np.ndarray,
    adjacency: np.ndarray,
    n_hidden: int = 32,
    n_out: int = 16,
    seed: int = 42,
) -> np.ndarray:
    """Stateless GCN forward pass — creates a fresh GCN each call."""
    features = np.asarray(features, dtype=np.float64)
    n_features = features.shape[1] if features.ndim == 2 else 1
    model = GCN(n_features=n_features, n_hidden=n_hidden, n_out=n_out, seed=seed)
    return model.forward(features, adjacency)
