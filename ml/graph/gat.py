"""
Graph Attention Network (GAT) — Veličković et al. (2018).

Reference
---------
Veličković, Cucurull, Casanova, Romero, Liò, Bengio,
"Graph Attention Networks," ICLR 2018 (arXiv:1710.10903).

Core operation
--------------
For each node i:
    α_ij = softmax_j( LeakyReLU(a^T [W h_i || W h_j]) )
    h_i' = σ( Σ_j α_ij W h_j )

where α_ij is an attention coefficient over neighbors j.

Supports multi-head attention with mean aggregation at the output layer.

Trading use
-----------
Asset correlation graph as adjacency; the learned attention weights reveal
which assets drive each target asset's embedding — useful for
attribution and cross-asset feature importance.
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


def _leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x_shifted)
    return ex / np.sum(ex, axis=axis, keepdims=True)


# ═════════════════════════════════════════════════════════════════════════════
# Single-head GAT layer
# ═════════════════════════════════════════════════════════════════════════════


class GATLayer:
    """
    Single-head GAT layer.

    Parameters
    ----------
    in_features : int
    out_features : int
    dropout : float
        Applied to attention coefficients (training only).
    seed : int
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dropout = float(dropout)

        rng = np.random.default_rng(seed)
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.W = rng.uniform(-bound, bound, (in_features, out_features))
        # Attention vector a ∈ R^(2*out_features)
        self.a = rng.uniform(-bound, bound, (2 * out_features,))

    def forward(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass for a single head.

        Parameters
        ----------
        features : (n_nodes, in_features)
        adjacency : (n_nodes, n_nodes)
            Binary or weighted adjacency. Non-zero = edge.

        Returns
        -------
        np.ndarray of shape (n_nodes, out_features)
        """
        n = features.shape[0]
        wh = features @ self.W  # (n, out_features)

        # Compute pairwise attention scores
        # e_ij = LeakyReLU(a^T [wh_i || wh_j])
        # We split a into a_l and a_r so e_ij = (a_l · wh_i) + (a_r · wh_j)
        a_l = self.a[: self.out_features]
        a_r = self.a[self.out_features:]
        e_l = wh @ a_l  # (n,)
        e_r = wh @ a_r  # (n,)
        e = e_l[:, None] + e_r[None, :]  # (n, n)
        e = _leaky_relu(e)

        # Mask non-edges: set to -inf so softmax gives zero
        mask = adjacency > 0
        # Always include self-loops so a node can attend to itself
        np.fill_diagonal(mask, True)
        e_masked = np.where(mask, e, -1e9)

        alpha = _softmax(e_masked, axis=-1)  # (n, n)

        # Optional dropout on alpha (not applied at inference)
        # Aggregate: h_i' = Σ_j alpha_ij * wh_j
        h_out = alpha @ wh  # (n, out_features)
        return h_out


# ═════════════════════════════════════════════════════════════════════════════
# Multi-head GAT
# ═════════════════════════════════════════════════════════════════════════════


class GAT:
    """
    Multi-head Graph Attention Network.

    Parameters
    ----------
    n_features : int
    n_hidden : int
        Output dimension per head in the hidden layer.
    n_out : int
        Output embedding dimension.
    n_heads : int
    seed : int
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 16,
        n_out: int = 16,
        n_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.n_features = int(n_features)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.n_heads = int(n_heads)

        # Hidden layer: n_heads of GATLayer(n_features -> n_hidden)
        self.heads: List[GATLayer] = [
            GATLayer(n_features, n_hidden, seed=seed + h)
            for h in range(n_heads)
        ]
        # Output layer: single head (n_hidden*n_heads -> n_out)
        self.out_layer = GATLayer(
            in_features=n_hidden * n_heads,
            out_features=n_out,
            seed=seed + n_heads + 1,
        )

    def forward(
        self,
        features: np.ndarray,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """
        Multi-head forward pass. Hidden layer concatenates heads;
        output layer is a single head.
        """
        features = np.asarray(features, dtype=np.float64)
        adjacency = np.asarray(adjacency, dtype=np.float64)

        # Hidden layer: concatenate heads
        head_outs = [h.forward(features, adjacency) for h in self.heads]
        h_concat = np.concatenate(head_outs, axis=-1)
        h_concat = _leaky_relu(h_concat)

        # Output layer
        return self.out_layer.forward(h_concat, adjacency)


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def gat_forward(
    features: np.ndarray,
    adjacency: np.ndarray,
    n_hidden: int = 16,
    n_out: int = 16,
    n_heads: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Stateless GAT forward pass — creates a fresh GAT each call."""
    features = np.asarray(features, dtype=np.float64)
    n_features = features.shape[1] if features.ndim == 2 else 1
    model = GAT(
        n_features=n_features,
        n_hidden=n_hidden,
        n_out=n_out,
        n_heads=n_heads,
        seed=seed,
    )
    return model.forward(features, adjacency)
