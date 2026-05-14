"""
Multi-modal fusion layer — combines numeric, text, and graph features.

Architecture
------------
Each modality is first projected into a shared latent space ``d_shared``
via modality-specific linear projections:

    z_numeric = W_n @ numeric_features
    z_text    = W_t @ text_embedding
    z_graph   = W_g @ graph_embedding

Then cross-attention is applied:

    Each modality attends to the others.
    Final fused vector = concat(z_numeric, z_text, z_graph) @ W_out

Supports graceful degradation: if text or graph modality is missing,
the fusion uses only what's provided, re-normalizing the output.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x_shifted)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


# ═════════════════════════════════════════════════════════════════════════════
# MultiModalFusion
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MultiModalFusion:
    """
    Cross-attention fusion of numeric, text, and graph modalities.

    Parameters
    ----------
    d_numeric : int
    d_text : int
    d_graph : int
    d_shared : int
        Shared latent dimension after projection.
    d_out : int
        Final fused vector dimension.
    seed : int
    """

    d_numeric: int
    d_text: int
    d_graph: int
    d_shared: int = 32
    d_out: int = 16
    seed: int = 42

    W_n: np.ndarray = field(init=False)
    W_t: np.ndarray = field(init=False)
    W_g: np.ndarray = field(init=False)
    W_out: np.ndarray = field(init=False)
    W_q: np.ndarray = field(init=False)
    W_k: np.ndarray = field(init=False)
    W_v: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Modality projections → shared latent
        bound_n = np.sqrt(6.0 / (self.d_numeric + self.d_shared))
        bound_t = np.sqrt(6.0 / (self.d_text + self.d_shared))
        bound_g = np.sqrt(6.0 / (self.d_graph + self.d_shared))

        self.W_n = rng.uniform(-bound_n, bound_n, (self.d_numeric, self.d_shared))
        self.W_t = rng.uniform(-bound_t, bound_t, (self.d_text, self.d_shared))
        self.W_g = rng.uniform(-bound_g, bound_g, (self.d_graph, self.d_shared))

        # Cross-attention QKV projections (shared across modalities)
        scale = np.sqrt(2.0 / self.d_shared)
        self.W_q = rng.normal(0, scale, (self.d_shared, self.d_shared))
        self.W_k = rng.normal(0, scale, (self.d_shared, self.d_shared))
        self.W_v = rng.normal(0, scale, (self.d_shared, self.d_shared))

        # Output projection (concat all modalities)
        bound_o = np.sqrt(6.0 / (3 * self.d_shared + self.d_out))
        self.W_out = rng.uniform(-bound_o, bound_o, (3 * self.d_shared, self.d_out))

    def forward(
        self,
        numeric: Optional[np.ndarray] = None,
        text: Optional[np.ndarray] = None,
        graph: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse available modalities.

        Parameters
        ----------
        numeric : (d_numeric,) or None
        text : (d_text,) or None
        graph : (d_graph,) or None

        Returns
        -------
        np.ndarray of shape (d_out,) — fused feature vector.
        """
        # Project available modalities into shared space; use zeros for missing
        z_n = (numeric @ self.W_n) if numeric is not None else np.zeros(self.d_shared)
        z_t = (text @ self.W_t) if text is not None else np.zeros(self.d_shared)
        z_g = (graph @ self.W_g) if graph is not None else np.zeros(self.d_shared)

        # Stack into (3, d_shared) token matrix
        tokens = np.stack([z_n, z_t, z_g], axis=0)

        # Cross-attention
        q = tokens @ self.W_q
        k = tokens @ self.W_k
        v = tokens @ self.W_v

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(max(self.d_shared, 1))
        scores = q @ k.T * scale  # (3, 3)

        # Mask missing modalities (attention weight → 0)
        mask = np.array(
            [
                numeric is not None,
                text is not None,
                graph is not None,
            ],
            dtype=bool,
        )
        scores = np.where(mask[None, :], scores, -1e9)
        attn = _softmax(scores, axis=-1)
        attended = attn @ v  # (3, d_shared)

        # Concat all modalities with attention-refined features
        fused = attended.reshape(-1)  # (3 * d_shared,)

        # Output projection + tanh squash
        out = _tanh(fused @ self.W_out)
        return out

    def snapshot(self) -> Dict:
        return {
            "d_numeric": self.d_numeric,
            "d_text": self.d_text,
            "d_graph": self.d_graph,
            "d_shared": self.d_shared,
            "d_out": self.d_out,
            "W_n_norm": float(np.linalg.norm(self.W_n)),
            "W_t_norm": float(np.linalg.norm(self.W_t)),
            "W_g_norm": float(np.linalg.norm(self.W_g)),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def fuse_modalities(
    numeric: Optional[np.ndarray] = None,
    text: Optional[np.ndarray] = None,
    graph: Optional[np.ndarray] = None,
    d_shared: int = 32,
    d_out: int = 16,
    seed: int = 42,
) -> np.ndarray:
    """One-shot modality fusion with a fresh MultiModalFusion instance."""
    d_numeric = numeric.shape[-1] if numeric is not None else 1
    d_text = text.shape[-1] if text is not None else 1
    d_graph = graph.shape[-1] if graph is not None else 1

    fusion = MultiModalFusion(
        d_numeric=d_numeric,
        d_text=d_text,
        d_graph=d_graph,
        d_shared=d_shared,
        d_out=d_out,
        seed=seed,
    )
    return fusion.forward(numeric=numeric, text=text, graph=graph)
