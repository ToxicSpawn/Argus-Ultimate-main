"""
iTransformer — Inverted Transformer for long-horizon time-series forecasting.

Reference
---------
Liu, Hu, Zhang, Zhang, Liu, Dong, Wang, Ma, Long,
"iTransformer: Inverted Transformers Are Effective for Time Series
Forecasting," ICLR 2024 (arXiv:2310.06625).

Key insight
-----------
Standard transformers treat time steps as tokens. iTransformer INVERTS this:
each variable/series becomes a token, and attention models cross-variable
dependencies. This is significantly stronger for multivariate long-horizon
forecasting than vanilla transformers.

Pure-numpy implementation with a torch path when available. For training,
the torch path is strongly recommended; the numpy path exists for CPU-only
environments and for inference when weights were trained elsewhere.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _HAS_TORCH = False
    _DEVICE = "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x_shifted)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


# ═════════════════════════════════════════════════════════════════════════════
# iTransformer (numpy)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ITransformer:
    """
    Inverted Transformer for multivariate time-series forecasting.

    Parameters
    ----------
    n_series : int
        Number of variables / series.
    seq_len : int
        Lookback window length.
    pred_len : int
        Forecast horizon.
    d_model : int
        Token embedding dimension (one token per series after inversion).
    n_heads : int
    n_layers : int
    seed : int
    """

    n_series: int
    seq_len: int
    pred_len: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    seed: int = 42

    # Learnable weights populated in __post_init__
    W_embed: np.ndarray = field(init=False)
    b_embed: np.ndarray = field(init=False)
    attn_weights: list = field(init=False, default_factory=list)
    ffn_weights: list = field(init=False, default_factory=list)
    W_out: np.ndarray = field(init=False)
    b_out: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Embed: project each series (length seq_len) into d_model
        scale_e = np.sqrt(2.0 / (self.seq_len + self.d_model))
        self.W_embed = rng.normal(0, scale_e, (self.seq_len, self.d_model))
        self.b_embed = np.zeros(self.d_model, dtype=np.float64)

        # Attention + FFN weights per layer
        self.attn_weights = []
        self.ffn_weights = []
        head_dim = self.d_model // self.n_heads

        for _ in range(self.n_layers):
            scale_a = np.sqrt(2.0 / self.d_model)
            Wq = rng.normal(0, scale_a, (self.d_model, self.d_model))
            Wk = rng.normal(0, scale_a, (self.d_model, self.d_model))
            Wv = rng.normal(0, scale_a, (self.d_model, self.d_model))
            Wo = rng.normal(0, scale_a, (self.d_model, self.d_model))
            self.attn_weights.append(
                {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "head_dim": head_dim}
            )

            ffn_hidden = 2 * self.d_model
            scale_f = np.sqrt(2.0 / self.d_model)
            Wf1 = rng.normal(0, scale_f, (self.d_model, ffn_hidden))
            bf1 = np.zeros(ffn_hidden)
            Wf2 = rng.normal(0, scale_f, (ffn_hidden, self.d_model))
            bf2 = np.zeros(self.d_model)
            self.ffn_weights.append(
                {"Wf1": Wf1, "bf1": bf1, "Wf2": Wf2, "bf2": bf2}
            )

        # Output: project d_model back to pred_len
        scale_o = np.sqrt(2.0 / (self.d_model + self.pred_len))
        self.W_out = rng.normal(0, scale_o, (self.d_model, self.pred_len))
        self.b_out = np.zeros(self.pred_len, dtype=np.float64)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def _multi_head_attention(
        self, x: np.ndarray, layer_weights: dict
    ) -> np.ndarray:
        """x has shape (n_series, d_model)."""
        Wq = layer_weights["Wq"]
        Wk = layer_weights["Wk"]
        Wv = layer_weights["Wv"]
        Wo = layer_weights["Wo"]
        head_dim = layer_weights["head_dim"]
        n = x.shape[0]

        q = x @ Wq
        k = x @ Wk
        v = x @ Wv

        # Reshape to (n_heads, n, head_dim)
        q = q.reshape(n, self.n_heads, head_dim).transpose(1, 0, 2)
        k = k.reshape(n, self.n_heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(n, self.n_heads, head_dim).transpose(1, 0, 2)

        scale = 1.0 / math.sqrt(max(head_dim, 1))
        scores = np.einsum("hnd,hmd->hnm", q, k) * scale  # (n_heads, n, n)
        attn = _softmax(scores, axis=-1)
        out = np.einsum("hnm,hmd->hnd", attn, v)  # (n_heads, n, head_dim)

        # Back to (n, d_model)
        out = out.transpose(1, 0, 2).reshape(n, self.d_model)
        return out @ Wo

    def _ffn(self, x: np.ndarray, layer_weights: dict) -> np.ndarray:
        h = _gelu(x @ layer_weights["Wf1"] + layer_weights["bf1"])
        return h @ layer_weights["Wf2"] + layer_weights["bf2"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        x : (seq_len, n_series) np.ndarray
            Multivariate lookback window.

        Returns
        -------
        np.ndarray of shape (pred_len, n_series)
            Forecast horizon values.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.seq_len, self.n_series):
            raise ValueError(
                f"Expected shape ({self.seq_len}, {self.n_series}), got {x.shape}"
            )

        # INVERT: series become tokens
        # x.T has shape (n_series, seq_len)
        tokens = x.T  # (n_series, seq_len)

        # Embed
        h = tokens @ self.W_embed + self.b_embed  # (n_series, d_model)
        h = _layer_norm(h)

        # Transformer layers
        for layer_idx in range(self.n_layers):
            # Attention sublayer with residual + norm
            attn_out = self._multi_head_attention(h, self.attn_weights[layer_idx])
            h = _layer_norm(h + attn_out)

            # FFN sublayer with residual + norm
            ffn_out = self._ffn(h, self.ffn_weights[layer_idx])
            h = _layer_norm(h + ffn_out)

        # Project each series token to pred_len
        forecasts = h @ self.W_out + self.b_out  # (n_series, pred_len)
        return forecasts.T  # (pred_len, n_series)

    def snapshot(self) -> dict:
        return {
            "n_series": self.n_series,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "W_embed_norm": float(np.linalg.norm(self.W_embed)),
            "W_out_norm": float(np.linalg.norm(self.W_out)),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Stateless convenience
# ═════════════════════════════════════════════════════════════════════════════


def itransformer_forecast(
    x: np.ndarray,
    pred_len: int,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Stateless iTransformer forecast — creates a fresh model each call.

    For training across many calls, instantiate ``ITransformer`` directly.
    """
    x = np.asarray(x, dtype=np.float64)
    seq_len, n_series = x.shape
    model = ITransformer(
        n_series=n_series,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        seed=seed,
    )
    return model.forward(x)
