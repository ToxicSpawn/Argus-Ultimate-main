"""
Attention-based Order Flow Predictor — uses multi-head self-attention
over order flow feature sequences to predict short-term price direction.

Pure numpy implementation. Features per timestep:
  price_change, volume, bid_vol, ask_vol, spread, trade_imbalance,
  vwap_deviation, volatility

The attention mechanism identifies which historical ticks are most
relevant for the next price move prediction.

Usage:
    predictor = AttentionOrderFlowPredictor(seq_len=50, n_heads=4)
    for tick in ticks:
        predictor.update(tick)
    result = predictor.predict_direction()
    attention_map = predictor.get_attention_map()
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Feature names and their indices
FEATURE_NAMES = [
    "price_change",
    "volume",
    "bid_vol",
    "ask_vol",
    "spread",
    "trade_imbalance",
    "vwap_deviation",
    "volatility",
]
_N_FEATURES = len(FEATURE_NAMES)


class AttentionOrderFlowPredictor:
    """
    Multi-head self-attention on order flow sequences for price direction
    prediction.

    Architecture:
      Input: (seq_len, feature_dim) sequence of tick features
      → Positional encoding (sinusoidal)
      → Multi-head self-attention
      → Layer norm + residual
      → Feed-forward (hidden → output)
      → Direction prediction [-1, 1]
    """

    def __init__(
        self,
        seq_len: int = 50,
        feature_dim: int = 8,
        n_heads: int = 4,
        ff_dim: int = 32,
    ) -> None:
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        assert feature_dim % n_heads == 0, (
            f"feature_dim ({feature_dim}) must be divisible by n_heads ({n_heads})"
        )
        self.head_dim = feature_dim // n_heads

        # Tick buffer
        self._buffer: Deque[np.ndarray] = deque(maxlen=seq_len)

        # Attention projection weights (per-head)
        # Q, K, V projections: (feature_dim, feature_dim) each
        self._W_Q = self._xavier(feature_dim, feature_dim)
        self._W_K = self._xavier(feature_dim, feature_dim)
        self._W_V = self._xavier(feature_dim, feature_dim)
        self._W_O = self._xavier(feature_dim, feature_dim)  # output projection

        # Layer norm parameters
        self._ln_gamma = np.ones(feature_dim)
        self._ln_beta = np.zeros(feature_dim)

        # Feed-forward network
        self._W_ff1 = self._xavier(feature_dim, ff_dim)
        self._b_ff1 = np.zeros(ff_dim)
        self._W_ff2 = self._xavier(ff_dim, 1)
        self._b_ff2 = np.zeros(1)

        # Positional encoding (precomputed)
        self._pos_encoding = self._build_positional_encoding(seq_len, feature_dim)

        # Cached attention weights from last prediction
        self._last_attention_weights: Optional[np.ndarray] = None

        # Feature normalization (running stats)
        self._feat_mean = np.zeros(feature_dim)
        self._feat_var = np.ones(feature_dim)
        self._feat_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tick_features: dict) -> None:
        """
        Add a new tick observation.

        tick_features: dict with keys from FEATURE_NAMES.
        Missing features default to 0.
        """
        vec = np.zeros(self.feature_dim)
        for i, name in enumerate(FEATURE_NAMES[:self.feature_dim]):
            val = tick_features.get(name, 0.0)
            vec[i] = float(val) if val is not None else 0.0

        # Update running normalization stats
        self._feat_count += 1
        alpha = 1.0 / self._feat_count
        delta = vec - self._feat_mean
        self._feat_mean += alpha * delta
        self._feat_var = (1.0 - alpha) * self._feat_var + alpha * (delta ** 2)

        # Normalize before storing
        std = np.sqrt(self._feat_var + 1e-8)
        normalized = (vec - self._feat_mean) / std
        self._buffer.append(normalized)

    def predict_direction(self) -> dict:
        """
        Predict next 5-minute price direction.

        Returns:
            {direction: float [-1 to 1], confidence: float,
             attention_weights: list, dominant_feature: str}
        """
        if len(self._buffer) < 5:
            return {
                "direction": 0.0,
                "confidence": 0.0,
                "attention_weights": [],
                "dominant_feature": "insufficient_data",
            }

        # Build sequence matrix
        seq = np.array(list(self._buffer))  # (T, feature_dim)
        T = seq.shape[0]

        # Add positional encoding
        seq = seq + self._pos_encoding[:T, :]

        # Multi-head self-attention
        attn_out, attn_weights = self._multi_head_attention(seq)

        # Store for later inspection
        self._last_attention_weights = attn_weights

        # Residual + layer norm
        x = self._layer_norm(seq + attn_out)

        # Global average pooling over sequence dimension
        pooled = x.mean(axis=0)  # (feature_dim,)

        # Feed-forward to scalar prediction
        h = self._relu(pooled @ self._W_ff1 + self._b_ff1)
        raw = float((h @ self._W_ff2 + self._b_ff2)[0])

        # Tanh activation for bounded direction
        direction = float(np.tanh(raw))

        # Confidence from attention entropy
        # Low entropy = model is focused = high confidence
        if attn_weights is not None and attn_weights.size > 0:
            # Average attention across heads, use last token's attention
            avg_attn = attn_weights.mean(axis=0)  # (T, T)
            last_attn = avg_attn[-1]  # attention from last position
            entropy = -float(np.sum(last_attn * np.log(np.maximum(last_attn, 1e-10))))
            max_entropy = math.log(max(T, 1))
            confidence = max(0.0, 1.0 - entropy / max(max_entropy, 1e-8))
        else:
            confidence = 0.0

        # Dominant feature: which feature dimension has highest absolute
        # attention-weighted contribution
        feature_importance = np.abs(pooled)
        dominant_idx = int(np.argmax(feature_importance))
        dominant_feature = (
            FEATURE_NAMES[dominant_idx]
            if dominant_idx < len(FEATURE_NAMES)
            else f"feature_{dominant_idx}"
        )

        return {
            "direction": round(direction, 6),
            "confidence": round(confidence, 4),
            "attention_weights": (
                avg_attn[-1].tolist() if attn_weights is not None else []
            ),
            "dominant_feature": dominant_feature,
        }

    def get_attention_map(self) -> np.ndarray:
        """
        Return attention weights from last prediction.

        Shape: (n_heads, T, T) where T is the sequence length used.
        """
        if self._last_attention_weights is None:
            return np.array([])
        return self._last_attention_weights.copy()

    # ------------------------------------------------------------------
    # Multi-head self-attention
    # ------------------------------------------------------------------

    def _multi_head_attention(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-head self-attention.

        Args:
            x: (T, feature_dim) input sequence

        Returns:
            (output, attention_weights) where output is (T, feature_dim)
            and attention_weights is (n_heads, T, T).
        """
        T, d = x.shape

        # Linear projections
        Q = x @ self._W_Q  # (T, d)
        K = x @ self._W_K
        V = x @ self._W_V

        # Split into heads
        Q_heads = Q.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)  # (h, T, hd)
        K_heads = K.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
        V_heads = V.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)

        # Scaled dot-product attention per head
        scale = math.sqrt(self.head_dim)
        scores = np.matmul(Q_heads, K_heads.transpose(0, 2, 1)) / scale  # (h, T, T)

        # Causal mask: attend only to past positions
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[:, mask] = -1e9

        # Softmax
        attn_weights = np.zeros_like(scores)
        for head in range(self.n_heads):
            for i in range(T):
                row = scores[head, i]
                row = row - row.max()
                exp_row = np.exp(row)
                attn_weights[head, i] = exp_row / (exp_row.sum() + 1e-10)

        # Weighted sum of values
        context = np.matmul(attn_weights, V_heads)  # (h, T, hd)

        # Concatenate heads
        context = context.transpose(1, 0, 2).reshape(T, d)  # (T, d)

        # Output projection
        output = context @ self._W_O

        return output, attn_weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization along last dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return self._ln_gamma * x_norm + self._ln_beta

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
        """Sinusoidal positional encoding."""
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term[:d_model // 2])
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])

        return pe
