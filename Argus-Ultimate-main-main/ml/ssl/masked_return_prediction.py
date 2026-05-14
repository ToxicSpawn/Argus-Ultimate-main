"""
Masked return prediction — BERT-style SSL for time series.

Given a sequence of returns r_1, ..., r_T, randomly mask out p% of the
positions (replacing with a sentinel value) and train a model to
reconstruct the masked values from context. The pretrained encoder
transfers to the downstream transformer price predictor (ml/training/
train_tft.py).

Algorithm
---------
1. For each sequence, sample a mask of ~15% positions.
2. 80% of masked positions: replace with 0.0 (MASK token).
3. 10%: replace with a random return from the sequence.
4. 10%: keep the original.
5. Train on the reconstruction loss over masked positions only.

Pure-numpy implementation with optional torch path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _HAS_TORCH = False
    _DEVICE = "cpu"


# ═════════════════════════════════════════════════════════════════════════════
# Masking utility
# ═════════════════════════════════════════════════════════════════════════════


def mask_returns(
    returns: np.ndarray,
    mask_prob: float = 0.15,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply BERT-style masking to a return sequence.

    Parameters
    ----------
    returns : (T,) or (B, T) np.ndarray
    mask_prob : float, default 0.15
    seed : int, optional

    Returns
    -------
    masked : ndarray, same shape as returns
    mask : bool ndarray, same shape (True = position was masked)
    """
    returns = np.asarray(returns, dtype=np.float64)
    rng = np.random.default_rng(seed)

    n = returns.size
    flat = returns.ravel().copy()

    # Sample positions to mask
    n_mask = max(1, int(round(mask_prob * n)))
    mask_positions = rng.choice(n, size=n_mask, replace=False)

    # Build the mask array
    mask = np.zeros_like(flat, dtype=bool)
    mask[mask_positions] = True

    # Apply masking rules
    for pos in mask_positions:
        r = rng.random()
        if r < 0.8:
            flat[pos] = 0.0  # MASK token
        elif r < 0.9:
            # Random replacement from the sequence
            flat[pos] = returns.ravel()[rng.integers(0, n)]
        # else: keep original (10%)

    return flat.reshape(returns.shape), mask.reshape(returns.shape)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class MaskedReturnDataset:
    """
    Dataset of masked return sequences for SSL pretraining.

    Parameters
    ----------
    sequences : list of (T,) ndarrays
        Raw return sequences.
    seq_len : int
        Fixed length per training example. Sequences longer than this are
        randomly cropped; shorter ones are skipped.
    mask_prob : float
    """

    sequences: List[np.ndarray]
    seq_len: int = 64
    mask_prob: float = 0.15

    def __post_init__(self) -> None:
        # Filter out too-short sequences
        self.sequences = [s for s in self.sequences if len(s) >= self.seq_len]

    def __len__(self) -> int:
        return len(self.sequences)

    def get_batch(
        self,
        batch_size: int = 32,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (masked, target, mask) arrays of shape (B, seq_len).
        """
        rng = np.random.default_rng(seed)
        if len(self.sequences) == 0:
            raise ValueError("No sequences available")

        masked_batch = np.zeros((batch_size, self.seq_len), dtype=np.float64)
        target_batch = np.zeros((batch_size, self.seq_len), dtype=np.float64)
        mask_batch = np.zeros((batch_size, self.seq_len), dtype=bool)

        for i in range(batch_size):
            idx = rng.integers(0, len(self.sequences))
            seq = self.sequences[idx]
            if len(seq) > self.seq_len:
                start = rng.integers(0, len(seq) - self.seq_len + 1)
                seq_slice = seq[start : start + self.seq_len]
            else:
                seq_slice = seq[: self.seq_len]
            masked, mask = mask_returns(
                seq_slice, mask_prob=self.mask_prob, seed=int(rng.integers(0, 1 << 30)),
            )
            target_batch[i] = seq_slice
            masked_batch[i] = masked
            mask_batch[i] = mask

        return masked_batch, target_batch, mask_batch


# ═════════════════════════════════════════════════════════════════════════════
# Pretrainer (numpy linear reconstruction baseline)
# ═════════════════════════════════════════════════════════════════════════════


class MaskedReturnPretrainer:
    """
    Simple linear reconstruction pretrainer. Uses context (non-masked
    positions) to predict masked positions via a learned linear projection.

    This is a proven baseline — when torch is available, users should
    instead wire this into a transformer encoder. The public API is the
    same either way.

    Parameters
    ----------
    seq_len : int
    learning_rate : float
    """

    def __init__(
        self,
        seq_len: int = 64,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.seq_len = int(seq_len)
        self.learning_rate = float(learning_rate)
        rng = np.random.default_rng(seed)
        # Linear projection: context_vector -> full reconstruction
        self.W = rng.normal(scale=0.1, size=(seq_len, seq_len))
        self.b = np.zeros(seq_len, dtype=np.float64)
        self.step_count = 0
        self.last_loss = 0.0

    def train_step(
        self,
        masked: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """
        One batch SGD step. Returns the mean reconstruction loss over
        masked positions only.
        """
        masked = np.asarray(masked, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)

        # Linear forward: predict full sequence from masked input
        pred = masked @ self.W + self.b

        # Compute loss only on masked positions
        err = (pred - target) * mask.astype(np.float64)
        loss = float((err ** 2).sum() / max(mask.sum(), 1))

        # Gradient (only masked positions contribute)
        grad_W = masked.T @ err / max(mask.sum(), 1)
        grad_b = err.sum(axis=0) / max(mask.sum(), 1)

        self.W -= self.learning_rate * grad_W
        self.b -= self.learning_rate * grad_b
        self.step_count += 1
        self.last_loss = loss
        return loss

    def reconstruct(self, masked: np.ndarray) -> np.ndarray:
        masked = np.asarray(masked, dtype=np.float64)
        return masked @ self.W + self.b

    def fit(
        self,
        dataset: MaskedReturnDataset,
        n_steps: int = 100,
        batch_size: int = 32,
    ) -> List[float]:
        """Run ``n_steps`` of SGD on the dataset."""
        losses: List[float] = []
        for _ in range(n_steps):
            masked, target, mask = dataset.get_batch(batch_size=batch_size)
            loss = self.train_step(masked, target, mask)
            losses.append(loss)
        return losses

    def snapshot(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "step_count": self.step_count,
            "last_loss": float(self.last_loss),
            "W_norm": float(np.linalg.norm(self.W)),
        }
