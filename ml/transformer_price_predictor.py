"""
Transformer-Based Price Predictor — uses a lightweight self-attention model
to predict next-bar price direction and magnitude from OHLCV sequences.

When PyTorch is available, trains a small Transformer encoder on normalised
OHLCV windows.  When PyTorch is unavailable, falls back to a momentum-based
prediction using exponential-weighted trend of the last N bars.

Tracks every prediction for calibration so ``get_accuracy()`` reports
realised hit-rate over a rolling window.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PricePrediction:
    """Single next-bar prediction."""
    direction: str           # "up" or "down"
    magnitude_pct: float     # expected percentage move (always positive)
    confidence: float        # 0.0 – 1.0
    timestamp: float         # epoch seconds when prediction was made


@dataclass
class _PredictionRecord:
    """Internal bookkeeping for calibration."""
    prediction: PricePrediction
    price_at_prediction: float
    actual_return: Optional[float] = None   # filled once next bar arrives


# ---------------------------------------------------------------------------
# Torch Transformer (optional)
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]


def _build_transformer(input_dim: int = 5, d_model: int = 32,
                       nhead: int = 4, num_layers: int = 2,
                       dropout: float = 0.1) -> Any:
    """Build a small Transformer encoder + linear head (returns nn.Module)."""
    if not _TORCH_AVAILABLE:
        return None

    class _PriceTransformer(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 2)   # [direction_logit, magnitude]

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            """x: (batch, seq_len, input_dim) -> (batch, 2)."""
            h = self.input_proj(x)
            h = self.encoder(h)
            h = h[:, -1, :]          # last token representation
            return self.head(h)

    return _PriceTransformer()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TransformerPricePredictor:
    """
    Transformer-based (or momentum-fallback) next-bar price predictor.

    Parameters
    ----------
    fallback_lookback : int
        Number of recent bars used by the momentum fallback predictor.
    max_history : int
        Maximum prediction records kept for calibration.
    """

    def __init__(self, fallback_lookback: int = 20, max_history: int = 2000) -> None:
        self._fallback_lookback = fallback_lookback
        self._max_history = max_history

        # Torch model (None until fit() or if torch unavailable)
        self._model: Any = None
        self._seq_len: int = 60
        self._trained: bool = False

        # Fallback statistics
        self._mean_return: float = 0.0
        self._std_return: float = 1e-8
        self._ema_alpha: float = 2.0 / (fallback_lookback + 1)

        # Calibration tracking
        self._predictions: Deque[_PredictionRecord] = deque(maxlen=max_history)
        self._last_close: Optional[float] = None

        logger.info(
            "TransformerPricePredictor initialised (torch=%s, fallback_lookback=%d)",
            _TORCH_AVAILABLE, fallback_lookback,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, ohlcv_sequences: list, seq_len: int = 60, epochs: int = 10) -> Dict[str, Any]:
        """
        Train on OHLCV sequences.

        Parameters
        ----------
        ohlcv_sequences : list
            List of bars, each bar is [open, high, low, close, volume].
        seq_len : int
            Lookback window length.
        epochs : int
            Training epochs (only used when PyTorch is available).

        Returns
        -------
        dict
            Training summary with keys: method, epochs_run, final_loss, n_samples.
        """
        self._seq_len = seq_len
        bars = np.asarray(ohlcv_sequences, dtype=np.float64)

        if len(bars) < seq_len + 1:
            logger.warning("fit: need at least %d bars, got %d — storing stats only",
                           seq_len + 1, len(bars))
            return self._fit_fallback(bars)

        # Compute returns for fallback stats regardless
        closes = bars[:, 3]
        returns = np.diff(closes) / (closes[:-1] + 1e-12)
        self._mean_return = float(np.mean(returns))
        self._std_return = float(np.std(returns)) + 1e-12

        if _TORCH_AVAILABLE:
            return self._fit_torch(bars, seq_len, epochs)

        logger.info("fit: PyTorch unavailable — using momentum fallback")
        return self._fit_fallback(bars)

    def _fit_fallback(self, bars: np.ndarray) -> Dict[str, Any]:
        """Store descriptive statistics for momentum-based prediction."""
        if len(bars) > 1:
            closes = bars[:, 3]
            returns = np.diff(closes) / (closes[:-1] + 1e-12)
            self._mean_return = float(np.mean(returns))
            self._std_return = float(np.std(returns)) + 1e-12
        self._trained = True
        logger.info("fit fallback: mean_return=%.6f, std_return=%.6f",
                     self._mean_return, self._std_return)
        return {"method": "momentum_fallback", "epochs_run": 0,
                "final_loss": 0.0, "n_samples": len(bars)}

    def _fit_torch(self, bars: np.ndarray, seq_len: int, epochs: int) -> Dict[str, Any]:
        """Train Transformer on OHLCV windows."""
        # Normalise each feature column to zero-mean unit-variance
        means = bars.mean(axis=0)
        stds = bars.std(axis=0) + 1e-12
        normed = (bars - means) / stds

        # Build sliding windows → (X, y)
        X_list, y_list = [], []
        closes = bars[:, 3]
        for i in range(len(normed) - seq_len):
            X_list.append(normed[i:i + seq_len])
            # Target: next-bar return
            ret = (closes[i + seq_len] - closes[i + seq_len - 1]) / (closes[i + seq_len - 1] + 1e-12)
            direction = 1.0 if ret >= 0 else 0.0
            magnitude = abs(ret)
            y_list.append([direction, magnitude])

        X_t = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_t = torch.tensor(np.array(y_list), dtype=torch.float32)

        self._model = _build_transformer(input_dim=bars.shape[1])
        self._norm_means = means
        self._norm_stds = stds

        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        n_samples = len(X_t)
        batch_size = min(64, n_samples)
        final_loss = 0.0

        self._model.train()
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                idx = indices[start:start + batch_size]
                xb, yb = X_t[idx], y_t[idx]
                pred = self._model(xb)
                loss = bce(pred[:, 0], yb[:, 0]) + mse(pred[:, 1], yb[:, 1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % max(epochs // 3, 1) == 0 or epoch == epochs - 1:
                logger.info("fit torch: epoch %d/%d  loss=%.6f", epoch + 1, epochs, final_loss)

        self._model.eval()
        self._trained = True
        logger.info("fit torch complete: %d samples, %d epochs, final_loss=%.6f",
                     n_samples, epochs, final_loss)
        return {"method": "transformer", "epochs_run": epochs,
                "final_loss": final_loss, "n_samples": n_samples}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next_bar(self, recent_bars: list) -> PricePrediction:
        """
        Predict the next bar's direction and magnitude.

        Parameters
        ----------
        recent_bars : list
            Recent OHLCV bars (at least ``fallback_lookback`` bars).

        Returns
        -------
        PricePrediction
        """
        now = time.time()
        bars = np.asarray(recent_bars, dtype=np.float64)

        if len(bars) < 2:
            pred = PricePrediction(direction="up", magnitude_pct=0.0,
                                   confidence=0.0, timestamp=now)
            self._record_prediction(pred, 0.0)
            return pred

        current_close = float(bars[-1, 3])

        # Update previous prediction's actual return
        if self._predictions and self._predictions[-1].actual_return is None:
            prev_close = self._predictions[-1].price_at_prediction
            if prev_close > 0:
                self._predictions[-1].actual_return = (current_close - prev_close) / prev_close

        # Try torch prediction first
        if self._model is not None and _TORCH_AVAILABLE and len(bars) >= self._seq_len:
            pred = self._predict_torch(bars, now)
        else:
            pred = self._predict_momentum(bars, now)

        self._record_prediction(pred, current_close)
        self._last_close = current_close
        return pred

    def _predict_torch(self, bars: np.ndarray, now: float) -> PricePrediction:
        """Use trained Transformer for prediction."""
        window = bars[-self._seq_len:]
        normed = (window - self._norm_means) / self._norm_stds
        x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = self._model(x)
        direction_logit = float(out[0, 0])
        magnitude_raw = float(out[0, 1])

        prob_up = 1.0 / (1.0 + math.exp(-direction_logit))
        direction = "up" if prob_up >= 0.5 else "down"
        confidence = abs(prob_up - 0.5) * 2.0   # 0-1 scale
        magnitude_pct = abs(magnitude_raw) * 100.0

        return PricePrediction(
            direction=direction,
            magnitude_pct=round(magnitude_pct, 4),
            confidence=round(min(confidence, 1.0), 4),
            timestamp=now,
        )

    def _predict_momentum(self, bars: np.ndarray, now: float) -> PricePrediction:
        """Momentum fallback using exponential-weighted trend."""
        lookback = min(self._fallback_lookback, len(bars))
        closes = bars[-lookback:, 3]

        if len(closes) < 2:
            return PricePrediction(direction="up", magnitude_pct=0.0,
                                   confidence=0.0, timestamp=now)

        # Exponential weighted returns
        returns = np.diff(closes) / (closes[:-1] + 1e-12)
        weights = np.array([self._ema_alpha * (1 - self._ema_alpha) ** i
                            for i in range(len(returns) - 1, -1, -1)])
        weights /= weights.sum() + 1e-12
        ema_return = float(np.dot(weights, returns))

        direction = "up" if ema_return >= 0 else "down"
        magnitude_pct = abs(ema_return) * 100.0

        # Confidence based on consistency of returns
        if len(returns) > 1:
            sign_agreement = np.mean(np.sign(returns) == np.sign(ema_return))
            confidence = float(sign_agreement) * min(1.0, abs(ema_return) / (self._std_return + 1e-12))
        else:
            confidence = 0.1

        return PricePrediction(
            direction=direction,
            magnitude_pct=round(magnitude_pct, 4),
            confidence=round(min(confidence, 1.0), 4),
            timestamp=now,
        )

    def _record_prediction(self, pred: PricePrediction, price: float) -> None:
        """Store prediction for later calibration."""
        self._predictions.append(_PredictionRecord(
            prediction=pred,
            price_at_prediction=price,
        ))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def get_accuracy(self, lookback: int = 100) -> float:
        """
        Fraction of recent predictions where predicted direction matched
        the actual next-bar direction.

        Parameters
        ----------
        lookback : int
            Number of most-recent predictions to evaluate.

        Returns
        -------
        float
            Accuracy in [0.0, 1.0].  Returns 0.0 if no resolved predictions.
        """
        resolved = [r for r in list(self._predictions)[-lookback:]
                    if r.actual_return is not None]
        if not resolved:
            return 0.0

        correct = 0
        for rec in resolved:
            actual_dir = "up" if rec.actual_return >= 0 else "down"
            if rec.prediction.direction == actual_dir:
                correct += 1

        accuracy = correct / len(resolved)
        logger.debug("get_accuracy: %d/%d = %.2f%% (lookback=%d)",
                     correct, len(resolved), accuracy * 100, lookback)
        return accuracy

    @property
    def prediction_count(self) -> int:
        """Total predictions stored."""
        return len(self._predictions)

    @property
    def is_trained(self) -> bool:
        """Whether fit() has been called successfully."""
        return self._trained
