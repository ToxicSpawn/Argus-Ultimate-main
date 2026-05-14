"""
deep_lob.py — DeepLOB microstructure ML signal.

Based on: Zhang et al. (2019) "DeepLOB: Deep Learning for Limit Order Books"
https://arxiv.org/abs/1901.04716

Architecture
------------
    Conv1D(1→32, k=2) → LeakyReLU → Conv1D(32→32, k=2) → MaxPool(2)
    → LSTM(32→64, batch_first) → FC(64→3)

Output: 3 classes — UP (0), FLAT (1), DOWN (2)  over a 5-tick horizon.

Usage
-----
    features = DeepLOBFeatures()
    feat = features.extract(bids, asks, n_levels=10)

    model = DeepLOBModel()
    prediction = model.predict(feat)
    # {'direction': 'UP', 'probability': 0.71, 'confidence': 0.71}

    predictor = OnlineLOBPredictor()
    signal = predictor.predict({'bids': bids, 'asks': asks})
    predictor.update(feat, outcome=0)      # outcome: 0=UP, 1=FLAT, 2=DOWN
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional PyTorch import — graceful fallback if not installed
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.info("deep_lob: torch not available — model disabled, OBI fallback active")

# Class labels
DIRECTION_LABELS = {0: "UP", 1: "FLAT", 2: "DOWN"}


# ─── DeepLOBFeatures ─────────────────────────────────────────────────────────

class DeepLOBFeatures:
    """Extract normalised LOB features compatible with the DeepLOB architecture.

    Output shape: (40,)  =  10 bid prices + 10 ask prices
                           + 10 bid sizes  + 10 ask sizes
    All prices are normalised by mid price; all sizes by total visible volume.
    """

    def extract(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        n_levels: int = 10,
    ) -> Optional[np.ndarray]:
        """
        Parameters
        ----------
        bids : list of (price, size) sorted descending (best bid first)
        asks : list of (price, size) sorted ascending  (best ask first)
        n_levels : number of price levels to include (default 10)

        Returns
        -------
        np.ndarray of shape (4 * n_levels,) or None if data insufficient.
        """
        if len(bids) < 1 or len(asks) < 1:
            return None

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
            return None

        mid = (best_bid + best_ask) / 2.0

        # Pad / truncate to n_levels
        def _pad(levels: list, n: int) -> List[Tuple[float, float]]:
            levels = list(levels[:n])
            while len(levels) < n:
                levels.append((0.0, 0.0))
            return levels

        bids_p = _pad(bids, n_levels)
        asks_p = _pad(asks, n_levels)

        bid_prices = np.array([p for p, _ in bids_p], dtype=np.float32)
        ask_prices = np.array([p for p, _ in asks_p], dtype=np.float32)
        bid_sizes  = np.array([s for _, s in bids_p], dtype=np.float32)
        ask_sizes  = np.array([s for _, s in asks_p], dtype=np.float32)

        # Normalise prices by mid
        bid_prices = (bid_prices - mid) / mid
        ask_prices = (ask_prices - mid) / mid

        # Normalise sizes by total visible volume (avoid div-by-zero)
        total_vol = bid_sizes.sum() + ask_sizes.sum()
        if total_vol > 0.0:
            bid_sizes = bid_sizes / total_vol
            ask_sizes = ask_sizes / total_vol

        features = np.concatenate([bid_prices, ask_prices, bid_sizes, ask_sizes])
        return features.astype(np.float32)


# ─── DeepLOBNet (PyTorch module) ─────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _DeepLOBNet(nn.Module):
        """Conv1D → LSTM → FC architecture for mid-price direction prediction."""

        def __init__(self, input_dim: int = 40, n_classes: int = 3) -> None:
            super().__init__()
            # Convolutional feature extraction
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2)
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2)
            self.pool  = nn.MaxPool1d(kernel_size=2)
            self.leaky = nn.LeakyReLU(negative_slope=0.01)

            # Compute flattened size after conv+pool
            with torch.no_grad():
                dummy = torch.zeros(1, 1, input_dim)
                x = self.pool(self.leaky(self.conv2(self.leaky(self.conv1(dummy)))))
                # x shape: (1, 32, L) → transpose to (1, L, 32) for LSTM
                self._lstm_input = x.shape[2]   # sequence length
                self._lstm_features = x.shape[1] # 32

            # Temporal modelling
            self.lstm = nn.LSTM(
                input_size=self._lstm_features,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
            )
            # Classification head
            self.fc = nn.Linear(64, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : Tensor shape (B, 40)

            Returns
            -------
            logits : Tensor shape (B, 3)
            """
            # Add channel dim: (B, 1, 40)
            x = x.unsqueeze(1)
            x = self.leaky(self.conv1(x))      # (B, 32, 39)
            x = self.leaky(self.conv2(x))      # (B, 32, 38)
            x = self.pool(x)                   # (B, 32, 19)
            # Transpose for LSTM: (B, seq_len, features)
            x = x.transpose(1, 2)             # (B, 19, 32)
            _, (h, _) = self.lstm(x)          # h: (1, B, 64)
            h = h.squeeze(0)                  # (B, 64)
            return self.fc(h)                  # (B, 3)


# ─── DeepLOBModel ─────────────────────────────────────────────────────────────

class DeepLOBModel:
    """Wraps _DeepLOBNet for prediction, training, and persistence.

    Falls back gracefully when torch is not available.
    """

    def __init__(self, input_dim: int = 40, lr: float = 1e-3) -> None:
        self._input_dim = input_dim
        self._trained = False

        if _TORCH_AVAILABLE:
            self._net = _DeepLOBNet(input_dim=input_dim)
            self._opt = torch.optim.Adam(self._net.parameters(), lr=lr)
            self._criterion = nn.CrossEntropyLoss()
        else:
            self._net = None
            self._opt = None
            self._criterion = None

    def predict(self, features: np.ndarray) -> Dict:
        """Return direction prediction for a single feature vector.

        Parameters
        ----------
        features : np.ndarray shape (input_dim,)

        Returns
        -------
        dict: direction ('UP'|'FLAT'|'DOWN'), probability (float), confidence (float)
        """
        if not _TORCH_AVAILABLE or self._net is None:
            return {"direction": "FLAT", "probability": 0.333, "confidence": 0.0,
                    "source": "fallback_no_torch"}

        if features is None or features.shape[0] != self._input_dim:
            return {"direction": "FLAT", "probability": 0.333, "confidence": 0.0,
                    "source": "fallback_bad_features"}

        self._net.eval()
        with torch.no_grad():
            t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, D)
            logits = self._net(t)                                          # (1, 3)
            probs  = F.softmax(logits, dim=-1).squeeze(0).numpy()         # (3,)

        cls_idx = int(np.argmax(probs))
        return {
            "direction":   DIRECTION_LABELS[cls_idx],
            "class_index": cls_idx,
            "probability": float(probs[cls_idx]),
            "confidence":  float(probs[cls_idx]),
            "probs":       probs.tolist(),
            "source":      "deeplob_model" if self._trained else "deeplob_untrained",
        }

    def train_step(
        self,
        features_batch: np.ndarray,
        labels_batch: np.ndarray,
    ) -> float:
        """One mini-batch gradient step.

        Parameters
        ----------
        features_batch : shape (B, input_dim)
        labels_batch   : shape (B,) of ints in {0, 1, 2}

        Returns
        -------
        float: scalar cross-entropy loss
        """
        if not _TORCH_AVAILABLE or self._net is None:
            return float("nan")

        self._net.train()
        x = torch.tensor(features_batch, dtype=torch.float32)
        y = torch.tensor(labels_batch, dtype=torch.long)

        self._opt.zero_grad()
        logits = self._net(x)
        loss = self._criterion(logits, y)
        loss.backward()
        self._opt.step()

        self._trained = True
        return float(loss.item())

    def save(self, path: str) -> None:
        """Persist model weights to *path* (PyTorch .pt file)."""
        if not _TORCH_AVAILABLE or self._net is None:
            logger.warning("deep_lob: torch unavailable, cannot save model")
            return
        torch.save({
            "state_dict": self._net.state_dict(),
            "input_dim":  self._input_dim,
            "trained":    self._trained,
        }, path)
        logger.info("deep_lob: model saved to %s", path)

    def load(self, path: str) -> bool:
        """Load model weights from *path*. Returns True on success."""
        if not _TORCH_AVAILABLE or self._net is None:
            logger.warning("deep_lob: torch unavailable, cannot load model")
            return False
        if not os.path.exists(path):
            logger.warning("deep_lob: model file not found: %s", path)
            return False
        try:
            ckpt = torch.load(path, map_location="cpu")
            self._net.load_state_dict(ckpt["state_dict"])
            self._trained = ckpt.get("trained", True)
            logger.info("deep_lob: model loaded from %s", path)
            return True
        except Exception as exc:
            logger.warning("deep_lob: failed to load model: %s", exc)
            return False


# ─── OnlineLOBPredictor ───────────────────────────────────────────────────────

class OnlineLOBPredictor:
    """Online-learning wrapper around DeepLOBModel.

    Maintains a ring-buffer of (features, outcome) pairs and triggers a
    mini-batch retrain every ``retrain_every`` samples.  Falls back to a
    simple OBI-threshold signal when the model has not been trained yet.
    """

    OBI_THRESHOLD = 0.15      # OBI > threshold → UP, < -threshold → DOWN

    def __init__(
        self,
        buffer_size: int = 1000,
        retrain_every: int = 100,
        mini_batch: int = 32,
        model_path: Optional[str] = None,
    ) -> None:
        self._feature_extractor = DeepLOBFeatures()
        self._model = DeepLOBModel()
        self._buffer: deque = deque(maxlen=buffer_size)
        self._retrain_every = retrain_every
        self._mini_batch = mini_batch
        self._sample_count = 0

        # Attempt to load a pre-trained model
        if model_path and os.path.exists(model_path):
            self._model.load(model_path)

    # ── OBI-based fallback ───────────────────────────────────────────────────

    @staticmethod
    def _obi_signal(bids, asks, levels: int = 5) -> Dict:
        """Simple order book imbalance threshold signal."""
        bid_qty = sum(s for _, s in list(bids)[:levels])
        ask_qty = sum(s for _, s in list(asks)[:levels])
        total = bid_qty + ask_qty
        if total == 0.0:
            obi = 0.0
        else:
            obi = (bid_qty - ask_qty) / total

        if obi > OnlineLOBPredictor.OBI_THRESHOLD:
            direction = "UP"
        elif obi < -OnlineLOBPredictor.OBI_THRESHOLD:
            direction = "DOWN"
        else:
            direction = "FLAT"

        confidence = min(abs(obi) / 0.5, 1.0)
        return {
            "direction": direction,
            "probability": confidence,
            "confidence": confidence,
            "obi": obi,
            "source": "obi_fallback",
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, book_snapshot: Dict) -> Dict:
        """Generate a trading signal from a book snapshot.

        Parameters
        ----------
        book_snapshot : dict with keys 'bids' and 'asks', each a list of
                        (price, size) tuples or [price, size] lists.

        Returns
        -------
        dict: direction, probability, confidence, source, timestamp_ms
        """
        bids_raw = book_snapshot.get("bids", [])
        asks_raw = book_snapshot.get("asks", [])

        # Normalise to (price, size) tuples
        def _to_pairs(levels):
            result = []
            for lv in levels:
                if isinstance(lv, (list, tuple)) and len(lv) >= 2:
                    result.append((float(lv[0]), float(lv[1])))
                elif isinstance(lv, dict):
                    result.append((float(lv.get("price", 0)), float(lv.get("size", 0))))
            return result

        bids = _to_pairs(bids_raw)
        asks = _to_pairs(asks_raw)

        features = self._feature_extractor.extract(bids, asks)

        if features is None or not self._model._trained:
            # Fallback to OBI threshold
            signal = self._obi_signal(bids, asks)
            signal["timestamp_ms"] = int(time.time() * 1000)
            signal["features_available"] = features is not None
            return signal

        signal = self._model.predict(features)
        signal["timestamp_ms"] = int(time.time() * 1000)
        return signal

    def update(self, features: np.ndarray, outcome: int) -> None:
        """Record a labelled observation and optionally retrain.

        Parameters
        ----------
        features : np.ndarray shape (40,) — feature vector from DeepLOBFeatures
        outcome  : int in {0=UP, 1=FLAT, 2=DOWN}
        """
        if features is None:
            return
        self._buffer.append((features.copy(), int(outcome)))
        self._sample_count += 1

        if (
            _TORCH_AVAILABLE
            and len(self._buffer) >= self._mini_batch
            and self._sample_count % self._retrain_every == 0
        ):
            self._mini_batch_train()

    def _mini_batch_train(self) -> None:
        """Sample a mini-batch from the buffer and do one gradient step."""
        buf_list = list(self._buffer)
        # Random mini-batch
        idx = np.random.choice(len(buf_list), size=min(self._mini_batch, len(buf_list)),
                               replace=False)
        feats = np.stack([buf_list[i][0] for i in idx])
        labels = np.array([buf_list[i][1] for i in idx], dtype=np.int64)
        loss = self._model.train_step(feats, labels)
        logger.debug("OnlineLOBPredictor: mini-batch train loss=%.4f (n=%d)", loss, len(idx))

    def save_model(self, path: str) -> None:
        self._model.save(path)

    def load_model(self, path: str) -> bool:
        return self._model.load(path)
