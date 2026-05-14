"""DeepLOBLiveBridge — Push 39.

Lightweight LOB-feature inference bridge for the DEEPLOB SignalGateway
source. Consumes raw order book snapshots, extracts a 40-feature vector
(10 bid/ask price + volume levels), and runs inference via:

  1. A loaded PyTorch model (models/deeplob_weights.pt) if available.
  2. A fallback numpy MLP (random init on first run, useful for
     integration testing and graceful degradation in production).

The bridge is intentionally stateless between bar calls — each call to
get_signal() uses the most recent book snapshot stored via update_book().

LOB feature layout (40 features)
---------------------------------
  [bid_p0..bid_p9]  10 best bid prices   (normalised by mid)
  [ask_p0..ask_p9]  10 best ask prices   (normalised by mid)
  [bid_v0..bid_v9]  10 best bid volumes  (log-scaled, clipped)
  [ask_v0..ask_v9]  10 best ask volumes  (log-scaled, clipped)

Output
------
  get_signal() → {"direction": str, "confidence": float, "logits": list}
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "..", "models", "deeplob_weights.pt")
_N_LEVELS      = 10
_N_FEATURES    = 4 * _N_LEVELS  # 40
_N_HIDDEN      = 64
_N_CLASSES     = 3              # 0=short, 1=flat, 2=long
_CONF_THRESHOLD = 0.45          # below this -> flat


class DeepLOBLiveBridge:
    """Real-time LOB inference bridge.

    Parameters
    ----------
    model_path     : Path to PyTorch weights file (optional).
    n_levels       : Number of LOB price levels to use (default 10).
    conf_threshold : Minimum softmax confidence to emit non-flat signal.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_levels: int = _N_LEVELS,
        conf_threshold: float = _CONF_THRESHOLD,
    ) -> None:
        self._n_levels      = n_levels
        self._n_features    = 4 * n_levels
        self._conf_threshold = conf_threshold
        self._book: Optional[Dict] = None      # latest order book snapshot
        self._last_mid: float = 1.0

        # Try to load PyTorch model
        self._torch_model = None
        self._np_weights  = None
        path = model_path or _MODEL_PATH
        if os.path.exists(path):
            self._load_torch(path)
        else:
            logger.info(
                "DeepLOBLiveBridge: no weights at %s — using numpy fallback MLP", path
            )
            self._init_numpy_mlp()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_book(self, book: Dict) -> None:
        """Feed a new order book snapshot.

        Expected format::

            {
                "bids": [[price, volume], ...],  # best first
                "asks": [[price, volume], ...],
            }
        """
        self._book = book
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        if bids and asks:
            mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
            if mid > 0:
                self._last_mid = mid

    def get_signal(self) -> Optional[Dict]:
        """Run inference on the latest book snapshot.

        Returns
        -------
        {"direction": str, "confidence": float, "logits": list}
        or None if no book snapshot available.
        """
        if self._book is None:
            return None
        try:
            features = self._extract_features(self._book)
            logits   = self._infer(features)
            probs    = self._softmax(logits)
            cls      = int(np.argmax(probs))
            conf     = float(probs[cls])

            if conf < self._conf_threshold:
                direction = "flat"
            else:
                direction = ["short", "flat", "long"][cls]

            return {
                "direction":  direction,
                "confidence": conf,
                "logits":     logits.tolist(),
            }
        except Exception as exc:
            logger.debug("DeepLOBLiveBridge.get_signal() error: %s", exc)
            return None

    def get_features(self) -> Optional[np.ndarray]:
        """Return the 40-feature vector for the latest book (for logging/debug)."""
        if self._book is None:
            return None
        return self._extract_features(self._book)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, book: Dict) -> np.ndarray:
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        mid  = self._last_mid

        def _pad_levels(levels, n):
            padded = list(levels)[:n]
            while len(padded) < n:
                padded.append([mid, 0.0])
            return padded

        bids_p = _pad_levels(bids, self._n_levels)
        asks_p = _pad_levels(asks, self._n_levels)

        bid_prices = np.array([float(b[0]) for b in bids_p]) / mid - 1.0
        ask_prices = np.array([float(a[0]) for a in asks_p]) / mid - 1.0
        bid_vols   = np.log1p(np.clip([float(b[1]) for b in bids_p], 0, None))
        ask_vols   = np.log1p(np.clip([float(a[1]) for a in asks_p], 0, None))

        return np.concatenate([bid_prices, ask_prices, bid_vols, ask_vols]).astype(np.float32)

    # ------------------------------------------------------------------
    # Inference backends
    # ------------------------------------------------------------------

    def _infer(self, features: np.ndarray) -> np.ndarray:
        if self._torch_model is not None:
            return self._infer_torch(features)
        return self._infer_numpy(features)

    def _infer_torch(self, features: np.ndarray) -> np.ndarray:
        try:
            import torch
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self._torch_model(x).squeeze(0).numpy()
            return logits.astype(np.float32)
        except Exception as exc:
            logger.debug("Torch inference failed (%s) — falling back to numpy", exc)
            return self._infer_numpy(features)

    def _infer_numpy(self, features: np.ndarray) -> np.ndarray:
        """2-layer MLP: features → hidden → 3 logits."""
        assert self._np_weights is not None
        W1, b1, W2, b2 = self._np_weights
        h = np.tanh(features @ W1 + b1)
        return (h @ W2 + b2).astype(np.float32)

    # ------------------------------------------------------------------
    # Model loading / init
    # ------------------------------------------------------------------

    def _load_torch(self, path: str) -> None:
        try:
            import torch
            self._torch_model = torch.jit.load(path, map_location="cpu")
            self._torch_model.eval()
            logger.info("DeepLOBLiveBridge: loaded TorchScript model from %s", path)
        except Exception as exc:
            logger.warning(
                "DeepLOBLiveBridge: could not load torch model (%s) — using numpy MLP", exc
            )
            self._init_numpy_mlp()

    def _init_numpy_mlp(self) -> None:
        """Initialise a small random-weight numpy MLP (glorot uniform)."""
        rng = np.random.default_rng(seed=42)
        n_in, n_h, n_out = self._n_features, _N_HIDDEN, _N_CLASSES

        def _glorot(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)

        self._np_weights = (
            _glorot(n_in, n_h),
            np.zeros(n_h, dtype=np.float32),
            _glorot(n_h, n_out),
            np.zeros(n_out, dtype=np.float32),
        )
        logger.debug("DeepLOBLiveBridge: numpy MLP initialised (seed=42)")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
