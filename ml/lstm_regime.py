"""
LSTM/GRU regime or next-bar boost. Numpy-based recurrent forward (GRU-like) plus
optional PyTorch LSTM when available (set LSTM_REGIME_MODEL_PATH to .pt/.pth file).
Re-exports regime_boost for compatibility.
"""

from __future__ import annotations
import logging

import logging

import os
from typing import List, Optional, Sequence

import numpy as np

from ml.regime_boost import apply_regime_boost as _apply_regime_boost
from ml.regime_boost import regime_boost_from_closes as _regime_boost_from_closes

logger = logging.getLogger(__name__)

# Optional PyTorch LSTM: when LSTM_REGIME_MODEL_PATH points to a .pt/.pth file, use it for forward.
_torch_lstm_model: Optional[object] = None


def _load_torch_lstm_if_available() -> Optional[object]:
    global _torch_lstm_model
    if _torch_lstm_model is not None:
        return _torch_lstm_model
    path = os.environ.get("LSTM_REGIME_MODEL_PATH", "").strip()
    if not path or not os.path.isfile(path):
        return None
    try:
        import torch  # noqa: F401
        _torch_lstm_model = torch.jit.load(path)
        _torch_lstm_model.eval()
        return _torch_lstm_model
    except Exception:
        return None


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-max(-20, min(20, x)))))


def _tanh(x: float) -> float:
    return float(np.tanh(max(-20, min(20, x))))


def _gru_forward_numpy(closes: Sequence[float], lookback: int = 20) -> float:
    """
    Minimal GRU-like forward over returns: single hidden state h, update gate z,
    reset gate r. No training; fixed weights to smooth returns and emphasize trend.
    Returns scalar in [0.5, 1.0] for use as regime boost.
    """
    if not closes or len(closes) < max(5, lookback):
        return 0.75
    arr = np.array(closes[-lookback:], dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
        return 0.75
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    if len(rets) < 2:
        return 0.75
    h = 0.0
    # Fixed small weights: z gate = sigmoid(0.3 * r + 0.2), so ~0.55 update
    wz, wr = 0.3, 0.25
    for r in rets:
        z = _sigmoid(wz * r * 100 + 0.2)
        r_gate = _sigmoid(wr * r * 100)
        h_new = (1 - z) * h + z * _tanh(r * 50)
        h = h_new
    # Map h in [-1,1] to boost in [0.5, 1.0]
    boost = 0.5 + 0.5 * (h + 1.0) / 2.0
    return float(np.clip(boost, 0.5, 1.0))


def regime_boost_from_closes(closes: Sequence[float], lookback: int = 20) -> float:
    """
    Regime/next-bar boost from recent closes. Uses GRU-like numpy forward when
    lookback >= 5; else re-exports regime_boost baseline.
    """
    if closes and len(closes) >= max(5, lookback):
        return _gru_forward_numpy(closes, lookback=lookback)
    return float(_regime_boost_from_closes(closes, lookback=lookback))


def apply_regime_boost(
    confidence: float,
    closes: Optional[Sequence[float]] = None,
    lookback: int = 20,
) -> float:
    """Scale confidence by regime boost. Same as ml.regime_boost.apply_regime_boost."""
    return _apply_regime_boost(confidence, closes, lookback=lookback)


def lstm_regime_forward(closes: List[float], lookback: int = 20) -> float:
    """
    Regime/next-bar forward. When LSTM_REGIME_MODEL_PATH is set and points to a
    TorchScript .pt/.pth file, uses that model (input: [1, lookback, 1], output: scalar).
    Otherwise uses numpy GRU-like recurrence. Returns scalar in [0.5, 1.0].
    """
    model = _load_torch_lstm_if_available()
    if model is not None and closes and len(closes) >= lookback:
        try:
            import torch
            arr = np.array(closes[-lookback:], dtype=np.float32)
            if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
                return _gru_forward_numpy(closes, lookback=lookback)
            x = torch.from_numpy(arr).reshape(1, lookback, 1)
            with torch.no_grad():
                out = model(x)
            if isinstance(out, torch.Tensor):
                out = out.cpu().numpy()
            if hasattr(out, "item"):
                boost = float(out.item())
            else:
                boost = float(out) if isinstance(out, (int, float)) else float(out.flat[0])
            return float(np.clip(0.5 + 0.5 * (1.0 / (1.0 + np.exp(-boost))), 0.5, 1.0))
        except Exception as _e:
            logger.debug("lstm_regime error: %s", _e)
    return _gru_forward_numpy(closes, lookback=lookback)
