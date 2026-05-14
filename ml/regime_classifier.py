"""
XGBoost / GBM Regime Classifier — supervised regime labelling.

Classifies market regime from features:
  - Rolling returns (1h, 4h, 24h, 7d)
  - Rolling volatility (1h, 24h, 7d)
  - Trend strength (ADX proxy)
  - Volume ratio

Output labels: TREND_UP / TREND_DOWN / RANGING / VOLATILE / CRISIS

Falls back to a rule-based classifier when XGBoost is unavailable or
insufficient training data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb  # type: ignore
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.debug("xgboost not available; using rule-based regime classifier")

# Detect GPU availability once at import time
_GPU_DEVICE: Optional[str] = None
try:
    if _XGB_AVAILABLE:
        import subprocess, sys as _sys
        _r = subprocess.run(
            [_sys.executable, "-c", "import xgboost as xgb; xgb.train({'tree_method':'hist','device':'cuda','verbosity':0}, xgb.DMatrix([[1],[2]], label=[0,1]), num_boost_round=1)"],
            capture_output=True, timeout=10
        )
        if _r.returncode == 0:
            _GPU_DEVICE = "cuda"
            logger.info("RegimeClassifier: CUDA GPU detected — using gpu_hist tree method")
except Exception as _e:
    logger.debug("regime_classifier error: %s", _e)

REGIME_LABELS = ["TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE", "CRISIS"]
LABEL_TO_INT = {r: i for i, r in enumerate(REGIME_LABELS)}
INT_TO_LABEL = {i: r for i, r in enumerate(REGIME_LABELS)}

BARS_HOUR = 12
BARS_4H = 48
BARS_DAY = 288
BARS_WEEK = 2016


@dataclass
class RegimePrediction:
    regime: str
    probabilities: Dict[str, float]
    confidence: float
    method: str  # xgboost | rules


def _returns(prices: np.ndarray, bars: int) -> float:
    if len(prices) < bars + 1:
        return 0.0
    return float(math.log(prices[-1] / prices[-bars - 1]))


def _vol(prices: np.ndarray, bars: int) -> float:
    if len(prices) < bars + 1:
        return 0.0
    r = np.diff(np.log(prices[-bars - 1:]))
    return float(np.std(r) * math.sqrt(BARS_DAY * 252))  # annualised


def _adx_proxy(prices: np.ndarray, period: int = 14) -> float:
    """Simplified directional movement proxy (abs mean return / vol)."""
    if len(prices) < period + 1:
        return 0.0
    r = np.diff(np.log(prices[-period - 1:]))
    sigma = float(np.std(r))
    if sigma < 1e-10:
        return 0.0
    return abs(float(np.mean(r))) / sigma


def _build_features(prices: np.ndarray) -> Optional[np.ndarray]:
    if len(prices) < BARS_WEEK + 2:
        return None
    return np.array([
        _returns(prices, BARS_HOUR),
        _returns(prices, BARS_4H),
        _returns(prices, BARS_DAY),
        _returns(prices, BARS_WEEK),
        _vol(prices, BARS_HOUR),
        _vol(prices, BARS_DAY),
        _vol(prices, BARS_WEEK),
        _adx_proxy(prices, 14),
        _adx_proxy(prices, 50),
    ], dtype=np.float32)


def build_quantum_regime_features(
    prices: np.ndarray,
    *,
    use_vqe: bool = False,
    n_qubits: int = 3,
) -> Optional[np.ndarray]:
    """
    Phase S/G3: build quantum-augmented regime features.

    Computes the standard 9 features plus 3 quantum-derived features:
        [ground_energy, eigenmode_norm, vqe_iters]

    Where the quantum features come from running VQE on a small Pauli
    Hamiltonian whose coefficients are derived from the recent return moments.

    Returns the augmented (12,)-shape feature vector.
    """
    base = _build_features(prices)
    if base is None:
        return None

    if not use_vqe or len(prices) < 30:
        return base

    try:
        from quantum.algorithms.vqe import VQESolver

        recent = prices[-30:]
        rets = np.diff(np.log(recent + 1e-10))
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets)) or 1e-6
        skew = float(np.mean((rets - mean_r) ** 3) / max(std_r ** 3, 1e-9))

        # Build a 2-qubit Pauli Hamiltonian
        pauli_terms = [
            ("ZI", -mean_r * 50.0),
            ("IZ", -std_r * 30.0),
            ("ZZ", skew * 5.0),
        ]
        solver = VQESolver(n_qubits=2, n_layers=2)
        result = solver.solve_hamiltonian(pauli_terms, max_iter=15, shots=256, n_restarts=1)

        quantum_features = np.array([
            float(result.get("ground_energy", 0.0)),
            float(np.linalg.norm(result.get("optimal_params", [0.0]))),
            float(result.get("n_iterations", 0)) / 100.0,
        ], dtype=np.float32)

        return np.concatenate([base, quantum_features])
    except Exception as exc:
        # Quantum failed - return base features only
        return base


FEATURE_NAMES = [
    "ret_1h", "ret_4h", "ret_1d", "ret_7d",
    "vol_1h", "vol_1d", "vol_7d",
    "adx_14", "adx_50",
]


def _rule_based_regime(feats: np.ndarray) -> RegimePrediction:
    """Heuristic regime classification when XGBoost is unavailable."""
    ret_1d = feats[2]
    ret_7d = feats[3]
    vol_1d = feats[5]
    adx = feats[7]

    probs = {r: 0.0 for r in REGIME_LABELS}

    # Crisis: extreme drawdown (>15% 7d) + high vol
    if ret_7d < -0.15 and vol_1d > 1.0:
        probs["CRISIS"] = 0.80
        probs["VOLATILE"] = 0.20
    # Volatile: high vol, no clear direction
    elif vol_1d > 0.80:
        probs["VOLATILE"] = 0.70
        probs["CRISIS"] = 0.30
    # Trend up: positive momentum + trending
    elif ret_1d > 0.01 and adx > 0.3:
        probs["TREND_UP"] = 0.75
        probs["RANGING"] = 0.25
    # Trend down: negative momentum + trending
    elif ret_1d < -0.01 and adx > 0.3:
        probs["TREND_DOWN"] = 0.75
        probs["RANGING"] = 0.25
    # Ranging: low vol, low ADX
    else:
        probs["RANGING"] = 0.80
        probs["TREND_UP"] = 0.10
        probs["TREND_DOWN"] = 0.10

    regime = max(probs, key=lambda k: probs[k])
    return RegimePrediction(
        regime=regime,
        probabilities=probs,
        confidence=max(probs.values()),
        method="rules",
    )


class RegimeClassifier:
    """
    Trains on labelled (features, regime) pairs and predicts current regime.

    Usage::

        clf = RegimeClassifier()
        # Feed labelled data for training
        clf.add_training_sample(price_array, "TREND_UP")
        clf.train()
        # Predict current regime
        prediction = clf.predict(recent_prices)
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 4, use_gpu: bool = True) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._use_gpu = use_gpu
        self._model: Optional[Any] = None
        self._X: List[np.ndarray] = []
        self._y: List[int] = []
        self._trained = False

    def add_training_sample(
        self, prices: Sequence[float], regime_label: str
    ) -> bool:
        """Add one labelled sample. Returns False if insufficient price data."""
        arr = np.array(prices, dtype=np.float64)
        feats = _build_features(arr)
        if feats is None:
            return False
        if regime_label not in LABEL_TO_INT:
            raise ValueError(f"Unknown regime: {regime_label}. Valid: {REGIME_LABELS}")
        self._X.append(feats)
        self._y.append(LABEL_TO_INT[regime_label])
        return True

    def train(self) -> bool:
        """Train classifier. Returns True if successful."""
        if len(self._X) < 20:
            logger.warning("Need at least 20 labelled samples to train; have %d", len(self._X))
            return False
        if not _XGB_AVAILABLE:
            logger.info("XGBoost not available; will use rule-based classifier")
            return False
        X = np.vstack(self._X)
        y = np.array(self._y, dtype=int)
        # Remap labels to contiguous 0..N-1 if not all classes present
        unique_classes = sorted(set(y))
        if unique_classes != list(range(len(REGIME_LABELS))):
            logger.info("Remapping %d classes to contiguous range: %s", len(unique_classes), unique_classes)
            remap = {old: new for new, old in enumerate(unique_classes)}
            y = np.array([remap[v] for v in y], dtype=int)
            self._class_remap = {new: old for old, new in remap.items()}
        else:
            self._class_remap = None
        try:
            use_cuda = self._use_gpu and _GPU_DEVICE == "cuda"
            n_classes = len(unique_classes)
            xgb_kwargs: Dict[str, Any] = dict(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                eval_metric="mlogloss",
                num_class=n_classes if n_classes > 2 else None,
                verbosity=0,
            )
            # Remove None values
            xgb_kwargs = {k: v for k, v in xgb_kwargs.items() if v is not None}
            if use_cuda:
                xgb_kwargs["tree_method"] = "hist"
                xgb_kwargs["device"] = "cuda"
                logger.info("RegimeClassifier: training on GPU (cuda)")
            else:
                xgb_kwargs["n_jobs"] = -1  # all CPU cores
            self._model = xgb.XGBClassifier(**xgb_kwargs)
            self._model.fit(X, y)
            self._trained = True
            logger.info(
                "RegimeClassifier trained on %d samples (%s)",
                len(self._X),
                "GPU" if use_cuda else "CPU",
            )
            return True
        except Exception as exc:
            logger.error("RegimeClassifier training failed: %s", exc)
            return False

    def predict(self, prices: Sequence[float]) -> Optional[RegimePrediction]:
        """Predict regime for the most recent portion of a price series."""
        arr = np.array(prices, dtype=np.float64)
        feats = _build_features(arr)
        if feats is None:
            return None

        if self._trained and self._model is not None:
            try:
                proba = self._model.predict_proba(feats.reshape(1, -1))[0]
                idx = int(np.argmax(proba))
                probs = {INT_TO_LABEL[i]: float(p) for i, p in enumerate(proba)}
                return RegimePrediction(
                    regime=INT_TO_LABEL[idx],
                    probabilities=probs,
                    confidence=float(proba[idx]),
                    method="xgboost",
                )
            except Exception as exc:
                logger.warning("XGBoost predict failed: %s — falling back to rules", exc)

        return _rule_based_regime(feats)

    @property
    def n_training_samples(self) -> int:
        return len(self._X)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES.copy()

    def feature_importances(self) -> Optional[Dict[str, float]]:
        """Returns feature importances if XGBoost model trained."""
        if not self._trained or self._model is None:
            return None
        imp = self._model.feature_importances_
        return {name: float(v) for name, v in zip(FEATURE_NAMES, imp)}
