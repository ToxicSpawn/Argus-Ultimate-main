"""
MetaLearner Retraining Script  (Push 27)
=========================================
Retrains the Argus MetaLearner ensemble on FeaturePipeline output.

Walk-forward cross-validation is used to prevent lookahead bias.
The best model is saved to models/meta_learner.pkl.

Usage:
    python training/train_meta_learner.py --symbol XBT/USD --bars 5000
    python training/train_meta_learner.py --synthetic

Requires:
    pip install lightgbm scikit-learn
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Imports (graceful)
# ---------------------------------------------------------------------------
try:
    from ml.feature_pipeline import FeaturePipeline
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    logger.error("FeaturePipeline not found — cannot train MetaLearner without it.")

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    logger.warning("LightGBM not installed (pip install lightgbm) — will use RandomForest fallback")

try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipeline
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.error("scikit-learn not installed — pip install scikit-learn")


# ---------------------------------------------------------------------------
# Candle generation
# ---------------------------------------------------------------------------

def _synthetic_candles(n: int = 5000, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate synthetic BTC-like OHLCV candles."""
    rng   = np.random.default_rng(seed)
    price = 50_000.0
    candles: List[Dict] = []
    for i in range(n):
        ret  = rng.normal(0.0002, 0.004)
        o    = price
        h    = price * (1 + abs(rng.normal(0, 0.002)))
        l    = price * (1 - abs(rng.normal(0, 0.002)))
        c    = price * (1 + ret)
        c    = max(l, min(h, c))
        v    = rng.uniform(1.0, 50.0)
        candles.append({
            "timestamp": 1_700_000_000 + i * 60,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        })
        price = c
    return candles


def _fetch_live_candles(symbol: str, bars: int) -> Optional[List[Dict[str, Any]]]:
    """Fetch real OHLCV candles via CCXTAdapter if available."""
    try:
        from execution.ccxt_adapter import build_adapter_from_env
        adapter = build_adapter_from_env("kraken", dry_run=True)
        candles = adapter.fetch_ohlcv(symbol, timeframe="1m", limit=bars)
        logger.info("Fetched %d live candles for %s", len(candles), symbol)
        return candles
    except Exception as exc:
        logger.warning("Live candle fetch failed (%s) — using synthetic data", exc)
        return None


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _make_labels(candles: List[Dict], horizon: int = 5, threshold: float = 0.001) -> np.ndarray:
    """
    Binary label: 1 if close price rises > threshold over next `horizon` bars.
    Last `horizon` rows are set to 0 (no future data).
    """
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    labels = np.zeros(len(closes), dtype=np.int32)
    for i in range(len(closes) - horizon):
        fwd_ret = (closes[i + horizon] - closes[i]) / closes[i]
        labels[i] = 1 if fwd_ret > threshold else 0
    return labels


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------

def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Tuple[Any, float]:
    """
    Walk-forward cross-validation.
    Returns the model retrained on all data + mean OOS AUC.
    """
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required")

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    aucs   = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = _build_model()
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, proba)
        except ValueError:
            auc = 0.5
        aucs.append(auc)
        logger.info("Fold %d/%d  OOS AUC=%.4f", fold + 1, n_splits, auc)

    mean_auc = float(np.mean(aucs))
    logger.info("Mean OOS AUC = %.4f across %d folds", mean_auc, n_splits)

    # Final model trained on all data
    final_model = _build_model()
    final_model.fit(X, y)
    return final_model, mean_auc


def _build_model() -> Any:
    """Build the ensemble classifier (LightGBM + RF + LR stacked via VotingClassifier)."""
    estimators = []

    if _LGB_AVAILABLE:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
        estimators.append(("lgbm", lgb_clf))

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    estimators.append(("rf", rf))

    lr = SkPipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=0.1, max_iter=500, random_state=42)),
    ])
    estimators.append(("lr", lr))

    voting = VotingClassifier(estimators=estimators, voting="soft")
    return voting


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    symbol: str,
    bars: int,
    output_dir: str,
    synthetic: bool,
    horizon: int,
    threshold: float,
    n_splits: int,
) -> None:
    if not _PIPELINE_AVAILABLE or not _SKLEARN_AVAILABLE:
        logger.error("Missing dependencies — aborting MetaLearner training.")
        return

    # 1. Candles
    candles: Optional[List[Dict]] = None
    if not synthetic:
        candles = _fetch_live_candles(symbol, bars)
    if candles is None or len(candles) < 500:
        logger.info("Using synthetic candles (n=%d)", bars)
        candles = _synthetic_candles(bars)

    # 2. Build features
    fp     = FeaturePipeline(timeframes=["1m", "5m", "15m", "1h", "4h"])
    result = fp.build(candles)
    X_raw  = result.X
    names  = result.feature_names if hasattr(result, "feature_names") else None
    logger.info("Feature matrix: %s  (features=%d)", X_raw.shape, X_raw.shape[1])

    # 3. Labels (aligned to feature rows)
    labels = _make_labels(candles, horizon=horizon, threshold=threshold)
    # Trim to match feature rows (pipeline may drop leading NaN bars)
    n_feat = X_raw.shape[0]
    labels = labels[-n_feat:]

    # 4. Drop NaN rows
    mask   = ~np.isnan(X_raw).any(axis=1)
    X      = X_raw[mask].astype(np.float32)
    y      = labels[mask]
    logger.info("Clean rows: %d  (dropped %d NaN rows)", X.shape[0], (~mask).sum())

    if X.shape[0] < 200:
        logger.error("Not enough clean rows (%d) to train.", X.shape[0])
        return

    # 5. Walk-forward train
    model, mean_auc = walk_forward_train(X, y, n_splits=n_splits)

    # 6. Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "meta_learner.pkl")
    meta: Dict[str, Any] = {
        "model":          model,
        "obs_dim":        X.shape[1],
        "feature_names":  names,
        "mean_oos_auc":   mean_auc,
        "horizon_bars":   horizon,
        "threshold":      threshold,
        "symbol":         symbol,
        "n_train_rows":   X.shape[0],
    }
    with open(out_path, "wb") as fh:
        pickle.dump(meta, fh)
    logger.info("MetaLearner saved → %s", out_path)
    logger.info("AUC=%.4f | features=%d | rows=%d", mean_auc, X.shape[1], X.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Argus MetaLearner (Push 27)")
    parser.add_argument("--symbol",    default="XBT/USD")
    parser.add_argument("--bars",      type=int,   default=5000)
    parser.add_argument("--output",    default="models")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--horizon",   type=int,   default=5,     help="Forward bars for label")
    parser.add_argument("--threshold", type=float, default=0.001, help="Min return to label as 1")
    parser.add_argument("--splits",    type=int,   default=5,     help="Walk-forward folds")
    args = parser.parse_args()
    train(
        symbol=args.symbol,
        bars=args.bars,
        output_dir=args.output,
        synthetic=args.synthetic,
        horizon=args.horizon,
        threshold=args.threshold,
        n_splits=args.splits,
    )


if __name__ == "__main__":
    main()
