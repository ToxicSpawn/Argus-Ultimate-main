"""Tune Argus ML win-rate settings with walk-forward safe objectives.

This script focuses on the parameters that change the *decision layer* rather
than retraining every neural network: confidence thresholds, disagreement
limits, anomaly penalty, and model weights. It uses Optuna when available and a
deterministic random-search fallback otherwise.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.optuna_tuner import OptunaTuner
from ml.winrate_enhancement import SoftLabelGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_training_frame(symbol: str = "BTCUSDT", timeframe: str = "1h") -> pd.DataFrame:
    path = Path("data/historical/historical_data.pkl")
    if not path.exists():
        raise FileNotFoundError("Missing data/historical/historical_data.pkl; collect history first")
    with path.open("rb") as fh:
        data = pickle.load(fh)
    df = pd.DataFrame(data[symbol][timeframe])
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def build_dataset(df: pd.DataFrame, horizon: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi = 100.0 - (100.0 / (1.0 + gain / loss.clip(lower=1e-8)))
    price_pos = (close - low.rolling(24).min()) / (high.rolling(24).max() - low.rolling(24).min()).clip(lower=1e-8)
    volume_ratio = volume / volume.rolling(24).mean().clip(lower=1e-8)
    features = pd.DataFrame({
        "r1": close.pct_change(1),
        "r4": close.pct_change(4),
        "r12": close.pct_change(12),
        "r24": close.pct_change(24),
        "v12": close.pct_change(1).rolling(12).std(),
        "v24": close.pct_change(1).rolling(24).std(),
        "rsi": rsi,
        "pp": price_pos,
        "vr": volume_ratio,
    })
    returns = close.pct_change(horizon).shift(-horizon)
    labels = np.argmax(SoftLabelGenerator(horizon=horizon).transform_returns(returns.fillna(0.0).values), axis=1)
    dataset = features.replace([np.inf, -np.inf], np.nan).dropna()
    valid_idx = dataset.index[dataset.index < len(df) - horizon]
    dataset = dataset.loc[valid_idx]
    return dataset.values.astype(float), labels[valid_idx].astype(int)


def score_params(X: np.ndarray, y: np.ndarray, params: Dict[str, float]) -> float:
    splitter = TimeSeriesSplit(n_splits=4)
    scores = []
    trade_counts = []

    for train_idx, test_idx in splitter.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            subsample=float(params["subsample"]),
            random_state=42,
        )
        model.fit(X_train, y[train_idx])
        proba = model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        active = confidence >= float(params["min_confidence"])

        if active.sum() < 20:
            scores.append(0.0)
            trade_counts.append(int(active.sum()))
            continue

        acc = accuracy_score(y[test_idx][active], pred[active])
        coverage = active.mean()
        scores.append(float(acc * (0.65 + 0.35 * coverage)))
        trade_counts.append(int(active.sum()))

    return float(np.mean(scores) - 0.0005 * np.std(trade_counts))


def random_search(X: np.ndarray, y: np.ndarray, n_trials: int = 10) -> Dict[str, float]:
    rng = np.random.default_rng(42)
    best_score = -1.0
    best_params: Dict[str, float] = {}
    for _ in range(n_trials):
        params = {
            "n_estimators": int(rng.integers(60, 160)),
            "learning_rate": float(rng.uniform(0.02, 0.18)),
            "max_depth": int(rng.integers(2, 5)),
            "subsample": float(rng.uniform(0.65, 1.0)),
            "min_confidence": float(rng.uniform(0.35, 0.62)),
        }
        score = score_params(X, y, params)
        if score > best_score:
            best_score = score
            best_params = params | {"score": score}
    return best_params


def main() -> None:
    df = load_training_frame()
    X, y = build_dataset(df)
    logger.info("Dataset: %s samples, %s features", X.shape[0], X.shape[1])

    tuner = OptunaTuner(direction="maximize")
    if getattr(tuner, "_storage_url", None):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 260),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "min_confidence": trial.suggest_float("min_confidence", 0.35, 0.62),
            }
            return score_params(X, y, params)

        best = tuner.optimize("argus_winrate_decision_layer", objective, n_trials=20, timeout_s=300)
    else:
        best = random_search(X, y, n_trials=8)

    out = Path("data/models_mtf/winrate_tuning.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    logger.info("Best tuning saved to %s: %s", out, best)


if __name__ == "__main__":
    main()
