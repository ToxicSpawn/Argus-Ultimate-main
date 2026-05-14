"""Walk-forward validation for the compact Argus ML signal layer."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.winrate_enhancement import SoftLabelGenerator


def load_data(symbol: str = "BTCUSDT") -> pd.DataFrame:
    with Path("data/historical/historical_data.pkl").open("rb") as fh:
        data = pickle.load(fh)
    return pd.DataFrame(data[symbol]["1h"])[["open", "high", "low", "close", "volume"]].astype(float)


def build_dataset(df: pd.DataFrame, horizon: int = 4):
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
    return dataset.values.astype(float), labels[valid_idx].astype(int), returns.loc[valid_idx].values.astype(float)


def main() -> None:
    X, y, future_returns = build_dataset(load_data())
    splitter = TimeSeriesSplit(n_splits=5)
    folds = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = GradientBoostingClassifier(n_estimators=160, learning_rate=0.06, max_depth=3, random_state=42)
        model.fit(X_train, y[train_idx])

        proba = model.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        active = confidence >= 0.40
        fold_returns = future_returns[test_idx]
        directional_pnl = np.where(pred == 2, fold_returns, np.where(pred == 0, -fold_returns, 0.0))

        folds.append({
            "fold": fold,
            "samples": int(len(test_idx)),
            "active_trades": int(active.sum()),
            "coverage": float(active.mean()),
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "active_accuracy": float(accuracy_score(y[test_idx][active], pred[active])) if active.any() else 0.0,
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
            "mean_directional_return": float(directional_pnl[active].mean()) if active.any() else 0.0,
        })

    summary = {
        "folds": folds,
        "mean_active_accuracy": float(np.mean([f["active_accuracy"] for f in folds])),
        "mean_coverage": float(np.mean([f["coverage"] for f in folds])),
        "mean_directional_return": float(np.mean([f["mean_directional_return"] for f in folds])),
    }
    out = Path("data/models_mtf/walk_forward_winrate_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
