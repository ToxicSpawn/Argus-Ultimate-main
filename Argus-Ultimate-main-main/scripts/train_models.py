#!/usr/bin/env python
"""
ARGUS ML Training Pipeline — one-command model training.

Downloads/generates historical crypto data, computes features, and trains:
  1. Regime classifier (XGBoost-style GradientBoosting)
  2. Volatility forecaster (GradientBoosting regressor)
  3. Alpha model (direction prediction, walk-forward validated)

Usage:
    py scripts/train_models.py [--symbols BTC-USD ETH-USD] [--years 5] [--output-dir models/]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.training_features import (
    compute_features,
    label_regimes,
    FEATURE_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_models")


# ─── Synthetic data generation ──────────────────────────────────────────────


def generate_synthetic_crypto(
    symbol: str = "BTC-USD",
    years: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV data mimicking real crypto characteristics:
      - Fat tails (kurtosis > 3)
      - Volatility clustering (GARCH-like)
      - Regime switches (bull/bear/sideways lasting 30-90 days)
      - Mean daily return ~0.05%, daily vol ~3-4%
    """
    rng = np.random.RandomState(seed)
    n_days = years * 365

    # --- Regime generation ---
    regimes: List[str] = []
    regime_labels = ["bull", "bear", "sideways"]
    remaining = n_days
    while remaining > 0:
        regime = rng.choice(regime_labels)
        duration = rng.randint(30, 91)
        duration = min(duration, remaining)
        regimes.extend([regime] * duration)
        remaining -= duration

    # --- Regime-dependent return parameters ---
    regime_params = {
        "bull":     {"mu": 0.0012, "base_vol": 0.030},
        "bear":     {"mu": -0.0008, "base_vol": 0.035},
        "sideways": {"mu": 0.0001, "base_vol": 0.020},
    }

    # --- GARCH(1,1)-like volatility clustering ---
    omega = 0.00001
    alpha_garch = 0.10
    beta_garch = 0.85
    sigma2 = 0.03 ** 2
    log_returns = np.zeros(n_days)

    for i in range(n_days):
        params = regime_params[regimes[i]]
        mu = params["mu"]
        base_vol = params["base_vol"]

        # Mix GARCH variance with regime base vol
        target_var = base_vol ** 2
        sigma2 = omega + alpha_garch * (log_returns[i - 1] ** 2 if i > 0 else target_var) + beta_garch * sigma2
        sigma2 = max(sigma2, 1e-8)

        # Fat tails via t-distribution (df=5)
        z = rng.standard_t(df=5)
        log_returns[i] = mu + np.sqrt(sigma2) * z

    # --- Build price series ---
    initial_price = {"BTC-USD": 30000.0, "ETH-USD": 2000.0}.get(symbol, 1000.0)
    close_prices = initial_price * np.exp(np.cumsum(log_returns))

    # --- Build OHLCV ---
    dates = pd.date_range(
        end=datetime.now(timezone.utc).date(),
        periods=n_days,
        freq="D",
    )

    # Intraday range simulation
    daily_vol = np.abs(log_returns)
    high_factor = 1.0 + rng.uniform(0.001, 0.01, n_days) + daily_vol * 0.5
    low_factor = 1.0 - rng.uniform(0.001, 0.01, n_days) - daily_vol * 0.5
    open_factor = 1.0 + rng.normal(0, 0.005, n_days)

    highs = close_prices * high_factor
    lows = close_prices * low_factor
    opens = close_prices * open_factor

    # Ensure high >= max(open, close) and low <= min(open, close)
    highs = np.maximum(highs, np.maximum(opens, close_prices))
    lows = np.minimum(lows, np.minimum(opens, close_prices))

    # Volume: correlated with volatility + regime
    base_volume = {"BTC-USD": 1e9, "ETH-USD": 5e8}.get(symbol, 1e8)
    volumes = base_volume * (1.0 + daily_vol * 10) * rng.lognormal(0, 0.5, n_days)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": close_prices,
        "volume": volumes,
        "symbol": symbol,
    })

    return df


# ─── Training functions ─────────────────────────────────────────────────────


def train_regime_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
) -> Tuple[Any, Dict[str, Any]]:
    """Train regime classifier. Returns (model, metrics_dict)."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, shuffle=False,  # time-series: no shuffle
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Classification report as dict
    all_labels = list(range(len(label_names)))
    report = classification_report(
        y_test, y_pred_test,
        labels=all_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # Feature importances
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))

    metrics = {
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 200,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_time_seconds": round(train_time, 2),
        "feature_importances": importances,
        "classification_report": {
            k: v for k, v in report.items()
            if k in label_names or k in ("accuracy", "macro avg", "weighted avg")
        },
    }

    logger.info(
        "Regime classifier: train_acc=%.4f, test_acc=%.4f (%d train, %d test, %.1fs)",
        train_acc, test_acc, len(X_train), len(X_test), train_time,
    )

    return model, metrics


def train_volatility_forecaster(
    features: np.ndarray,
    target: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    """Train volatility forecaster (regressor). Returns (model, metrics_dict)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    split = int(len(features) * 0.8)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    metrics = {
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 200,
        "r2_score": round(r2, 4),
        "mae": round(mae, 6),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_time_seconds": round(train_time, 2),
        "target": "next_5d_realized_vol",
    }

    logger.info(
        "Volatility forecaster: R²=%.4f, MAE=%.6f (%d train, %d test, %.1fs)",
        r2, mae, len(X_train), len(X_test), train_time,
    )

    return model, metrics


def train_alpha_model(
    features: np.ndarray,
    target: np.ndarray,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train alpha model (binary direction classifier) with walk-forward validation.
    Target: sign of next-day return (1 = up, 0 = down).
    Returns (model, metrics_dict).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    split = int(len(features) * 0.8)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Compute signal-based Sharpe approximation
    # Use probability of class 1 as signal strength
    y_proba = model.predict_proba(X_test)[:, 1]
    signal = y_proba - 0.5  # center at 0
    # Assume unit returns aligned with signal
    actual_returns = np.where(y_test == 1, 1.0, -1.0)
    strategy_returns = signal * actual_returns
    sharpe = float(np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10)) * np.sqrt(252)

    metrics = {
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 150,
        "test_accuracy": round(test_acc, 4),
        "signal_sharpe": round(sharpe, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_time_seconds": round(train_time, 2),
        "walk_forward_split": "80/20 temporal",
        "target": "sign_of_next_day_return",
    }

    logger.info(
        "Alpha model: test_acc=%.4f, signal_sharpe=%.4f (%d train, %d test, %.1fs)",
        test_acc, sharpe, len(X_train), len(X_test), train_time,
    )

    return model, metrics


# ─── Main pipeline ───────────────────────────────────────────────────────────


def run_pipeline(
    symbols: List[str],
    years: int = 5,
    output_dir: str = "models",
    data_dir: str = "data/historical",
) -> Dict[str, Any]:
    """Execute the full training pipeline. Returns training report."""
    import joblib

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "years": years,
        "models": {},
    }

    # ── Step 1: Generate / load data ─────────────────────────────────────
    logger.info("Step 1: Generating synthetic crypto data for %s (%d years)", symbols, years)
    all_dfs = []
    for i, sym in enumerate(symbols):
        seed = 42 + i
        df = generate_synthetic_crypto(symbol=sym, years=years, seed=seed)
        parquet_path = data_path / f"{sym.replace('-', '_')}_daily.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info("  Saved %d rows to %s", len(df), parquet_path)
        all_dfs.append(df)

    report["data"] = {
        "source": "synthetic",
        "total_rows": sum(len(d) for d in all_dfs),
        "date_range": {
            "start": str(all_dfs[0]["timestamp"].min().date()),
            "end": str(all_dfs[0]["timestamp"].max().date()),
        },
    }

    # ── Step 2: Compute features ─────────────────────────────────────────
    logger.info("Step 2: Computing features")
    # Combine all symbols for training (use close prices and volumes)
    feature_frames = []
    for df in all_dfs:
        feat_df = compute_features(df)
        feature_frames.append(feat_df)

    combined = pd.concat(feature_frames, ignore_index=True)
    combined = combined.dropna()
    logger.info("  Feature matrix: %d rows x %d features", len(combined), len(FEATURE_NAMES))

    features = combined[FEATURE_NAMES].values

    # ── Step 3: Train regime classifier ──────────────────────────────────
    logger.info("Step 3: Training regime classifier")
    regime_labels, regime_names = label_regimes(combined)
    regime_model, regime_metrics = train_regime_classifier(
        features, regime_labels, regime_names,
    )

    regime_path = output_path / "regime_classifier.pkl"
    metadata = {
        "model_type": "regime_classifier",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_NAMES,
        "classes": regime_names,
        "version": "1.0.0",
    }
    joblib.dump({"model": regime_model, "metadata": metadata}, regime_path)
    logger.info("  Saved to %s", regime_path)
    report["models"]["regime_classifier"] = regime_metrics

    # ── Step 4: Train volatility forecaster ──────────────────────────────
    logger.info("Step 4: Training volatility forecaster")
    # Target: next-5-day realized volatility
    close_all = combined["close"].values
    returns_all = np.diff(np.log(close_all))
    vol_target = np.full(len(combined), np.nan)
    for i in range(len(combined) - 5):
        vol_target[i] = np.std(returns_all[i:i + 5]) * np.sqrt(252)

    # Align: drop rows where target is NaN
    valid_mask = ~np.isnan(vol_target)
    vol_features = features[valid_mask]
    vol_y = vol_target[valid_mask]

    vol_model, vol_metrics = train_volatility_forecaster(vol_features, vol_y)

    vol_path = output_path / "volatility_forecaster.pkl"
    metadata_vol = {
        "model_type": "volatility_forecaster",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_NAMES,
        "target": "next_5d_realized_vol_annualized",
        "version": "1.0.0",
    }
    joblib.dump({"model": vol_model, "metadata": metadata_vol}, vol_path)
    logger.info("  Saved to %s", vol_path)
    report["models"]["volatility_forecaster"] = vol_metrics

    # ── Step 5: Train alpha model ────────────────────────────────────────
    logger.info("Step 5: Training alpha model (direction prediction)")
    # Target: sign of next-day return (binary)
    next_day_return = np.zeros(len(combined))
    for i in range(len(combined) - 1):
        next_day_return[i] = np.log(close_all[i + 1] / close_all[i])
    alpha_target = (next_day_return > 0).astype(int)

    # Drop last row (no next-day target)
    alpha_features = features[:-1]
    alpha_y = alpha_target[:-1]

    alpha_model, alpha_metrics = train_alpha_model(alpha_features, alpha_y)

    alpha_path = output_path / "alpha_model.pkl"
    metadata_alpha = {
        "model_type": "alpha_model",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_NAMES,
        "target": "sign_of_next_day_return",
        "classes": ["down", "up"],
        "version": "1.0.0",
    }
    joblib.dump({"model": alpha_model, "metadata": metadata_alpha}, alpha_path)
    logger.info("  Saved to %s", alpha_path)
    report["models"]["alpha_model"] = alpha_metrics

    # ── Step 6: Generate training report ─────────────────────────────────
    report_path = output_path / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Training report saved to %s", report_path)

    # ── Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ARGUS ML Training Pipeline — Summary")
    print("=" * 70)
    print(f"  Date:       {report['training_date']}")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Data:       {report['data']['total_rows']} rows ({report['data']['source']})")
    print(f"  Features:   {len(FEATURE_NAMES)} features")
    print()
    print("  Regime Classifier:")
    print(f"    Train accuracy: {regime_metrics['train_accuracy']:.4f}")
    print(f"    Test accuracy:  {regime_metrics['test_accuracy']:.4f}")
    print()
    print("  Volatility Forecaster:")
    print(f"    R² score: {vol_metrics['r2_score']:.4f}")
    print(f"    MAE:      {vol_metrics['mae']:.6f}")
    print()
    print("  Alpha Model (Direction):")
    print(f"    Test accuracy:  {alpha_metrics['test_accuracy']:.4f}")
    print(f"    Signal Sharpe:  {alpha_metrics['signal_sharpe']:.4f}")
    print()
    print(f"  Models saved to: {output_path.resolve()}")
    print(f"  Report:          {report_path.resolve()}")
    print("=" * 70)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARGUS ML Training Pipeline",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC-USD", "ETH-USD"],
        help="Symbols to generate data for (default: BTC-USD ETH-USD)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Years of historical data to generate (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained models (default: models/)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/historical",
        help="Directory to save historical data (default: data/historical/)",
    )

    args = parser.parse_args()
    run_pipeline(
        symbols=args.symbols,
        years=args.years,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
