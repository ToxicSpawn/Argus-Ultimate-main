"""
Bayesian Parameter Optimisation via Optuna
==========================================
Optimises strategy thresholds (RSI, z-score, ATR multiplier, MTF score,
scalping TP/SL, ARM limits) against a rolling Kraken OHLCV dataset.

Usage:
    python training/optimise_params.py --symbol BTC/AUD --trials 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Lightweight vectorised backtester
# ---------------------------------------------------------------------------

def _compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    ema = np.empty_like(prices)
    ema[:period] = np.nan
    ema[period - 1] = prices[:period].mean()
    for i in range(period, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(prices, prepend=prices[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(period) / period, mode="same")
    avg_loss = np.convolve(loss, np.ones(period) / period, mode="same")
    rs  = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def _compute_zscore(prices: np.ndarray, window: int = 20) -> np.ndarray:
    result = np.full_like(prices, np.nan)
    for i in range(window, len(prices)):
        w   = prices[i - window:i]
        mu  = w.mean()
        std = w.std()
        result[i] = (prices[i] - mu) / std if std > 1e-10 else 0.0
    return result


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr  = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.convolve(tr, np.ones(period) / period, mode="same")
    return atr


def fast_backtest(
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    params: Dict[str, Any],
    fee_pct: float = 0.0026,
) -> Dict[str, float]:
    """
    Vectorised single-strategy backtest.
    Returns Sharpe, total return, max drawdown, win_rate.
    """
    rsi_os     = params.get("rsi_oversold",       30.0)
    rsi_ob     = params.get("rsi_overbought",      70.0)
    z_entry    = params.get("zscore_entry",         1.8)
    ema_fast   = int(params.get("ema_fast",           9))
    ema_slow   = int(params.get("ema_slow",          21))
    tp_mult    = params.get("tp_atr_mult",           3.0)
    sl_mult    = params.get("sl_atr_mult",           1.5)
    min_conf   = params.get("min_confidence",        0.45)

    rsi    = _compute_rsi(closes)
    zscore = _compute_zscore(closes)
    fast_e = _compute_ema(closes, ema_fast)
    slow_e = _compute_ema(closes, ema_slow)
    atr    = _compute_atr(highs, lows, closes)

    in_trade   = False
    entry_px   = 0.0
    tp_px      = 0.0
    sl_px      = 0.0
    pnl_series: List[float] = []
    n = len(closes)

    for i in range(50, n):
        if np.isnan(fast_e[i]) or np.isnan(slow_e[i]) or np.isnan(rsi[i]):
            continue

        if in_trade:
            px = closes[i]
            if px >= tp_px:
                pnl = (tp_px - entry_px) / entry_px - fee_pct
                pnl_series.append(pnl)
                in_trade = False
            elif px <= sl_px:
                pnl = (sl_px - entry_px) / entry_px - fee_pct
                pnl_series.append(pnl)
                in_trade = False
        else:
            trend_up   = fast_e[i] > slow_e[i]
            oversold   = rsi[i] < rsi_os
            z_low      = zscore[i] < -z_entry

            if trend_up and (oversold or z_low):
                entry_px = closes[i] * (1 + fee_pct / 2)
                tp_px    = entry_px + atr[i] * tp_mult
                sl_px    = entry_px - atr[i] * sl_mult
                in_trade = True

    if not pnl_series:
        return {"sharpe": -999.0, "total_return": -1.0, "max_drawdown": 1.0, "win_rate": 0.0}

    arr    = np.array(pnl_series)
    mean_r = arr.mean()
    std_r  = arr.std() if arr.std() > 0 else 1e-10
    sharpe = mean_r / std_r * np.sqrt(252)

    cum    = np.cumprod(1 + arr)
    peak   = np.maximum.accumulate(cum)
    dd     = (cum - peak) / peak
    mdd    = float(-dd.min())

    return {
        "sharpe":       float(sharpe),
        "total_return": float(cum[-1] - 1),
        "max_drawdown": mdd,
        "win_rate":     float((arr > 0).mean()),
        "n_trades":     len(pnl_series),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray):
    def objective(trial) -> float:
        params = {
            "rsi_oversold":     trial.suggest_float("rsi_oversold",    20.0, 40.0),
            "rsi_overbought":   trial.suggest_float("rsi_overbought",  60.0, 80.0),
            "zscore_entry":     trial.suggest_float("zscore_entry",     1.0,  3.0),
            "ema_fast":         trial.suggest_int(  "ema_fast",          5,   15),
            "ema_slow":         trial.suggest_int(  "ema_slow",         15,   50),
            "tp_atr_mult":      trial.suggest_float("tp_atr_mult",      1.5,  5.0),
            "sl_atr_mult":      trial.suggest_float("sl_atr_mult",      0.5,  3.0),
            "min_confidence":   trial.suggest_float("min_confidence",   0.35, 0.65),
        }
        # Enforce ema_fast < ema_slow
        if params["ema_fast"] >= params["ema_slow"]:
            return -999.0

        result = fast_backtest(closes, highs, lows, params)

        # Penalise drawdown > 20% and fewer than 10 trades
        if result["max_drawdown"] > 0.20 or result["n_trades"] < 10:
            return -999.0

        return result["sharpe"]
    return objective


# ---------------------------------------------------------------------------
# Data loader (synthetic if no real data)
# ---------------------------------------------------------------------------

def load_ohlcv(symbol: str, bars: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load OHLCV from local CSV cache or generate synthetic data.
    CSV expected at: data/cache/{symbol.replace('/','-')}_1h.csv
    Columns: timestamp,open,high,low,close,volume
    """
    cache_path = f"data/cache/{symbol.replace('/', '-')}_1h.csv"
    if os.path.exists(cache_path):
        import csv
        rows = []
        with open(cache_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        closes = np.array([float(r["close"]) for r in rows[-bars:]])
        highs  = np.array([float(r["high"])  for r in rows[-bars:]])
        lows   = np.array([float(r["low"])   for r in rows[-bars:]])
        logger.info("Loaded %d bars from %s", len(closes), cache_path)
        return closes, highs, lows

    # Synthetic GBM data
    logger.info("No cache found — generating %d synthetic bars for %s", bars, symbol)
    rng    = np.random.default_rng(42)
    prices = [50000.0]
    for _ in range(bars - 1):
        ret = rng.normal(0.0002, 0.015)
        prices.append(prices[-1] * (1 + ret))
    closes = np.array(prices)
    noise  = rng.uniform(0.995, 1.005, size=bars)
    highs  = closes * rng.uniform(1.001, 1.015, size=bars)
    lows   = closes * rng.uniform(0.985, 0.999, size=bars)
    return closes, highs, lows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bayesian param optimisation for Argus")
    parser.add_argument("--symbol",  default="BTC/AUD",  help="Trading pair")
    parser.add_argument("--trials",  type=int, default=200, help="Optuna trials")
    parser.add_argument("--output",  default="models/best_params.json", help="Output JSON")
    args = parser.parse_args()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna not installed. Run: pip install optuna")
        return

    closes, highs, lows = load_ohlcv(args.symbol)
    logger.info("Starting Bayesian optimisation: %d trials on %s", args.trials, args.symbol)

    study = optuna.create_study(direction="maximize", study_name="argus_params")
    study.optimize(make_objective(closes, highs, lows), n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    best["best_sharpe"] = study.best_value
    logger.info("Best params: %s", json.dumps(best, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(best, f, indent=2)
    logger.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
