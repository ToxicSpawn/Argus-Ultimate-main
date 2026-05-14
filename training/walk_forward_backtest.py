"""
Walk-Forward Backtest Engine
============================
Rolling train/test windows across historical OHLCV data.
Reports per-window and aggregate Sharpe, drawdown, win rate.

Usage:
    python training/walk_forward_backtest.py --symbol BTC/AUD --windows 6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from training.optimise_params import fast_backtest, load_ohlcv, make_objective

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def walk_forward(
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    n_windows:  int = 6,
    train_frac: float = 0.75,
    optuna_trials: int = 50,
) -> List[Dict[str, Any]]:
    """
    Perform rolling walk-forward optimisation + test.

    For each window:
      1. Train slice  → Optuna optimises params
      2. Test slice   → fast_backtest with best params
      3. Record results

    Returns list of per-window result dicts.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna not installed")
        return []

    n      = len(closes)
    window = n // n_windows
    results = []

    for w in range(n_windows - 1):
        start = w * window
        mid   = start + int(window * train_frac)
        end   = start + window

        train_c, train_h, train_l = closes[start:mid], highs[start:mid], lows[start:mid]
        test_c,  test_h,  test_l  = closes[mid:end],   highs[mid:end],   lows[mid:end]

        if len(train_c) < 100 or len(test_c) < 20:
            continue

        # Optimise on train
        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(train_c, train_h, train_l),
            n_trials=optuna_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        train_sharpe = study.best_value

        # Evaluate on test (out-of-sample)
        test_result = fast_backtest(test_c, test_h, test_l, best_params)

        result = {
            "window":        w,
            "train_start":   int(start),
            "train_end":     int(mid),
            "test_start":    int(mid),
            "test_end":      int(end),
            "train_sharpe":  round(train_sharpe, 3),
            "test_sharpe":   round(test_result["sharpe"], 3),
            "test_return":   round(test_result["total_return"], 4),
            "test_mdd":      round(test_result["max_drawdown"], 4),
            "test_win_rate": round(test_result["win_rate"], 3),
            "n_trades":      test_result["n_trades"],
            "best_params":   best_params,
        }
        results.append(result)
        logger.info(
            "Window %d/%d | Train Sharpe=%.2f | Test Sharpe=%.2f | "
            "Return=%.2f%% | MDD=%.2f%% | WinRate=%.0f%% | Trades=%d",
            w + 1, n_windows - 1,
            train_sharpe,
            test_result["sharpe"],
            test_result["total_return"] * 100,
            test_result["max_drawdown"] * 100,
            test_result["win_rate"] * 100,
            test_result["n_trades"],
        )

    return results


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}
    sharpes    = [r["test_sharpe"]   for r in results]
    returns    = [r["test_return"]   for r in results]
    mdds       = [r["test_mdd"]      for r in results]
    win_rates  = [r["test_win_rate"] for r in results]
    return {
        "mean_sharpe":    round(float(np.mean(sharpes)),   3),
        "median_sharpe":  round(float(np.median(sharpes)), 3),
        "mean_return":    round(float(np.mean(returns)),   4),
        "mean_mdd":       round(float(np.mean(mdds)),      4),
        "mean_win_rate":  round(float(np.mean(win_rates)), 3),
        "n_windows":      len(results),
        "profitable_windows": sum(1 for r in results if r["test_return"] > 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest for Argus")
    parser.add_argument("--symbol",  default="BTC/AUD",  help="Trading pair")
    parser.add_argument("--windows", type=int, default=6,  help="Number of windows")
    parser.add_argument("--trials",  type=int, default=50, help="Optuna trials per window")
    parser.add_argument("--output",  default="models/wf_results.json", help="Output JSON")
    args = parser.parse_args()

    closes, highs, lows = load_ohlcv(args.symbol, bars=5000)
    results = walk_forward(closes, highs, lows, n_windows=args.windows, optuna_trials=args.trials)
    agg     = aggregate(results)

    logger.info("\n=== WALK-FORWARD AGGREGATE ===")
    for k, v in agg.items():
        logger.info("  %s: %s", k, v)

    out = {"aggregate": agg, "windows": results}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s", args.output)

    # CI gate: fail if mean Sharpe < 0.5
    if agg.get("mean_sharpe", -999) < 0.5:
        logger.error("WALK-FORWARD FAILED: mean Sharpe %.3f < 0.5", agg.get("mean_sharpe", 0))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
