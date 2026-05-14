"""
CI Backtest Gate
================
Runs a fast walk-forward backtest on synthetic data.
Fails (exit code 1) if:
  - Mean Sharpe ratio < MIN_SHARPE
  - Mean max drawdown  > MAX_DRAWDOWN

Intended for GitHub Actions:
    - name: CI Backtest
      run: python scripts/ci_backtest.py
"""
from __future__ import annotations

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.optimise_params import load_ohlcv, fast_backtest
from training.walk_forward_backtest import walk_forward, aggregate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ci_backtest")

MIN_SHARPE   = 0.5
MAX_DRAWDOWN = 0.20


def main():
    logger.info("Starting CI backtest gate...")

    closes, highs, lows = load_ohlcv("BTC/AUD", bars=2000)
    results = walk_forward(
        closes, highs, lows,
        n_windows=4,
        train_frac=0.75,
        optuna_trials=30,
    )

    if not results:
        logger.error("CI FAIL: walk-forward returned no results")
        sys.exit(1)

    agg = aggregate(results)
    mean_sharpe = agg.get("mean_sharpe", -999)
    mean_mdd    = agg.get("mean_mdd",     1.0)

    logger.info("Mean Sharpe:    %.3f  (min=%.1f)",   mean_sharpe, MIN_SHARPE)
    logger.info("Mean Drawdown:  %.2f%%  (max=%.0f%%)", mean_mdd * 100, MAX_DRAWDOWN * 100)

    failed = False
    if mean_sharpe < MIN_SHARPE:
        logger.error("CI FAIL: mean Sharpe %.3f < %.1f", mean_sharpe, MIN_SHARPE)
        failed = True
    if mean_mdd > MAX_DRAWDOWN:
        logger.error("CI FAIL: mean drawdown %.2f%% > %.0f%%", mean_mdd * 100, MAX_DRAWDOWN * 100)
        failed = True

    if failed:
        sys.exit(1)

    logger.info("CI PASS: Sharpe=%.3f  MDD=%.2f%%  Windows=%d",
                mean_sharpe, mean_mdd * 100, agg.get("n_windows", 0))


if __name__ == "__main__":
    main()
