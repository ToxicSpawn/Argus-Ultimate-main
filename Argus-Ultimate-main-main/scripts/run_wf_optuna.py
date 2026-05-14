#!/usr/bin/env python3
"""Walk-forward + Optuna optimisation runner.

Usage:
    python scripts/run_wf_optuna.py \
        --symbol BTC/USDT \
        --timeframe 1d \
        --n-trials 100 \
        --train-days 365 \
        --test-days 90 \
        --output results/wf_optuna.json

The script fetches historical OHLCV, runs Optuna to find the best
hyperparameters on the in-sample window, then evaluates on out-of-sample
using walk-forward analysis.  Results are written as JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("wf_optuna_runner")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward + Optuna strategy optimiser")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--exchange", default="kraken")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--step-days", type=int, default=90)
    p.add_argument("--limit", type=int, default=1000, help="Max candles to fetch")
    p.add_argument("--output", default="results/wf_optuna.json")
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


async def fetch_ohlcv(symbol: str, timeframe: str, exchange_id: str, limit: int) -> pd.DataFrame:
    from core.async_ohlcv_fetcher import AsyncOHLCVFetcher
    import os

    fetcher = AsyncOHLCVFetcher(
        exchange_id=exchange_id,
        api_key=os.getenv("KRAKEN_API_KEY", ""),
        api_secret=os.getenv("KRAKEN_API_SECRET", ""),
    )
    try:
        results = await fetcher.fetch_many([(symbol, timeframe)], limit=limit)
        df = results.get((symbol, timeframe), pd.DataFrame())
        return df
    finally:
        await fetcher.close()


def build_ma_strategy(fast: int, slow: int):
    """Simple dual-MA crossover strategy factory for demonstration."""
    def factory(train_df: pd.DataFrame):
        # Nothing to fit for MA — params are fixed
        def strategy_fn(test_df: pd.DataFrame) -> pd.Series:
            close = test_df["close"]
            ma_fast = close.rolling(fast).mean()
            ma_slow = close.rolling(slow).mean()
            signal = np.where(ma_fast > ma_slow, 1.0, -1.0)
            return pd.Series(signal, index=test_df.index)
        return strategy_fn
    return factory


def run_optuna_on_train(
    train_df: pd.DataFrame,
    n_trials: int,
    n_jobs: int,
) -> Dict[str, Any]:
    from backtesting.optuna_optimiser import OptunaOptimiser
    from backtesting.walk_forward_backtester import WalkForwardBacktester
    import optuna

    inner_wf = WalkForwardBacktester(
        train_periods=min(180, len(train_df) // 2),
        test_periods=max(30, len(train_df) // 5),
        step_periods=max(30, len(train_df) // 5),
    )

    def param_space(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "fast": trial.suggest_int("fast", 5, 50),
            "slow": trial.suggest_int("slow", 20, 200),
        }

    def objective(params: Dict[str, Any], df: pd.DataFrame) -> float:
        fast = int(params["fast"])
        slow = int(params["slow"])
        if fast >= slow:
            return -999.0
        summary = inner_wf.run(df, build_ma_strategy(fast, slow))
        return summary.mean_oos_sharpe

    optimiser = OptunaOptimiser(
        n_trials=n_trials,
        n_jobs=n_jobs,
        direction="maximize",
        study_name="argus_wf_optuna",
    )
    best_params = optimiser.optimise(param_space, objective, train_df)
    importance = optimiser.get_importance()
    return {"best_params": best_params, "importance": importance}


def run_walk_forward(
    df: pd.DataFrame,
    best_params: Dict[str, Any],
    train_days: int,
    test_days: int,
    step_days: int,
) -> Dict[str, Any]:
    from backtesting.walk_forward_backtester import WalkForwardBacktester

    fast = int(best_params.get("fast", 20))
    slow = int(best_params.get("slow", 50))
    wf = WalkForwardBacktester(
        train_periods=train_days,
        test_periods=test_days,
        step_periods=step_days,
        anchored=True,
    )
    summary = wf.run(df, build_ma_strategy(fast, slow))
    folds_out = []
    for f in summary.folds:
        folds_out.append({
            "fold": f.fold,
            "train_start": str(f.train_start),
            "train_end": str(f.train_end),
            "test_start": str(f.test_start),
            "test_end": str(f.test_end),
            "sharpe": round(f.sharpe, 4),
            "sortino": round(f.sortino, 4),
            "calmar": round(f.calmar, 4),
            "total_return_pct": round(f.total_return * 100, 3),
            "max_drawdown_pct": round(f.max_drawdown * 100, 3),
            "win_rate": round(f.win_rate, 4),
            "num_trades": f.num_trades,
        })
    return {
        "mean_oos_sharpe": round(summary.mean_oos_sharpe, 4),
        "mean_oos_return_pct": round(summary.mean_oos_return * 100, 3),
        "consistency_ratio": round(summary.consistency_ratio, 4),
        "num_folds": len(summary.folds),
        "folds": folds_out,
    }


async def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching %s candles for %s @ %s ...", args.limit, args.symbol, args.timeframe)
    df = await fetch_ohlcv(args.symbol, args.timeframe, args.exchange, args.limit)

    if df.empty:
        logger.error("No data fetched — check API keys and symbol name")
        sys.exit(1)

    logger.info("Fetched %d candles (%s → %s)", len(df), df.index[0], df.index[-1])

    # Use first 70% as in-sample for Optuna optimisation
    split = int(len(df) * 0.70)
    train_df = df.iloc[:split].copy()

    logger.info("Running Optuna optimisation on %d candles (n_trials=%d) ...", len(train_df), args.n_trials)
    optuna_result = run_optuna_on_train(train_df, args.n_trials, args.n_jobs)
    best_params = optuna_result["best_params"]
    logger.info("Best params: %s | Importance: %s", best_params, optuna_result["importance"])

    logger.info("Running walk-forward analysis on full dataset ...")
    wf_result = run_walk_forward(
        df,
        best_params,
        args.train_days,
        args.test_days,
        args.step_days,
    )

    output = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "total_candles": len(df),
        "optuna": optuna_result,
        "walk_forward": wf_result,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info("Results written to %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Symbol:             {args.symbol}")
    print(f"Best params:        fast={best_params.get('fast')}  slow={best_params.get('slow')}")
    print(f"WF Mean OOS Sharpe: {wf_result['mean_oos_sharpe']}")
    print(f"WF Consistency:     {wf_result['consistency_ratio']:.0%}")
    print(f"WF Folds:           {wf_result['num_folds']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
