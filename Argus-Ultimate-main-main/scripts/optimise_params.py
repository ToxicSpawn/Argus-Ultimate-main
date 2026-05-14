"""
optimise_params.py — Bayesian Optuna parameter optimiser for Argus Ultimate.

Uses Optuna's TPE sampler to search strategy hyperparameters, evaluated
via the WalkForwardEngine with the Kraken maker fee model.

Usage
-----
    python scripts/optimise_params.py \
        --symbol XBT/USD \
        --bars 2000 \
        --trials 200 \
        --train-bars 500 \
        --test-bars 100 \
        --output params_best.json

Outputs
-------
  - Best parameter dict printed to stdout and saved as JSON
  - Optuna study persisted to SQLite (optuna_study.db) for resumability
  - Pareto-front summary if multi-objective mode enabled
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    print("[ERROR] optuna not installed. Run: pip install optuna", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting.walk_forward import WalkForwardEngine, KRAKEN_MAKER_FEE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter search space definition
# ---------------------------------------------------------------------------

def suggest_params(trial: "optuna.Trial") -> Dict[str, Any]:
    """
    Define the hyperparameter search space.
    Extend this function to add / remove parameters.
    """
    return {
        # Trend-following
        "fast_ema": trial.suggest_int("fast_ema", 5, 50),
        "slow_ema": trial.suggest_int("slow_ema", 20, 200),
        "atr_period": trial.suggest_int("atr_period", 7, 28),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 1.0, 4.0),
        # Mean reversion
        "rsi_period": trial.suggest_int("rsi_period", 7, 21),
        "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
        "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 80),
        # Risk
        "risk_per_trade": trial.suggest_float("risk_per_trade", 0.005, 0.03),
        "max_position": trial.suggest_float("max_position", 0.1, 1.0),
        # Regime filter
        "regime_lookback": trial.suggest_int("regime_lookback", 20, 100),
    }


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategy(params: Dict[str, Any]) -> Any:
    """
    Build a strategy callable from a parameter dict.
    Returns fn(test_prices, train_prices) -> positions array.

    This default implementation is a simple EMA crossover + RSI filter.
    Replace with your actual strategy factory.
    """
    fast  = int(params["fast_ema"])
    slow  = int(params["slow_ema"])
    rsi_p = int(params["rsi_period"])
    ob    = int(params["rsi_overbought"])
    os_   = int(params["rsi_oversold"])

    def _ema(prices: np.ndarray, period: int) -> np.ndarray:
        out = np.full_like(prices, np.nan)
        if len(prices) < period:
            return out
        k = 2.0 / (period + 1)
        out[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            out[i] = prices[i] * k + out[i - 1] * (1 - k)
        return out

    def _rsi(prices: np.ndarray, period: int) -> np.ndarray:
        delta = np.diff(prices, prepend=prices[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = np.convolve(gain, np.ones(period) / period, mode="same")
        avg_l = np.convolve(loss, np.ones(period) / period, mode="same")
        rs    = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-10))
        return 100 - (100 / (1 + rs))

    def strategy(test_prices: np.ndarray, _train_prices: np.ndarray) -> np.ndarray:
        prices = test_prices
        ema_f  = _ema(prices, fast)
        ema_s  = _ema(prices, slow)
        rsi    = _rsi(prices, rsi_p)
        n = len(prices)
        pos = np.zeros(n)
        for i in range(slow, n):
            if np.isnan(ema_f[i]) or np.isnan(ema_s[i]):
                continue
            long_signal  = ema_f[i] > ema_s[i] and rsi[i] < ob
            short_signal = ema_f[i] < ema_s[i] and rsi[i] > os_
            if long_signal:
                pos[i] = 1.0
            elif short_signal:
                pos[i] = -1.0
        return pos

    return strategy


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(
    prices: np.ndarray,
    train_bars: int,
    test_bars: int,
    fee_rate: float,
    anchored: bool,
) -> Any:
    """Return an Optuna objective function closed over the price data."""

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial)

        # Constraint: fast EMA must be shorter than slow EMA
        if params["fast_ema"] >= params["slow_ema"]:
            raise optuna.exceptions.TrialPruned()

        strategy_fn = build_strategy(params)
        engine = WalkForwardEngine(
            prices=prices,
            strategy_fn=strategy_fn,
            train_bars=train_bars,
            test_bars=test_bars,
            fee_rate=fee_rate,
            anchored=anchored,
        )
        wf_result = engine.run()
        if not wf_result.folds:
            return -999.0

        summary = wf_result.summary()
        # Primary objective: mean Sharpe across folds
        # Penalty for negative fold consistency
        sharpe    = summary["mean_sharpe"]
        pos_folds = summary["positive_folds_pct"] / 100.0
        score = sharpe * 0.7 + pos_folds * 0.3
        return float(score)

    return objective


# ---------------------------------------------------------------------------
# Price loader (stub — replace with real data source)
# ---------------------------------------------------------------------------

def load_prices(symbol: str, bars: int) -> np.ndarray:
    """
    Load OHLCV close prices for `symbol`.
    Stub: returns synthetic random-walk data.
    Replace with Kraken REST / ccxt / CSV loader.
    """
    logger.warning(
        "Using synthetic price data for %s (%d bars). "
        "Replace load_prices() with a real data source.",
        symbol, bars,
    )
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, bars)
    prices  = 30000.0 * np.cumprod(1 + returns)
    return prices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian Optuna parameter optimiser")
    p.add_argument("--symbol",      default="XBT/USD",         help="Trading pair")
    p.add_argument("--bars",        type=int, default=2000,     help="Price history length")
    p.add_argument("--trials",      type=int, default=100,      help="Optuna trials")
    p.add_argument("--train-bars",  type=int, default=500,      help="In-sample bars per fold")
    p.add_argument("--test-bars",   type=int, default=100,      help="OOS bars per fold")
    p.add_argument("--fee-rate",    type=float, default=KRAKEN_MAKER_FEE, help="Fee rate")
    p.add_argument("--no-anchored", action="store_true",        help="Use rolling (non-anchored) WFA")
    p.add_argument("--study-db",    default="optuna_study.db",  help="SQLite storage path")
    p.add_argument("--study-name",  default="argus_optimise",   help="Optuna study name")
    p.add_argument("--output",      default="params_best.json", help="Output JSON path")
    p.add_argument("--jobs",        type=int, default=1,        help="Parallel Optuna jobs")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    prices = load_prices(args.symbol, args.bars)
    logger.info("Loaded %d price bars for %s", len(prices), args.symbol)

    storage = f"sqlite:///{args.study_db}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(
        prices=prices,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        fee_rate=args.fee_rate,
        anchored=not args.no_anchored,
    )

    logger.info(
        "Starting optimisation: %d trials, %d jobs, study=%s",
        args.trials, args.jobs, args.study_name,
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        n_jobs=args.jobs,
        show_progress_bar=True,
        catch=(Exception,),
    )

    best = study.best_trial
    logger.info("Best trial #%d | score=%.4f", best.number, best.value)
    logger.info("Best params: %s", json.dumps(best.params, indent=2))

    out_path = Path(args.output)
    out_path.write_text(json.dumps(best.params, indent=2))
    logger.info("Best params saved to %s", out_path)

    # Print top-5 trials
    print("\n--- Top 5 trials ---")
    top5 = sorted(study.trials, key=lambda t: t.value or -999, reverse=True)[:5]
    for t in top5:
        print(f"  #{t.number:4d}  score={t.value:.4f}  params={t.params}")


if __name__ == "__main__":
    main()
