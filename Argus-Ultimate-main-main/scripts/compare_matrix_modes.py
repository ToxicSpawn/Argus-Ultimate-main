"""
compare_matrix_modes.py — Head-to-head comparison of MatrixEvaluator aggregation modes
on real Kraken OHLCV data.  (Push 29a)

Runs TentacleBacktestRunner with WEIGHTED_MEAN, MAJORITY_VOTE, and MIN_AGREEMENT
across 6-12 months of live Kraken price history and outputs a full performance
comparison table + per-tentacle attribution for each mode.

New in Push 29a
---------------
  --auto-apply flag: after comparison, writes the winning mode (by Sharpe) to
  config/matrix_mode.json so argus_bot.py picks it up automatically on next start.

Usage
-----
    python scripts/compare_matrix_modes.py
    python scripts/compare_matrix_modes.py --auto-apply          # write winner to config
    python scripts/compare_matrix_modes.py --symbol XBTUSD --months 12 --interval 60
    python scripts/compare_matrix_modes.py --output results/matrix_comparison.json

Outputs
-------
    - Console: formatted comparison table
    - JSON:    full results with fold-level detail and attribution
    - CSV:     summary table for spreadsheet analysis
    - config/matrix_mode.json  (only when --auto-apply is set)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.kraken_feed import fetch_ohlcv_rest
from backtesting.tentacle_backtest import TentacleBacktestRunner
from strategies.tentacles.matrix_evaluator import AggregationMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODES = [
    AggregationMode.WEIGHTED_MEAN,
    AggregationMode.MAJORITY_VOTE,
    AggregationMode.MIN_AGREEMENT,
]

KRAKEN_INTERVAL_BARS: Dict[int, int] = {
    1:    720,
    5:    288,
    15:   96,
    60:   24,
    240:  6,
    1440: 1,
}

# Path where the winning mode config is written when --auto-apply is set
_REPO_ROOT   = Path(__file__).resolve().parents[1]
_MODE_CONFIG = _REPO_ROOT / "config" / "matrix_mode.json"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_data(
    symbol: str,
    interval: int,
    months: int,
) -> np.ndarray:
    """
    Fetch approximately `months` months of OHLCV data from Kraken REST.
    Kraken REST returns max 720 bars per call; we page backwards for longer history.
    """
    bars_per_day = KRAKEN_INTERVAL_BARS.get(interval, 24)
    target_bars  = bars_per_day * 30 * months
    logger.info(
        "Fetching ~%d bars (%d months) of %s %dm candles from Kraken REST...",
        target_bars, months, symbol, interval,
    )

    all_candles: List[np.ndarray] = []
    since: int | None = None
    max_pages = max(1, target_bars // 720 + 1)

    for page in range(max_pages):
        try:
            candles = await fetch_ohlcv_rest(symbol=symbol, interval=interval, since=since)
            if len(candles) == 0:
                break
            all_candles.append(candles)
            since = int(candles[-1, 0]) + interval * 60
            logger.debug(
                "Page %d: %d candles (total so far: %d)",
                page + 1, len(candles), sum(len(c) for c in all_candles),
            )
            if sum(len(c) for c in all_candles) >= target_bars:
                break
            await asyncio.sleep(0.5)
        except Exception as exc:
            logger.warning("REST fetch error on page %d: %s", page + 1, exc)
            break

    if not all_candles:
        raise RuntimeError(
            f"No data returned from Kraken REST for {symbol} {interval}m. "
            "Check symbol name (e.g. XBTUSD not XBT/USD for REST API)."
        )

    combined = np.vstack(all_candles)
    _, idx   = np.unique(combined[:, 0], return_index=True)
    combined = combined[idx]
    logger.info("Total unique candles fetched: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def run_mode(
    mode: AggregationMode,
    candles: np.ndarray,
    train_bars: int,
    test_bars: int,
    obs_window: int,
) -> Dict[str, Any]:
    """Run TentacleBacktestRunner for a single matrix mode and return results dict."""
    logger.info("Running mode: %s ...", mode.value)
    t0 = time.time()

    prices = candles[:, 4].astype(np.float64)

    runner = TentacleBacktestRunner(
        prices=prices,
        ohlcv=candles,
        train_bars=train_bars,
        test_bars=test_bars,
        obs_window=obs_window,
        anchored=True,
        matrix_kwargs={"mode": mode},
        log_matrix=False,
    )

    result      = runner.run()
    elapsed     = time.time() - t0
    summary     = result.summary()
    attribution = result.attribution_report()

    folds_detail = [
        {
            "fold"       : f.fold_index,
            "return_pct" : round(f.total_return * 100, 4),
            "sharpe"     : round(f.sharpe_ratio, 4),
            "max_dd_pct" : round(f.max_drawdown * 100, 4),
            "win_rate"   : round(f.win_rate, 4),
            "num_trades" : f.num_trades,
        }
        for f in result.wf_result.folds
    ]

    logger.info(
        "%-16s | sharpe=%+.3f | return=%+.2f%% | dd=%.2f%% | win=%.1f%% | folds=%d | %.1fs",
        mode.value,
        summary.get("mean_sharpe", 0),
        summary.get("mean_return_pct", 0),
        summary.get("mean_max_drawdown_pct", 0),
        summary.get("mean_win_rate", 0) * 100,
        summary.get("num_folds", 0),
        elapsed,
    )

    return {
        "mode"       : mode.value,
        "summary"    : summary,
        "folds"      : folds_detail,
        "attribution": attribution,
        "elapsed_sec": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Auto-apply: write winning mode to config/matrix_mode.json
# ---------------------------------------------------------------------------

def apply_winning_mode(
    best: Dict[str, Any],
    all_results: List[Dict[str, Any]],
    symbol: str,
    interval: int,
    months: int,
) -> None:
    """
    Write the winning mode and its key metrics to config/matrix_mode.json.

    argus_bot.py and MatrixEvaluator will read this file on startup (Push 29b)
    so the bot always uses the empirically best mode without manual edits.

    Schema
    ------
    {
      "mode"          : "WEIGHTED_MEAN",          # winning AggregationMode value
      "mean_sharpe"   : 1.234,
      "mean_return_pct": 8.45,
      "mean_win_rate" : 0.54,
      "positive_folds_pct": 80.0,
      "generated_at"  : "2026-04-16T07:51:00Z",
      "backtest_params": {
        "symbol"   : "XBTUSD",
        "interval" : 60,
        "months"   : 6
      },
      "all_modes": [
        {"mode": "WEIGHTED_MEAN", "mean_sharpe": 1.234, ...},
        ...
      ]
    }
    """
    _MODE_CONFIG.parent.mkdir(parents=True, exist_ok=True)

    s = best["summary"]
    payload: Dict[str, Any] = {
        "mode"               : best["mode"],
        "mean_sharpe"        : round(s.get("mean_sharpe", 0.0), 4),
        "mean_return_pct"    : round(s.get("mean_return_pct", 0.0), 4),
        "mean_win_rate"      : round(s.get("mean_win_rate", 0.0), 4),
        "positive_folds_pct" : round(s.get("positive_folds_pct", 0.0), 2),
        "generated_at"       : datetime.now(timezone.utc).isoformat(),
        "backtest_params"    : {
            "symbol"  : symbol,
            "interval": interval,
            "months"  : months,
        },
        "all_modes": [
            {
                "mode"               : r["mode"],
                "mean_sharpe"        : round(r["summary"].get("mean_sharpe", 0.0), 4),
                "mean_return_pct"    : round(r["summary"].get("mean_return_pct", 0.0), 4),
                "mean_win_rate"      : round(r["summary"].get("mean_win_rate", 0.0), 4),
                "positive_folds_pct" : round(r["summary"].get("positive_folds_pct", 0.0), 2),
            }
            for r in all_results
        ],
    }

    with open(_MODE_CONFIG, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(
        "[auto-apply] Winning mode '%s' (Sharpe=%.4f) written to %s",
        best["mode"], s.get("mean_sharpe", 0.0), _MODE_CONFIG,
    )
    print(
        f"\n  [auto-apply] config/matrix_mode.json updated → mode={best['mode']} "
        f"Sharpe={s.get('mean_sharpe', 0.0):+.3f}\n"
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    sep    = "-" * 90
    header = (
        f"{'Mode':<18} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
        f"{'WinRate%':>10} {'PosFolds%':>11} {'Folds':>7} {'Time(s)':>8}"
    )
    print("\n" + sep)
    print("  ARGUS MATRIX MODE COMPARISON — REAL KRAKEN DATA")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        s = r["summary"]
        print(
            f"{r['mode']:<18} "
            f"{s.get('mean_sharpe', 0):>+8.3f} "
            f"{s.get('mean_return_pct', 0):>+9.3f} "
            f"{s.get('mean_max_drawdown_pct', 0):>8.2f} "
            f"{s.get('mean_win_rate', 0)*100:>10.1f} "
            f"{s.get('positive_folds_pct', 0):>11.1f} "
            f"{s.get('num_folds', 0):>7} "
            f"{r['elapsed_sec']:>8.1f}"
        )
    print(sep)
    best = max(results, key=lambda r: r["summary"].get("mean_sharpe", -99))
    print(
        f"\n  BEST MODE (by Sharpe): {best['mode']} "
        f"(Sharpe={best['summary'].get('mean_sharpe', 0):+.3f})"
    )
    print(sep + "\n")


def print_attribution(results: List[Dict[str, Any]]) -> None:
    for r in results:
        print(f"  Attribution — {r['mode']}")
        print(
            f"  {'Tentacle':<22} {'MeanSignal':>12} {'MeanConf':>10} "
            f"{'Buy%':>7} {'Sell%':>7} {'Neutral%':>9}"
        )
        for row in r["attribution"][:5]:
            print(
                f"  {row['tentacle']:<22} "
                f"{row['mean_signal']:>+12.4f} "
                f"{row['mean_confidence']:>10.4f} "
                f"{row['buy_pct']:>7.1f} "
                f"{row['sell_pct']:>7.1f} "
                f"{row['neutral_pct']:>9.1f}"
            )
        print()


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(results: List[Dict[str, Any]], path: str) -> None:
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "modes"       : results,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("JSON results written to %s", path)


def write_csv(results: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode", "mean_sharpe", "mean_return_pct", "mean_max_drawdown_pct",
        "mean_win_rate", "positive_folds_pct", "num_folds", "elapsed_sec",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            s = r["summary"]
            writer.writerow({
                "mode"                  : r["mode"],
                "mean_sharpe"           : round(s.get("mean_sharpe", 0), 4),
                "mean_return_pct"       : round(s.get("mean_return_pct", 0), 4),
                "mean_max_drawdown_pct" : round(s.get("mean_max_drawdown_pct", 0), 4),
                "mean_win_rate"         : round(s.get("mean_win_rate", 0), 4),
                "positive_folds_pct"    : round(s.get("positive_folds_pct", 0), 2),
                "num_folds"             : s.get("num_folds", 0),
                "elapsed_sec"           : r["elapsed_sec"],
            })
    logger.info("CSV results written to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare Argus MatrixEvaluator modes on real Kraken data (Push 29a)"
    )
    p.add_argument("--symbol",      default="XBTUSD")
    p.add_argument("--interval",    default=60,  type=int,
                   help="Candle interval minutes (default: 60)")
    p.add_argument("--months",      default=6,   type=int,
                   help="Months of history (default: 6)")
    p.add_argument("--train-bars",  default=300, type=int)
    p.add_argument("--test-bars",   default=60,  type=int)
    p.add_argument("--obs-window",  default=50,  type=int)
    p.add_argument("--output",      default="results/matrix_comparison",
                   help="Output path prefix (default: results/matrix_comparison)")
    p.add_argument("--modes",       nargs="+",
                   choices=[m.value for m in AggregationMode],
                   default=[m.value for m in MODES])
    p.add_argument(
        "--auto-apply",
        action="store_true",
        default=False,
        help=(
            "After comparison, write the winning mode to config/matrix_mode.json "
            "so argus_bot.py picks it up automatically on next start. "
            "Off by default."
        ),
    )
    return p.parse_args()


async def async_main() -> None:
    args = parse_args()
    selected_modes = [AggregationMode(m) for m in args.modes]

    candles = await fetch_data(
        symbol=args.symbol,
        interval=args.interval,
        months=args.months,
    )

    if len(candles) < args.train_bars + args.test_bars + args.obs_window:
        raise RuntimeError(
            f"Insufficient data: {len(candles)} bars fetched, need at least "
            f"{args.train_bars + args.test_bars + args.obs_window}"
        )

    logger.info(
        "Data ready: %d bars | %.1f days | %s — %s",
        len(candles),
        len(candles) * args.interval / 1440,
        datetime.fromtimestamp(candles[0, 0]).strftime("%Y-%m-%d"),
        datetime.fromtimestamp(candles[-1, 0]).strftime("%Y-%m-%d"),
    )

    results = []
    for mode in selected_modes:
        r = run_mode(
            mode=mode,
            candles=candles,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            obs_window=args.obs_window,
        )
        results.append(r)

    print_comparison_table(results)
    print_attribution(results)

    write_json(results, args.output + ".json")
    write_csv(results,  args.output + ".csv")

    best = max(results, key=lambda r: r["summary"].get("mean_sharpe", -99))

    # --- Push 29a: optionally persist winner ---
    if args.auto_apply:
        apply_winning_mode(
            best=best,
            all_results=results,
            symbol=args.symbol,
            interval=args.interval,
            months=args.months,
        )
    else:
        logger.info(
            "RECOMMENDATION: run with --auto-apply to hardcode '%s' into config/matrix_mode.json",
            best["mode"],
        )

    logger.info(
        "Best mode: %s  Sharpe=%.4f",
        best["mode"], best["summary"].get("mean_sharpe", 0),
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
