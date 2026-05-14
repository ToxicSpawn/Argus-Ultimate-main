"""
run_backtest_gate.py

Called by .github/workflows/backtest_check.yml.
Runs a walk-forward backtest on synthetic data (or real data if a
CSV path is provided via BACKTEST_PRICE_CSV env var) and fails with
exit code 1 if the strategy does not meet minimum quality thresholds.

Thresholds (env vars with defaults)
------------------------------------
  BACKTEST_MIN_SHARPE              default: 0.0
  BACKTEST_MIN_POSITIVE_FOLDS_PCT  default: 40.0
  BACKTEST_TRAIN_BARS              default: 300
  BACKTEST_TEST_BARS               default: 60
  BACKTEST_TOTAL_BARS              default: 1200
  BACKTEST_PRICE_CSV               optional: path to CSV with a 'close' column
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backtesting.walk_forward import WalkForwardEngine, KRAKEN_MAKER_FEE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_SHARPE = float(os.getenv("BACKTEST_MIN_SHARPE", "0.0"))
MIN_POS_FOLDS = float(os.getenv("BACKTEST_MIN_POSITIVE_FOLDS_PCT", "40.0"))
TRAIN_BARS = int(os.getenv("BACKTEST_TRAIN_BARS", "300"))
TEST_BARS = int(os.getenv("BACKTEST_TEST_BARS", "60"))
TOTAL_BARS = int(os.getenv("BACKTEST_TOTAL_BARS", "1200"))
PRICE_CSV = os.getenv("BACKTEST_PRICE_CSV", "")
REPORT_PATH = "backtest_report.json"


# ---------------------------------------------------------------------------
# Price loader
# ---------------------------------------------------------------------------

def load_prices() -> np.ndarray:
    if PRICE_CSV and Path(PRICE_CSV).exists():
        try:
            import pandas as pd
            df = pd.read_csv(PRICE_CSV)
            col = next((c for c in df.columns if c.lower() == "close"), df.columns[-1])
            prices = df[col].dropna().to_numpy(dtype=np.float64)
            logger.info("Loaded %d bars from %s (column: %s)", len(prices), PRICE_CSV, col)
            return prices
        except Exception as exc:  # noqa: BLE001
            logger.warning("CSV load failed (%s); falling back to synthetic data", exc)

    logger.info("Using synthetic random-walk prices (%d bars)", TOTAL_BARS)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, TOTAL_BARS)
    return 30_000.0 * np.cumprod(1 + returns)


# ---------------------------------------------------------------------------
# Reference strategy (EMA crossover — same as optimise_params.py default)
# ---------------------------------------------------------------------------

def reference_strategy(test_prices: np.ndarray, _train: np.ndarray) -> np.ndarray:
    """Simple EMA-20/50 crossover used as the gate benchmark."""
    def ema(arr: np.ndarray, period: int) -> np.ndarray:
        out = np.zeros_like(arr)
        k = 2.0 / (period + 1)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
        return out

    fast = ema(test_prices, 20)
    slow = ema(test_prices, 50)
    pos = np.zeros(len(test_prices))
    for i in range(50, len(test_prices)):
        if fast[i] > slow[i]:
            pos[i] = 1.0
        elif fast[i] < slow[i]:
            pos[i] = -1.0
    return pos


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def main() -> None:
    prices = load_prices()

    engine = WalkForwardEngine(
        prices=prices,
        strategy_fn=reference_strategy,
        train_bars=TRAIN_BARS,
        test_bars=TEST_BARS,
        fee_rate=KRAKEN_MAKER_FEE,
        anchored=True,
    )

    result = engine.run()
    summary = result.summary()

    logger.info("Backtest summary: %s", json.dumps(summary, indent=2))

    # Persist report for artifact upload
    with open(REPORT_PATH, "w") as fh:
        json.dump({
            "summary": summary,
            "thresholds": {
                "min_sharpe": MIN_SHARPE,
                "min_positive_folds_pct": MIN_POS_FOLDS,
            },
            "folds": [
                {
                    "fold": f.fold_index,
                    "return_pct": round(f.total_return * 100, 4),
                    "sharpe": round(f.sharpe_ratio, 4),
                    "max_drawdown_pct": round(f.max_drawdown * 100, 4),
                    "win_rate": round(f.win_rate, 4),
                    "num_trades": f.num_trades,
                }
                for f in result.folds
            ],
        }, fh, indent=2)
    logger.info("Report written to %s", REPORT_PATH)

    # Gate checks
    passed = True
    mean_sharpe = summary.get("mean_sharpe", 0.0)
    pos_folds   = summary.get("positive_folds_pct", 0.0)

    if mean_sharpe < MIN_SHARPE:
        logger.error(
            "GATE FAILED: mean_sharpe=%.4f < threshold=%.4f",
            mean_sharpe, MIN_SHARPE,
        )
        passed = False

    if pos_folds < MIN_POS_FOLDS:
        logger.error(
            "GATE FAILED: positive_folds_pct=%.1f%% < threshold=%.1f%%",
            pos_folds, MIN_POS_FOLDS,
        )
        passed = False

    if passed:
        logger.info(
            "GATE PASSED ✔  mean_sharpe=%.4f  positive_folds=%.1f%%",
            mean_sharpe, pos_folds,
        )
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
