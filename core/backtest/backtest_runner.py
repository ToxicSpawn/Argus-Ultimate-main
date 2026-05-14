"""Push 76 — BacktestRunner: orchestrates full backtest pipeline.

Pipeline:
  1. Load OHLCV from CSV (or generate synthetic GBM data)
  2. Run strategy tick-loop to build equity curve
  3. Compute BacktestMetrics
  4. Run WalkForwardEngine
  5. Run MonteCarloSimulator
  6. Save JSON summary + CSV equity curve + CSV trade log

CLI usage:
  python -m core.backtest.backtest_runner \
      --strategy momentum \
      --symbol BTCUSDT \
      --csv data/BTCUSDT_1d.csv \
      --initial-equity 10000 \
      --output reports/
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.backtest.metrics import compute_metrics, BacktestMetrics
from core.backtest.walk_forward import WalkForwardEngine
from core.backtest.monte_carlo import MonteCarloSimulator
from core.strategy.base_strategy import StrategyConfig
from core.strategy.strategy_registry import get_registry


@dataclass
class BacktestConfig:
    strategy_name:    str   = "momentum"
    symbol:           str   = "BTCUSDT"
    initial_equity:   float = 10_000.0
    periods_per_year: int   = 252
    wf_n_splits:      int   = 5
    wf_is_pct:        float = 0.70
    mc_n_simulations: int   = 2_000     # lower default for speed
    mc_ruin_threshold: float = 0.50
    mc_seed:          int   = 42
    output_dir:       str   = "reports"
    strategy_params:  Dict[str, Any] = field(default_factory=dict)


class BacktestRunner:
    """Full backtest pipeline runner.

    Args:
        config: BacktestConfig
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._registry = get_registry()

    def _load_prices_from_csv(self, csv_path: str) -> List[float]:
        prices = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ("close", "Close", "price", "Price"):
                    if col in row:
                        try:
                            prices.append(float(row[col]))
                        except ValueError:
                            pass
                        break
        return prices

    @staticmethod
    def generate_synthetic_prices(
        n: int = 500,
        start: float = 50_000.0,
        mu: float = 0.0003,
        sigma: float = 0.018,
        seed: int = 42,
    ) -> List[float]:
        """Generate GBM price series for testing."""
        rng = random.Random(seed)
        prices = [start]
        for _ in range(n - 1):
            r = mu + sigma * _box_muller(rng)
            prices.append(prices[-1] * math.exp(r))
        return prices

    def _strategy_factory(self):
        cfg = StrategyConfig(
            strategy_id=f"{self.config.strategy_name}_backtest",
            symbol=self.config.symbol,
            initial_equity=self.config.initial_equity,
            params=self.config.strategy_params,
        )
        return self._registry.instantiate(self.config.strategy_name, cfg)

    def run(
        self,
        prices: Optional[List[float]] = None,
        csv_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute full backtest. Returns summary dict."""
        # 1. Load data
        if prices is None:
            if csv_path and Path(csv_path).exists():
                prices = self._load_prices_from_csv(csv_path)
            else:
                prices = self.generate_synthetic_prices()

        if len(prices) < 60:
            raise ValueError(f"Need at least 60 price bars, got {len(prices)}")

        # 2. Full strategy run
        strategy = self._strategy_factory()
        from core.backtest.walk_forward import WalkForwardEngine
        wf_engine = WalkForwardEngine(
            n_splits=1, is_pct=0.0, min_oos_bars=10
        )
        equity_curve, trade_pnls = wf_engine._run_strategy(
            prices, strategy, self.config.initial_equity
        )

        # 3. Metrics
        full_metrics = compute_metrics(
            equity_curve, trade_pnls,
            periods_per_year=self.config.periods_per_year,
            initial_equity=self.config.initial_equity,
        )

        # 4. Walk-forward
        wf_result = None
        if len(prices) >= self.config.wf_n_splits * 60:
            try:
                wf_engine_full = WalkForwardEngine(
                    n_splits=self.config.wf_n_splits,
                    is_pct=self.config.wf_is_pct,
                    periods_per_year=self.config.periods_per_year,
                )
                wf_result = wf_engine_full.run(
                    prices, self._strategy_factory,
                    self.config.initial_equity,
                )
            except Exception as e:
                wf_result = None

        # 5. Monte Carlo
        mc_result = None
        try:
            mc = MonteCarloSimulator(
                n_simulations=self.config.mc_n_simulations,
                ruin_threshold=self.config.mc_ruin_threshold,
                periods_per_year=self.config.periods_per_year,
                seed=self.config.mc_seed,
            )
            mc_result = mc.run(equity_curve, self.config.initial_equity)
        except Exception:
            pass

        # 6. Build summary
        summary: Dict[str, Any] = {
            "strategy":       self.config.strategy_name,
            "symbol":         self.config.symbol,
            "n_bars":         len(prices),
            "initial_equity": self.config.initial_equity,
            "metrics":        full_metrics.to_dict(),
            "walk_forward":   wf_result.to_dict() if wf_result else None,
            "monte_carlo":    mc_result.to_dict()  if mc_result else None,
            "run_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        # 7. Save outputs
        self._save_outputs(summary, equity_curve, trade_pnls)
        return summary

    def _save_outputs(
        self,
        summary: Dict[str, Any],
        equity_curve: List[float],
        trade_pnls: List[float],
    ) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{self.config.strategy_name}_{self.config.symbol}_{ts}"

        # JSON summary
        json_path = out_dir / f"{stem}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # CSV equity curve
        eq_path = out_dir / f"{stem}_equity.csv"
        with open(eq_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bar", "equity"])
            for i, eq in enumerate(equity_curve):
                w.writerow([i, round(eq, 4)])

        # CSV trade log
        tl_path = out_dir / f"{stem}_trades.csv"
        with open(tl_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trade_idx", "pnl"])
            for i, pnl in enumerate(trade_pnls):
                w.writerow([i, round(pnl, 4)])


def _box_muller(rng: random.Random) -> float:
    """Box-Muller transform for normal random variate."""
    import math
    u1 = max(rng.random(), 1e-10)
    u2 = rng.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Argus backtest runner")
    parser.add_argument("--strategy",       default="momentum")
    parser.add_argument("--symbol",         default="BTCUSDT")
    parser.add_argument("--csv",            default=None)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--wf-splits",      type=int,   default=5)
    parser.add_argument("--mc-sims",        type=int,   default=2_000)
    parser.add_argument("--output",         default="reports")
    args = parser.parse_args()

    cfg = BacktestConfig(
        strategy_name=args.strategy,
        symbol=args.symbol,
        initial_equity=args.initial_equity,
        wf_n_splits=args.wf_splits,
        mc_n_simulations=args.mc_sims,
        output_dir=args.output,
    )
    runner = BacktestRunner(cfg)
    result = runner.run(csv_path=args.csv)
    print(json.dumps(result["metrics"], indent=2))
    if result["monte_carlo"]:
        print("\nMonte Carlo:")
        print(json.dumps(result["monte_carlo"], indent=2))
