"""Push 76 — MonteCarloSimulator: bootstrap return stream analysis.

Algorithm:
  1. Compute daily returns from equity curve
  2. Resample with replacement N times (bootstrap)
  3. For each simulation: rebuild equity curve, compute Sharpe + drawdown
  4. Aggregate: ruin probability, percentile distributions

Ruin defined as: equity falls below ruin_threshold * initial_equity

Outputs (MonteCarloResult):
  ruin_probability
  median / p5 / p95 final equity
  median / p5 / p95 Sharpe
  median / p5 / p95 max drawdown
  full percentile table (5th, 25th, 50th, 75th, 95th)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.backtest.metrics import BacktestMetrics, compute_metrics


@dataclass
class MonteCarloResult:
    n_simulations:     int
    ruin_probability:  float
    median_final_equity: float
    p5_final_equity:   float
    p95_final_equity:  float
    median_sharpe:     float
    p5_sharpe:         float
    p95_sharpe:        float
    median_max_dd:     float
    p5_max_dd:         float
    p95_max_dd:        float
    percentile_table:  Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_simulations":      self.n_simulations,
            "ruin_probability":   round(self.ruin_probability, 4),
            "final_equity": {
                "p5":    round(self.p5_final_equity, 2),
                "p50":   round(self.median_final_equity, 2),
                "p95":   round(self.p95_final_equity, 2),
            },
            "sharpe": {
                "p5":    round(self.p5_sharpe, 4),
                "p50":   round(self.median_sharpe, 4),
                "p95":   round(self.p95_sharpe, 4),
            },
            "max_drawdown_pct": {
                "p5":    round(self.p5_max_dd, 4),
                "p50":   round(self.median_max_dd, 4),
                "p95":   round(self.p95_max_dd, 4),
            },
        }


def _percentile(values: List[float], p: float) -> float:
    """Linear interpolation percentile."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo  = int(idx)
    hi  = lo + 1
    frac = idx - lo
    if hi >= len(sorted_v):
        return sorted_v[-1]
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulator for return streams.

    Args:
        n_simulations:    Number of bootstrap paths (default 10_000)
        ruin_threshold:   Equity fraction below which ruin is declared (default 0.5)
        periods_per_year: For Sharpe annualisation
        seed:             Optional random seed for reproducibility
    """

    def __init__(
        self,
        n_simulations:    int   = 10_000,
        ruin_threshold:   float = 0.5,
        periods_per_year: int   = 252,
        seed:             Optional[int] = None,
    ):
        self.n_simulations    = n_simulations
        self.ruin_threshold   = ruin_threshold
        self.periods_per_year = periods_per_year
        self._rng = random.Random(seed)

    def run(
        self,
        equity_curve:   List[float],
        initial_equity: float = 10_000.0,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on equity curve."""
        if len(equity_curve) < 2:
            raise ValueError("equity_curve must have at least 2 points")

        returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            if equity_curve[i - 1] != 0 else 0.0
            for i in range(1, len(equity_curve))
        ]
        n_bars = len(returns)
        ruin_level = initial_equity * self.ruin_threshold

        final_equities: List[float] = []
        sharpes:        List[float] = []
        max_dds:        List[float] = []
        ruin_count = 0

        for _ in range(self.n_simulations):
            sampled = [self._rng.choice(returns) for _ in range(n_bars)]
            equity  = initial_equity
            eq_curve = [equity]
            ruined   = False

            for r in sampled:
                equity = equity * (1 + r)
                eq_curve.append(equity)
                if equity <= ruin_level:
                    ruined = True

            if ruined:
                ruin_count += 1

            m = compute_metrics(
                eq_curve,
                periods_per_year=self.periods_per_year,
                initial_equity=initial_equity,
            )
            final_equities.append(eq_curve[-1])
            sharpes.append(m.sharpe)
            max_dds.append(m.max_drawdown_pct)

        pcts = [5, 25, 50, 75, 95]
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            ruin_probability=ruin_count / self.n_simulations,
            median_final_equity=_percentile(final_equities, 50),
            p5_final_equity=_percentile(final_equities, 5),
            p95_final_equity=_percentile(final_equities, 95),
            median_sharpe=_percentile(sharpes, 50),
            p5_sharpe=_percentile(sharpes, 5),
            p95_sharpe=_percentile(sharpes, 95),
            median_max_dd=_percentile(max_dds, 50),
            p5_max_dd=_percentile(max_dds, 5),
            p95_max_dd=_percentile(max_dds, 95),
            percentile_table={
                "final_equity":    {str(p): round(_percentile(final_equities, p), 2) for p in pcts},
                "sharpe":          {str(p): round(_percentile(sharpes, p), 4)         for p in pcts},
                "max_drawdown_pct":{str(p): round(_percentile(max_dds, p), 4)         for p in pcts},
            },
        )
