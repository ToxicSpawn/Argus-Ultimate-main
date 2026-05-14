"""Push 76 — WalkForwardEngine: rolling in-sample / out-of-sample splits.

Two modes:
  anchored: IS window always starts at bar 0 (expanding IS)
  rolling:  IS window slides forward (fixed IS size)

For each split:
  1. Fit / run strategy on IS window (warm-up)
  2. Evaluate strategy on OOS window
  3. Collect OOS BacktestMetrics

Aggregated output:
  WalkForwardResult with per-window metrics + aggregate stats
  WF Efficiency = mean_oos_sharpe / mean_is_sharpe

Usage:
    engine = WalkForwardEngine(n_splits=5, is_pct=0.7)
    result = engine.run(prices, strategy_factory)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from core.backtest.metrics import BacktestMetrics, compute_metrics
from core.strategy.base_strategy import BaseStrategy


@dataclass
class WindowResult:
    window_idx:   int
    is_start:     int
    is_end:       int
    oos_start:    int
    oos_end:      int
    is_metrics:   BacktestMetrics
    oos_metrics:  BacktestMetrics
    oos_equity:   List[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    windows:          List[WindowResult]
    aggregate:        BacktestMetrics
    wf_efficiency:    float    # mean_oos_sharpe / mean_is_sharpe
    mean_oos_sharpe:  float
    mean_is_sharpe:   float
    n_splits:         int
    mode:             str      # "rolling" | "anchored"

    def to_dict(self) -> dict:
        return {
            "n_splits":        self.n_splits,
            "mode":            self.mode,
            "wf_efficiency":   round(self.wf_efficiency, 4),
            "mean_oos_sharpe": round(self.mean_oos_sharpe, 4),
            "mean_is_sharpe":  round(self.mean_is_sharpe, 4),
            "aggregate":       self.aggregate.to_dict(),
            "windows": [
                {
                    "idx":         w.window_idx,
                    "is":          [w.is_start, w.is_end],
                    "oos":         [w.oos_start, w.oos_end],
                    "is_sharpe":   round(w.is_metrics.sharpe, 4),
                    "oos_sharpe":  round(w.oos_metrics.sharpe, 4),
                    "oos_maxdd":   round(w.oos_metrics.max_drawdown_pct, 4),
                    "oos_cagr":    round(w.oos_metrics.cagr_pct, 4),
                }
                for w in self.windows
            ],
        }


StrategyFactory = Callable[[], BaseStrategy]


class WalkForwardEngine:
    """Rolling / anchored walk-forward backtest engine.

    Args:
        n_splits:    Number of IS/OOS windows
        is_pct:      Fraction of each window used as IS (0 < is_pct < 1)
        anchored:    True = anchored (expanding IS), False = rolling
        min_oos_bars: Minimum bars required in OOS window
        periods_per_year: For annualisation
    """

    def __init__(
        self,
        n_splits:         int   = 5,
        is_pct:           float = 0.70,
        anchored:         bool  = False,
        min_oos_bars:     int   = 30,
        periods_per_year: int   = 252,
    ):
        self.n_splits          = n_splits
        self.is_pct            = is_pct
        self.anchored          = anchored
        self.min_oos_bars      = min_oos_bars
        self.periods_per_year  = periods_per_year

    def _make_splits(
        self, n_bars: int
    ) -> List[Tuple[int, int, int, int]]:
        """Return list of (is_start, is_end, oos_start, oos_end)."""
        splits = []
        window_size = n_bars // self.n_splits
        for i in range(self.n_splits):
            if self.anchored:
                is_start = 0
                oos_end  = min((i + 1) * window_size + int(window_size * (1 - self.is_pct)), n_bars)
                oos_start = (i + 1) * window_size - int(window_size * (1 - self.is_pct))
                is_end    = oos_start
            else:
                start     = i * window_size
                is_end    = start + int(window_size * self.is_pct)
                oos_start = is_end
                oos_end   = min(start + window_size, n_bars)
                is_start  = start
            if oos_end - oos_start < self.min_oos_bars:
                continue
            splits.append((is_start, is_end, oos_start, oos_end))
        return splits

    def _run_strategy(
        self,
        prices:   List[float],
        strategy: BaseStrategy,
        initial_equity: float = 10_000.0,
    ) -> Tuple[List[float], List[float]]:
        """Tick strategy through prices. Return (equity_curve, trade_pnls)."""
        equity       = initial_equity
        equity_curve = [equity]
        trade_pnls   = []
        position     = 0.0   # base units held
        entry_price  = 0.0

        strategy.start()
        for price in prices:
            signal = strategy.tick(price)
            if signal is None:
                equity_curve.append(equity_curve[-1])
                continue

            from core.strategy.signal import SignalSide
            if signal.side == SignalSide.LONG and position <= 0:
                # Close short if open
                if position < 0:
                    pnl = position * (entry_price - price)
                    equity += pnl
                    trade_pnls.append(pnl)
                    position = 0.0
                # Open long
                size = strategy.kelly_size(equity, signal.strength, price)
                position    = size
                entry_price = price

            elif signal.side == SignalSide.SHORT and position >= 0:
                if position > 0:
                    pnl = position * (price - entry_price)
                    equity += pnl
                    trade_pnls.append(pnl)
                    position = 0.0
                size = strategy.kelly_size(equity, signal.strength, price)
                position    = -size
                entry_price = price

            elif signal.side == SignalSide.FLAT and position != 0:
                if position > 0:
                    pnl = position * (price - entry_price)
                else:
                    pnl = abs(position) * (entry_price - price)
                equity  += pnl
                trade_pnls.append(pnl)
                position = 0.0

            # Mark-to-market equity
            if position != 0:
                if position > 0:
                    mtm = equity + position * (price - entry_price)
                else:
                    mtm = equity + abs(position) * (entry_price - price)
                equity_curve.append(mtm)
            else:
                equity_curve.append(equity)

        strategy.stop()
        return equity_curve, trade_pnls

    def run(
        self,
        prices:           List[float],
        strategy_factory: StrategyFactory,
        initial_equity:   float = 10_000.0,
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Args:
            prices:           Full price series
            strategy_factory: Callable returning a fresh BaseStrategy
            initial_equity:   Starting equity per window
        """
        splits = self._make_splits(len(prices))
        if not splits:
            raise ValueError(
                f"Not enough bars ({len(prices)}) for {self.n_splits} splits "
                f"with min_oos_bars={self.min_oos_bars}"
            )

        windows: List[WindowResult] = []
        all_oos_equity: List[float] = [initial_equity]
        all_trade_pnls: List[float] = []

        for idx, (is0, is1, oos0, oos1) in enumerate(splits):
            # IS run
            is_strategy = strategy_factory()
            is_equity, _ = self._run_strategy(
                prices[is0:is1], is_strategy, initial_equity
            )
            is_metrics = compute_metrics(
                is_equity, periods_per_year=self.periods_per_year,
                initial_equity=initial_equity,
            )

            # OOS run (fresh strategy, warmed-up on IS first)
            oos_strategy = strategy_factory()
            # Warm up on IS silently
            oos_strategy.start()
            for p in prices[is0:is1]:
                oos_strategy.tick(p)
            oos_strategy.stop()

            oos_equity, oos_pnls = self._run_strategy(
                prices[oos0:oos1], oos_strategy, initial_equity
            )
            oos_metrics = compute_metrics(
                oos_equity, oos_pnls,
                periods_per_year=self.periods_per_year,
                initial_equity=initial_equity,
            )

            windows.append(WindowResult(
                window_idx=idx,
                is_start=is0, is_end=is1,
                oos_start=oos0, oos_end=oos1,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                oos_equity=oos_equity,
            ))
            all_oos_equity.extend(oos_equity[1:])
            all_trade_pnls.extend(oos_pnls)

        # Aggregate
        aggregate = compute_metrics(
            all_oos_equity, all_trade_pnls,
            periods_per_year=self.periods_per_year,
            initial_equity=initial_equity,
        )
        mean_oos_sharpe = sum(w.oos_metrics.sharpe for w in windows) / len(windows)
        mean_is_sharpe  = sum(w.is_metrics.sharpe  for w in windows) / len(windows)
        wf_eff = (mean_oos_sharpe / mean_is_sharpe) if mean_is_sharpe != 0 else 0.0

        return WalkForwardResult(
            windows=windows,
            aggregate=aggregate,
            wf_efficiency=wf_eff,
            mean_oos_sharpe=mean_oos_sharpe,
            mean_is_sharpe=mean_is_sharpe,
            n_splits=len(windows),
            mode="anchored" if self.anchored else "rolling",
        )
