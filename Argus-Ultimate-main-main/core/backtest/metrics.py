"""Push 76 — BacktestMetrics: compute standard performance metrics
from a daily returns series.

Metrics computed:
  Sharpe ratio (annualised)
  Sortino ratio (annualised, downside std)
  Calmar ratio (CAGR / max_drawdown)
  Max drawdown % and duration (bars)
  CAGR %
  Total return %
  Win rate
  Profit factor
  Expectancy (avg $ per trade)
  Avg winning / losing trade
  Number of trades

All ratios annualised assuming `periods_per_year` (default 252 for daily).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BacktestMetrics:
    sharpe:            float = 0.0
    sortino:           float = 0.0
    calmar:            float = 0.0
    max_drawdown_pct:  float = 0.0
    max_dd_duration:   int   = 0     # bars in longest drawdown
    cagr_pct:          float = 0.0
    total_return_pct:  float = 0.0
    win_rate:          float = 0.0
    profit_factor:     float = 0.0
    expectancy:        float = 0.0
    avg_win:           float = 0.0
    avg_loss:          float = 0.0
    n_trades:          int   = 0
    n_bars:            int   = 0
    periods_per_year:  int   = 252

    def to_dict(self) -> dict:
        return {
            "sharpe":           round(self.sharpe, 4),
            "sortino":          round(self.sortino, 4),
            "calmar":           round(self.calmar, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "max_dd_duration":  self.max_dd_duration,
            "cagr_pct":         round(self.cagr_pct, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "win_rate":         round(self.win_rate, 4),
            "profit_factor":    round(self.profit_factor, 4),
            "expectancy":       round(self.expectancy, 4),
            "avg_win":          round(self.avg_win, 4),
            "avg_loss":         round(self.avg_loss, 4),
            "n_trades":         self.n_trades,
            "n_bars":           self.n_bars,
        }


def _drawdown_series(equity: List[float]) -> Tuple[List[float], float, int]:
    """Return (drawdown_pct_series, max_dd_pct, max_dd_duration_bars)."""
    peak     = equity[0]
    dd_series = []
    max_dd    = 0.0
    dd_dur    = 0
    max_dur   = 0
    in_dd     = False

    for eq in equity:
        if eq > peak:
            peak  = eq
            if in_dd:
                max_dur = max(max_dur, dd_dur)
            dd_dur = 0
            in_dd  = False
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0.0
        if dd_pct > 0:
            in_dd   = True
            dd_dur += 1
        max_dd = max(max_dd, dd_pct)
        dd_series.append(dd_pct)

    if in_dd:
        max_dur = max(max_dur, dd_dur)
    return dd_series, max_dd, max_dur


def compute_metrics(
    equity_curve: List[float],
    trade_pnls:   Optional[List[float]] = None,
    periods_per_year: int = 252,
    initial_equity: float = 10_000.0,
) -> BacktestMetrics:
    """Compute full BacktestMetrics from equity curve.

    Args:
        equity_curve:     List of equity values (one per bar).
        trade_pnls:       Optional list of per-trade PnL values.
        periods_per_year: 252 (daily), 365 (crypto daily), 8760 (hourly).
        initial_equity:   Starting equity for return calculation.
    """
    if len(equity_curve) < 2:
        return BacktestMetrics(n_bars=len(equity_curve))

    # Daily returns
    returns = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        if equity_curve[i - 1] != 0 else 0.0
        for i in range(1, len(equity_curve))
    ]

    n      = len(returns)
    mean_r = sum(returns) / n
    std_r  = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / n) if n > 1 else 0.0

    # Sharpe
    sharpe = (mean_r / std_r * math.sqrt(periods_per_year)) if std_r > 0 else 0.0

    # Sortino (downside std)
    neg_returns  = [r for r in returns if r < 0]
    down_std     = math.sqrt(sum(r ** 2 for r in neg_returns) / len(neg_returns)) \
                   if neg_returns else 0.0
    sortino      = (mean_r / down_std * math.sqrt(periods_per_year)) if down_std > 0 else 0.0

    # Drawdown
    _, max_dd, max_dd_dur = _drawdown_series(equity_curve)

    # CAGR
    final_equity    = equity_curve[-1]
    total_return    = (final_equity - initial_equity) / initial_equity * 100
    years           = n / periods_per_year
    cagr            = ((final_equity / initial_equity) ** (1 / years) - 1) * 100 \
                      if years > 0 and initial_equity > 0 else 0.0

    # Calmar
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    # Trade stats
    m = BacktestMetrics(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=max_dd,
        max_dd_duration=max_dd_dur,
        cagr_pct=cagr,
        total_return_pct=total_return,
        n_bars=len(equity_curve),
        periods_per_year=periods_per_year,
    )

    if trade_pnls:
        wins  = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        m.n_trades      = len(trade_pnls)
        m.win_rate      = len(wins) / len(trade_pnls)
        m.avg_win       = sum(wins)   / len(wins)   if wins   else 0.0
        m.avg_loss      = sum(losses) / len(losses) if losses else 0.0
        m.profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")
        m.expectancy    = sum(trade_pnls) / len(trade_pnls)

    return m
