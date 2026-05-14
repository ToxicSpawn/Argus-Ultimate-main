"""
Regime-Conditional Backtester — measures strategy performance per market regime.

Breaks backtest results down by:
  - Regime (TREND_UP, TREND_DOWN, RANGING, VOLATILE, CRISIS)
  - Time of day (session)
  - Volatility quartile

Useful for identifying which regimes a strategy excels or degrades in,
and for regime-conditional position sizing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REGIMES = ["TREND_UP", "TREND_DOWN", "RANGING", "VOLATILE", "CRISIS", "UNKNOWN"]


@dataclass
class Trade:
    entry_price: float
    exit_price: float
    side: str          # LONG / SHORT
    size: float        # base currency
    entry_bar: int
    exit_bar: int
    regime: str


@dataclass
class RegimeStats:
    regime: str
    n_trades: int
    win_rate: float
    avg_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    total_pnl: float
    avg_hold_bars: float


@dataclass
class RegimeBacktestResult:
    by_regime: Dict[str, RegimeStats]
    overall: RegimeStats
    regime_distribution: Dict[str, float]   # fraction of time in each regime
    best_regime: str
    worst_regime: str


def _sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / 252
    sigma = float(np.std(excess))
    if sigma < 1e-10:
        return 0.0
    return float(np.mean(excess) / sigma * math.sqrt(252))


def _max_dd(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_stats(regime: str, trades: List[Trade]) -> RegimeStats:
    if not trades:
        return RegimeStats(regime=regime, n_trades=0, win_rate=0.0,
                           avg_return_pct=0.0, sharpe=0.0,
                           max_drawdown_pct=0.0, total_pnl=0.0,
                           avg_hold_bars=0.0)
    returns = []
    pnls = []
    hold_bars = []
    for t in trades:
        if t.side == "LONG":
            ret = (t.exit_price - t.entry_price) / t.entry_price
        else:
            ret = (t.entry_price - t.exit_price) / t.entry_price
        pnl = ret * t.size * t.entry_price
        returns.append(ret)
        pnls.append(pnl)
        hold_bars.append(t.exit_bar - t.entry_bar)

    arr = np.array(returns)
    equity = np.cumprod(1 + arr) * 1000  # normalised equity curve

    wins = sum(1 for r in returns if r > 0)
    return RegimeStats(
        regime=regime,
        n_trades=len(trades),
        win_rate=wins / len(trades),
        avg_return_pct=float(np.mean(arr)) * 100,
        sharpe=_sharpe(arr),
        max_drawdown_pct=_max_dd(equity) * 100,
        total_pnl=sum(pnls),
        avg_hold_bars=float(np.mean(hold_bars)),
    )


class RegimeBacktester:
    """
    Regime-conditional backtest analyser.

    Usage::

        bt = RegimeBacktester()
        # Feed trade records with regime labels
        bt.add_trade(Trade(entry_price=50000, exit_price=51000, side="LONG",
                           size=0.1, entry_bar=0, exit_bar=10, regime="TREND_UP"))
        result = bt.analyse()
        print(result.best_regime)
    """

    def __init__(self) -> None:
        self._trades: List[Trade] = []
        self._regime_bars: Dict[str, int] = {r: 0 for r in REGIMES}
        self._total_bars: int = 0

    def add_trade(self, trade: Trade) -> None:
        self._trades.append(trade)

    def record_bar(self, regime: str) -> None:
        """Record each bar's regime for distribution calculation."""
        self._total_bars += 1
        self._regime_bars[regime if regime in REGIMES else "UNKNOWN"] += 1

    def analyse(self) -> RegimeBacktestResult:
        """Run regime-conditional analysis on all recorded trades."""
        by_regime: Dict[str, RegimeStats] = {}

        for regime in REGIMES:
            regime_trades = [t for t in self._trades if t.regime == regime]
            by_regime[regime] = _compute_stats(regime, regime_trades)

        overall = _compute_stats("OVERALL", self._trades)

        # Distribution
        total_bars = max(1, self._total_bars)
        dist = {r: self._regime_bars.get(r, 0) / total_bars for r in REGIMES}

        # Best / worst by Sharpe (min 5 trades to qualify)
        qualified = {r: s for r, s in by_regime.items() if s.n_trades >= 5}
        if qualified:
            best = max(qualified, key=lambda r: qualified[r].sharpe)
            worst = min(qualified, key=lambda r: qualified[r].sharpe)
        else:
            best = worst = "UNKNOWN"

        return RegimeBacktestResult(
            by_regime=by_regime,
            overall=overall,
            regime_distribution=dist,
            best_regime=best,
            worst_regime=worst,
        )

    def from_trades_and_regimes(
        self,
        price_series: Sequence[float],
        regime_series: Sequence[str],
        trade_signals: Sequence[int],  # +1 long, -1 short, 0 flat
        trade_size: float = 1.0,
    ) -> RegimeBacktestResult:
        """
        Convenience: build trades from parallel series.

        Args:
            price_series: close prices
            regime_series: regime label per bar
            trade_signals: +1/−1/0 per bar
            trade_size: position size in base units
        """
        prices = list(price_series)
        regimes = list(regime_series)
        signals = list(trade_signals)
        n = min(len(prices), len(regimes), len(signals))

        self._trades.clear()
        self._regime_bars = {r: 0 for r in REGIMES}
        self._total_bars = 0

        in_trade = False
        entry_price = 0.0
        entry_bar = 0
        entry_side = "LONG"
        entry_regime = "UNKNOWN"

        for i in range(n):
            reg = regimes[i] if regimes[i] in REGIMES else "UNKNOWN"
            self.record_bar(reg)

            sig = signals[i]
            price = prices[i]

            if not in_trade and sig != 0:
                in_trade = True
                entry_price = price
                entry_bar = i
                entry_side = "LONG" if sig > 0 else "SHORT"
                entry_regime = reg
            elif in_trade and sig == 0:
                self.add_trade(Trade(
                    entry_price=entry_price,
                    exit_price=price,
                    side=entry_side,
                    size=trade_size,
                    entry_bar=entry_bar,
                    exit_bar=i,
                    regime=entry_regime,
                ))
                in_trade = False

        # Close any open trade
        if in_trade and n > 0:
            self.add_trade(Trade(
                entry_price=entry_price,
                exit_price=prices[-1],
                side=entry_side,
                size=trade_size,
                entry_bar=entry_bar,
                exit_bar=n - 1,
                regime=entry_regime,
            ))

        return self.analyse()
