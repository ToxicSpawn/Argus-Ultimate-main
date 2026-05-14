"""
WalkForwardEngine — walk-forward backtest engine with Kraken fee model.

Splits a price series into anchored in-sample / out-of-sample windows,
runs a user-supplied strategy function on each OOS window, and aggregates
performance metrics across all folds.

Kraken fee model (2026 30-day volume tiers, maker/taker):
  Taker : 0.40% (default for market orders)
  Maker : 0.16% (limit orders that rest)

Usage
-----
    engine = WalkForwardEngine(
        prices=df["close"],
        strategy_fn=my_strategy,   # (prices_slice) -> pd.Series of positions {-1, 0, 1}
        train_bars=500,
        test_bars=100,
        fee_rate=0.0016,           # maker
    )
    results = engine.run()
    print(results.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kraken fee constants
# ---------------------------------------------------------------------------
KRAKEN_TAKER_FEE: float = 0.0040   # 0.40%
KRAKEN_MAKER_FEE: float = 0.0016   # 0.16%


@dataclass
class FoldResult:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    profit_factor: float
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    folds: List[FoldResult] = field(default_factory=list)
    fee_rate: float = KRAKEN_MAKER_FEE
    elapsed_sec: float = 0.0

    def summary(self) -> dict:
        if not self.folds:
            return {"folds": 0}
        returns      = [f.total_return  for f in self.folds]
        sharpes      = [f.sharpe_ratio  for f in self.folds]
        drawdowns    = [f.max_drawdown  for f in self.folds]
        win_rates    = [f.win_rate      for f in self.folds]
        pfs          = [f.profit_factor for f in self.folds]
        return {
            "folds"             : len(self.folds),
            "fee_rate_pct"      : self.fee_rate * 100,
            "mean_return_pct"   : float(np.mean(returns)) * 100,
            "median_return_pct" : float(np.median(returns)) * 100,
            "mean_sharpe"       : float(np.mean(sharpes)),
            "mean_max_drawdown" : float(np.mean(drawdowns)),
            "mean_win_rate"     : float(np.mean(win_rates)),
            "mean_profit_factor": float(np.mean(pfs)),
            "positive_folds_pct": sum(r > 0 for r in returns) / len(returns) * 100,
            "elapsed_sec"       : self.elapsed_sec,
        }


class WalkForwardEngine:
    """
    Anchored walk-forward backtest engine.

    Parameters
    ----------
    prices       : 1-D array-like of close prices (numpy or pandas Series)
    strategy_fn  : callable(prices_window) -> positions array same length as prices_window
                   positions should be -1 (short), 0 (flat), or 1 (long)
    train_bars   : number of bars in the in-sample training window
    test_bars    : number of bars in each out-of-sample test window
    fee_rate     : per-trade fee applied on entry AND exit (default: Kraken maker 0.16%)
    anchored     : if True, training window grows with each fold (anchored WFA)
                   if False, rolling fixed-size window
    """

    def __init__(
        self,
        prices,
        strategy_fn: Callable,
        train_bars: int = 500,
        test_bars: int = 100,
        fee_rate: float = KRAKEN_MAKER_FEE,
        anchored: bool = True,
    ) -> None:
        self._prices = np.asarray(prices, dtype=np.float64)
        self._strategy_fn = strategy_fn
        self._train_bars = train_bars
        self._test_bars = test_bars
        self._fee_rate = fee_rate
        self._anchored = anchored

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> WalkForwardResult:
        t0 = time.time()
        result = WalkForwardResult(fee_rate=self._fee_rate)
        n = len(self._prices)
        fold_idx = 0

        train_start = 0
        test_start  = self._train_bars

        while test_start + self._test_bars <= n:
            train_end = test_start
            test_end  = test_start + self._test_bars

            train_prices = self._prices[train_start:train_end]
            test_prices  = self._prices[test_start:test_end]

            # Run strategy on test window (strategy may use train window for calibration)
            try:
                positions = np.asarray(
                    self._strategy_fn(test_prices, train_prices), dtype=np.float64
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Fold %d strategy error: %s", fold_idx, exc)
                positions = np.zeros(len(test_prices))

            fold_result = self._evaluate(
                fold_index=fold_idx,
                prices=test_prices,
                positions=positions,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            result.folds.append(fold_result)
            logger.debug(
                "Fold %d: return=%.2f%% sharpe=%.2f dd=%.2f%%",
                fold_idx,
                fold_result.total_return * 100,
                fold_result.sharpe_ratio,
                fold_result.max_drawdown * 100,
            )

            fold_idx += 1
            if self._anchored:
                # train_start stays at 0; window grows
                test_start += self._test_bars
            else:
                train_start += self._test_bars
                test_start  += self._test_bars

        result.elapsed_sec = time.time() - t0
        logger.info(
            "WalkForward complete: %d folds in %.2fs", len(result.folds), result.elapsed_sec
        )
        return result

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        fold_index: int,
        prices: np.ndarray,
        positions: np.ndarray,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
    ) -> FoldResult:
        n = len(prices)
        if n < 2:
            return FoldResult(
                fold_index=fold_index,
                train_start=train_start, train_end=train_end,
                test_start=test_start,   test_end=test_end,
                total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                num_trades=0, win_rate=0.0, profit_factor=0.0,
            )

        # Raw price returns
        price_rets = np.diff(prices) / prices[:-1]       # shape (n-1,)
        pos = positions[:n-1]                             # align with returns

        # Gross strategy returns
        gross = pos * price_rets

        # Apply fees on position changes (entry + exit)
        trades = np.diff(np.concatenate([[0], pos]))      # non-zero = trade
        trade_costs = np.abs(trades) * self._fee_rate * 2  # entry + exit
        net = gross - trade_costs

        # Equity curve
        equity = np.cumprod(1 + net)
        equity_list = equity.tolist()

        total_return = float(equity[-1] - 1.0)
        sharpe = self._sharpe(net)
        max_dd = self._max_drawdown(equity)

        # Trade-level stats
        trade_indices = np.where(np.abs(trades) > 0)[0]
        num_trades = int(len(trade_indices))
        win_rate, profit_factor = self._trade_stats(pos, price_rets, trade_costs)

        return FoldResult(
            fold_index=fold_index,
            train_start=train_start, train_end=train_end,
            test_start=test_start,   test_end=test_end,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            equity_curve=equity_list,
        )

    @staticmethod
    def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        mu  = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        if std == 0:
            return 0.0
        return float(mu / std * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        if len(equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / peak
        return float(np.min(dd))

    @staticmethod
    def _trade_stats(
        positions: np.ndarray,
        price_rets: np.ndarray,
        trade_costs: np.ndarray,
    ):
        """Returns (win_rate, profit_factor) across individual position holding periods."""
        gains, losses = [], []
        in_trade = False
        entry_idx = 0
        current_pos = 0.0

        for i in range(len(positions)):
            p = positions[i]
            if not in_trade and p != 0:
                in_trade = True
                entry_idx = i
                current_pos = p
            elif in_trade and (p == 0 or p != current_pos):
                # Close trade
                trade_ret = float(np.sum(positions[entry_idx:i] * price_rets[entry_idx:i])
                                  - np.sum(trade_costs[entry_idx:i]))
                if trade_ret >= 0:
                    gains.append(trade_ret)
                else:
                    losses.append(abs(trade_ret))
                in_trade = False
                if p != 0:
                    in_trade = True
                    entry_idx = i
                    current_pos = p

        total_trades = len(gains) + len(losses)
        win_rate = len(gains) / total_trades if total_trades > 0 else 0.0
        sum_gains  = sum(gains)  if gains  else 0.0
        sum_losses = sum(losses) if losses else 1e-9
        profit_factor = sum_gains / sum_losses
        return win_rate, profit_factor
