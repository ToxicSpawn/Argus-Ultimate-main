"""Batch 3 — Walk-forward backtester.

Splits a historical DataFrame into anchored in-sample / out-of-sample
folds.  For each fold it calls a user-supplied strategy factory,
fits on in-sample, evaluates on OOS, and aggregates metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WFFoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    sharpe: float
    sortino: float
    calmar: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series = field(default_factory=pd.Series)


@dataclass
class WFSummary:
    folds: List[WFFoldResult]
    mean_oos_sharpe: float
    mean_oos_return: float
    consistency_ratio: float  # fraction of profitable folds


StrategyFactory = Callable[[pd.DataFrame], Callable[[pd.DataFrame], pd.Series]]
# Factory takes train_df → returns a function that takes test_df → pd.Series of positions


class WalkForwardBacktester:
    """Anchored walk-forward analysis engine."""

    def __init__(
        self,
        train_periods: int = 252,
        test_periods: int = 63,
        step_periods: int = 63,
        anchored: bool = True,
        risk_free_rate: float = 0.04,
    ) -> None:
        self._train = train_periods
        self._test = test_periods
        self._step = step_periods
        self._anchored = anchored
        self._rfr = risk_free_rate / 252

    def run(
        self,
        data: pd.DataFrame,
        strategy_factory: StrategyFactory,
        price_col: str = "close",
    ) -> WFSummary:
        """Execute walk-forward analysis.

        Parameters
        ----------
        data : DataFrame indexed by datetime with at least `price_col`.
        strategy_factory : callable(train_df) → callable(test_df) → positions.
        """
        n = len(data)
        folds: List[WFFoldResult] = []
        start = 0 if self._anchored else 0
        fold_idx = 0

        cursor = self._train
        while cursor + self._test <= n:
            train_slice = data.iloc[start:cursor]
            test_slice = data.iloc[cursor : cursor + self._test]

            strategy_fn = strategy_factory(train_slice)
            positions: pd.Series = strategy_fn(test_slice)
            prices: pd.Series = test_slice[price_col]

            result = self._eval_fold(
                fold_idx,
                train_slice,
                test_slice,
                positions,
                prices,
            )
            folds.append(result)
            logger.info(
                "Fold %d OOS Sharpe=%.3f Return=%.2f%%",
                fold_idx,
                result.sharpe,
                result.total_return * 100,
            )

            fold_idx += 1
            cursor += self._step
            if not self._anchored:
                start += self._step

        return self._summarise(folds)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_fold(
        self,
        fold: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        positions: pd.Series,
        prices: pd.Series,
    ) -> WFFoldResult:
        rets = prices.pct_change().fillna(0)
        strat_rets = (positions.shift(1).fillna(0) * rets).fillna(0)
        equity = (1 + strat_rets).cumprod()

        total_return = float(equity.iloc[-1] - 1)
        excess = strat_rets - self._rfr
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
        down = strat_rets[strat_rets < 0]
        sortino = (
            float(excess.mean() / down.std() * np.sqrt(252)) if down.std() > 0 else 0.0
        )
        drawdown = (equity / equity.cummax() - 1).min()
        max_dd = float(drawdown)
        calmar = total_return / abs(max_dd) if max_dd < 0 else 0.0
        wins = (strat_rets > 0).sum()
        num_trades = int((positions.diff().abs() > 0).sum())
        win_rate = float(wins / len(strat_rets)) if len(strat_rets) > 0 else 0.0

        return WFFoldResult(
            fold=fold,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            total_return=total_return,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=num_trades,
            equity_curve=equity,
        )

    @staticmethod
    def _summarise(folds: List[WFFoldResult]) -> WFSummary:
        sharpes = [f.sharpe for f in folds]
        returns = [f.total_return for f in folds]
        profitable = sum(1 for r in returns if r > 0)
        return WFSummary(
            folds=folds,
            mean_oos_sharpe=float(np.mean(sharpes)) if sharpes else 0.0,
            mean_oos_return=float(np.mean(returns)) if returns else 0.0,
            consistency_ratio=float(profitable / len(folds)) if folds else 0.0,
        )
