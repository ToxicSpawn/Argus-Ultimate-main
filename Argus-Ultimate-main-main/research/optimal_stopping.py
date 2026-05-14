"""
Optimal Trade Exit Timing — dynamic-programming approach.

Uses backward induction (optimal stopping theory) to determine the best
holding period for a trade given its historical return profile.  The core
idea: at each bar the trader decides whether to *exit now* and collect
the current P&L, or *continue holding* at the cost of one more bar of
risk, discounted by a factor gamma.

Key methods:

  ``compute_optimal_exit``
      Given a vector of returns-after-entry, finds the bar at which
      expected discounted payoff is maximised.

  ``get_exit_distribution``
      Builds a histogram of avg P&L by holding period from historical
      trades.

  ``should_exit_now``
      Real-time decision: exit or hold, given current P&L and bars held.

Pure Python + numpy.  No exchange or config dependencies.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    """Result of an optimal exit computation."""

    optimal_bar: int          # bar at which to exit (0 = exit immediately)
    expected_payoff: float    # expected discounted payoff at optimal bar
    value_function: List[float]  # V(t) for each bar


class OptimalStopper:
    """Optimal trade exit timing using dynamic programming.

    The optimal stopping value function is computed via backward induction:

    .. math::

        V(T) = R(T)  \\quad\\text{(terminal: must exit)}

        V(t) = \\max\\bigl( R(t),\\; \\gamma \\, E[V(t+1)] \\bigr)
        \\quad t = T{-}1, \\ldots, 0

    where R(t) is the cumulative return at bar t and gamma is the
    discount factor (accounts for time-value of capital and risk).

    Parameters
    ----------
    discount_factor : float
        Per-bar discount factor (default 0.99).  Lower values penalise
        holding longer.
    max_bars : int
        Maximum holding period to consider (default 50).
    exit_threshold : float
        Minimum expected-payoff advantage to justify holding.  If the
        continuation value exceeds exit value by less than this, exit
        early (default 0.0005 = 0.05%).
    """

    def __init__(
        self,
        discount_factor: float = 0.99,
        max_bars: int = 50,
        exit_threshold: float = 0.0005,
    ) -> None:
        if not 0.0 < discount_factor <= 1.0:
            raise ValueError(f"discount_factor must be in (0, 1], got {discount_factor}")
        if max_bars < 1:
            raise ValueError(f"max_bars must be >= 1, got {max_bars}")

        self.discount_factor = discount_factor
        self.max_bars = max_bars
        self.exit_threshold = exit_threshold

        # Historical trade data for exit distribution
        self._trade_history: List[List[float]] = []

        logger.info(
            "OptimalStopper initialised: gamma=%.4f max_bars=%d threshold=%.6f",
            self.discount_factor, self.max_bars, self.exit_threshold,
        )

    # ------------------------------------------------------------------
    # Core: backward induction
    # ------------------------------------------------------------------

    def compute_optimal_exit(
        self,
        returns_after_entry: Sequence[float],
        discount_factor: Optional[float] = None,
    ) -> int:
        """Compute the optimal holding period using backward induction.

        Parameters
        ----------
        returns_after_entry : list[float]
            Cumulative return at each bar after trade entry.
            ``returns_after_entry[0]`` is the return after 1 bar,
            ``returns_after_entry[k]`` is the return after k+1 bars.
        discount_factor : float, optional
            Override the instance discount factor.

        Returns
        -------
        int
            Optimal holding period in bars (0-indexed: 0 means exit after
            first bar, etc.).  Returns 0 if the series is empty.
        """
        gamma = discount_factor if discount_factor is not None else self.discount_factor
        r = np.asarray(returns_after_entry, dtype=np.float64)
        n = len(r)

        if n == 0:
            return 0

        # Truncate to max_bars
        if n > self.max_bars:
            r = r[: self.max_bars]
            n = self.max_bars

        # Backward induction: V[t] = max(R[t], gamma * V[t+1])
        v = np.zeros(n, dtype=np.float64)
        v[-1] = r[-1]  # Terminal condition: must exit at last bar

        for t in range(n - 2, -1, -1):
            continuation = gamma * v[t + 1]
            v[t] = max(r[t], continuation)

        # Forward scan: first bar where stopping is optimal
        # (V[t] == R[t] means stopping is at least as good as continuing)
        for t in range(n):
            continuation = gamma * v[t + 1] if t + 1 < n else 0.0
            if r[t] >= continuation - self.exit_threshold:
                logger.debug(
                    "Optimal exit at bar %d: R=%.6f, continuation=%.6f",
                    t, r[t], continuation,
                )
                return t

        return n - 1

    def compute_optimal_exit_detailed(
        self,
        returns_after_entry: Sequence[float],
        discount_factor: Optional[float] = None,
    ) -> ExitDecision:
        """Like :meth:`compute_optimal_exit` but returns the full value function."""
        gamma = discount_factor if discount_factor is not None else self.discount_factor
        r = np.asarray(returns_after_entry, dtype=np.float64)
        n = len(r)

        if n == 0:
            return ExitDecision(optimal_bar=0, expected_payoff=0.0, value_function=[])

        if n > self.max_bars:
            r = r[: self.max_bars]
            n = self.max_bars

        v = np.zeros(n, dtype=np.float64)
        v[-1] = r[-1]

        for t in range(n - 2, -1, -1):
            v[t] = max(r[t], gamma * v[t + 1])

        optimal_bar = 0
        for t in range(n):
            continuation = gamma * v[t + 1] if t + 1 < n else 0.0
            if r[t] >= continuation - self.exit_threshold:
                optimal_bar = t
                break
        else:
            optimal_bar = n - 1

        return ExitDecision(
            optimal_bar=optimal_bar,
            expected_payoff=float(v[optimal_bar]),
            value_function=v.tolist(),
        )

    # ------------------------------------------------------------------
    # Exit distribution from historical trades
    # ------------------------------------------------------------------

    def record_trade(self, returns_path: Sequence[float]) -> None:
        """Record a historical trade's return path for distribution analysis.

        Parameters
        ----------
        returns_path : list[float]
            Cumulative returns at each bar after entry for a single trade.
        """
        self._trade_history.append(list(returns_path))
        logger.debug("Recorded trade path (%d bars)", len(returns_path))

    def get_exit_distribution(
        self,
        strategy: Optional[str] = None,
        lookback: int = 100,
    ) -> Dict[int, float]:
        """Build a histogram of average P&L by holding period.

        Uses the last *lookback* recorded trades.  For each holding period
        ``h`` (in bars), reports the average cumulative return at bar ``h``
        across all trades that lasted at least ``h`` bars.

        Parameters
        ----------
        strategy : str, optional
            Currently unused; reserved for per-strategy filtering.
        lookback : int
            Number of most-recent trades to use.

        Returns
        -------
        dict[int, float]
            Mapping of holding_period → average P&L.
        """
        trades = self._trade_history[-lookback:]
        if not trades:
            logger.warning("No trade history for exit distribution")
            return {}

        # Aggregate returns by holding period
        pnl_by_bar: Dict[int, List[float]] = defaultdict(list)
        for path in trades:
            for bar_idx, ret in enumerate(path):
                pnl_by_bar[bar_idx].append(ret)

        distribution = {
            bar: float(np.mean(returns))
            for bar, returns in sorted(pnl_by_bar.items())
        }

        # Log the peak
        if distribution:
            best_bar = max(distribution, key=distribution.get)  # type: ignore[arg-type]
            logger.info(
                "Exit distribution peak: bar %d → avg P&L %.6f (%d trades)",
                best_bar, distribution[best_bar], len(trades),
            )

        return distribution

    # ------------------------------------------------------------------
    # Real-time exit decision
    # ------------------------------------------------------------------

    def should_exit_now(
        self,
        current_pnl: float,
        bars_held: int,
        max_bars: Optional[int] = None,
    ) -> bool:
        """Decide whether to exit a live trade right now.

        Uses a simple heuristic informed by the historical exit distribution:

        1. If ``bars_held >= max_bars``, always exit (timeout).
        2. If we have historical data, compare ``current_pnl`` to the
           expected P&L at ``bars_held + 1``.  Exit if continuing is
           unlikely to improve.
        3. If no history, use a simple rule: exit if P&L is positive and
           bars_held > max_bars / 2, or if P&L is deeply negative.

        Parameters
        ----------
        current_pnl : float
            Current unrealised P&L (as fractional return).
        bars_held : int
            Number of bars since trade entry.
        max_bars : int, optional
            Override instance max_bars.

        Returns
        -------
        bool
            True if the trade should be exited now.
        """
        mb = max_bars if max_bars is not None else self.max_bars

        # Hard timeout
        if bars_held >= mb:
            logger.debug("Exit: hard timeout at %d bars", bars_held)
            return True

        # Use historical distribution if available
        dist = self.get_exit_distribution(lookback=200)
        if dist and bars_held in dist:
            current_avg = dist.get(bars_held, 0.0)
            next_avg = dist.get(bars_held + 1)

            if next_avg is not None:
                # Discount continuation value
                expected_continuation = self.discount_factor * next_avg
                if current_pnl >= expected_continuation - self.exit_threshold:
                    logger.debug(
                        "Exit: current P&L %.6f >= discounted next %.6f",
                        current_pnl, expected_continuation,
                    )
                    return True
                return False

        # Fallback heuristic
        if current_pnl > 0 and bars_held > mb / 2:
            logger.debug("Exit: positive P&L %.6f past halfway (%d/%d)", current_pnl, bars_held, mb)
            return True
        if current_pnl < -0.03:  # 3% loss
            logger.debug("Exit: deep loss %.6f at bar %d", current_pnl, bars_held)
            return True

        return False
