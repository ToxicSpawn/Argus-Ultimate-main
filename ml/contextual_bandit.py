"""
Contextual Thompson Sampling Bandit — regime-aware strategy capital allocation.

Unlike the flat BanditStrategyAllocator, this bandit conditions strategy
performance estimates on market context (regime, volatility tier, time bucket).
Each strategy × context_bucket pair has its own Beta(alpha, beta) posterior,
so a strategy that performs well in TREND but poorly in RANGE will be allocated
more capital in trending markets without polluting its range-market estimate.

Context features used:
  - regime    : ARGUS regime label (TREND_UP, TREND_DOWN, RANGE, HIGH_VOL, CRISIS)
  - vol_tier  : "low" | "mid" | "high" (percentile-based from recent vol history)
  - time_slot : "asia" | "london" | "us" | "off" (UTC hour buckets)

Usage::

    bandit = ContextualBandit(
        strategy_names=["momentum", "mean_reversion", "arb"],
    )

    # Each cycle — get allocations given current context
    ctx = bandit.make_context(regime="TREND_UP", vol_estimate=0.02, utc_hour=14)
    allocs = bandit.get_allocations(total_capital=1000.0, context=ctx)

    # After trade closes — update posterior for that context
    bandit.record_outcome("momentum", pnl=12.5, context=ctx)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Context helpers ───────────────────────────────────────────────────────────

_KNOWN_REGIMES = {
    "TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL", "CRISIS", "UNKNOWN",
}

_TIME_SLOTS = {
    "asia":   range(0, 7),    # 00-06 UTC
    "london": range(7, 14),   # 07-13 UTC
    "us":     range(14, 21),  # 14-20 UTC
    "off":    range(21, 24),  # 21-23 UTC
}


def _time_slot(utc_hour: int) -> str:
    for slot, hours in _TIME_SLOTS.items():
        if utc_hour in hours:
            return slot
    return "off"


@dataclass(frozen=True)
class BanditContext:
    """Immutable market context used to key the conditional posterior."""
    regime: str
    vol_tier: str    # "low" | "mid" | "high"
    time_slot: str   # "asia" | "london" | "us" | "off"

    def key(self) -> str:
        return f"{self.regime}|{self.vol_tier}|{self.time_slot}"


@dataclass
class StrategyStats:
    """Per-strategy per-context performance accumulator."""
    alpha: float = 1.0   # Beta distribution success param (uninformative prior)
    beta: float = 1.0    # Beta distribution failure param
    total_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.5
        return self.win_count / self.trade_count

    @property
    def mean_pnl(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.total_pnl / self.trade_count

    def sample(self) -> float:
        """Draw a Thompson sample from Beta(alpha, beta)."""
        return float(np.random.beta(self.alpha, self.beta))


class ContextualBandit:
    """
    Regime-aware Thompson Sampling bandit for strategy capital allocation.

    Maintains independent Beta posteriors per (strategy, context_bucket).
    On each allocation call, the current context determines which posterior
    is sampled, so strategies are evaluated relative to the current regime.

    A global posterior is also maintained per strategy as a fallback for
    context buckets with fewer than ``min_context_trades`` observations.

    Parameters
    ----------
    strategy_names : list[str]
        Strategies to track. New strategies auto-register on first outcome.
    exploration_rate : float
        Minimum fractional allocation per strategy (exploration floor).
    min_context_trades : int
        Minimum trades in a context bucket before context-specific posterior
        is trusted over the global posterior.
    vol_history_len : int
        Number of recent vol samples for percentile-based tier computation.
    """

    def __init__(
        self,
        strategy_names: List[str],
        exploration_rate: float = 0.05,
        min_context_trades: int = 10,
        vol_history_len: int = 200,
    ) -> None:
        if not strategy_names:
            raise ValueError("strategy_names must not be empty")
        self._strategy_names: List[str] = list(strategy_names)
        self._exploration_rate = max(0.0, min(0.5, float(exploration_rate)))
        self._min_context_trades = int(min_context_trades)

        # context_key → strategy → StrategyStats
        self._context_stats: Dict[str, Dict[str, StrategyStats]] = {}

        # Global fallback posteriors per strategy
        self._global_stats: Dict[str, StrategyStats] = {
            name: StrategyStats() for name in strategy_names
        }

        # Volatility history for tier computation
        self._vol_history: Deque[float] = deque(maxlen=vol_history_len)

        # Outcome history for diagnostics (ring buffer)
        self._outcome_history: Deque[Dict[str, Any]] = deque(maxlen=5000)

        self._rng = np.random.default_rng(seed=None)

    # ── Context construction ───────────────────────────────────────────────

    def make_context(
        self,
        regime: str,
        vol_estimate: float = 0.0,
        utc_hour: Optional[int] = None,
    ) -> BanditContext:
        """
        Build a BanditContext from current market state.

        Parameters
        ----------
        regime : str
            Current ARGUS regime label.
        vol_estimate : float
            Recent realised volatility (e.g. daily vol fraction, 0.02 = 2%).
        utc_hour : int | None
            Current UTC hour (0-23). Defaults to current system UTC hour.
        """
        # Record vol for percentile computation
        if vol_estimate > 0.0:
            self._vol_history.append(float(vol_estimate))

        # Compute vol tier
        if len(self._vol_history) >= 10:
            arr = np.array(self._vol_history)
            p33 = float(np.percentile(arr, 33))
            p67 = float(np.percentile(arr, 67))
            if vol_estimate <= p33:
                vol_tier = "low"
            elif vol_estimate <= p67:
                vol_tier = "mid"
            else:
                vol_tier = "high"
        else:
            vol_tier = "mid"  # insufficient history

        # Normalise regime
        regime_norm = regime if regime in _KNOWN_REGIMES else "UNKNOWN"

        # Time slot
        if utc_hour is None:
            from datetime import datetime, timezone
            utc_hour = datetime.now(tz=timezone.utc).hour

        return BanditContext(
            regime=regime_norm,
            vol_tier=vol_tier,
            time_slot=_time_slot(int(utc_hour)),
        )

    # ── Recording ─────────────────────────────────────────────────────────

    def record_outcome(
        self,
        strategy: str,
        pnl: float,
        context: Optional[BanditContext] = None,
    ) -> None:
        """
        Update Beta posteriors for the given strategy and context.

        The magnitude of |pnl| is used to scale the update strength,
        capped at 3× the base update to prevent outlier dominance.

        Parameters
        ----------
        strategy : str
            Name of the strategy that generated the trade.
        pnl : float
            Realised P&L of the closed trade (any currency unit, sign matters).
        context : BanditContext | None
            Market context at time of entry. If None, only global posterior updated.
        """
        self._ensure_strategy(strategy)

        magnitude = min(3.0, 1.0 + abs(pnl) / 100.0)
        is_win = pnl > 0

        # Update global posterior
        g = self._global_stats[strategy]
        g.trade_count += 1
        g.total_pnl += float(pnl)
        if is_win:
            g.alpha += magnitude
            g.win_count += 1
        elif pnl < 0:
            g.beta += magnitude

        # Update context-specific posterior
        if context is not None:
            ckey = context.key()
            if ckey not in self._context_stats:
                self._context_stats[ckey] = {}
            ctx_strats = self._context_stats[ckey]
            if strategy not in ctx_strats:
                # Start new context bucket inheriting half the global prior
                ctx_strats[strategy] = StrategyStats(
                    alpha=1.0 + (g.alpha - 1.0) * 0.25,
                    beta=1.0 + (g.beta - 1.0) * 0.25,
                )
            cs = ctx_strats[strategy]
            cs.trade_count += 1
            cs.total_pnl += float(pnl)
            if is_win:
                cs.alpha += magnitude
                cs.win_count += 1
            elif pnl < 0:
                cs.beta += magnitude

        self._outcome_history.append({
            "strategy": strategy,
            "pnl": float(pnl),
            "context_key": context.key() if context else None,
            "timestamp": time.time(),
        })

    # ── Allocation ────────────────────────────────────────────────────────

    def get_allocations(
        self,
        total_capital: float,
        context: Optional[BanditContext] = None,
    ) -> Dict[str, float]:
        """
        Thompson Sampling allocation of ``total_capital`` across strategies.

        When a context is provided and has sufficient observations, the
        context-specific posterior is used; otherwise falls back to the
        global posterior (warm-started with global prior).

        Parameters
        ----------
        total_capital : float
            Total capital to allocate.
        context : BanditContext | None
            Current market context for conditional allocation.

        Returns
        -------
        dict[str, float]
            strategy → allocated capital amount.
        """
        if total_capital <= 0:
            return {name: 0.0 for name in self._strategy_names}

        n = len(self._strategy_names)
        if n == 0:
            return {}

        ckey = context.key() if context is not None else None
        ctx_strats = self._context_stats.get(ckey, {}) if ckey else {}

        samples: Dict[str, float] = {}
        for name in self._strategy_names:
            # Use context posterior if it has enough observations
            if (
                context is not None
                and name in ctx_strats
                and ctx_strats[name].trade_count >= self._min_context_trades
            ):
                stats = ctx_strats[name]
            else:
                stats = self._global_stats.get(name, StrategyStats())
            samples[name] = stats.sample()

        # Normalize to sum to 1, apply exploration floor
        total_sample = sum(samples.values())
        if total_sample <= 0:
            weights = {name: 1.0 / n for name in self._strategy_names}
        else:
            weights = {name: s / total_sample for name, s in samples.items()}

        # Apply exploration floor: every strategy gets at least exploration_rate
        floor = self._exploration_rate / n
        weights = {name: max(floor, w) for name, w in weights.items()}
        total_w = sum(weights.values())
        weights = {name: w / total_w for name, w in weights.items()}

        return {name: weights[name] * total_capital for name in self._strategy_names}

    # ── Advisory ──────────────────────────────────────────────────────────

    def get_rankings(self, context: Optional[BanditContext] = None) -> List[Dict[str, Any]]:
        """
        Return strategies sorted by expected win-rate in the given context.
        Falls back to global stats for low-observation context buckets.
        """
        ckey = context.key() if context is not None else None
        ctx_strats = self._context_stats.get(ckey, {}) if ckey else {}
        rows = []
        for name in self._strategy_names:
            if (
                context is not None
                and name in ctx_strats
                and ctx_strats[name].trade_count >= self._min_context_trades
            ):
                stats = ctx_strats[name]
                source = "context"
            else:
                stats = self._global_stats.get(name, StrategyStats())
                source = "global"
            # Expected value of Beta distribution = alpha / (alpha + beta)
            ev = stats.alpha / (stats.alpha + stats.beta)
            rows.append({
                "strategy": name,
                "expected_win_rate": round(ev, 4),
                "trade_count": stats.trade_count,
                "win_count": stats.win_count,
                "total_pnl": round(stats.total_pnl, 4),
                "posterior_source": source,
            })
        rows.sort(key=lambda r: r["expected_win_rate"], reverse=True)
        return rows

    def should_pause(self, strategy: str, context: Optional[BanditContext] = None) -> bool:
        """
        Return True if the strategy's expected win-rate in this context
        has dropped below 30% (posterior mean < 0.30).
        """
        ckey = context.key() if context is not None else None
        ctx_strats = self._context_stats.get(ckey, {}) if ckey else {}
        if (
            context is not None
            and strategy in ctx_strats
            and ctx_strats[strategy].trade_count >= self._min_context_trades
        ):
            stats = ctx_strats[strategy]
        else:
            stats = self._global_stats.get(strategy, StrategyStats())
        ev = stats.alpha / (stats.alpha + stats.beta)
        return ev < 0.30

    def snapshot(self) -> Dict[str, Any]:
        """Return diagnostic snapshot for monitoring / API dashboard."""
        return {
            "strategy_count": len(self._strategy_names),
            "context_buckets": len(self._context_stats),
            "total_outcomes": len(self._outcome_history),
            "global_stats": {
                name: {
                    "alpha": round(s.alpha, 3),
                    "beta": round(s.beta, 3),
                    "ev": round(s.alpha / (s.alpha + s.beta), 4),
                    "trade_count": s.trade_count,
                    "total_pnl": round(s.total_pnl, 4),
                }
                for name, s in self._global_stats.items()
            },
        }

    # ── Internal ──────────────────────────────────────────────────────────

    def _ensure_strategy(self, strategy: str) -> None:
        if strategy not in self._global_stats:
            self._global_stats[strategy] = StrategyStats()
            if strategy not in self._strategy_names:
                self._strategy_names.append(strategy)
