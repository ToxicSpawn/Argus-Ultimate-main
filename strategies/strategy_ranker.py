"""
StrategyRanker
==============
Tracks per-strategy P&L, Sharpe, win-rate, and regime-conditional
performance online. Exposes a ranked list so the BanditAllocator
can weight capital allocation toward the currently best strategies.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StrategyStats:
    name: str
    total_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=200))
    regime_pnl: Dict[str, float] = field(default_factory=dict)
    regime_count: Dict[str, int] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.monotonic)

    # Exponentially-weighted mean / variance for online Sharpe
    _ew_mean: float = 0.0
    _ew_var: float = 1e-9
    _alpha: float = 0.05          # EW decay  (~20-bar half-life)

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5

    @property
    def sharpe(self) -> float:
        """Online EW-Sharpe (annualised for 5-s bars ≈ 1440*365/5 bars/yr)."""
        std = math.sqrt(max(self._ew_var, 1e-12))
        return self._ew_mean / std

    @property
    def score(self) -> float:
        """Composite score balancing Sharpe, win-rate, and recency."""
        sharpe_clamp = max(-3.0, min(3.0, self.sharpe))
        recency_bonus = 1.0 / (1.0 + max(0.0, time.monotonic() - self.last_updated) / 3600.0)
        return 0.50 * sharpe_clamp + 0.30 * (self.win_rate - 0.5) * 2 + 0.20 * recency_bonus

    def record(self, pnl: float, regime: str = "unknown") -> None:
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl >= 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.pnl_history.append(pnl)
        self.regime_pnl[regime] = self.regime_pnl.get(regime, 0.0) + pnl
        self.regime_count[regime] = self.regime_count.get(regime, 0) + 1
        self.last_updated = time.monotonic()

        # EW update
        delta = pnl - self._ew_mean
        self._ew_mean += self._alpha * delta
        self._ew_var = (1 - self._alpha) * (self._ew_var + self._alpha * delta ** 2)


class StrategyRanker:
    """
    Maintains a live leaderboard of strategy performance.

    Usage::

        ranker = StrategyRanker()
        ranker.record_trade("momentum", pnl=12.5, regime="trending")
        top3 = ranker.top_k(3)
    """

    def __init__(self) -> None:
        self._stats: Dict[str, StrategyStats] = {}

    def _ensure(self, name: str) -> StrategyStats:
        if name not in self._stats:
            self._stats[name] = StrategyStats(name=name)
        return self._stats[name]

    def record_trade(self, strategy_name: str, pnl: float, regime: str = "unknown") -> None:
        self._ensure(strategy_name).record(pnl, regime)

    def stats(self, strategy_name: str) -> Optional[StrategyStats]:
        return self._stats.get(strategy_name)

    def all_stats(self) -> Dict[str, StrategyStats]:
        return dict(self._stats)

    def ranked(self) -> List[Tuple[str, float]]:
        """Return [(name, score)] sorted best-first."""
        pairs = [(n, s.score) for n, s in self._stats.items()]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def top_k(self, k: int) -> List[str]:
        return [n for n, _ in self.ranked()[:k]]

    def regime_best(self, regime: str, k: int = 3) -> List[str]:
        """Best strategies within a specific market regime."""
        def regime_score(stats: StrategyStats) -> float:
            cnt = stats.regime_count.get(regime, 0)
            if cnt == 0:
                return -999.0
            avg = stats.regime_pnl.get(regime, 0.0) / cnt
            return avg
        pairs = sorted(
            self._stats.items(),
            key=lambda kv: regime_score(kv[1]),
            reverse=True,
        )
        return [n for n, _ in pairs[:k]]

    def snapshot(self) -> List[dict]:
        """Rich-printable snapshot for dashboard / logging."""
        return [
            {
                "name": s.name,
                "score": round(s.score, 4),
                "sharpe": round(s.sharpe, 3),
                "win_rate": round(s.win_rate, 3),
                "trades": s.trade_count,
                "total_pnl": round(s.total_pnl, 2),
            }
            for s in sorted(self._stats.values(), key=lambda x: x.score, reverse=True)
        ]
