"""
Online Strategy Weight Adapter
===============================
Adapts SignalConsensus strategy weights after every trade outcome.
Strategies that have been profitable over recent trades get boosted;
strategies on a losing streak are penalised.

Usage:
    learner = OnlineLearner(strategies=["momentum", "mean_reversion", ...])
    learner.record_trade(strategy_name="momentum", profitable=True, pnl=12.5)
    weights = learner.get_weights()   # dict[str, float]
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    name:         str
    window:       int        = 50     # rolling window for win-rate calc
    min_weight:   float      = 0.20   # never drop below 20% of base
    max_weight:   float      = 3.00   # never exceed 3x base weight
    base_weight:  float      = 1.00

    _outcomes:    Deque[int]   = field(default_factory=lambda: deque(maxlen=50), repr=False)
    _pnls:        Deque[float] = field(default_factory=lambda: deque(maxlen=50), repr=False)
    _weight:      float        = 1.00

    def record(self, profitable: bool, pnl: float = 0.0) -> None:
        self._outcomes.append(int(profitable))
        self._pnls.append(pnl)
        self._recompute_weight()

    def _recompute_weight(self) -> None:
        if len(self._outcomes) < 5:
            return
        n        = len(self._outcomes)
        win_rate = sum(self._outcomes) / n
        avg_pnl  = sum(self._pnls) / n if self._pnls else 0.0

        # Kelly-inspired: f = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
        wins  = [p for p in self._pnls if p > 0]
        losses= [abs(p) for p in self._pnls if p < 0]
        avg_w = sum(wins)   / len(wins)   if wins   else 1.0
        avg_l = sum(losses) / len(losses) if losses else 1.0
        kelly = (win_rate * avg_w - (1.0 - win_rate) * avg_l) / max(1e-10, avg_w)

        # Map kelly [-inf, +inf] to weight [min_weight, max_weight]
        # kelly > 0.5 → max weight, kelly < 0 → min weight
        raw = self.base_weight + kelly * 2.0
        self._weight = max(self.min_weight, min(self.max_weight, raw))

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def win_rate(self) -> float:
        if not self._outcomes:
            return 0.5
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def n_trades(self) -> int:
        return len(self._outcomes)


class OnlineLearner:
    """
    Online strategy weight adapter.

    After each trade close, call record_trade() with the strategy name and
    outcome.  get_weights() returns a dict of normalised weights ready to
    be passed into SignalConsensus.
    """

    def __init__(
        self,
        strategies: List[str],
        window: int = 50,
        save_path: Optional[str] = "models/online_weights.json",
    ):
        self._stats: Dict[str, StrategyStats] = {
            s: StrategyStats(name=s, window=window)
            for s in strategies
        }
        self._save_path = save_path
        self._total_trades = 0
        self._load()

    def record_trade(
        self,
        strategy_name: str,
        profitable: bool,
        pnl: float = 0.0,
    ) -> None:
        """Record outcome for a strategy. Auto-saves every 10 trades."""
        if strategy_name not in self._stats:
            self._stats[strategy_name] = StrategyStats(name=strategy_name)
        self._stats[strategy_name].record(profitable, pnl)
        self._total_trades += 1
        if self._total_trades % 10 == 0:
            self._save()

    def get_weights(self) -> Dict[str, float]:
        """Return normalised strategy weights."""
        raw = {name: s.weight for name, s in self._stats.items()}
        total = sum(raw.values())
        if total < 1e-10:
            return {name: 1.0 for name in raw}
        norm_factor = len(raw) / total   # normalise so mean weight = 1.0
        return {name: w * norm_factor for name, w in raw.items()}

    def get_stats_summary(self) -> Dict[str, Any]:
        """Return per-strategy stats for logging/dashboard."""
        return {
            name: {
                "weight":    round(s.weight, 3),
                "win_rate":  round(s.win_rate, 3),
                "n_trades":  s.n_trades,
            }
            for name, s in self._stats.items()
        }

    def _save(self) -> None:
        if not self._save_path:
            return
        os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)
        data = {
            name: {
                "weight":   s._weight,
                "outcomes": list(s._outcomes),
                "pnls":     list(s._pnls),
            }
            for name, s in self._stats.items()
        }
        with open(self._save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not self._save_path or not os.path.exists(self._save_path):
            return
        try:
            with open(self._save_path) as f:
                data = json.load(f)
            for name, d in data.items():
                if name not in self._stats:
                    self._stats[name] = StrategyStats(name=name)
                s = self._stats[name]
                s._weight = float(d.get("weight", 1.0))
                for o in d.get("outcomes", []):
                    s._outcomes.append(int(o))
                for p in d.get("pnls", []):
                    s._pnls.append(float(p))
            logger.info("OnlineLearner loaded weights from %s", self._save_path)
        except Exception as exc:
            logger.warning("OnlineLearner load failed: %s", exc)
