"""
Evolution-strategy style reward loop (stub). Record reward (PnL or negative IS) and optionally jitter params.

Ref: Stock-Prediction-Models style – reward = PnL or -IS; jitter strategy/engine params; update.
Use as a hook: after each closed trade (or batch), call record_reward(); periodically or on trigger
call suggest_param_jitter() to get small deltas to apply to config/strategy params for exploration.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default: keep last N rewards per (strategy, symbol) for running stats
_DEFAULT_REWARD_HISTORY = 50


class EvolutionStrategyReward:
    """
    Tracks reward (e.g. PnL or negative implementation shortfall) per strategy/symbol.
    Can suggest param jitter for exploration (evolution-strategy style).
    """

    def __init__(self, history_size: int = _DEFAULT_REWARD_HISTORY) -> None:
        self.history_size = max(10, history_size)
        self._rewards: Dict[Tuple[str, str], deque] = {}  # (strategy, symbol) -> deque of floats
        self._total_reward: float = 0.0
        self._n_records: int = 0

    def record_reward(
        self,
        reward: float,
        strategy: str = "default",
        symbol: str = "default",
    ) -> None:
        """Record one reward (e.g. trade PnL in AUD, or negative IS in bps)."""
        key = (str(strategy), str(symbol))
        if key not in self._rewards:
            self._rewards[key] = deque(maxlen=self.history_size)
        self._rewards[key].append(reward)
        self._total_reward += reward
        self._n_records += 1

    def get_avg_reward(self, strategy: Optional[str] = None, symbol: Optional[str] = None) -> Optional[float]:
        """Return average reward for (strategy, symbol) or overall if both None."""
        if strategy is None and symbol is None:
            if self._n_records == 0:
                return None
            return float(self._total_reward / self._n_records)
        key = (str(strategy or "default"), str(symbol or "default"))
        d = self._rewards.get(key)
        if not d or len(d) == 0:
            return None
        return float(sum(d) / len(d))

    def suggest_param_jitter(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        sigma: float = 0.1,
        rng: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Suggest small jitter deltas for params (for evolution-strategy exploration).
        param_bounds: param_name -> (low, high). Returns param_name -> delta in [-sigma*(range), +sigma*(range)].
        """
        import random
        rng = rng or random
        out: Dict[str, float] = {}
        for name, (lo, hi) in param_bounds.items():
            span = float(hi - lo)
            delta = (rng.random() * 2 - 1) * sigma * span
            out[name] = delta
        return out

    def should_explore(self, strategy: str = "default", symbol: str = "default") -> bool:
        """True if recent reward for (strategy, symbol) is negative or missing (suggest exploring params)."""
        avg = self.get_avg_reward(strategy, symbol)
        if avg is None:
            return True
        return float(avg) < 0


def get_reward_tracker(singleton: bool = True) -> EvolutionStrategyReward:
    """Return shared or new EvolutionStrategyReward instance."""
    if singleton:
        if not hasattr(get_reward_tracker, "_instance"):
            setattr(get_reward_tracker, "_instance", EvolutionStrategyReward())
        out: EvolutionStrategyReward = getattr(get_reward_tracker, "_instance")
        return out
    return EvolutionStrategyReward()
