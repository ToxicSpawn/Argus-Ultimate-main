"""
BanditAllocator
===============
Thompson-Sampling multi-armed bandit that adjusts per-strategy
capital allocation based on observed P&L rewards.  Works hand-in-hand
with StrategyRanker — each pull updates Beta(alpha, beta) posteriors
and returns a normalised weight vector for the active strategy set.

Key design choices
------------------
* Reward signal   : sign(pnl) — win (+1) / loss (-1 treated as 0)
* Prior           : Beta(1, 1) — uniform, no warm-up required
* Thompson sample : draw θ_i ~ Beta(α_i, β_i); weight ∝ θ_i
* Decay           : configurable half-life prevents the bandit
                    getting locked into early winners
* Min weight floor: every active strategy gets at least 5 % share
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("argus.bandit_allocator")


@dataclass
class ArmState:
    name: str
    alpha: float = 1.0   # Beta prior successes
    beta: float = 1.0    # Beta prior failures
    total_reward: float = 0.0
    pull_count: int = 0

    def update(self, reward: float) -> None:
        """reward should be in [0, 1] — typically 1 for win, 0 for loss."""
        self.alpha += reward
        self.beta += 1.0 - reward
        self.total_reward += reward
        self.pull_count += 1

    def thompson_sample(self) -> float:
        return random.betavariate(max(0.1, self.alpha), max(0.1, self.beta))

    def decay(self, factor: float) -> None:
        """Shrink counts toward prior — factor in (0, 1]."""
        self.alpha = max(1.0, self.alpha * factor)
        self.beta = max(1.0, self.beta * factor)


class BanditAllocator:
    """
    Thompson-Sampling bandit over strategy arms.

    Usage::

        bandit = BanditAllocator(decay_halflife_trades=100)
        # After each closed trade:
        bandit.update("momentum", pnl=12.5)
        weights = bandit.weights(["momentum", "mean_reversion", "breakout"])
        # weights is a dict {name: float} summing to 1.0
    """

    def __init__(
        self,
        *,
        decay_halflife_trades: int = 100,
        min_weight: float = 0.05,
        n_thompson_samples: int = 1,
    ) -> None:
        self._arms: Dict[str, ArmState] = {}
        self._decay_factor = 0.5 ** (1.0 / max(1, decay_halflife_trades))
        self._min_weight = min_weight
        self._n_samples = n_thompson_samples
        self._global_pulls = 0

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def _ensure(self, name: str) -> ArmState:
        if name not in self._arms:
            self._arms[name] = ArmState(name=name)
        return self._arms[name]

    def update(self, strategy_name: str, pnl: float) -> None:
        """Record a trade outcome.  Positive pnl = reward 1, else 0."""
        arm = self._ensure(strategy_name)
        reward = 1.0 if pnl > 0 else 0.0
        arm.update(reward)
        self._global_pulls += 1

        # Periodic decay of all arms to forget stale performance
        if self._global_pulls % 50 == 0:
            for a in self._arms.values():
                a.decay(self._decay_factor)

    def weights(self, active_names: List[str]) -> Dict[str, float]:
        """
        Return normalised capital weights for *active_names* strategies.
        Names absent from arms get the uniform prior (Beta(1,1)).
        """
        if not active_names:
            return {}

        samples: Dict[str, float] = {}
        for name in active_names:
            arm = self._ensure(name)
            # Average over n_thompson_samples for stability
            samples[name] = sum(
                arm.thompson_sample() for _ in range(self._n_samples)
            ) / self._n_samples

        raw_sum = sum(samples.values())
        if raw_sum <= 0:
            equal = 1.0 / len(active_names)
            return {n: equal for n in active_names}

        # Normalise then apply floor
        n = len(active_names)
        floor = self._min_weight
        weights: Dict[str, float] = {}
        for name, s in samples.items():
            weights[name] = max(floor, s / raw_sum)

        # Re-normalise after floor
        total = sum(weights.values())
        return {n: w / total for n, w in weights.items()}

    def best_strategy(self, active_names: List[str]) -> Optional[str]:
        """Return the arm with the highest current Thompson sample."""
        w = self.weights(active_names)
        if not w:
            return None
        return max(w, key=lambda k: w[k])

    def snapshot(self) -> List[dict]:
        return [
            {
                "name": a.name,
                "alpha": round(a.alpha, 2),
                "beta": round(a.beta, 2),
                "est_win_rate": round(a.alpha / (a.alpha + a.beta), 3),
                "pulls": a.pull_count,
            }
            for a in sorted(self._arms.values(), key=lambda x: x.pull_count, reverse=True)
        ]
