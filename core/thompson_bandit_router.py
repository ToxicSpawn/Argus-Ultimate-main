"""Thompson Sampling multi-armed bandit for ARGUS strategy selection.

This module provides a pure-numpy Thompson Sampling router that decides which
strategies to allocate capital to on a given cycle. Each strategy is modelled
by two conjugate posteriors:

- A **Beta(alpha, beta)** posterior on the win probability (binary outcome).
- A **Normal-Inverse-Gamma**-inspired posterior on the mean P&L (continuous
  outcome) updated via simple online mean/variance estimation and Bayesian
  credibility weighting against a global prior.

Thompson Sampling draws one sample from each posterior and picks the top ``n``
strategies.  Cold strategies receive an exploration bonus so they are not
starved when established strategies already look profitable, and stale
strategies can optionally have their evidence decayed toward the prior so they
have to re-prove themselves after a quiet period.

Designed for small-to-medium strategy counts (typical ARGUS deployment:
10 to 60 strategies) with per-cycle update cost O(n_strategies).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BanditArm dataclass
# ---------------------------------------------------------------------------


@dataclass
class BanditArm:
    """Posterior state for a single strategy arm.

    ``alpha`` and ``beta`` parametrise the Beta posterior on win probability.
    ``mean_pnl``, ``variance_pnl``, and ``n`` track an online P&L estimator.
    ``last_updated`` is a UNIX timestamp used to decay stale arms.
    """

    name: str
    alpha: float = 1.0  # prior successes
    beta: float = 1.0  # prior failures
    mean_pnl: float = 0.0
    variance_pnl: float = 1.0
    n: int = 0
    total_pnl: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # Numerical internals for Welford's online variance.
    _m2: float = 0.0

    def update(self, pnl_aud: float, won: bool) -> None:
        """Fold a new observation into the arm's posterior."""
        if won:
            self.alpha += 1.0
        else:
            self.beta += 1.0

        # Welford's online mean & variance.
        self.n += 1
        delta = pnl_aud - self.mean_pnl
        self.mean_pnl += delta / self.n
        delta2 = pnl_aud - self.mean_pnl
        self._m2 += delta * delta2
        self.variance_pnl = (self._m2 / self.n) if self.n > 1 else 1.0
        self.total_pnl += pnl_aud
        self.last_updated = time.time()

    def win_rate_mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    def decay(self, factor: float) -> None:
        """Pull Beta posterior toward the prior; softens cold historical data."""
        factor = float(np.clip(factor, 0.0, 1.0))
        # Shrink both alpha and beta toward the baseline of 1.
        self.alpha = 1.0 + (self.alpha - 1.0) * factor
        self.beta = 1.0 + (self.beta - 1.0) * factor

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "n": int(self.n),
            "mean_pnl": float(self.mean_pnl),
            "variance_pnl": float(self.variance_pnl),
            "total_pnl": float(self.total_pnl),
            "win_rate_mean": self.win_rate_mean(),
            "last_updated": float(self.last_updated),
        }


# ---------------------------------------------------------------------------
# ThompsonBanditRouter
# ---------------------------------------------------------------------------


class ThompsonBanditRouter:
    """Multi-armed bandit for picking strategies each cycle.

    Arms are registered with :meth:`register_strategy`.  On each cycle call
    :meth:`select_strategies(n)` to get the top-n to activate.  After the
    fills land, call :meth:`record_outcome(name, pnl, won)` to update the
    posteriors.  Arms that have been idle for longer than ``stale_seconds``
    have their Beta posteriors decayed toward the prior by ``decay_factor``.
    """

    def __init__(
        self,
        exploration_bonus: float = 0.15,
        decay_factor: float = 0.97,
        stale_seconds: float = 3600.0,
        seed: Optional[int] = None,
    ) -> None:
        self.exploration_bonus = float(exploration_bonus)
        self.decay_factor = float(decay_factor)
        self.stale_seconds = float(stale_seconds)
        self._rng = np.random.default_rng(seed)
        self._arms: Dict[str, BanditArm] = {}

    # -- registration --------------------------------------------------------

    def register_strategy(self, name: str) -> None:
        if name in self._arms:
            logger.debug("register_strategy: %s already registered", name)
            return
        self._arms[name] = BanditArm(name=name)
        logger.debug("Registered Thompson arm %s", name)

    def _decay_stale(self) -> None:
        if self.stale_seconds <= 0:
            return
        now = time.time()
        for arm in self._arms.values():
            if arm.n == 0:
                continue
            if now - arm.last_updated > self.stale_seconds:
                arm.decay(self.decay_factor)

    # -- sampling ------------------------------------------------------------

    def _sample_beta(self, alpha: float, beta: float) -> float:
        return float(self._rng.beta(alpha, beta))

    def _sample_pnl(self, arm: BanditArm) -> float:
        """Sample from an approximate posterior over mean P&L."""
        if arm.n == 0:
            return 0.0  # prior mean is zero P&L
        sigma2 = max(arm.variance_pnl, 1e-6) / max(arm.n, 1)
        return float(self._rng.normal(arm.mean_pnl, math.sqrt(sigma2)))

    def select_strategies(self, n: int) -> List[Tuple[str, float]]:
        """Return the top ``n`` strategies by combined Thompson score.

        Score combines the Beta draw (win probability) with a normalised P&L
        draw, plus an exploration bonus inversely proportional to ``1 + sqrt(n)``.
        """
        if n <= 0 or not self._arms:
            return []

        self._decay_stale()

        scores: List[Tuple[str, float]] = []
        for name, arm in self._arms.items():
            p_win = self._sample_beta(arm.alpha, arm.beta)
            pnl_sample = self._sample_pnl(arm)
            # Normalise P&L sample against arm's own std; clip to avoid blowups.
            pnl_std = math.sqrt(max(arm.variance_pnl, 1e-6))
            pnl_norm = float(np.clip(pnl_sample / (pnl_std + 1e-6), -3.0, 3.0))
            # Convert normalised value in [-3, 3] to [0, 1] for clean fusion.
            pnl_score = (pnl_norm + 3.0) / 6.0

            # Exploration bonus: shrinks as observations accumulate.
            bonus = self.exploration_bonus / (1.0 + math.sqrt(max(arm.n, 0)))
            total = 0.55 * p_win + 0.35 * pnl_score + bonus
            scores.append((name, float(total)))

        scores.sort(key=lambda kv: kv[1], reverse=True)
        top = scores[: max(1, int(n))]
        return top

    # -- outcome update ------------------------------------------------------

    def record_outcome(self, name: str, pnl_aud: float, won: bool) -> None:
        if name not in self._arms:
            logger.warning("record_outcome: unknown strategy %s (auto-registering)", name)
            self.register_strategy(name)
        arm = self._arms[name]
        arm.update(float(pnl_aud), bool(won))

    # -- Phase W10 off-policy regret update ----------------------------------

    def record_counterfactual_regret(
        self,
        strategy: str,
        regret: float,
        *,
        weight: float = 0.5,
    ) -> None:
        """Fold off-policy counterfactual regret into an arm's posterior.

        This is much more sample-efficient than waiting for real fills. High
        positive regret means "a different action would have earned more" and
        pulls the Beta posterior toward ``beta`` (evidence the arm is worse
        than alternatives). ``weight`` dampens the pseudo-observation to
        prevent off-policy overfitting (counterfactuals are noisier than
        realised fills — typical value 0.3-0.7).

        Parameters
        ----------
        strategy : str
            Name of the strategy whose arm to update.
        regret : float
            Per-decision regret (non-negative).
        weight : float, default 0.5
            Multiplier on the pseudo-observation (0 = ignore, 1 = full).
        """
        if strategy not in self._arms:
            return
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight <= 0.0 or regret <= 0.0:
            return
        arm = self._arms[strategy]
        # Each unit of regret adds a weighted "failure" vote on the Beta arm.
        # Cap per-call contribution so a single outlier can't dominate.
        pseudo_failure = float(np.clip(regret, 0.0, 5.0)) * weight
        arm.beta += pseudo_failure
        arm.last_updated = time.time()
        logger.debug(
            "record_counterfactual_regret: %s regret=%.4f weight=%.2f pseudo=%.4f",
            strategy, regret, weight, pseudo_failure,
        )

    def ingest_regret_batch(
        self,
        regret_records: List[Dict[str, Any]],
        *,
        weight: float = 0.5,
    ) -> int:
        """Batch-ingest ``[{"strategy": str, "regret": float}, ...]``.

        Returns the number of records successfully applied.
        """
        applied = 0
        for rec in regret_records:
            try:
                strat = str(rec.get("strategy", ""))
                reg = float(rec.get("regret", 0.0))
                if strat and reg > 0.0:
                    self.record_counterfactual_regret(strat, reg, weight=weight)
                    applied += 1
            except (TypeError, ValueError):
                continue
        return applied

    # -- inspection ----------------------------------------------------------

    def get_rankings(self) -> List[Tuple[str, float, float]]:
        """Return ``[(name, win_rate_mean, mean_pnl)]`` sorted by win rate."""
        rows = [
            (arm.name, arm.win_rate_mean(), float(arm.mean_pnl))
            for arm in self._arms.values()
        ]
        rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
        return rows

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_arms": len(self._arms),
            "exploration_bonus": float(self.exploration_bonus),
            "decay_factor": float(self.decay_factor),
            "stale_seconds": float(self.stale_seconds),
            "arms": {name: arm.snapshot() for name, arm in self._arms.items()},
            "rankings": [
                {"name": n, "win_rate": w, "mean_pnl": p}
                for (n, w, p) in self.get_rankings()
            ],
        }


__all__ = ["BanditArm", "ThompsonBanditRouter"]
