"""Contextual Regime-Aware Thompson Sampling Bandit.

Improvements over the original:
1. Regime-contextual arms — each arm has separate Beta distributions per regime
   (TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOL_CRASH) instead of one global
   distribution. The bandit learns which strategy wins in each specific regime.
2. Alpha/beta exponential decay (half-life ~200 trades) so early history doesn't
   permanently dominate — the bandit stays adaptive to regime shifts.
3. Proportional capital allocation — arms with higher estimated win probability
   get proportionally more capital weight, not just binary selection.
4. Forced exploration floor (epsilon=0.05) to prevent permanent arm starvation.
5. Regime confidence weighting — when regime is uncertain (low confidence),
   falls back to the global (non-regime-specific) distribution.
"""
from __future__ import annotations

import logging
import math
import random
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported regime keys — must match RegimeDetector output
REGIMES = ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL_CRASH", "UNKNOWN")

# Exponential decay factor per update (half-life ≈ 200 trades at 0.9965)
_DECAY = 0.9965

# Minimum pseudo-count to prevent division collapse after heavy decay
_MIN_COUNT = 0.5

# Exploration floor — with this probability pick a random arm regardless
_EPSILON = 0.05


@dataclass
class ArmState:
    """Per-arm, per-regime Beta distribution state."""
    alpha: float = 1.0   # successes + 1 (Beta prior)
    beta: float = 1.0    # failures  + 1 (Beta prior)
    total_pnl: float = 0.0
    n_pulls: int = 0

    @property
    def estimated_win_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Draw one sample from Beta(alpha, beta)."""
        # Use gamma sampling: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
        try:
            g1 = random.gammavariate(max(self.alpha, _MIN_COUNT), 1.0)
            g2 = random.gammavariate(max(self.beta, _MIN_COUNT), 1.0)
            return g1 / (g1 + g2) if (g1 + g2) > 0 else 0.5
        except Exception:
            return self.estimated_win_rate

    def decay(self) -> None:
        """Apply exponential decay to both alpha and beta, preserving ratio."""
        # Shift toward prior (1.0, 1.0) by decaying excess above prior
        self.alpha = max(_MIN_COUNT, 1.0 + (self.alpha - 1.0) * _DECAY)
        self.beta  = max(_MIN_COUNT, 1.0 + (self.beta  - 1.0) * _DECAY)


@dataclass
class BanditArm:
    """One strategy arm with per-regime Beta distributions."""
    name: str
    # Keyed by regime string
    regime_states: Dict[str, ArmState] = field(default_factory=dict)
    # Global state used when regime confidence is low
    global_state: ArmState = field(default_factory=ArmState)

    def _get_state(self, regime: str) -> ArmState:
        if regime not in self.regime_states:
            self.regime_states[regime] = ArmState()
        return self.regime_states[regime]

    def sample(self, regime: str, regime_confidence: float = 1.0) -> float:
        """Sample estimated win probability, blending regime-specific and global."""
        regime_sample = self._get_state(regime).sample()
        global_sample = self.global_state.sample()
        # Blend: high confidence → use regime-specific; low → blend toward global
        return regime_confidence * regime_sample + (1.0 - regime_confidence) * global_sample

    def update(self, regime: str, win: bool, pnl: float) -> None:
        """Record outcome, update both regime-specific and global distributions."""
        # Decay first
        state = self._get_state(regime)
        state.decay()
        self.global_state.decay()

        # Update
        if win:
            state.alpha += 1.0
            self.global_state.alpha += 1.0
        else:
            state.beta += 1.0
            self.global_state.beta += 1.0

        state.total_pnl += pnl
        state.n_pulls += 1
        self.global_state.total_pnl += pnl
        self.global_state.n_pulls += 1

    def get_stats(self, regime: Optional[str] = None) -> dict:
        state = self._get_state(regime) if regime else self.global_state
        return {
            "name": self.name,
            "regime": regime or "global",
            "alpha": round(state.alpha, 3),
            "beta": round(state.beta, 3),
            "estimated_win_rate": round(state.estimated_win_rate, 4),
            "total_pnl": round(state.total_pnl, 4),
            "n_pulls": state.n_pulls,
        }


class ContextualThompsonBandit:
    """
    Contextual Thompson Sampling bandit for strategy selection.

    Usage:
        bandit = ContextualThompsonBandit(["momentum", "mean_reversion", "trend_following"])

        # Select arm for this bar
        arm, weights = bandit.select(regime="TRENDING_UP", regime_confidence=0.85)

        # After trade closes:
        bandit.update(arm_name="momentum", regime="TRENDING_UP", win=True, pnl=120.0)
    """

    def __init__(
        self,
        arm_names: List[str],
        epsilon: float = _EPSILON,
    ) -> None:
        self._arms: Dict[str, BanditArm] = {
            name: BanditArm(name=name) for name in arm_names
        }
        self._epsilon = epsilon
        self._lock = threading.Lock()
        self._total_selections = 0
        self._arm_selection_counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select(
        self,
        regime: str = "UNKNOWN",
        regime_confidence: float = 1.0,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select the best arm via Thompson Sampling.

        Returns:
            (selected_arm_name, capital_weights_dict)
            capital_weights_dict sums to 1.0 and can be used for
            proportional capital allocation across strategies.
        """
        with self._lock:
            regime = regime if regime in REGIMES else "UNKNOWN"

            # Epsilon-greedy exploration floor
            if random.random() < self._epsilon:
                selected = random.choice(list(self._arms.keys()))
                logger.debug("Bandit: epsilon-exploration selected '%s'", selected)
            else:
                # Thompson sample each arm
                samples = {
                    name: arm.sample(regime, regime_confidence)
                    for name, arm in self._arms.items()
                }
                selected = max(samples, key=lambda k: samples[k])

            self._total_selections += 1
            self._arm_selection_counts[selected] += 1

            # Compute proportional capital weights from current win-rate estimates
            weights = self._compute_capital_weights(regime, regime_confidence)

            return selected, weights

    def update(
        self,
        arm_name: str,
        regime: str,
        win: bool,
        pnl: float,
    ) -> None:
        """Record the outcome of a trade and update the arm's distribution."""
        with self._lock:
            regime = regime if regime in REGIMES else "UNKNOWN"
            arm = self._arms.get(arm_name)
            if arm is None:
                logger.warning("Bandit: unknown arm '%s' — skipping update", arm_name)
                return
            arm.update(regime, win, pnl)
            logger.debug(
                "Bandit updated: arm=%s regime=%s win=%s pnl=%.2f "
                "new_wr=%.3f (alpha=%.2f beta=%.2f)",
                arm_name, regime, win, pnl,
                arm._get_state(regime).estimated_win_rate,
                arm._get_state(regime).alpha,
                arm._get_state(regime).beta,
            )

    # ------------------------------------------------------------------
    # Capital weight computation
    # ------------------------------------------------------------------

    def _compute_capital_weights(
        self,
        regime: str,
        regime_confidence: float,
    ) -> Dict[str, float]:
        """
        Compute proportional capital weights based on estimated win rates.
        Arms with higher estimated win probability get more capital.
        Minimum weight floor of 0.05 prevents full starvation.
        """
        raw: Dict[str, float] = {}
        for name, arm in self._arms.items():
            wr = arm.sample(regime, regime_confidence)
            # Boost signal: square to amplify differences
            raw[name] = max(0.05, wr ** 2)

        total = sum(raw.values())
        return {name: v / total for name, v in raw.items()}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_arm_stats(self, regime: Optional[str] = None) -> List[dict]:
        """Get stats for all arms, optionally filtered to a specific regime."""
        with self._lock:
            stats = []
            for name, arm in self._arms.items():
                s = arm.get_stats(regime)
                s["selection_count"] = self._arm_selection_counts.get(name, 0)
                s["selection_pct"] = (
                    self._arm_selection_counts.get(name, 0) / max(1, self._total_selections)
                )
                stats.append(s)
            return sorted(stats, key=lambda x: x["estimated_win_rate"], reverse=True)

    def get_best_arm(self, regime: str = "UNKNOWN") -> str:
        """Return the arm with highest estimated win rate for this regime."""
        with self._lock:
            regime = regime if regime in REGIMES else "UNKNOWN"
            return max(
                self._arms.keys(),
                key=lambda n: self._arms[n]._get_state(regime).estimated_win_rate,
            )

    def get_summary(self) -> dict:
        with self._lock:
            return {
                "total_selections": self._total_selections,
                "arm_counts": dict(self._arm_selection_counts),
                "epsilon": self._epsilon,
                "decay_factor": _DECAY,
                "arms": [arm.get_stats() for arm in self._arms.values()],
            }
