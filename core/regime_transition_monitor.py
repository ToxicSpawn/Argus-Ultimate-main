"""
RegimeTransitionMonitor — Forward-looking regime risk from HMM transition probabilities.

Calls hmm_regime.predict_transition_probs(returns, horizon_steps) each cycle to get
the probability distribution over future regimes.  Converts this into a scalar
transition_risk_score and a pre_hedge_signal flag that feeds into position sizing.

The transition cost matrix quantifies how *bad* each possible regime transition is:
  any → CRISIS      = 1.0  (worst)
  any → HIGH_VOL    = 0.7
  TRENDING_UP → TRENDING_DOWN = 0.6
  any → RANGING     = 0.3
  benign transitions = 0.1

transition_risk_score = sum over next_regimes of (cost(current → next) × prob(next))
Clamped to [0, 1].

pre_hedge_signal = True when any *adverse* regime (default: HIGH_VOL, CRISIS)
                  has probability > pre_hedge_threshold (default 0.40).

Output: advisory["regime_transition"]
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Default transition cost matrix ────────────────────────────────────────────
# Keys: ("current_regime", "next_regime") → cost float [0, 1]
# Special key ("*", "next") means "any current regime"
_DEFAULT_COSTS: Dict[str, float] = {
    # Catastrophic transitions
    "CRISIS":       1.0,
    "HIGH_VOL":     0.7,
    # Directional regime flip
    "TRENDING_DOWN": 0.6,
    # Neutral/ranging — mild
    "RANGING":      0.3,
    "RANGE":        0.3,
    # Benign: moving to trending up is good for long-biased
    "TRENDING_UP":  0.1,
    "TRENDING":     0.1,
    "BULL":         0.1,
    "BEAR":         0.7,
    # Unknown / catch-all
    "UNKNOWN":      0.4,
}


@dataclass
class RegimeTransitionResult:
    transition_risk_score: float        # 0.0–1.0; weighted over all next-regime probs
    dominant_next_regime: str           # highest-probability next regime label
    pre_hedge_signal: bool              # True when adverse regime prob > threshold
    probs: Dict[str, float]             # full {regime_label: prob} from HMM
    horizon_steps: int
    current_regime: str
    ts: float = field(default_factory=time.time)


_NEUTRAL_RESULT = RegimeTransitionResult(
    transition_risk_score=0.0,
    dominant_next_regime="UNKNOWN",
    pre_hedge_signal=False,
    probs={},
    horizon_steps=0,
    current_regime="UNKNOWN",
)


class RegimeTransitionMonitor:
    """
    Wraps an HMMRegimeDetector to surface forward transition risk.

    Parameters
    ----------
    hmm_detector          : HMMRegimeDetector (must have predict_transition_probs())
    horizon_steps         : look-ahead steps for transition probability query
    pre_hedge_threshold   : adverse-regime prob above this → pre_hedge_signal=True
    adverse_regimes       : list of regime labels considered adverse
    transition_cost_matrix: override cost per next-regime label (dict)
    """

    def __init__(
        self,
        hmm_detector: Any,
        horizon_steps: int = 12,
        pre_hedge_threshold: float = 0.40,
        adverse_regimes: Optional[List[str]] = None,
        transition_cost_matrix: Optional[Dict[str, float]] = None,
    ) -> None:
        self.hmm_detector = hmm_detector
        self.horizon_steps = max(1, int(horizon_steps))
        self.pre_hedge_threshold = float(pre_hedge_threshold)
        self.adverse_regimes: List[str] = adverse_regimes or ["HIGH_VOL", "CRISIS", "BEAR"]
        self._cost_matrix: Dict[str, float] = dict(_DEFAULT_COSTS)
        if transition_cost_matrix:
            self._cost_matrix.update(transition_cost_matrix)

        self._last_result: Optional[RegimeTransitionResult] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        returns: np.ndarray,
        current_regime: str = "UNKNOWN",
    ) -> RegimeTransitionResult:
        """
        Query the HMM for transition probabilities and compute risk score.

        Parameters
        ----------
        returns        : 1-D float array of recent log-returns (at least 5 values)
        current_regime : current regime label (used for logging)
        """
        if self.hmm_detector is None:
            return _NEUTRAL_RESULT

        if returns is None or len(returns) < 5:
            return _NEUTRAL_RESULT

        try:
            # Ask HMM for forward-looking probabilities
            # predict_transition_probs returns Dict[str, float] or None
            raw_probs = self.hmm_detector.predict_transition_probs(
                returns, horizon_steps=self.horizon_steps
            )
        except Exception as exc:
            logger.debug("RegimeTransitionMonitor: hmm predict_transition_probs failed: %s", exc)
            return _NEUTRAL_RESULT

        if not raw_probs or not isinstance(raw_probs, dict):
            return _NEUTRAL_RESULT

        # Normalise probabilities (should sum to ~1 but guard for rounding)
        total = sum(float(v) for v in raw_probs.values() if v is not None)
        if total <= 0:
            return _NEUTRAL_RESULT
        probs = {k: float(v or 0.0) / total for k, v in raw_probs.items()}

        # ── Transition risk score ─────────────────────────────────────────────
        risk_score = 0.0
        for next_regime, prob in probs.items():
            cost = self._get_cost(next_regime)
            risk_score += cost * prob
        risk_score = max(0.0, min(1.0, risk_score))

        # ── Dominant next regime ──────────────────────────────────────────────
        dominant = max(probs, key=probs.get) if probs else "UNKNOWN"

        # ── Pre-hedge signal ──────────────────────────────────────────────────
        pre_hedge = any(
            probs.get(r, 0.0) >= self.pre_hedge_threshold
            for r in self.adverse_regimes
        )

        result = RegimeTransitionResult(
            transition_risk_score=round(risk_score, 4),
            dominant_next_regime=dominant,
            pre_hedge_signal=pre_hedge,
            probs={k: round(v, 4) for k, v in probs.items()},
            horizon_steps=self.horizon_steps,
            current_regime=current_regime,
        )
        self._last_result = result

        if pre_hedge:
            logger.info(
                "RegimeTransitionMonitor: PRE-HEDGE signal — risk=%.3f, dominant=%s, regime=%s",
                risk_score, dominant, current_regime,
            )

        return result

    def snapshot(self) -> Dict[str, Any]:
        r = self._last_result
        if r is None:
            return {
                "transition_risk_score": 0.0,
                "dominant_next_regime": "UNKNOWN",
                "pre_hedge_signal": False,
                "probs": {},
                "horizon_steps": self.horizon_steps,
                "thresholds": {
                    "pre_hedge_threshold": self.pre_hedge_threshold,
                    "adverse_regimes": self.adverse_regimes,
                },
            }
        return {
            "transition_risk_score": r.transition_risk_score,
            "dominant_next_regime":  r.dominant_next_regime,
            "pre_hedge_signal":      r.pre_hedge_signal,
            "probs":                 r.probs,
            "horizon_steps":         r.horizon_steps,
            "current_regime":        r.current_regime,
            "thresholds": {
                "pre_hedge_threshold": self.pre_hedge_threshold,
                "adverse_regimes":     self.adverse_regimes,
            },
            "ts": r.ts,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_cost(self, next_regime: str) -> float:
        """Return transition cost for a given next regime label."""
        # Direct match
        cost = self._cost_matrix.get(str(next_regime).upper())
        if cost is not None:
            return float(cost)
        # Partial match (e.g., "REGIME_HIGH_VOL" contains "HIGH_VOL")
        upper = str(next_regime).upper()
        for key, val in self._cost_matrix.items():
            if key in upper:
                return float(val)
        return 0.4  # default cost for unknown transitions
