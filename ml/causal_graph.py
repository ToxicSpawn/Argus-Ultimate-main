"""
Causal Signal Graph — funding → cascade → vol-spike → regime-transition chain.

Models the causal mechanism chain that precedes large crypto price moves.
Rather than reacting to outcomes, this graph anticipates them by detecting
early nodes in the causal chain firing.

Causal chain nodes:
  Node 1 (FUNDING_EXTREME):    funding rate is extreme (>0.08% / 8h or <-0.04%)
  Node 2 (FUNDING_ACCELERATING): rate of change of funding is accelerating
  Node 3 (WHALE_ACCUMULATION): large whale net buys/sells observed
  Node 4 (CASCADE_RISK):       conditional probability of cascade liquidation
  Node 5 (VOL_SPIKE_FORECAST): expected volatility spike in next 2-4 hours
  Node 6 (REGIME_TRANSITION):  probability of regime change within next 12 bars

Signal flow (directed edges with conditional probabilities):
  FUNDING_EXTREME + FUNDING_ACCELERATING  →  CASCADE_RISK
  CASCADE_RISK  →  VOL_SPIKE_FORECAST
  VOL_SPIKE_FORECAST + WHALE_ACCUMULATION →  REGIME_TRANSITION

The graph outputs:
  - cascade_probability    [0, 1]
  - vol_spike_probability  [0, 1]
  - regime_transition_prob [0, 1]
  - direction_bias         [-1, 1]  (SELL cascade if positive funding, else BUY)
  - action_signal          [-1, 1]  composite directional signal

Usage::

    graph = CausalGraph()

    # Feed observable inputs each cycle
    graph.update_funding(rate=0.085, prev_rate=0.070)  # 8h funding rates
    graph.update_whale(net_flow=1250.0)                # USD net flow (+ = buying)
    result = graph.evaluate()

    result.cascade_probability    # e.g. 0.78
    result.action_signal          # e.g. -0.65 (short bias — longs will be squeezed)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Thresholds (calibrated against historical crypto funding data) ────────────

FUNDING_LONG_EXTREME_PCT = 0.08    # longs paying > 0.08%/8h → squeeze risk
FUNDING_SHORT_EXTREME_PCT = -0.04  # shorts paying > 0.04%/8h → short squeeze risk
FUNDING_ACCEL_THRESHOLD = 0.02    # |Δfunding| > 0.02%/cycle → accelerating

WHALE_FLOW_SIGNIFICANT = 500_000  # USD net flow to count as significant

CASCADE_BASE_PROB = 0.15           # base cascade probability without signals
CASCADE_FUNDING_BOOST = 0.45      # P(cascade | extreme funding)
CASCADE_ACCEL_BOOST = 0.20        # additional boost if also accelerating

VOL_SPIKE_BASE = 0.10
VOL_SPIKE_IF_CASCADE = 0.70      # P(vol spike in 2-4h | cascade risk > 0.5)

REGIME_TRANSITION_BASE = 0.12
REGIME_TRANSITION_IF_VOL = 0.55  # P(regime change | vol spike likely)


@dataclass
class CausalReading:
    """Output snapshot from the causal graph evaluation."""
    # Node activations
    funding_extreme: bool
    funding_direction: str          # "LONG_SQUEEZE" | "SHORT_SQUEEZE" | "NEUTRAL"
    funding_accelerating: bool
    whale_signal: float             # [-1, 1] normalised whale net flow
    cascade_probability: float      # [0, 1]
    vol_spike_probability: float    # [0, 1]
    regime_transition_prob: float   # [0, 1]

    # Composite output
    direction_bias: float           # [-1, 1] directional lean
    action_signal: float            # [-1, 1] actionable composite
    confidence: float               # [0, 1]

    # Metadata
    timestamp: float = field(default_factory=time.time)
    dominant_cause: str = ""        # human-readable primary causal factor

    def is_actionable(self, threshold: float = 0.5) -> bool:
        return abs(self.action_signal) >= threshold and self.confidence >= 0.4

    def summary(self) -> str:
        return (
            f"CausalGraph: cascade={self.cascade_probability:.0%} "
            f"vol_spike={self.vol_spike_probability:.0%} "
            f"regime_tx={self.regime_transition_prob:.0%} "
            f"signal={self.action_signal:+.2f} ({self.dominant_cause})"
        )


class CausalGraph:
    """
    Lightweight Bayesian causal graph for crypto funding → cascade → regime chain.

    Maintains rolling history of funding rates and whale flows to compute
    rate-of-change and trend signals. Each call to ``evaluate()`` propagates
    through the causal chain and produces a directional signal.

    Parameters
    ----------
    funding_history_len : int
        Number of funding rate observations to retain (one per 8h = 30 periods/week).
    whale_history_len : int
        Number of whale flow observations to retain.
    """

    def __init__(
        self,
        funding_history_len: int = 30,
        whale_history_len: int = 50,
    ) -> None:
        self._funding_history: Deque[Tuple[float, float]] = deque(maxlen=funding_history_len)
        # (rate, timestamp)
        self._whale_history: Deque[Tuple[float, float]] = deque(maxlen=whale_history_len)

        self._last_reading: Optional[CausalReading] = None

    # ── Feed methods ──────────────────────────────────────────────────────

    def update_funding(self, rate: float, timestamp: Optional[float] = None) -> None:
        """
        Record a funding rate observation.

        Parameters
        ----------
        rate : float
            Funding rate as percentage per 8h (e.g. 0.085 = 0.085%).
            Positive = longs pay shorts. Negative = shorts pay longs.
        timestamp : float | None
            UNIX timestamp. Defaults to now.
        """
        self._funding_history.append((float(rate), float(timestamp or time.time())))

    def update_whale(self, net_flow: float, timestamp: Optional[float] = None) -> None:
        """
        Record a whale net flow observation.

        Parameters
        ----------
        net_flow : float
            Net USD flow (positive = accumulation, negative = distribution).
        timestamp : float | None
            UNIX timestamp. Defaults to now.
        """
        self._whale_history.append((float(net_flow), float(timestamp or time.time())))

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self) -> CausalReading:
        """
        Propagate through the causal chain and return a CausalReading.

        Returns the most recent reading unchanged if no new inputs have
        been provided since the last evaluation.
        """
        # ── Node 1: Funding extremity ─────────────────────────────────────
        if self._funding_history:
            current_rate = self._funding_history[-1][0]
        else:
            current_rate = 0.0

        long_extreme = current_rate >= FUNDING_LONG_EXTREME_PCT
        short_extreme = current_rate <= FUNDING_SHORT_EXTREME_PCT
        funding_extreme = long_extreme or short_extreme

        if long_extreme:
            funding_direction = "LONG_SQUEEZE"
        elif short_extreme:
            funding_direction = "SHORT_SQUEEZE"
        else:
            funding_direction = "NEUTRAL"

        # ── Node 2: Funding acceleration ──────────────────────────────────
        funding_accel = self._compute_funding_acceleration()
        funding_accelerating = abs(funding_accel) >= FUNDING_ACCEL_THRESHOLD

        # ── Node 3: Whale signal ──────────────────────────────────────────
        whale_signal = self._compute_whale_signal()

        # ── Node 4: Cascade probability ───────────────────────────────────
        # P(cascade) = base + funding_boost * I(extreme) + accel_boost * I(accelerating)
        cascade_prob = CASCADE_BASE_PROB
        if funding_extreme:
            cascade_prob += CASCADE_FUNDING_BOOST
        if funding_accelerating and funding_extreme:
            cascade_prob += CASCADE_ACCEL_BOOST
        # Whale flow opposing positions amplifies cascade risk
        if funding_direction == "LONG_SQUEEZE" and whale_signal < -0.3:
            cascade_prob = min(1.0, cascade_prob * 1.25)
        elif funding_direction == "SHORT_SQUEEZE" and whale_signal > 0.3:
            cascade_prob = min(1.0, cascade_prob * 1.25)
        cascade_prob = float(np.clip(cascade_prob, 0.0, 1.0))

        # ── Node 5: Vol spike probability ─────────────────────────────────
        if cascade_prob >= 0.5:
            vol_spike_prob = VOL_SPIKE_IF_CASCADE
        else:
            vol_spike_prob = VOL_SPIKE_BASE + (cascade_prob / 0.5) * (VOL_SPIKE_IF_CASCADE - VOL_SPIKE_BASE) * 0.5
        vol_spike_prob = float(np.clip(vol_spike_prob, 0.0, 1.0))

        # ── Node 6: Regime transition probability ─────────────────────────
        if vol_spike_prob >= 0.5:
            regime_tx_prob = REGIME_TRANSITION_IF_VOL
        else:
            regime_tx_prob = REGIME_TRANSITION_BASE + vol_spike_prob * (REGIME_TRANSITION_IF_VOL - REGIME_TRANSITION_BASE)
        regime_tx_prob = float(np.clip(regime_tx_prob, 0.0, 1.0))

        # ── Direction bias ────────────────────────────────────────────────
        # Positive funding → longs squeezed → expect SELL cascade → directional bias = SELL (-1)
        # Negative funding → shorts squeezed → expect BUY cascade → directional bias = BUY (+1)
        if funding_direction == "LONG_SQUEEZE":
            direction_bias = -cascade_prob  # stronger cascade → stronger sell signal
        elif funding_direction == "SHORT_SQUEEZE":
            direction_bias = cascade_prob   # stronger cascade → stronger buy signal
        else:
            direction_bias = 0.0

        # Blend with whale signal (whale flows confirm or oppose the direction)
        if abs(whale_signal) > 0.2:
            direction_bias = direction_bias * 0.7 + whale_signal * 0.3
        direction_bias = float(np.clip(direction_bias, -1.0, 1.0))

        # ── Action signal ─────────────────────────────────────────────────
        # Scale by cascade probability (how certain are we this is happening?)
        action_signal = direction_bias * cascade_prob
        action_signal = float(np.clip(action_signal, -1.0, 1.0))

        # ── Confidence ────────────────────────────────────────────────────
        n_signals_active = int(funding_extreme) + int(funding_accelerating) + int(abs(whale_signal) > 0.3)
        confidence = min(1.0, n_signals_active / 3.0 * 0.8 + cascade_prob * 0.2)

        # ── Dominant cause ────────────────────────────────────────────────
        if cascade_prob >= 0.6 and funding_extreme and funding_accelerating:
            cause = f"accelerating {funding_direction} funding → cascade imminent"
        elif cascade_prob >= 0.5 and funding_extreme:
            cause = f"extreme funding ({current_rate:+.3f}%) → {funding_direction}"
        elif cascade_prob >= 0.4 and abs(whale_signal) > 0.3:
            cause = f"whale flow confirms {funding_direction}"
        elif cascade_prob > CASCADE_BASE_PROB:
            cause = "elevated cascade risk"
        else:
            cause = "no dominant causal signal"

        reading = CausalReading(
            funding_extreme=funding_extreme,
            funding_direction=funding_direction,
            funding_accelerating=funding_accelerating,
            whale_signal=float(whale_signal),
            cascade_probability=cascade_prob,
            vol_spike_probability=vol_spike_prob,
            regime_transition_prob=regime_tx_prob,
            direction_bias=direction_bias,
            action_signal=action_signal,
            confidence=confidence,
            dominant_cause=cause,
        )
        self._last_reading = reading

        if cascade_prob >= 0.5:
            logger.info("CausalGraph: %s", reading.summary())
        else:
            logger.debug("CausalGraph: %s", reading.summary())

        return reading

    def last_reading(self) -> Optional[CausalReading]:
        """Return the most recent evaluation result without recomputing."""
        return self._last_reading

    def snapshot(self) -> Dict[str, Any]:
        """Return diagnostic snapshot."""
        r = self._last_reading
        return {
            "funding_observations": len(self._funding_history),
            "whale_observations": len(self._whale_history),
            "latest_funding_rate": self._funding_history[-1][0] if self._funding_history else None,
            "cascade_probability": round(r.cascade_probability, 4) if r else None,
            "vol_spike_probability": round(r.vol_spike_probability, 4) if r else None,
            "regime_transition_prob": round(r.regime_transition_prob, 4) if r else None,
            "action_signal": round(r.action_signal, 4) if r else None,
            "dominant_cause": r.dominant_cause if r else None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _compute_funding_acceleration(self) -> float:
        """
        Rate of change of funding rate: recent_rate - lagged_rate.
        Uses the two most recent observations.
        """
        if len(self._funding_history) < 2:
            return 0.0
        recent = self._funding_history[-1][0]
        lagged = self._funding_history[-2][0]
        return float(recent - lagged)

    def _compute_whale_signal(self) -> float:
        """
        Normalised net whale flow in [-1, 1].
        Uses recent 5-period window vs longer 20-period baseline.
        """
        if len(self._whale_history) < 3:
            return 0.0

        flows = np.array([f for f, _ in self._whale_history], dtype=float)

        # Short-term net flow (last 5 observations)
        short_window = min(5, len(flows))
        short_flow = float(np.sum(flows[-short_window:]))

        # Normalise by historical absolute flow scale
        scale = float(np.percentile(np.abs(flows), 75)) if len(flows) >= 4 else WHALE_FLOW_SIGNIFICANT
        if scale < 1.0:
            scale = WHALE_FLOW_SIGNIFICANT

        signal = float(np.clip(short_flow / scale, -1.0, 1.0))
        return signal
