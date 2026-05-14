"""
Regime-Conditional Strategy Rotation — automatically enables/disables
strategies based on the current market regime.

When regime changes (e.g. TREND_UP → RANGING), the rotator adjusts which
strategies are active. Hysteresis prevents flip-flopping on transient
regime changes shorter than 30 minutes.

Strategies not in the current regime map still run at reduced weight (0.3),
so they can capture unexpected opportunities.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default hysteresis: ignore regime changes younger than this (seconds)
_HYSTERESIS_SECONDS = 1800  # 30 minutes

# Reduced weight for strategies NOT matching current regime
_OFF_REGIME_WEIGHT = 0.3

# Max rotation history entries kept in memory
_MAX_HISTORY = 100


@dataclass
class RotationEvent:
    """Record of a single rotation decision."""
    timestamp: float
    old_regime: str
    new_regime: str
    enabled: List[str]
    disabled: List[str]
    confidence: float


class RegimeStrategyRotator:
    """
    Automatically enable/disable strategies based on detected market regime.

    Hysteresis logic: a new regime must persist for at least ``hysteresis_s``
    seconds before strategies are rotated. This prevents whipsaw on noisy
    regime transitions.
    """

    # Regime → strategies that perform well in that regime
    REGIME_STRATEGY_MAP: Dict[str, List[str]] = {
        "TRENDING_UP": ["momentum", "breakout", "peak_alpha"],
        "TREND_UP": ["momentum", "breakout", "peak_alpha"],
        "TRENDING_DOWN": ["mean_reversion", "peak_alpha"],
        "TREND_DOWN": ["mean_reversion", "peak_alpha"],
        "RANGING": ["mean_reversion", "stat_arb_cointegration", "kalman_pairs", "market_maker"],
        "RANGE": ["mean_reversion", "stat_arb_cointegration", "kalman_pairs", "market_maker"],
        "HIGH_VOL": ["volatility_arb", "breakout", "liquidation_cascade"],
        "LOW_VOL": ["mean_reversion", "funding_rate_harvester", "futures_basis_arb"],
        "CRISIS": ["macro_event_filter"],
        "BREAKOUT": ["momentum", "breakout", "peak_alpha"],
        "NORMAL": ["peak_alpha", "mean_reversion", "momentum", "stat_arb_cointegration"],
    }

    def __init__(
        self,
        strategy_router: Any = None,
        hysteresis_s: float = _HYSTERESIS_SECONDS,
        off_regime_weight: float = _OFF_REGIME_WEIGHT,
    ) -> None:
        self._router = strategy_router
        self._hysteresis_s = hysteresis_s
        self._off_regime_weight = off_regime_weight

        # Current state
        self._current_regime: str = "NORMAL"
        self._current_confidence: float = 0.0
        self._regime_set_ts: float = 0.0  # when current regime was first seen

        # Pending regime (waiting for hysteresis to confirm)
        self._pending_regime: Optional[str] = None
        self._pending_confidence: float = 0.0
        self._pending_ts: float = 0.0

        # Last applied rotation
        self._last_rotation: Dict[str, bool] = {}
        self._weights: Dict[str, float] = {}

        # History
        self._history: Deque[RotationEvent] = deque(maxlen=_MAX_HISTORY)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_regime(self, regime: str, confidence: float) -> None:
        """Called each cycle with the latest detected regime."""
        now = time.time()

        if regime == self._current_regime:
            # Same regime — update confidence, clear any pending transition
            self._current_confidence = confidence
            self._pending_regime = None
            self._pending_ts = 0.0
            return

        # Different regime detected
        if regime == self._pending_regime:
            # Already pending — check hysteresis
            self._pending_confidence = confidence
            return

        # New pending regime
        self._pending_regime = regime
        self._pending_confidence = confidence
        self._pending_ts = now

    def rotate(self) -> Dict[str, bool]:
        """
        Enable strategies matching current regime, disable others.

        Returns ``{strategy_name: True/False}`` showing enable state per strategy.
        Applies hysteresis: a pending regime must be at least ``hysteresis_s``
        old before it replaces the current regime.
        """
        now = time.time()

        # Check if pending regime has passed hysteresis
        if (
            self._pending_regime is not None
            and self._pending_regime != self._current_regime
            and (now - self._pending_ts) >= self._hysteresis_s
        ):
            old_regime = self._current_regime
            self._current_regime = self._pending_regime
            self._current_confidence = self._pending_confidence
            self._regime_set_ts = self._pending_ts
            self._pending_regime = None
            self._pending_ts = 0.0

            # Record rotation
            preferred = set(self.REGIME_STRATEGY_MAP.get(self._current_regime, []))
            old_preferred = set(self.REGIME_STRATEGY_MAP.get(old_regime, []))
            newly_enabled = sorted(preferred - old_preferred)
            newly_disabled = sorted(old_preferred - preferred)

            self._history.append(RotationEvent(
                timestamp=now,
                old_regime=old_regime,
                new_regime=self._current_regime,
                enabled=newly_enabled,
                disabled=newly_disabled,
                confidence=self._current_confidence,
            ))

            logger.info(
                "RegimeStrategyRotator: %s → %s (conf=%.2f) — "
                "enabled=[%s] disabled=[%s]",
                old_regime, self._current_regime, self._current_confidence,
                ", ".join(newly_enabled), ", ".join(newly_disabled),
            )

        # Apply current regime to router
        preferred = set(self.REGIME_STRATEGY_MAP.get(self._current_regime, []))
        result: Dict[str, bool] = {}

        if self._router is not None:
            all_strategies = list(getattr(self._router, "_strategies", {}).keys())
            for name in all_strategies:
                should_enable = name in preferred or not preferred
                result[name] = should_enable
                if should_enable:
                    self._router.enable(name)
                else:
                    # Don't fully disable — reduce weight instead
                    # Keep enabled but with reduced weight via get_regime_weights()
                    self._router.enable(name)
        else:
            # No router: just return theoretical map
            for regime_strats in self.REGIME_STRATEGY_MAP.values():
                for s in regime_strats:
                    result[s] = s in preferred

        self._last_rotation = result
        return result

    def get_regime_weights(self) -> Dict[str, float]:
        """
        Return per-strategy weight multiplier based on regime fit.

        Strategies in the current REGIME_STRATEGY_MAP get 1.0.
        All others get ``off_regime_weight`` (default 0.3) — not fully disabled
        so they can catch unexpected opportunities.
        """
        preferred = set(self.REGIME_STRATEGY_MAP.get(self._current_regime, []))
        weights: Dict[str, float] = {}

        if self._router is not None:
            for name in getattr(self._router, "_strategies", {}):
                weights[name] = 1.0 if name in preferred else self._off_regime_weight
        else:
            all_names: set = set()
            for strats in self.REGIME_STRATEGY_MAP.values():
                all_names.update(strats)
            for name in all_names:
                weights[name] = 1.0 if name in preferred else self._off_regime_weight

        self._weights = weights
        return weights

    def get_rotation_history(self) -> List[Dict]:
        """Return last rotation events for debugging."""
        return [
            {
                "timestamp": e.timestamp,
                "old_regime": e.old_regime,
                "new_regime": e.new_regime,
                "enabled": e.enabled,
                "disabled": e.disabled,
                "confidence": e.confidence,
            }
            for e in self._history
        ]

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def current_confidence(self) -> float:
        return self._current_confidence

    @property
    def pending_regime(self) -> Optional[str]:
        return self._pending_regime
