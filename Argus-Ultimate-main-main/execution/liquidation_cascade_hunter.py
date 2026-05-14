"""
Liquidation Cascade Hunter - Ultimate Edge Module

PROFITS from liquidation cascades by detecting and trading:
- Cascading liquidations before they happen
- Leverage ratio extremes
- Funding rate spikes
- Open interest deleveraging
- Whale liquidations

This module turns market crashes into profit opportunities.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CascadePhase(str, Enum):
    """Phase of liquidation cascade."""
    BUILDING = "building"  # Pressure building
    TRIGGERING = "triggering"  # Cascade starting
    ACCELERATING = "accelerating"  # Cascade in full force
    PEAK = "peak"  # Maximum liquidation
    REVERSING = "reversing"  # Bounce starting
    RECOVERY = "recovery"  # Recovery phase


@dataclass
class LiquidationLevel:
    """Liquidation cluster level."""
    price: float
    estimated_liquidation_volume: float
    severity: float
    is_whale_liquidation: bool


@dataclass
class CascadeSignal:
    """Cascade trading signal."""
    action: str
    confidence: float
    phase: CascadePhase
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    reasons: List[str]
    estimated_liquidation_volume: float
    expected_move_pct: float
    holding_period_minutes: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CascadeMetrics:
    """Current cascade metrics."""
    leverage_ratio: float
    funding_rate: float
    open_interest: float
    cascade_probability: float
    estimated_liquidation_volume: float
    cascade_phase: CascadePhase
    direction: str
    time_to_cascade: Optional[str]


class LiquidationCascadeHunter:
    """
    Liquidation Cascade Hunter.

    Detects and profits from liquidation cascades by:
    - Monitoring leverage ratios across exchanges
    - Tracking funding rate spikes
    - Detecting open interest deleveraging
    - Identifying whale liquidation clusters
    - Trading the cascade direction

    Key insight: Liquidations cause cascading sells, which creates
    oversold conditions that bounce. Trade WITH the initial cascade,
    then fade it for the bounce.
    """

    def __init__(
        self,
        leverage_threshold: float = 10.0,
        funding_rate_threshold: float = 0.01,
        volume_spike_multiplier: float = 3.0,
        cascade_confidence: float = 0.65,
    ):
        self.leverage_threshold = leverage_threshold
        self.funding_threshold = funding_rate_threshold
        self.volume_spike = volume_spike_multiplier
        self.confidence = cascade_confidence

        # Price tracking
        self._prices: Deque[float] = deque(maxlen=100)
        self._volumes: Deque[float] = deque(maxlen=100)
        self._leverage_ratios: Deque[float] = deque(maxlen=50)
        self._funding_rates: Deque[float] = deque(maxlen=50)
        self._open_interests: Deque[float] = deque(maxlen=50)

        # Liquidation levels
        self._liquidation_levels: List[LiquidationLevel] = []
        self._whale_liquidations: Deque[Dict] = deque(maxlen=20)

        # Cascade state
        self._cascade_in_progress = False
        self._cascade_phase = CascadePhase.BUILDING
        self._cascade_direction = "none"
        self._cascade_start_price: Optional[float] = None
        self._cascade_start_time: Optional[datetime] = None

    def update(
        self,
        price: float,
        volume: float,
        leverage_ratio: Optional[float] = None,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        exchange: str = "unknown",
    ) -> Optional[CascadeSignal]:
        """
        Update with new market data.

        Args:
            price: Current price
            volume: Current volume
            leverage_ratio: System-wide leverage ratio
            funding_rate: Current funding rate
            open_interest: Open interest
            exchange: Exchange name

        Returns:
            CascadeSignal if cascade detected
        """
        self._prices.append(price)
        self._volumes.append(volume)

        if leverage_ratio is not None:
            self._leverage_ratios.append(leverage_ratio)

        if funding_rate is not None:
            self._funding_rates.append(funding_rate)

        if open_interest is not None:
            self._open_interests.append(open_interest)

        # Check for cascade conditions
        cascade = self._check_cascade_conditions()

        if cascade:
            return self._generate_cascade_signal(price)

        return None

    def add_liquidation_level(
        self,
        price: float,
        volume: float,
        is_whale: bool = False,
    ) -> None:
        """Add a liquidation level to track."""
        severity = min(1.0, volume / 1000000)  # Normalize

        level = LiquidationLevel(
            price=price,
            estimated_liquidation_volume=volume,
            severity=severity,
            is_whale_liquidation=is_whale,
        )
        self._liquidation_levels.append(level)

        if is_whale:
            self._whale_liquidations.append({
                "price": price,
                "volume": volume,
                "timestamp": datetime.now(),
            })

        self._liquidation_levels.sort(key=lambda x: x.severity, reverse=True)

    def _check_cascade_conditions(self) -> bool:
        """Check if cascade conditions are met."""
        if len(self._prices) < 20:
            return False

        metrics = self._get_cascade_metrics()

        # Count conditions met
        conditions = []

        # High leverage
        if len(self._leverage_ratios) >= 5:
            avg_leverage = np.mean(list(self._leverage_ratios)[-5:])
            if avg_leverage > self.leverage_threshold:
                conditions.append(("leverage", 0.8))

        # High funding rate (bear funding = shorts paying longs = bearish)
        if len(self._funding_rates) >= 3:
            recent_funding = list(self._funding_ratios)[-3:] if len(self._funding_ratios) >= 3 else list(self._funding_ratios)
            avg_funding = np.mean(recent_funding)
            if avg_funding > self.funding_threshold:
                conditions.append(("funding", 0.7))
            elif avg_funding < -self.funding_threshold:
                conditions.append(("funding", 0.7))  # Bullish funding

        # Volume spike
        if len(self._volumes) >= 20:
            recent_vol = np.mean(list(self._volumes)[-5:])
            avg_vol = np.mean(list(self._volumes)[-20:])
            if recent_vol > avg_vol * self.volume_spike:
                conditions.append(("volume", 0.75))

        # Open interest deleveraging
        if len(self._open_interests) >= 5:
            recent_oi = np.mean(list(self._open_interests)[-3:])
            earlier_oi = np.mean(list(self._open_interests)[-10:-5])
            if earlier_oi > 0 and recent_oi < earlier_oi * 0.9:
                conditions.append(("oi_deleverage", 0.85))

        # Price drop acceleration
        if len(self._prices) >= 10:
            recent_returns = np.diff(list(self._prices)[-5:]) / list(self._prices)[-6:-1]
            earlier_returns = np.diff(list(self._prices)[-15:-10]) / list(self._prices)[-16:-11]

            if np.std(recent_returns) > np.std(earlier_returns) * 2:
                conditions.append(("volatility_clustering", 0.7))

        # Whale liquidation detected
        if len(self._whale_liquidations) >= 3:
            recent_whales = list(self._whale_liquidations)[-3:]
            if all(datetime.now() - w["timestamp"] < timedelta(minutes=30) for w in recent_whales):
                conditions.append(("whale_liquidations", 0.9))

        # Calculate cascade probability
        if not conditions:
            self._cascade_in_progress = False
            return False

        total_weight = sum(c[1] for c in conditions)
        cascade_prob = min(0.95, total_weight / len(conditions) * 1.5)

        if cascade_prob >= self.confidence:
            self._cascade_in_progress = True
            self._cascade_direction = self._determine_cascade_direction()
            self._cascade_phase = self._determine_cascade_phase()
            return True

        return False

    def _get_cascade_metrics(self) -> CascadeMetrics:
        """Get current cascade metrics."""
        if len(self._leverage_ratios) < 5:
            leverage = 5.0  # Default moderate
        else:
            leverage = np.mean(list(self._leverage_ratios)[-5:])

        if len(self._funding_rates) < 3:
            funding = 0.0
        else:
            funding = np.mean(list(self._funding_rates)[-3:])

        if len(self._open_interests) < 3:
            oi = 0.0
        else:
            oi = np.mean(list(self._open_interests)[-3:])

        if len(self._prices) < 20:
            cascade_prob = 0.0
        else:
            cascade_prob = self._calculate_cascade_probability()

        total_liq = sum(l.estimated_liquidation_volume for l in self._liquidation_levels)

        return CascadeMetrics(
            leverage_ratio=leverage,
            funding_rate=funding,
            open_interest=oi,
            cascade_probability=cascade_prob,
            estimated_liquidation_volume=total_liq,
            cascade_phase=self._cascade_phase,
            direction=self._cascade_direction,
            time_to_cascade=self._estimate_time_to_cascade(),
        )

    def _calculate_cascade_probability(self) -> float:
        """Calculate cascade probability."""
        if len(self._prices) < 20:
            return 0.0

        conditions_met = 0

        # Leverage
        if len(self._leverage_ratios) >= 5 and np.mean(list(self._leverage_ratios)[-5:]) > self.leverage_threshold:
            conditions_met += 1

        # Funding
        if len(self._funding_rates) >= 3 and abs(np.mean(list(self._funding_rates)[-3:])) > self.funding_threshold:
            conditions_met += 1

        # Volume
        if len(self._volumes) >= 20:
            recent_vol = np.mean(list(self._volumes)[-5:])
            avg_vol = np.mean(list(self._volumes)[-20:])
            if recent_vol > avg_vol * self.volume_spike:
                conditions_met += 1

        # OI deleverage
        if len(self._open_interests) >= 10:
            recent = np.mean(list(self._open_interests)[-3:])
            earlier = np.mean(list(self._open_interests)[-10:-5])
            if earlier > 0 and recent < earlier * 0.9:
                conditions_met += 1

        return min(0.95, conditions_met / 4 * 1.2)

    def _determine_cascade_direction(self) -> str:
        """Determine if cascade is bullish or bearish."""
        if len(self._funding_rates) >= 3:
            avg_funding = np.mean(list(self._funding_rates)[-3:])
            if avg_funding > self.funding_threshold:
                return "down"  # Bear funding = bearish cascade
            elif avg_funding < -self.funding_threshold:
                return "up"  # Bull funding = bullish cascade

        # Check recent price action
        if len(self._prices) >= 10:
            recent = list(self._prices)[-5:]
            if recent[-1] < recent[0]:
                return "down"
            elif recent[-1] > recent[0]:
                return "up"

        return "down"  # Default to bearish

    def _determine_cascade_phase(self) -> CascadePhase:
        """Determine current phase of cascade."""
        if len(self._prices) < 20:
            return CascadePhase.BUILDING

        # Check volume acceleration
        if len(self._volumes) >= 20:
            recent_vol = np.mean(list(self._volumes)[-3:])
            earlier_vol = np.mean(list(self._volumes)[-10:-3])

            if earlier_vol > 0 and recent_vol > earlier_vol * 3:
                return CascadePhase.TRIGGERING
            elif recent_vol > earlier_vol * 2:
                return CascadePhase.ACCELERATING

        # Check if we've already moved significantly
        if len(self._prices) >= 10:
            recent_returns = np.diff(list(self._prices)[-10:])
            total_move = abs(sum(recent_returns) / list(self._prices)[-10])

            if total_move > 0.10:  # > 10% move
                return CascadePhase.PEAK
            elif total_move > 0.05:
                return CascadePhase.ACCELERATING

        return CascadePhase.BUILDING

    def _estimate_time_to_cascade(self) -> Optional[str]:
        """Estimate time until cascade or phase completion."""
        phase = self._cascade_phase

        if phase == CascadePhase.BUILDING:
            return "Minutes to hours"
        elif phase == CascadePhase.TRIGGERING:
            return "Minutes"
        elif phase == CascadePhase.ACCELERATING:
            return "Ongoing"
        elif phase == CascadePhase.PEAK:
            return "Imminent reversal"
        elif phase == CascadePhase.REVERSING:
            return "Bounce in progress"
        else:
            return "Unknown"

    def _generate_cascade_signal(self, current_price: float) -> CascadeSignal:
        """Generate trading signal for cascade."""
        direction = self._cascade_direction
        phase = self._cascade_phase

        reasons = [f"Cascade phase: {phase.value}"]

        # Entry, target, stop based on phase
        if phase == CascadePhase.BUILDING:
            # Early entry
            if direction == "down":
                action = "sell"
                entry = current_price * 0.99
                target = current_price * 0.92
                stop = current_price * 1.01
                reasons.append("Early short - building pressure")
            else:
                action = "buy"
                entry = current_price * 1.01
                target = current_price * 1.08
                stop = current_price * 0.99
                reasons.append("Early long - bull cascade")

            confidence = 0.60
            expected_move = 0.07
            holding = 60

        elif phase == CascadePhase.TRIGGERING:
            # Middle entry
            if direction == "down":
                action = "sell"
                entry = current_price * 0.98
                target = current_price * 0.88
                stop = current_price * 1.02
                reasons.append("Short - cascade triggering")
            else:
                action = "buy"
                entry = current_price * 1.02
                target = current_price * 1.12
                stop = current_price * 0.98
                reasons.append("Long - bull cascade triggering")

            confidence = 0.72
            expected_move = 0.10
            holding = 45

        elif phase == CascadePhase.ACCELERATING:
            # Ride the cascade
            if direction == "down":
                action = "sell"
                entry = current_price * 0.97
                target = current_price * 0.85
                stop = current_price * 1.02
                reasons.append("Short - cascade accelerating")
            else:
                action = "buy"
                entry = current_price * 1.03
                target = current_price * 1.15
                stop = current_price * 0.98
                reasons.append("Long - bull cascade accelerating")

            confidence = 0.78
            expected_move = 0.12
            holding = 30

        elif phase == CascadePhase.PEAK:
            # Fade the move - counter-trend
            if direction == "down":
                action = "buy"  # BUY THE DIP
                entry = current_price * 0.95
                target = current_price * 1.08
                stop = current_price * 0.88
                reasons.append("BUY THE DIP - cascade peaked")
            else:
                action = "sell"  # SELL THE RALLY
                entry = current_price * 1.05
                target = current_price * 0.92
                stop = current_price * 1.12
                reasons.append("SELL THE RALLY - bull cascade peaked")

            confidence = 0.82
            expected_move = 0.13
            holding = 30

        elif phase == CascadePhase.REVERSING:
            # Continue fade
            if direction == "down":
                action = "buy"
                entry = current_price * 1.02
                target = current_price * 1.10
                stop = current_price * 0.95
                reasons.append("Buy continuation - reversal in progress")
            else:
                action = "sell"
                entry = current_price * 0.98
                target = current_price * 0.90
                stop = current_price * 1.05
                reasons.append("Sell continuation - reversal in progress")

            confidence = 0.75
            expected_move = 0.08
            holding = 20

        else:
            action = "hold"
            entry = current_price
            target = current_price
            stop = current_price
            confidence = 0.0
            expected_move = 0.0
            holding = 0

        total_liq = sum(l.estimated_liquidation_volume for l in self._liquidation_levels)

        return CascadeSignal(
            action=action,
            confidence=min(0.90, confidence),
            phase=phase,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            position_size=min(1.0, confidence * 0.3),  # Size by confidence
            reasons=reasons,
            estimated_liquidation_volume=total_liq,
            expected_move_pct=expected_move,
            holding_period_minutes=holding,
        )

    def get_metrics(self) -> CascadeMetrics:
        """Get current cascade metrics."""
        return self._get_cascade_metrics()

    def is_cascade_in_progress(self) -> bool:
        """Check if cascade is in progress."""
        return self._cascade_in_progress

    def get_liquidation_levels(self) -> List[LiquidationLevel]:
        """Get tracked liquidation levels."""
        return self._liquidation_levels[:10]

    def get_whale_liquidations(self) -> List[Dict]:
        """Get recent whale liquidations."""
        return list(self._whale_liquidations)

    def reset(self) -> None:
        """Reset all state."""
        self._prices.clear()
        self._volumes.clear()
        self._leverage_ratios.clear()
        self._funding_rates.clear()
        self._open_interests.clear()
        self._liquidation_levels.clear()
        self._whale_liquidations.clear()
        self._cascade_in_progress = False
        self._cascade_phase = CascadePhase.BUILDING
        self._cascade_direction = "none"
        self._cascade_start_price = None
        self._cascade_start_time = None
        logger.info("LiquidationCascadeHunter reset")
