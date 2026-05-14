# pyright: reportMissingImports=false
"""
Latency-Based Stops
====================
Dynamic stop loss adjustment based on execution latency.

Problem: Traditional fixed-percentage stops don't account for execution latency.
When latency spikes (network issues, exchange load), your stop may trigger on
stale prices, resulting in worse fills.

Solution: Widen stops when latency is high, tighten when latency is low.

This provides:
1. Better fill prices during normal conditions (tighter stops)
2. Protection against slippage during high latency (wider stops)
3. Automatic adaptation to exchange/venue latency characteristics
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LatencyTier(Enum):
    """Latency classification tiers."""
    ULTRA_LOW = auto()     # < 5ms (colocated, HFT)
    LOW = auto()           # 5-20ms (normal retail)
    MODERATE = auto()      # 20-100ms (slightly slow)
    HIGH = auto()          # 100-500ms (congested)
    EXTREME = auto()       # > 500ms (degraded)


@dataclass
class LatencySample:
    """A single latency measurement."""
    timestamp: datetime
    venue: str
    symbol: str
    order_send_time: datetime
    ack_receive_time: datetime
    latency_ms: float
    
    @property
    def latency_tier(self) -> LatencyTier:
        if self.latency_ms < 5:
            return LatencyTier.ULTRA_LOW
        elif self.latency_ms < 20:
            return LatencyTier.LOW
        elif self.latency_ms < 100:
            return LatencyTier.MODERATE
        elif self.latency_ms < 500:
            return LatencyTier.HIGH
        else:
            return LatencyTier.EXTREME


@dataclass
class StopAdjustment:
    """Stop loss adjustment based on latency."""
    base_stop_pct: float         # Original stop percentage (e.g., 0.02 = 2%)
    adjusted_stop_pct: float     # New stop percentage after latency adjustment
    latency_ms: float            # Current latency
    latency_tier: LatencyTier    # Latency classification
    adjustment_factor: float     # Multiplier applied (e.g., 1.5 = 50% wider)
    reason: str                  # Human-readable explanation


@dataclass
class VenueLatencyProfile:
    """Latency characteristics for a specific exchange/venue."""
    venue: str
    avg_latency_ms: float
    p50_latency_ms: float        # 50th percentile
    p95_latency_ms: float        # 95th percentile
    p99_latency_ms: float        # 99th percentile
    jitter_ms: float             # Latency variance
    samples: int                 # Number of samples
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def is_healthy(self) -> bool:
        """Check if venue latency is within acceptable range."""
        return self.p95_latency_ms < 200  # < 200ms at 95th percentile
    
    def get_stop_multiplier(self) -> float:
        """Get stop widening multiplier based on latency profile."""
        if self.p95_latency_ms < 20:
            return 1.0      # Normal stops
        elif self.p95_latency_ms < 50:
            return 1.2      # 20% wider
        elif self.p95_latency_ms < 100:
            return 1.5      # 50% wider
        elif self.p95_latency_ms < 200:
            return 2.0      # 100% wider
        else:
            return 2.5      # 150% wider (extreme latency)


class LatencyBasedStops:
    """
    Dynamic stop loss adjustment based on real-time latency measurements.
    
    The core idea: Your stop loss should be wider when latency is high because:
    1. The price you see is stale
    2. The fill you get will be worse
    3. You need more buffer to avoid premature stops
    
    During low latency, you can use tighter stops for better risk management.
    """
    
    # Default adjustment multipliers by latency tier
    DEFAULT_MULTIPLIERS = {
        LatencyTier.ULTRA_LOW: 0.8,    # Tighter stops (10% tighter)
        LatencyTier.LOW: 1.0,          # Normal stops
        LatencyTier.MODERATE: 1.2,     # 20% wider
        LatencyTier.HIGH: 1.5,         # 50% wider
        LatencyTier.EXTREME: 2.0,      # 100% wider
    }
    
    # Minimum stop percentage (can't go below this)
    MIN_STOP_PCT = 0.005  # 0.5%
    
    # Maximum stop percentage (can't go above this)
    MAX_STOP_PCT = 0.15   # 15%
    
    def __init__(
        self,
        base_stop_pct: float = 0.02,
        multipliers: Optional[Dict[LatencyTier, float]] = None,
        history_size: int = 1000
    ):
        """
        Initialize latency-based stops.
        
        Args:
            base_stop_pct: Base stop loss percentage (e.g., 0.02 = 2%)
            multipliers: Custom adjustment multipliers by latency tier
            history_size: Number of latency samples to keep
        """
        self.base_stop_pct = base_stop_pct
        self.multipliers = multipliers or dict(self.DEFAULT_MULTIPLIERS)
        
        # Latency tracking
        self.latency_history: Deque[LatencySample] = deque(maxlen=history_size)
        self.venue_profiles: Dict[str, VenueLatencyProfile] = {}
        
        # Current state
        self.current_latency_ms: float = 20.0  # Default assumption
        self.current_tier: LatencyTier = LatencyTier.LOW
        
        # Statistics
        self.total_adjustments: int = 0
        self.adjustments_by_tier: Dict[LatencyTier, int] = {t: 0 for t in LatencyTier}
        
        logger.info("Latency-Based Stops initialized (base_stop=%.2f%%)", 
                    base_stop_pct * 100)
    
    def record_latency(
        self,
        venue: str,
        symbol: str,
        order_send_time: datetime,
        ack_receive_time: datetime
    ) -> LatencySample:
        """Record a latency measurement from order execution."""
        latency_ms = (ack_receive_time - order_send_time).total_seconds() * 1000
        
        sample = LatencySample(
            timestamp=datetime.now(),
            venue=venue,
            symbol=symbol,
            order_send_time=order_send_time,
            ack_receive_time=ack_receive_time,
            latency_ms=latency_ms
        )
        
        self.latency_history.append(sample)
        self._update_venue_profile(venue, latency_ms)
        self._update_current_state()
        
        return sample
    
    def _update_venue_profile(self, venue: str, latency_ms: float) -> None:
        """Update venue latency profile with new sample."""
        # Get recent samples for this venue
        venue_samples = [
            s.latency_ms for s in self.latency_history 
            if s.venue == venue
        ]
        
        if not venue_samples:
            return
        
        # Calculate statistics
        samples_array = np.array(venue_samples)
        
        self.venue_profiles[venue] = VenueLatencyProfile(
            venue=venue,
            avg_latency_ms=float(np.mean(samples_array)),
            p50_latency_ms=float(np.percentile(samples_array, 50)),
            p95_latency_ms=float(np.percentile(samples_array, 95)),
            p99_latency_ms=float(np.percentile(samples_array, 99)),
            jitter_ms=float(np.std(samples_array)),
            samples=len(venue_samples),
            last_updated=datetime.now()
        )
    
    def _update_current_state(self) -> None:
        """Update current latency state from recent samples."""
        if not self.latency_history:
            return
        
        # Use recent samples (last 50)
        recent = list(self.latency_history)[-50:]
        recent_latencies = [s.latency_ms for s in recent]
        
        # Use p95 of recent samples (more robust than mean)
        self.current_latency_ms = float(np.percentile(recent_latencies, 95))
        
        # Classify latency tier
        if self.current_latency_ms < 5:
            self.current_tier = LatencyTier.ULTRA_LOW
        elif self.current_latency_ms < 20:
            self.current_tier = LatencyTier.LOW
        elif self.current_latency_ms < 100:
            self.current_tier = LatencyTier.MODERATE
        elif self.current_latency_ms < 500:
            self.current_tier = LatencyTier.HIGH
        else:
            self.current_tier = LatencyTier.EXTREME
    
    def calculate_stop(
        self,
        entry_price: float,
        side: str = "long",
        venue: Optional[str] = None,
        custom_base: Optional[float] = None
    ) -> StopAdjustment:
        """
        Calculate the latency-adjusted stop loss price.
        
        Args:
            entry_price: Entry price
            side: "long" or "short"
            venue: Specific venue (uses venue-specific profile if provided)
            custom_base: Override base stop percentage
            
        Returns:
            StopAdjustment with adjusted stop price
        """
        base_stop = custom_base or self.base_stop_pct
        
        # Get adjustment factor
        if venue and venue in self.venue_profiles:
            # Use venue-specific multiplier
            adjustment_factor = self.venue_profiles[venue].get_stop_multiplier()
            latency_tier = self._latency_tier_from_ms(
                self.venue_profiles[venue].p95_latency_ms
            )
            latency_ms = self.venue_profiles[venue].p95_latency_ms
        else:
            # Use global multiplier
            adjustment_factor = self.multipliers.get(
                self.current_tier, 1.0
            )
            latency_tier = self.current_tier
            latency_ms = self.current_latency_ms
        
        # Apply adjustment
        adjusted_stop = base_stop * adjustment_factor
        
        # Clamp to min/max
        adjusted_stop = max(self.MIN_STOP_PCT, min(self.MAX_STOP_PCT, adjusted_stop))
        
        # Calculate stop price
        if side == "long":
            stop_price = entry_price * (1 - adjusted_stop)
        else:
            stop_price = entry_price * (1 + adjusted_stop)
        
        # Build reason
        if adjustment_factor > 1.0:
            reason = f"Latency {latency_ms:.1f}ms ({latency_tier.name}): stop widened {adjustment_factor:.1f}x"
        elif adjustment_factor < 1.0:
            reason = f"Low latency {latency_ms:.1f}ms: stop tightened {adjustment_factor:.1f}x"
        else:
            reason = f"Normal latency {latency_ms:.1f}ms: standard stop"
        
        # Track adjustment
        self.total_adjustments += 1
        self.adjustments_by_tier[latency_tier] += 1
        
        return StopAdjustment(
            base_stop_pct=base_stop,
            adjusted_stop_pct=adjusted_stop,
            latency_ms=latency_ms,
            latency_tier=latency_tier,
            adjustment_factor=adjustment_factor,
            reason=reason
        )
    
    def calculate_dynamic_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,  # For trailing stop
        side: str = "long",
        trail_pct: float = 0.02
    ) -> Tuple[float, str]:
        """
        Calculate a dynamic stop that combines latency adjustment with trailing.
        
        This provides:
        1. Latency-adjusted base stop
        2. Trailing component that follows favorable price movement
        3. Never moves stop in unfavorable direction
        """
        # Get latency-adjusted stop
        adjustment = self.calculate_stop(entry_price, side)
        base_stop_pct = adjustment.adjusted_stop_pct
        
        # Calculate trailing stop
        if side == "long":
            trail_stop = highest_price * (1 - trail_pct)
            base_stop = entry_price * (1 - base_stop_pct)
            # Use whichever is higher (trailing never moves down for longs)
            final_stop = max(base_stop, trail_stop)
        else:
            trail_stop = highest_price * (1 + trail_pct)
            base_stop = entry_price * (1 + base_stop_pct)
            # Use whichever is lower (trailing never moves up for shorts)
            final_stop = min(base_stop, trail_stop)
        
        reason = f"Combined: latency stop={base_stop:.2f}, trail stop={trail_stop:.2f}, final={final_stop:.2f}"
        
        return final_stop, reason
    
    def _latency_tier_from_ms(self, latency_ms: float) -> LatencyTier:
        """Convert latency in ms to tier."""
        if latency_ms < 5:
            return LatencyTier.ULTRA_LOW
        elif latency_ms < 20:
            return LatencyTier.LOW
        elif latency_ms < 100:
            return LatencyTier.MODERATE
        elif latency_ms < 500:
            return LatencyTier.HIGH
        else:
            return LatencyTier.EXTREME
    
    def get_status(self) -> Dict[str, Any]:
        """Get current latency-based stops status."""
        return {
            "base_stop_pct": self.base_stop_pct,
            "current_latency_ms": self.current_latency_ms,
            "current_tier": self.current_tier.name,
            "current_multiplier": self.multipliers.get(self.current_tier, 1.0),
            "adjusted_stop_pct": self.base_stop_pct * self.multipliers.get(
                self.current_tier, 1.0
            ),
            "total_adjustments": self.total_adjustments,
            "adjustments_by_tier": {
                k.name: v for k, v in self.adjustments_by_tier.items()
            },
            "venue_profiles": {
                k: {
                    "avg_ms": v.avg_latency_ms,
                    "p95_ms": v.p95_latency_ms,
                    "p99_ms": v.p99_latency_ms,
                    "samples": v.samples,
                    "healthy": v.is_healthy
                }
                for k, v in self.venue_profiles.items()
            },
            "multipliers": {
                k.name: v for k, v in self.multipliers.items()
            }
        }
    
    def update_multipliers(self, new_multipliers: Dict[LatencyTier, float]) -> None:
        """Update the adjustment multipliers."""
        self.multipliers.update(new_multipliers)
        logger.info("Updated latency multipliers: %s", 
                    {k.name: v for k, v in new_multipliers.items()})


# Singleton instance
_stops: Optional[LatencyBasedStops] = None


def get_latency_based_stops(
    base_stop_pct: float = 0.02,
    config: Optional[Dict[str, Any]] = None
) -> LatencyBasedStops:
    """Get or create the Latency-Based Stops singleton."""
    global _stops
    if _stops is None:
        _stops = LatencyBasedStops(
            base_stop_pct=base_stop_pct,
            multipliers=config.get("multipliers") if config else None
        )
    return _stops


__all__ = [
    "LatencyBasedStops",
    "LatencySample",
    "LatencyTier",
    "StopAdjustment",
    "VenueLatencyProfile",
    "get_latency_based_stops",
]
