# pyright: reportMissingImports=false
"""
Dynamic Latency Sizing
=======================
Adjusts position size based on execution latency and market conditions.

Problem: Position sizing that doesn't account for execution quality leads to
over-trading during high latency (when fills are worse) and under-trading
during low latency (when fills are better).

Solution: Scale position size inversely with latency metrics:
- Low latency → Full position size (best execution)
- High latency → Reduced position size (worse execution)
- Extreme latency → Minimum position or skip trade

Additional factors:
- Market volatility (higher vol = smaller positions)
- Order book depth (thinner = smaller positions)
- Time of day (off-hours = smaller positions)
- Recent fill quality (bad fills = smaller positions)
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


class SizingMode(Enum):
    """Position sizing modes."""
    CONSERVATIVE = auto()   # Heavy latency discount
    BALANCED = auto()       # Moderate latency discount
    AGGRESSIVE = auto()     # Light latency discount
    ADAPTIVE = auto()       # Self-adjusting based on fill quality


@dataclass
class FillQuality:
    """Metrics for a single fill execution."""
    timestamp: datetime
    venue: str
    symbol: str
    intended_price: float
    actual_price: float
    intended_qty: float
    actual_qty: float
    latency_ms: float
    slippage_bps: float       # Intended vs actual price
    fill_rate: float          # Actual / intended quantity
    
    @property
    def quality_score(self) -> float:
        """Calculate fill quality score (0-1, higher is better)."""
        # Penalize slippage
        slippage_penalty = max(0, 1 - abs(self.slippage_bps) / 100)
        
        # Penalize partial fills
        fill_penalty = self.fill_rate
        
        # Penalize latency (mild)
        latency_penalty = max(0.5, 1 - self.latency_ms / 1000)
        
        return slippage_penalty * fill_penalty * latency_penalty


@dataclass
class LatencySizingAdjustment:
    """Result of latency-based position sizing adjustment."""
    base_size_usd: float           # Original intended size
    adjusted_size_usd: float       # Size after adjustments
    adjustment_factor: float       # Overall multiplier (0-1)
    
    # Component adjustments
    latency_factor: float          # Adjustment due to latency
    volatility_factor: float       # Adjustment due to volatility
    depth_factor: float            # Adjustment due to order book depth
    time_factor: float             # Adjustment due to time of day
    fill_quality_factor: float     # Adjustment due to recent fill quality
    
    # Reasoning
    reasons: List[str]             # Why adjustments were made


class DynamicLatencySizing:
    """
    Dynamic position sizing based on execution conditions.
    
    The core principle: Size positions inversely to execution friction.
    When execution is clean (low latency, deep books, good fills), take
    full positions. When execution is degraded, scale down.
    """
    
    # Default adjustment factors
    DEFAULT_LATENCY_FACTORS = {
        # Latency (ms) -> size factor
        (0, 10): 1.0,       # Ultra-low latency: full size
        (10, 50): 0.9,      # Low latency: 90%
        (50, 100): 0.75,    # Moderate: 75%
        (100, 200): 0.5,    # High: 50%
        (200, 500): 0.25,   # Very high: 25%
        (500, float('inf')): 0.1,  # Extreme: 10%
    }
    
    # Time of day factors (hour in UTC -> factor)
    TIME_FACTORS = {
        (0, 6): 0.7,        # Asia session (moderate)
        (6, 8): 0.9,        # Pre-EU (building)
        (8, 16): 1.0,       # EU + US overlap (best liquidity)
        (16, 20): 0.9,      # US session
        (20, 24): 0.7,      # Late US / pre-Asia
    }
    
    # Minimum position size (as fraction of base)
    MIN_SIZE_FACTOR = 0.1
    
    def __init__(
        self,
        mode: SizingMode = SizingMode.BALANCED,
        base_position_usd: float = 1000.0,
        max_position_usd: float = 10000.0,
        fill_history_size: int = 100
    ):
        """
        Initialize dynamic latency sizing.
        
        Args:
            mode: Sizing mode (conservative/balanced/aggressive/adaptive)
            base_position_usd: Base position size before adjustments
            max_position_usd: Maximum allowed position size
            fill_history_size: Number of recent fills to track
        """
        self.mode = mode
        self.base_position_usd = base_position_usd
        self.max_position_usd = max_position_usd
        
        # Fill quality tracking
        self.fill_history: Deque[FillQuality] = deque(maxlen=fill_history_size)
        self.recent_fill_quality: float = 0.8  # Running average
        
        # Current market conditions
        self.current_latency_ms: float = 20.0
        self.current_volatility: float = 0.02   # Daily vol
        self.current_depth_score: float = 0.8   # 0-1, 1 = deep
        
        # Statistics
        self.total_adjustments: int = 0
        self.total_size_reduction: float = 0.0
        
        # Mode-specific multipliers
        self.mode_multipliers = {
            SizingMode.CONSERVATIVE: 0.7,
            SizingMode.BALANCED: 0.85,
            SizingMode.AGGRESSIVE: 1.0,
            SizingMode.ADAPTIVE: 0.85,  # Starts balanced, adjusts
        }
        
        logger.info("Dynamic Latency Sizing initialized (mode=%s, base=$%.0f)",
                    mode.name, base_position_usd)
    
    def update_conditions(
        self,
        latency_ms: Optional[float] = None,
        volatility: Optional[float] = None,
        depth_score: Optional[float] = None
    ) -> None:
        """Update current market conditions."""
        if latency_ms is not None:
            self.current_latency_ms = latency_ms
        if volatility is not None:
            self.current_volatility = volatility
        if depth_score is not None:
            self.current_depth_score = depth_score
    
    def record_fill(self, fill: FillQuality) -> None:
        """Record a fill for quality tracking."""
        self.fill_history.append(fill)
        
        # Update running average
        if self.fill_history:
            recent = list(self.fill_history)[-20:]  # Last 20 fills
            self.recent_fill_quality = np.mean([f.quality_score for f in recent])
        
        # Adaptive mode adjusts based on fill quality
        if self.mode == SizingMode.ADAPTIVE:
            self._adapt_to_fill_quality()
    
    def _adapt_to_fill_quality(self) -> None:
        """Adapt mode multiplier based on recent fill quality."""
        if self.recent_fill_quality < 0.5:
            # Poor fills → become more conservative
            self.mode_multipliers[SizingMode.ADAPTIVE] = max(
                0.5, self.mode_multipliers[SizingMode.ADAPTIVE] * 0.95
            )
        elif self.recent_fill_quality > 0.8:
            # Good fills → become more aggressive
            self.mode_multipliers[SizingMode.ADAPTIVE] = min(
                1.0, self.mode_multipliers[SizingMode.ADAPTIVE] * 1.02
            )
    
    def calculate_size(
        self,
        symbol: str,
        venue: str,
        intended_size_usd: Optional[float] = None,
        current_price: Optional[float] = None
    ) -> LatencySizingAdjustment:
        """
        Calculate adjusted position size based on current conditions.
        
        Args:
            symbol: Trading symbol
            venue: Exchange/venue
            intended_size_usd: Override base position size
            current_price: Current market price (for validation)
            
        Returns:
            LatencySizingAdjustment with all factors
        """
        base_size = min(
            intended_size_usd or self.base_position_usd,
            self.max_position_usd
        )
        
        reasons = []
        
        # 1. Latency factor
        latency_factor = self._get_latency_factor()
        if latency_factor < 1.0:
            reasons.append(f"Latency {self.current_latency_ms:.0f}ms: {latency_factor:.0%} size")
        
        # 2. Volatility factor
        vol_factor = self._get_volatility_factor()
        if vol_factor < 1.0:
            reasons.append(f"Volatility {self.current_volatility:.1%}: {vol_factor:.0%} size")
        
        # 3. Depth factor
        depth_factor = self._get_depth_factor()
        if depth_factor < 1.0:
            reasons.append(f"Thin order book: {depth_factor:.0%} size")
        
        # 4. Time of day factor
        time_factor = self._get_time_factor()
        if time_factor < 1.0:
            reasons.append(f"Off-peak hours: {time_factor:.0%} size")
        
        # 5. Fill quality factor
        fill_factor = self._get_fill_quality_factor()
        if fill_factor < 1.0:
            reasons.append(f"Recent fill quality {self.recent_fill_quality:.0%}: {fill_factor:.0%} size")
        
        # 6. Mode multiplier
        mode_factor = self.mode_multipliers[self.mode]
        
        # Combine all factors
        combined_factor = (
            latency_factor *
            vol_factor *
            depth_factor *
            time_factor *
            fill_factor *
            mode_factor
        )
        
        # Apply minimum
        combined_factor = max(self.MIN_SIZE_FACTOR, combined_factor)
        
        # Calculate adjusted size
        adjusted_size = base_size * combined_factor
        
        # Track statistics
        self.total_adjustments += 1
        self.total_size_reduction += (base_size - adjusted_size)
        
        if not reasons:
            reasons.append("Normal conditions: full position size")
        
        return LatencySizingAdjustment(
            base_size_usd=base_size,
            adjusted_size_usd=adjusted_size,
            adjustment_factor=combined_factor,
            latency_factor=latency_factor,
            volatility_factor=vol_factor,
            depth_factor=depth_factor,
            time_factor=time_factor,
            fill_quality_factor=fill_factor,
            reasons=reasons
        )
    
    def _get_latency_factor(self) -> float:
        """Get size factor based on current latency."""
        for (low, high), factor in self.DEFAULT_LATENCY_FACTORS.items():
            if low <= self.current_latency_ms < high:
                return factor
        return 0.1  # Default to minimum
    
    def _get_volatility_factor(self) -> float:
        """Get size factor based on volatility."""
        # Higher volatility → smaller positions
        if self.current_volatility < 0.01:
            return 1.0       # Low vol: full size
        elif self.current_volatility < 0.03:
            return 0.85      # Normal: 85%
        elif self.current_volatility < 0.05:
            return 0.7       # High: 70%
        elif self.current_volatility < 0.10:
            return 0.5       # Very high: 50%
        else:
            return 0.3       # Extreme: 30%
    
    def _get_depth_factor(self) -> float:
        """Get size factor based on order book depth."""
        # Thinner books → smaller positions (less slippage risk)
        return max(0.3, self.current_depth_score)
    
    def _get_time_factor(self) -> float:
        """Get size factor based on time of day."""
        hour = datetime.utcnow().hour
        
        for (start, end), factor in self.TIME_FACTORS.items():
            if start <= hour < end:
                return factor
        
        return 0.7  # Default
    
    def _get_fill_quality_factor(self) -> float:
        """Get size factor based on recent fill quality."""
        # Map fill quality (0-1) to size factor (0.5-1.0)
        return 0.5 + (self.recent_fill_quality * 0.5)
    
    def should_trade(
        self,
        symbol: str,
        venue: str,
        min_size_usd: float = 10.0
    ) -> Tuple[bool, str]:
        """
        Determine if we should trade based on conditions.
        
        Returns:
            Tuple of (should_trade, reason)
        """
        adjustment = self.calculate_size(symbol, venue)
        
        if adjustment.adjusted_size_usd < min_size_usd:
            return False, f"Adjusted size ${adjustment.adjusted_size_usd:.2f} below minimum ${min_size_usd}"
        
        if adjustment.latency_factor < 0.2:
            return False, f"Latency too high: {self.current_latency_ms:.0f}ms"
        
        if adjustment.volatility_factor < 0.3:
            return False, f"Volatility too high: {self.current_volatility:.1%}"
        
        return True, "Conditions acceptable for trading"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sizing status."""
        return {
            "mode": self.mode.name,
            "mode_multiplier": self.mode_multipliers[self.mode],
            "base_position_usd": self.base_position_usd,
            "max_position_usd": self.max_position_usd,
            "current_conditions": {
                "latency_ms": self.current_latency_ms,
                "volatility": self.current_volatility,
                "depth_score": self.current_depth_score,
                "fill_quality": self.recent_fill_quality,
            },
            "factors": {
                "latency": self._get_latency_factor(),
                "volatility": self._get_volatility_factor(),
                "depth": self._get_depth_factor(),
                "time": self._get_time_factor(),
                "fill_quality": self._get_fill_quality_factor(),
            },
            "statistics": {
                "total_adjustments": self.total_adjustments,
                "avg_reduction_pct": (
                    (self.total_size_reduction / max(1, self.total_adjustments)) / 
                    self.base_position_usd * 100
                ),
            },
            "fill_history_size": len(self.fill_history),
        }
    
    def set_mode(self, mode: SizingMode) -> None:
        """Change sizing mode."""
        self.mode = mode
        logger.info("Sizing mode changed to %s", mode.name)


# Singleton instance
_sizing: Optional[DynamicLatencySizing] = None


def get_dynamic_latency_sizing(
    mode: SizingMode = SizingMode.BALANCED,
    base_position_usd: float = 1000.0,
    config: Optional[Dict[str, Any]] = None
) -> DynamicLatencySizing:
    """Get or create the Dynamic Latency Sizing singleton."""
    global _sizing
    if _sizing is None:
        _sizing = DynamicLatencySizing(
            mode=mode,
            base_position_usd=base_position_usd,
            max_position_usd=config.get("max_position_usd", 10000.0) if config else 10000.0
        )
    return _sizing


__all__ = [
    "DynamicLatencySizing",
    "FillQuality",
    "LatencySizingAdjustment",
    "SizingMode",
    "get_dynamic_latency_sizing",
]
