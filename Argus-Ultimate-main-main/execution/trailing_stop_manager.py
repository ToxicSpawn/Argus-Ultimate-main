"""
Trailing Stop Manager
=====================
Dynamic trailing stops that adapt to market volatility and capture more profit.

Features:
1. ATR-based trailing stop (adapts to volatility)
2. Profit-tiered trailing (tighter as profit increases)
3. Breakeven lock (move stop to breakeven after certain profit)
4. Time-based stop relaxation (give trades more room over time)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stops."""
    
    # ATR-based settings
    atr_period: int = 14
    atr_multiplier: float = 2.0      # Stop distance = ATR * multiplier
    min_atr_multiplier: float = 1.5   # Minimum multiplier
    max_atr_multiplier: float = 3.0   # Maximum multiplier
    
    # Profit tiers (trail tighter as profit increases)
    profit_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.01, 2.0),   # 1% profit → 2x ATR trail
        (0.02, 1.5),   # 2% profit → 1.5x ATR trail
        (0.05, 1.0),   # 5% profit → 1x ATR trail (tight)
    ])
    
    # Breakeven settings
    breakeven_trigger: float = 0.02    # Move to breakeven at 2% profit
    breakeven_offset: float = 0.001    # 0.1% above entry for breakeven
    
    # Time-based relaxation
    enable_time_relaxation: bool = True
    time_relaxation_start: float = 3600  # Start relaxing after 1 hour
    time_relaxation_rate: float = 0.1    # 10% wider per hour after start


class TrailingStopManager:
    """
    Manages dynamic trailing stops for open positions.
    """
    
    def __init__(self, config: Optional[TrailingStopConfig] = None):
        self.config = config or TrailingStopConfig()
        
        # Active trailing stops
        self.active_trails: Dict[str, Dict] = {}
        
        # Statistics
        self.trails_created: int = 0
        self.trails_updated: int = 0
        self.breakeven_hits: int = 0
    
    def create_trail(
        self,
        position_id: str,
        entry_price: float,
        side: str,
        initial_stop: float,
        atr: float,
    ) -> Dict[str, Any]:
        """Create a new trailing stop for a position."""
        trail = {
            "position_id": position_id,
            "entry_price": entry_price,
            "side": side,
            "current_stop": initial_stop,
            "initial_stop": initial_stop,
            "best_price": entry_price,
            "atr": atr,
            "entry_time": None,  # Will be set when tracking starts
            "breakeven_hit": False,
            "highest_pnl_pct": 0.0,
        }
        
        self.active_trails[position_id] = trail
        self.trails_created += 1
        
        return trail
    
    def update_trail(
        self,
        position_id: str,
        current_price: float,
        current_atr: Optional[float] = None,
    ) -> Optional[float]:
        """
        Update trailing stop based on current price.
        
        Returns:
            New stop price if updated, None otherwise
        """
        if position_id not in self.active_trails:
            return None
        
        trail = self.active_trails[position_id]
        side = trail["side"]
        entry_price = trail["entry_price"]
        
        # Update ATR if provided
        if current_atr and current_atr > 0:
            trail["atr"] = current_atr
        
        # Calculate current P&L percentage
        if side == "buy":
            pnl_pct = (current_price - entry_price) / entry_price
            is_better = current_price > trail["best_price"]
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            is_better = current_price < trail["best_price"]
        
        # Track best price
        if is_better:
            trail["best_price"] = current_price
            trail["highest_pnl_pct"] = max(trail["highest_pnl_pct"], pnl_pct)
        
        # Check breakeven trigger
        if not trail["breakeven_hit"] and pnl_pct >= self.config.breakeven_trigger:
            if side == "buy":
                breakeven_stop = entry_price * (1 + self.config.breakeven_offset)
            else:
                breakeven_stop = entry_price * (1 - self.config.breakeven_offset)
            
            if self._is_stop_better(trail["current_stop"], breakeven_stop, side):
                trail["current_stop"] = breakeven_stop
                trail["breakeven_hit"] = True
                self.breakeven_hits += 1
                logger.debug(f"Breakeven hit for {position_id}: stop -> {breakeven_stop:.2f}")
        
        # Calculate profit-tiered ATR multiplier
        atr_multiplier = self._get_profit_tier_multiplier(pnl_pct)
        
        # Apply time-based relaxation if enabled
        if self.config.enable_time_relaxation and trail.get("entry_time"):
            import time
            elapsed = time.time() - trail["entry_time"]
            if elapsed > self.config.time_relaxation_start:
                hours_elapsed = (elapsed - self.config.time_relaxation_start) / 3600
                relaxation = 1.0 + (hours_elapsed * self.config.time_relaxation_rate)
                atr_multiplier *= relaxation
        
        # Calculate new stop
        atr_value = trail["atr"]
        stop_distance = atr_value * atr_multiplier
        
        if side == "buy":
            new_stop = trail["best_price"] - stop_distance
        else:
            new_stop = trail["best_price"] + stop_distance
        
        # Only update if new stop is better (closer to price for profit protection)
        if self._is_stop_better(trail["current_stop"], new_stop, side):
            trail["current_stop"] = new_stop
            self.trails_updated += 1
            return new_stop
        
        return None
    
    def _get_profit_tier_multiplier(self, pnl_pct: float) -> float:
        """Get ATR multiplier based on current profit level."""
        # Start with default multiplier
        multiplier = self.config.atr_multiplier
        
        # Check profit tiers (sorted by profit threshold descending)
        for profit_threshold, tier_multiplier in sorted(
            self.config.profit_tiers, 
            key=lambda x: x[0], 
            reverse=True
        ):
            if pnl_pct >= profit_threshold:
                multiplier = tier_multiplier
                break
        
        return np.clip(multiplier, self.config.min_atr_multiplier, self.config.max_atr_multiplier)
    
    def _is_stop_better(self, current_stop: float, new_stop: float, side: str) -> bool:
        """Check if new stop is better than current stop."""
        if side == "buy":
            return new_stop > current_stop  # Higher stop = better for longs
        else:
            return new_stop < current_stop  # Lower stop = better for shorts
    
    def get_stop(self, position_id: str) -> Optional[float]:
        """Get current stop price for a position."""
        if position_id in self.active_trails:
            return self.active_trails[position_id]["current_stop"]
        return None
    
    def remove_trail(self, position_id: str) -> None:
        """Remove trailing stop for closed position."""
        if position_id in self.active_trails:
            del self.active_trails[position_id]
    
    def should_exit(self, position_id: str, current_price: float) -> bool:
        """Check if position should be exited based on trailing stop."""
        stop = self.get_stop(position_id)
        if stop is None:
            return False
        
        trail = self.active_trails[position_id]
        side = trail["side"]
        
        if side == "buy":
            return current_price <= stop
        else:
            return current_price >= stop
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trailing stop statistics."""
        return {
            "active_trails": len(self.active_trails),
            "trails_created": self.trails_created,
            "trails_updated": self.trails_updated,
            "breakeven_hits": self.breakeven_hits,
        }


# Singleton
_trailing_manager: Optional[TrailingStopManager] = None


def get_trailing_manager() -> TrailingStopManager:
    """Get or create singleton trailing stop manager."""
    global _trailing_manager
    if _trailing_manager is None:
        _trailing_manager = TrailingStopManager()
    return _trailing_manager


def reset_trailing_manager() -> None:
    """Reset singleton (for testing)."""
    global _trailing_manager
    _trailing_manager = None
