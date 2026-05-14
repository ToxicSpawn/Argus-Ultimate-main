"""
Trailing Stop System
====================
Protects profits by automatically moving stop loss up as price moves in your favor.

Features:
- Dynamic trailing stop that follows price
- Percentage or ATR-based trailing
- Time-based stop adjustment
- Partial exit support
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("trailing_stops")


class TrailType(Enum):
    PERCENTAGE = "percentage"  # Trail by X% from high
    ATR = "atr"               # Trail by X ATR from high
    STEP = "step"             # Trail in discrete steps


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop."""
    trail_type: TrailType = TrailType.PERCENTAGE
    trail_percent: float = 2.0      # 2% trailing distance
    atr_multiplier: float = 2.0     # 2x ATR trailing distance
    step_size: float = 1.0          # 1% step increments
    min_profit_percent: float = 1.0 # Only activate after 1% profit


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    quantity: float
    highest_price: float  # Highest price since entry (for longs)
    lowest_price: float   # Lowest price since entry (for shorts)
    stop_price: float     # Current stop price
    trail_active: bool    # Whether trailing is active
    partial_exits: List[Tuple[float, float]]  # [(price, qty), ...]
    

class TrailingStopManager:
    """Manages trailing stops for all positions."""
    
    def __init__(self, config: TrailingStopConfig = None):
        self.config = config or TrailingStopConfig()
        self.positions: Dict[str, Position] = {}
        self._atr_cache: Dict[str, float] = {}
    
    def open_position(self, symbol: str, side: str, entry_price: float, 
                      quantity: float, atr: float = None) -> Position:
        """Open a new position with trailing stop.
        
        Args:
            symbol: Trading pair
            side: "long" or "short"
            entry_price: Entry price
            quantity: Position size
            atr: Current ATR value (needed for ATR-based trailing)
        """
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            highest_price=entry_price,
            lowest_price=entry_price,
            stop_price=self._calculate_initial_stop(side, entry_price, atr),
            trail_active=False,
            partial_exits=[]
        )
        
        self.positions[symbol] = pos
        
        if atr:
            self._atr_cache[symbol] = atr
        
        logger.info(f"Opened {side} {symbol} @ ${entry_price:,.2f}, stop @ ${pos.stop_price:,.2f}")
        return pos
    
    def _calculate_initial_stop(self, side: str, entry_price: float, atr: float = None) -> float:
        """Calculate initial stop price."""
        if self.config.trail_type == TrailType.ATR and atr:
            distance = atr * self.config.atr_multiplier
        else:
            distance = entry_price * (self.config.trail_percent / 100)
        
        if side == "long":
            return entry_price - distance
        else:
            return entry_price + distance
    
    def update_price(self, symbol: str, price: float, atr: float = None) -> Tuple[bool, Optional[float]]:
        """Update position with new price and check for stop hit.
        
        Returns:
            Tuple of (should_exit, exit_price)
        """
        pos = self.positions.get(symbol)
        if not pos:
            return False, None
        
        pos.current_price = price
        
        if atr:
            self._atr_cache[symbol] = atr
        
        # Check if trailing should activate
        profit_pct = self._get_profit_percent(pos)
        
        if not pos.trail_active and profit_pct >= self.config.min_profit_percent:
            pos.trail_active = True
            logger.info(f"Trailing stop ACTIVATED for {symbol} at {profit_pct:.1f}% profit")
        
        # Update highest/lowest
        if pos.side == "long" and price > pos.highest_price:
            pos.highest_price = price
            if pos.trail_active:
                self._update_stop(pos)
        elif pos.side == "short" and price < pos.lowest_price:
            pos.lowest_price = price
            if pos.trail_active:
                self._update_stop(pos)
        
        # Check if stop hit
        if self._is_stop_hit(pos):
            logger.warning(f"STOP HIT for {symbol} @ ${price:,.2f} (stop was ${pos.stop_price:,.2f})")
            return True, pos.stop_price
        
        return False, None
    
    def _get_profit_percent(self, pos: Position) -> float:
        """Get current profit percentage."""
        if pos.side == "long":
            return ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
        else:
            return ((pos.entry_price - pos.current_price) / pos.entry_price) * 100
    
    def _update_stop(self, pos: Position):
        """Update trailing stop based on new high/low."""
        if pos.side == "long":
            ref_price = pos.highest_price
        else:
            ref_price = pos.lowest_price
        
        if self.config.trail_type == TrailType.ATR:
            atr = self._atr_cache.get(pos.symbol, ref_price * 0.02)
            distance = atr * self.config.atr_multiplier
        elif self.config.trail_type == TrailType.STEP:
            distance = ref_price * (self.config.step_size / 100)
        else:  # PERCENTAGE
            distance = ref_price * (self.config.trail_percent / 100)
        
        if pos.side == "long":
            new_stop = ref_price - distance
            # Stop can only go UP for longs
            if new_stop > pos.stop_price:
                old_stop = pos.stop_price
                pos.stop_price = new_stop
                logger.info(f"Trail UP: {pos.symbol} stop ${old_stop:,.2f} → ${new_stop:,.2f}")
        else:
            new_stop = ref_price + distance
            # Stop can only go DOWN for shorts
            if new_stop < pos.stop_price:
                old_stop = pos.stop_price
                pos.stop_price = new_stop
                logger.info(f"Trail DOWN: {pos.symbol} stop ${old_stop:,.2f} → ${new_stop:,.2f}")
    
    def _is_stop_hit(self, pos: Position) -> bool:
        """Check if stop price has been hit."""
        if pos.side == "long":
            return pos.current_price <= pos.stop_price
        else:
            return pos.current_price >= pos.stop_price
    
    def get_partial_exit(self, pos: Position, profit_target: float) -> Optional[Tuple[float, float]]:
        """Calculate partial exit at profit target.
        
        Returns:
            Tuple of (exit_price, exit_quantity) or None
        """
        profit_pct = self._get_profit_percent(pos)
        
        if profit_pct >= profit_target:
            # Exit 50% at first target
            exit_qty = pos.quantity * 0.5
            pos.quantity -= exit_qty
            pos.partial_exits.append((pos.current_price, exit_qty))
            logger.info(f"Partial exit: {pos.symbol} {exit_qty:.4f} @ ${pos.current_price:,.2f}")
            return pos.current_price, exit_qty
        
        return None
    
    def get_stats(self, symbol: str = None) -> Dict:
        """Get trailing stop statistics."""
        if symbol:
            pos = self.positions.get(symbol)
            if not pos:
                return {}
            return {
                "symbol": pos.symbol,
                "side": pos.side,
                "entry": pos.entry_price,
                "current": pos.current_price,
                "stop": pos.stop_price,
                "profit_pct": self._get_profit_percent(pos),
                "trail_active": pos.trail_active,
                "highest": pos.highest_price,
                "lowest": pos.lowest_price,
                "distance_pct": abs(pos.current_price - pos.stop_price) / pos.current_price * 100
            }
        
        return {sym: self.get_stats(sym) for sym in self.positions}
    
    def close_position(self, symbol: str):
        """Close a position."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Closed position: {symbol}")


# ── Demo ─────────────────────────────────────────────────────────────────────

def demo():
    """Demo the trailing stop system."""
    config = TrailingStopConfig(
        trail_type=TrailType.PERCENTAGE,
        trail_percent=2.0,
        min_profit_percent=1.0
    )
    
    manager = TrailingStopManager(config)
    
    # Open a long position
    manager.open_position("XBT/AUD", "long", 107000.0, 0.01)
    
    # Simulate price movement
    prices = [
        107500, 108000, 108500, 109000, 108500,  # Rise, trail activates
        109500, 110000, 109800, 109500, 109000,  # Continue up, then pullback
        108500, 108000, 107500, 107000, 106500   # Decline to stop
    ]
    
    print("Trailing Stop Demo")
    print("=" * 60)
    
    for price in prices:
        should_exit, exit_price = manager.update_price("XBT/AUD", price)
        stats = manager.get_stats("XBT/AUD")
        
        print(f"Price: ${price:>10,.2f} | Stop: ${stats['stop']:>10,.2f} | "
              f"Profit: {stats['profit_pct']:>+6.2f}% | "
              f"Trail: {'ACTIVE' if stats['trail_active'] else 'WAITING'}")
        
        if should_exit:
            print(f"\n🛑 STOP HIT! Exit at ${exit_price:,.2f}")
            break


if __name__ == "__main__":
    demo()
