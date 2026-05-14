"""
Argus Trading System - Stop Loss Management
==========================================

Dynamic stop loss management with multiple strategies.

Supported Stop Types:
- Fixed: Static stop at entry - N%
- ATR-based: Stop at entry - N * ATR
- Trailing: Follows price by fixed amount or ATR
- Breakeven: Moves to entry after N% profit
- Time-based: Exit after max holding period
- Chandelier: High/Low based trailing stop
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple

from core.types import Side, Position

logger = logging.getLogger(__name__)


class StopType(str, Enum):
    """Stop loss strategy types."""
    FIXED = "fixed"
    ATR = "atr"
    TRAILING = "trailing"
    TRAILING_ATR = "trailing_atr"
    BREAKEVEN = "breakeven"
    CHANDELIER = "chandelier"
    TIME_BASED = "time_based"


@dataclass
class StopConfig:
    """Configuration for stop loss management."""
    # Primary stop type
    primary_type: StopType = StopType.ATR

    # Fixed stop parameters
    fixed_stop_pct: float = 0.02  # 2% fixed stop

    # ATR-based parameters
    atr_multiplier: float = 2.0  # Stop at 2x ATR
    atr_period: int = 14

    # Trailing stop parameters
    trailing_distance_pct: float = 0.015  # 1.5% trailing distance
    trailing_atr_multiplier: float = 1.5  # Trailing at 1.5x ATR
    trailing_activation_pct: float = 0.01  # Activate after 1% profit

    # Breakeven parameters
    breakeven_activation_pct: float = 0.01  # Move to breakeven after 1% profit
    breakeven_buffer_pct: float = 0.001  # Small buffer above/below entry

    # Chandelier parameters
    chandelier_period: int = 22
    chandelier_multiplier: float = 3.0

    # Time-based parameters
    max_holding_hours: float = 72.0  # 3 days max holding

    # Risk management
    max_loss_pct: float = 0.05  # Hard stop at 5% loss
    use_hard_stop: bool = True


@dataclass
class StopLevel:
    """Current stop loss level with metadata."""
    price: float
    stop_type: StopType
    is_trailing: bool = False
    is_breakeven: bool = False
    activated_at: Optional[datetime] = None
    reason: str = ""

    def should_exit(self, current_price: float, side: Side) -> bool:
        """Check if stop has been triggered."""
        if side == Side.BUY:
            return current_price <= self.price
        else:
            return current_price >= self.price


class StopLossManager:
    """
    Dynamic stop loss management.

    Manages stop levels for positions, supporting multiple
    stop strategies and automatic adjustment.
    """

    def __init__(self, config: Optional[StopConfig] = None) -> None:
        self.config = config or StopConfig()

        # Track stops per position
        self._stops: dict[str, StopLevel] = {}

        # Track highs/lows for trailing
        self._highest_price: dict[str, float] = {}
        self._lowest_price: dict[str, float] = {}

    def initialize_stop(
        self,
        position: Position,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> StopLevel:
        """
        Initialize stop loss for a new position.

        Args:
            position: The position to protect
            atr: Current ATR value
            volatility: Current volatility (alternative to ATR)

        Returns:
            StopLevel with initial stop price
        """
        config = self.config
        entry = position.entry_price
        side = position.side

        # Calculate stop based on primary type
        if config.primary_type == StopType.FIXED:
            stop_price = self._fixed_stop(entry, side)
            reason = f"Fixed {config.fixed_stop_pct:.1%} stop"

        elif config.primary_type == StopType.ATR:
            if atr is None and volatility is not None:
                atr = volatility
            stop_price = self._atr_stop(entry, side, atr)
            reason = f"ATR-based stop ({config.atr_multiplier}x ATR)"

        elif config.primary_type == StopType.TRAILING:
            stop_price = self._trailing_stop_initial(entry, side)
            reason = f"Trailing {config.trailing_distance_pct:.1%} stop"

        elif config.primary_type == StopType.TRAILING_ATR:
            if atr is None and volatility is not None:
                atr = volatility
            stop_price = self._trailing_atr_stop_initial(entry, side, atr)
            reason = f"Trailing ATR stop ({config.trailing_atr_multiplier}x)"

        elif config.primary_type == StopType.CHANDELIER:
            stop_price = self._chandelier_stop_initial(entry, side, atr)
            reason = "Chandelier stop"

        else:  # Default to fixed
            stop_price = self._fixed_stop(entry, side)
            reason = "Default fixed stop"

        # Apply hard stop limit
        if config.use_hard_stop:
            hard_stop = self._hard_stop(entry, side)
            if side == Side.BUY:
                stop_price = max(stop_price, hard_stop)
            else:
                stop_price = min(stop_price, hard_stop)

        # Initialize tracking
        self._highest_price[position.symbol] = entry
        self._lowest_price[position.symbol] = entry

        stop = StopLevel(
            price=stop_price,
            stop_type=config.primary_type,
            is_trailing=config.primary_type in (StopType.TRAILING, StopType.TRAILING_ATR),
            activated_at=datetime.utcnow(),
            reason=reason,
        )

        self._stops[position.symbol] = stop
        return stop

    def update_stop(
        self,
        position: Position,
        current_price: float,
        atr: Optional[float] = None,
    ) -> StopLevel:
        """
        Update stop loss based on current price.

        Handles trailing stops, breakeven moves, etc.

        Args:
            position: The position
            current_price: Current market price
            atr: Current ATR value

        Returns:
            Updated StopLevel
        """
        symbol = position.symbol
        side = position.side
        entry = position.entry_price
        config = self.config

        # Get or initialize stop
        if symbol not in self._stops:
            return self.initialize_stop(position, atr)

        current_stop = self._stops[symbol]

        # Update high/low tracking
        if symbol in self._highest_price:
            self._highest_price[symbol] = max(self._highest_price[symbol], current_price)
        else:
            self._highest_price[symbol] = current_price

        if symbol in self._lowest_price:
            self._lowest_price[symbol] = min(self._lowest_price[symbol], current_price)
        else:
            self._lowest_price[symbol] = current_price

        # Calculate profit percentage
        if side == Side.BUY:
            profit_pct = (current_price - entry) / entry
        else:
            profit_pct = (entry - current_price) / entry

        # Check for breakeven activation
        if (not current_stop.is_breakeven and
            profit_pct >= config.breakeven_activation_pct):
            breakeven_price = self._breakeven_stop(entry, side)
            # Only move to breakeven if it's better than current stop
            if side == Side.BUY and breakeven_price > current_stop.price:
                current_stop = StopLevel(
                    price=breakeven_price,
                    stop_type=StopType.BREAKEVEN,
                    is_trailing=current_stop.is_trailing,
                    is_breakeven=True,
                    activated_at=datetime.utcnow(),
                    reason="Moved to breakeven",
                )
                logger.info(
                    "%s: Stop moved to breakeven at %.2f",
                    symbol,
                    breakeven_price,
                )
            elif side == Side.SELL and breakeven_price < current_stop.price:
                current_stop = StopLevel(
                    price=breakeven_price,
                    stop_type=StopType.BREAKEVEN,
                    is_trailing=current_stop.is_trailing,
                    is_breakeven=True,
                    activated_at=datetime.utcnow(),
                    reason="Moved to breakeven",
                )

        # Update trailing stop
        if current_stop.is_trailing and profit_pct >= config.trailing_activation_pct:
            if current_stop.stop_type == StopType.TRAILING:
                new_stop = self._update_trailing_stop(
                    symbol, side, current_price, current_stop.price
                )
            elif current_stop.stop_type == StopType.TRAILING_ATR:
                new_stop = self._update_trailing_atr_stop(
                    symbol, side, current_price, current_stop.price, atr
                )
            else:
                new_stop = current_stop.price

            # Only move stop in profitable direction
            if side == Side.BUY and new_stop > current_stop.price:
                current_stop = StopLevel(
                    price=new_stop,
                    stop_type=current_stop.stop_type,
                    is_trailing=True,
                    is_breakeven=current_stop.is_breakeven,
                    activated_at=current_stop.activated_at,
                    reason="Trailing stop adjusted",
                )
            elif side == Side.SELL and new_stop < current_stop.price:
                current_stop = StopLevel(
                    price=new_stop,
                    stop_type=current_stop.stop_type,
                    is_trailing=True,
                    is_breakeven=current_stop.is_breakeven,
                    activated_at=current_stop.activated_at,
                    reason="Trailing stop adjusted",
                )

        self._stops[symbol] = current_stop
        return current_stop

    def check_stop_triggered(
        self,
        position: Position,
        current_price: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if stop loss has been triggered.

        Args:
            position: The position to check
            current_price: Current market price

        Returns:
            Tuple of (triggered, reason)
        """
        symbol = position.symbol

        if symbol not in self._stops:
            return False, None

        stop = self._stops[symbol]

        if stop.should_exit(current_price, position.side):
            return True, f"Stop triggered at {stop.price:.2f}: {stop.reason}"

        return False, None

    def check_time_stop(
        self,
        position: Position,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if time-based stop has been triggered.

        Args:
            position: The position to check

        Returns:
            Tuple of (triggered, reason)
        """
        config = self.config
        holding_time = datetime.utcnow() - position.entry_time
        max_holding = timedelta(hours=config.max_holding_hours)

        if holding_time >= max_holding:
            return True, f"Max holding time exceeded ({config.max_holding_hours}h)"

        return False, None

    def remove_stop(self, symbol: str) -> None:
        """Remove stop tracking for a closed position."""
        self._stops.pop(symbol, None)
        self._highest_price.pop(symbol, None)
        self._lowest_price.pop(symbol, None)

    def get_stop(self, symbol: str) -> Optional[StopLevel]:
        """Get current stop level for a symbol."""
        return self._stops.get(symbol)

    def _fixed_stop(self, entry: float, side: Side) -> float:
        """Calculate fixed percentage stop."""
        distance = entry * self.config.fixed_stop_pct
        if side == Side.BUY:
            return entry - distance
        else:
            return entry + distance

    def _atr_stop(
        self,
        entry: float,
        side: Side,
        atr: Optional[float],
    ) -> float:
        """Calculate ATR-based stop."""
        if atr is None or atr <= 0:
            return self._fixed_stop(entry, side)

        distance = atr * self.config.atr_multiplier
        if side == Side.BUY:
            return entry - distance
        else:
            return entry + distance

    def _trailing_stop_initial(self, entry: float, side: Side) -> float:
        """Calculate initial trailing stop (same as fixed)."""
        distance = entry * self.config.trailing_distance_pct
        if side == Side.BUY:
            return entry - distance
        else:
            return entry + distance

    def _trailing_atr_stop_initial(
        self,
        entry: float,
        side: Side,
        atr: Optional[float],
    ) -> float:
        """Calculate initial trailing ATR stop."""
        if atr is None or atr <= 0:
            return self._trailing_stop_initial(entry, side)

        distance = atr * self.config.trailing_atr_multiplier
        if side == Side.BUY:
            return entry - distance
        else:
            return entry + distance

    def _chandelier_stop_initial(
        self,
        entry: float,
        side: Side,
        atr: Optional[float],
    ) -> float:
        """Calculate initial Chandelier stop."""
        if atr is None or atr <= 0:
            return self._fixed_stop(entry, side)

        distance = atr * self.config.chandelier_multiplier
        if side == Side.BUY:
            return entry - distance
        else:
            return entry + distance

    def _breakeven_stop(self, entry: float, side: Side) -> float:
        """Calculate breakeven stop with buffer."""
        buffer = entry * self.config.breakeven_buffer_pct
        if side == Side.BUY:
            return entry + buffer  # Slightly above entry
        else:
            return entry - buffer  # Slightly below entry

    def _hard_stop(self, entry: float, side: Side) -> float:
        """Calculate hard maximum loss stop."""
        distance = entry * self.config.max_loss_pct
        if side == Side.BUY:
            return entry - distance
        else:
            return entry + distance

    def _update_trailing_stop(
        self,
        symbol: str,
        side: Side,
        current_price: float,
        current_stop: float,
    ) -> float:
        """Update trailing stop based on price movement."""
        distance = current_price * self.config.trailing_distance_pct

        if side == Side.BUY:
            # Trail below the highest price
            highest = self._highest_price.get(symbol, current_price)
            new_stop = highest - distance
            return max(new_stop, current_stop)  # Only move up
        else:
            # Trail above the lowest price
            lowest = self._lowest_price.get(symbol, current_price)
            new_stop = lowest + distance
            return min(new_stop, current_stop)  # Only move down

    def _update_trailing_atr_stop(
        self,
        symbol: str,
        side: Side,
        current_price: float,
        current_stop: float,
        atr: Optional[float],
    ) -> float:
        """Update trailing ATR stop based on price movement."""
        if atr is None or atr <= 0:
            return self._update_trailing_stop(symbol, side, current_price, current_stop)

        distance = atr * self.config.trailing_atr_multiplier

        if side == Side.BUY:
            highest = self._highest_price.get(symbol, current_price)
            new_stop = highest - distance
            return max(new_stop, current_stop)
        else:
            lowest = self._lowest_price.get(symbol, current_price)
            new_stop = lowest + distance
            return min(new_stop, current_stop)


def calculate_atr_stop(
    entry_price: float,
    side: Side,
    atr: float,
    multiplier: float = 2.0,
) -> float:
    """
    Convenience function to calculate ATR-based stop.

    Args:
        entry_price: Entry price
        side: Position side
        atr: Average True Range
        multiplier: ATR multiplier

    Returns:
        Stop loss price
    """
    distance = atr * multiplier
    if side == Side.BUY:
        return entry_price - distance
    else:
        return entry_price + distance


def calculate_trailing_stop(
    current_price: float,
    side: Side,
    highest_price: float,
    lowest_price: float,
    trail_pct: float = 0.02,
) -> float:
    """
    Convenience function to calculate trailing stop.

    Args:
        current_price: Current price
        side: Position side
        highest_price: Highest price since entry
        lowest_price: Lowest price since entry
        trail_pct: Trail distance as percentage

    Returns:
        Trailing stop price
    """
    if side == Side.BUY:
        return highest_price * (1 - trail_pct)
    else:
        return lowest_price * (1 + trail_pct)


__all__ = [
    "StopLossManager",
    "StopConfig",
    "StopLevel",
    "StopType",
    "calculate_atr_stop",
    "calculate_trailing_stop",
]
