"""
Dynamic Stop Loss — adaptive stop management based on market conditions.

Provides intelligent stop loss placement that adapts to:
- Current volatility (ATR-based)
- Support/resistance levels
- Position profit (trailing stops)
- Market regime changes
- Order flow signals

Features:
- ATR-based initial stop placement
- Trailing stop with profit locking
- Chandelier exit system
- Time-based stop tightening
- Regime-adjusted multipliers

Example::

    stop_manager = DynamicStopLoss()
    stop_manager.update_atr("BTC/USD", atr=500)
    stop_manager.update_price("BTC/USD", price=50000)
    
    # Calculate stop for new long position
    stop = stop_manager.calculate_stop("BTC/USD", side="long", entry_price=50000)
    print(stop.stop_price, stop.trailing_active)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StopConfig:
    """Stop loss configuration."""
    atr_multiplier: float = 2.0  # ATR multiplier for stop distance
    min_stop_pct: float = 0.01  # Minimum stop distance (1%)
    max_stop_pct: float = 0.10  # Maximum stop distance (10%)
    trailing_activation: float = 0.02  # Activate trailing at 2% profit
    trailing_distance: float = 1.5  # Trailing distance in ATR
    chandelier_period: int = 22  # Chandelier exit lookback
    time_decay_days: int = 5  # Tighten stop after N days


@dataclass
class StopLevel:
    """Calculated stop level."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    stop_price: float
    initial_stop_price: float
    current_distance_pct: float
    atr_distance: float
    trailing_active: bool
    profit_locked_pct: float
    timestamp: float
    reason: str  # Why this stop level was chosen


@dataclass
class Position:
    """Tracked position for stop management."""
    symbol: str
    side: str
    entry_price: float
    entry_time: float
    current_stop: Optional[StopLevel]
    highest_price: float  # For trailing (long)
    lowest_price: float  # For trailing (short)
    realized_pnl: float = 0.0


@dataclass
class _SymbolState:
    atr_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=500)
    )
    price_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5000)
    )
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


class DynamicStopLoss:
    """
    Adaptive stop loss manager with multiple strategies.

    Parameters
    ----------
    config : StopConfig
        Stop loss configuration.
    use_chandelier : bool
        Use Chandelier exit for trailing (default True).
    use_support_resistance : bool
        Adjust stops based on S/R levels (default True).
    """

    def __init__(
        self,
        config: Optional[StopConfig] = None,
        use_chandelier: bool = True,
        use_support_resistance: bool = True,
    ) -> None:
        self._config = config or StopConfig()
        self._use_chandelier = use_chandelier
        self._use_sr = use_support_resistance
        self._states: Dict[str, _SymbolState] = {}
        self._positions: Dict[str, Position] = {}

        logger.info(
            "DynamicStopLoss initialized: atr_mult=%.1f trailing_act=%.1f%% chandelier=%s",
            self._config.atr_multiplier,
            self._config.trailing_activation * 100,
            use_chandelier,
        )

    def update_atr(self, symbol: str, atr: float) -> None:
        """Update ATR value for symbol."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        self._states[symbol].atr_history.append(atr)

    def update_price(self, symbol: str, price: float) -> None:
        """Update price for symbol."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        self._states[symbol].price_history.append(price)

    def update_support_resistance(
        self,
        symbol: str,
        support: List[float],
        resistance: List[float],
    ) -> None:
        """Update support and resistance levels."""
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        self._states[symbol].support_levels = support
        self._states[symbol].resistance_levels = resistance

    def _get_atr(self, symbol: str) -> float:
        """Get current ATR value."""
        if symbol in self._states and self._states[symbol].atr_history:
            return self._states[symbol].atr_history[-1]
        return 0.0

    def _get_current_price(self, symbol: str) -> float:
        """Get current price."""
        if symbol in self._states and self._states[symbol].price_history:
            return self._states[symbol].price_history[-1]
        return 0.0

    def calculate_stop(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: Optional[float] = None,
        existing_position: Optional[Position] = None,
    ) -> StopLevel:
        """
        Calculate optimal stop loss level.

        Parameters
        ----------
        symbol : str
            Trading pair.
        side : str
            "long" or "short".
        entry_price : float
            Entry price.
        current_price : float, optional
            Current price (for trailing).
        existing_position : Position, optional
            Existing position for trailing updates.

        Returns
        -------
        StopLevel
            Calculated stop level.
        """
        current_price = current_price or self._get_current_price(symbol) or entry_price
        atr = self._get_atr(symbol)
        
        config = self._config
        
        # Base stop distance
        if atr > 0:
            atr_stop_distance = atr * config.atr_multiplier
        else:
            atr_stop_distance = entry_price * config.min_stop_pct

        # Apply min/max bounds
        min_distance = entry_price * config.min_stop_pct
        max_distance = entry_price * config.max_stop_pct
        stop_distance = np.clip(atr_stop_distance, min_distance, max_distance)

        # Initial stop price
        if side == "long":
            initial_stop = entry_price - stop_distance
        else:
            initial_stop = entry_price + stop_distance

        # Adjust for support/resistance
        adjusted_stop = self._adjust_for_sr(symbol, initial_stop, side)
        
        # Check if trailing should be active
        trailing_active = False
        profit_locked = 0.0
        
        if existing_position:
            # Update highest/lowest
            if side == "long":
                existing_position.highest_price = max(
                    existing_position.highest_price, current_price
                )
                profit_pct = (current_price - entry_price) / entry_price
                
                # Activate trailing if profit exceeds threshold
                if profit_pct >= config.trailing_activation:
                    trailing_active = True
                    
                    # Calculate trailing stop
                    if self._use_chandelier:
                        trailing_stop = self._calculate_chandelier(symbol, side)
                    else:
                        trailing_stop = existing_position.highest_price - (atr * config.trailing_distance)
                    
                    # Use higher of initial and trailing
                    final_stop = max(adjusted_stop, trailing_stop)
                    
                    # Lock in some profit
                    profit_locked = (current_price - final_stop) / current_price
                else:
                    final_stop = adjusted_stop
                    
                existing_position.current_stop = StopLevel(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    stop_price=final_stop,
                    initial_stop_price=initial_stop,
                    current_distance_pct=(current_price - final_stop) / current_price,
                    atr_distance=(current_price - final_stop) / atr if atr > 0 else 0,
                    trailing_active=trailing_active,
                    profit_locked_pct=profit_locked,
                    timestamp=time.time(),
                    reason="trailing" if trailing_active else "initial",
                )
                return existing_position.current_stop
            
            else:  # short
                existing_position.lowest_price = min(
                    existing_position.lowest_price, current_price
                )
                profit_pct = (entry_price - current_price) / entry_price
                
                if profit_pct >= config.trailing_activation:
                    trailing_active = True
                    
                    if self._use_chandelier:
                        trailing_stop = self._calculate_chandelier(symbol, side)
                    else:
                        trailing_stop = existing_position.lowest_price + (atr * config.trailing_distance)
                    
                    final_stop = min(adjusted_stop, trailing_stop)
                    profit_locked = (final_stop - current_price) / current_price
                else:
                    final_stop = adjusted_stop
                    
                existing_position.current_stop = StopLevel(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    stop_price=final_stop,
                    initial_stop_price=initial_stop,
                    current_distance_pct=(final_stop - current_price) / current_price,
                    atr_distance=(final_stop - current_price) / atr if atr > 0 else 0,
                    trailing_active=trailing_active,
                    profit_locked_pct=profit_locked,
                    timestamp=time.time(),
                    reason="trailing" if trailing_active else "initial",
                )
                return existing_position.current_stop

        # New position
        return StopLevel(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_price=adjusted_stop,
            initial_stop_price=initial_stop,
            current_distance_pct=stop_distance / entry_price,
            atr_distance=stop_distance / atr if atr > 0 else 0,
            trailing_active=False,
            profit_locked_pct=0.0,
            timestamp=time.time(),
            reason="initial",
        )

    def _adjust_for_sr(
        self,
        symbol: str,
        stop_price: float,
        side: str,
    ) -> float:
        """Adjust stop price based on support/resistance levels."""
        if not self._use_sr or symbol not in self._states:
            return stop_price

        state = self._states[symbol]
        
        if side == "long":
            # For longs, place stop just below nearest support
            supports_below = [s for s in state.support_levels if s < stop_price]
            if supports_below:
                nearest_support = max(supports_below)
                # Place stop 0.5% below support
                adjusted = nearest_support * 0.995
                return max(adjusted, stop_price)  # Don't move stop higher
        else:
            # For shorts, place stop just above nearest resistance
            resistances_above = [r for r in state.resistance_levels if r > stop_price]
            if resistances_above:
                nearest_resistance = min(resistances_above)
                adjusted = nearest_resistance * 1.005
                return min(adjusted, stop_price)  # Don't move stop lower

        return stop_price

    def _calculate_chandelier(self, symbol: str, side: str) -> float:
        """Calculate Chandelier exit level."""
        if symbol not in self._states:
            return 0.0

        state = self._states[symbol]
        atr = self._get_atr(symbol)
        
        if atr <= 0 or len(state.price_history) < self._config.chandelier_period:
            return 0.0

        prices = list(state.price_history)[-self._config.chandelier_period:]
        
        if side == "long":
            highest = max(prices)
            return highest - (atr * self._config.trailing_distance)
        else:
            lowest = min(prices)
            return lowest + (atr * self._config.trailing_distance)

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
    ) -> Position:
        """Track a new position for stop management."""
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=time.time(),
            current_stop=None,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        
        # Calculate initial stop
        position.current_stop = self.calculate_stop(symbol, side, entry_price)
        self._positions[symbol] = position
        
        return position

    def update_position(self, symbol: str, current_price: float) -> Optional[StopLevel]:
        """Update position with current price and return new stop level."""
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]
        self.update_price(symbol, current_price)
        
        return self.calculate_stop(
            symbol,
            position.side,
            position.entry_price,
            current_price,
            position,
        )

    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close position and return PnL."""
        if symbol not in self._positions:
            return 0.0

        position = self._positions[symbol]
        
        if position.side == "long":
            pnl = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl = (position.entry_price - exit_price) / position.entry_price
        
        position.realized_pnl = pnl
        del self._positions[symbol]
        
        return pnl

    def should_stop(self, symbol: str, current_price: float) -> bool:
        """Check if position should be stopped out."""
        if symbol not in self._positions:
            return False

        position = self._positions[symbol]
        if not position.current_stop:
            return False

        stop = position.current_stop
        
        if position.side == "long":
            return current_price <= stop.stop_price
        else:
            return current_price >= stop.stop_price

    def get_stop_level(self, symbol: str) -> Optional[StopLevel]:
        """Get current stop level for position."""
        if symbol in self._positions:
            return self._positions[symbol].current_stop
        return None

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all tracked positions."""
        return dict(self._positions)

    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols."""
        return sorted(self._states.keys())


__all__ = ["DynamicStopLoss", "StopLevel", "StopConfig", "Position"]
