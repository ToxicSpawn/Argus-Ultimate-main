"""
Trailing Stop Loss - Production Implementation
Tracks price movement and adjusts stop level to lock in profits.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrailingStopState:
    entry_price: float
    side: str  # "long" or "short"
    trail_pct: float
    trail_atr_mult: float
    current_atr: float
    highest_price: float  # For longs
    lowest_price: float   # For shorts
    stop_price: float
    activated: bool = False


class TrailingStop:
    """ATR-based trailing stop that widens/tightens with volatility."""

    def __init__(
        self,
        trail_pct: float = 0.02,
        trail_atr_mult: float = 2.0,
        activation_pct: float = 0.005,
        use_atr: bool = True,
        ratchet_only: bool = True,
    ) -> None:
        self.trail_pct = float(trail_pct)
        self.trail_atr_mult = float(trail_atr_mult)
        self.activation_pct = float(activation_pct)
        self.use_atr = use_atr
        self.ratchet_only = ratchet_only

    def init_state(
        self, entry_price: float, side: str = "long", current_atr: float = 0.0,
    ) -> TrailingStopState:
        trail_dist = self._trail_distance(entry_price, current_atr)
        if side == "long":
            stop = entry_price - trail_dist
        else:
            stop = entry_price + trail_dist
        return TrailingStopState(
            entry_price=entry_price, side=side,
            trail_pct=self.trail_pct, trail_atr_mult=self.trail_atr_mult,
            current_atr=current_atr,
            highest_price=entry_price, lowest_price=entry_price,
            stop_price=stop, activated=False,
        )

    def update(self, state: TrailingStopState, current_price: float, current_atr: float = 0.0) -> TrailingStopState:
        state.current_atr = current_atr if current_atr > 0 else state.current_atr
        trail_dist = self._trail_distance(current_price, state.current_atr)

        if state.side == "long":
            if current_price > state.highest_price:
                state.highest_price = current_price
            pnl_pct = (current_price - state.entry_price) / state.entry_price
            if pnl_pct >= self.activation_pct:
                state.activated = True
            if state.activated:
                new_stop = state.highest_price - trail_dist
                if self.ratchet_only:
                    state.stop_price = max(state.stop_price, new_stop)
                else:
                    state.stop_price = new_stop
        else:
            if current_price < state.lowest_price:
                state.lowest_price = current_price
            pnl_pct = (state.entry_price - current_price) / state.entry_price
            if pnl_pct >= self.activation_pct:
                state.activated = True
            if state.activated:
                new_stop = state.lowest_price + trail_dist
                if self.ratchet_only:
                    state.stop_price = min(state.stop_price, new_stop)
                else:
                    state.stop_price = new_stop
        return state

    def is_triggered(self, state: TrailingStopState, current_price: float) -> bool:
        if state.side == "long":
            return current_price <= state.stop_price
        return current_price >= state.stop_price

    def _trail_distance(self, price: float, atr: float) -> float:
        if self.use_atr and atr > 0:
            return self.trail_atr_mult * atr
        return price * self.trail_pct
