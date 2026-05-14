"""Push 75 — Signal: the unit of communication between strategy and order manager.

A Signal represents a trade intent produced by a strategy. It carries
enough information for the order manager to size and place an order
without further querying the strategy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SignalSide(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"   # close / no position


class SignalStrength(float, Enum):
    """Canonical strength buckets — strategies may use continuous [0,1]."""
    WEAK     = 0.25
    MODERATE = 0.50
    STRONG   = 0.75
    VERY_STRONG = 1.00


@dataclass
class Signal:
    """Trade signal produced by a strategy.

    Args:
        symbol:      Trading pair, e.g. "BTCUSDT"
        side:        LONG | SHORT | FLAT
        strength:    Conviction in [0, 1]. Used for Kelly position sizing.
        strategy_id: Identifier of the originating strategy
        order_type:  "Market" | "Limit"
        price:       Limit price (None for Market orders)
        stop_loss:   Absolute stop-loss price
        take_profit: Absolute take-profit price
        timestamp:   Unix epoch seconds (auto-set)
        metadata:    Arbitrary extra context from the strategy
    """
    symbol:      str
    side:        SignalSide
    strength:    float = 0.5          # [0, 1]
    strategy_id: str   = "unknown"
    order_type:  str   = "Market"
    price:       Optional[float] = None
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None
    timestamp:   float = field(default_factory=time.time)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be in [0,1], got {self.strength}")
        if isinstance(self.side, str):
            self.side = SignalSide(self.side)

    @property
    def is_entry(self) -> bool:
        return self.side in (SignalSide.LONG, SignalSide.SHORT)

    @property
    def is_exit(self) -> bool:
        return self.side == SignalSide.FLAT

    @property
    def age_secs(self) -> float:
        return time.time() - self.timestamp

    def __repr__(self) -> str:
        return (
            f"Signal({self.symbol} {self.side.value} "
            f"str={self.strength:.2f} [{self.strategy_id}])"
        )
