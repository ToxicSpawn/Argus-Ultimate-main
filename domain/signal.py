"""
Typed Signal dataclass — replaces raw dict passing between scanner → brain → executor.

All fields are immutable (frozen=True) so a Signal can be safely shared across
async tasks without defensive copying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Signal:
    """A trading signal produced by a strategy or AI brain."""

    symbol: str
    """Trading pair, e.g. 'BTC/USD'."""

    side: Literal["buy", "sell"]
    """Direction of the trade."""

    confidence: float
    """Model confidence in [0.0, 1.0]."""

    strategy_id: str
    """Identifier of the strategy that generated this signal."""

    timestamp: float
    """Unix epoch seconds when the signal was generated."""

    entry_price: float = 0.0
    """Suggested entry price (0 = market order)."""

    stop_loss: float = 0.0
    """Suggested stop-loss price (0 = not set)."""

    take_profit: float = 0.0
    """Suggested take-profit price (0 = not set)."""

    reasoning: str = ""
    """Human-readable explanation from the strategy/AI."""

    def is_valid(self) -> bool:
        """Basic sanity check — confidence in range and symbol non-empty."""
        return bool(self.symbol) and 0.0 <= self.confidence <= 1.0
