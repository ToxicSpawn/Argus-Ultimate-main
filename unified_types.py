"""
Canonical shared types for the Unified System.

Keep these lightweight and dependency-free so they can be imported by strategy,
AI, execution, and testing layers without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TradingSignal:
    """Trading signal for the Unified System."""

    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    strength: float  # 0-1
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    agent_consensus: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

