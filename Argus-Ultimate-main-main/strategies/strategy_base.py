"""
Base strategy interfaces for optional strategy packs.

These are used by the `strategies.router.initialization.strategy_initializer` loader.
They are not the canonical unified signal type (see `unified_types.TradingSignal`),
so the unified runtime should adapt/normalize them when consuming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """
    Lightweight strategy-pack signal.
    """

    symbol: str
    side: str  # "buy" or "sell"
    amount: float
    price: Optional[float] = None  # None for market orders
    confidence: float = 0.5
    strategy_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """
    Base class for optional strategies.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.name = str(name)
        self.config: Dict[str, Any] = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.weight = float(self.config.get("weight", 1.0) or 1.0)
        self.performance_metrics: Dict[str, Any] = {
            "signals_generated": 0,
            "signals_executed": 0,
            "total_pnl": 0.0,
        }
        logger.info("Strategy initialized: %s", self.name)

    @abstractmethod
    async def analyze(self, market_data: Any) -> Optional[TradingSignal]:
        """
        Analyze market data and generate a signal, or None.
        """

    @abstractmethod
    def get_required_indicators(self) -> list[str]:
        """
        Return indicator names needed by this strategy.
        """

    def validate_signal(self, signal: TradingSignal) -> bool:
        if not signal:
            return False
        side = str(signal.side).lower()
        if side not in ("buy", "sell"):
            return False
        try:
            if float(signal.amount) <= 0:
                return False
        except Exception:
            return False
        try:
            c = float(signal.confidence)
            if c < 0.0 or c > 1.0:
                return False
        except Exception:
            return False
        return True
