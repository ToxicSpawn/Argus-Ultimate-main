"""Exchange reserve monitor for on-chain analysis.

Tracks cryptocurrency reserves on exchanges to detect
accumulation or distribution patterns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReserveChange:
    """Single reserve change event.
    
    Attributes
    ----------
    symbol : str
        Cryptocurrency symbol (e.g., "BTC", "ETH")
    exchange : str
        Exchange name
    change_amount : float
        Change in reserve amount
    change_pct : float
        Percentage change
    current_reserve : float
        Current total reserve
    timestamp : float
        Unix timestamp
    """
    symbol: str = ""
    exchange: str = ""
    change_amount: float = 0.0
    change_pct: float = 0.0
    current_reserve: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def direction(self) -> str:
        """Return 'inflow', 'outflow', or 'neutral'."""
        if self.change_amount > 0:
            return "inflow"
        elif self.change_amount < 0:
            return "outflow"
        return "neutral"


class ExchangeReserveMonitor:
    """Monitor for exchange cryptocurrency reserves.
    
    Parameters
    ----------
    cache_ttl_s : float
        Cache time-to-live in seconds (default 300)
    exchanges : list[str]
        Exchanges to monitor (default ["binance", "kraken", "coinbase"])
    """
    
    def __init__(
        self,
        cache_ttl_s: float = 300.0,
        exchanges: Optional[List[str]] = None,
    ) -> None:
        self._cache_ttl_s = cache_ttl_s
        self._exchanges = exchanges or ["binance", "kraken", "coinbase"]
        self._reserves: Dict[str, Dict[str, float]] = {}
        self._changes: List[ReserveChange] = []
        self._last_fetch_time: float = 0.0
    
    def get_reserve(self, symbol: str, exchange: str) -> float:
        """Get current reserve for a symbol on an exchange."""
        return self._reserves.get(exchange, {}).get(symbol, 0.0)
    
    def get_total_reserve(self, symbol: str) -> float:
        """Get total reserve across all exchanges."""
        total = 0.0
        for exchange in self._exchanges:
            total += self.get_reserve(symbol, exchange)
        return total
    
    def get_recent_changes(
        self,
        symbol: Optional[str] = None,
        hours: float = 24.0,
    ) -> List[ReserveChange]:
        """Get recent reserve changes."""
        now = time.time()
        cutoff = now - (hours * 3600)
        changes = [c for c in self._changes if c.timestamp >= cutoff]
        if symbol:
            changes = [c for c in changes if c.symbol == symbol]
        return changes
    
    def update_reserve(
        self,
        symbol: str,
        exchange: str,
        new_reserve: float,
    ) -> Optional[ReserveChange]:
        """Update reserve and record change."""
        old_reserve = self.get_reserve(symbol, exchange)
        
        if exchange not in self._reserves:
            self._reserves[exchange] = {}
        
        self._reserves[exchange][symbol] = new_reserve
        
        change_amount = new_reserve - old_reserve
        change_pct = (change_amount / max(old_reserve, 1.0)) * 100
        
        change = ReserveChange(
            symbol=symbol,
            exchange=exchange,
            change_amount=change_amount,
            change_pct=change_pct,
            current_reserve=new_reserve,
        )
        self._changes.append(change)
        
        # Keep only last 10000 changes
        if len(self._changes) > 10000:
            self._changes = self._changes[-10000:]
        
        return change
