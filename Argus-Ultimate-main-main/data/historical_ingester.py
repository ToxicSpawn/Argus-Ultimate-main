"""Historical data ingester for backtesting."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IngestedBar:
    """A single ingested OHLCV bar.
    
    Attributes
    ----------
    symbol : str
        Trading pair symbol
    timestamp : float
        Bar timestamp
    open : float
        Opening price
    high : float
        Highest price
    low : float
        Lowest price
    close : float
        Closing price
    volume : float
        Trading volume
    """
    symbol: str = ""
    timestamp: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


class HistoricalDataIngester:
    """Ingester for historical market data.
    
    Fetches and stores historical OHLCV data from exchanges
    for backtesting purposes.
    """
    
    def __init__(
        self,
        exchange_id: str = "binance",
        cache_dir: str = "data/historical",
    ) -> None:
        self.exchange_id = exchange_id
        self.cache_dir = cache_dir
        self._bars: Dict[str, List[IngestedBar]] = {}
    
    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000,
    ) -> List[IngestedBar]:
        """Fetch historical bars from exchange.
        
        Returns cached bars if available.
        """
        cache_key = f"{symbol}:{timeframe}"
        
        if cache_key in self._bars:
            bars = self._bars[cache_key]
            if start_time:
                bars = [b for b in bars if b.timestamp >= start_time]
            if end_time:
                bars = [b for b in bars if b.timestamp <= end_time]
            return bars[:limit]
        
        return []
    
    def store_bars(self, symbol: str, timeframe: str, bars: List[IngestedBar]) -> None:
        """Store bars in cache."""
        cache_key = f"{symbol}:{timeframe}"
        self._bars[cache_key] = bars
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols with cached data."""
        return list(set(k.split(":")[0] for k in self._bars.keys()))
