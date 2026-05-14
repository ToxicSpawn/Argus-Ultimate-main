"""Tick engine for real-time OHLCV candle generation.

Converts raw trade ticks into OHLCV candles in real-time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Timeframe to seconds mapping
TIMEFRAME_SECONDS: Dict[str, int] = {
    "1s": 1,
    "5s": 5,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


@dataclass
class OHLCVCandle:
    """OHLCV candle data.
    
    Attributes
    ----------
    symbol : str
        Trading pair symbol
    timeframe : str
        Candle timeframe (e.g., "1m", "5m", "1h")
    open : float
        Opening price
    high : float
        Highest price
    low : float
        Lowest price
    close : float
        Closing price
    volume : float
        Total volume
    timestamp : float
        Candle start timestamp
    trades : int
        Number of trades
    """
    symbol: str = ""
    timeframe: str = "1m"
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    timestamp: float = field(default_factory=time.time)
    trades: int = 0
    
    @property
    def vwap(self) -> float:
        """Volume-weighted average price (simplified)."""
        if self.volume == 0:
            return (self.high + self.low + self.close) / 3
        return self.close  # Simplified
    
    @property
    def price_range(self) -> float:
        """Price range (high - low)."""
        return self.high - self.low


class TickEngine:
    """Real-time tick to OHLCV candle converter.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol
    timeframe : str
        Candle timeframe (default "1m")
    on_candle_close : callable, optional
        Callback when a candle closes
    """
    
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        on_candle_close: Optional[Callable[[OHLCVCandle], None]] = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._on_candle_close = on_candle_close
        self._interval_s = TIMEFRAME_SECONDS.get(timeframe, 60)
        
        self._current_candle: Optional[OHLCVCandle] = None
        self._candle_start: float = 0.0
        self._candles: List[OHLCVCandle] = []
        self._max_candles = 1000
    
    def ingest_tick(
        self,
        price: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> Optional[OHLCVCandle]:
        """Ingest a trade tick and update the current candle.
        
        Returns the closed candle if the tick causes a candle close.
        """
        now = timestamp or time.time()
        
        # Calculate candle start time
        candle_start = (now // self._interval_s) * self._interval_s
        
        # Check if we need to start a new candle
        if self._current_candle is None or candle_start > self._candle_start:
            # Close previous candle
            closed_candle = None
            if self._current_candle is not None:
                closed_candle = self._current_candle
                self._candles.append(closed_candle)
                if len(self._candles) > self._max_candles:
                    self._candles = self._candles[-self._max_candles:]
                
                if self._on_candle_close:
                    self._on_candle_close(closed_candle)
            
            # Start new candle
            self._candle_start = candle_start
            self._current_candle = OHLCVCandle(
                symbol=self.symbol,
                timeframe=self.timeframe,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                timestamp=candle_start,
                trades=1,
            )
            return closed_candle
        
        # Update current candle
        candle = self._current_candle
        candle.high = max(candle.high, price)
        candle.low = min(candle.low, price)
        candle.close = price
        candle.volume += volume
        candle.trades += 1
        
        return None
    
    def get_current_candle(self) -> Optional[OHLCVCandle]:
        """Get the current (in-progress) candle."""
        return self._current_candle
    
    def get_recent_candles(self, count: int = 100) -> List[OHLCVCandle]:
        """Get recent completed candles."""
        return self._candles[-count:]
    
    def get_candle_at(self, timestamp: float) -> Optional[OHLCVCandle]:
        """Get candle at a specific timestamp."""
        candle_start = (timestamp // self._interval_s) * self._interval_s
        for candle in reversed(self._candles):
            if candle.timestamp == candle_start:
                return candle
        return None
