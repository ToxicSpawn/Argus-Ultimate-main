"""
Data Management Module
======================

Market data ingestion, storage, and retrieval.
Refactored from unified_trading_system.py.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from collections import deque
import numpy as np

from core.exception_manager import (
    DataFeedError,
    DataValidationError,
    handle_errors
)

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data point."""
    symbol: str
    timestamp: datetime
    price: Decimal
    volume: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OHLCV:
    """OHLCV candle."""
    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timeframe: str = "1m"


class DataFeed:
    """Base class for data feeds."""
    
    def __init__(self, name: str, symbols: List[str]):
        self.name = name
        self.symbols = symbols
        self._connected = False
        self._callbacks: List[callable] = []
    
    async def connect(self) -> bool:
        """Connect to data feed."""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from data feed."""
        raise NotImplementedError
    
    def on_data(self, callback: callable):
        """Register data callback."""
        self._callbacks.append(callback)
    
    async def _notify(self, data: MarketData):
        """Notify all callbacks."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Data callback error: {e}")


class DataManager:
    """
    Manages market data from multiple sources.
    """
    
    def __init__(self):
        self._feeds: Dict[str, DataFeed] = {}
        self._latest_data: Dict[str, MarketData] = {}
        self._historical_data: Dict[str, deque] = {}
        self._ohlcv_data: Dict[str, List[OHLCV]] = {}
        self._lock = asyncio.Lock()
        
        # Configuration
        self._max_history = 10000  # Max data points per symbol
        self._cleanup_interval = 3600  # Cleanup every hour
        
        logger.info("DataManager initialized")
    
    async def initialize(self):
        """Initialize data manager."""
        logger.info("Data manager initialized")
    
    async def register_feed(self, feed: DataFeed):
        """Register a data feed."""
        self._feeds[feed.name] = feed
        feed.on_data(self._on_market_data)
        logger.info(f"Data feed registered: {feed.name}")
    
    @handle_errors(logger_name=__name__, reraise=False)
    async def _on_market_data(self, data: MarketData):
        """Process incoming market data."""
        async with self._lock:
            # Store latest data
            self._latest_data[data.symbol] = data
            
            # Add to historical
            if data.symbol not in self._historical_data:
                self._historical_data[data.symbol] = deque(maxlen=self._max_history)
            
            self._historical_data[data.symbol].append(data)
            
            # Update OHLCV
            await self._update_ohlcv(data)
    
    async def _update_ohlcv(self, data: MarketData):
        """Update OHLCV data."""
        symbol = data.symbol
        now = datetime.utcnow()
        
        # Round to nearest minute
        minute = now.replace(second=0, microsecond=0)
        
        if symbol not in self._ohlcv_data:
            self._ohlcv_data[symbol] = []
        
        candles = self._ohlcv_data[symbol]
        
        if candles and candles[-1].timestamp == minute:
            # Update existing candle
            candle = candles[-1]
            candle.high = max(candle.high, data.price)
            candle.low = min(candle.low, data.price)
            candle.close = data.price
            if data.volume:
                candle.volume += data.volume
        else:
            # Create new candle
            candles.append(OHLCV(
                symbol=symbol,
                timestamp=minute,
                open=data.price,
                high=data.price,
                low=data.price,
                close=data.price,
                volume=data.volume or Decimal("0")
            ))
            
            # Limit history
            if len(candles) > self._max_history:
                self._ohlcv_data[symbol] = candles[-self._max_history:]
    
    async def update_market_data(
        self,
        symbol: str,
        price: float,
        **kwargs
    ):
        """Update market data for symbol."""
        data = MarketData(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            price=Decimal(str(price)),
            volume=Decimal(str(kwargs.get('volume', 0))) if kwargs.get('volume') else None,
            high=Decimal(str(kwargs.get('high', price))) if kwargs.get('high') else None,
            low=Decimal(str(kwargs.get('low', price))) if kwargs.get('low') else None,
            bid=Decimal(str(kwargs.get('bid', 0))) if kwargs.get('bid') else None,
            ask=Decimal(str(kwargs.get('ask', 0))) if kwargs.get('ask') else None
        )
        
        await self._on_market_data(data)
    
    async def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol."""
        return self._latest_data.get(symbol)
    
    async def get_history(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[MarketData]:
        """Get historical market data."""
        async with self._lock:
            data = self._historical_data.get(symbol, deque())
            return list(data)[-limit:]
    
    async def get_ohlcv(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[OHLCV]:
        """Get OHLCV candles."""
        async with self._lock:
            candles = self._ohlcv_data.get(symbol, [])
            return candles[-limit:]
    
    async def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price."""
        data = await self.get_latest(symbol)
        return data.price if data else None
    
    async def get_prices(self, symbols: List[str]) -> Dict[str, Decimal]:
        """Get prices for multiple symbols."""
        prices = {}
        for symbol in symbols:
            price = await self.get_price(symbol)
            if price:
                prices[symbol] = price
        return prices
    
    async def cleanup_old_data(self):
        """Clean up old data to free memory."""
        cutoff = datetime.utcnow() - timedelta(hours=24)
        
        async with self._lock:
            # Clean historical data
            for symbol, data in list(self._historical_data.items()):
                # Remove old data points
                while data and data[0].timestamp < cutoff:
                    data.popleft()
            
            logger.info("Old data cleaned up")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get data manager status."""
        return {
            "feeds": len(self._feeds),
            "symbols": len(self._latest_data),
            "data_points": sum(
                len(d) for d in self._historical_data.values()
            ),
            "ohlcv_candles": sum(
                len(c) for c in self._ohlcv_data.values()
            ),
            "connected_feeds": sum(
                1 for f in self._feeds.values() if f._connected
            )
        }
    
    async def start(self):
        """Start data manager."""
        # Connect to all feeds
        for feed in self._feeds.values():
            try:
                await feed.connect()
            except Exception as e:
                logger.error(f"Failed to connect feed {feed.name}: {e}")
        
        logger.info("Data manager started")
    
    async def stop(self):
        """Stop data manager."""
        # Disconnect from all feeds
        for feed in self._feeds.values():
            try:
                await feed.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting feed {feed.name}: {e}")
        
        logger.info("Data manager stopped")
