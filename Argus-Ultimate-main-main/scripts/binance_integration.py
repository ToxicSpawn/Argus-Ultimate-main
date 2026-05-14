"""
Binance Integration - Real Market Data

Features:
- Connects to Binance WebSocket for real market data
- Handles kline (candlestick) data
- Handles order book data
- Handles trades data
- Error handling and reconnection
- Rate limiting

Usage:
    from scripts.binance_integration import BinanceDataStream
    
    stream = BinanceDataStream(symbols=['BTCUSDT', 'ETHUSDT'])
    stream.start()
    
    # Callback for new data
    def on_kline(symbol, kline):
        print(f"{symbol}: {kline['close']}")
    
    stream.on_kline = on_kline

Requirements:
    pip install python-binance

Run: py scripts/binance_integration.py
"""

import asyncio
import importlib
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BinanceDataStream:
    """
    Real-time Binance data stream.
    
    Supports:
    - Kline (candlestick) data
    - Order book data
    - Trade data
    - Multiple symbols
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        interval: str = '1m',
        depth: int = 5000
    ):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.interval = interval
        self.depth = depth
        self.client: Any = None
        self.bm: Any = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnects = 5
        
        # Callbacks
        self.on_kline: Any = None
        self.on_order_book: Any = None
        self.on_trade: Any = None
        self.on_error: Any = None
        
        logger.info(f"Binance Data Stream initialized for {self.symbols}")
    
    async def connect(self):
        """Connect to Binance API."""
        try:
            binance_module = importlib.import_module("binance")
            async_client = getattr(binance_module, "AsyncClient")
            socket_manager = getattr(binance_module, "BinanceSocketManager")
        except ImportError:
            logger.error("python-binance is not installed; install it to use live Binance streams")
            return False
        try:
            self.client = await async_client.create()
            self.bm = socket_manager(self.client)
            self.is_running = True
            self.reconnect_attempts = 0
            logger.info("Connected to Binance API")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def start_kline_stream(self):
        """Start kline stream for all symbols."""
        if not self.is_running:
            if not await self.connect():
                return
        
        tasks = []
        for symbol in self.symbols:
            stream = self.bm.kline_socket(symbol, interval=self.interval)
            task = asyncio.create_task(self._process_kline(stream, symbol))
            tasks.append(task)
            logger.info(f"Started kline stream for {symbol}")
        
        await asyncio.gather(*tasks)
    
    async def _process_kline(self, stream: Any, symbol: str) -> None:
        """Process kline data."""
        try:
            async with stream as ts:
                while self.is_running:
                    try:
                        receiver: Any = ts
                        msg = await receiver.recv()
                        if msg:
                            kline = self._parse_kline(msg)
                            if self.on_kline:
                                await self.on_kline(symbol, kline)
                    except Exception as e:
                        logger.warning(f"Kline error for {symbol}: {e}")
                        await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Kline stream failed for {symbol}: {e}")
            await self._handle_error()
    
    def _parse_kline(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Parse kline message."""
        return {
            'symbol': msg['s'],
            'open_time': msg['k']['t'],
            'open': float(msg['k']['o']),
            'high': float(msg['k']['h']),
            'low': float(msg['k']['l']),
            'close': float(msg['k']['c']),
            'volume': float(msg['k']['v']),
            'close_time': msg['k']['T'],
            'quote_volume': float(msg['k']['q']),
            'trades': int(msg['k']['n']),
            'taker_buy_base': float(msg['k']['V']),
            'taker_buy_quote': float(msg['k']['Q']),
            'ignore': msg['k']['B']
        }
    
    async def start_order_book_stream(self):
        """Start order book stream."""
        if not self.is_running:
            if not await self.connect():
                return
        
        tasks = []
        for symbol in self.symbols:
            stream = self.bm.depth_socket(symbol, depth=self.depth)
            task = asyncio.create_task(self._process_order_book(stream, symbol))
            tasks.append(task)
            logger.info(f"Started order book stream for {symbol}")
        
        await asyncio.gather(*tasks)
    
    async def _process_order_book(self, stream: Any, symbol: str) -> None:
        """Process order book data."""
        try:
            async with stream as ts:
                while self.is_running:
                    try:
                        receiver: Any = ts
                        msg = await receiver.recv()
                        if msg:
                            book = self._parse_order_book(msg)
                            if self.on_order_book:
                                await self.on_order_book(symbol, book)
                    except Exception as e:
                        logger.warning(f"Order book error for {symbol}: {e}")
                        await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Order book stream failed for {symbol}: {e}")
            await self._handle_error()
    
    def _parse_order_book(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Parse order book message."""
        last_update_id = msg['lastUpdateId']
        bids = [(float(b[0]), float(b[1])) for b in msg['bids']]
        asks = [(float(a[0]), float(a[1])) for a in msg['asks']]
        
        return {
            'symbol': msg['s'],
            'last_update_id': last_update_id,
            'bids': bids,
            'asks': asks,
            'bid_volume': sum(b[1] for b in bids),
            'ask_volume': sum(a[1] for a in asks),
            'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
            'timestamp': time.time()
        }
    
    async def start_trade_stream(self):
        """Start trade stream."""
        if not self.is_running:
            if not await self.connect():
                return
        
        tasks = []
        for symbol in self.symbols:
            stream = self.bm.trade_socket(symbol)
            task = asyncio.create_task(self._process_trade(stream, symbol))
            tasks.append(task)
            logger.info(f"Started trade stream for {symbol}")
        
        await asyncio.gather(*tasks)
    
    async def _process_trade(self, stream: Any, symbol: str) -> None:
        """Process trade data."""
        try:
            async with stream as ts:
                while self.is_running:
                    try:
                        receiver: Any = ts
                        msg = await receiver.recv()
                        if msg:
                            trade = self._parse_trade(msg)
                            if self.on_trade:
                                await self.on_trade(symbol, trade)
                    except Exception as e:
                        logger.warning(f"Trade error for {symbol}: {e}")
                        await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Trade stream failed for {symbol}: {e}")
            await self._handle_error()
    
    def _parse_trade(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Parse trade message."""
        return {
            'symbol': msg['s'],
            'trade_id': msg['t'],
            'price': float(msg['p']),
            'quantity': float(msg['q']),
            'buyer_maker': msg['m'],
            'timestamp': msg['T']
        }
    
    async def _handle_error(self):
        """Handle connection errors."""
        self.reconnect_attempts += 1
        if self.reconnect_attempts <= self.max_reconnects:
            logger.info(f"Reconnecting... (attempt {self.reconnect_attempts})")
            await asyncio.sleep(5)
            await self.connect()
        else:
            logger.error("Max reconnect attempts reached")
            if self.on_error:
                await self.on_error("Max reconnect attempts reached")
    
    async def stop(self):
        """Stop all streams."""
        self.is_running = False
        if self.client:
            await self.client.close_connection()
        logger.info("Binance data stream stopped")


# ============================================================================
# DEMO
# ============================================================================

async def demo():
    """Demo the Binance integration."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("BINANCE INTEGRATION DEMO")
    print("=" * 60)
    print()
    
    # Create stream
    stream = BinanceDataStream(symbols=['BTCUSDT', 'ETHUSDT'], interval='1m')
    
    # Set callbacks
    async def on_kline_callback(symbol, kline):
        print(f"{symbol} {kline['close']:.2f} | Volume: {kline['volume']:.2f}")
    
    stream.on_kline = on_kline_callback
    
    # Start streams
    print("Starting data streams...")
    print("(Press Ctrl+C to stop)")
    print()
    
    try:
        await stream.start_kline_stream()
    except KeyboardInterrupt:
        print("\nStopping...")
        await stream.stop()


if __name__ == "__main__":
    asyncio.run(demo())
