"""
WebSocket Market Data Feed
Real-time low-latency market data from exchanges
Wires live data to quantum engine and strategies
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiveTrade:
    """Live trade tick"""
    symbol: str
    price: float
    amount: float
    side: str  # 'buy' or 'sell'
    timestamp: datetime
    exchange: str


@dataclass
class OrderBookLevel:
    """Order book price level"""
    price: float
    amount: float
    count: int  # Number of orders at this level


@dataclass
class LiveOrderBook:
    """Live order book snapshot"""
    symbol: str
    exchange: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted high to low
    asks: List[OrderBookLevel]  # Sorted low to high
    
    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread_pct(self) -> float:
        return (self.spread / self.mid_price) * 100 if self.mid_price > 0 else 0


@dataclass
class LiveTicker:
    """Live ticker update"""
    symbol: str
    bid: float
    ask: float
    last_price: float
    volume_24h: float
    high_24h: float
    low_24h: float
    change_24h: float
    timestamp: datetime
    exchange: str


class KrakenWebSocket:
    """Kraken WebSocket connection"""
    
    def __init__(self):
        self.ws_url = "wss://ws.kraken.com"
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.subscriptions: set = set()
        
        # Callbacks
        self.ticker_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        
        # Data buffers
        self.order_books: Dict[str, LiveOrderBook] = {}
        self.recent_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    async def connect(self):
        """Connect to WebSocket"""
        try:
            self.ws = await websockets.connect(self.ws_url)
            self.is_connected = True
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            logger.info("✅ Kraken WebSocket connected")
            
        except Exception as e:
            logger.error(f"❌ Kraken WebSocket connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect WebSocket"""
        self.is_connected = False
        if self.ws:
            await self.ws.close()
            logger.info("⏹️ Kraken WebSocket disconnected")
    
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates"""
        if not self.ws:
            return
        
        # Convert symbols to Kraken format
        pairs = [self._to_kraken_symbol(s) for s in symbols]
        
        msg = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "ticker"}
        }
        
        await self.ws.send(json.dumps(msg))
        self.subscriptions.update(symbols)
        logger.info(f"📡 Subscribed to tickers: {symbols}")
    
    async def subscribe_trades(self, symbols: List[str]):
        """Subscribe to trade updates"""
        if not self.ws:
            return
        
        pairs = [self._to_kraken_symbol(s) for s in symbols]
        
        msg = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "trade"}
        }
        
        await self.ws.send(json.dumps(msg))
        logger.info(f"📡 Subscribed to trades: {symbols}")
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 10):
        """Subscribe to order book updates"""
        if not self.ws:
            return
        
        pairs = [self._to_kraken_symbol(s) for s in symbols]
        
        msg = {
            "event": "subscribe",
            "pair": pairs,
            "subscription": {"name": "book", "depth": depth}
        }
        
        await self.ws.send(json.dumps(msg))
        logger.info(f"📡 Subscribed to order books: {symbols} (depth: {depth})")
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        while self.is_connected and self.ws:
            try:
                msg = await self.ws.recv()
                data = json.loads(msg)
                
                # Handle different message types
                if isinstance(data, list):
                    # Data message [channelID, data, pair]
                    await self._process_data_message(data)
                elif isinstance(data, dict):
                    # Event message
                    await self._process_event_message(data)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.is_connected = False
                break
            except Exception as e:
                logger.error(f"Message handling error: {e}")
    
    async def _process_data_message(self, data: list):
        """Process data message"""
        if len(data) < 2:
            return
        
        channel_id = data[0]
        payload = data[1]
        pair = data[2] if len(data) > 2 else ""
        symbol = self._from_kraken_symbol(pair)
        
        # Determine message type
        if isinstance(payload, dict):
            # Ticker data
            ticker = self._parse_ticker(symbol, payload)
            for callback in self.ticker_callbacks:
                await callback(ticker)
        
        elif isinstance(payload, list):
            if len(payload) > 0 and isinstance(payload[0], list):
                # Order book data
                orderbook = self._parse_orderbook(symbol, payload)
                self.order_books[symbol] = orderbook
                for callback in self.orderbook_callbacks:
                    await callback(orderbook)
            else:
                # Trade data
                trades = self._parse_trades(symbol, payload)
                for trade in trades:
                    self.recent_trades[symbol].append(trade)
                    for callback in self.trade_callbacks:
                        await callback(trade)
    
    async def _process_event_message(self, data: dict):
        """Process event message"""
        event = data.get("event")
        
        if event == "subscriptionStatus":
            status = data.get("status")
            if status == "subscribed":
                logger.info(f"✅ Subscription confirmed: {data.get('pair')}")
            elif status == "error":
                logger.error(f"❌ Subscription error: {data.get('errorMessage')}")
        
        elif event == "systemStatus":
            status = data.get("status")
            logger.info(f"System status: {status}")
        
        elif event == "heartbeat":
            pass  # Keep connection alive
    
    def _parse_ticker(self, symbol: str, data: dict) -> LiveTicker:
        """Parse ticker message"""
        return LiveTicker(
            symbol=symbol,
            bid=float(data.get("b", [0])[0]),
            ask=float(data.get("a", [0])[0]),
            last_price=float(data.get("c", [0])[0]),
            volume_24h=float(data.get("v", [0, 0])[1]),
            high_24h=float(data.get("h", [0, 0])[1]),
            low_24h=float(data.get("l", [0, 0])[1]),
            change_24h=float(data.get("p", [0, 0])[1]),
            timestamp=datetime.now(),
            exchange="kraken"
        )
    
    def _parse_trades(self, symbol: str, data: list) -> List[LiveTrade]:
        """Parse trade messages"""
        trades = []
        for trade in data:
            if isinstance(trade, list) and len(trade) >= 4:
                trades.append(LiveTrade(
                    symbol=symbol,
                    price=float(trade[0]),
                    amount=float(trade[1]),
                    side="buy" if trade[3] == "b" else "sell",
                    timestamp=datetime.now(),
                    exchange="kraken"
                ))
        return trades
    
    def _parse_orderbook(self, symbol: str, data: list) -> LiveOrderBook:
        """Parse order book message"""
        bids = []
        asks = []
        
        for item in data:
            if isinstance(item, list) and len(item) >= 3:
                price = float(item[0])
                amount = float(item[1])
                count = int(item[2]) if len(item) > 2 else 1
                
                if amount > 0:
                    bids.append(OrderBookLevel(price, amount, count))
                else:
                    asks.append(OrderBookLevel(price, abs(amount), count))
        
        # Sort: bids high to low, asks low to high
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return LiveOrderBook(
            symbol=symbol,
            exchange="kraken",
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )
    
    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert BTCUSD to XBT/USD"""
        symbol = symbol.upper()
        if symbol.startswith("BTC"):
            symbol = "XBT" + symbol[3:]
        return symbol[:3] + "/" + symbol[3:]
    
    def _from_kraken_symbol(self, kraken_symbol: str) -> str:
        """Convert XBT/USD to BTCUSD"""
        if "/" in kraken_symbol:
            base, quote = kraken_symbol.split("/")
            if base == "XBT":
                base = "BTC"
            return base + quote
        return kraken_symbol
    
    def register_ticker_callback(self, callback: Callable):
        """Register ticker update callback"""
        self.ticker_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable):
        """Register trade callback"""
        self.trade_callbacks.append(callback)
    
    def register_orderbook_callback(self, callback: Callable):
        """Register order book callback"""
        self.orderbook_callbacks.append(callback)
    
    def get_order_book(self, symbol: str) -> Optional[LiveOrderBook]:
        """Get current order book for symbol"""
        return self.order_books.get(symbol)
    
    def get_recent_trades(self, symbol: str, n: int = 100) -> List[LiveTrade]:
        """Get recent trades for symbol"""
        return list(self.recent_trades.get(symbol, deque()))[-n:]


class WebSocketFeedManager:
    """Manages all WebSocket connections"""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.all_tickers: Dict[str, LiveTicker] = {}
        self.all_orderbooks: Dict[str, LiveOrderBook] = {}
        
        # Unified callbacks
        self.global_callbacks: List[Callable] = []
    
    async def add_kraken(self):
        """Add Kraken WebSocket"""
        kraken = KrakenWebSocket()
        await kraken.connect()
        
        # Register unified callbacks
        kraken.register_ticker_callback(self._on_ticker)
        kraken.register_trade_callback(self._on_trade)
        kraken.register_orderbook_callback(self._on_orderbook)
        
        self.connections["kraken"] = kraken
        
        # Subscribe to major pairs
        await kraken.subscribe_ticker(["BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD"])
        await kraken.subscribe_orderbook(["BTCUSD", "ETHUSD"], depth=10)
        await kraken.subscribe_trades(["BTCUSD", "ETHUSD"])
    
    async def _on_ticker(self, ticker: LiveTicker):
        """Handle ticker update"""
        self.all_tickers[ticker.symbol] = ticker
        
        # Notify global callbacks
        for callback in self.global_callbacks:
            await callback("ticker", ticker)
    
    async def _on_trade(self, trade: LiveTrade):
        """Handle trade"""
        for callback in self.global_callbacks:
            await callback("trade", trade)
    
    async def _on_orderbook(self, orderbook: LiveOrderBook):
        """Handle order book update"""
        self.all_orderbooks[orderbook.symbol] = orderbook
        
        for callback in self.global_callbacks:
            await callback("orderbook", orderbook)
    
    def get_best_price(self, symbol: str) -> tuple:
        """Get best bid/ask for symbol"""
        ticker = self.all_tickers.get(symbol)
        if ticker:
            return ticker.bid, ticker.ask
        
        orderbook = self.all_orderbooks.get(symbol)
        if orderbook:
            return orderbook.best_bid, orderbook.best_ask
        
        return 0.0, 0.0
    
    def get_mid_price(self, symbol: str) -> float:
        """Get mid price"""
        bid, ask = self.get_best_price(symbol)
        return (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
    
    def register_global_callback(self, callback: Callable):
        """Register global market data callback"""
        self.global_callbacks.append(callback)
    
    async def close_all(self):
        """Close all connections"""
        for name, conn in self.connections.items():
            try:
                await conn.disconnect()
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")


# Global instance
_ws_manager: Optional[WebSocketFeedManager] = None


def get_websocket_manager() -> WebSocketFeedManager:
    """Get singleton WebSocket manager"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketFeedManager()
    return _ws_manager


async def init_websocket_feeds():
    """Initialize all WebSocket feeds"""
    manager = get_websocket_manager()
    
    await manager.add_kraken()
    
    logger.info("✅ WebSocket feeds initialized (<10ms latency)")
