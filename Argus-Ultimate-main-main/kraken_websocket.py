"""
Kraken WebSocket Feed - Real-Time Market Data
==============================================
Provides real-time price updates via WebSocket instead of REST polling.

100x faster than REST polling - get ticks as they happen.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("kraken_ws")


class KrakenWebSocket:
    """Real-time Kraken WebSocket feed for tick-level data."""
    
    WS_URL = "wss://ws.kraken.com"
    
    def __init__(self):
        self._ws = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._subscriptions: Set[str] = set()
        self._running = False
        self._last_price: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}
        self._feed_task = None
    
    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self.WS_URL)
            logger.info("Kraken WebSocket connected")
            self._running = True
            self._feed_task = asyncio.create_task(self._feed_loop())
            return True
        except Exception as e:
            logger.error(f"WebSocket connect failed: {e}")
            return False
    
    async def subscribe_ticker(self, symbols: List[str]) -> bool:
        """Subscribe to real-time ticker data.
        
        Args:
            symbols: List of Kraken pair names (e.g., ["XXBTZUSD", "XETHZUSD"])
        """
        if not self._ws:
            logger.error("Not connected")
            return False
        
        try:
            subscription = {
                "event": "subscribe",
                "subscription": {"name": "ticker"},
                "pair": symbols
            }
            await self._ws.send_json(subscription)
            
            for sym in symbols:
                self._subscriptions.add(f"ticker:{sym}")
            
            logger.info(f"Subscribed to ticker: {symbols}")
            return True
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False
    
    async def subscribe_ohlcv(self, symbols: List[str], interval: int = 1) -> bool:
        """Subscribe to OHLCV candle updates.
        
        Args:
            symbols: List of Kraken pair names
            interval: Candle interval in minutes (1, 5, 15, 30, 60)
        """
        if not self._ws:
            logger.error("Not connected")
            return False
        
        try:
            subscription = {
                "event": "subscribe",
                "subscription": {"name": "ohlc", "interval": interval},
                "pair": symbols
            }
            await self._ws.send_json(subscription)
            
            for sym in symbols:
                self._subscriptions.add(f"ohlc:{sym}:{interval}")
            
            logger.info(f"Subscribed to OHLC {interval}m: {symbols}")
            return True
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False
    
    async def subscribe_trades(self, symbols: List[str]) -> bool:
        """Subscribe to real-time trade feed."""
        if not self._ws:
            logger.error("Not connected")
            return False
        
        try:
            subscription = {
                "event": "subscribe",
                "subscription": {"name": "trade"},
                "pair": symbols
            }
            await self._ws.send_json(subscription)
            
            for sym in symbols:
                self._subscriptions.add(f"trade:{sym}")
            
            logger.info(f"Subscribed to trades: {symbols}")
            return True
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False
    
    def on_ticker(self, symbol: str, callback: Callable[[Dict], None]):
        """Register callback for ticker updates."""
        key = f"ticker:{symbol}"
        self._callbacks.setdefault(key, []).append(callback)
    
    def on_ohlcv(self, symbol: str, callback: Callable[[Dict], None]):
        """Register callback for OHLCV updates."""
        key = f"ohlc:{symbol}"
        self._callbacks.setdefault(key, []).append(callback)
    
    def on_trade(self, symbol: str, callback: Callable[[Dict], None]):
        """Register callback for trade updates."""
        key = f"trade:{symbol}"
        self._callbacks.setdefault(key, []).append(callback)
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get most recent price for a symbol."""
        return self._last_price.get(symbol)
    
    def get_last_update(self, symbol: str) -> Optional[float]:
        """Get timestamp of last update for a symbol."""
        return self._last_update.get(symbol)
    
    async def _feed_loop(self):
        """Main WebSocket message loop."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket closed")
                    break
        except Exception as e:
            logger.error(f"Feed loop error: {e}")
        finally:
            self._running = False
    
    async def _handle_message(self, data):
        """Process incoming WebSocket message."""
        # Subscription confirmation
        if isinstance(data, dict) and data.get("event") == "subscriptionStatus":
            if data.get("status") == "subscribed":
                logger.info(f"Subscribed: {data.get('pair')} {data.get('subscription', {}).get('name')}")
            return
        
        # Heartbeat
        if isinstance(data, dict) and data.get("event") == "heartbeat":
            return
        
        # Data array: [channelID, data, channelName, pair]
        if isinstance(data, list) and len(data) >= 4:
            channel_name = data[2]
            pair = data[3]
            payload = data[1]
            
            if channel_name == "ticker":
                await self._handle_ticker(pair, payload)
            elif channel_name.startswith("ohlc"):
                await self._handle_ohlcv(pair, channel_name, payload)
            elif channel_name == "trade":
                await self._handle_trade(pair, payload)
    
    async def _handle_ticker(self, pair: str, data: dict):
        """Process ticker update."""
        # Kraken ticker format
        close_price = float(data.get("c", [0])[0])  # Last trade price
        
        if close_price > 0:
            self._last_price[pair] = close_price
            self._last_update[pair] = time.time()
            
            ticker_data = {
                "symbol": pair,
                "price": close_price,
                "bid": float(data.get("b", [0])[0]),
                "ask": float(data.get("a", [0])[0]),
                "volume": float(data.get("v", [0, 0])[1]),  # 24h volume
                "timestamp": time.time()
            }
            
            # Fire callbacks
            for cb in self._callbacks.get(f"ticker:{pair}", []):
                try:
                    cb(ticker_data)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")
    
    async def _handle_ohlcv(self, pair: str, channel: str, data: list):
        """Process OHLCV candle update."""
        # OHLCV format: [time, open, high, low, close, volume, vwap, count]
        if len(data) >= 6:
            ohlcv_data = {
                "symbol": pair,
                "timestamp": float(data[0]),
                "open": float(data[1]),
                "high": float(data[2]),
                "low": float(data[3]),
                "close": float(data[4]),
                "volume": float(data[5])
            }
            
            # Fire callbacks
            for cb in self._callbacks.get(f"ohlc:{pair}", []):
                try:
                    cb(ohlcv_data)
                except Exception as e:
                    logger.error(f"OHLCV callback error: {e}")
    
    async def _handle_trade(self, pair: str, data: list):
        """Process trade update."""
        # Trade format: [price, volume, time, side, orderType, misc]
        for trade in data:
            if len(trade) >= 4:
                trade_data = {
                    "symbol": pair,
                    "price": float(trade[0]),
                    "volume": float(trade[1]),
                    "timestamp": float(trade[2]),
                    "side": trade[3],  # "b" (buy) or "s" (sell)
                    "order_type": trade[4] if len(trade) > 4 else "limit"
                }
                
                # Update last price
                self._last_price[pair] = trade_data["price"]
                self._last_update[pair] = time.time()
                
                # Fire callbacks
                for cb in self._callbacks.get(f"trade:{pair}", []):
                    try:
                        cb(trade_data)
                    except Exception as e:
                        logger.error(f"Trade callback error: {e}")
    
    async def disconnect(self):
        """Close WebSocket connection."""
        self._running = False
        if self._feed_task:
            self._feed_task.cancel()
        if self._ws:
            await self._ws.close()
        if hasattr(self, '_session') and self._session:
            await self._session.close()
        logger.info("Kraken WebSocket disconnected")


# ── Price Feed Manager ──────────────────────────────────────────────────────

class RealTimePriceFeed:
    """Manages real-time price feeds for multiple symbols."""
    
    def __init__(self):
        self.ws = KrakenWebSocket()
        self._prices: Dict[str, float] = {}
        self._ohlcvs: Dict[str, List[Dict]] = {}
        self._subscribed = False
    
    async def start(self, symbols: List[str]) -> bool:
        """Start real-time feed for symbols.
        
        Args:
            symbols: List of symbols in format "XBT/USD", "ETH/USD", "SOL/USD"
        """
        # Convert to Kraken format
        kraken_symbols = [self._to_kraken(s) for s in symbols]
        
        connected = await self.ws.connect()
        if not connected:
            logger.error("Failed to connect WebSocket")
            return False
        
        # Subscribe to ticker and OHLCV
        await self.ws.subscribe_ticker(kraken_symbols)
        await self.ws.subscribe_ohlcv(kraken_symbols, interval=1)
        
        # Register callbacks
        for orig, kraken in zip(symbols, kraken_symbols):
            self.ws.on_ticker(kraken, lambda data, s=orig: self._on_price(s, data))
            self.ws.on_ohlcv(kraken, lambda data, s=orig: self._on_ohlcv(s, data))
        
        self._subscribed = True
        logger.info(f"Real-time feed started for: {symbols}")
        return True
    
    def _to_kraken(self, symbol: str) -> str:
        """Convert symbol format to Kraken pair name."""
        mapping = {
            "XBT/USD": "XXBTZUSD",
            "XBT/AUD": "XXBTZAUD",
            "ETH/USD": "XETHZUSD",
            "ETH/AUD": "XETHZAUD",
            "SOL/USD": "SOLUSD",
            "SOL/AUD": "SOLAUD",
        }
        return mapping.get(symbol, symbol.replace("/", ""))
    
    def _on_price(self, symbol: str, data: Dict):
        """Handle price update."""
        self._prices[symbol] = data["price"]
    
    def _on_ohlcv(self, symbol: str, data: Dict):
        """Handle OHLCV update."""
        if symbol not in self._ohlcvs:
            self._ohlcvs[symbol] = []
        self._ohlcvs[symbol].append(data)
        # Keep last 200 candles
        if len(self._ohlcvs[symbol]) > 200:
            self._ohlcvs[symbol] = self._ohlcvs[symbol][-200:]
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        return self._prices.get(symbol) or self.ws.get_last_price(self._to_kraken(symbol))
    
    def get_ohlcvs(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent OHLCV data for symbol."""
        return self._ohlcvs.get(symbol, [])[-limit:]
    
    async def stop(self):
        """Stop the price feed."""
        await self.ws.disconnect()
        self._subscribed = False


# ── Demo ─────────────────────────────────────────────────────────────────────

async def demo():
    """Demo the WebSocket feed."""
    feed = RealTimePriceFeed()
    
    async def on_xbt_price(data):
        print(f"XBT: ${data['price']:,.2f} (bid: ${data.get('bid', 0):,.2f}, ask: ${data.get('ask', 0):,.2f})")
    
    async def on_eth_price(data):
        print(f"ETH: ${data['price']:,.2f}")
    
    symbols = ["XBT/USD", "ETH/USD", "SOL/USD"]
    
    print("Starting real-time WebSocket feed...")
    print("Press Ctrl+C to stop\n")
    
    connected = await feed.start(symbols)
    if not connected:
        print("Failed to connect!")
        return
    
    # Register price callbacks
    feed.ws.on_ticker("XXBTZUSD", on_xbt_price)
    feed.ws.on_ticker("XETHZUSD", on_eth_price)
    
    try:
        while True:
            await asyncio.sleep(1)
            # Show current prices
            for sym in symbols:
                price = feed.get_price(sym)
                if price:
                    print(f"  {sym}: ${price:,.2f}")
            print()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await feed.stop()


if __name__ == "__main__":
    asyncio.run(demo())
