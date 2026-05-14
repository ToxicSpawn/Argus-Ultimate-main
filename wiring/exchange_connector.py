"""
Exchange Connector - Live Order Execution
Wires Argus to real exchanges for live trading
Supports: Kraken, Binance, Coinbase, Bybit
"""

import asyncio
import aiohttp
import json
import hmac
import hashlib
import base64
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class LiveOrder:
    """Live order structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float]
    status: OrderStatus
    filled_amount: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    client_order_id: str = ""
    
    @property
    def remaining(self) -> float:
        return self.amount - self.filled_amount
    
    @property
    def is_complete(self) -> bool:
        return self.status == OrderStatus.FILLED or self.status == OrderStatus.CANCELLED


@dataclass
class LivePosition:
    """Live position from exchange"""
    symbol: str
    amount: float
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    exchange: str
    updated_at: datetime = field(default_factory=datetime.now)


class KrakenConnector:
    """Live Kraken exchange connector"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.kraken.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        
    async def connect(self):
        """Initialize connection"""
        self.session = aiohttp.ClientSession()
        
        # Test connection
        try:
            await self._private_request("/0/private/Balance", {})
            self.is_connected = True
            logger.info("✅ Kraken connected")
        except Exception as e:
            logger.error(f"❌ Kraken connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()
            self.is_connected = False
            logger.info("⏹️ Kraken disconnected")
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[float] = None,
        client_order_id: str = None
    ) -> LiveOrder:
        """Submit live order to Kraken"""
        
        # Kraken symbol format
        kraken_symbol = self._to_kraken_symbol(symbol)
        
        # Build order params
        params = {
            "pair": kraken_symbol,
            "type": side.value,
            "ordertype": order_type.value,
            "volume": str(amount)
        }
        
        if price and order_type != OrderType.MARKET:
            params["price"] = str(price)
        
        if client_order_id:
            params["userref"] = client_order_id
        
        # Submit order
        try:
            response = await self._private_request("/0/private/AddOrder", params)
            
            if response.get("error") == []:
                result = response["result"]
                txid = result.get("txid", [None])[0]
                
                order = LiveOrder(
                    order_id=txid,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    status=OrderStatus.PENDING,
                    filled_amount=0.0,
                    exchange="kraken",
                    client_order_id=client_order_id or ""
                )
                
                logger.info(f"📤 Order submitted: {side.value.upper()} {amount} {symbol} @ Kraken (ID: {txid})")
                return order
            else:
                error = response.get("error", ["Unknown error"])[0]
                raise Exception(f"Kraken order failed: {error}")
                
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        try:
            response = await self._private_request(
                "/0/private/CancelOrder",
                {"txid": order_id}
            )
            
            if response.get("error") == []:
                logger.info(f"❌ Order cancelled: {order_id}")
                return True
            else:
                error = response.get("error", ["Unknown"])[0]
                logger.warning(f"Cancel failed: {error}")
                return False
                
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[LiveOrder]:
        """Get order status from exchange"""
        try:
            response = await self._private_request(
                "/0/private/QueryOrders",
                {"txid": order_id}
            )
            
            if response.get("error") == [] and response.get("result"):
                result = response["result"].get(order_id, {})
                
                status_map = {
                    "pending": OrderStatus.PENDING,
                    "open": OrderStatus.OPEN,
                    "closed": OrderStatus.FILLED,
                    "canceled": OrderStatus.CANCELLED,
                    "expired": OrderStatus.CANCELLED
                }
                
                order = LiveOrder(
                    order_id=order_id,
                    symbol=self._from_kraken_symbol(result.get("descr", {}).get("pair", "")),
                    side=OrderSide(result.get("descr", {}).get("type", "buy")),
                    order_type=OrderType(result.get("descr", {}).get("ordertype", "limit")),
                    amount=float(result.get("vol", 0)),
                    price=float(result.get("descr", {}).get("price", 0)) if result.get("descr", {}).get("price") else None,
                    status=status_map.get(result.get("status"), OrderStatus.PENDING),
                    filled_amount=float(result.get("vol_exec", 0)),
                    average_price=float(result.get("price", 0)) if result.get("price") else 0,
                    exchange="kraken"
                )
                
                return order
            
            return None
            
        except Exception as e:
            logger.error(f"Get order status error: {e}")
            return None
    
    async def get_positions(self) -> List[LivePosition]:
        """Get current positions"""
        try:
            response = await self._private_request("/0/private/Balance", {})
            
            positions = []
            if response.get("error") == [] and response.get("result"):
                for currency, amount in response["result"].items():
                    if float(amount) > 0:
                        positions.append(LivePosition(
                            symbol=currency,
                            amount=float(amount),
                            entry_price=0.0,  # Would need trade history
                            unrealized_pnl=0.0,
                            realized_pnl=0.0,
                            exchange="kraken"
                        ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []
    
    async def get_ticker(self, symbol: str) -> Dict[str, float]:
        """Get current price"""
        try:
            kraken_symbol = self._to_kraken_symbol(symbol)
            response = await self._public_request("/0/public/Ticker", {"pair": kraken_symbol})
            
            if response.get("error") == [] and response.get("result"):
                ticker = list(response["result"].values())[0]
                return {
                    "bid": float(ticker["b"][0]),
                    "ask": float(ticker["a"][0]),
                    "last": float(ticker["c"][0])
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Get ticker error: {e}")
            return {}
    
    def _to_kraken_symbol(self, symbol: str) -> str:
        """Convert BTCUSD to XBTUSD"""
        symbol = symbol.upper().replace("/", "").replace("-", "")
        if symbol.startswith("BTC"):
            symbol = "XBT" + symbol[3:]
        return symbol
    
    def _from_kraken_symbol(self, kraken_symbol: str) -> str:
        """Convert XBTUSD to BTCUSD"""
        if kraken_symbol.startswith("XBT"):
            kraken_symbol = "BTC" + kraken_symbol[3:]
        return kraken_symbol
    
    def _sign_request(self, urlpath: str, data: Dict) -> str:
        """Sign Kraken API request"""
        postdata = urlencode(data)
        encoded = (str(data["nonce"]) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        signature = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()
    
    async def _private_request(self, endpoint: str, params: Dict) -> Dict:
        """Make authenticated request"""
        if not self.session:
            raise Exception("Not connected")
        
        # Add nonce
        params["nonce"] = str(int(time.time() * 1000))
        
        # Sign request
        signature = self._sign_request(endpoint, params)
        
        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature
        }
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.post(url, data=params, headers=headers) as response:
            return await response.json()
    
    async def _public_request(self, endpoint: str, params: Dict) -> Dict:
        """Make public request"""
        if not self.session:
            raise Exception("Not connected")
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.session.get(url, params=params) as response:
            return await response.json()


class ExchangeConnectorManager:
    """Manages connections to multiple exchanges"""
    
    def __init__(self):
        self.connectors: Dict[str, Any] = {}
        self.active_orders: Dict[str, LiveOrder] = {}
        self.order_callbacks: List[Callable] = []
    
    async def add_kraken(self, api_key: str, api_secret: str):
        """Add Kraken connection"""
        connector = KrakenConnector(api_key, api_secret)
        await connector.connect()
        self.connectors["kraken"] = connector
    
    async def submit_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "limit",
        price: Optional[float] = None
    ) -> LiveOrder:
        """Submit order to specified exchange"""
        connector = self.connectors.get(exchange)
        if not connector:
            raise Exception(f"Exchange {exchange} not connected")
        
        order = await connector.submit_order(
            symbol=symbol,
            side=OrderSide(side.lower()),
            amount=amount,
            order_type=OrderType(order_type.lower()),
            price=price
        )
        
        # Track order
        self.active_orders[order.order_id] = order
        
        # Notify callbacks
        for callback in self.order_callbacks:
            await callback(order)
        
        return order
    
    async def cancel_order(self, exchange: str, order_id: str) -> bool:
        """Cancel order on exchange"""
        connector = self.connectors.get(exchange)
        if not connector:
            return False
        
        return await connector.cancel_order(order_id)
    
    async def sync_orders(self):
        """Sync order statuses from all exchanges"""
        for order_id, order in list(self.active_orders.items()):
            if not order.is_complete:
                connector = self.connectors.get(order.exchange)
                if connector:
                    updated = await connector.get_order_status(order_id)
                    if updated:
                        self.active_orders[order_id] = updated
                        
                        # Notify if status changed
                        if updated.status != order.status:
                            logger.info(f"Order {order_id} status: {order.status.value} → {updated.status.value}")
                            
                            for callback in self.order_callbacks:
                                await callback(updated)
    
    def register_order_callback(self, callback: Callable):
        """Register callback for order updates"""
        self.order_callbacks.append(callback)
    
    async def get_all_positions(self) -> List[LivePosition]:
        """Get positions from all exchanges"""
        all_positions = []
        
        for name, connector in self.connectors.items():
            try:
                positions = await connector.get_positions()
                all_positions.extend(positions)
            except Exception as e:
                logger.error(f"Failed to get positions from {name}: {e}")
        
        return all_positions
    
    async def close_all(self):
        """Close all exchange connections"""
        for name, connector in self.connectors.items():
            try:
                await connector.disconnect()
            except Exception as e:
                logger.error(f"Error closing {name}: {e}")


# Global instance
_exchange_manager: Optional[ExchangeConnectorManager] = None


def get_exchange_manager() -> ExchangeConnectorManager:
    """Get singleton exchange manager"""
    global _exchange_manager
    if _exchange_manager is None:
        _exchange_manager = ExchangeConnectorManager()
    return _exchange_manager


async def init_exchange_connections(config: Dict):
    """Initialize all exchange connections from config"""
    manager = get_exchange_manager()
    
    if config.get("kraken", {}).get("enabled"):
        await manager.add_kraken(
            api_key=config["kraken"]["api_key"],
            api_secret=config["kraken"]["api_secret"]
        )
    
    logger.info("✅ Exchange connections initialized")
