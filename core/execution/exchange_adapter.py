"""Push 77 — ExchangeAdapter: abstract base + PaperAdapter + BinanceAdapter stub.

AbstractExchangeAdapter defines the interface:
  place_order(order) -> str          exchange_id
  cancel_order(order_id, symbol)
  get_balance(asset) -> float
  subscribe_price(symbol, callback)
  unsubscribe_price(symbol)

PaperAdapter (dry-run):
  Simulates instant fills at current price with configurable slippage.
  Maintains internal order book and fake balance.
  No network calls — safe for backtesting and CI.

BinanceAdapter (stub, ready for keys):
  REST via aiohttp (place, cancel, balance)
  WS via websockets (price feed, trade stream)
  HMAC-SHA256 signature
  Reconnect loop with exponential backoff
  Activated only when api_key + api_secret provided
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from core.execution.order import Fill, Order, OrderSide, OrderStatus


class AbstractExchangeAdapter(ABC):
    """Interface all exchange adapters must implement."""

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Submit order to exchange. Returns exchange_id."""

    @abstractmethod
    async def cancel_order(self, exchange_id: str, symbol: str) -> bool:
        """Cancel order by exchange ID. Returns True on success."""

    @abstractmethod
    async def get_balance(self, asset: str = "USDT") -> float:
        """Return available balance for asset."""

    @abstractmethod
    async def subscribe_price(
        self,
        symbol: str,
        callback: Callable[[str, float], None],
    ) -> None:
        """Subscribe to real-time price updates."""

    @abstractmethod
    async def unsubscribe_price(self, symbol: str) -> None:
        """Unsubscribe from price feed."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""


# ---------------------------------------------------------------------------
# Paper adapter
# ---------------------------------------------------------------------------

class PaperAdapter(AbstractExchangeAdapter):
    """Simulated exchange for dry-run and testing.

    Args:
        initial_balance: Starting USDT balance
        slippage_bps:    Simulated slippage in basis points (default 2 bps)
        fill_callback:   Called with Fill when an order is simulated-filled
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        slippage_bps:    float = 2.0,
        fee_rate:        float = 0.001,
        fill_callback:   Optional[Callable] = None,
    ):
        self._balance      = {"USDT": initial_balance}
        self._slippage_bps = slippage_bps
        self._fee_rate     = fee_rate
        self._fill_cb      = fill_callback
        self._prices:      Dict[str, float] = {}
        self._price_cbs:   Dict[str, Callable] = {}
        self._counter      = 0

    def set_price(self, symbol: str, price: float) -> None:
        """Inject a price tick (used by tests and backtest runner)."""
        self._prices[symbol] = price
        if symbol in self._price_cbs:
            self._price_cbs[symbol](symbol, price)

    async def place_order(self, order: Order) -> str:
        """Simulate instant fill at current price ± slippage."""
        self._counter += 1
        exchange_id = f"PAPER_{self._counter:06d}"
        price = self._prices.get(order.symbol, order.price or 0.0)
        slip  = price * self._slippage_bps / 10_000
        if order.side == OrderSide.BUY:
            fill_price = price + slip
        else:
            fill_price = price - slip

        fee  = order.qty * fill_price * self._fee_rate
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=fee,
        )
        order.status     = OrderStatus.FILLED
        order.exchange_id = exchange_id

        # Update paper balance
        if order.side == OrderSide.BUY:
            cost = order.qty * fill_price + fee
            self._balance["USDT"] = self._balance.get("USDT", 0) - cost
            self._balance[order.symbol.replace("USDT", "")] = (
                self._balance.get(order.symbol.replace("USDT", ""), 0) + order.qty
            )
        else:
            proceeds = order.qty * fill_price - fee
            self._balance["USDT"] = self._balance.get("USDT", 0) + proceeds

        if self._fill_cb:
            result = self._fill_cb(fill)
            if asyncio.iscoroutine(result):
                await result

        return exchange_id

    async def cancel_order(self, exchange_id: str, symbol: str) -> bool:
        return True   # Paper: always succeeds

    async def get_balance(self, asset: str = "USDT") -> float:
        return self._balance.get(asset, 0.0)

    async def subscribe_price(
        self,
        symbol: str,
        callback: Callable[[str, float], None],
    ) -> None:
        self._price_cbs[symbol] = callback

    async def unsubscribe_price(self, symbol: str) -> None:
        self._price_cbs.pop(symbol, None)

    async def close(self) -> None:
        self._price_cbs.clear()

    @property
    def balances(self) -> Dict[str, float]:
        return dict(self._balance)


# ---------------------------------------------------------------------------
# Binance adapter stub
# ---------------------------------------------------------------------------

class BinanceAdapter(AbstractExchangeAdapter):
    """Binance REST + WebSocket adapter.

    Requires: aiohttp, websockets
    Set api_key + api_secret for live trading.
    Without keys, all REST calls raise RuntimeError.

    Args:
        api_key:    Binance API key
        api_secret: Binance API secret
        testnet:    Use Binance testnet endpoints
    """

    REST_BASE    = "https://api.binance.com"
    TESTNET_BASE = "https://testnet.binance.vision"
    WS_BASE      = "wss://stream.binance.com:9443/ws"
    WS_TESTNET   = "wss://testnet.binance.vision/ws"

    def __init__(
        self,
        api_key:    str  = "",
        api_secret: str  = "",
        testnet:    bool = False,
    ):
        self._key     = api_key
        self._secret  = api_secret
        self._testnet = testnet
        self._base    = self.TESTNET_BASE if testnet else self.REST_BASE
        self._ws_base = self.WS_TESTNET   if testnet else self.WS_BASE
        self._session  = None
        self._ws_tasks: Dict[str, asyncio.Task] = {}
        self._price_cbs: Dict[str, Callable]    = {}

    def _sign(self, params: dict) -> dict:
        query = urllib.parse.urlencode(params)
        sig   = hmac.new(
            self._secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = sig
        return params

    async def _get_session(self):
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    headers={"X-MBX-APIKEY": self._key}
                )
            except ImportError:
                raise RuntimeError("aiohttp required: pip install aiohttp")
        return self._session

    async def place_order(self, order: Order) -> str:
        if not self._key:
            raise RuntimeError("Binance API key not set")
        session = await self._get_session()
        params = self._sign({
            "symbol":    order.symbol,
            "side":      order.side.value,
            "type":      order.order_type.value,
            "quantity":  str(round(order.qty, 6)),
            "timestamp": int(time.time() * 1000),
        })
        if order.price and order.order_type.value != "MARKET":
            params["price"]    = str(order.price)
            params["timeInForce"] = "GTC"
        async with session.post(f"{self._base}/api/v3/order", params=params) as r:
            data = await r.json()
            if r.status != 200:
                raise RuntimeError(f"Binance place_order error: {data}")
            return str(data.get("orderId", ""))

    async def cancel_order(self, exchange_id: str, symbol: str) -> bool:
        if not self._key:
            raise RuntimeError("Binance API key not set")
        session = await self._get_session()
        params = self._sign({
            "symbol":    symbol,
            "orderId":   exchange_id,
            "timestamp": int(time.time() * 1000),
        })
        async with session.delete(f"{self._base}/api/v3/order", params=params) as r:
            return r.status == 200

    async def get_balance(self, asset: str = "USDT") -> float:
        if not self._key:
            raise RuntimeError("Binance API key not set")
        session = await self._get_session()
        params = self._sign({"timestamp": int(time.time() * 1000)})
        async with session.get(f"{self._base}/api/v3/account", params=params) as r:
            data = await r.json()
            for b in data.get("balances", []):
                if b["asset"] == asset:
                    return float(b["free"])
            return 0.0

    async def subscribe_price(
        self,
        symbol: str,
        callback: Callable[[str, float], None],
    ) -> None:
        self._price_cbs[symbol] = callback
        task = asyncio.create_task(self._ws_loop(symbol))
        self._ws_tasks[symbol] = task

    async def _ws_loop(self, symbol: str) -> None:
        """WebSocket price feed with exponential backoff reconnect."""
        backoff = 1.0
        stream  = f"{symbol.lower()}@trade"
        uri     = f"{self._ws_base}/{stream}"
        while True:
            try:
                import websockets
                async with websockets.connect(uri) as ws:
                    backoff = 1.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        price = float(msg.get("p", 0))
                        if price > 0 and symbol in self._price_cbs:
                            self._price_cbs[symbol](symbol, price)
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def unsubscribe_price(self, symbol: str) -> None:
        task = self._ws_tasks.pop(symbol, None)
        if task:
            task.cancel()
        self._price_cbs.pop(symbol, None)

    async def close(self) -> None:
        for task in self._ws_tasks.values():
            task.cancel()
        self._ws_tasks.clear()
        if self._session:
            await self._session.close()
            self._session = None
