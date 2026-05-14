"""
Bybit Perpetual Futures Connector (USDT Linear Perps) — V5 API.

Full trading connector supporting:
- Order placement (market/limit) and cancellation
- Position management (get/set leverage)
- Funding rate queries (current + predicted)
- Balance queries
- HMAC-SHA256 authentication
- Rate limiting (10 req/s orders, 50 req/s market data)

Bybit supports Australian users and offers:
- USDT-margined linear perpetuals (BTC, ETH, SOL, etc.)
- 8-hour funding rate payments
- Deep liquidity for BTC/ETH perps
- No geo-restriction for Australian residents

API keys: Set BYBIT_API_KEY / BYBIT_API_SECRET in environment.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Bybit V5 API base URLs
BYBIT_V5_MAINNET = "https://api.bybit.com"
BYBIT_V5_TESTNET = "https://api-testnet.bybit.com"

# Rate limits
_ORDER_RATE_LIMIT = 10     # max 10 order requests per second
_MARKET_RATE_LIMIT = 50    # max 50 market data requests per second
_RECV_WINDOW = "5000"      # 5 second receive window


class BybitRateLimiter:
    """Token-bucket rate limiter for Bybit API calls."""

    def __init__(self, max_per_second: int = 10):
        self._max = max_per_second
        self._tokens: float = float(max_per_second)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._max, self._tokens + elapsed * self._max)
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._max
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._last_refill = time.monotonic()
            else:
                self._tokens -= 1.0


class BybitAPIError(Exception):
    """Raised when Bybit V5 API returns retCode != 0."""

    def __init__(self, ret_code: int, ret_msg: str, endpoint: str = ""):
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.endpoint = endpoint
        super().__init__(f"Bybit API error {ret_code}: {ret_msg} (endpoint={endpoint})")


class BybitConnector:
    """
    Bybit USDT linear perpetuals connector using V5 REST API.

    Supports both direct V5 API calls (aiohttp + HMAC-SHA256) and
    CCXT fallback for compatibility with existing code.
    """

    health_check_symbol = "BTC/USDT:USDT"  # Bybit linear perp format

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        self.api_key = api_key or os.environ.get("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BYBIT_API_SECRET", "")
        self.testnet = testnet
        self.base_url = BYBIT_V5_TESTNET if testnet else BYBIT_V5_MAINNET
        self._exchange: Any = None
        self._session: Any = None  # aiohttp.ClientSession
        self.connected: bool = False

        # Rate limiters
        self._order_limiter = BybitRateLimiter(_ORDER_RATE_LIMIT)
        self._market_limiter = BybitRateLimiter(_MARKET_RATE_LIMIT)

    # ------------------------------------------------------------------
    # Authentication — HMAC-SHA256 signing per Bybit V5 spec
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, params_str: str) -> str:
        """
        Generate HMAC-SHA256 signature for Bybit V5 API.

        Sign string = timestamp + api_key + recv_window + params_str
        """
        sign_str = timestamp + self.api_key + _RECV_WINDOW + params_str
        return hmac.new(
            self.api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, params_str: str) -> Dict[str, str]:
        """Build authenticated headers for V5 API request."""
        timestamp = str(int(time.time() * 1000))
        signature = self._sign(timestamp, params_str)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                logger.warning("aiohttp not available, V5 direct API disabled")
                return None
        return self._session

    async def _v5_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None, auth: bool = False
    ) -> Dict[str, Any]:
        """Make authenticated GET request to Bybit V5 API."""
        session = await self._get_session()
        if session is None:
            raise RuntimeError("aiohttp session not available")

        url = f"{self.base_url}{endpoint}"
        params_str = urlencode(params) if params else ""

        headers: Dict[str, str] = {}
        if auth:
            headers = self._auth_headers(params_str)

        await self._market_limiter.acquire()

        async with session.get(url, params=params, headers=headers, timeout=10) as resp:
            data = await resp.json()
            ret_code = data.get("retCode", -1)
            if ret_code != 0:
                raise BybitAPIError(ret_code, data.get("retMsg", "unknown"), endpoint)
            return data.get("result", {})

    async def _v5_post(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make authenticated POST request to Bybit V5 API."""
        session = await self._get_session()
        if session is None:
            raise RuntimeError("aiohttp session not available")

        url = f"{self.base_url}{endpoint}"
        import json
        payload_str = json.dumps(payload)
        headers = self._auth_headers(payload_str)

        await self._order_limiter.acquire()

        async with session.post(url, data=payload_str, headers=headers, timeout=10) as resp:
            data = await resp.json()
            ret_code = data.get("retCode", -1)
            if ret_code != 0:
                raise BybitAPIError(ret_code, data.get("retMsg", "unknown"), endpoint)
            return data.get("result", {})

    # ------------------------------------------------------------------
    # CCXT fallback (for environments without aiohttp or legacy callers)
    # ------------------------------------------------------------------

    def _get_exchange(self) -> Any:
        if self._exchange is None:
            try:
                import ccxt  # type: ignore[import]
                params: Dict[str, Any] = {
                    "apiKey": self.api_key,
                    "secret": self.api_secret,
                    "enableRateLimit": True,
                    "options": {"defaultType": "linear"},
                }
                if self.testnet:
                    params["options"]["sandboxMode"] = True
                self._exchange = ccxt.bybit(params)
            except Exception as exc:
                logger.error("Failed to create Bybit CCXT exchange: %s", exc)
                raise
        return self._exchange

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect and verify API access."""
        try:
            # Try V5 direct first
            session = await self._get_session()
            if session is not None:
                result = await self._v5_get("/v5/market/tickers", {"category": "linear", "symbol": "BTCUSDT"})
                if result and result.get("list"):
                    self.connected = True
                    logger.info("Bybit V5 API connected (direct)")
                    return True
        except Exception as exc:
            logger.debug("V5 direct connect attempt: %s — falling back to CCXT", exc)

        try:
            ex = self._get_exchange()
            markets = ex.load_markets()
            self.connected = len(markets) > 0
            logger.info("Bybit connected via CCXT: %d markets", len(markets))
            return self.connected
        except Exception as exc:
            logger.error("Bybit connect failed: %s", exc)
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        self.connected = False
        self._exchange = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> Dict[str, Any]:
        """
        Place an order on Bybit perpetual futures (V5 API).

        Args:
            symbol: Trading pair, e.g. "BTCUSDT" or "BTC/USDT:USDT"
            side: "Buy" or "Sell"
            quantity: Order quantity
            order_type: "Market" or "Limit"
            price: Required for limit orders
            reduce_only: If True, only reduce existing position
            time_in_force: "GTC", "IOC", "FOK"

        Returns:
            Order response dict with orderId, orderLinkId, etc.
        """
        # Normalise symbol: "BTC/USDT:USDT" -> "BTCUSDT"
        api_symbol = self._normalise_symbol(symbol)
        # Normalise side: "buy"/"BUY" -> "Buy"
        api_side = side.capitalize()

        payload: Dict[str, Any] = {
            "category": "linear",
            "symbol": api_symbol,
            "side": api_side,
            "orderType": order_type.capitalize(),
            "qty": str(quantity),
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
        }
        if price is not None and order_type.lower() == "limit":
            payload["price"] = str(price)

        try:
            result = await self._v5_post("/v5/order/create", payload)
            logger.info(
                "Bybit order placed: %s %s %.6f %s (id=%s)",
                api_side, api_symbol, quantity, order_type,
                result.get("orderId", "?"),
            )
            return result
        except BybitAPIError as exc:
            logger.error("Bybit place_order failed: %s", exc)
            raise
        except Exception:
            # Fallback to CCXT
            return await self.create_order(symbol, side.lower(), quantity, order_type.lower(), price, reduce_only)

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order.

        Args:
            order_id: Bybit order ID
            symbol: Trading pair

        Returns:
            Cancellation response dict
        """
        api_symbol = self._normalise_symbol(symbol)
        payload = {
            "category": "linear",
            "symbol": api_symbol,
            "orderId": order_id,
        }
        try:
            result = await self._v5_post("/v5/order/cancel", payload)
            logger.info("Bybit order cancelled: %s %s", order_id, api_symbol)
            return result
        except BybitAPIError as exc:
            logger.error("Bybit cancel_order failed: %s", exc)
            raise
        except Exception:
            # CCXT fallback
            try:
                ex = self._get_exchange()
                return ex.cancel_order(order_id, symbol)
            except Exception as exc2:
                logger.error("cancel_order CCXT fallback failed: %s", exc2)
                raise

    async def create_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Create a futures order on Bybit (CCXT-compatible interface)."""
        try:
            ex = self._get_exchange()
            params: Dict[str, Any] = {"reduceOnly": reduce_only}
            if order_type == "market":
                return ex.create_market_order(symbol, side, amount, params=params)
            return ex.create_limit_order(symbol, side, amount, price, params=params)
        except Exception as exc:
            logger.error("Bybit create_order %s %s %s: %s", side, amount, symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current perpetual position for a symbol.

        Returns dict with: symbol, size, side, entry_price, unrealized_pnl, leverage
        """
        api_symbol = self._normalise_symbol(symbol)
        try:
            result = await self._v5_get(
                "/v5/position/list",
                {"category": "linear", "symbol": api_symbol},
                auth=True,
            )
            positions = result.get("list", [])
            for pos in positions:
                size = float(pos.get("size", 0))
                if size != 0:
                    return {
                        "symbol": symbol,
                        "api_symbol": api_symbol,
                        "size": size,
                        "side": pos.get("side", "").lower(),
                        "entry_price": float(pos.get("avgPrice", 0) or 0),
                        "unrealized_pnl": float(pos.get("unrealisedPnl", 0) or 0),
                        "leverage": float(pos.get("leverage", 1) or 1),
                        "liq_price": float(pos.get("liqPrice", 0) or 0),
                        "position_value": float(pos.get("positionValue", 0) or 0),
                    }
            return None
        except Exception:
            # CCXT fallback
            try:
                ex = self._get_exchange()
                positions = ex.fetch_positions([symbol])
                for pos in positions:
                    if pos.get("symbol") == symbol and float(pos.get("contracts", 0) or 0) != 0:
                        return {
                            "symbol": symbol,
                            "size": abs(float(pos.get("contracts", 0) or 0)),
                            "side": pos.get("side", ""),
                            "entry_price": float(pos.get("entryPrice", 0) or 0),
                            "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                            "leverage": float(pos.get("leverage", 1) or 1),
                        }
                return None
            except Exception as exc:
                logger.warning("get_position %s failed: %s", symbol, exc)
                return None

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open perpetual positions."""
        try:
            result = await self._v5_get(
                "/v5/position/list",
                {"category": "linear", "settleCoin": "USDT"},
                auth=True,
            )
            positions = []
            for pos in result.get("list", []):
                size = float(pos.get("size", 0))
                if size != 0:
                    positions.append({
                        "symbol": pos.get("symbol", ""),
                        "size": size,
                        "side": pos.get("side", "").lower(),
                        "entry_price": float(pos.get("avgPrice", 0) or 0),
                        "unrealized_pnl": float(pos.get("unrealisedPnl", 0) or 0),
                        "leverage": float(pos.get("leverage", 1) or 1),
                        "liq_price": float(pos.get("liqPrice", 0) or 0),
                        "position_value": float(pos.get("positionValue", 0) or 0),
                    })
            return positions
        except Exception:
            # CCXT fallback
            try:
                ex = self._get_exchange()
                raw = ex.fetch_positions()
                return [
                    {
                        "symbol": p.get("symbol", ""),
                        "size": abs(float(p.get("contracts", 0) or 0)),
                        "side": p.get("side", ""),
                        "entry_price": float(p.get("entryPrice", 0) or 0),
                        "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                        "leverage": float(p.get("leverage", 1) or 1),
                    }
                    for p in raw
                    if float(p.get("contracts", 0) or 0) != 0
                ]
            except Exception as exc:
                logger.warning("get_positions failed: %s", exc)
                return []

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair
            leverage: Leverage value (1-100)

        Returns:
            API response
        """
        api_symbol = self._normalise_symbol(symbol)
        lev_str = str(leverage)
        payload = {
            "category": "linear",
            "symbol": api_symbol,
            "buyLeverage": lev_str,
            "sellLeverage": lev_str,
        }
        try:
            result = await self._v5_post("/v5/position/set-leverage", payload)
            logger.info("Bybit leverage set: %s -> %dx", api_symbol, leverage)
            return result
        except BybitAPIError as exc:
            # retCode 110043 = leverage not modified (already set)
            if exc.ret_code == 110043:
                logger.debug("Leverage already set to %dx for %s", leverage, api_symbol)
                return {"already_set": True}
            raise
        except Exception:
            try:
                ex = self._get_exchange()
                ex.set_leverage(leverage, symbol)
                return {"leverage": leverage, "symbol": symbol}
            except Exception as exc2:
                logger.warning("set_leverage %s failed: %s", symbol, exc2)
                raise

    # ------------------------------------------------------------------
    # Funding rate queries
    # ------------------------------------------------------------------

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current and predicted funding rate for a symbol.

        Returns:
            Dict with funding_rate, predicted_rate, next_funding_time, symbol
        """
        api_symbol = self._normalise_symbol(symbol)
        try:
            result = await self._v5_get(
                "/v5/market/tickers",
                {"category": "linear", "symbol": api_symbol},
            )
            tickers = result.get("list", [])
            if tickers:
                t = tickers[0]
                return {
                    "symbol": symbol,
                    "api_symbol": api_symbol,
                    "funding_rate": float(t.get("fundingRate", 0) or 0),
                    "next_funding_time": int(t.get("nextFundingTime", 0) or 0),
                    "predicted_rate": float(t.get("fundingRate", 0) or 0),
                    "exchange": "bybit",
                }
            return {"symbol": symbol, "funding_rate": 0.0, "exchange": "bybit"}
        except Exception:
            # Fall back to existing CCXT method
            return await self.fetch_funding_rate(symbol)

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Fetch current 8-hour funding rate for a linear perp symbol (CCXT)."""
        try:
            ex = self._get_exchange()
            info = ex.fetch_funding_rate(symbol)
            return {
                "symbol": symbol,
                "funding_rate": float(info.get("fundingRate", 0.0) or 0.0),
                "next_funding_time": info.get("fundingDatetime"),
                "timestamp": info.get("timestamp"),
                "exchange": "bybit",
            }
        except Exception as exc:
            logger.warning("fetch_funding_rate %s failed: %s", symbol, exc)
            return {"symbol": symbol, "funding_rate": 0.0, "exchange": "bybit", "error": str(exc)}

    async def fetch_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch funding rates for multiple symbols. Returns {symbol: rate}."""
        rates: Dict[str, float] = {}
        for sym in symbols:
            info = await self.get_funding_rate(sym)
            rates[sym] = info.get("funding_rate", 0.0)
        return rates

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price/bid/ask for a symbol.

        Returns dict with: last, bid, ask, volume, symbol
        """
        api_symbol = self._normalise_symbol(symbol)
        try:
            result = await self._v5_get(
                "/v5/market/tickers",
                {"category": "linear", "symbol": api_symbol},
            )
            tickers = result.get("list", [])
            if tickers:
                t = tickers[0]
                return {
                    "symbol": symbol,
                    "last": float(t.get("lastPrice", 0) or 0),
                    "bid": float(t.get("bid1Price", 0) or 0),
                    "ask": float(t.get("ask1Price", 0) or 0),
                    "bid_size": float(t.get("bid1Size", 0) or 0),
                    "ask_size": float(t.get("ask1Size", 0) or 0),
                    "volume_24h": float(t.get("turnover24h", 0) or 0),
                    "funding_rate": float(t.get("fundingRate", 0) or 0),
                }
            return None
        except Exception:
            # CCXT fallback
            try:
                return self._get_exchange().fetch_ticker(symbol)
            except Exception as exc:
                logger.warning("Bybit get_ticker %s failed: %s", symbol, exc)
                return None

    # ------------------------------------------------------------------
    # Account / Balance
    # ------------------------------------------------------------------

    async def get_balance(self) -> Dict[str, Any]:
        """
        Get USDT balance for the unified trading account.

        Returns:
            Dict with USDT free/total balances
        """
        try:
            result = await self._v5_get(
                "/v5/account/wallet-balance",
                {"accountType": "UNIFIED"},
                auth=True,
            )
            accounts = result.get("list", [])
            for acct in accounts:
                for coin in acct.get("coin", []):
                    if coin.get("coin") == "USDT":
                        return {
                            "USDT": {
                                "free": float(coin.get("availableToWithdraw", 0) or 0),
                                "total": float(coin.get("walletBalance", 0) or 0),
                                "equity": float(coin.get("equity", 0) or 0),
                                "unrealized_pnl": float(coin.get("unrealisedPnl", 0) or 0),
                            }
                        }
            return {"USDT": {"free": 0.0, "total": 0.0}}
        except Exception:
            # CCXT fallback
            try:
                bal = self._get_exchange().fetch_balance({"type": "linear"})
                usdt = bal.get("USDT", {})
                return {"USDT": {"free": usdt.get("free", 0.0), "total": usdt.get("total", 0.0)}}
            except Exception as exc:
                logger.warning("get_balance failed: %s", exc)
                return {}

    # ------------------------------------------------------------------
    # Position close
    # ------------------------------------------------------------------

    async def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Close an open position at market."""
        try:
            pos = await self.get_position(symbol)
            if not pos or pos.get("size", 0) == 0:
                logger.info("No open position to close for %s", symbol)
                return None
            side = "Sell" if pos.get("side") in ("long", "buy", "Buy") else "Buy"
            amount = abs(pos.get("size", 0))
            return await self.place_order(symbol, side, amount, "Market", reduce_only=True)
        except Exception as exc:
            logger.error("close_position %s failed: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """
        Convert CCXT-style symbol to Bybit V5 API symbol.

        "BTC/USDT:USDT" -> "BTCUSDT"
        "BTC/USDT"      -> "BTCUSDT"
        "BTCUSDT"        -> "BTCUSDT" (no-op)
        """
        s = symbol.replace(":USDT", "").replace("/", "")
        return s
