"""
dYdX v4 (Cosmos-based) Perpetual Futures Connector.

dYdX v4 is a fully decentralized orderbook exchange built on a sovereign
Cosmos appchain. Key features:
- Zero gas fees for trading (fees are in USDC)
- Fully on-chain orderbook
- Perpetual markets only (no spot)
- Uses USD denomination (not USDT)
- Deep liquidity for BTC, ETH, SOL and 30+ markets

Public data requires NO authentication (REST indexer API).
Private data (positions, fills, orders) uses API key authentication.

REST indexer: https://indexer.dydx.trade/v4
Env vars: DYDX_API_KEY, DYDX_API_SECRET, DYDX_PASSPHRASE (optional for public)
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# dYdX v4 indexer base URL
DYDX_V4_INDEXER = "https://indexer.dydx.trade/v4"

# Rate limits — dYdX indexer is generous
_PUBLIC_RATE_LIMIT = 100   # 100 req/s for public
_PRIVATE_RATE_LIMIT = 20   # 20 req/s for private (conservative)

# Retry settings
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 0.5


class DYDXRateLimiter:
    """Token-bucket rate limiter for dYdX API calls."""

    def __init__(self, max_per_second: int = 100):
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


class DYDXAPIError(Exception):
    """Raised when dYdX indexer API returns an error."""

    def __init__(self, message: str, status_code: int = 0, endpoint: str = ""):
        self.status_code = status_code
        self.endpoint = endpoint
        super().__init__(f"dYdX API error (HTTP {status_code}): {message} [endpoint={endpoint}]")


class DYDXv4Connector:
    """
    dYdX v4 perpetual futures connector using the indexer REST API.

    Provides async methods for market data (no auth) and authenticated
    account queries via aiohttp.
    """

    health_check_symbol = "BTC-USD"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        base_url: str = DYDX_V4_INDEXER,
    ):
        self.api_key = api_key or os.environ.get("DYDX_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("DYDX_API_SECRET", "")
        self.passphrase = passphrase or os.environ.get("DYDX_PASSPHRASE", "")
        self.base_url = base_url.rstrip("/")
        self._session: Any = None  # aiohttp.ClientSession
        self.connected: bool = False

        # Rate limiters
        self._public_limiter = DYDXRateLimiter(_PUBLIC_RATE_LIMIT)
        self._private_limiter = DYDXRateLimiter(_PRIVATE_RATE_LIMIT)

    # ------------------------------------------------------------------
    # Symbol mapping
    # ------------------------------------------------------------------

    @staticmethod
    def to_dydx_symbol(symbol: str) -> str:
        """
        Convert standard symbol to dYdX market ticker.

        "BTC/USD" -> "BTC-USD"
        "ETH/USDT" -> "ETH-USD" (dYdX uses USD, not USDT)
        "BTC-USD" -> "BTC-USD" (no-op)
        """
        # Already in dYdX format
        if "/" not in symbol and "-" in symbol:
            # Normalize USDT -> USD
            return symbol.replace("-USDT", "-USD")

        # Strip CCXT perp suffix
        s = symbol.replace(":USDT", "").replace(":USD", "")
        s = s.replace("/", "-")
        # dYdX uses USD denomination
        s = s.replace("-USDT", "-USD")
        return s

    @staticmethod
    def from_dydx_symbol(ticker: str) -> str:
        """
        Convert dYdX market ticker to standard symbol.

        "BTC-USD" -> "BTC/USD"
        """
        return ticker.replace("-", "/")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Generate HMAC-SHA256 signature for dYdX private API.

        Sign string = timestamp + method + path + body
        """
        if not self.api_secret:
            return ""
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Build authenticated headers for dYdX private requests."""
        timestamp = str(int(time.time()))
        signature = self._sign(timestamp, method, path, body)
        return {
            "DYDX-SIGNATURE": signature,
            "DYDX-API-KEY": self.api_key,
            "DYDX-TIMESTAMP": timestamp,
            "DYDX-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    @property
    def has_auth(self) -> bool:
        """Whether API keys are configured for private endpoints."""
        return bool(self.api_key and self.api_secret)

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                logger.error("aiohttp not available — cannot use DYDXv4Connector")
                raise RuntimeError("aiohttp is required for DYDXv4Connector")
        return self._session

    async def _public_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a public GET request to dYdX indexer.

        Args:
            endpoint: API path, e.g. "/perpetualMarkets"
            params: Query parameters

        Returns:
            Parsed JSON response
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        await self._public_limiter.acquire()

        for attempt in range(1, _MAX_RETRIES + 1):
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status == 429:
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                        logger.warning(
                            "dYdX rate limited on %s, retrying in %.1fs (attempt %d/%d)",
                            endpoint, delay, attempt, _MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise DYDXAPIError("Rate limited after max retries", 429, endpoint)

                if resp.status >= 400:
                    text = await resp.text()
                    raise DYDXAPIError(text, resp.status, endpoint)

                return await resp.json()

        raise DYDXAPIError("Max retries exceeded", 0, endpoint)

    async def _private_get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make an authenticated GET request to dYdX indexer.

        Args:
            endpoint: API path
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            DYDXAPIError: On auth or API errors
        """
        if not self.has_auth:
            raise DYDXAPIError(
                "API key and secret are required for private endpoints", 401, endpoint
            )

        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"

        # Build query string for signing
        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        sign_path = endpoint + query_string
        headers = self._auth_headers("GET", sign_path)

        await self._private_limiter.acquire()

        for attempt in range(1, _MAX_RETRIES + 1):
            async with session.get(
                url, params=params, headers=headers, timeout=15
            ) as resp:
                if resp.status == 429:
                    if attempt < _MAX_RETRIES:
                        delay = _RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                        continue
                    raise DYDXAPIError("Rate limited after max retries", 429, endpoint)

                if resp.status in (401, 403):
                    text = await resp.text()
                    raise DYDXAPIError(f"Auth failed: {text}", resp.status, endpoint)

                if resp.status >= 400:
                    text = await resp.text()
                    raise DYDXAPIError(text, resp.status, endpoint)

                return await resp.json()

        raise DYDXAPIError("Max retries exceeded", 0, endpoint)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect and verify API access by pinging a public endpoint."""
        try:
            result = await self._public_get(
                "/perpetualMarkets",
                {"ticker": "BTC-USD"},
            )
            if result and "markets" in result:
                self.connected = True
                logger.info("dYdX v4 indexer connected")
                return True
            self.connected = False
            return False
        except Exception as exc:
            logger.error("dYdX v4 connect failed: %s", exc)
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect and clean up resources."""
        self.connected = False
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("dYdX v4 disconnected")

    async def __aenter__(self) -> "DYDXv4Connector":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Public endpoints — Market data (no auth)
    # ------------------------------------------------------------------

    async def get_markets(self) -> Dict[str, Any]:
        """
        Get all available perpetual markets.

        Returns:
            Dict of market ticker -> market info
        """
        try:
            result = await self._public_get("/perpetualMarkets")
            return result.get("markets", {})
        except Exception as exc:
            logger.warning("dYdX get_markets failed: %s", exc)
            return {}

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair, e.g. "BTC/USD" or "BTC-USD"

        Returns:
            {bid, ask, last, volume_24h, funding_rate, open_interest, symbol, exchange}
        """
        ticker = self.to_dydx_symbol(symbol)
        try:
            result = await self._public_get(
                "/perpetualMarkets",
                {"ticker": ticker},
            )
            markets = result.get("markets", {})
            if ticker in markets:
                m = markets[ticker]
                # Get orderbook for bid/ask
                ob = await self.get_orderbook(symbol, depth=1)
                bid = ob["bids"][0][0] if ob and ob.get("bids") else 0.0
                ask = ob["asks"][0][0] if ob and ob.get("asks") else 0.0

                return {
                    "symbol": symbol,
                    "ticker": ticker,
                    "last": float(m.get("oraclePrice", 0) or 0),
                    "bid": bid,
                    "ask": ask,
                    "volume_24h": float(m.get("volume24H", 0) or 0),
                    "open_interest": float(m.get("openInterest", 0) or 0),
                    "funding_rate": float(m.get("nextFundingRate", 0) or 0),
                    "index_price": float(m.get("oraclePrice", 0) or 0),
                    "price_change_24h": float(m.get("priceChange24H", 0) or 0),
                    "exchange": "dydx",
                }
            return None
        except Exception as exc:
            logger.warning("dYdX get_ticker %s failed: %s", symbol, exc)
            return None

    async def get_orderbook(
        self, symbol: str, depth: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        Get orderbook for a symbol.

        Args:
            symbol: Trading pair
            depth: Number of levels per side (not used by API, returns all)

        Returns:
            {bids: [(price, qty), ...], asks: [(price, qty), ...], symbol}
        """
        ticker = self.to_dydx_symbol(symbol)
        try:
            result = await self._public_get(f"/orderbooks/perpetualMarket/{ticker}")
            bids = [
                (float(b.get("price", 0)), float(b.get("size", 0)))
                for b in result.get("bids", [])[:depth]
            ]
            asks = [
                (float(a.get("price", 0)), float(a.get("size", 0)))
                for a in result.get("asks", [])[:depth]
            ]
            return {
                "symbol": symbol,
                "ticker": ticker,
                "bids": bids,
                "asks": asks,
                "exchange": "dydx",
            }
        except Exception as exc:
            logger.warning("dYdX get_orderbook %s failed: %s", symbol, exc)
            return None

    async def get_funding_rates(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical funding rates for a symbol.

        Args:
            symbol: Trading pair
            limit: Number of historical rates to return

        Returns:
            List of {rate, effective_at, ticker} sorted newest first
        """
        ticker = self.to_dydx_symbol(symbol)
        try:
            result = await self._public_get(
                f"/historicalFunding/{ticker}",
                {"limit": str(limit)},
            )
            rates: List[Dict[str, Any]] = []
            for fr in result.get("historicalFunding", []):
                rates.append({
                    "ticker": ticker,
                    "rate": float(fr.get("rate", 0) or 0),
                    "effective_at": fr.get("effectiveAt", ""),
                    "exchange": "dydx",
                })
            return rates
        except Exception as exc:
            logger.warning("dYdX get_funding_rates %s failed: %s", symbol, exc)
            return []

    async def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Get current funding rate for a symbol.

        Uses perpetualMarkets endpoint for next funding rate.

        Returns:
            {funding_rate, next_funding_rate, symbol, exchange}
        """
        ticker = self.to_dydx_symbol(symbol)
        try:
            result = await self._public_get(
                "/perpetualMarkets",
                {"ticker": ticker},
            )
            markets = result.get("markets", {})
            if ticker in markets:
                m = markets[ticker]
                return {
                    "symbol": symbol,
                    "ticker": ticker,
                    "funding_rate": float(m.get("nextFundingRate", 0) or 0),
                    "next_funding_rate": float(m.get("nextFundingRate", 0) or 0),
                    "exchange": "dydx",
                }
            return {"symbol": symbol, "funding_rate": 0.0, "exchange": "dydx"}
        except Exception as exc:
            logger.warning("dYdX get_funding_rate %s failed: %s", symbol, exc)
            return {"symbol": symbol, "funding_rate": 0.0, "exchange": "dydx", "error": str(exc)}

    # Alias for compatibility
    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """Alias for get_funding_rate()."""
        return await self.get_funding_rate(symbol)

    async def fetch_funding_rates(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch funding rates for multiple symbols. Returns {symbol: rate}."""
        rates: Dict[str, float] = {}
        for sym in symbols:
            info = await self.get_funding_rate(sym)
            rates[sym] = info.get("funding_rate", 0.0)
        return rates

    # ------------------------------------------------------------------
    # Private endpoints — Account data
    # ------------------------------------------------------------------

    async def get_balances(self, address: str = "") -> Dict[str, Any]:
        """
        Get account balances (subaccount equity, free collateral, margin).

        Args:
            address: dYdX address (uses API key's default if empty)

        Returns:
            {equity, free_collateral, margin_used, open_perpetual_positions_count}
        """
        if not address and not self.has_auth:
            return {"equity": 0.0, "free_collateral": 0.0, "margin_used": 0.0}

        try:
            # dYdX v4 indexer uses address-based subaccounts
            endpoint = f"/addresses/{address}/subaccountNumber/0" if address else "/accounts"
            result = await self._private_get(endpoint)

            # Handle different response shapes
            subaccount = result
            if isinstance(result, dict) and "subaccount" in result:
                subaccount = result["subaccount"]
            elif isinstance(result, dict) and "subaccounts" in result:
                subs = result["subaccounts"]
                subaccount = subs[0] if subs else {}

            return {
                "equity": float(subaccount.get("equity", 0) or 0),
                "free_collateral": float(subaccount.get("freeCollateral", 0) or 0),
                "margin_used": float(subaccount.get("marginEnabled", 0) or 0),
                "open_perpetual_positions_count": int(
                    subaccount.get("openPerpetualPositionsCount", 0) or 0
                ),
                "exchange": "dydx",
            }
        except Exception as exc:
            logger.warning("dYdX get_balances failed: %s", exc)
            return {"equity": 0.0, "free_collateral": 0.0, "margin_used": 0.0}

    async def get_balance(self) -> Dict[str, Any]:
        """Alias for get_balances() — ExchangeManager compatibility."""
        return await self.get_balances()

    async def get_positions(self, address: str = "") -> List[Dict[str, Any]]:
        """
        Get all open perpetual positions.

        Args:
            address: dYdX address

        Returns:
            List of position dicts
        """
        if not self.has_auth and not address:
            return []

        try:
            endpoint = f"/addresses/{address}/subaccountNumber/0" if address else "/accounts"
            result = await self._private_get(endpoint)

            subaccount = result
            if isinstance(result, dict) and "subaccount" in result:
                subaccount = result["subaccount"]
            elif isinstance(result, dict) and "subaccounts" in result:
                subs = result["subaccounts"]
                subaccount = subs[0] if subs else {}

            perp_positions = subaccount.get("openPerpetualPositions", {})
            positions: List[Dict[str, Any]] = []

            for ticker, pos in perp_positions.items():
                size = float(pos.get("size", 0) or 0)
                if size != 0:
                    positions.append({
                        "symbol": self.from_dydx_symbol(ticker),
                        "ticker": ticker,
                        "size": abs(size),
                        "side": "long" if size > 0 else "short",
                        "entry_price": float(pos.get("entryPrice", 0) or 0),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                        "realized_pnl": float(pos.get("realizedPnl", 0) or 0),
                        "sum_open": float(pos.get("sumOpen", 0) or 0),
                        "sum_close": float(pos.get("sumClose", 0) or 0),
                        "exchange": "dydx",
                    })
            return positions
        except Exception as exc:
            logger.warning("dYdX get_positions failed: %s", exc)
            return []

    async def get_fills(
        self, symbol: str, address: str = "", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent fills (trade executions) for a symbol.

        Args:
            symbol: Trading pair
            address: dYdX address
            limit: Max fills to return

        Returns:
            List of fill dicts
        """
        if not self.has_auth and not address:
            return []

        ticker = self.to_dydx_symbol(symbol)
        try:
            params: Dict[str, Any] = {
                "market": ticker,
                "limit": str(limit),
            }
            endpoint = (
                f"/addresses/{address}/subaccountNumber/0/fills"
                if address
                else "/fills"
            )
            result = await self._private_get(endpoint, params)

            fills: List[Dict[str, Any]] = []
            for f in result.get("fills", []):
                fills.append({
                    "fill_id": f.get("id", ""),
                    "symbol": self.from_dydx_symbol(f.get("market", ticker)),
                    "ticker": f.get("market", ticker),
                    "side": f.get("side", "").lower(),
                    "size": float(f.get("size", 0) or 0),
                    "price": float(f.get("price", 0) or 0),
                    "fee": float(f.get("fee", 0) or 0),
                    "type": f.get("type", ""),
                    "created_at": f.get("createdAt", ""),
                    "exchange": "dydx",
                })
            return fills
        except Exception as exc:
            logger.warning("dYdX get_fills %s failed: %s", symbol, exc)
            return []

    async def get_orders(
        self, symbol: str, address: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Get open orders for a symbol.

        Args:
            symbol: Trading pair
            address: dYdX address

        Returns:
            List of open order dicts
        """
        if not self.has_auth and not address:
            return []

        ticker = self.to_dydx_symbol(symbol)
        try:
            params: Dict[str, Any] = {
                "ticker": ticker,
                "status": "OPEN",
            }
            endpoint = (
                f"/addresses/{address}/subaccountNumber/0/orders"
                if address
                else "/orders"
            )
            result = await self._private_get(endpoint, params)

            orders: List[Dict[str, Any]] = []
            for o in (result if isinstance(result, list) else result.get("orders", [])):
                orders.append({
                    "order_id": o.get("id", ""),
                    "client_id": o.get("clientId", ""),
                    "symbol": self.from_dydx_symbol(o.get("ticker", ticker)),
                    "ticker": o.get("ticker", ticker),
                    "side": o.get("side", "").lower(),
                    "size": float(o.get("size", 0) or 0),
                    "price": float(o.get("price", 0) or 0),
                    "status": o.get("status", ""),
                    "type": o.get("type", ""),
                    "created_at": o.get("createdAtHeight", ""),
                    "exchange": "dydx",
                })
            return orders
        except Exception as exc:
            logger.warning("dYdX get_orders %s failed: %s", symbol, exc)
            return []

    # ------------------------------------------------------------------
    # Funding rate analysis — cross-venue arb
    # ------------------------------------------------------------------

    async def get_funding_opportunity(
        self,
        bybit_rates: Optional[Dict[str, float]] = None,
        okx_rates: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare dYdX funding rates against Bybit and OKX to find arb opportunities.

        The strategy: go long on the venue with the most negative (or least positive)
        funding rate and short on the venue with the most positive rate.

        Args:
            bybit_rates: {symbol: rate} from Bybit (optional)
            okx_rates: {symbol: rate} from OKX (optional)

        Returns:
            Best arb opportunity dict or None if spread is too small
        """
        dydx_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        best_opportunity: Optional[Dict[str, Any]] = None
        best_spread = 0.0

        for dydx_sym in dydx_symbols:
            standard_sym = self.from_dydx_symbol(dydx_sym)

            # Get dYdX rate
            dydx_rate_info = await self.get_funding_rate(dydx_sym)
            dydx_rate = dydx_rate_info.get("funding_rate", 0.0)

            # Collect rates from all venues
            venue_rates: Dict[str, float] = {"dydx": dydx_rate}

            if bybit_rates:
                # Map standard symbol to bybit
                for sym, rate in bybit_rates.items():
                    base = sym.split("/")[0] if "/" in sym else sym.split("-")[0]
                    if base == dydx_sym.split("-")[0]:
                        venue_rates["bybit"] = rate
                        break

            if okx_rates:
                for sym, rate in okx_rates.items():
                    base = sym.split("/")[0] if "/" in sym else sym.split("-")[0]
                    if base == dydx_sym.split("-")[0]:
                        venue_rates["okx"] = rate
                        break

            if len(venue_rates) < 2:
                continue

            # Find max spread: long on lowest rate, short on highest rate
            min_venue = min(venue_rates, key=lambda v: venue_rates[v])
            max_venue = max(venue_rates, key=lambda v: venue_rates[v])

            spread = venue_rates[max_venue] - venue_rates[min_venue]

            if spread > best_spread:
                best_spread = spread
                spread_bps = spread * 10_000  # Convert to basis points
                # Annualized: 3 settlements per day * 365
                annualized_apr = spread * 3 * 365 * 100

                best_opportunity = {
                    "symbol": standard_sym,
                    "long_venue": min_venue,
                    "short_venue": max_venue,
                    "long_rate": venue_rates[min_venue],
                    "short_rate": venue_rates[max_venue],
                    "spread": spread,
                    "spread_bps": round(spread_bps, 2),
                    "annualized_apr": round(annualized_apr, 2),
                    "venue_rates": dict(venue_rates),
                }

        return best_opportunity

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Ping dYdX indexer to verify connectivity."""
        t0 = time.perf_counter()
        try:
            result = await self._public_get(
                "/perpetualMarkets",
                {"ticker": "BTC-USD"},
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            healthy = result is not None and "markets" in (result or {})
            return {
                "healthy": healthy,
                "latency_ms": round(latency_ms, 2),
                "exchange": "dydx",
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "healthy": False,
                "latency_ms": round(latency_ms, 2),
                "exchange": "dydx",
                "error": str(exc),
            }
