"""
dYdX v4 Exchange Client
========================

Production-quality REST + WebSocket client for dYdX v4 (decentralised
perpetuals exchange, CosmosSDK chain), compatible with the Argus
exchange interface pattern.

Key characteristics:
- Self-custodial: no KYC required, wallet-based authentication
- Perpetuals only (no spot markets)
- Maker fee: 0.01%  Taker fee: 0.05%
- Decentralised: CosmosSDK chain — no custody risk
- REST indexer: https://indexer.dydx.trade/v4
- WebSocket: wss://indexer.dydx.trade/v4/ws
- Auth: Cosmos private key signing (not API key)

Order signing:
  dYdX v4 uses Cosmos wallet signing. Orders are broadcast as on-chain
  transactions. This client supports both mnemonic (BIP-39) and raw
  private key (hex) initialisation.

  If the dydx-v4-client Python SDK is not installed, order placement
  raises NotImplementedError with a helpful message.

REST indexer endpoints are public (no auth needed for market data).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

log = logging.getLogger("argus.dydx_client")

# ---------------------------------------------------------------------------
# Module-level fee & URL constants
# ---------------------------------------------------------------------------

DYDX_MAKER_FEE: float = 0.0001     # 0.01%
DYDX_TAKER_FEE: float = 0.0005     # 0.05%

DYDX_INDEXER_URL: str = "https://indexer.dydx.trade/v4"
DYDX_WS_URL: str = "wss://indexer.dydx.trade/v4/ws"

DYDX_INDEXER_TESTNET_URL: str = "https://indexer.v4testnet.dydx.exchange/v4"
DYDX_WS_TESTNET_URL: str = "wss://indexer.v4testnet.dydx.exchange/v4/ws"

# dYdX node RPC (for on-chain transactions)
DYDX_NODE_URL: str = "https://dydx-ops-rpc.kingnodes.com:443"
DYDX_NODE_TESTNET_URL: str = "https://dydx-testnet-archive.allthatnode.com:26657"

DYDX_IS_DECENTRALISED: bool = True
DYDX_REQUIRES_KYC: bool = False

# Chain IDs
DYDX_CHAIN_ID: str = "dydx-mainnet-1"
DYDX_CHAIN_ID_TESTNET: str = "dydx-testnet-4"

# Rate limit: conservative 9 req/s for indexer
_RATE_LIMIT_PER_SECOND: float = 9.0


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_exchange_info() -> Dict[str, Any]:
    """Return a dict with dYdX fee constants and endpoint info."""
    return {
        "exchange": "dydx",
        "display_name": "dYdX v4",
        "fee_rates": {
            "maker": DYDX_MAKER_FEE,
            "taker": DYDX_TAKER_FEE,
        },
        "urls": {
            "indexer_rest": DYDX_INDEXER_URL,
            "ws": DYDX_WS_URL,
            "node_rpc": DYDX_NODE_URL,
        },
        "properties": {
            "is_decentralised": DYDX_IS_DECENTRALISED,
            "requires_kyc": DYDX_REQUIRES_KYC,
            "custody": "self-custodial",
            "markets": "perpetuals_only",
        },
        "notes": (
            "Self-custodial perps on CosmosSDK. "
            "0.01% maker (near-zero), 0.05% taker. "
            "No KYC required. "
            "Orders require private key signing — no API key custody."
        ),
    }


# ---------------------------------------------------------------------------
# Symbol normalisation helpers (module-level)
# ---------------------------------------------------------------------------

def to_dydx_symbol(symbol: str) -> str:
    """
    Convert a symbol to dYdX v4 format (dash-separated, e.g. "BTC-USD").

    Examples
    --------
    "BTC/USD"   →  "BTC-USD"
    "BTC/USDT"  →  "BTC-USD"  (USDT→USD normalisation)
    "BTCUSD"    →  "BTC-USD"
    "BTC-USD"   →  "BTC-USD"  (no-op)
    """
    symbol = symbol.strip().upper()
    if "-" in symbol:
        return symbol
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        # dYdX uses USD not USDT
        quote = "USD" if quote in ("USDT", "USDC", "USD") else quote
        return f"{base}-{quote}"
    # Concatenated — try known patterns
    for quote in ("USDT", "USDC", "USD"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: len(symbol) - len(quote)]
            return f"{base}-USD"
    if len(symbol) == 6:
        return f"{symbol[:3]}-{symbol[3:]}"
    return symbol


def from_dydx_symbol(symbol: str) -> str:
    """
    Convert a dYdX symbol to slash-separated format.

    Example:  "BTC-USD"  →  "BTC/USD"
    """
    return symbol.replace("-", "/")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DYDXAPIError(Exception):
    """Raised when the dYdX indexer returns an error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"dYdX API error {code}: {message}")
        self.code: int = code
        self.message: str = message

    def __repr__(self) -> str:  # pragma: no cover
        return f"DYDXAPIError(code={self.code}, message={self.message!r})"


class DYDXOrderSigningError(Exception):
    """Raised when order signing fails (missing key or signing library)."""
    pass


# ---------------------------------------------------------------------------
# Internal: token-bucket rate limiter
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, capacity: float = _RATE_LIMIT_PER_SECOND) -> None:
        self._capacity = capacity
        self._tokens: float = capacity
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._capacity)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._capacity
        await asyncio.sleep(wait)
        await self.acquire()


# ---------------------------------------------------------------------------
# DYDXClient
# ---------------------------------------------------------------------------

class DYDXClient:
    """
    Async REST + WebSocket client for dYdX v4 (decentralised perps).

    Parameters
    ----------
    wallet_address:
        dYdX / Cosmos wallet address (bech32, e.g. "dydx1abc...").
        Required for order placement; optional for read-only market data.
    mnemonic:
        BIP-39 mnemonic phrase for key derivation.
        One of mnemonic or private_key_hex must be provided for orders.
    private_key_hex:
        Raw 32-byte private key as a hex string (64 hex chars).
        One of mnemonic or private_key_hex must be provided for orders.
    testnet:
        If True, connect to dYdX v4 testnet.
    """

    def __init__(
        self,
        wallet_address: str,
        mnemonic: Optional[str] = None,
        private_key_hex: Optional[str] = None,
        testnet: bool = False,
    ) -> None:
        self._wallet_address = wallet_address
        self._mnemonic = mnemonic
        self._private_key_hex = private_key_hex
        self._testnet = testnet

        self._has_signing_key = bool(mnemonic or private_key_hex)

        if testnet:
            self._indexer_url = DYDX_INDEXER_TESTNET_URL
            self._ws_url = DYDX_WS_TESTNET_URL
            self._node_url = DYDX_NODE_TESTNET_URL
            self._chain_id = DYDX_CHAIN_ID_TESTNET
            log.info("DYDXClient: using testnet endpoints")
        else:
            self._indexer_url = DYDX_INDEXER_URL
            self._ws_url = DYDX_WS_URL
            self._node_url = DYDX_NODE_URL
            self._chain_id = DYDX_CHAIN_ID

        self._rate_limiter = _TokenBucket()
        self._session: Optional[aiohttp.ClientSession] = None

        # Active WebSocket tasks
        self._ws_tasks: Dict[str, asyncio.Task] = {}

        # Cache for account sequence number (needed for tx signing)
        self._account_sequence: int = 0
        self._account_number: int = 0

        if not self._has_signing_key:
            log.info(
                "DYDXClient: no private key provided — order methods will raise "
                "NotImplementedError. Market data methods are fully available."
            )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session and cancel WS tasks."""
        for task in self._ws_tasks.values():
            task.cancel()
        self._ws_tasks.clear()
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def __aenter__(self) -> "DYDXClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_custodial(self) -> bool:
        """
        Return False — dYdX is always self-custodial.

        Unlike centralised exchanges, dYdX never takes custody of funds.
        Assets remain in the user's wallet and are secured by the
        CosmosSDK chain.
        """
        return False

    def get_estimated_fees(self, size_usd: float) -> Dict[str, float]:
        """
        Estimate trading fees for a given notional USD size.

        Parameters
        ----------
        size_usd:
            Notional position size in USD.

        Returns
        -------
        dict with keys: maker_fee_usd, taker_fee_usd, maker_fee_pct, taker_fee_pct
        """
        return {
            "maker_fee_usd": size_usd * DYDX_MAKER_FEE,
            "taker_fee_usd": size_usd * DYDX_TAKER_FEE,
            "maker_fee_pct": DYDX_MAKER_FEE * 100,
            "taker_fee_pct": DYDX_TAKER_FEE * 100,
            "maker_fee_bps": DYDX_MAKER_FEE * 10_000,
            "taker_fee_bps": DYDX_TAKER_FEE * 10_000,
            "note": "Fees may be lower or negative at high volume tiers.",
        }

    def _require_signing_key(self, operation: str) -> None:
        """Raise NotImplementedError if no signing key was provided."""
        if not self._has_signing_key:
            raise NotImplementedError(
                f"Cannot {operation}: no private key was provided to DYDXClient.\n"
                "To enable order placement, initialise with either:\n"
                "  DYDXClient(wallet_address=..., mnemonic='your twelve word mnemonic ...')\n"
                "  DYDXClient(wallet_address=..., private_key_hex='deadbeef...')\n"
                "Market data methods (fetch_ticker, fetch_order_book, etc.) "
                "work without a key."
            )

    # ------------------------------------------------------------------
    # Low-level HTTP (indexer)
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Perform a rate-limited HTTP request to the dYdX indexer.

        Parameters
        ----------
        method:
            HTTP method ("GET", "POST").
        path:
            Indexer API path, e.g. "/perpetualMarkets".
        params:
            Query-string parameters.
        json_body:
            JSON request body.

        Raises
        ------
        DYDXAPIError
            If the indexer returns a non-200 response with an error.
        """
        await self._rate_limiter.acquire()

        session = await self._get_session()
        url = f"{self._indexer_url}{path}"

        for attempt in range(1, 4):
            try:
                async with session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers={"Accept": "application/json"},
                ) as resp:
                    raw = await resp.text()

                    if resp.status == 429:
                        wait = 1.0 * attempt
                        log.warning("dYdX indexer rate limited, retry in %.1fs", wait)
                        await asyncio.sleep(wait)
                        continue

                    if resp.status >= 400:
                        try:
                            err = json.loads(raw)
                            msg = err.get("errors", [{}])[0].get("msg", raw) if err.get("errors") else err.get("message", raw)
                        except Exception:
                            msg = raw
                        raise DYDXAPIError(resp.status, str(msg))

                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return raw

            except DYDXAPIError:
                raise
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                if attempt < 3:
                    await asyncio.sleep(1.0 * attempt)
                    continue
                raise DYDXAPIError(-1, f"Network error: {exc}") from exc

        raise DYDXAPIError(-1, "Max retries exceeded")

    # ------------------------------------------------------------------
    # REST — market data
    # ------------------------------------------------------------------

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker (bid/ask, last, 24h volume) for *symbol*.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USD", "BTC-USD".

        Returns
        -------
        dict with keys: symbol, bid, ask, last, volume, open_interest,
                        funding_rate, timestamp, exchange
        """
        dydx_sym = to_dydx_symbol(symbol)
        raw = await self._request(
            "GET",
            f"/perpetualMarkets",
            params={"ticker": dydx_sym},
        )

        markets = raw.get("markets", {})
        market = markets.get(dydx_sym, {})

        return {
            "symbol": from_dydx_symbol(dydx_sym),
            "bid": float(market.get("bidPrice", market.get("oraclePrice", 0)) or 0),
            "ask": float(market.get("askPrice", market.get("oraclePrice", 0)) or 0),
            "last": float(market.get("lastTradedPrice", market.get("oraclePrice", 0)) or 0),
            "oracle_price": float(market.get("oraclePrice", 0) or 0),
            "volume": float(market.get("volume24H", 0) or 0),
            "trades_24h": int(market.get("trades24H", 0) or 0),
            "open_interest": float(market.get("openInterest", 0) or 0),
            "funding_rate": float(market.get("nextFundingRate", 0) or 0),
            "status": market.get("status", ""),
            "timestamp": int(time.time() * 1000),
            "exchange": "dydx",
            "raw": market,
        }

    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch order book depth from the dYdX indexer.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USD", "BTC-USD".
        limit:
            Number of price levels per side to return.

        Returns
        -------
        dict with keys: symbol, bids [[price, qty]...], asks [[price, qty]...], timestamp
        """
        dydx_sym = to_dydx_symbol(symbol)
        raw = await self._request(
            "GET",
            f"/orderbooks/perpetualMarket/{dydx_sym}",
        )

        raw_bids = raw.get("bids", [])
        raw_asks = raw.get("asks", [])

        bids = sorted(
            [
                [float(entry.get("price", 0)), float(entry.get("size", 0))]
                for entry in raw_bids
            ],
            key=lambda x: x[0],
            reverse=True,
        )[:limit]

        asks = sorted(
            [
                [float(entry.get("price", 0)), float(entry.get("size", 0))]
                for entry in raw_asks
            ],
            key=lambda x: x[0],
        )[:limit]

        return {
            "symbol": from_dydx_symbol(dydx_sym),
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000),
            "exchange": "dydx",
        }

    async def fetch_account(self) -> Dict[str, Any]:
        """
        Fetch account balance, equity, and margin information.

        Returns
        -------
        dict with keys: wallet_address, equity, free_collateral,
                        open_positions, timestamp, exchange
        """
        if not self._wallet_address:
            raise DYDXAPIError(400, "wallet_address required for fetch_account")

        raw = await self._request(
            "GET",
            f"/addresses/{self._wallet_address}",
        )

        subaccounts = raw.get("subaccounts", [])
        if not subaccounts:
            return {
                "wallet_address": self._wallet_address,
                "equity": 0.0,
                "free_collateral": 0.0,
                "open_positions": 0,
                "timestamp": int(time.time() * 1000),
                "exchange": "dydx",
                "raw": raw,
            }

        # Use subaccount 0 (default)
        sub = subaccounts[0]
        return {
            "wallet_address": self._wallet_address,
            "subaccount_number": sub.get("subaccountNumber", 0),
            "equity": float(sub.get("equity", 0) or 0),
            "free_collateral": float(sub.get("freeCollateral", 0) or 0),
            "margin_usage": float(sub.get("marginUsage", 0) or 0),
            "leverage": float(sub.get("leverage", 0) or 0),
            "open_positions": int(sub.get("openPerpetualPositions", {}) and
                                  len(sub.get("openPerpetualPositions", {})) or 0),
            "timestamp": int(time.time() * 1000),
            "exchange": "dydx",
            "raw": raw,
        }

    async def fetch_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch all open perpetual positions for the wallet.

        Returns
        -------
        list of position dicts.
        """
        if not self._wallet_address:
            return []

        raw = await self._request(
            "GET",
            f"/perpetualPositions",
            params={
                "address": self._wallet_address,
                "subaccountNumber": 0,
                "status": "OPEN",
            },
        )

        positions = []
        for entry in raw.get("positions", []):
            size = float(entry.get("size", 0) or 0)
            positions.append({
                "symbol": from_dydx_symbol(entry.get("market", "")),
                "market": entry.get("market", ""),
                "side": entry.get("side", "LONG"),
                "size": abs(size),
                "size_signed": size,
                "entry_price": float(entry.get("entryPrice", 0) or 0),
                "realized_pnl": float(entry.get("realizedPnl", 0) or 0),
                "unrealized_pnl": float(entry.get("unrealizedPnl", 0) or 0),
                "created_at": entry.get("createdAt", ""),
                "sum_open": float(entry.get("sumOpen", 0) or 0),
                "sum_close": float(entry.get("sumClose", 0) or 0),
                "net_funding": float(entry.get("netFunding", 0) or 0),
                "status": entry.get("status", "OPEN"),
                "exchange": "dydx",
                "raw": entry,
            })
        return positions

    async def fetch_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all open orders, optionally filtered by *symbol*.

        Returns
        -------
        list of open order dicts.
        """
        if not self._wallet_address:
            return []

        params: Dict[str, Any] = {
            "address": self._wallet_address,
            "subaccountNumber": 0,
            "status": "OPEN",
        }
        if symbol:
            params["ticker"] = to_dydx_symbol(symbol)

        raw = await self._request(
            "GET",
            "/orders",
            params=params,
        )

        orders = []
        for item in raw if isinstance(raw, list) else raw.get("orders", []):
            orders.append({
                "order_id": str(item.get("id", "")),
                "client_id": str(item.get("clientId", "")),
                "symbol": from_dydx_symbol(item.get("ticker", "")),
                "ticker": item.get("ticker", ""),
                "side": item.get("side", ""),
                "type": item.get("type", ""),
                "size": float(item.get("size", 0) or 0),
                "remaining_size": float(item.get("remainingSize", 0) or 0),
                "price": float(item.get("price", 0) or 0),
                "status": item.get("status", ""),
                "time_in_force": item.get("timeInForce", ""),
                "post_only": bool(item.get("postOnly", False)),
                "created_at": item.get("createdAt", ""),
                "exchange": "dydx",
            })
        return orders

    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch the current funding rate for a perpetual contract.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USD", "BTC-USD".

        Returns
        -------
        dict with keys: symbol, rate, next_time, predicted, exchange
        """
        dydx_sym = to_dydx_symbol(symbol)
        raw = await self._request(
            "GET",
            f"/perpetualMarkets",
            params={"ticker": dydx_sym},
        )

        markets = raw.get("markets", {})
        market = markets.get(dydx_sym, {})

        return {
            "symbol": from_dydx_symbol(dydx_sym),
            "rate": float(market.get("nextFundingRate", 0) or 0),
            "next_time": market.get("nextFundingAt"),
            "predicted": float(market.get("nextFundingRate", 0) or 0),
            "exchange": "dydx",
            "raw": market,
        }

    async def fetch_all_funding_rates(self) -> List[Dict[str, Any]]:
        """
        Fetch funding rates for all perpetual markets.

        Returns
        -------
        list of dicts (one per market), each with symbol, rate, next_time.
        """
        raw = await self._request("GET", "/perpetualMarkets")

        results = []
        for sym, market in raw.get("markets", {}).items():
            results.append({
                "symbol": from_dydx_symbol(sym),
                "ticker": sym,
                "rate": float(market.get("nextFundingRate", 0) or 0),
                "next_time": market.get("nextFundingAt"),
                "oracle_price": float(market.get("oraclePrice", 0) or 0),
                "exchange": "dydx",
            })
        return results

    async def fetch_historical_funding(
        self, symbol: str, days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical funding rate payments for *symbol*.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USD", "BTC-USD".
        days:
            Number of days of history (default 7).

        Returns
        -------
        list of historical funding rate dicts, newest first.
        """
        dydx_sym = to_dydx_symbol(symbol)
        raw = await self._request(
            "GET",
            f"/historicalFunding/{dydx_sym}",
            params={"limit": days * 3},  # ~3 funding events per day (8h intervals)
        )

        results = []
        for entry in raw.get("historicalFunding", []):
            results.append({
                "symbol": from_dydx_symbol(dydx_sym),
                "rate": float(entry.get("rate", 0) or 0),
                "price": float(entry.get("price", 0) or 0),
                "effective_at": entry.get("effectiveAt", ""),
                "exchange": "dydx",
            })
        return results

    # ------------------------------------------------------------------
    # Order methods (require on-chain signing)
    # ------------------------------------------------------------------

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        post_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Place a perpetual order on the dYdX v4 chain.

        Orders are signed with the private key and broadcast as
        CosmosSDK transactions. Requires dydx-v4-client library.

        Parameters
        ----------
        symbol:
            Any format: "BTC/USD", "BTC-USD".
        side:
            "BUY" or "SELL".
        order_type:
            "LIMIT" or "MARKET".
        size:
            Position size in base asset (e.g. BTC units).
        price:
            Limit price (required for LIMIT orders).
        post_only:
            If True (default), order is flagged as post-only (maker-only).
            POST_ONLY rejects orders that would immediately fill as taker,
            guaranteeing the 0.01% maker fee.

        Returns
        -------
        dict with order details and transaction hash.

        Raises
        ------
        NotImplementedError
            If no private key was provided at initialisation.
        DYDXOrderSigningError
            If the dydx-v4-client library is not installed or signing fails.
        """
        self._require_signing_key("place_order")
        dydx_sym = to_dydx_symbol(symbol)

        try:
            return await self._place_order_via_sdk(
                symbol=dydx_sym,
                side=side.upper(),
                order_type=order_type.upper(),
                size=size,
                price=price,
                post_only=post_only,
            )
        except NotImplementedError:
            raise
        except ImportError as exc:
            raise DYDXOrderSigningError(
                "dydx-v4-client library not installed. "
                "Install with: pip install dydx-v4-client\n"
                f"Original error: {exc}"
            ) from exc

    async def _place_order_via_sdk(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float],
        post_only: bool,
    ) -> Dict[str, Any]:
        """
        Internal: place order using dydx-v4-client SDK.

        Supports both mnemonic and private_key_hex initialisation.
        Falls back to a detailed error if the SDK is not available.
        """
        try:
            # Attempt to import dydx v4 SDK
            from dydx_v4_client import NodeClient, Wallet  # type: ignore
            from dydx_v4_client.indexer.rest.indexer_client import IndexerClient  # type: ignore
            from dydx_v4_client.node.market import Market  # type: ignore
            from dydx_v4_client.node.order_flags import ORDER_FLAGS_LONG_TERM  # type: ignore
        except ImportError:
            raise ImportError(
                "dydx-v4-client not installed. "
                "Run: pip install dydx-v4-client"
            )

        # Derive wallet from mnemonic or private key
        if self._mnemonic:
            wallet = await Wallet.from_mnemonic(self._mnemonic)
        elif self._private_key_hex:
            wallet = await Wallet.from_hex_key(self._private_key_hex)
        else:
            raise DYDXOrderSigningError("No signing key available")

        # Connect to node and indexer
        node_client = await NodeClient.connect(self._node_url)
        indexer = IndexerClient(host=self._indexer_url)

        # Fetch market params for tick/step size
        markets_data = await self._request("GET", "/perpetualMarkets",
                                           params={"ticker": symbol})
        market_info = markets_data.get("markets", {}).get(symbol, {})
        atomic_resolution = int(market_info.get("atomicResolution", -10))
        step_base_quantums = int(market_info.get("stepBaseQuantums", 1_000_000))
        quantum_conversion_exponent = int(market_info.get("quantumConversionExponent", -9))

        # Build market helper
        market = Market(
            market=market_info,
        )

        # Determine order side flag
        from dydx_v4_client.node.order_flags import ORDER_FLAGS_SHORT_TERM  # type: ignore
        from v4_proto.dydxprotocol.clob.order_pb2 import Order  # type: ignore

        side_flag = Order.SIDE_BUY if side == "BUY" else Order.SIDE_SELL

        # Build order
        current_block = await node_client.latest_block_height()
        good_til_block = current_block + 20  # short-term: good for 20 blocks

        if order_type == "MARKET":
            # Market orders: use oracle price ± slippage
            oracle = float(market_info.get("oraclePrice", price or 0))
            limit_price = oracle * (1.02 if side == "BUY" else 0.98)
            time_in_force = Order.TIME_IN_FORCE_IOC
            post_only = False
        else:
            limit_price = price
            time_in_force = (
                Order.TIME_IN_FORCE_POST_ONLY
                if post_only
                else Order.TIME_IN_FORCE_UNSPECIFIED
            )

        if limit_price is None:
            raise DYDXOrderSigningError("price required for LIMIT orders")

        order_id, order = market.order(
            wallet=wallet,
            side=side_flag,
            size=size,
            price=limit_price,
            time_in_force=time_in_force,
            reduce_only=False,
            order_flags=ORDER_FLAGS_SHORT_TERM,
            good_til_block=good_til_block,
        )

        # Sign and broadcast
        tx = await node_client.place_order(wallet, order)
        tx_hash = tx.get("txhash", "")

        log.info(
            "DYDXClient: order placed symbol=%s side=%s size=%s price=%s tx=%s",
            symbol, side, size, limit_price, tx_hash,
        )

        return {
            "order_id": str(order_id),
            "tx_hash": tx_hash,
            "symbol": from_dydx_symbol(symbol),
            "side": side,
            "type": order_type,
            "size": size,
            "price": limit_price,
            "post_only": post_only,
            "status": "PENDING",
            "good_til_block": good_til_block,
            "exchange": "dydx",
        }

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        good_til_block: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Cancel an open order on the dYdX v4 chain.

        Parameters
        ----------
        symbol:
            Trading pair (for logging/context).
        order_id:
            dYdX order ID.
        good_til_block:
            Good-til-block for the cancellation transaction.
            If None, uses current block + 10.

        Returns
        -------
        dict with cancellation transaction hash.

        Raises
        ------
        NotImplementedError
            If no private key was provided at initialisation.
        """
        self._require_signing_key("cancel_order")
        dydx_sym = to_dydx_symbol(symbol)

        try:
            from dydx_v4_client import NodeClient, Wallet  # type: ignore
        except ImportError as exc:
            raise DYDXOrderSigningError(
                "dydx-v4-client not installed. Run: pip install dydx-v4-client"
            ) from exc

        if self._mnemonic:
            wallet = await Wallet.from_mnemonic(self._mnemonic)
        elif self._private_key_hex:
            wallet = await Wallet.from_hex_key(self._private_key_hex)
        else:
            raise DYDXOrderSigningError("No signing key available")

        node_client = await NodeClient.connect(self._node_url)

        if good_til_block is None:
            good_til_block = (await node_client.latest_block_height()) + 10

        tx = await node_client.cancel_order(
            wallet=wallet,
            order_id=order_id,
            good_til_block=good_til_block,
        )
        tx_hash = tx.get("txhash", "")

        log.info("DYDXClient: order cancelled order_id=%s tx=%s", order_id, tx_hash)

        return {
            "order_id": order_id,
            "symbol": from_dydx_symbol(dydx_sym),
            "status": "CANCELLED",
            "tx_hash": tx_hash,
            "exchange": "dydx",
        }

    # ------------------------------------------------------------------
    # WebSocket subscriptions
    # ------------------------------------------------------------------

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe to the dYdX v4 orderbook WebSocket channel.

        dYdX WS subscription format:
        {"type": "subscribe", "channel": "v4_orderbook", "id": "BTC-USD"}

        *callback* is called with (symbol, bids, asks, timestamp_ns).

        Parameters
        ----------
        symbol:
            Any format: "BTC/USD", "BTC-USD".
        callback:
            Coroutine or callable accepting (symbol, bids, asks, timestamp_ns).
        """
        dydx_sym = to_dydx_symbol(symbol)
        task_key = f"ob_{dydx_sym}"

        task = asyncio.create_task(
            self._ws_stream(
                channel="v4_orderbook",
                channel_id=dydx_sym,
                symbol=dydx_sym,
                handler=self._handle_order_book,
                callback=callback,
            ),
            name=f"dydx_ob_{dydx_sym}",
        )
        self._ws_tasks[task_key] = task

    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Subscribe to the dYdX v4 trades WebSocket channel.

        dYdX WS subscription format:
        {"type": "subscribe", "channel": "v4_trades", "id": "BTC-USD"}

        *callback* is called with (symbol, side, size, price, timestamp_ns).

        Parameters
        ----------
        symbol:
            Any format: "BTC/USD", "BTC-USD".
        callback:
            Coroutine or callable accepting (symbol, side, size, price, timestamp_ns).
        """
        dydx_sym = to_dydx_symbol(symbol)
        task_key = f"trades_{dydx_sym}"

        task = asyncio.create_task(
            self._ws_stream(
                channel="v4_trades",
                channel_id=dydx_sym,
                symbol=dydx_sym,
                handler=self._handle_trades,
                callback=callback,
            ),
            name=f"dydx_trades_{dydx_sym}",
        )
        self._ws_tasks[task_key] = task

    # ------------------------------------------------------------------
    # WebSocket internals
    # ------------------------------------------------------------------

    async def _ws_stream(
        self,
        channel: str,
        channel_id: str,
        symbol: str,
        handler: Callable,
        callback: Callable,
    ) -> None:
        """Internal WS connection loop with auto-reconnect."""
        backoff = 1.0
        while True:
            try:
                session = await self._get_session()
                async with session.ws_connect(
                    self._ws_url,
                    heartbeat=30.0,
                    receive_timeout=60.0,
                ) as ws:
                    sub_msg = json.dumps({
                        "type": "subscribe",
                        "channel": channel,
                        "id": channel_id,
                    })
                    await ws.send_str(sub_msg)
                    log.info(
                        "DYDXClient: subscribed channel=%s id=%s",
                        channel, channel_id,
                    )
                    backoff = 1.0  # reset on successful connection

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await handler(msg.data, symbol, callback)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            log.warning(
                                "DYDXClient WS closed/error channel=%s", channel
                            )
                            break

            except asyncio.CancelledError:
                log.info("DYDXClient: WS stream %s/%s cancelled", channel, channel_id)
                return
            except Exception as exc:
                log.warning(
                    "DYDXClient WS error channel=%s: %s — reconnect in %.1fs",
                    channel, exc, backoff,
                )

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

    @staticmethod
    async def _handle_order_book(
        raw: str,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Parse dYdX v4 orderbook message and invoke callback.

        dYdX v4 orderbook format:
        {
            "type": "channel_data" | "subscribed",
            "channel": "v4_orderbook",
            "id": "BTC-USD",
            "contents": {
                "bids": [{"price": "...", "size": "..."}, ...],
                "asks": [{"price": "...", "size": "..."}, ...]
            }
        }
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        msg_type = data.get("type", "")
        if msg_type in ("connected", "subscribed", "ping", "pong"):
            return

        contents = data.get("contents", {})
        if not contents:
            return

        ts_ns = int(time.time() * 1_000_000_000)

        raw_bids = contents.get("bids", [])
        raw_asks = contents.get("asks", [])

        bids = [
            [float(entry.get("price", 0)), float(entry.get("size", 0))]
            for entry in raw_bids
            if isinstance(entry, dict)
        ]
        asks = [
            [float(entry.get("price", 0)), float(entry.get("size", 0))]
            for entry in raw_asks
            if isinstance(entry, dict)
        ]

        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(symbol, bids, asks, ts_ns)
            else:
                callback(symbol, bids, asks, ts_ns)
        except Exception as exc:
            log.debug("DYDXClient order_book callback error: %s", exc)

    @staticmethod
    async def _handle_trades(
        raw: str,
        symbol: str,
        callback: Callable,
    ) -> None:
        """
        Parse dYdX v4 trade message and invoke callback.

        dYdX v4 trade format:
        {
            "type": "channel_data",
            "channel": "v4_trades",
            "id": "BTC-USD",
            "contents": {
                "trades": [
                    {
                        "id": "...",
                        "side": "BUY" | "SELL",
                        "size": "0.001",
                        "price": "29000.00",
                        "createdAt": "2024-01-01T00:00:00.000Z",
                        "createdAtHeight": 123456
                    },
                    ...
                ]
            }
        }
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict):
            return

        msg_type = data.get("type", "")
        if msg_type in ("connected", "subscribed", "ping", "pong"):
            return

        contents = data.get("contents", {})
        trades = contents.get("trades", [])
        if not trades:
            return

        ts_ns = int(time.time() * 1_000_000_000)

        for trade in trades:
            side = str(trade.get("side", "BUY")).upper()
            price = float(trade.get("price", 0))
            size = float(trade.get("size", 0))
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, side, size, price, ts_ns)
                else:
                    callback(symbol, side, size, price, ts_ns)
            except Exception as exc:
                log.debug("DYDXClient trades callback error: %s", exc)
