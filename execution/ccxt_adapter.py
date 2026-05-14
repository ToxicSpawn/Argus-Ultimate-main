"""
ccxt_adapter.py — Multi-exchange execution adapter via ccxt.

Drop-in replacement for the Kraken-only MakerOrderManager.
Supports 50+ exchanges (Binance, Kraken, OKX, Bybit, Coinbase, etc.)
through a unified async interface.

Features
--------
- Unified place/cancel/fetch across all ccxt-supported exchanges
- Maker-only limit orders with optional post-only flag
- Automatic FOC (Fill-or-Cancel) retry with configurable attempts
- Order status polling with timeout
- Position and balance queries
- Rate-limit aware (respects ccxt rateLimit)
- Dry-run mode: logs all orders without submitting
- Exchange capability detection (futures, margin, spot)
- Factory helper: build_adapter_from_env() reads credentials from env vars

Usage
-----
    adapter = CCXTAdapter(
        exchange_id="binance",
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        sandbox=True,
        dry_run=False,
    )
    await adapter.connect()
    order = await adapter.place_foc_order(
        symbol="BTC/USDT", side="buy", amount=0.001, price=60000.0
    )
    await adapter.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import ccxt.async_support as ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _CCXT_AVAILABLE = False
    logger.warning("ccxt not installed. Install with: pip install ccxt")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str
    filled: float = 0.0
    remaining: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)
    raw: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def is_filled(self) -> bool:
        return self.status == "closed" and self.filled >= self.amount * 0.999

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def is_cancelled(self) -> bool:
        return self.status in ("canceled", "cancelled")


@dataclass
class Balance:
    total: Dict[str, float] = field(default_factory=dict)
    free: Dict[str, float] = field(default_factory=dict)
    used: Dict[str, float] = field(default_factory=dict)

    def get_free(self, currency: str) -> float:
        return self.free.get(currency, 0.0)

    def get_total(self, currency: str) -> float:
        return self.total.get(currency, 0.0)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class CCXTAdapter:
    """
    Unified async ccxt execution adapter.

    Parameters
    ----------
    exchange_id   : ccxt exchange id e.g. "binance", "kraken", "okx", "bybit"
    api_key       : exchange API key (or set {EXCHANGE}_API_KEY env var)
    api_secret    : exchange API secret
    passphrase    : API passphrase (required for OKX, Coinbase Pro)
    sandbox       : use exchange testnet/sandbox (default False)
    dry_run       : log orders only, never submit (default True)
    maker_only    : add postOnly flag to limit orders (default True)
    foc_retries   : fill-or-cancel retry attempts (default 3)
    foc_timeout   : seconds to wait for fill before cancel (default 10.0)
    poll_interval : seconds between order status polls (default 1.0)
    """

    def __init__(
        self,
        exchange_id: str = "kraken",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        sandbox: bool = False,
        dry_run: bool = True,
        maker_only: bool = True,
        foc_retries: int = 3,
        foc_timeout: float = 10.0,
        poll_interval: float = 1.0,
    ) -> None:
        if not _CCXT_AVAILABLE:
            raise RuntimeError("ccxt not installed. Run: pip install ccxt")

        self._exchange_id   = exchange_id.lower()
        self._api_key       = api_key or os.getenv(f"{exchange_id.upper()}_API_KEY", "")
        self._api_secret    = api_secret or os.getenv(f"{exchange_id.upper()}_API_SECRET", "")
        self._passphrase    = passphrase or os.getenv(f"{exchange_id.upper()}_PASSPHRASE", "")
        self._sandbox       = sandbox
        self._dry_run       = dry_run
        self._maker_only    = maker_only
        self._foc_retries   = foc_retries
        self._foc_timeout   = foc_timeout
        self._poll_interval = poll_interval
        self._exchange: Optional[Any] = None
        self._connected     = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise ccxt exchange instance and load markets."""
        exchange_class = getattr(ccxt, self._exchange_id, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{self._exchange_id}' not found in ccxt. "
                f"Try one of: {SUPPORTED_EXCHANGES}"
            )

        config: Dict[str, Any] = {
            "apiKey": self._api_key,
            "secret": self._api_secret,
            "enableRateLimit": True,
        }
        if self._passphrase:
            config["password"] = self._passphrase

        self._exchange = exchange_class(config)

        if self._sandbox:
            if self._exchange.has.get("sandbox"):
                self._exchange.set_sandbox_mode(True)
                logger.info("CCXTAdapter: sandbox mode enabled")
            else:
                logger.warning(
                    "CCXTAdapter: %s does not support sandbox", self._exchange_id
                )

        await self._exchange.load_markets()
        self._connected = True
        logger.info(
            "CCXTAdapter connected: %s | dry_run=%s | maker_only=%s | %d markets",
            self._exchange_id, self._dry_run, self._maker_only,
            len(self._exchange.markets),
        )

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._connected = False
            logger.info("CCXTAdapter closed: %s", self._exchange_id)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        """
        Place a maker limit order.

        Parameters
        ----------
        symbol : unified ccxt symbol e.g. "BTC/USDT"
        side   : "buy" or "sell"
        amount : base currency quantity
        price  : limit price in quote currency
        params : extra exchange-specific params dict
        """
        self._check_connected()
        params = params or {}
        if self._maker_only:
            params["postOnly"] = True

        if self._dry_run:
            logger.info(
                "[DRY RUN] %s LIMIT %s %s amt=%.6f price=%.4f",
                self._exchange_id.upper(), side.upper(), symbol, amount, price,
            )
            return OrderResult(
                order_id=f"dryrun_{int(time.time()*1000)}",
                symbol=symbol, side=side, amount=amount, price=price,
                status="closed", filled=amount, dry_run=True,
            )

        try:
            raw = await self._exchange.create_limit_order(
                symbol, side, amount, price, params=params
            )
            result = self._parse_order(raw)
            logger.info(
                "Order placed: %s %s %s amt=%.6f price=%.4f id=%s",
                self._exchange_id, side, symbol, amount, price, result.order_id,
            )
            return result
        except Exception as exc:
            logger.error("place_limit_order error: %s", exc)
            raise

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict[str, Any]] = None,
    ) -> OrderResult:
        """Place a market order (taker fees apply — use sparingly)."""
        self._check_connected()
        params = params or {}

        if self._dry_run:
            price = await self.get_mid_price(symbol)
            logger.info(
                "[DRY RUN] %s MARKET %s %s amt=%.6f ~price=%.4f",
                self._exchange_id.upper(), side.upper(), symbol, amount, price,
            )
            return OrderResult(
                order_id=f"dryrun_mkt_{int(time.time()*1000)}",
                symbol=symbol, side=side, amount=amount, price=price,
                status="closed", filled=amount, dry_run=True,
            )

        raw = await self._exchange.create_market_order(symbol, side, amount, params=params)
        return self._parse_order(raw)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True on success."""
        self._check_connected()
        if self._dry_run:
            logger.info("[DRY RUN] cancel order %s", order_id)
            return True
        try:
            await self._exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order: %s", order_id)
            return True
        except Exception as exc:
            logger.warning("cancel_order error: %s", exc)
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders. Returns count cancelled."""
        self._check_connected()
        if self._dry_run:
            logger.info("[DRY RUN] cancel_all_orders symbol=%s", symbol)
            return 0
        orders = await self.fetch_open_orders(symbol)
        count = 0
        for o in orders:
            if await self.cancel_order(o["id"], o["symbol"]):
                count += 1
        logger.info("Cancelled %d orders", count)
        return count

    async def place_foc_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
    ) -> Optional[OrderResult]:
        """
        Fill-or-Cancel limit order with automatic retry.

        Places a limit order, polls until filled or timeout,
        then cancels and retries at a refreshed mid-price.
        Returns None if all retries exhausted without fill.
        """
        for attempt in range(self._foc_retries):
            order = await self.place_limit_order(symbol, side, amount, price)
            if order.dry_run or order.is_filled:
                return order

            filled = await self._poll_until_filled(
                order.order_id, symbol, self._foc_timeout
            )
            if filled:
                return filled

            await self.cancel_order(order.order_id, symbol)
            price = await self.get_mid_price(symbol)
            logger.info(
                "FOC retry %d/%d: refreshed price=%.4f",
                attempt + 1, self._foc_retries, price,
            )
            await asyncio.sleep(0.5)

        logger.warning(
            "FOC order failed after %d retries: %s %s",
            self._foc_retries, side, symbol,
        )
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def fetch_order(
        self, order_id: str, symbol: str
    ) -> Optional[OrderResult]:
        self._check_connected()
        try:
            raw = await self._exchange.fetch_order(order_id, symbol)
            return self._parse_order(raw)
        except Exception as exc:
            logger.warning("fetch_order error: %s", exc)
            return None

    async def fetch_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        self._check_connected()
        try:
            return await self._exchange.fetch_open_orders(symbol)
        except Exception as exc:
            logger.warning("fetch_open_orders error: %s", exc)
            return []

    async def fetch_balance(self) -> Balance:
        self._check_connected()
        if self._dry_run:
            return Balance()
        try:
            raw = await self._exchange.fetch_balance()
            return Balance(
                total={k: float(v) for k, v in raw.get("total", {}).items() if v},
                free ={k: float(v) for k, v in raw.get("free",  {}).items() if v},
                used ={k: float(v) for k, v in raw.get("used",  {}).items() if v},
            )
        except Exception as exc:
            logger.error("fetch_balance error: %s", exc)
            return Balance()

    async def get_mid_price(self, symbol: str) -> float:
        self._check_connected()
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            bid  = float(ticker.get("bid") or 0)
            ask  = float(ticker.get("ask") or 0)
            last = float(ticker.get("last") or 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            return last
        except Exception as exc:
            logger.warning("get_mid_price error: %s", exc)
            return 0.0

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 500,
        since: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Fetch OHLCV candles via ccxt unified interface.
        Returns list of [timestamp_ms, open, high, low, close, volume].
        Converts timestamp_ms to seconds for Argus compatibility.
        """
        self._check_connected()
        try:
            raw = await self._exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, since=since
            )
            # Normalise timestamp from ms to seconds
            return [[r[0] / 1000.0, r[1], r[2], r[3], r[4], r[5]] for r in raw]
        except Exception as exc:
            logger.error("fetch_ohlcv error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Exchange info
    # ------------------------------------------------------------------

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def supports(self, capability: str) -> bool:
        if not self._exchange:
            return False
        return bool(self._exchange.has.get(capability, False))

    def available_symbols(self) -> List[str]:
        if not self._exchange:
            return []
        return list(self._exchange.markets.keys())

    def get_min_order_amount(self, symbol: str) -> float:
        if not self._exchange:
            return 0.0
        market = self._exchange.markets.get(symbol, {})
        limits = market.get("limits", {})
        return float(limits.get("amount", {}).get("min", 0.0) or 0.0)

    def get_price_precision(self, symbol: str) -> int:
        if not self._exchange:
            return 8
        market = self._exchange.markets.get(symbol, {})
        return int((market.get("precision", {}).get("price") or 8))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_connected(self) -> None:
        if not self._connected or not self._exchange:
            raise RuntimeError(
                "CCXTAdapter not connected. Call await adapter.connect() first."
            )

    async def _poll_until_filled(
        self, order_id: str, symbol: str, timeout: float
    ) -> Optional[OrderResult]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            order = await self.fetch_order(order_id, symbol)
            if order and order.is_filled:
                return order
            if order and order.is_cancelled:
                return None
            await asyncio.sleep(self._poll_interval)
        return None

    @staticmethod
    def _parse_order(raw: Dict[str, Any]) -> OrderResult:
        fee_cost = 0.0
        if raw.get("fee") and raw["fee"].get("cost"):
            fee_cost = float(raw["fee"]["cost"])
        return OrderResult(
            order_id =str(raw.get("id", "")),
            symbol   =str(raw.get("symbol", "")),
            side     =str(raw.get("side", "")),
            amount   =float(raw.get("amount") or 0),
            price    =float(raw.get("price") or raw.get("average") or 0),
            status   =str(raw.get("status", "open")),
            filled   =float(raw.get("filled") or 0),
            remaining=float(raw.get("remaining") or 0),
            cost     =float(raw.get("cost") or 0),
            fee      =fee_cost,
            timestamp=float(raw.get("timestamp") or time.time() * 1000) / 1000,
            raw      =raw,
        )

    def __repr__(self) -> str:
        return (
            f"<CCXTAdapter exchange={self._exchange_id} "
            f"connected={self._connected} dry_run={self._dry_run}>"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_adapter_from_env(exchange_id: str, **kwargs) -> CCXTAdapter:
    """
    Build CCXTAdapter reading credentials from environment variables.

    Reads:
        {EXCHANGE_ID}_API_KEY
        {EXCHANGE_ID}_API_SECRET
        {EXCHANGE_ID}_PASSPHRASE  (optional, for OKX/Coinbase)

    Examples
    --------
        adapter = build_adapter_from_env("binance", dry_run=False)
        adapter = build_adapter_from_env("okx", sandbox=True)
        adapter = build_adapter_from_env("kraken", maker_only=True)
    """
    eid = exchange_id.upper()
    return CCXTAdapter(
        exchange_id=exchange_id,
        api_key    =os.getenv(f"{eid}_API_KEY",    ""),
        api_secret =os.getenv(f"{eid}_API_SECRET", ""),
        passphrase =os.getenv(f"{eid}_PASSPHRASE", ""),
        **kwargs,
    )


SUPPORTED_EXCHANGES = [
    "binance", "kraken", "okx", "bybit", "coinbase",
    "kucoin", "huobi", "bitfinex", "gemini", "bitstamp",
    "gateio", "mexc", "bitmex", "phemex", "cryptocom",
]
