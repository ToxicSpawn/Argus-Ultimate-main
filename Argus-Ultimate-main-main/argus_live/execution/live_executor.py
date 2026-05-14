"""
Argus Live — LiveExecutor  (Push 28a)
======================================
Bridge layer that replaces all legacy Kraken-specific connector calls
inside argus_live with CCXTAdapter.

Drop-in usage
-------------
Anywhere argus_live previously called a Kraken connector directly::

    # OLD
    from exchanges.kraken_connector import KrakenConnector
    conn = KrakenConnector(api_key=..., api_secret=...)
    conn.place_order(...)

    # NEW
    from argus_live.execution.live_executor import LiveExecutor
    executor = LiveExecutor.from_env()          # reads ARGUS_EXCHANGE, keys from env
    await executor.place_limit_order(...)       # identical interface, any exchange

Exchange selection
------------------
Set the environment variable::

    ARGUS_EXCHANGE=kraken       # default
    ARGUS_EXCHANGE=binance
    ARGUS_EXCHANGE=okx
    ARGUS_EXCHANGE=bybit

All credentials are read automatically from env vars matching the pattern
``{EXCHANGE}_API_KEY`` / ``{EXCHANGE}_API_SECRET`` (uppercase exchange name).
See ``.env.example`` for full list.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LiveExecutor:
    """
    Unified live execution bridge for argus_live.

    Wraps CCXTAdapter with the exact method signatures that argus_live
    components expect, so swapping exchange requires only one env var change.

    All methods are synchronous wrappers — CCXTAdapter handles async
    internally via ccxt's synchronous REST client.
    """

    def __init__(self, adapter: Any, exchange_name: str, dry_run: bool = False):
        self._adapter      = adapter
        self._exchange     = exchange_name
        self._dry_run      = dry_run
        logger.info(
            "LiveExecutor ready | exchange=%s dry_run=%s",
            exchange_name, dry_run,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        exchange: Optional[str] = None,
        dry_run: Optional[bool] = None,
        maker_only: bool = True,
        sandbox: bool = False,
    ) -> "LiveExecutor":
        """
        Build a LiveExecutor from environment variables.

        Args:
            exchange:   Exchange id (e.g. 'kraken').  Reads ``ARGUS_EXCHANGE``
                        env var if not provided.  Defaults to 'kraken'.
            dry_run:    If True, log orders without sending.  Reads
                        ``ARGUS_DRY_RUN`` env var if not provided.
            maker_only: Enforce postOnly on all limit orders.
            sandbox:    Use exchange sandbox/testnet endpoint.

        Returns:
            Configured LiveExecutor.
        """
        try:
            from execution.ccxt_adapter import build_adapter_from_env
        except ImportError as exc:
            raise RuntimeError(
                "CCXTAdapter not found — ensure execution/ccxt_adapter.py exists (Push 26)"
            ) from exc

        xch = exchange or os.environ.get("ARGUS_EXCHANGE", "kraken")
        dr  = dry_run  if dry_run is not None else (
            os.environ.get("ARGUS_DRY_RUN", "false").lower() in ("1", "true", "yes")
        )

        adapter = build_adapter_from_env(
            exchange_id=xch,
            dry_run=dr,
            maker_only=maker_only,
            sandbox=sandbox,
        )
        return cls(adapter=adapter, exchange_name=xch, dry_run=dr)

    @classmethod
    def from_adapter(cls, adapter: Any, dry_run: bool = False) -> "LiveExecutor":
        """Build from a pre-constructed CCXTAdapter (e.g. in tests)."""
        name = getattr(adapter, "exchange_id", "unknown")
        return cls(adapter=adapter, exchange_name=name, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def exchange_name(self) -> str:
        return self._exchange

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def adapter(self) -> Any:
        """Direct access to the underlying CCXTAdapter."""
        return self._adapter

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Place a post-only limit order.

        Args:
            symbol: Market symbol e.g. 'XBT/USD'.
            side:   'buy' or 'sell'.
            amount: Order size in base currency.
            price:  Limit price.
            params: Extra ccxt params (merged with adapter defaults).

        Returns:
            ccxt order dict.
        """
        logger.debug(
            "[%s] limit %s %s @ %.2f  qty=%.6f  dry=%s",
            self._exchange, side, symbol, price, amount, self._dry_run,
        )
        return self._adapter.place_limit_order(
            symbol=symbol, side=side, amount=amount, price=price, params=params or {}
        )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Place a market order (taker — use sparingly)."""
        logger.debug(
            "[%s] market %s %s  qty=%.6f  dry=%s",
            self._exchange, side, symbol, amount, self._dry_run,
        )
        return self._adapter.place_market_order(
            symbol=symbol, side=side, amount=amount, params=params or {}
        )

    def place_foc_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        max_retries: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Fill-or-Cancel limit order with automatic mid-price refresh + retry.

        Identical to CCXTAdapter.place_foc_order — exposed here so
        argus_live components don't need to import the adapter directly.
        """
        return self._adapter.place_foc_order(
            symbol=symbol, side=side, amount=amount, max_retries=max_retries
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel a single order by ID."""
        return self._adapter.cancel_order(order_id=order_id, symbol=symbol)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Cancel all open orders, optionally filtered to one symbol."""
        return self._adapter.cancel_all_orders(symbol=symbol)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all open orders for a symbol (or all symbols)."""
        try:
            return self._adapter.exchange.fetch_open_orders(symbol) if symbol else \
                   self._adapter.exchange.fetch_open_orders()
        except Exception as exc:
            logger.warning("fetch_open_orders failed: %s", exc)
            return []

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Fetch a single order by ID."""
        try:
            return self._adapter.exchange.fetch_order(order_id, symbol)
        except Exception as exc:
            logger.warning("fetch_order(%s) failed: %s", order_id, exc)
            return {}

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_mid_price(self, symbol: str) -> float:
        """Return current mid-price (best_bid + best_ask) / 2."""
        return self._adapter.get_mid_price(symbol)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV candles.

        Returns:
            List of dicts with keys: timestamp, open, high, low, close, volume.
        """
        return self._adapter.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return full ticker dict for a symbol."""
        try:
            return self._adapter.exchange.fetch_ticker(symbol)
        except Exception as exc:
            logger.warning("fetch_ticker(%s) failed: %s", symbol, exc)
            return {}

    def fetch_order_book(
        self, symbol: str, depth: int = 20
    ) -> Dict[str, Any]:
        """Return order book with bids/asks up to `depth` levels."""
        try:
            return self._adapter.exchange.fetch_order_book(symbol, depth)
        except Exception as exc:
            logger.warning("fetch_order_book(%s) failed: %s", symbol, exc)
            return {"bids": [], "asks": []}

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def fetch_balance(self) -> Dict[str, Any]:
        """Return full balance dict from exchange."""
        return self._adapter.fetch_balance()

    def fetch_free_balance(self, currency: str = "USD") -> float:
        """Return free (available) balance for a currency."""
        try:
            bal = self.fetch_balance()
            return float(bal.get("free", {}).get(currency, 0.0))
        except Exception as exc:
            logger.warning("fetch_free_balance(%s) failed: %s", currency, exc)
            return 0.0

    def fetch_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return open positions (futures/margin exchanges)."""
        try:
            ex = self._adapter.exchange
            if hasattr(ex, "fetch_positions"):
                return ex.fetch_positions([symbol] if symbol else [])
            return []
        except Exception as exc:
            logger.warning("fetch_positions failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Exchange metadata
    # ------------------------------------------------------------------

    def get_min_order_amount(self, symbol: str) -> float:
        """Return minimum order size for a symbol."""
        return self._adapter.get_min_order_amount(symbol)

    def get_price_precision(self, symbol: str) -> int:
        """Return price decimal precision for a symbol."""
        return self._adapter.get_price_precision(symbol)

    def get_markets(self) -> Dict[str, Any]:
        """Return all available markets on the exchange."""
        try:
            return self._adapter.exchange.load_markets()
        except Exception as exc:
            logger.warning("get_markets failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Ping the exchange to verify connectivity."""
        try:
            self._adapter.exchange.fetch_time()
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"LiveExecutor(exchange={self._exchange!r}, "
            f"dry_run={self._dry_run})"
        )
