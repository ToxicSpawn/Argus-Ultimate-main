"""exchanges/__init__.py — hardened exchange factory with GuardedExchange.

Every exchange adapter returned by create_exchange() is automatically
wrapped in GuardedExchange, injecting per-exchange token-bucket rate
limiting and exponential-backoff retry.

Usage:
    from exchanges import create_exchange, ExchangeRegistry

    # Simple factory
    ex = create_exchange("kraken", api_key="...", api_secret="...", dry_run=True)
    ticker = ex.fetch_ticker("BTC/USD")  # rate-limited automatically

    # Registry (idempotent, thread-safe)
    registry = ExchangeRegistry.instance()
    ex = registry.get_or_create("kraken", api_key="...", api_secret="...")
    available = registry.list_available()
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wrap_with_guard(exchange: Any, name: str) -> Any:
    """Wrap an exchange object in GuardedExchange. No-op if already guarded."""
    try:
        from exchanges.guard import GuardedExchange
        from utils.rate_limit_guard import ExchangeRateLimiter
        if isinstance(exchange, GuardedExchange):
            return exchange
        limiter = ExchangeRateLimiter()
        guarded = GuardedExchange(exchange, exchange_name=name, limiter=limiter)
        logger.debug(
            "exchanges: wrapped '%s' in GuardedExchange "
            "(capacity=%.0f/s from DEFAULTS)",
            name,
            ExchangeRateLimiter.DEFAULTS.get(name.lower(), {}).get("capacity", 10),
        )
        return guarded
    except ImportError as e:
        logger.warning("Could not wrap exchange in GuardedExchange: %s", e)
        return exchange


def _build_ccxt_exchange(name: str, api_key: str, api_secret: str,
                         dry_run: bool, sandbox: bool) -> Any:
    """Build a raw ccxt exchange instance."""
    try:
        import ccxt
        ExchangeClass = getattr(ccxt, name.lower(), None)
        if ExchangeClass is None:
            logger.error("ccxt has no exchange named '%s'", name)
            return None
        cfg: Dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": False,  # We handle rate limiting ourselves
        }
        if sandbox:
            cfg["sandbox"] = True
        ex = ExchangeClass(cfg)
        return ex
    except ImportError:
        logger.warning("ccxt not installed — exchange '%s' unavailable", name)
        return None


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_exchange(
    name: str,
    api_key: str = "",
    api_secret: str = "",
    dry_run: bool = True,
    sandbox: bool = False,
) -> Optional[Any]:
    """
    Create and return a GuardedExchange-wrapped ccxt exchange.

    Args:
        name:       Exchange name (kraken, coinbase, binance, bybit, okx)
        api_key:    API key (or empty string for public data only)
        api_secret: API secret
        dry_run:    If True, log orders but don't execute
        sandbox:    If True, use exchange testnet

    Returns:
        GuardedExchange wrapping a ccxt exchange, or None on failure.
    """
    raw = _build_ccxt_exchange(name, api_key, api_secret, dry_run, sandbox)
    if raw is None:
        return None
    guarded = _wrap_with_guard(raw, name)
    if dry_run:
        logger.info("Exchange '%s' created in DRY-RUN mode (orders simulated)", name)
    else:
        logger.info("Exchange '%s' created in LIVE mode", name)
    return guarded


async def health_check(exchange: Any, symbol: str = "BTC/USDT") -> bool:
    """
    Check if an exchange is reachable by fetching a ticker.
    Returns True if healthy, False otherwise.
    """
    try:
        import asyncio
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_ticker", None)):
            ticker = await exchange.fetch_ticker(symbol)
        else:
            ticker = exchange.fetch_ticker(symbol)
        return ticker is not None
    except Exception as exc:
        logger.warning("Exchange health check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ExchangeRegistry:
    """Thread-safe registry of exchange instances.

    Use ExchangeRegistry.instance() for the process-wide singleton.
    """

    _singleton: Optional["ExchangeRegistry"] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._exchanges: Dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ExchangeRegistry":
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def get_or_create(
        self,
        name: str,
        api_key: str = "",
        api_secret: str = "",
        dry_run: bool = True,
        sandbox: bool = False,
    ) -> Optional[Any]:
        """Return existing exchange or create a new one. Thread-safe."""
        key = name.lower()
        with self._lock:
            if key not in self._exchanges:
                ex = create_exchange(name, api_key, api_secret, dry_run, sandbox)
                if ex is not None:
                    self._exchanges[key] = ex
            return self._exchanges.get(key)

    def register(self, name: str, exchange: Any) -> None:
        """Register a pre-built exchange instance."""
        with self._lock:
            self._exchanges[name.lower()] = exchange

    def get(self, name: str) -> Optional[Any]:
        with self._lock:
            return self._exchanges.get(name.lower())

    def remove(self, name: str) -> None:
        with self._lock:
            self._exchanges.pop(name.lower(), None)

    def list_available(self) -> List[str]:
        with self._lock:
            return list(self._exchanges.keys())

    def clear(self) -> None:
        with self._lock:
            self._exchanges.clear()
