"""
CCXT async exchange provider for ARGUS.

Creates and configures ccxt.pro async exchange instances with API keys
from environment variables. Uses ccxt.pro for WebSocket streaming when available,
falls back to ccxt.async_support for REST-only.

M21: Network-error resilience via tenacity retry decorators on all fetch methods.
     fetch_ohlcv, fetch_ticker, and fetch_order_book each retry up to 3 times
     with exponential back-off (1 s min, 10 s max) and re-raise on final failure.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tenacity retry policy — shared across all fetch helpers
# ---------------------------------------------------------------------------

# Retry on any exception that looks like a transient network error.
# We catch broad Exception here; the `reraise=True` flag ensures the final
# attempt's exception propagates to callers as normal.
_RETRY_POLICY = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


# ---------------------------------------------------------------------------
# Exchange registry
# ---------------------------------------------------------------------------

_EXCHANGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "kraken": {
        "class": "kraken",
        "key_env": "KRAKEN_API_KEY",
        "secret_env": "KRAKEN_API_SECRET",
        "options": {
            "defaultType": "spot",
        },
    },
    "coinbase": {
        "class": "coinbase",
        "key_env": "COINBASE_API_KEY",
        "secret_env": "COINBASE_SECRET_KEY",
        "options": {},
    },
    "bybit": {
        "class": "bybit",
        "key_env": "BYBIT_API_KEY",
        "secret_env": "BYBIT_SECRET_KEY",
        "options": {
            "defaultType": "spot",
        },
    },
    "okx": {
        "class": "okx",
        "key_env": "OKX_API_KEY",
        "secret_env": "OKX_SECRET_KEY",
        "options": {},
    },
}


# ---------------------------------------------------------------------------
# Retry-wrapped fetch helpers
# ---------------------------------------------------------------------------

@retry(**_RETRY_POLICY)
async def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str = "1m",
    since: Optional[int] = None,
    limit: int = 100,
    params: Optional[Dict[str, Any]] = None,
) -> List[List[Any]]:
    """
    Fetch OHLCV candles from a ccxt exchange with automatic retry on network errors.

    Args:
        exchange:  ccxt async exchange instance
        symbol:    Trading pair, e.g. "BTC/USDT"
        timeframe: Candle interval, e.g. "1m", "5m", "1h"
        since:     Unix timestamp ms to start from (optional)
        limit:     Maximum number of candles to return
        params:    Extra ccxt params forwarded verbatim

    Returns:
        List of [timestamp, open, high, low, close, volume] lists
    """
    result = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit, params=params or {})
    logger.debug("fetch_ohlcv %s %s: %d candles", symbol, timeframe, len(result) if result else 0)
    return result  # type: ignore[return-value]


@retry(**_RETRY_POLICY)
async def fetch_ticker(
    exchange: Any,
    symbol: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fetch current ticker for *symbol* with automatic retry on network errors.

    Args:
        exchange: ccxt async exchange instance
        symbol:   Trading pair, e.g. "BTC/USDT"
        params:   Extra ccxt params forwarded verbatim

    Returns:
        ccxt ticker dict (last, bid, ask, volume, timestamp, …)
    """
    result = await exchange.fetch_ticker(symbol, params=params or {})
    logger.debug("fetch_ticker %s: last=%.4f", symbol, result.get("last", 0))
    return result  # type: ignore[return-value]


@retry(**_RETRY_POLICY)
async def fetch_order_book(
    exchange: Any,
    symbol: str,
    limit: int = 20,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fetch order book for *symbol* with automatic retry on network errors.

    Args:
        exchange: ccxt async exchange instance
        symbol:   Trading pair, e.g. "BTC/USDT"
        limit:    Depth levels to retrieve
        params:   Extra ccxt params forwarded verbatim

    Returns:
        ccxt order book dict (bids, asks, timestamp, …)
    """
    result = await exchange.fetch_order_book(symbol, limit=limit, params=params or {})
    bids = len(result.get("bids", []))
    asks = len(result.get("asks", []))
    logger.debug("fetch_order_book %s: %d bids / %d asks", symbol, bids, asks)
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Exchange factory
# ---------------------------------------------------------------------------

def get_ccxt_async_exchange(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = False,
) -> Any:
    """
    Create and return a ccxt async exchange instance.

    Args:
        exchange_id: Exchange name (kraken, coinbase, bybit, okx)
        api_key: API key (optional, falls back to env var)
        api_secret: API secret (optional, falls back to env var)
        sandbox: Use sandbox/testnet if available

    Returns:
        ccxt.pro async exchange instance (or ccxt async if pro unavailable)
    """
    # Prefer ccxt.pro for WebSocket support, fall back to async_support
    try:
        import ccxt.pro as ccxt_module
        _using_pro = True
        logger.info("Using ccxt.pro for WebSocket support")
    except ImportError:
        try:
            import ccxt.async_support as ccxt_module
            _using_pro = False
            logger.info("Using ccxt.async_support (REST only)")
        except ImportError as e:
            logger.error(f"Failed to import ccxt.async_support: {e}")
            raise

    exchange_id = exchange_id.lower().strip()
    cfg = _EXCHANGE_CONFIG.get(exchange_id, {})
    ccxt_class_name = cfg.get("class", exchange_id)

    # Resolve API credentials
    key = api_key or os.getenv(cfg.get("key_env", ""), "")
    secret = api_secret or os.getenv(cfg.get("secret_env", ""), "")

    # Get the exchange class from ccxt
    if not hasattr(ccxt_module, ccxt_class_name):
        raise ValueError(f"Unknown CCXT exchange: {ccxt_class_name}")

    exchange_class = getattr(ccxt_module, ccxt_class_name)

    # Build config
    config: Dict[str, Any] = {
        "enableRateLimit": True,
        "options": cfg.get("options", {}),
        "timeout": 30000,  # 30 second timeout
    }

    if key:
        config["apiKey"] = key
    if secret:
        config["secret"] = secret

    if sandbox:
        config["sandbox"] = True

    exchange = exchange_class(config)

    has_keys = bool(key and secret)
    _ws_available = hasattr(exchange, "watch_ticker")
    logger.info(
        "CCXT exchange created: %s (keys=%s, sandbox=%s, ws=%s, pro=%s)",
        exchange_id, "present" if has_keys else "public-only", sandbox,
        "yes" if _ws_available else "no", "yes" if _using_pro else "no",
    )

    return exchange


def get_ccxt_exchange(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = False,
) -> Any:
    """
    Create and return a ccxt sync exchange instance.
    
    This is a synchronous wrapper for get_ccxt_async_exchange.
    
    Args:
        exchange_id: Exchange name (kraken, coinbase, bybit, okx)
        api_key: API key (optional, falls back to env var)
        api_secret: API secret (optional, falls back to env var)
        sandbox: Use sandbox/testnet if available
    
    Returns:
        ccxt exchange instance (sync version)
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        raise
    
    key = api_key or os.environ.get(f"{exchange_id.upper()}_API_KEY", "")
    secret = api_secret or os.environ.get(f"{exchange_id.upper()}_API_SECRET", "")
    
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")
    
    config: Dict[str, Any] = {"enableRateLimit": True}
    if key:
        config["apiKey"] = key
    if secret:
        config["secret"] = secret
    if sandbox:
        config["sandbox"] = True
    
    exchange = exchange_class(config)
    
    has_keys = bool(key and secret)
    logger.info(
        "CCXT sync exchange created: %s (keys=%s, sandbox=%s)",
        exchange_id, "present" if has_keys else "public-only", sandbox,
    )
    
    return exchange
