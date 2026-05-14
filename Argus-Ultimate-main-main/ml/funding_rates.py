"""
Funding rates feed (crypto perpetuals). Optional stub for regime/filter.

When use_funding_rate_filter is True and symbol has a funding rate above
funding_rate_skip_long_threshold, the strategy/execution layer can skip opening
longs (e.g. avoid paying high funding). Replace fetch_funding_rate with real
exchange API (CCXT fetch_funding_rate) when available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def fetch_funding_rate(
    symbol: str,
    exchange_id: str = "kraken",
    url_override: Optional[str] = None,
) -> Optional[float]:
    """
    Return current funding rate for symbol (e.g. 0.0001 = 0.01%).
    Stub: returns None (no filter). Replace with CCXT fetch_funding_rate or
    HTTP GET to funding_rates_url when configured.
    """
    if url_override:
        try:
            import urllib.request
            with urllib.request.urlopen(url_override, timeout=5) as resp:
                data = resp.read().decode()
                import json
                obj = json.loads(data)
                if isinstance(obj, dict) and symbol in obj:
                    return float(obj[symbol])
                if isinstance(obj, list):
                    for row in obj:
                        if isinstance(row, dict) and row.get("symbol") == symbol:
                            return float(row.get("funding_rate", row.get("rate", 0)) or 0)
        except Exception as e:
            logger.debug("Funding rate fetch %s: %s", url_override, e)
    try:
        import ccxt
        ex = getattr(ccxt, exchange_id, None)
        if ex and hasattr(ex, "fetch_funding_rate"):
            exchange = ex()
            rate = exchange.fetch_funding_rate(symbol)
            if isinstance(rate, dict) and "fundingRate" in rate:
                return float(rate["fundingRate"])
            if isinstance(rate, (int, float)):
                return float(rate)
    except Exception as e:
        logger.debug("Funding rate CCXT %s %s: %s", exchange_id, symbol, e)
    return None


def should_skip_long(
    symbol: str,
    funding_rate: Optional[float],
    threshold: float = 0.0001,
) -> bool:
    """Return True if long should be skipped (e.g. funding rate above threshold)."""
    if funding_rate is None:
        return False
    return float(funding_rate) >= threshold
