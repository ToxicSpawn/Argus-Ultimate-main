"""
Live AUD/USD FX rate fetcher with caching.

Replaces the hardcoded 0.65 fallback throughout the codebase.
Uses free public APIs with graceful degradation to config default.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache: (rate, timestamp)
_cached_rate: Optional[float] = None
_cached_at: float = 0.0
_CACHE_TTL_SECONDS: float = 300.0  # 5 minutes


def get_aud_usd_rate(*, fallback: float = 0.65, cache_ttl: float = _CACHE_TTL_SECONDS) -> float:
    """
    Fetch live AUD/USD rate.  Returns cached value if fresh enough.

    Tries (in order):
      1. exchangerate.host (free, no key)
      2. open.er-api.com (free, no key)
      3. Falls back to `fallback` (config default, typically 0.65)

    This is synchronous (blocking HTTP) — call from an executor in async code,
    or accept ~200ms latency on cache miss.
    """
    global _cached_rate, _cached_at

    now = time.monotonic()
    if _cached_rate is not None and (now - _cached_at) < cache_ttl:
        return _cached_rate

    rate = _try_fetch()
    if rate is not None:
        _cached_rate = rate
        _cached_at = now
        logger.info("AUD/USD FX rate updated: %.5f", rate)
        return rate

    if _cached_rate is not None:
        logger.warning("FX fetch failed; using stale cached rate: %.5f", _cached_rate)
        return _cached_rate

    logger.warning("FX fetch failed; using fallback rate: %.5f", fallback)
    return fallback


def _try_fetch() -> Optional[float]:
    """Try multiple free FX APIs, return rate or None."""
    import urllib.request
    import json

    # Source 1: exchangerate.host
    try:
        url = "https://api.exchangerate.host/latest?base=AUD&symbols=USD"
        req = urllib.request.Request(url, headers={"User-Agent": "Argus-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            rate = float(data["rates"]["USD"])
            if 0.30 < rate < 1.20:  # sanity check
                return rate
    except Exception as exc:
        logger.debug("exchangerate.host failed: %s", exc)

    # Source 2: open.er-api.com
    try:
        url = "https://open.er-api.com/v6/latest/AUD"
        req = urllib.request.Request(url, headers={"User-Agent": "Argus-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            rate = float(data["rates"]["USD"])
            if 0.30 < rate < 1.20:
                return rate
    except Exception as exc:
        logger.debug("open.er-api.com failed: %s", exc)

    return None


def invalidate_cache() -> None:
    """Force next call to fetch fresh rate."""
    global _cached_rate, _cached_at
    _cached_rate = None
    _cached_at = 0.0
