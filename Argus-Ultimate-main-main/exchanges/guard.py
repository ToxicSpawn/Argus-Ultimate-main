"""exchanges/guard.py — rate-limit middleware for exchange adapters.

Wraps any exchange object (ccxt-compatible) with:
  1. Per-exchange token bucket rate limiting
  2. Exponential backoff retry on RateLimitExceeded
  3. Prometheus counter: argus_rate_limit_hits_total{exchange}

Usage:
    from exchanges.guard import GuardedExchange
    from utils.rate_limit_guard import get_limiter

    raw_exchange = ccxt.kraken({...})
    exchange = GuardedExchange(raw_exchange, exchange_name="kraken")
    ticker = exchange.fetch_ticker("BTC/USDT")   # rate-limited
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Prometheus counter — graceful no-op if prometheus_client not installed
try:
    from prometheus_client import Counter
    _RATE_LIMIT_HITS = Counter(
        "argus_rate_limit_hits_total",
        "Exchange API rate limit hits",
        ["exchange"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    _RATE_LIMIT_HITS = None  # type: ignore[assignment]


def _inc_counter(exchange: str) -> None:
    if _PROMETHEUS_AVAILABLE and _RATE_LIMIT_HITS is not None:
        try:
            _RATE_LIMIT_HITS.labels(exchange=exchange).inc()
        except Exception:
            pass


class GuardedExchange:
    """
    Proxy that wraps a ccxt-compatible exchange and enforces rate limits.

    Any attribute access that returns a callable is wrapped with:
    - Acquire a token before calling
    - Retry up to max_retries times on RateLimitExceeded
    - Exponential backoff: 0.5s, 1.0s, 2.0s
    """

    MAX_RETRIES = 3
    BACKOFF_BASE = 0.5  # seconds

    def __init__(
        self,
        exchange: Any,
        exchange_name: str = "",
        limiter: Optional[Any] = None,
    ) -> None:
        self._exchange = exchange
        self._name = exchange_name or getattr(exchange, "id", "unknown")

        if limiter is None:
            from utils.rate_limit_guard import get_limiter
            limiter = get_limiter()
        self._limiter = limiter

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._exchange, name)
        if callable(attr):
            return self._wrap(attr)
        return attr

    def _wrap(self, fn: Callable) -> Callable:
        from utils.rate_limit_guard import RateLimitExceeded

        def guarded(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(self.MAX_RETRIES):
                try:
                    self._limiter.acquire(self._name)
                    return fn(*args, **kwargs)
                except RateLimitExceeded as exc:
                    _inc_counter(self._name)
                    sleep_s = self.BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "GuardedExchange[%s]: rate limited (attempt %d/%d), backing off %.1fs",
                        self._name, attempt + 1, self.MAX_RETRIES, sleep_s,
                    )
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    time.sleep(sleep_s)
            return None  # unreachable

        return guarded
