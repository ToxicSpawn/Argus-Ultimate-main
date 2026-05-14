"""
Rate Limit Manager — centralised per-exchange API rate limit tracker.

Prevents 429 errors by tracking request counts per exchange per endpoint type
using a token bucket algorithm.  All operations are thread-safe.

Conservative default limits (tuned below published maximums to provide
headroom for burst absorption):

  Kraken:   REST 15 req/s public, 15 req/s private; WS unrestricted
  Coinbase: REST 10 req/s public, 15 req/s private
  Bybit:    REST 20 req/s public, 10 req/s private
  Deribit:  REST 20 req/s (single tier for all endpoints)

Usage:
    mgr = RateLimitManager()
    if mgr.check("kraken", EndpointType.PRIVATE):
        # fire request immediately
        ...
    else:
        allowed = await asyncio.get_running_loop().run_in_executor(
            None, mgr.wait_for_slot, "kraken", EndpointType.PRIVATE, 5.0
        )
"""
from __future__ import annotations

import asyncio
import enum
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EndpointType(enum.Enum):
    """Category of exchange API endpoint."""

    PUBLIC = "public"
    PRIVATE = "private"
    WEBSOCKET = "websocket"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RateLimit:
    """Configuration for a single bucket (endpoint type on an exchange)."""

    requests_per_second: float   # sustained throughput
    burst_size: int              # maximum tokens that can accumulate


# ---------------------------------------------------------------------------
# Token Bucket
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Thread-safe token bucket for rate limiting.

    Tokens accumulate at ``refill_rate`` per second up to ``max_tokens``.
    Each request consumes one (or more) tokens.
    """

    def __init__(self, requests_per_second: float, burst_size: int) -> None:
        self.refill_rate: float = requests_per_second
        self.max_tokens: float = float(burst_size)
        self.tokens: float = float(burst_size)          # start full
        self.last_refill: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consume(self, n: int = 1) -> bool:
        """
        Attempt to consume ``n`` tokens.

        Returns True and deducts tokens if sufficient tokens are available;
        returns False without any state change if tokens are insufficient.
        """
        with self._lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def wait_time(self) -> float:
        """
        Seconds until at least one token is available.

        Returns 0.0 if a token is available right now.
        """
        with self._lock:
            self._refill()
            if self.tokens >= 1.0:
                return 0.0
            deficit = 1.0 - self.tokens
            return deficit / self.refill_rate if self.refill_rate > 0 else float("inf")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time.  Caller must hold _lock."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


# ---------------------------------------------------------------------------
# Default limit configuration
# ---------------------------------------------------------------------------

DEFAULT_LIMITS: Dict[str, Dict[EndpointType, RateLimit]] = {
    "kraken": {
        EndpointType.PUBLIC:    RateLimit(requests_per_second=15.0, burst_size=20),
        EndpointType.PRIVATE:   RateLimit(requests_per_second=15.0, burst_size=20),
        EndpointType.WEBSOCKET: RateLimit(requests_per_second=100.0, burst_size=200),
    },
    "coinbase": {
        EndpointType.PUBLIC:    RateLimit(requests_per_second=10.0, burst_size=15),
        EndpointType.PRIVATE:   RateLimit(requests_per_second=15.0, burst_size=20),
        EndpointType.WEBSOCKET: RateLimit(requests_per_second=100.0, burst_size=200),
    },
    "bybit": {
        EndpointType.PUBLIC:    RateLimit(requests_per_second=20.0, burst_size=30),
        EndpointType.PRIVATE:   RateLimit(requests_per_second=10.0, burst_size=15),
        EndpointType.WEBSOCKET: RateLimit(requests_per_second=100.0, burst_size=200),
    },
    "deribit": {
        EndpointType.PUBLIC:    RateLimit(requests_per_second=20.0, burst_size=30),
        EndpointType.PRIVATE:   RateLimit(requests_per_second=20.0, burst_size=30),
        EndpointType.WEBSOCKET: RateLimit(requests_per_second=100.0, burst_size=200),
    },
    "okx": {
        EndpointType.PUBLIC:    RateLimit(requests_per_second=20.0, burst_size=30),
        EndpointType.PRIVATE:   RateLimit(requests_per_second=10.0, burst_size=15),
        EndpointType.WEBSOCKET: RateLimit(requests_per_second=100.0, burst_size=200),
    },
}


# ---------------------------------------------------------------------------
# RateLimitManager
# ---------------------------------------------------------------------------

class RateLimitManager:
    """
    Central registry of token buckets for all known exchanges.

    Unknown exchanges are accepted with a permissive default bucket
    (10 req/s public, 5 req/s private) to avoid blocking unexpected venues.
    """

    _PERMISSIVE_LIMITS: Dict[EndpointType, RateLimit] = {
        EndpointType.PUBLIC:    RateLimit(requests_per_second=10.0, burst_size=15),
        EndpointType.PRIVATE:   RateLimit(requests_per_second=5.0,  burst_size=8),
        EndpointType.WEBSOCKET: RateLimit(requests_per_second=100.0, burst_size=200),
    }

    def __init__(
        self,
        custom_limits: Optional[Dict[str, Dict[EndpointType, RateLimit]]] = None,
    ) -> None:
        # Per-exchange per-endpoint token buckets
        limits = dict(DEFAULT_LIMITS) if custom_limits is None else custom_limits
        self._buckets: Dict[str, Dict[EndpointType, TokenBucket]] = {}
        for exchange, ep_map in limits.items():
            self._buckets[exchange] = {
                ep: TokenBucket(rl.requests_per_second, rl.burst_size)
                for ep, rl in ep_map.items()
            }
        logger.info(
            "RateLimitManager initialised for %d exchanges: %s",
            len(self._buckets), sorted(self._buckets.keys()),
        )

    # ------------------------------------------------------------------
    # FIX 20: Build from unified_config.yaml rate_limits section
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "RateLimitManager":
        """
        Build a RateLimitManager with limits from config.

        Expected config format::

            rate_limits:
              kraken:
                private: 15
                public: 15
              coinbase:
                private: 15
                public: 10

        Missing values fall back to DEFAULT_LIMITS.
        """
        if not config:
            return cls()

        custom: Dict[str, Dict[EndpointType, RateLimit]] = {}
        for exchange, ep_limits in config.items():
            if not isinstance(ep_limits, dict):
                continue
            exchange_lower = exchange.lower()
            base = DEFAULT_LIMITS.get(exchange_lower, {})
            ep_map: Dict[EndpointType, RateLimit] = {}
            for ep_type in EndpointType:
                ep_name = ep_type.value.lower()
                if ep_name in ep_limits:
                    rps = float(ep_limits[ep_name])
                    burst = int(rps * 1.5) + 1
                    ep_map[ep_type] = RateLimit(requests_per_second=rps, burst_size=burst)
                elif ep_type in base:
                    ep_map[ep_type] = base[ep_type]
                else:
                    ep_map[ep_type] = RateLimit(requests_per_second=10.0, burst_size=15)
            custom[exchange_lower] = ep_map

        # Merge in exchanges not mentioned in config
        for exchange, ep_map in DEFAULT_LIMITS.items():
            if exchange not in custom:
                custom[exchange] = ep_map

        return cls(custom_limits=custom)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, exchange: str, endpoint_type: EndpointType) -> bool:
        """
        Attempt to consume one token for the given exchange and endpoint.

        Returns True if the request is permitted; False if rate-limited.
        Registers unknown exchanges with permissive defaults.
        """
        bucket = self._get_bucket(exchange, endpoint_type)
        allowed = bucket.consume(1)
        if not allowed:
            logger.debug(
                "RateLimitManager: rate limit hit exchange=%s endpoint=%s",
                exchange, endpoint_type.value,
            )
        return allowed

    def wait_for_slot(
        self,
        exchange: str,
        endpoint_type: EndpointType,
        timeout: float = 5.0,
    ) -> bool:
        """
        Block until a token is available or ``timeout`` seconds have elapsed.

        Returns True if a token was successfully consumed; False on timeout.
        Safe to call from a thread pool executor for async callers.
        """
        deadline = time.monotonic() + timeout
        bucket = self._get_bucket(exchange, endpoint_type)

        while True:
            if bucket.consume(1):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "RateLimitManager: timeout waiting for slot exchange=%s endpoint=%s",
                    exchange, endpoint_type.value,
                )
                return False
            wait = min(bucket.wait_time(), remaining)
            if wait > 0:
                time.sleep(wait)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Return per-exchange utilisation statistics.

        Each exchange entry maps endpoint type name to:
            tokens        — current token level
            max_tokens    — bucket capacity
            utilisation   — percentage consumed (0-100)
        """
        stats: Dict[str, Dict[str, float]] = {}
        for exchange, ep_map in self._buckets.items():
            exchange_stats: Dict[str, float] = {}
            for ep, bucket in ep_map.items():
                max_t = bucket.max_tokens
                current_t = bucket.tokens
                util = max(0.0, (max_t - current_t) / max_t * 100.0) if max_t > 0 else 0.0
                exchange_stats[ep.value] = {  # type: ignore[assignment]
                    "tokens": current_t,
                    "max_tokens": max_t,
                    "utilisation_pct": util,
                }
            stats[exchange] = exchange_stats  # type: ignore[assignment]
        return stats  # type: ignore[return-value]

    async def wait_if_needed(self, exchange: str, endpoint_type: EndpointType, timeout: float = 5.0) -> bool:
        """
        Async version of wait_for_slot. Waits in a thread executor to avoid blocking the event loop.

        Returns True if a token was consumed; False on timeout.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.wait_for_slot, exchange, endpoint_type, timeout,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_bucket(self, exchange: str, endpoint_type: EndpointType) -> TokenBucket:
        """Return the bucket for the given exchange+endpoint, creating if absent."""
        if exchange not in self._buckets:
            logger.warning(
                "RateLimitManager: unknown exchange '%s'; applying permissive defaults",
                exchange,
            )
            self._buckets[exchange] = {
                ep: TokenBucket(rl.requests_per_second, rl.burst_size)
                for ep, rl in self._PERMISSIVE_LIMITS.items()
            }

        ep_map = self._buckets[exchange]
        if endpoint_type not in ep_map:
            # Register with permissive limits for unknown endpoint type
            perm = self._PERMISSIVE_LIMITS.get(
                endpoint_type,
                RateLimit(requests_per_second=5.0, burst_size=8),
            )
            ep_map[endpoint_type] = TokenBucket(perm.requests_per_second, perm.burst_size)

        return ep_map[endpoint_type]


# ---------------------------------------------------------------------------
# Decorator for mandatory rate limiting
# ---------------------------------------------------------------------------

def rate_limited(endpoint_type: EndpointType, exchange_arg: str = "exchange") -> Callable:
    """
    Async decorator that enforces rate limiting before function execution.

    The decorated function must either:
      - Accept ``exchange`` as a keyword argument, or
      - Accept it as the first positional argument after ``self``.

    Alternatively, ``exchange_arg`` names the keyword argument to read.

    The decorator looks up the RateLimitManager instance from the object's
    ``_rate_limit_manager`` attribute, or falls back to a module-level
    ``_default_rate_limit_manager`` if set.

    Usage::

        class MyClient:
            def __init__(self):
                self._rate_limit_manager = RateLimitManager()

            @rate_limited(EndpointType.PRIVATE)
            async def fetch_balance(self, exchange: str = "kraken"):
                ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Resolve exchange name
            exchange = kwargs.get(exchange_arg)
            if exchange is None and len(args) >= 2:
                # args[0] is self, args[1] is exchange
                exchange = args[1]
            if exchange is None:
                exchange = "unknown"

            # Resolve rate limit manager
            mgr = None
            if args and hasattr(args[0], "_rate_limit_manager"):
                mgr = args[0]._rate_limit_manager
            if mgr is None:
                mgr = globals().get("_default_rate_limit_manager")

            if mgr is not None:
                await mgr.wait_if_needed(str(exchange), endpoint_type)

            return await fn(*args, **kwargs)
        return wrapper
    return decorator
