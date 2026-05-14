"""rate_limit_guard.py — token-bucket rate limiter for exchange API calls.

Usage (direct):
    guard = RateLimitGuard(capacity=10, refill_rate=10.0)  # 10 calls/sec
    with guard:
        response = exchange.fetch_ticker(symbol)

Usage (decorator):
    @guard.limit
    def fetch(symbol):
        return exchange.fetch_ticker(symbol)

Usage (per-exchange registry):
    limiter = ExchangeRateLimiter()
    limiter.register("kraken", capacity=10, refill_rate=10.0)
    limiter.register("coinbase", capacity=15, refill_rate=15.0)
    with limiter.guard("kraken"):
        ...
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Callable, Dict, Optional, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


class RateLimitExceeded(Exception):
    """Raised when token bucket is empty and block=False."""
    def __init__(self, exchange: str = "", retry_after_s: float = 0.0) -> None:
        self.exchange = exchange
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Rate limit exceeded for '{exchange}'. "
            f"Retry after {retry_after_s:.3f}s."
        )


class RateLimitGuard:
    """
    Thread-safe token bucket rate limiter.

    Args:
        capacity:    Max tokens (burst size)
        refill_rate: Tokens added per second
        exchange:    Name used in logs/metrics
        block:       If True, sleep until token available. If False, raise.
    """

    def __init__(
        self,
        capacity: float = 10.0,
        refill_rate: float = 10.0,
        exchange: str = "default",
        block: bool = True,
    ) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.exchange = exchange
        self.block = block
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.RLock()
        self._denied_count = 0
        self._allowed_count = 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + added)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> None:
        """Acquire tokens. Blocks if block=True, raises if block=False."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._allowed_count += 1
                return

            # Not enough tokens
            wait_s = (tokens - self._tokens) / max(self.refill_rate, 1e-9)

            if self.block:
                self._denied_count += 1
                logger.debug(
                    "RateLimitGuard[%s]: waiting %.3fs (tokens=%.2f)",
                    self.exchange, wait_s, self._tokens,
                )

            if not self.block:
                self._denied_count += 1
                logger.warning(
                    "RateLimitGuard[%s]: DENIED (tokens=%.2f, need=%.2f)",
                    self.exchange, self._tokens, tokens,
                )
                raise RateLimitExceeded(self.exchange, wait_s)

        # Block outside the lock to avoid holding it during sleep
        time.sleep(wait_s)
        with self._lock:
            self._refill()
            self._tokens -= tokens
            self._allowed_count += 1

    def __enter__(self) -> "RateLimitGuard":
        self.acquire()
        return self

    def __exit__(self, *_) -> None:
        pass

    def limit(self, fn: F) -> F:
        """Decorator: apply rate limit to a function."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            self.acquire()
            return fn(*args, **kwargs)
        return cast(F, wrapper)

    @property
    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "exchange": self.exchange,
                "tokens_available": round(self._tokens, 3),
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "allowed": self._allowed_count,
                "denied": self._denied_count,
            }


class ExchangeRateLimiter:
    """
    Registry of per-exchange RateLimitGuard instances.

    Exchange rate limits (conservative defaults based on API docs):
        kraken:   10 calls/sec, burst 10
        coinbase: 15 calls/sec, burst 15
        binance:  20 calls/sec, burst 20
        bybit:    10 calls/sec, burst 10
    """

    DEFAULTS: Dict[str, Dict[str, float]] = {
        "kraken":   {"capacity": 10.0, "refill_rate": 10.0},
        "coinbase": {"capacity": 15.0, "refill_rate": 15.0},
        "binance":  {"capacity": 20.0, "refill_rate": 20.0},
        "bybit":    {"capacity": 10.0, "refill_rate": 10.0},
        "okx":      {"capacity": 10.0, "refill_rate": 10.0},
    }

    def __init__(self) -> None:
        self._guards: Dict[str, RateLimitGuard] = {}
        self._lock = threading.Lock()
        # Auto-register known exchanges
        for name, cfg in self.DEFAULTS.items():
            self.register(name, **cfg)

    def register(
        self,
        exchange: str,
        capacity: float = 10.0,
        refill_rate: float = 10.0,
        block: bool = True,
    ) -> None:
        with self._lock:
            self._guards[exchange.lower()] = RateLimitGuard(
                capacity=capacity,
                refill_rate=refill_rate,
                exchange=exchange,
                block=block,
            )

    def guard(self, exchange: str) -> RateLimitGuard:
        """Get the guard for a named exchange. Auto-creates with defaults if unknown."""
        key = exchange.lower()
        with self._lock:
            if key not in self._guards:
                logger.warning(
                    "ExchangeRateLimiter: unknown exchange '%s', using defaults", exchange
                )
                self._guards[key] = RateLimitGuard(exchange=exchange)
            return self._guards[key]

    def acquire(self, exchange: str, tokens: float = 1.0) -> None:
        """Convenience: acquire tokens for an exchange."""
        self.guard(exchange).acquire(tokens)

    def all_stats(self) -> Dict[str, object]:
        with self._lock:
            return {k: v.stats for k, v in self._guards.items()}


# Module-level singleton — import and use directly
_limiter: Optional[ExchangeRateLimiter] = None


def get_limiter() -> ExchangeRateLimiter:
    """Get or create the module-level ExchangeRateLimiter singleton."""
    global _limiter
    if _limiter is None:
        _limiter = ExchangeRateLimiter()
    return _limiter
