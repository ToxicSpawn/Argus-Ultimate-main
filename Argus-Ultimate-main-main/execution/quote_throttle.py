"""
quote_throttle.py — Smart Quote Refresh Throttling.

Prevents over-quoting by suppressing quote updates that are too small,
too frequent, or occur too soon after the previous quote.

Three suppression conditions (all must pass for a refresh to be allowed):

1. **Min-tick filter**: both bid and ask must move by at least *min_tick*
   from the previous sent quote.
2. **Min-age filter**: the last sent quote must be at least *min_age_ms*
   milliseconds ago.
3. **Rate limiter**: a token-bucket allows at most *max_rate_per_sec*
   refreshes per second per symbol.

Usage::

    filt = QuoteThrottleFilter(min_tick=0.01, min_age_ms=50.0, max_rate_per_sec=20)

    if filt.should_refresh("BTC/USD", new_bid=30_000.0, new_ask=30_001.0,
                           last_bid=29_999.99, last_ask=30_000.99):
        # Send quote
        exchange.quote(...)
        filt.record_refresh("BTC/USD")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

log = logging.getLogger("argus.quote_throttle")

# ---------------------------------------------------------------------------
# Token bucket (module-level, re-exported for tests and external use)
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Leaky-bucket / token-bucket rate limiter.

    Tokens are replenished at *rate* tokens per second up to *capacity*.
    Each ``consume()`` call tries to take one token; returns ``True`` if
    successful, ``False`` if the bucket is currently empty.

    Parameters
    ----------
    rate : float
        Tokens added per second.
    capacity : float, optional
        Maximum bucket depth (defaults to *rate*, i.e. 1-second burst).
    """

    def __init__(self, rate: float, capacity: Optional[float] = None) -> None:
        if rate <= 0:
            raise ValueError(f"TokenBucket rate must be positive, got {rate}")
        self.rate = float(rate)
        self.capacity = float(capacity) if capacity is not None else float(rate)
        self._tokens: float = self.capacity
        self._last_ts: float = time.monotonic()

    # ── Internal ─────────────────────────────────────────────────────────

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_ts
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_ts = now

    # ── Public API ────────────────────────────────────────────────────────

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume *tokens* from the bucket.

        Returns
        -------
        bool
            ``True`` if tokens were available and consumed; ``False`` otherwise.
        """
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def reset(self) -> None:
        """Refill the bucket to capacity immediately."""
        self._tokens = self.capacity
        self._last_ts = time.monotonic()

    @property
    def available(self) -> float:
        """Token count currently available (after implicit refill)."""
        self._refill()
        return self._tokens

    def __repr__(self) -> str:
        return (
            f"TokenBucket(rate={self.rate}, capacity={self.capacity}, "
            f"available={self._tokens:.3f})"
        )


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------

@dataclass
class _SymbolState:
    """Mutable tracking state for one symbol."""

    last_sent_bid: Optional[float] = None
    last_sent_ask: Optional[float] = None
    last_refresh_ns: int = 0       # monotonic ns timestamp of last sent refresh
    total_sent: int = 0
    total_suppressed: int = 0

    # Per-symbol token bucket (created lazily)
    bucket: Optional[TokenBucket] = field(default=None, repr=False)

    def ms_since_last_refresh(self) -> float:
        """Milliseconds elapsed since the last recorded refresh."""
        if self.last_refresh_ns == 0:
            return float("inf")
        elapsed_ns = time.monotonic_ns() - self.last_refresh_ns
        return elapsed_ns / 1_000_000.0


# ---------------------------------------------------------------------------
# QuoteThrottleFilter
# ---------------------------------------------------------------------------

class QuoteThrottleFilter:
    """
    Decide whether a new quote should be sent to the exchange.

    Parameters
    ----------
    min_tick : float
        Minimum price movement (absolute) required on either side before a
        refresh is considered.  E.g. ``0.01`` for a 1-cent tick.
    min_age_ms : float
        Minimum milliseconds that must have elapsed since the last sent
        quote before a new one is allowed.  Default ``50.0`` (20 Hz max).
    max_rate_per_sec : int
        Token-bucket rate cap in quotes per second per symbol.
        Default ``20``.
    """

    def __init__(
        self,
        min_tick: float,
        min_age_ms: float = 50.0,
        max_rate_per_sec: int = 20,
    ) -> None:
        if min_tick <= 0:
            raise ValueError(f"min_tick must be positive, got {min_tick}")
        if min_age_ms < 0:
            raise ValueError(f"min_age_ms must be non-negative, got {min_age_ms}")
        if max_rate_per_sec <= 0:
            raise ValueError(f"max_rate_per_sec must be positive, got {max_rate_per_sec}")

        self.min_tick = float(min_tick)
        self.min_age_ms = float(min_age_ms)
        self.max_rate_per_sec = int(max_rate_per_sec)

        self._symbols: Dict[str, _SymbolState] = {}

    # ── Internal helpers ──────────────────────────────────────────────────

    def _state(self, symbol: str) -> _SymbolState:
        """Return (creating if absent) the state for *symbol*."""
        if symbol not in self._symbols:
            st = _SymbolState()
            st.bucket = TokenBucket(
                rate=self.max_rate_per_sec,
                capacity=self.max_rate_per_sec,
            )
            self._symbols[symbol] = st
        return self._symbols[symbol]

    # ── Public API ────────────────────────────────────────────────────────

    def should_refresh(
        self,
        symbol: str,
        new_bid: float,
        new_ask: float,
        last_bid: Optional[float] = None,
        last_ask: Optional[float] = None,
    ) -> bool:
        """
        Evaluate whether a new quote should be sent.

        Parameters
        ----------
        symbol : str
            The trading pair, e.g. ``"BTC/USD"``.
        new_bid : float
            Proposed new best bid price.
        new_ask : float
            Proposed new best ask price.
        last_bid : float, optional
            The previously *sent* bid price.  If omitted, falls back to the
            internally tracked last-sent bid (set by ``record_refresh``).
        last_ask : float, optional
            The previously *sent* ask price.

        Returns
        -------
        bool
            ``True`` if the refresh should proceed; ``False`` if suppressed.
        """
        st = self._state(symbol)

        # Resolve last prices
        prev_bid = last_bid if last_bid is not None else st.last_sent_bid
        prev_ask = last_ask if last_ask is not None else st.last_sent_ask

        # ── Check 1: min-tick filter ───────────────────────────────────
        if prev_bid is not None and prev_ask is not None:
            bid_move = abs(new_bid - prev_bid)
            ask_move = abs(new_ask - prev_ask)
            if bid_move < self.min_tick and ask_move < self.min_tick:
                st.total_suppressed += 1
                log.debug(
                    "QuoteThrottle[%s]: suppressed — move bid=%.6f ask=%.6f < min_tick=%.6f",
                    symbol, bid_move, ask_move, self.min_tick,
                )
                return False

        # ── Check 2: min-age filter ────────────────────────────────────
        age_ms = st.ms_since_last_refresh()
        if age_ms < self.min_age_ms:
            st.total_suppressed += 1
            log.debug(
                "QuoteThrottle[%s]: suppressed — age=%.2fms < min_age=%.2fms",
                symbol, age_ms, self.min_age_ms,
            )
            return False

        # ── Check 3: token-bucket rate limit ──────────────────────────
        assert st.bucket is not None
        if not st.bucket.consume():
            st.total_suppressed += 1
            log.debug(
                "QuoteThrottle[%s]: suppressed — rate limit exceeded (%.2f tok/s)",
                symbol, self.max_rate_per_sec,
            )
            return False

        return True

    def record_refresh(self, symbol: str, bid: Optional[float] = None, ask: Optional[float] = None) -> None:
        """
        Record that a quote was actually sent for *symbol*.

        Updates the last-sent prices and the refresh timestamp.

        Parameters
        ----------
        symbol : str
        bid : float, optional
            The bid price that was sent.  Use ``None`` to leave unchanged.
        ask : float, optional
            The ask price that was sent.
        """
        st = self._state(symbol)
        st.last_refresh_ns = time.monotonic_ns()
        st.total_sent += 1
        if bid is not None:
            st.last_sent_bid = bid
        if ask is not None:
            st.last_sent_ask = ask
        log.debug(
            "QuoteThrottle[%s]: refresh recorded bid=%s ask=%s",
            symbol, bid, ask,
        )

    def get_stats(self, symbol: str) -> Dict[str, object]:
        """
        Return throttle statistics for *symbol*.

        Returns
        -------
        dict with keys:
            ``total_suppressed``, ``total_sent``, ``suppression_rate``,
            ``last_refresh_ms_ago``, ``available_tokens``.
        """
        st = self._state(symbol)
        total = st.total_sent + st.total_suppressed
        suppression_rate = st.total_suppressed / total if total > 0 else 0.0
        assert st.bucket is not None
        return {
            "total_suppressed":   st.total_suppressed,
            "total_sent":         st.total_sent,
            "suppression_rate":   suppression_rate,
            "last_refresh_ms_ago": st.ms_since_last_refresh(),
            "available_tokens":   st.bucket.available,
        }

    def reset_symbol(self, symbol: str) -> None:
        """
        Reset all state for *symbol*.

        Call this when a position is closed and you want to start fresh.
        """
        if symbol in self._symbols:
            del self._symbols[symbol]
        log.debug("QuoteThrottle[%s]: state reset", symbol)

    def reset_all(self) -> None:
        """Reset state for all tracked symbols."""
        self._symbols.clear()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def tracked_symbols(self) -> list:
        """List of symbols currently being tracked."""
        return list(self._symbols.keys())

    def __repr__(self) -> str:
        return (
            f"QuoteThrottleFilter("
            f"min_tick={self.min_tick}, "
            f"min_age_ms={self.min_age_ms}, "
            f"max_rate_per_sec={self.max_rate_per_sec}, "
            f"symbols={len(self._symbols)}"
            f")"
        )
