"""
microprice_drift.py — Short-term microprice drift signal.

The microprice (Stoikov 2018) is a volume-weighted mid that adjusts the
raw mid-price toward the side with more liquidity:

    microprice = (bid_size × ask_price + ask_size × bid_price)
                 / (bid_size + ask_size)

When microprice > mid consistently, it indicates upward short-term drift;
when microprice < mid it indicates downward drift.

This module tracks a rolling window of (microprice > mid) booleans and
derives bullish/bearish/neutral signals and quote-skew recommendations.

Classes
-------
DriftState              — snapshot dataclass
MicropriceDriftSignal   — main signal class
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class DriftState:
    """Complete drift snapshot for a single symbol."""

    symbol: str
    drift: float          # proportion of snapshots where microprice > mid [0, 1]
    signal: str           # "bullish" / "bearish" / "neutral"
    magnitude: float      # signed drift: (above - below) / window, range [-1, 1]
    quote_skew: float     # recommended bid/ask skew in ticks, range [-2, +2]
    window: int           # current window fill depth
    timestamp_ns: int


# ─── Internal per-symbol state ────────────────────────────────────────────────


@dataclass
class _SymbolDriftState:
    window: int                           # max window size
    threshold: float                      # bullish/bearish threshold (e.g. 0.6)

    # Boolean ring-buffer: True = microprice > mid at that snapshot
    observations: Deque[bool] = field(default_factory=deque)

    last_microprice: float = 0.0
    last_mid: float = 0.0
    last_update_ns: int = 0

    def __post_init__(self) -> None:
        self.observations = deque(maxlen=self.window)

    def add_observation(self, microprice: float, mid: float) -> None:
        self.observations.append(microprice > mid)
        self.last_microprice = microprice
        self.last_mid = mid

    def above_count(self) -> int:
        return sum(1 for v in self.observations if v)

    def below_count(self) -> int:
        return sum(1 for v in self.observations if not v)

    def n(self) -> int:
        return len(self.observations)


# ─── MicropriceDriftSignal ────────────────────────────────────────────────────


class MicropriceDriftSignal:
    """
    Short-term microprice drift signal for market-making quote adjustment.

    Parameters
    ----------
    window : int
        Number of LOB snapshots in the rolling drift window.
    threshold : float
        Proportion cutoff for directional signals.
        drift > threshold        → bullish
        drift < (1 - threshold)  → bearish
        else                     → neutral

    Example
    -------
        signal = MicropriceDriftSignal(window=20, threshold=0.6)

        # On each LOB update:
        micro = compute_microprice(bids, asks)
        mid   = (best_bid + best_ask) / 2
        signal.on_book_update("BTCUSDT", micro, mid, time.time_ns())

        # Read signal:
        drift   = signal.get_drift("BTCUSDT")
        skew    = signal.get_quote_skew("BTCUSDT")
        state   = signal.get_drift_state("BTCUSDT")
    """

    def __init__(self, window: int = 20, threshold: float = 0.6) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if not (0.5 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0.5, 1.0], got {threshold}")

        self._window = window
        self._threshold = threshold
        self._states: Dict[str, _SymbolDriftState] = {}

        logger.info(
            "MicropriceDriftSignal initialised: window=%d threshold=%.2f",
            window,
            threshold,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def on_book_update(
        self,
        symbol: str,
        microprice: float,
        mid: float,
        timestamp_ns: int,
    ) -> None:
        """
        Ingest a new LOB snapshot.

        Parameters
        ----------
        symbol       : trading pair
        microprice   : volume-weighted mid (Stoikov microprice)
        mid          : arithmetic mid-price = (best_bid + best_ask) / 2
        timestamp_ns : nanosecond timestamp
        """
        if mid <= 0.0:
            logger.debug(
                "MicropriceDriftSignal: invalid mid %.6f for %s — skipping",
                mid,
                symbol,
            )
            return

        state = self._get_state(symbol)
        state.add_observation(microprice, mid)
        state.last_update_ns = timestamp_ns

        logger.debug(
            "MicropriceDrift %s: micro=%.6f mid=%.6f above=%s drift=%.3f",
            symbol,
            microprice,
            mid,
            microprice > mid,
            self.get_drift(symbol),
        )

    def get_drift(self, symbol: str) -> float:
        """
        Proportion of the last N snapshots where microprice > mid.

        Range [0, 1]:
          > threshold      → bullish signal
          < (1-threshold)  → bearish signal
          else             → neutral

        Returns 0.5 if no data.
        """
        state = self._states.get(symbol)
        if state is None or state.n() == 0:
            return 0.5

        n = state.n()
        above = state.above_count()
        return above / n

    def get_drift_signal(self, symbol: str) -> str:
        """
        Directional signal: "bullish" / "bearish" / "neutral".

        Thresholds:
          drift >  threshold       → "bullish"
          drift < (1 - threshold)  → "bearish"
          else                     → "neutral"
        """
        drift = self.get_drift(symbol)
        if drift > self._threshold:
            return "bullish"
        if drift < (1.0 - self._threshold):
            return "bearish"
        return "neutral"

    def get_drift_magnitude(self, symbol: str) -> float:
        """
        Signed drift magnitude: (above_count - below_count) / window.

        Range [-1, 1]:
          +1.0 → every snapshot had microprice > mid (strongly bullish)
          -1.0 → every snapshot had microprice < mid (strongly bearish)
           0.0 → perfectly balanced or no data
        """
        state = self._states.get(symbol)
        if state is None or state.n() == 0:
            return 0.0

        n = state.n()
        above = state.above_count()
        below = state.below_count()
        return (above - below) / n

    def get_quote_skew(self, symbol: str) -> float:
        """
        Suggested bid/ask quote skew in ticks.

        Linearly maps drift_magnitude [-1, 1] → skew [-2, +2] ticks.

        Interpretation:
          positive → skew ask price up (anticipate upward move; be more aggressive bid)
          negative → skew bid price down (anticipate downward move; be more aggressive ask)
        """
        magnitude = self.get_drift_magnitude(symbol)
        # Linear map: [-1, 1] → [-2, +2]
        return magnitude * 2.0

    def get_drift_state(self, symbol: str) -> DriftState:
        """Return a complete DriftState snapshot for *symbol*."""
        state = self._states.get(symbol)
        ts = state.last_update_ns if state else time.time_ns()
        n = state.n() if state else 0

        return DriftState(
            symbol=symbol,
            drift=self.get_drift(symbol),
            signal=self.get_drift_signal(symbol),
            magnitude=self.get_drift_magnitude(symbol),
            quote_skew=self.get_quote_skew(symbol),
            window=n,
            timestamp_ns=ts,
        )

    def reset(self, symbol: str) -> None:
        """
        Clear all drift history for *symbol*.

        Should be called when a position is closed to avoid stale signal bleed.
        """
        if symbol in self._states:
            del self._states[symbol]
            logger.info("MicropriceDriftSignal: reset state for %s", symbol)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_microprice(
        best_bid: float,
        best_ask: float,
        bid_size: float,
        ask_size: float,
    ) -> float:
        """
        Compute Stoikov microprice from best-level bid/ask prices and sizes.

        microprice = (bid_size × ask_price + ask_size × bid_price)
                     / (bid_size + ask_size)

        Falls back to arithmetic mid if sizes are both zero.
        """
        total = bid_size + ask_size
        if total <= 0.0 or best_bid <= 0.0 or best_ask <= 0.0:
            return (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0.0 else 0.0
        return (bid_size * best_ask + ask_size * best_bid) / total

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def _get_state(self, symbol: str) -> _SymbolDriftState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolDriftState(
                window=self._window,
                threshold=self._threshold,
            )
        return self._states[symbol]

    def symbols(self) -> List[str]:
        return list(self._states.keys())

    def snapshot_all(self) -> Dict[str, DriftState]:
        return {sym: self.get_drift_state(sym) for sym in self._states}
