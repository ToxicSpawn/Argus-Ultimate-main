"""
Multi-Timeframe Confluence — requires signal agreement across 15m / 1h / 4h.

A signal is only approved if:
  1. The 15m signal direction matches the 1h signal direction
  2. The 1h signal does NOT contradict the 4h regime/trend
  3. The combined confidence meets a minimum threshold

**Partial-Candle Safety (v2)**
-----------------------------
The original implementation consumed close lists fed inline from the main
5-second loop.  When a higher-timeframe candle (e.g. 4h) had not yet
closed, the last element of the close list was a *partial* (live) candle,
causing signal contamination — effectively lookahead bias in live trading.

This version adds two defences:

1. **Push-based closed-candle cache** — callers call
   ``feed_closed_candle(symbol, tf, price, close_ts)`` on confirmed bar
   close events (exchange callback or ws message).  The internal cache
   only ever stores confirmed-closed candles.

2. **Partial-candle guard on check()** — if a market_data dict IS passed
   (legacy path), the guard inspects the newest close timestamp against
   ``now`` and drops the last element if the candle's period has not
   elapsed.  This is a conservative safety net; prefer the push-based API.

Usage (push-based — preferred)::

    filter = MTFConfluenceFilter(timeframes=["15m", "1h", "4h"])

    # On each confirmed bar close from the exchange WS:
    filter.feed_closed_candle("BTC/USDT", "1h", close_price=68432.0,
                               close_ts=datetime.utcnow())

    # On signal check (no market_data needed if cache is warm):
    approved, score, reason = filter.check(
        symbol="BTC/USDT",
        signal_direction="buy",
    )

Usage (legacy dict path)::

    approved, score, reason = filter.check(
        symbol="BTC/USD",
        signal_direction="buy",
        market_data={
            "15m": {"close": [...], "close_ts": [...], "volume": [...]},
            "1h":  {"close": [...], "close_ts": [...], "volume": [...]},
            "4h":  {"close": [...], "close_ts": [...], "volume": [...]},
        }
    )
    # Each "close_ts" list contains UTC datetime objects aligned 1:1 with
    # "close".  The guard will drop the last bar if its candle period has
    # not elapsed relative to datetime.utcnow().
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------

_TF_SECONDS: Dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


def _tf_to_seconds(tf: str) -> int:
    """Return the period of a timeframe label in seconds."""
    s = _TF_SECONDS.get(tf)
    if s is None:
        raise ValueError(f"Unknown timeframe: {tf!r}.  Known: {list(_TF_SECONDS)!r}")
    return s


def _is_candle_closed(newest_close_ts: datetime, tf: str, now: Optional[datetime] = None) -> bool:
    """
    Return True if the candle whose bar *opened* at ``newest_close_ts - period``
    has fully elapsed.

    Strategy: if ``now - newest_close_ts < tf_period`` the bar is still live.
    We give a 2-second grace margin to account for WS latency.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if newest_close_ts.tzinfo is None:
        newest_close_ts = newest_close_ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    period = timedelta(seconds=_tf_to_seconds(tf))
    grace = timedelta(seconds=2)
    elapsed_since_bar_close = now - newest_close_ts
    return elapsed_since_bar_close >= -grace


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MTFCandle:
    """A single confirmed-closed OHLCV candle on a given timeframe."""

    symbol: str
    timeframe: str
    close: float
    close_ts: datetime  # UTC close timestamp (bar end)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0


@dataclass
class _TFState:
    """Internal per-(symbol, timeframe) rolling close buffer."""

    closes: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    close_ts: Deque[datetime] = field(default_factory=lambda: deque(maxlen=200))
    direction: int = 0
    fast_ema: float = float("nan")
    slow_ema: float = float("nan")
    trend_ema: float = float("nan")
    last_close: float = float("nan")


# ---------------------------------------------------------------------------
# EMA helpers
# ---------------------------------------------------------------------------


def _compute_ema_series(data: List[float], period: int) -> np.ndarray:
    """
    Compute the full EMA series for a list of prices using numpy.

    Uses the standard smoothing factor k = 2 / (period + 1).
    The first EMA value is seeded with the simple average of the first
    ``period`` data points.

    Args:
        data:   List of price values (oldest first).
        period: EMA lookback period (must be >= 1).

    Returns:
        numpy array of EMA values, same length as ``data``.
        Returns an empty array if ``len(data) < period``.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")

    arr = np.array(data, dtype=float)
    n = len(arr)
    if n < period:
        return np.array([], dtype=float)

    k = 2.0 / (period + 1)
    ema = np.empty(n, dtype=float)
    ema[period - 1] = np.mean(arr[:period])

    for i in range(period, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)

    ema[: period - 1] = np.nan
    return ema


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MTFConfluenceFilter:
    """
    Multi-timeframe confluence filter with partial-candle safety.

    Attributes:
        timeframes:                  Ordered list of timeframe labels (lowest → highest).
        min_agreeing_tfs:            Minimum number of TFs that must agree.
        min_confluence_score:        Minimum fraction of TFs that must agree (0.0–1.0).
        fast_ema:                    Fast EMA period for crossover.
        slow_ema:                    Slow EMA period for crossover.
        trend_ema:                   Long-term EMA period for price trend.
        require_higher_tf_agreement: If True, the highest TF must not explicitly oppose
                                     the signal direction.
        enforce_closed_candles:      If True (default), the partial-candle guard is
                                     active on the legacy market_data path.
    """

    def __init__(
        self,
        timeframes: List[str] = None,
        min_agreeing_tfs: int = 2,
        min_confluence_score: float = 0.6,
        fast_ema: int = 9,
        slow_ema: int = 21,
        trend_ema: int = 50,
        require_higher_tf_agreement: bool = True,
        enforce_closed_candles: bool = True,
    ):
        self.timeframes = timeframes or ["15m", "1h", "4h"]
        self.min_agreeing_tfs = min_agreeing_tfs
        self.min_confluence_score = min_confluence_score
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.trend_ema = trend_ema
        self.require_higher_tf_agreement = require_higher_tf_agreement
        self.enforce_closed_candles = enforce_closed_candles

        # Push-based closed-candle state: {symbol: {tf: _TFState}}
        self._state: Dict[str, Dict[str, _TFState]] = {}

        # Legacy compat cache (used by update_cache / get_cached_direction)
        self._tf_cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Push-based API (preferred — use from live loop on bar-close events)
    # ------------------------------------------------------------------

    def feed_closed_candle(self, symbol: str, timeframe: str, close: float,
                           close_ts: datetime, open: float = 0.0,
                           high: float = 0.0, low: float = 0.0,
                           volume: float = 0.0) -> None:
        """
        Record a confirmed-closed candle into the rolling buffer.

        Call this from the exchange WS on_bar_close callback ONLY after the
        exchange confirms the bar has closed.  Never call it with a live
        (partial) candle.

        Args:
            symbol:    Trading symbol.
            timeframe: Timeframe label (e.g. "1h").
            close:     Confirmed close price of the closed bar.
            close_ts:  UTC datetime of the bar close.
        """
        if symbol not in self._state:
            self._state[symbol] = {}
        if timeframe not in self._state[symbol]:
            self._state[symbol][timeframe] = _TFState()

        state = self._state[symbol][timeframe]
        state.closes.append(close)
        state.close_ts.append(close_ts)

        closes_list = list(state.closes)
        if len(closes_list) >= self.trend_ema:
            direction = self._get_tf_direction(closes_list)
            fast_s = _compute_ema_series(closes_list, self.fast_ema)
            slow_s = _compute_ema_series(closes_list, self.slow_ema)
            trend_s = _compute_ema_series(closes_list, self.trend_ema)
            state.direction = direction
            state.fast_ema = float(fast_s[-1]) if len(fast_s) else float("nan")
            state.slow_ema = float(slow_s[-1]) if len(slow_s) else float("nan")
            state.trend_ema = float(trend_s[-1]) if len(trend_s) else float("nan")
            state.last_close = close

        logger.debug(
            "MTF feed_closed_candle: %s[%s] close=%.6f ts=%s direction=%d",
            symbol, timeframe, close, close_ts.isoformat(), state.direction,
        )

    def feed_candle(self, candle: MTFCandle) -> None:
        """Convenience wrapper accepting an MTFCandle dataclass."""
        self.feed_closed_candle(
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            close=candle.close,
            close_ts=candle.close_ts,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            volume=candle.volume,
        )

    def check(
        self,
        symbol: str,
        signal_direction: str,
        market_data: Optional[Dict[str, Dict]] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, float, str]:
        """
        Check if a signal passes multi-timeframe confluence.

        Preferred: call without ``market_data`` after warming the cache via
        ``feed_closed_candle()``.  The legacy ``market_data`` dict path is
        supported but will have the partial-candle guard applied.

        Args:
            symbol:           Trading symbol (e.g. "BTC/USD").
            signal_direction: "buy"/"long" or "sell"/"short".
            market_data:      Optional legacy dict: tf → {"close": [...],
                              "close_ts": [datetime, ...], "volume": [...]}.
                              If omitted, the push-based cache is used.
            now:              Reference UTC datetime for the partial-candle
                              guard (defaults to datetime.utcnow()).

        Returns:
            Tuple of:
              - approved (bool):          True if confluence passes.
              - confluence_score (float): Fraction of TFs agreeing (0.0–1.0).
              - reason (str):             Human-readable reason string.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        tf_directions: Dict[str, int] = {}

        # ---- Push-based path (preferred) ----------------------------------
        sym_state = self._state.get(symbol, {})
        for tf in self.timeframes:
            if tf in sym_state:
                state = sym_state[tf]
                if len(state.closes) >= self.trend_ema:
                    tf_directions[tf] = state.direction

        # ---- Legacy market_data path (with partial-candle guard) ----------
        if market_data:
            for tf in self.timeframes:
                if tf in tf_directions:
                    continue  # push-based already handled
                if tf not in market_data:
                    continue

                closes: List[float] = list(market_data[tf].get("close", []))
                close_ts_list: List[datetime] = list(market_data[tf].get("close_ts", []))

                # Partial-candle guard
                if self.enforce_closed_candles and closes and close_ts_list:
                    newest_ts = close_ts_list[-1]
                    if not _is_candle_closed(newest_ts, tf, now=now):
                        logger.warning(
                            "MTF[%s][%s]: partial candle detected (newest_ts=%s, now=%s) "
                            "— dropping last bar to prevent lookahead",
                            symbol, tf,
                            newest_ts.isoformat() if hasattr(newest_ts, "isoformat") else newest_ts,
                            now.isoformat(),
                        )
                        closes = closes[:-1]
                        close_ts_list = close_ts_list[:-1]

                if len(closes) < self.trend_ema + 1:
                    logger.debug(
                        "MTF[%s][%s]: insufficient data (%d closes, need %d)",
                        symbol, tf, len(closes), self.trend_ema + 1,
                    )
                    continue

                direction = self._get_tf_direction(closes)
                tf_directions[tf] = direction
                self.update_cache(symbol, tf, closes)

        if not tf_directions:
            logger.debug("MTF[%s]: no timeframe data available — pass-through", symbol)
            return True, 0.5, "no_mtf_data_available"

        direction_sign = 1 if signal_direction.lower() in ("buy", "long") else -1

        agreeing_tfs = [tf for tf, d in tf_directions.items() if d == direction_sign]
        agreements = len(agreeing_tfs)
        score = agreements / len(tf_directions)

        # Critical guard: highest timeframe must not explicitly oppose
        if self.require_higher_tf_agreement and self.timeframes:
            highest_tf = self.timeframes[-1]
            if highest_tf in tf_directions:
                highest_dir = tf_directions[highest_tf]
                if highest_dir == -direction_sign:
                    reason = f"highest_tf_{highest_tf}_opposes"
                    logger.info(
                        "MTF[%s]: REJECTED — %s explicitly opposes %s (score=%.2f)",
                        symbol, highest_tf, signal_direction, score,
                    )
                    return False, score, reason

        approved = agreements >= self.min_agreeing_tfs and score >= self.min_confluence_score
        reason = f"score={score:.2f} agreeing={agreeing_tfs}"

        if not approved:
            logger.info(
                "MTF[%s]: REJECTED — %s (need min_tfs=%d score>=%.2f)",
                symbol, reason, self.min_agreeing_tfs, self.min_confluence_score,
            )
        else:
            logger.debug("MTF[%s]: APPROVED — %s", symbol, reason)

        return approved, score, reason

    def get_timeframe_analysis(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Return a detailed per-timeframe analysis dictionary.

        Uses push-based cache if available; falls back to ``market_data``.
        """
        result: Dict[str, Any] = {}

        sym_state = self._state.get(symbol, {})

        for tf in self.timeframes:
            # Push-based state
            if tf in sym_state:
                state = sym_state[tf]
                closes_list = list(state.closes)
                n = len(closes_list)
                if n < self.trend_ema:
                    result[tf] = {"error": f"insufficient_data_{n}_of_{self.trend_ema}_required"}
                    continue
                result[tf] = {
                    "direction": state.direction,
                    "fast_ema": state.fast_ema,
                    "slow_ema": state.slow_ema,
                    "trend_ema": state.trend_ema,
                    "last_close": state.last_close,
                    "bullish": state.direction == 1,
                    "bearish": state.direction == -1,
                    "neutral": state.direction == 0,
                    "data_points": n,
                    "source": "push_cache",
                }
                continue

            # Legacy fallback
            if market_data and tf in market_data:
                closes = market_data[tf].get("close", [])
                n = len(closes)
                if n < self.trend_ema + 1:
                    result[tf] = {"error": f"insufficient_data_{n}_of_{self.trend_ema + 1}_required"}
                    continue

                fast_series = _compute_ema_series(closes, self.fast_ema)
                slow_series = _compute_ema_series(closes, self.slow_ema)
                trend_series = _compute_ema_series(closes, self.trend_ema)
                direction = self._get_tf_direction(closes)

                result[tf] = {
                    "direction": direction,
                    "fast_ema": float(fast_series[-1]),
                    "slow_ema": float(slow_series[-1]),
                    "trend_ema": float(trend_series[-1]),
                    "last_close": float(closes[-1]),
                    "bullish": direction == 1,
                    "bearish": direction == -1,
                    "neutral": direction == 0,
                    "data_points": n,
                    "source": "market_data",
                }
            else:
                result[tf] = {"error": "no_data"}

        return result

    def update_cache(self, symbol: str, timeframe: str, closes: List[float]) -> None:
        """Legacy compat: cache the last computed direction for a symbol/timeframe pair."""
        if len(closes) < self.trend_ema + 1:
            return
        direction = self._get_tf_direction(closes)
        if symbol not in self._tf_cache:
            self._tf_cache[symbol] = {}
        fast_series = _compute_ema_series(closes, self.fast_ema)
        slow_series = _compute_ema_series(closes, self.slow_ema)
        trend_series = _compute_ema_series(closes, self.trend_ema)
        self._tf_cache[symbol][timeframe] = {
            "direction": direction,
            "fast_ema": float(fast_series[-1]),
            "slow_ema": float(slow_series[-1]),
            "trend_ema": float(trend_series[-1]),
            "last_close": float(closes[-1]),
        }

    def get_cached_direction(self, symbol: str, timeframe: str) -> Optional[int]:
        """Retrieve a previously cached trend direction (legacy compat)."""
        # Prefer push-based state
        push_state = self._state.get(symbol, {}).get(timeframe)
        if push_state and len(push_state.closes) >= self.trend_ema:
            return push_state.direction
        entry = self._tf_cache.get(symbol, {}).get(timeframe)
        return entry["direction"] if entry else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tf_direction(self, closes: List[float]) -> int:
        """
        Compute trend direction for a series of **confirmed-closed** close prices.

        Requires that all closes passed in are from fully-closed bars.
        """
        fast_val = self._ema(closes, self.fast_ema)
        slow_val = self._ema(closes, self.slow_ema)
        trend_val = self._ema(closes, self.trend_ema)
        last_close = float(closes[-1])

        if fast_val > slow_val and last_close > trend_val:
            return 1
        if fast_val < slow_val and last_close < trend_val:
            return -1
        return 0

    def _ema(self, data: List[float], period: int) -> float:
        series = _compute_ema_series(data, period)
        if len(series) == 0:
            raise ValueError(
                f"Cannot compute EMA({period}): need >= {period} data points, got {len(data)}"
            )
        return float(series[-1])
