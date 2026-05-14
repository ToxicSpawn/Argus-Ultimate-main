"""Indicator Cache — LRU+TTL cache AND incremental ring-buffer engine.

Improvements over the original LRU-only cache:
1. IncrementalIndicators class — maintains per-symbol deque ring buffers and
   computes EMA, ATR, RSI, and SMA with O(1) incremental updates per bar
   instead of recomputing from the full OHLCV history on every call.
   For a 200-bar EMA on 1m candles this is ~200x cheaper per bar.
2. IndicatorCache (original LRU+TTL) is preserved unchanged for any
   indicators that can't be made incremental yet.
3. Thread-safe; each symbol gets its own lock.
4. get_or_compute() helper combines cache-lookup + compute in one call.

Typical usage:
    inc = IncrementalIndicators(ema_periods=[9, 21, 50, 200], rsi_period=14, atr_period=14)

    for bar in stream:
        result = inc.update(
            symbol="BTC/USDT",
            close=bar.close, high=bar.high, low=bar.low,
        )
        ema9   = result["ema_9"]
        rsi14  = result["rsi_14"]
        atr14  = result["atr_14"]
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Original LRU + TTL cache (preserved for backward compatibility)
# ---------------------------------------------------------------------------

CacheKey = Tuple[str, str, str, Any]   # (symbol, timeframe, indicator_name, params)


class IndicatorCache:
    """
    Thread-safe LRU cache with per-entry TTL.
    Unchanged from original — kept for any caller that uses it directly.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: float = 60.0):
        self.max_size = max(1, int(max_size))
        self.ttl      = max(0.0, float(ttl_seconds))
        self._store: OrderedDict[CacheKey, Tuple[Any, float]] = OrderedDict()
        self._lock  = threading.Lock()
        self._hits  = 0
        self._misses = 0

    def get(self, key: CacheKey) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if self.ttl > 0 and time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: CacheKey, value: Any) -> None:
        expires_at = time.monotonic() + self.ttl if self.ttl > 0 else float("inf")
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def get_or_compute(self, key: CacheKey, fn, *args, **kwargs) -> Any:
        """Return cached value or compute, cache, and return it."""
        v = self.get(key)
        if v is not None:
            return v
        v = fn(*args, **kwargs)
        self.set(key, v)
        return v

    def invalidate(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> int:
        with self._lock:
            to_del = [
                k for k in self._store
                if (symbol    is None or (isinstance(k, tuple) and len(k) > 0 and k[0] == symbol))
                and (timeframe is None or (isinstance(k, tuple) and len(k) > 1 and k[1] == timeframe))
            ]
            for k in to_del:
                del self._store[k]
            return len(to_del)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":     len(self._store),
                "max_size": self.max_size,
                "hits":     self._hits,
                "misses":   self._misses,
                "hit_rate": self._hits / total if total else 0.0,
            }


# ---------------------------------------------------------------------------
# Per-symbol incremental state
# ---------------------------------------------------------------------------

class _SymbolState:
    """Holds all rolling buffers and EMA states for one symbol."""

    def __init__(
        self,
        ema_periods: List[int],
        rsi_period:  int,
        atr_period:  int,
        sma_periods: List[int],
    ) -> None:
        self._ema_periods = ema_periods
        self._rsi_period  = rsi_period
        self._atr_period  = atr_period
        self._sma_periods = sma_periods

        # EMA: {period: current_ema_value}
        self._ema: Dict[int, Optional[float]] = {p: None for p in ema_periods}
        # EMA multipliers
        self._ema_k: Dict[int, float] = {p: 2.0 / (p + 1) for p in ema_periods}

        # ATR: Wilder smoothed (RMA)
        self._atr:       Optional[float] = None
        self._prev_close: Optional[float] = None
        self._atr_k = 1.0 / atr_period

        # RSI: Wilder RS
        self._rsi_avg_gain: Optional[float] = None
        self._rsi_avg_loss: Optional[float] = None
        self._rsi_prev_close: Optional[float] = None
        self._rsi_warmup: List[float] = []

        # SMA: ring buffers
        self._sma_bufs: Dict[int, deque] = {
            p: deque(maxlen=p) for p in sma_periods
        }

        # ADX components (14-period DX smoothing)
        self._adx_period = 14
        self._adx: Optional[float] = None
        self._adx_plus_dm_smooth:  Optional[float] = None
        self._adx_minus_dm_smooth: Optional[float] = None
        self._adx_tr_smooth:       Optional[float] = None
        self._adx_dx_buf: deque = deque(maxlen=self._adx_period)

        self._n_bars = 0
        self._lock   = threading.Lock()

    def update(
        self,
        close: float,
        high:  float,
        low:   float,
    ) -> Dict[str, Optional[float]]:
        """Ingest one bar, return dict of all updated indicator values."""
        with self._lock:
            result: Dict[str, Optional[float]] = {}
            self._n_bars += 1

            # --- EMAs ---------------------------------------------------
            for p in self._ema_periods:
                if self._ema[p] is None:
                    self._ema[p] = close
                else:
                    k = self._ema_k[p]
                    self._ema[p] = close * k + self._ema[p] * (1.0 - k)
                result[f"ema_{p}"] = self._ema[p]

            # --- SMAs ---------------------------------------------------
            for p in self._sma_periods:
                self._sma_bufs[p].append(close)
                buf = self._sma_bufs[p]
                result[f"sma_{p}"] = sum(buf) / len(buf) if len(buf) == p else None

            # --- True Range & ATR (Wilder RMA) --------------------------
            tr = self._true_range(high, low, self._prev_close)
            if self._atr is None:
                self._atr = tr
            else:
                self._atr = tr * self._atr_k + self._atr * (1.0 - self._atr_k)
            result[f"atr_{self._atr_period}"] = self._atr
            result["atr"] = self._atr

            # --- RSI (Wilder RS) ----------------------------------------
            rsi_val = self._update_rsi(close)
            result[f"rsi_{self._rsi_period}"] = rsi_val
            result["rsi"] = rsi_val

            # --- ADX (14-period) ----------------------------------------
            adx_val = self._update_adx(high, low, tr)
            result["adx"] = adx_val

            # Store for next bar
            self._prev_close = close

            return result

    # ------------------------------------------------------------------
    # Incremental RSI
    # ------------------------------------------------------------------

    def _update_rsi(self, close: float) -> Optional[float]:
        p = self._rsi_period
        if self._rsi_prev_close is None:
            self._rsi_prev_close = close
            return None

        change = close - self._rsi_prev_close
        gain   = max(0.0, change)
        loss   = max(0.0, -change)
        self._rsi_prev_close = close

        if self._rsi_avg_gain is None:
            # Warmup period
            self._rsi_warmup.append((gain, loss))
            if len(self._rsi_warmup) < p:
                return None
            # First Wilder average
            self._rsi_avg_gain = sum(g for g, _ in self._rsi_warmup) / p
            self._rsi_avg_loss = sum(l for _, l in self._rsi_warmup) / p
            self._rsi_warmup.clear()
        else:
            # Wilder smoothing
            k = 1.0 / p
            self._rsi_avg_gain = gain * k + self._rsi_avg_gain * (1.0 - k)
            self._rsi_avg_loss = loss * k + self._rsi_avg_loss * (1.0 - k)

        if self._rsi_avg_loss < 1e-12:
            return 100.0
        rs  = self._rsi_avg_gain / self._rsi_avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ------------------------------------------------------------------
    # Incremental ADX
    # ------------------------------------------------------------------

    def _update_adx(
        self, high: float, low: float, tr: float
    ) -> Optional[float]:
        p = self._adx_period
        k = 1.0 / p

        plus_dm  = 0.0
        minus_dm = 0.0
        if self._prev_close is not None:
            prev_high = getattr(self, "_prev_high", high)
            prev_low  = getattr(self, "_prev_low",  low)
            up   = high - prev_high
            down = prev_low  - low
            if up > down and up > 0:
                plus_dm = up
            if down > up and down > 0:
                minus_dm = down

        self._prev_high = high
        self._prev_low  = low

        if self._adx_tr_smooth is None:
            self._adx_tr_smooth       = tr
            self._adx_plus_dm_smooth  = plus_dm
            self._adx_minus_dm_smooth = minus_dm
        else:
            self._adx_tr_smooth       = tr       * k + self._adx_tr_smooth       * (1.0 - k)
            self._adx_plus_dm_smooth  = plus_dm  * k + self._adx_plus_dm_smooth  * (1.0 - k)
            self._adx_minus_dm_smooth = minus_dm * k + self._adx_minus_dm_smooth * (1.0 - k)

        if self._adx_tr_smooth and self._adx_tr_smooth > 0:
            plus_di  = 100.0 * self._adx_plus_dm_smooth  / self._adx_tr_smooth
            minus_di = 100.0 * self._adx_minus_dm_smooth / self._adx_tr_smooth
            di_sum   = plus_di + minus_di
            dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
            self._adx_dx_buf.append(dx)

            if len(self._adx_dx_buf) >= p:
                if self._adx is None:
                    self._adx = sum(self._adx_dx_buf) / p
                else:
                    self._adx = dx * k + self._adx * (1.0 - k)
                return self._adx

        return None

    # ------------------------------------------------------------------
    # True Range
    # ------------------------------------------------------------------

    @staticmethod
    def _true_range(
        high: float, low: float, prev_close: Optional[float]
    ) -> float:
        if prev_close is None:
            return high - low
        return max(
            high - low,
            abs(high - prev_close),
            abs(low  - prev_close),
        )


# ---------------------------------------------------------------------------
# Public incremental engine
# ---------------------------------------------------------------------------

class IncrementalIndicators:
    """
    O(1)-per-bar incremental indicator engine.

    Maintains one _SymbolState per symbol. Call .update() once per closed bar.
    Returns a dict of all indicator values for that symbol.

    Example:
        indicators = IncrementalIndicators(
            ema_periods=[9, 21, 50, 200],
            rsi_period=14,
            atr_period=14,
            sma_periods=[20, 50],
        )
        result = indicators.update("BTC/USDT", close=65000, high=65500, low=64800)
        ema9  = result["ema_9"]
        rsi14 = result["rsi_14"]
        atr14 = result["atr_14"]
        adx   = result["adx"]
    """

    def __init__(
        self,
        ema_periods: Optional[List[int]] = None,
        rsi_period:  int = 14,
        atr_period:  int = 14,
        sma_periods: Optional[List[int]] = None,
    ) -> None:
        self._ema_periods = ema_periods or [9, 21, 50, 200]
        self._rsi_period  = rsi_period
        self._atr_period  = atr_period
        self._sma_periods = sma_periods or [20, 50]
        self._symbols: Dict[str, _SymbolState] = {}
        self._global_lock = threading.Lock()

    def update(
        self,
        symbol: str,
        close:  float,
        high:   float,
        low:    float,
    ) -> Dict[str, Optional[float]]:
        """
        Ingest one closed bar for symbol and return all indicator values.
        First call initialises state for this symbol.
        """
        state = self._get_or_create(symbol)
        return state.update(close=close, high=high, low=low)

    def reset(self, symbol: str) -> None:
        """Clear all state for a symbol (e.g. after gap/halt)."""
        with self._global_lock:
            self._symbols.pop(symbol, None)

    def reset_all(self) -> None:
        with self._global_lock:
            self._symbols.clear()

    def get_tracked_symbols(self) -> List[str]:
        with self._global_lock:
            return list(self._symbols.keys())

    def stats(self) -> dict:
        with self._global_lock:
            return {
                "tracked_symbols": len(self._symbols),
                "ema_periods":     self._ema_periods,
                "rsi_period":      self._rsi_period,
                "atr_period":      self._atr_period,
                "sma_periods":     self._sma_periods,
            }

    def _get_or_create(self, symbol: str) -> _SymbolState:
        with self._global_lock:
            if symbol not in self._symbols:
                self._symbols[symbol] = _SymbolState(
                    ema_periods = self._ema_periods,
                    rsi_period  = self._rsi_period,
                    atr_period  = self._atr_period,
                    sma_periods = self._sma_periods,
                )
        return self._symbols[symbol]
