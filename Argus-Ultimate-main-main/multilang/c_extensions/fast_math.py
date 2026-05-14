"""
Python wrapper for the ARGUS fast_math C extension.

Loads the compiled shared library (fast_math.so / fast_math.dll) via ctypes.
Falls back to pure-numpy implementations when the library is not available.

Compile the C source:
    Linux/Mac:  gcc -O3 -shared -fPIC -o fast_math.so fast_math.c -lm
    Windows:    gcc -O3 -shared -o fast_math.dll fast_math.c

Usage:
    fm = FastMath()
    ema_out = fm.ema(prices, period=20)
    zsc_out = fm.rolling_zscore(values, window=30)
    wmid    = fm.weighted_mid(bids, asks, bid_sizes, ask_sizes)
"""

from __future__ import annotations

import ctypes
import logging
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_LIB_NAME = "fast_math.dll" if platform.system() == "Windows" else "fast_math.so"
_LIB_PATH = _THIS_DIR / _LIB_NAME


def _load_lib() -> Optional[ctypes.CDLL]:
    """Try to load the shared library; return None on failure."""
    if not _LIB_PATH.is_file():
        return None
    try:
        lib = ctypes.CDLL(str(_LIB_PATH))

        # exponential_moving_average(double*, int, int, double*)
        lib.exponential_moving_average.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.exponential_moving_average.restype = None

        # rolling_zscore(double*, int, int, double*)
        lib.rolling_zscore.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.rolling_zscore.restype = None

        # weighted_mid_price(double*, double*, double*, double*, int, double*)
        lib.weighted_mid_price.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.weighted_mid_price.restype = None

        return lib
    except OSError as exc:
        logger.warning("Failed to load fast_math shared lib: %s", exc)
        return None


class FastMath:
    """Hot-path math: C native with numpy fallback."""

    def __init__(self, lib_path: Optional[str] = None) -> None:
        if lib_path:
            global _LIB_PATH
            _LIB_PATH = Path(lib_path)
        self._lib = _load_lib()
        self._native_available = self._lib is not None
        self._backend = "native" if self._native_available else "fallback"
        self._call_count = 0
        self._total_latency = 0.0
        if self._native_available:
            logger.info("FastMath: C library loaded from %s", _LIB_PATH)
        else:
            logger.info("FastMath: shared library not found, using numpy fallback")

    @property
    def available(self) -> bool:
        return self._native_available

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def avg_latency_ms(self) -> float:
        if self._call_count == 0:
            return 0.0
        return (self._total_latency / self._call_count) * 1000.0

    # ── EMA ────────────────────────────────────────────────────────────────────

    def ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Exponential moving average."""
        prices = np.asarray(prices, dtype=np.float64)
        t0 = time.monotonic()
        try:
            if self._native_available:
                return self._c_ema(prices, period)
            return self._fb_ema(prices, period)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def _c_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        n = len(prices)
        output = np.empty(n, dtype=np.float64)
        self._lib.exponential_moving_average(
            prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_int(period),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return output

    @staticmethod
    def _fb_ema(prices: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1.0)
        output = np.empty_like(prices)
        output[0] = prices[0]
        for i in range(1, len(prices)):
            output[i] = alpha * prices[i] + (1.0 - alpha) * output[i - 1]
        return output

    # ── Rolling Z-Score ───────────────────────────────────────────────────────

    def rolling_zscore(self, values: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling z-score with given lookback window."""
        values = np.asarray(values, dtype=np.float64)
        t0 = time.monotonic()
        try:
            if self._native_available:
                return self._c_rolling_zscore(values, window)
            return self._fb_rolling_zscore(values, window)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def _c_rolling_zscore(self, values: np.ndarray, window: int) -> np.ndarray:
        n = len(values)
        output = np.empty(n, dtype=np.float64)
        self._lib.rolling_zscore(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_int(window),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return output

    @staticmethod
    def _fb_rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
        n = len(values)
        output = np.zeros(n, dtype=np.float64)
        if window <= 1:
            return output
        for i in range(window - 1, n):
            segment = values[i - window + 1: i + 1]
            m = np.mean(segment)
            s = np.std(segment, ddof=1)
            if s < 1e-15:
                output[i] = 0.0
            else:
                output[i] = (values[i] - m) / s
        return output

    # ── Weighted Mid Price ────────────────────────────────────────────────────

    def weighted_mid(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> np.ndarray:
        """Volume-weighted mid price."""
        bids = np.asarray(bids, dtype=np.float64)
        asks = np.asarray(asks, dtype=np.float64)
        bid_sizes = np.asarray(bid_sizes, dtype=np.float64)
        ask_sizes = np.asarray(ask_sizes, dtype=np.float64)
        t0 = time.monotonic()
        try:
            if self._native_available:
                return self._c_weighted_mid(bids, asks, bid_sizes, ask_sizes)
            return self._fb_weighted_mid(bids, asks, bid_sizes, ask_sizes)
        finally:
            self._call_count += 1
            self._total_latency += time.monotonic() - t0

    def _c_weighted_mid(
        self, bids: np.ndarray, asks: np.ndarray,
        bid_sizes: np.ndarray, ask_sizes: np.ndarray,
    ) -> np.ndarray:
        n = len(bids)
        output = np.empty(n, dtype=np.float64)
        self._lib.weighted_mid_price(
            bids.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            asks.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            bid_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ask_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return output

    @staticmethod
    def _fb_weighted_mid(
        bids: np.ndarray, asks: np.ndarray,
        bid_sizes: np.ndarray, ask_sizes: np.ndarray,
    ) -> np.ndarray:
        total = bid_sizes + ask_sizes
        # Where total is near zero, use simple mid
        safe_total = np.where(total < 1e-15, 1.0, total)
        weighted = (bids * ask_sizes + asks * bid_sizes) / safe_total
        simple_mid = (bids + asks) / 2.0
        wmid = np.where(total < 1e-15, simple_mid, weighted)
        return wmid
