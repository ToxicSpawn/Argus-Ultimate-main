"""
Cross-Asset Contagion Model — detects when a price shock in one asset
propagates to others, estimates cascade timing, and identifies hedge
candidates that move inversely during stress.

Operates in-memory with a configurable rolling window (default 24h).
Uses numpy for correlation and lag analysis when available, with a pure-Python
fallback.

Usage:
    model = ContagionModel(window_hours=24)
    model.update_price("BTC/USD", 65000)
    model.update_price("ETH/USD", 3400)
    model.update_price("SOL/USD", 145)
    ...
    report = model.detect_contagion("BTC/USD", threshold_pct=3.0)
    cascade = model.get_cascade_order("BTC/USD")
    hedges = model.get_hedge_symbols(["BTC/USD", "ETH/USD"])
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional numpy
# ---------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore[import]
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContagionReport:
    """Result of contagion detection from a source symbol."""
    source: str
    affected_symbols: List[str]
    lag_seconds: Dict[str, float]       # symbol -> seconds lag from source
    correlation: Dict[str, float]       # symbol -> correlation coefficient
    severity: str                       # "low" / "medium" / "high" / "critical"
    timestamp: float = field(default_factory=time.time)


@dataclass
class _PricePoint:
    """Single price observation."""
    timestamp: float
    price: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContagionModel:
    """
    Cross-asset contagion detector and cascade order estimator.

    Parameters
    ----------
    window_hours : int
        Rolling window for correlation/lag analysis (default 24).
    max_points : int
        Maximum price points retained per symbol.
    min_return_count : int
        Minimum number of return observations required for correlation.
    """

    def __init__(
        self,
        window_hours: int = 24,
        max_points: int = 10_000,
        min_return_count: int = 20,
    ) -> None:
        self._window_seconds = window_hours * 3600
        self._max_points = max(50, max_points)
        self._min_return_count = max(5, min_return_count)

        # symbol -> deque[_PricePoint]
        self._prices: Dict[str, Deque[_PricePoint]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        logger.info(
            "ContagionModel initialised  window=%dh  max_points=%d  numpy=%s",
            window_hours, self._max_points, _HAS_NUMPY,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_lock(self, symbol: str) -> threading.Lock:
        with self._global_lock:
            if symbol not in self._locks:
                self._locks[symbol] = threading.Lock()
            return self._locks[symbol]

    def _prune(self, symbol: str) -> None:
        cutoff = time.time() - self._window_seconds
        dq = self._prices.get(symbol)
        if dq:
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()

    def _get_returns(self, symbol: str) -> List[Tuple[float, float]]:
        """
        Compute log returns for a symbol.

        Returns list of (timestamp, log_return).
        """
        lock = self._get_lock(symbol)
        with lock:
            points = list(self._prices.get(symbol, []))

        if len(points) < 2:
            return []

        returns = []
        for i in range(1, len(points)):
            if points[i - 1].price > 0:
                lr = math.log(points[i].price / points[i - 1].price)
                returns.append((points[i].timestamp, lr))
        return returns

    @staticmethod
    def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
        """Pure-Python Pearson correlation coefficient."""
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
        sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
        if sx == 0 or sy == 0:
            return 0.0
        return cov / (sx * sy)

    def _align_returns(
        self,
        returns_a: List[Tuple[float, float]],
        returns_b: List[Tuple[float, float]],
        max_lag_seconds: float = 300,
    ) -> Tuple[List[float], List[float]]:
        """
        Align two return series by timestamp with tolerance.

        Returns paired lists of returns.
        """
        aligned_a: List[float] = []
        aligned_b: List[float] = []

        j = 0
        for ts_a, ret_a in returns_a:
            while j < len(returns_b) and returns_b[j][0] < ts_a - max_lag_seconds:
                j += 1
            if j < len(returns_b) and abs(returns_b[j][0] - ts_a) <= max_lag_seconds:
                aligned_a.append(ret_a)
                aligned_b.append(returns_b[j][1])
                j += 1

        return aligned_a, aligned_b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float, timestamp: Optional[float] = None) -> None:
        """
        Record a price observation.

        Parameters
        ----------
        symbol : str
        price : float
        timestamp : float or None
            Unix timestamp; uses current time if None.
        """
        ts = timestamp if timestamp is not None else time.time()
        point = _PricePoint(timestamp=ts, price=price)

        lock = self._get_lock(symbol)
        with lock:
            if symbol not in self._prices:
                self._prices[symbol] = deque(maxlen=self._max_points)
            self._prices[symbol].append(point)
            self._prune(symbol)

    def detect_contagion(
        self,
        source_symbol: str,
        threshold_pct: float = 3.0,
    ) -> ContagionReport:
        """
        Detect whether a recent shock in *source_symbol* is propagating.

        A "shock" is defined as a recent return magnitude exceeding
        *threshold_pct*.  For each other tracked symbol, we compute
        correlation and estimated lag.

        Parameters
        ----------
        source_symbol : str
        threshold_pct : float
            Minimum recent return magnitude (%) to qualify as a shock.

        Returns
        -------
        ContagionReport
        """
        source_returns = self._get_returns(source_symbol)
        if len(source_returns) < self._min_return_count:
            return ContagionReport(
                source=source_symbol,
                affected_symbols=[],
                lag_seconds={},
                correlation={},
                severity="low",
            )

        # Check if source had a recent shock
        recent_returns = [r for _, r in source_returns[-10:]]
        if recent_returns:
            max_recent_abs = max(abs(r) for r in recent_returns) * 100  # to %
        else:
            max_recent_abs = 0.0

        with self._global_lock:
            all_symbols = [s for s in self._prices if s != source_symbol]

        affected: List[str] = []
        lag_seconds: Dict[str, float] = {}
        correlations: Dict[str, float] = {}

        for sym in all_symbols:
            sym_returns = self._get_returns(sym)
            if len(sym_returns) < self._min_return_count:
                continue

            # Align and correlate
            aligned_src, aligned_sym = self._align_returns(source_returns, sym_returns)
            if len(aligned_src) < self._min_return_count:
                continue

            if _HAS_NUMPY:
                corr = float(np.corrcoef(aligned_src, aligned_sym)[0, 1])
                if math.isnan(corr):
                    corr = 0.0
            else:
                corr = self._pearson_correlation(aligned_src, aligned_sym)

            correlations[sym] = round(corr, 4)

            # Estimate lag: find time offset of maximum cross-correlation
            lag = self._estimate_lag(source_returns, sym_returns)
            lag_seconds[sym] = round(lag, 1)

            # Consider "affected" if correlation > 0.5 and source had a shock
            if abs(corr) > 0.5 and max_recent_abs >= threshold_pct:
                affected.append(sym)

        # Severity
        severity = self._classify_severity(len(affected), correlations, max_recent_abs)

        report = ContagionReport(
            source=source_symbol,
            affected_symbols=affected,
            lag_seconds=lag_seconds,
            correlation=correlations,
            severity=severity,
        )

        if affected:
            logger.warning(
                "Contagion detected  source=%s  shock=%.2f%%  affected=%s  severity=%s",
                source_symbol, max_recent_abs, affected, severity,
            )

        return report

    def get_cascade_order(self, source_symbol: str) -> List[str]:
        """
        Return symbols ordered by expected cascade timing (fastest propagation first).

        Uses estimated lag from cross-correlation analysis.

        Parameters
        ----------
        source_symbol : str

        Returns
        -------
        list of str
            Symbols sorted by ascending lag.
        """
        source_returns = self._get_returns(source_symbol)
        if len(source_returns) < self._min_return_count:
            return []

        with self._global_lock:
            all_symbols = [s for s in self._prices if s != source_symbol]

        lags: List[Tuple[str, float]] = []
        for sym in all_symbols:
            sym_returns = self._get_returns(sym)
            if len(sym_returns) < self._min_return_count:
                continue
            lag = self._estimate_lag(source_returns, sym_returns)
            lags.append((sym, lag))

        # Sort by lag ascending (fastest propagation first)
        lags.sort(key=lambda x: x[1])

        result = [sym for sym, _ in lags]
        logger.debug("Cascade order from %s: %s", source_symbol, result)
        return result

    def get_hedge_symbols(self, portfolio_symbols: List[str]) -> List[str]:
        """
        Identify symbols that tend to move inversely during stress periods.

        Looks for negative correlation with portfolio constituents.

        Parameters
        ----------
        portfolio_symbols : list of str

        Returns
        -------
        list of str
            Symbols with negative average correlation to the portfolio.
        """
        with self._global_lock:
            all_symbols = list(self._prices.keys())

        candidates = [s for s in all_symbols if s not in portfolio_symbols]
        if not candidates:
            return []

        # Compute portfolio returns (equal-weight average)
        portfolio_returns_list = []
        for sym in portfolio_symbols:
            rets = self._get_returns(sym)
            if rets:
                portfolio_returns_list.append(rets)

        if not portfolio_returns_list:
            return []

        # Use the shortest series as reference length
        min_len = min(len(r) for r in portfolio_returns_list)
        if min_len < self._min_return_count:
            return []

        # Average returns across portfolio
        avg_returns: List[Tuple[float, float]] = []
        for i in range(min_len):
            ts = portfolio_returns_list[0][-(min_len - i)][0]
            avg_ret = sum(
                rl[-(min_len - i)][1] for rl in portfolio_returns_list
            ) / len(portfolio_returns_list)
            avg_returns.append((ts, avg_ret))

        hedges: List[Tuple[str, float]] = []
        for sym in candidates:
            sym_returns = self._get_returns(sym)
            if len(sym_returns) < self._min_return_count:
                continue
            aligned_port, aligned_sym = self._align_returns(avg_returns, sym_returns)
            if len(aligned_port) < self._min_return_count:
                continue

            if _HAS_NUMPY:
                corr = float(np.corrcoef(aligned_port, aligned_sym)[0, 1])
                if math.isnan(corr):
                    corr = 0.0
            else:
                corr = self._pearson_correlation(aligned_port, aligned_sym)

            # Negative correlation = good hedge
            if corr < -0.2:
                hedges.append((sym, corr))

        # Sort by most negative correlation (best hedge first)
        hedges.sort(key=lambda x: x[1])

        result = [sym for sym, _ in hedges]
        logger.debug("Hedge symbols for %s: %s", portfolio_symbols, result)
        return result

    def get_symbols(self) -> List[str]:
        """Return all tracked symbols."""
        with self._global_lock:
            return list(self._prices.keys())

    # ------------------------------------------------------------------
    # Lag estimation
    # ------------------------------------------------------------------

    def _estimate_lag(
        self,
        source_returns: List[Tuple[float, float]],
        target_returns: List[Tuple[float, float]],
    ) -> float:
        """
        Estimate propagation lag in seconds using cross-correlation at
        multiple offsets.

        Returns the lag (in seconds) that maximises correlation.
        """
        if not source_returns or not target_returns:
            return 0.0

        # Average time delta between observations (for converting index offset to seconds)
        src_timestamps = [ts for ts, _ in source_returns]
        if len(src_timestamps) < 2:
            return 0.0
        avg_dt = (src_timestamps[-1] - src_timestamps[0]) / (len(src_timestamps) - 1)
        if avg_dt <= 0:
            avg_dt = 60.0  # default 1 minute

        src_vals = [r for _, r in source_returns]
        tgt_vals = [r for _, r in target_returns]

        # Test offsets from -10 to +10 periods
        best_corr = -2.0
        best_offset = 0

        min_len = min(len(src_vals), len(tgt_vals))
        max_offset = min(10, min_len // 3)

        for offset in range(-max_offset, max_offset + 1):
            if offset >= 0:
                s = src_vals[:min_len - offset]
                t = tgt_vals[offset:offset + len(s)]
            else:
                t = tgt_vals[:min_len + offset]
                s = src_vals[-offset:-offset + len(t)]

            if len(s) < self._min_return_count:
                continue

            if _HAS_NUMPY and len(s) > 1:
                c = float(np.corrcoef(s, t)[0, 1])
                if math.isnan(c):
                    c = 0.0
            else:
                c = self._pearson_correlation(s, t)

            if c > best_corr:
                best_corr = c
                best_offset = offset

        lag_seconds = best_offset * avg_dt
        return max(0.0, lag_seconds)  # only positive lag makes sense for cascade

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_severity(
        affected_count: int,
        correlations: Dict[str, float],
        shock_pct: float,
    ) -> str:
        """Classify contagion severity."""
        high_corr_count = sum(1 for c in correlations.values() if abs(c) > 0.7)

        if shock_pct >= 10.0 and affected_count >= 3:
            return "critical"
        elif shock_pct >= 5.0 and affected_count >= 2:
            return "high"
        elif affected_count >= 1 or high_corr_count >= 2:
            return "medium"
        else:
            return "low"
