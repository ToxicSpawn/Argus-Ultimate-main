"""
alpha/regime_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~
Market regime classifier that gates market-making activity.

Only quotes in mean-reverting regimes; halts in trending / high-volatility
regimes.

Regime detection pipeline
--------------------------
1. Rolling price buffer (deque, maxlen=200) per symbol.
2. Hurst exponent via rescaled-range (R/S) method over `hurst_window` bars.
3. Volatility regime via fast-vol / slow-vol ratio.
4. Combined classifier combining Hurst + vol-ratio rules.

Regime definitions
------------------
  MEAN_REVERTING  → H < 0.45 AND vol_ratio < 1.5
  TRENDING_UP     → H > 0.55 AND vol_ratio > trend_threshold AND recent return > 0
  TRENDING_DOWN   → H > 0.55 AND vol_ratio > trend_threshold AND recent return ≤ 0
  HIGH_VOLATILITY → vol_ratio > 3.0 (overrides Hurst)
  LOW_LIQUIDITY   → not enough bars yet to classify
  UNKNOWN         → all other combinations
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------


class MarketRegime(str, Enum):
    """Possible market regimes detected by RegimeDetector."""

    MEAN_REVERTING = "MEAN_REVERTING"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RegimeConfig:
    """Tuning knobs for RegimeDetector.

    Parameters
    ----------
    vol_lookback_fast : int
        Number of bars used for fast (short-term) volatility estimation.
    vol_lookback_slow : int
        Number of bars used for slow (long-term) volatility estimation.
    trend_threshold : float
        vol_ratio above this value signals trending or high-volatility.
    hurst_window : int
        Number of *price* observations fed into the Hurst R/S calculation.
    hurst_mean_revert_threshold : float
        H below this → mean reverting.
    hurst_trend_threshold : float
        H above this → trending.
    min_bars : int
        Minimum bars accumulated before a regime is emitted (below → LOW_LIQUIDITY).
    update_interval_ms : float
        Minimum milliseconds between re-classifications per symbol.
    """

    vol_lookback_fast: int = 20
    vol_lookback_slow: int = 100
    trend_threshold: float = 2.0
    hurst_window: int = 50
    hurst_mean_revert_threshold: float = 0.45
    hurst_trend_threshold: float = 0.55
    min_bars: int = 50
    update_interval_ms: float = 5_000.0


# ---------------------------------------------------------------------------
# Internal per-symbol state
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    """All mutable state for one symbol."""

    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    timestamps_ns: Deque[int] = field(default_factory=lambda: deque(maxlen=200))

    # Cached classification
    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 0.0
    hurst: float = 0.5
    vol_ratio: float = 1.0
    trend_strength: float = 0.0

    # Regime tracking
    last_classified_ns: int = 0  # monotonic ns at last classification
    regime_start_ns: int = 0     # when current regime started (monotonic ns)
    bars_in_regime: int = 0
    prev_regime: MarketRegime = MarketRegime.UNKNOWN


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RegimeDetector:
    """
    Classifies market regime per symbol and gates MM activity.

    Usage
    -----
    detector = RegimeDetector(RegimeConfig())
    detector.on_price("BTC-USD", 29_500.0, time.time_ns())
    ...
    if detector.is_mm_safe("BTC-USD"):
        place_quotes(...)
    """

    # Annualisation factor for 5-minute bar vol → annual vol
    # Bars per year = 252 days × 24 hours × 12 bars/hour
    _ANNUALISE = math.sqrt(252 * 24 * 12)

    def __init__(self, config: RegimeConfig) -> None:
        self._cfg = config
        self._states: Dict[str, _SymbolState] = {}
        self._callbacks: List[Callable[[str, MarketRegime, MarketRegime], None]] = []

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_regime_change_callback(
        self, callback: Callable[[str, MarketRegime, MarketRegime], None]
    ) -> None:
        """Register a callback fired when a symbol transitions regimes.

        Signature: ``callback(symbol, old_regime, new_regime)``
        """
        self._callbacks.append(callback)

    def _fire_callbacks(
        self, symbol: str, old: MarketRegime, new: MarketRegime
    ) -> None:
        for cb in self._callbacks:
            try:
                cb(symbol, old, new)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "RegimeDetector callback raised for symbol=%s", symbol
                )

    # ------------------------------------------------------------------
    # Price ingestion
    # ------------------------------------------------------------------

    def on_price(
        self, symbol: str, price: float, timestamp_ns: int
    ) -> None:
        """Feed a new price tick.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"BTC-USD"``).
        price : float
            Mid-price or last-trade price.
        timestamp_ns : int
            Unix epoch nanoseconds of the tick.
        """
        state = self._get_or_create(symbol)
        state.prices.append(price)
        state.timestamps_ns.append(timestamp_ns)

        # Throttle classification to update_interval_ms
        now_ns = time.monotonic_ns()
        elapsed_ms = (now_ns - state.last_classified_ns) / 1_000_000.0
        if elapsed_ms >= self._cfg.update_interval_ms or state.last_classified_ns == 0:
            self._classify(symbol, state, now_ns)

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_regime(self, symbol: str) -> MarketRegime:
        """Return the latest regime for *symbol* (UNKNOWN if no data)."""
        state = self._states.get(symbol)
        if state is None:
            return MarketRegime.UNKNOWN
        return state.regime

    def is_mm_safe(self, symbol: str) -> bool:
        """Return True only if the regime is MEAN_REVERTING.

        All other regimes (TRENDING_*, HIGH_VOLATILITY, UNKNOWN, LOW_LIQUIDITY)
        are considered unsafe for passive market-making.
        """
        return self.get_regime(symbol) == MarketRegime.MEAN_REVERTING

    def get_regime_confidence(self, symbol: str) -> float:
        """Return a confidence score in [0, 1] for the current classification."""
        state = self._states.get(symbol)
        if state is None:
            return 0.0
        return state.confidence

    def get_stats(self, symbol: str) -> dict:
        """Return a diagnostic snapshot for *symbol*.

        Keys
        ----
        regime, confidence, hurst_exponent, vol_ratio, trend_strength,
        mm_safe, bars_in_regime, time_in_regime_s
        """
        state = self._states.get(symbol)
        if state is None:
            return {
                "regime": MarketRegime.UNKNOWN.value,
                "confidence": 0.0,
                "hurst_exponent": 0.5,
                "vol_ratio": 1.0,
                "trend_strength": 0.0,
                "mm_safe": False,
                "bars_in_regime": 0,
                "time_in_regime_s": 0.0,
            }
        now_ns = time.monotonic_ns()
        time_in_regime_s = (now_ns - state.regime_start_ns) / 1e9 if state.regime_start_ns else 0.0
        return {
            "regime": state.regime.value,
            "confidence": state.confidence,
            "hurst_exponent": state.hurst,
            "vol_ratio": state.vol_ratio,
            "trend_strength": state.trend_strength,
            "mm_safe": self.is_mm_safe(symbol),
            "bars_in_regime": state.bars_in_regime,
            "time_in_regime_s": time_in_regime_s,
        }

    # ------------------------------------------------------------------
    # Classification engine
    # ------------------------------------------------------------------

    def _classify(
        self, symbol: str, state: _SymbolState, now_ns: int
    ) -> None:
        """Re-classify regime for *symbol* and update state."""
        state.last_classified_ns = now_ns
        prices = list(state.prices)
        n = len(prices)

        if n < self._cfg.min_bars:
            new_regime = MarketRegime.LOW_LIQUIDITY
            new_confidence = 0.0
            new_hurst = 0.5
            new_vol_ratio = 1.0
            new_trend_strength = 0.0
        else:
            returns = self._log_returns(prices)
            new_hurst = self._hurst_exponent(prices[-self._cfg.hurst_window :])
            new_vol_ratio, new_trend_strength = self._vol_regime(returns)
            new_regime, new_confidence = self._combine(
                prices, returns, new_hurst, new_vol_ratio, new_trend_strength
            )

        # Detect transition
        old_regime = state.regime
        if new_regime != old_regime:
            state.prev_regime = old_regime
            state.regime_start_ns = now_ns
            state.bars_in_regime = 1
            logger.debug(
                "RegimeDetector [%s]: %s → %s (H=%.3f vr=%.3f)",
                symbol,
                old_regime.value,
                new_regime.value,
                new_hurst,
                new_vol_ratio,
            )
            self._fire_callbacks(symbol, old_regime, new_regime)
        else:
            state.bars_in_regime += 1

        state.regime = new_regime
        state.confidence = new_confidence
        state.hurst = new_hurst
        state.vol_ratio = new_vol_ratio
        state.trend_strength = new_trend_strength

    def _combine(
        self,
        prices: List[float],
        returns: np.ndarray,
        hurst: float,
        vol_ratio: float,
        trend_strength: float,
    ) -> Tuple[MarketRegime, float]:
        """Apply classification rules and compute confidence."""
        cfg = self._cfg

        # Rule 1: extreme vol overrides everything
        if vol_ratio > 3.0:
            confidence = min(1.0, (vol_ratio - 3.0) / 2.0 + 0.5)
            return MarketRegime.HIGH_VOLATILITY, confidence

        # Rule 2: clear mean-reversion signal
        if hurst < cfg.hurst_mean_revert_threshold and vol_ratio < 1.5:
            # Confidence scales with how far H is below threshold
            h_margin = (cfg.hurst_mean_revert_threshold - hurst) / cfg.hurst_mean_revert_threshold
            vr_margin = max(0.0, (1.5 - vol_ratio) / 1.5)
            confidence = 0.5 + 0.25 * h_margin + 0.25 * vr_margin
            return MarketRegime.MEAN_REVERTING, min(1.0, confidence)

        # Rule 3: trending
        if hurst > cfg.hurst_trend_threshold and vol_ratio > cfg.trend_threshold:
            # Determine direction from recent return
            recent_return = float(np.mean(returns[-5:])) if len(returns) >= 5 else 0.0
            h_margin = (hurst - cfg.hurst_trend_threshold) / (1.0 - cfg.hurst_trend_threshold)
            vr_margin = min(1.0, (vol_ratio - cfg.trend_threshold) / cfg.trend_threshold)
            confidence = 0.5 + 0.25 * h_margin + 0.25 * vr_margin
            regime = (
                MarketRegime.TRENDING_UP if recent_return > 0 else MarketRegime.TRENDING_DOWN
            )
            return regime, min(1.0, confidence)

        # Rule 4: moderate vol / neutral Hurst → UNKNOWN
        return MarketRegime.UNKNOWN, 0.2

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_returns(prices: List[float]) -> np.ndarray:
        """Compute log returns from a price series."""
        arr = np.asarray(prices, dtype=float)
        # Guard against zeros / negatives (shouldn't happen for asset prices)
        arr = np.maximum(arr, 1e-12)
        return np.diff(np.log(arr))

    def _vol_regime(
        self, returns: np.ndarray
    ) -> Tuple[float, float]:
        """Compute (vol_ratio, trend_strength).

        vol_ratio  = fast_vol / slow_vol  (annualised)
        trend_strength = |mean_return| / fast_vol  (signal-to-noise)
        """
        cfg = self._cfg
        n = len(returns)
        if n < cfg.vol_lookback_fast:
            return 1.0, 0.0

        fast_returns = returns[-cfg.vol_lookback_fast :]
        fast_std = float(np.std(fast_returns, ddof=1))
        fast_vol = fast_std * self._ANNUALISE

        if n >= cfg.vol_lookback_slow:
            slow_returns = returns[-cfg.vol_lookback_slow :]
        else:
            slow_returns = returns
        slow_std = float(np.std(slow_returns, ddof=1))
        slow_vol = slow_std * self._ANNUALISE

        vol_ratio = fast_vol / slow_vol if slow_vol > 1e-12 else 1.0

        mean_ret = float(np.mean(fast_returns))
        trend_strength = abs(mean_ret) / fast_std if fast_std > 1e-12 else 0.0

        return vol_ratio, trend_strength

    @staticmethod
    def _hurst_exponent(prices: List[float]) -> float:
        """Estimate Hurst exponent via the rescaled-range (R/S) method.

        The algorithm:
        1. Compute log returns of the price series.
        2. For sub-windows of sizes N/4, N/3, N/2, N (varying scales),
           compute the R/S statistic: R/S = (max cumdev − min cumdev) / std.
        3. Regress log(R/S) on log(n) to obtain H.

        Returns H in [0, 1].  H ~ 0.5 → random walk.
        """
        if len(prices) < 10:
            return 0.5

        arr = np.asarray(prices, dtype=float)
        arr = np.maximum(arr, 1e-12)
        log_prices = np.log(arr)
        rets = np.diff(log_prices)
        N = len(rets)

        if N < 8:
            return 0.5

        # Build a range of sub-window sizes (at least 4 sizes)
        sizes: List[int] = []
        for divisor in [1, 2, 3, 4, 6, 8]:
            sz = max(4, N // divisor)
            if sz not in sizes and sz <= N:
                sizes.append(sz)
        sizes = sorted(set(sizes))

        log_ns: List[float] = []
        log_rs: List[float] = []

        for sz in sizes:
            # Compute R/S over non-overlapping sub-windows of length sz
            rs_values: List[float] = []
            for start in range(0, N - sz + 1, sz):
                chunk = rets[start : start + sz]
                mean_c = float(np.mean(chunk))
                devs = np.cumsum(chunk - mean_c)
                R = float(devs.max() - devs.min())
                S = float(np.std(chunk, ddof=1))
                if S > 1e-12:
                    rs_values.append(R / S)

            if rs_values:
                avg_rs = float(np.mean(rs_values))
                if avg_rs > 0:
                    log_ns.append(math.log(sz))
                    log_rs.append(math.log(avg_rs))

        if len(log_ns) < 2:
            return 0.5

        # OLS regression: log(R/S) = H * log(n) + c
        log_ns_arr = np.array(log_ns)
        log_rs_arr = np.array(log_rs)
        # np.polyfit degree 1
        try:
            coeffs = np.polyfit(log_ns_arr, log_rs_arr, 1)
            H = float(coeffs[0])
        except Exception:
            return 0.5

        # Clamp to reasonable range
        return max(0.05, min(0.95, H))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState()
        return self._states[symbol]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def symbols(self) -> List[str]:
        """Return all symbols with accumulated price data."""
        return list(self._states.keys())

    def reset(self, symbol: str) -> None:
        """Clear all accumulated state for *symbol*."""
        if symbol in self._states:
            del self._states[symbol]
            logger.info("RegimeDetector: reset state for %s", symbol)

    def snapshot_all(self) -> Dict[str, dict]:
        """Return ``get_stats()`` for every tracked symbol."""
        return {sym: self.get_stats(sym) for sym in self._states}

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self._states)
        return f"<RegimeDetector symbols={n} cfg={self._cfg}>"
