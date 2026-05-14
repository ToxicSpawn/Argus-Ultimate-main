"""
Chart Pattern Detector — signal-processing-based pattern recognition.

Detects classic chart patterns using scipy.signal peak/trough detection,
numpy correlation/template matching, and linear regression. No CNN or
heavy CV libraries required.

Patterns detected:
  - Head and Shoulders / Inverse Head and Shoulders
  - Double Top / Double Bottom
  - Triangle (ascending, descending, symmetric)
  - Flag / Pennant (continuation)
  - Cup and Handle
  - Wedge (rising, falling)
  - Support and Resistance levels
  - Price/indicator divergence

Dependencies: numpy, scipy.signal (optional — graceful degradation to numpy).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try scipy.signal for peak detection; fall back to numpy
try:
    from scipy.signal import argrelextrema, find_peaks
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.debug("chart_pattern_cnn: scipy.signal unavailable, using numpy fallback")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """A detected chart pattern."""
    pattern: str            # pattern name
    confidence: float       # 0–1
    start_idx: int          # start index in price array
    end_idx: int            # end index in price array
    direction: str          # BULLISH / BEARISH / NEUTRAL
    target_price: float     # projected target price
    description: str = ""


@dataclass
class SupportResistance:
    """Support and resistance levels."""
    support: List[float]
    resistance: List[float]
    strength: List[float]   # strength score for each level


@dataclass
class DivergenceMatch:
    """Detected divergence between price and indicator."""
    type: str               # "bullish" or "bearish"
    start_idx: int
    end_idx: int
    confidence: float


@dataclass
class PatternSignal:
    """Aggregate pattern signal."""
    bias: float             # -1 to +1
    confidence: float       # 0–1
    patterns: List[PatternMatch]
    support: float          # nearest support
    resistance: float       # nearest resistance


# ---------------------------------------------------------------------------
# Numpy fallback peak/trough detection
# ---------------------------------------------------------------------------

def _find_peaks_numpy(data: np.ndarray, order: int = 5) -> np.ndarray:
    """Find local maxima using numpy (fallback when scipy unavailable)."""
    peaks = []
    n = len(data)
    for i in range(order, n - order):
        if all(data[i] >= data[i - j] for j in range(1, order + 1)) and \
           all(data[i] >= data[i + j] for j in range(1, order + 1)):
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _find_troughs_numpy(data: np.ndarray, order: int = 5) -> np.ndarray:
    """Find local minima using numpy."""
    return _find_peaks_numpy(-data, order)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_local_peaks(data: np.ndarray, order: int = 5) -> np.ndarray:
    """Find indices of local maxima."""
    if _SCIPY_AVAILABLE:
        indices = argrelextrema(data, np.greater_equal, order=order)[0]
        return indices
    return _find_peaks_numpy(data, order)


def _find_local_troughs(data: np.ndarray, order: int = 5) -> np.ndarray:
    """Find indices of local minima."""
    if _SCIPY_AVAILABLE:
        indices = argrelextrema(data, np.less_equal, order=order)[0]
        return indices
    return _find_troughs_numpy(data, order)


def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Simple OLS linear regression. Returns (slope, intercept)."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-10:
        return 0.0, float(y_mean)
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def _r_squared(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """Compute R-squared for linear fit."""
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _relative_diff(a: float, b: float) -> float:
    """Relative difference between two values."""
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ChartPatternDetector:
    """
    Detects chart patterns in price data using signal processing techniques.

    No heavy ML dependencies — uses scipy.signal for peak detection and
    numpy for correlation/template matching.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._min_pattern_bars = int(cfg.get("min_pattern_bars", 10))
        self._peak_order = int(cfg.get("peak_order", 5))
        self._sr_tolerance_pct = float(cfg.get("sr_tolerance_pct", 0.02))
        self._min_confidence = float(cfg.get("min_confidence", 0.3))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_patterns(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
    ) -> List[dict]:
        """
        Detect chart patterns in price series.

        Parameters
        ----------
        prices : list of float
            Close prices (oldest first).
        volumes : list of float, optional
            Volume data aligned with prices.

        Returns
        -------
        list of dict, each with keys:
            pattern, confidence, start_idx, end_idx, direction, target_price
        """
        arr = np.array(prices, dtype=np.float64)
        if len(arr) < self._min_pattern_bars:
            return []

        vol_arr = np.array(volumes, dtype=np.float64) if volumes else None
        patterns: List[PatternMatch] = []

        # Find peaks and troughs
        peaks = _find_local_peaks(arr, order=self._peak_order)
        troughs = _find_local_troughs(arr, order=self._peak_order)

        if len(peaks) < 2 or len(troughs) < 2:
            return self._to_dicts(patterns)

        # Detect each pattern type
        patterns.extend(self._detect_head_shoulders(arr, peaks, troughs))
        patterns.extend(self._detect_inv_head_shoulders(arr, peaks, troughs))
        patterns.extend(self._detect_double_top(arr, peaks, troughs))
        patterns.extend(self._detect_double_bottom(arr, peaks, troughs))
        patterns.extend(self._detect_triangles(arr, peaks, troughs))
        patterns.extend(self._detect_flags(arr, peaks, troughs, vol_arr))
        patterns.extend(self._detect_cup_handle(arr, peaks, troughs))
        patterns.extend(self._detect_wedges(arr, peaks, troughs))

        # Filter by minimum confidence
        patterns = [p for p in patterns if p.confidence >= self._min_confidence]

        # Sort by confidence descending
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return self._to_dicts(patterns)

    def find_support_resistance(
        self,
        prices: List[float],
        n_levels: int = 5,
    ) -> dict:
        """
        Find key support and resistance levels.

        Returns dict with:
            support: list of prices
            resistance: list of prices
            strength: list of floats (hit count normalised)
        """
        arr = np.array(prices, dtype=np.float64)
        if len(arr) < 10:
            return {"support": [], "resistance": [], "strength": []}

        current_price = arr[-1]

        # Find all peaks and troughs
        peaks = _find_local_peaks(arr, order=max(3, self._peak_order // 2))
        troughs = _find_local_troughs(arr, order=max(3, self._peak_order // 2))

        # Collect all turning point prices
        levels: List[float] = []
        if len(peaks) > 0:
            levels.extend(arr[peaks].tolist())
        if len(troughs) > 0:
            levels.extend(arr[troughs].tolist())

        if not levels:
            return {"support": [], "resistance": [], "strength": []}

        # Cluster nearby levels
        clusters = self._cluster_levels(levels, tolerance=self._sr_tolerance_pct)

        # Separate into support and resistance based on current price
        support_levels: List[Tuple[float, float]] = []  # (price, strength)
        resistance_levels: List[Tuple[float, float]] = []

        for price_level, count in clusters:
            strength = min(1.0, count / 5.0)
            if price_level < current_price:
                support_levels.append((price_level, strength))
            else:
                resistance_levels.append((price_level, strength))

        # Sort: support descending (nearest first), resistance ascending
        support_levels.sort(key=lambda x: x[0], reverse=True)
        resistance_levels.sort(key=lambda x: x[0])

        # Take top n_levels
        support_levels = support_levels[:n_levels]
        resistance_levels = resistance_levels[:n_levels]

        all_levels = support_levels + resistance_levels
        all_strengths = [s for _, s in all_levels]

        return {
            "support": [p for p, _ in support_levels],
            "resistance": [p for p, _ in resistance_levels],
            "strength": all_strengths,
        }

    def detect_divergence(
        self,
        prices: List[float],
        indicator: List[float],
    ) -> List[dict]:
        """
        Detect bullish/bearish divergence between price and indicator.

        Bullish divergence: price makes lower low, indicator makes higher low.
        Bearish divergence: price makes higher high, indicator makes lower high.

        Returns list of dicts with: type, start_idx, end_idx, confidence.
        """
        p = np.array(prices, dtype=np.float64)
        ind = np.array(indicator, dtype=np.float64)

        if len(p) < 10 or len(p) != len(ind):
            return []

        divergences: List[dict] = []
        order = max(3, self._peak_order // 2)

        # Find peaks and troughs in both series
        p_peaks = _find_local_peaks(p, order=order)
        p_troughs = _find_local_troughs(p, order=order)
        i_peaks = _find_local_peaks(ind, order=order)
        i_troughs = _find_local_troughs(ind, order=order)

        # Bearish divergence: price higher highs + indicator lower highs
        if len(p_peaks) >= 2 and len(i_peaks) >= 2:
            for j in range(1, len(p_peaks)):
                pk1, pk2 = p_peaks[j - 1], p_peaks[j]
                if p[pk2] > p[pk1]:  # price higher high
                    # Find closest indicator peaks
                    i_pk1 = self._nearest_idx(i_peaks, pk1)
                    i_pk2 = self._nearest_idx(i_peaks, pk2)
                    if i_pk1 is not None and i_pk2 is not None:
                        if ind[i_pk2] < ind[i_pk1]:  # indicator lower high
                            price_change = (p[pk2] - p[pk1]) / max(abs(p[pk1]), 1e-10)
                            ind_change = (ind[i_pk2] - ind[i_pk1]) / max(abs(ind[i_pk1]), 1e-10)
                            conf = min(1.0, abs(price_change - ind_change) * 5.0)
                            if conf >= 0.2:
                                divergences.append({
                                    "type": "bearish",
                                    "start_idx": int(pk1),
                                    "end_idx": int(pk2),
                                    "confidence": round(conf, 4),
                                })

        # Bullish divergence: price lower lows + indicator higher lows
        if len(p_troughs) >= 2 and len(i_troughs) >= 2:
            for j in range(1, len(p_troughs)):
                tr1, tr2 = p_troughs[j - 1], p_troughs[j]
                if p[tr2] < p[tr1]:  # price lower low
                    i_tr1 = self._nearest_idx(i_troughs, tr1)
                    i_tr2 = self._nearest_idx(i_troughs, tr2)
                    if i_tr1 is not None and i_tr2 is not None:
                        if ind[i_tr2] > ind[i_tr1]:  # indicator higher low
                            price_change = (p[tr2] - p[tr1]) / max(abs(p[tr1]), 1e-10)
                            ind_change = (ind[i_tr2] - ind[i_tr1]) / max(abs(ind[i_tr1]), 1e-10)
                            conf = min(1.0, abs(price_change - ind_change) * 5.0)
                            if conf >= 0.2:
                                divergences.append({
                                    "type": "bullish",
                                    "start_idx": int(tr1),
                                    "end_idx": int(tr2),
                                    "confidence": round(conf, 4),
                                })

        return divergences

    def get_pattern_signal(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
    ) -> dict:
        """
        Aggregate all patterns into a single trading signal.

        Returns dict with:
            bias: float (-1 to +1)
            confidence: float (0–1)
            patterns: list of pattern dicts
            support: float (nearest support level)
            resistance: float (nearest resistance level)
        """
        patterns = self.detect_patterns(prices, volumes)
        sr = self.find_support_resistance(prices)

        if not patterns:
            nearest_support = sr["support"][0] if sr["support"] else 0.0
            nearest_resistance = sr["resistance"][0] if sr["resistance"] else 0.0
            return {
                "bias": 0.0,
                "confidence": 0.0,
                "patterns": [],
                "support": nearest_support,
                "resistance": nearest_resistance,
            }

        # Compute weighted bias from all detected patterns
        total_weight = 0.0
        weighted_bias = 0.0
        for p in patterns:
            conf = p["confidence"]
            if p["direction"] == "BULLISH":
                bias = conf
            elif p["direction"] == "BEARISH":
                bias = -conf
            else:
                bias = 0.0
            weighted_bias += bias * conf
            total_weight += conf

        if total_weight > 0:
            aggregate_bias = weighted_bias / total_weight
        else:
            aggregate_bias = 0.0

        aggregate_bias = max(-1.0, min(1.0, aggregate_bias))
        avg_confidence = total_weight / len(patterns) if patterns else 0.0

        nearest_support = sr["support"][0] if sr["support"] else 0.0
        nearest_resistance = sr["resistance"][0] if sr["resistance"] else 0.0

        return {
            "bias": round(aggregate_bias, 4),
            "confidence": round(avg_confidence, 4),
            "patterns": patterns,
            "support": nearest_support,
            "resistance": nearest_resistance,
        }

    # ------------------------------------------------------------------
    # Pattern detection methods
    # ------------------------------------------------------------------

    def _detect_head_shoulders(
        self, arr: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
    ) -> List[PatternMatch]:
        """Detect Head and Shoulders (bearish reversal)."""
        results = []
        if len(peaks) < 3:
            return results

        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]
            h1, h2, h3 = arr[p1], arr[p2], arr[p3]

            # Head must be highest; shoulders roughly equal
            if h2 > h1 and h2 > h3:
                shoulder_diff = _relative_diff(h1, h3)
                head_prominence = (h2 - max(h1, h3)) / max(h2, 1e-10)

                if shoulder_diff < 0.05 and head_prominence > 0.01:
                    # Find neckline: troughs between peaks
                    neck_troughs = troughs[
                        (troughs > p1) & (troughs < p3)
                    ]
                    if len(neck_troughs) >= 1:
                        neckline = float(np.mean(arr[neck_troughs]))
                        # Target: neckline minus (head - neckline)
                        target = neckline - (h2 - neckline)

                        conf = min(1.0,
                            0.3 * (1.0 - shoulder_diff / 0.05) +
                            0.4 * min(1.0, head_prominence / 0.03) +
                            0.3 * (1.0 if len(neck_troughs) >= 2 else 0.5)
                        )

                        results.append(PatternMatch(
                            pattern="Head and Shoulders",
                            confidence=round(conf, 4),
                            start_idx=int(p1),
                            end_idx=int(p3),
                            direction="BEARISH",
                            target_price=round(target, 2),
                            description="Bearish reversal pattern",
                        ))
        return results

    def _detect_inv_head_shoulders(
        self, arr: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
    ) -> List[PatternMatch]:
        """Detect Inverse Head and Shoulders (bullish reversal)."""
        results = []
        if len(troughs) < 3:
            return results

        for i in range(len(troughs) - 2):
            t1, t2, t3 = troughs[i], troughs[i + 1], troughs[i + 2]
            l1, l2, l3 = arr[t1], arr[t2], arr[t3]

            # Head (t2) must be lowest; shoulders roughly equal
            if l2 < l1 and l2 < l3:
                shoulder_diff = _relative_diff(l1, l3)
                head_prominence = (min(l1, l3) - l2) / max(abs(l2), 1e-10)

                if shoulder_diff < 0.05 and head_prominence > 0.01:
                    neck_peaks = peaks[
                        (peaks > t1) & (peaks < t3)
                    ]
                    if len(neck_peaks) >= 1:
                        neckline = float(np.mean(arr[neck_peaks]))
                        target = neckline + (neckline - l2)

                        conf = min(1.0,
                            0.3 * (1.0 - shoulder_diff / 0.05) +
                            0.4 * min(1.0, head_prominence / 0.03) +
                            0.3 * (1.0 if len(neck_peaks) >= 2 else 0.5)
                        )

                        results.append(PatternMatch(
                            pattern="Inverse Head and Shoulders",
                            confidence=round(conf, 4),
                            start_idx=int(t1),
                            end_idx=int(t3),
                            direction="BULLISH",
                            target_price=round(target, 2),
                            description="Bullish reversal pattern",
                        ))
        return results

    def _detect_double_top(
        self, arr: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
    ) -> List[PatternMatch]:
        """Detect Double Top (bearish)."""
        results = []
        if len(peaks) < 2:
            return results

        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            h1, h2 = arr[p1], arr[p2]

            # Peaks should be roughly equal
            diff = _relative_diff(h1, h2)
            if diff < 0.03:
                # Must have a trough between them
                between = troughs[(troughs > p1) & (troughs < p2)]
                if len(between) >= 1:
                    valley = float(arr[between].min())
                    top = max(h1, h2)
                    drop = (top - valley) / max(top, 1e-10)

                    if drop > 0.01:
                        target = valley - (top - valley)
                        conf = min(1.0,
                            0.4 * (1.0 - diff / 0.03) +
                            0.3 * min(1.0, drop / 0.03) +
                            0.3 * min(1.0, (p2 - p1) / 20.0)
                        )

                        results.append(PatternMatch(
                            pattern="Double Top",
                            confidence=round(conf, 4),
                            start_idx=int(p1),
                            end_idx=int(p2),
                            direction="BEARISH",
                            target_price=round(target, 2),
                            description="Bearish reversal: two peaks at same level",
                        ))
        return results

    def _detect_double_bottom(
        self, arr: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
    ) -> List[PatternMatch]:
        """Detect Double Bottom (bullish)."""
        results = []
        if len(troughs) < 2:
            return results

        for i in range(len(troughs) - 1):
            t1, t2 = troughs[i], troughs[i + 1]
            l1, l2 = arr[t1], arr[t2]

            diff = _relative_diff(l1, l2)
            if diff < 0.03:
                between = peaks[(peaks > t1) & (peaks < t2)]
                if len(between) >= 1:
                    peak_val = float(arr[between].max())
                    bottom = min(l1, l2)
                    rise = (peak_val - bottom) / max(abs(bottom), 1e-10)

                    if rise > 0.01:
                        target = peak_val + (peak_val - bottom)
                        conf = min(1.0,
                            0.4 * (1.0 - diff / 0.03) +
                            0.3 * min(1.0, rise / 0.03) +
                            0.3 * min(1.0, (t2 - t1) / 20.0)
                        )

                        results.append(PatternMatch(
                            pattern="Double Bottom",
                            confidence=round(conf, 4),
                            start_idx=int(t1),
                            end_idx=int(t2),
                            direction="BULLISH",
                            target_price=round(target, 2),
                            description="Bullish reversal: two troughs at same level",
                        ))
        return results

    def _detect_triangles(
        self, arr: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
    ) -> List[PatternMatch]:
        """Detect ascending, descending, and symmetric triangles."""
        results = []
        if len(peaks) < 3 or len(troughs) < 3:
            return results

        # Use the last N peaks/troughs
        n_points = min(5, len(peaks), len(troughs))
        recent_peaks = peaks[-n_points:]
        recent_troughs = troughs[-n_points:]

        peak_vals = arr[recent_peaks]
        trough_vals = arr[recent_troughs]

        # Fit trendlines
        peak_slope, peak_int = _linear_regression(
            recent_peaks.astype(float), peak_vals
        )
        trough_slope, trough_int = _linear_regression(
            recent_troughs.astype(float), trough_vals
        )

        peak_r2 = _r_squared(
            recent_peaks.astype(float), peak_vals, peak_slope, peak_int
        )
        trough_r2 = _r_squared(
            recent_troughs.astype(float), trough_vals, trough_slope, trough_int
        )

        # Converging lines
        converging = (peak_slope < 0 and trough_slope > 0) or \
                     (abs(peak_slope) < abs(trough_slope * 0.3)) or \
                     (abs(trough_slope) < abs(peak_slope * 0.3))

        avg_r2 = (peak_r2 + trough_r2) / 2.0
        if avg_r2 < 0.3:
            return results  # poor trendline fit

        start_idx = int(min(recent_peaks[0], recent_troughs[0]))
        end_idx = int(max(recent_peaks[-1], recent_troughs[-1]))
        current = arr[-1]

        # Ascending triangle: flat resistance, rising support
        if abs(peak_slope) < 0.001 * current and trough_slope > 0.0005 * current:
            resistance = float(np.mean(peak_vals))
            target = resistance + (resistance - float(trough_vals[-1]))
            conf = min(1.0, 0.5 * avg_r2 + 0.5 * min(1.0, n_points / 4))
            results.append(PatternMatch(
                pattern="Ascending Triangle",
                confidence=round(conf, 4),
                start_idx=start_idx,
                end_idx=end_idx,
                direction="BULLISH",
                target_price=round(target, 2),
                description="Flat resistance + rising support",
            ))

        # Descending triangle: falling resistance, flat support
        elif peak_slope < -0.0005 * current and abs(trough_slope) < 0.001 * current:
            support = float(np.mean(trough_vals))
            target = support - (float(peak_vals[-1]) - support)
            conf = min(1.0, 0.5 * avg_r2 + 0.5 * min(1.0, n_points / 4))
            results.append(PatternMatch(
                pattern="Descending Triangle",
                confidence=round(conf, 4),
                start_idx=start_idx,
                end_idx=end_idx,
                direction="BEARISH",
                target_price=round(target, 2),
                description="Falling resistance + flat support",
            ))

        # Symmetric triangle: converging trendlines
        elif peak_slope < 0 and trough_slope > 0:
            # Direction depends on prior trend
            mid_idx = len(arr) // 2
            prior_trend = arr[mid_idx] - arr[0]
            direction = "BULLISH" if prior_trend > 0 else "BEARISH"
            height = float(peak_vals[0] - trough_vals[0])
            apex = (peak_int - trough_int) / max(abs(trough_slope - peak_slope), 1e-10)
            target = current + height if direction == "BULLISH" else current - height

            conf = min(1.0, 0.5 * avg_r2 + 0.3 * min(1.0, n_points / 4) + 0.2)
            results.append(PatternMatch(
                pattern="Symmetric Triangle",
                confidence=round(conf, 4),
                start_idx=start_idx,
                end_idx=end_idx,
                direction=direction,
                target_price=round(target, 2),
                description="Converging trendlines — breakout direction uncertain",
            ))

        return results

    def _detect_flags(
        self,
        arr: np.ndarray,
        peaks: np.ndarray,
        troughs: np.ndarray,
        volumes: Optional[np.ndarray],
    ) -> List[PatternMatch]:
        """Detect Flag/Pennant continuation patterns."""
        results = []
        n = len(arr)
        if n < 20:
            return results

        # Look for a strong impulse followed by a gentle counter-trend channel
        # Check last 30% of data for the flag part
        flag_start = int(n * 0.7)
        pole_section = arr[:flag_start]
        flag_section = arr[flag_start:]

        if len(flag_section) < 5:
            return results

        # Compute pole magnitude
        pole_move = pole_section[-1] - pole_section[0]
        pole_pct = pole_move / max(abs(pole_section[0]), 1e-10)

        # Flag slope (should be counter to pole)
        x = np.arange(len(flag_section), dtype=float)
        flag_slope, flag_int = _linear_regression(x, flag_section)

        # Normalise slope
        flag_slope_pct = flag_slope / max(abs(arr.mean()), 1e-10)

        # Bullish flag: strong up-pole, gentle down-slope flag
        if pole_pct > 0.03 and flag_slope_pct < -0.0001 and flag_slope_pct > -0.01:
            # Decreasing volume during flag is ideal
            vol_conf = 0.5
            if volumes is not None:
                pole_vol = np.mean(volumes[:flag_start])
                flag_vol = np.mean(volumes[flag_start:])
                if flag_vol < pole_vol:
                    vol_conf = 0.8

            target = arr[-1] + abs(pole_move)
            conf = min(1.0, 0.4 * min(1.0, abs(pole_pct) / 0.05) + 0.3 * vol_conf + 0.3)

            results.append(PatternMatch(
                pattern="Bull Flag",
                confidence=round(conf, 4),
                start_idx=0,
                end_idx=n - 1,
                direction="BULLISH",
                target_price=round(target, 2),
                description="Strong uptrend + gentle pullback channel",
            ))

        # Bearish flag: strong down-pole, gentle up-slope flag
        elif pole_pct < -0.03 and flag_slope_pct > 0.0001 and flag_slope_pct < 0.01:
            vol_conf = 0.5
            if volumes is not None:
                pole_vol = np.mean(volumes[:flag_start])
                flag_vol = np.mean(volumes[flag_start:])
                if flag_vol < pole_vol:
                    vol_conf = 0.8

            target = arr[-1] - abs(pole_move)
            conf = min(1.0, 0.4 * min(1.0, abs(pole_pct) / 0.05) + 0.3 * vol_conf + 0.3)

            results.append(PatternMatch(
                pattern="Bear Flag",
                confidence=round(conf, 4),
                start_idx=0,
                end_idx=n - 1,
                direction="BEARISH",
                target_price=round(target, 2),
                description="Strong downtrend + gentle bounce channel",
            ))

        return results

    def _detect_cup_handle(
        self, arr: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
    ) -> List[PatternMatch]:
        """Detect Cup and Handle (bullish)."""
        results = []
        n = len(arr)
        if n < 30:
            return results

        # Look for U-shaped curve in first 70%, small dip in last 30%
        cup_end = int(n * 0.75)
        cup = arr[:cup_end]
        handle = arr[cup_end:]

        if len(handle) < 3:
            return results

        # Cup should have a rounded bottom
        cup_start_val = cup[0]
        cup_end_val = cup[-1]
        cup_min_val = cup.min()
        cup_min_idx = int(np.argmin(cup))

        # Rim should be roughly equal
        rim_diff = _relative_diff(cup_start_val, cup_end_val)
        if rim_diff > 0.05:
            return results

        # Cup depth
        rim = max(cup_start_val, cup_end_val)
        depth = (rim - cup_min_val) / max(rim, 1e-10)
        if depth < 0.02 or depth > 0.5:
            return results

        # Bottom should be in middle third of cup
        relative_min = cup_min_idx / max(len(cup) - 1, 1)
        if relative_min < 0.25 or relative_min > 0.75:
            return results

        # Handle: small pullback (shallower than cup)
        handle_min = handle.min()
        handle_max = handle.max()
        handle_depth = (cup_end_val - handle_min) / max(cup_end_val, 1e-10)

        if handle_depth > depth * 0.5:
            return results  # handle too deep

        # Target: rim + cup depth
        target = rim + (rim - cup_min_val)

        # Confidence based on symmetry and depth
        mid_deviation = abs(relative_min - 0.5) / 0.25
        conf = min(1.0,
            0.3 * (1.0 - rim_diff / 0.05) +
            0.3 * min(1.0, depth / 0.1) +
            0.2 * (1.0 - mid_deviation) +
            0.2 * (1.0 - handle_depth / max(depth * 0.5, 1e-10))
        )

        if conf >= self._min_confidence:
            results.append(PatternMatch(
                pattern="Cup and Handle",
                confidence=round(max(0, conf), 4),
                start_idx=0,
                end_idx=n - 1,
                direction="BULLISH",
                target_price=round(target, 2),
                description="Rounded bottom with small handle pullback",
            ))

        return results

    def _detect_wedges(
        self, arr: np.ndarray, peaks: np.ndarray, troughs: np.ndarray,
    ) -> List[PatternMatch]:
        """Detect rising wedge (bearish) and falling wedge (bullish)."""
        results = []
        if len(peaks) < 3 or len(troughs) < 3:
            return results

        n_points = min(5, len(peaks), len(troughs))
        recent_peaks = peaks[-n_points:]
        recent_troughs = troughs[-n_points:]

        peak_vals = arr[recent_peaks]
        trough_vals = arr[recent_troughs]

        peak_slope, peak_int = _linear_regression(
            recent_peaks.astype(float), peak_vals
        )
        trough_slope, trough_int = _linear_regression(
            recent_troughs.astype(float), trough_vals
        )

        peak_r2 = _r_squared(
            recent_peaks.astype(float), peak_vals, peak_slope, peak_int
        )
        trough_r2 = _r_squared(
            recent_troughs.astype(float), trough_vals, trough_slope, trough_int
        )

        avg_r2 = (peak_r2 + trough_r2) / 2.0
        if avg_r2 < 0.3:
            return results

        start_idx = int(min(recent_peaks[0], recent_troughs[0]))
        end_idx = int(max(recent_peaks[-1], recent_troughs[-1]))
        current = arr[-1]

        # Rising wedge: both slopes positive, but converging
        if peak_slope > 0 and trough_slope > 0 and trough_slope > peak_slope:
            height = float(peak_vals[-1] - trough_vals[-1])
            target = current - height
            conf = min(1.0, 0.4 * avg_r2 + 0.3 * min(1.0, n_points / 4) + 0.3)
            results.append(PatternMatch(
                pattern="Rising Wedge",
                confidence=round(conf, 4),
                start_idx=start_idx,
                end_idx=end_idx,
                direction="BEARISH",
                target_price=round(target, 2),
                description="Both trendlines rising but converging — bearish",
            ))

        # Falling wedge: both slopes negative, but converging
        elif peak_slope < 0 and trough_slope < 0 and peak_slope > trough_slope:
            height = float(peak_vals[-1] - trough_vals[-1])
            target = current + height
            conf = min(1.0, 0.4 * avg_r2 + 0.3 * min(1.0, n_points / 4) + 0.3)
            results.append(PatternMatch(
                pattern="Falling Wedge",
                confidence=round(conf, 4),
                start_idx=start_idx,
                end_idx=end_idx,
                direction="BULLISH",
                target_price=round(target, 2),
                description="Both trendlines falling but converging — bullish",
            ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cluster_levels(
        self,
        levels: List[float],
        tolerance: float = 0.02,
    ) -> List[Tuple[float, int]]:
        """Cluster nearby price levels. Returns (avg_price, count) pairs."""
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters: List[Tuple[float, int]] = []
        current_cluster: List[float] = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            if _relative_diff(sorted_levels[i], current_cluster[-1]) < tolerance:
                current_cluster.append(sorted_levels[i])
            else:
                clusters.append((
                    float(np.mean(current_cluster)),
                    len(current_cluster),
                ))
                current_cluster = [sorted_levels[i]]

        clusters.append((
            float(np.mean(current_cluster)),
            len(current_cluster),
        ))

        # Sort by count descending
        clusters.sort(key=lambda x: x[1], reverse=True)
        return clusters

    @staticmethod
    def _nearest_idx(indices: np.ndarray, target: int) -> Optional[int]:
        """Find the index in `indices` closest to `target`."""
        if len(indices) == 0:
            return None
        diffs = np.abs(indices - target)
        best = np.argmin(diffs)
        if diffs[best] > 10:  # max allowed distance
            return None
        return int(indices[best])

    @staticmethod
    def _to_dicts(patterns: List[PatternMatch]) -> List[dict]:
        """Convert PatternMatch list to list of dicts."""
        return [
            {
                "pattern": p.pattern,
                "confidence": p.confidence,
                "start_idx": p.start_idx,
                "end_idx": p.end_idx,
                "direction": p.direction,
                "target_price": p.target_price,
            }
            for p in patterns
        ]
