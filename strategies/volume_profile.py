"""
Volume Profile Analysis — price-level volume distribution for support/resistance.

Volume Profile aggregates traded volume at each price level to identify:
- **Point of Control (POC)**: price level with the highest traded volume
- **Value Area**: price range containing a specified percentage (default 70%)
  of total volume — bounded by Value Area High (VAH) and Value Area Low (VAL)

Trading logic:
- BUY when price is near VAL (support) → price tends to revert to POC
- SELL when price is near VAH (resistance) → price tends to revert to POC
- Strong breakouts above VAH or below VAL may signal trend continuation

This is a pure statistical approach; no curve fitting or machine learning.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_LOOKBACK = 500  # bars
_DEFAULT_BIN_SIZE_PCT = 0.1  # 0.1% price bins
_DEFAULT_VALUE_AREA_PCT = 0.70
_PROXIMITY_PCT = 0.005  # 0.5% proximity to VA edges triggers signal


@dataclass
class VolumeProfileSignal:
    """Signal emitted by the Volume Profile analyzer."""

    symbol: str
    direction: str  # "buy" or "sell"
    confidence: float  # 0.0 – 1.0
    current_price: float
    poc: float
    vah: float
    val: float
    distance_to_edge_pct: float  # distance to nearest VA edge as %
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _PriceVolume:
    """Single price-volume observation."""

    price: float
    volume: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VolumeProfileAnalyzer:
    """
    Volume Profile analysis engine.

    Maintains a rolling window of price-volume data per symbol, bins by
    price level, and computes POC / Value Area for trade signals.

    Parameters
    ----------
    lookback : int
        Maximum number of observations to retain per symbol.
    bin_size_pct : float
        Width of each price bin as a percentage of the reference price.
    default_value_area_pct : float
        Default percentage of total volume for Value Area calculation.
    proximity_pct : float
        How close price must be to VAH/VAL to trigger a signal (as %).
    """

    def __init__(
        self,
        lookback: int = _DEFAULT_LOOKBACK,
        bin_size_pct: float = _DEFAULT_BIN_SIZE_PCT,
        default_value_area_pct: float = _DEFAULT_VALUE_AREA_PCT,
        proximity_pct: float = _PROXIMITY_PCT,
    ) -> None:
        self.lookback = max(lookback, 50)
        self.bin_size_pct = max(bin_size_pct, 0.01)
        self.default_value_area_pct = max(0.5, min(0.99, default_value_area_pct))
        self.proximity_pct = max(proximity_pct, 0.001)

        # Per-symbol rolling observation buffer
        self._data: Dict[str, Deque[_PriceVolume]] = {}

        logger.info(
            "VolumeProfileAnalyzer initialised (lookback=%d, bin=%.3f%%, "
            "VA=%.0f%%, proximity=%.3f%%)",
            self.lookback,
            self.bin_size_pct * 100,
            self.default_value_area_pct * 100,
            self.proximity_pct * 100,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, price: float, volume: float) -> None:
        """
        Record a price-volume observation for *symbol*.

        Parameters
        ----------
        symbol : str
            Trading pair (e.g. ``"BTC/USD"``).
        price : float
            Trade price or candle close.
        volume : float
            Trade volume (base asset) or candle volume.
        """
        if price <= 0 or volume <= 0:
            return

        if symbol not in self._data:
            self._data[symbol] = deque(maxlen=self.lookback)

        self._data[symbol].append(_PriceVolume(price=price, volume=volume))

    def get_poc(self, symbol: str) -> Optional[float]:
        """
        Return the Point of Control for *symbol*.

        The POC is the mid-price of the bin with the highest accumulated
        volume.  Returns ``None`` if insufficient data.
        """
        profile = self._build_profile(symbol)
        if not profile:
            return None

        # Find bin with max volume
        max_bin = max(profile, key=lambda x: x[1])
        return max_bin[0]

    def get_value_area(
        self,
        symbol: str,
        pct: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Return the Value Area as ``(vah, val)``.

        Parameters
        ----------
        pct : float or None
            Fraction of total volume to include (default: ``default_value_area_pct``).

        Returns
        -------
        tuple[float, float] or None
            ``(value_area_high, value_area_low)`` or None if insufficient data.
        """
        if pct is None:
            pct = self.default_value_area_pct
        pct = max(0.1, min(0.99, pct))

        profile = self._build_profile(symbol)
        if not profile:
            return None

        total_volume = sum(v for _, v in profile)
        if total_volume <= 0:
            return None

        target_volume = total_volume * pct

        # Sort by volume descending; greedily add bins until target reached
        sorted_bins = sorted(profile, key=lambda x: x[1], reverse=True)
        accumulated = 0.0
        included_prices: List[float] = []

        for bin_price, bin_vol in sorted_bins:
            accumulated += bin_vol
            included_prices.append(bin_price)
            if accumulated >= target_volume:
                break

        if not included_prices:
            return None

        vah = max(included_prices)
        val = min(included_prices)
        return round(vah, 8), round(val, 8)

    def get_signal(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[VolumeProfileSignal]:
        """
        Generate a volume-profile-based signal for *symbol*.

        - BUY when price is near VAL (expected bounce toward POC)
        - SELL when price is near VAH (expected rejection toward POC)
        - No signal in the middle of the value area

        Returns ``None`` if insufficient data or price is not near edges.
        """
        poc = self.get_poc(symbol)
        va = self.get_value_area(symbol)
        if poc is None or va is None:
            return None

        vah, val = va
        if vah <= val or current_price <= 0:
            return None

        # Distance from VA edges
        dist_to_val = abs(current_price - val) / current_price
        dist_to_vah = abs(current_price - vah) / current_price

        direction: Optional[str] = None
        distance_pct = 0.0

        if dist_to_val <= self.proximity_pct and current_price <= val * (1 + self.proximity_pct):
            direction = "buy"
            distance_pct = dist_to_val
        elif dist_to_vah <= self.proximity_pct and current_price >= vah * (1 - self.proximity_pct):
            direction = "sell"
            distance_pct = dist_to_vah

        if direction is None:
            return None

        # Confidence: closer to edge = higher confidence, also scale by VA width
        va_width = (vah - val) / poc if poc > 0 else 0
        proximity_score = max(0.0, 1.0 - distance_pct / self.proximity_pct)
        # Wider VA = stronger levels
        width_score = min(va_width * 20, 0.5)
        confidence = min(proximity_score * 0.6 + width_score, 1.0)

        signal = VolumeProfileSignal(
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 4),
            current_price=current_price,
            poc=poc,
            vah=vah,
            val=val,
            distance_to_edge_pct=round(distance_pct, 6),
        )
        logger.info(
            "VolumeProfile signal: %s %s @ %.2f (poc=%.2f vah=%.2f val=%.2f conf=%.3f)",
            symbol, direction, current_price, poc, vah, val, confidence,
        )
        return signal

    def get_profile(self, symbol: str) -> List[Tuple[float, float]]:
        """
        Return the full volume profile as a list of ``(price, volume)`` tuples.

        Sorted by price ascending.
        """
        profile = self._build_profile(symbol)
        return sorted(profile, key=lambda x: x[0])

    def get_observation_count(self, symbol: str) -> int:
        """Return the number of price-volume observations stored for *symbol*."""
        return len(self._data.get(symbol, []))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_profile(self, symbol: str) -> List[Tuple[float, float]]:
        """
        Bin price-volume data into a volume profile.

        Returns a list of ``(bin_mid_price, total_volume)`` tuples.
        """
        data = self._data.get(symbol)
        if not data or len(data) < 10:
            return []

        observations = list(data)
        prices = [obs.price for obs in observations]
        min_price = min(prices)
        max_price = max(prices)

        if min_price <= 0 or max_price <= 0:
            return []

        # Calculate bin width from percentage of midpoint
        mid = (min_price + max_price) / 2.0
        bin_width = mid * (self.bin_size_pct / 100.0)
        if bin_width <= 0:
            bin_width = (max_price - min_price) / 50.0
        if bin_width <= 0:
            return []

        # Accumulate volume per bin
        bins: Dict[int, float] = defaultdict(float)
        for obs in observations:
            bin_idx = int((obs.price - min_price) / bin_width)
            bins[bin_idx] += obs.volume

        # Convert to (mid_price, volume) list
        profile: List[Tuple[float, float]] = []
        for bin_idx, vol in bins.items():
            bin_mid = min_price + (bin_idx + 0.5) * bin_width
            profile.append((round(bin_mid, 8), round(vol, 8)))

        return profile
