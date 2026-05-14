"""
Cross-Venue Price Validation — detects stale or corrupt market data
by comparing prices across multiple venues.

If one venue's price diverges significantly from others, it's flagged
as potentially bad data and trading on that venue is blocked.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VenuePrice:
    """Price snapshot from a single venue."""
    venue: str
    symbol: str
    bid: float
    ask: float
    mid: float
    timestamp: float


@dataclass(frozen=True)
class PriceValidationResult:
    """Result of cross-venue price validation."""
    symbol: str
    valid: bool
    divergent_venue: Optional[str]
    max_divergence_pct: float
    venue_count: int
    consensus_mid: float
    reason: str
    timestamp: float = field(default_factory=time.time)


class CrossVenueValidator:
    """
    Validates market data by comparing prices across venues.

    If a venue's mid price diverges more than `max_divergence_pct` from
    the consensus (median of all venues), it's flagged as bad data.

    Usage:
        validator = CrossVenueValidator()
        validator.update("kraken", "BTC/USD", bid=50000, ask=50010)
        validator.update("coinbase", "BTC/USD", bid=50005, ask=50015)
        result = validator.validate("BTC/USD")
        if not result.valid:
            # Don't trade on result.divergent_venue
    """

    def __init__(
        self,
        max_divergence_pct: float = 1.0,
        stale_threshold_seconds: float = 60.0,
        min_venues: int = 2,
    ):
        self._max_div = max_divergence_pct
        self._stale_threshold = stale_threshold_seconds
        self._min_venues = min_venues
        self._prices: Dict[str, Dict[str, VenuePrice]] = {}  # symbol → venue → price
        self._alerts: List[PriceValidationResult] = []

    def update(self, venue: str, symbol: str, bid: float, ask: float) -> None:
        """Update price for a venue/symbol pair."""
        if symbol not in self._prices:
            self._prices[symbol] = {}
        mid = (bid + ask) / 2.0 if (bid + ask) > 0 else 0.0
        self._prices[symbol][venue] = VenuePrice(
            venue=venue, symbol=symbol, bid=bid, ask=ask,
            mid=mid, timestamp=time.time(),
        )

    def validate(self, symbol: str) -> PriceValidationResult:
        """Validate prices for a symbol across all venues."""
        venue_prices = self._prices.get(symbol, {})
        now = time.time()

        # Filter stale prices
        fresh = {
            v: p for v, p in venue_prices.items()
            if (now - p.timestamp) < self._stale_threshold and p.mid > 0
        }

        if len(fresh) < self._min_venues:
            return PriceValidationResult(
                symbol=symbol, valid=True, divergent_venue=None,
                max_divergence_pct=0.0, venue_count=len(fresh),
                consensus_mid=0.0,
                reason=f"insufficient venues ({len(fresh)}<{self._min_venues})",
            )

        # Compute consensus mid (median)
        mids = sorted(p.mid for p in fresh.values())
        if len(mids) % 2 == 0:
            consensus = (mids[len(mids) // 2 - 1] + mids[len(mids) // 2]) / 2
        else:
            consensus = mids[len(mids) // 2]

        if consensus <= 0:
            return PriceValidationResult(
                symbol=symbol, valid=True, divergent_venue=None,
                max_divergence_pct=0.0, venue_count=len(fresh),
                consensus_mid=0.0, reason="zero consensus price",
            )

        # Check each venue for divergence
        worst_venue = None
        worst_div = 0.0
        for venue, price in fresh.items():
            div_pct = abs(price.mid - consensus) / consensus * 100
            if div_pct > worst_div:
                worst_div = div_pct
                worst_venue = venue

        if worst_div > self._max_div:
            result = PriceValidationResult(
                symbol=symbol, valid=False, divergent_venue=worst_venue,
                max_divergence_pct=worst_div, venue_count=len(fresh),
                consensus_mid=consensus,
                reason=f"{worst_venue} diverged {worst_div:.2f}% from consensus",
            )
            self._alerts.append(result)
            logger.warning(
                "PRICE DIVERGENCE: %s on %s diverged %.2f%% from consensus $%.2f",
                symbol, worst_venue, worst_div, consensus,
            )
            return result

        return PriceValidationResult(
            symbol=symbol, valid=True, divergent_venue=None,
            max_divergence_pct=worst_div, venue_count=len(fresh),
            consensus_mid=consensus, reason="all venues within tolerance",
        )

    def get_alerts(self, last_n: int = 50) -> List[PriceValidationResult]:
        return self._alerts[-last_n:]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbols_tracked": len(self._prices),
            "total_alerts": len(self._alerts),
            "max_divergence_pct": self._max_div,
            "stale_threshold_s": self._stale_threshold,
        }
