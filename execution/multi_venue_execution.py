"""
Multi-venue execution: single decision, split across venues by spread/liquidity.

Routing logic:
  1. Score each venue by: spread (lower = better), liquidity (higher = better),
     latency (lower = better), fee tier.
  2. Allocate size proportionally to score, capped at max_venues.
  3. Below min_notional_usd threshold → route 100% to primary.

Live spread/liquidity is injected via update_venue_stats(); falls back to
static weights when not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum notional below which multi-venue split is not worth the overhead
_MIN_NOTIONAL_SPLIT_USD = 200.0

# Default venue parameters (overridden live via update_venue_stats)
_DEFAULT_VENUES: Dict[str, Dict[str, float]] = {
    "kraken": {
        "spread_bps": 5.0,
        "liquidity_score": 0.90,
        "latency_ms": 80.0,
        "taker_fee_bps": 26.0,   # 0.26%
    },
    "coinbase": {
        "spread_bps": 6.0,
        "liquidity_score": 0.85,
        "latency_ms": 60.0,
        "taker_fee_bps": 60.0,   # 0.60% advanced, lower with volume
    },
}


_DEFAULT_WEIGHTS: Dict[str, float] = {
    "spread": 0.40,
    "liquidity": 0.35,
    "latency": 0.15,
    "fee": 0.10,
}


@dataclass
class VenueStats:
    spread_bps: float = 5.0
    liquidity_score: float = 0.5    # 0-1
    latency_ms: float = 100.0
    taker_fee_bps: float = 30.0
    _weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))

    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Higher is better venue for routing."""
        w = weights if weights is not None else self._weights
        spread_score = 1.0 / max(self.spread_bps, 0.1)
        liquidity_score = self.liquidity_score
        latency_score = 1.0 / (1.0 + self.latency_ms / 100.0)
        fee_score = 1.0 / max(self.taker_fee_bps, 1.0)
        return (
            w.get("spread", 0.40) * spread_score
            + w.get("liquidity", 0.35) * liquidity_score
            + w.get("latency", 0.15) * latency_score
            + w.get("fee", 0.10) * fee_score
        )


@dataclass
class VenueOrder:
    venue: str
    symbol: str
    side: str
    size: float
    order_id: Optional[str] = None
    filled: float = 0.0
    fill_price: float = 0.0
    status: str = "pending"


@dataclass
class MultiVenueDecision:
    symbol: str
    side: str
    total_size: float
    urgency: str = "normal"     # low | normal | high
    max_venues: int = 2         # default: Kraken + Coinbase only


class MultiVenueExecutor:
    """
    Split a single decision across venues by real-time spread and liquidity.

    Usage::

        executor = MultiVenueExecutor(primary_venue="kraken")
        # update with live market data each cycle:
        executor.update_venue_stats("kraken", spread_bps=4.5, liquidity_score=0.92)
        executor.update_venue_stats("coinbase", spread_bps=7.1, liquidity_score=0.80)
        # split an order:
        orders = executor.split(MultiVenueDecision("BTC/USD", "BUY", 500.0))
    """

    def __init__(
        self,
        primary_venue: str = "kraken",
        min_notional_usd: float = _MIN_NOTIONAL_SPLIT_USD,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.primary_venue = primary_venue
        self._min_notional = min_notional_usd
        self._weights: Dict[str, float] = dict(weights) if weights else dict(_DEFAULT_WEIGHTS)
        # Initialise with exchange defaults
        self._stats: Dict[str, VenueStats] = {
            venue: VenueStats(**{k: v for k, v in params.items()}, _weights=self._weights)
            for venue, params in _DEFAULT_VENUES.items()
        }

    def update_venue_stats(
        self,
        venue: str,
        *,
        spread_bps: Optional[float] = None,
        liquidity_score: Optional[float] = None,
        latency_ms: Optional[float] = None,
        taker_fee_bps: Optional[float] = None,
    ) -> None:
        """Inject live market data for a venue."""
        if venue not in self._stats:
            self._stats[venue] = VenueStats(_weights=self._weights)
        s = self._stats[venue]
        if spread_bps is not None:
            s.spread_bps = max(0.0, float(spread_bps))
        if liquidity_score is not None:
            s.liquidity_score = max(0.0, min(1.0, float(liquidity_score)))
        if latency_ms is not None:
            s.latency_ms = max(0.0, float(latency_ms))
        if taker_fee_bps is not None:
            s.taker_fee_bps = max(0.0, float(taker_fee_bps))

    def _ranked_venues(self, symbol: str, max_venues: int) -> List[Tuple[str, float]]:
        """Return venues sorted by composite score, best first."""
        ranked = sorted(
            ((v, s.score(self._weights)) for v, s in self._stats.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:max_venues]

    def split(self, decision: MultiVenueDecision) -> List[VenueOrder]:
        """
        Split total_size across venues proportionally to their scores.

        Below min_notional_usd → 100% to primary (avoids fragmentation).
        With only one venue configured → 100% to that venue.
        """
        size = abs(float(decision.total_size))

        # Below threshold: single venue
        if size < self._min_notional or len(self._stats) <= 1:
            return [VenueOrder(
                venue=self.primary_venue,
                symbol=decision.symbol,
                side=decision.side,
                size=size,
            )]

        ranked = self._ranked_venues(decision.symbol, decision.max_venues)
        if not ranked:
            return [VenueOrder(
                venue=self.primary_venue,
                symbol=decision.symbol,
                side=decision.side,
                size=size,
            )]

        # Proportional allocation by score
        total_score = sum(sc for _, sc in ranked)
        if total_score <= 0:
            # Uniform split
            per = size / len(ranked)
            return [
                VenueOrder(venue=v, symbol=decision.symbol, side=decision.side, size=per)
                for v, _ in ranked
            ]

        orders: List[VenueOrder] = []
        allocated = 0.0
        for i, (venue, sc) in enumerate(ranked):
            if i == len(ranked) - 1:
                # Last venue gets remainder to avoid floating-point leakage
                chunk = size - allocated
            else:
                chunk = round(size * sc / total_score, 8)
            allocated += chunk
            if chunk > 0:
                orders.append(VenueOrder(
                    venue=venue,
                    symbol=decision.symbol,
                    side=decision.side,
                    size=chunk,
                ))

        logger.debug(
            "Multi-venue split %s %s %.4f → %s",
            decision.side, decision.symbol, size,
            [(o.venue, round(o.size, 4)) for o in orders],
        )
        return orders

    def aggregate_fills(self, orders: List[VenueOrder]) -> Dict[str, Any]:
        """Aggregate fills across venues into a single summary."""
        total_filled = sum(o.filled for o in orders)
        total_notional = sum(o.filled * o.fill_price for o in orders)
        vwap = total_notional / max(total_filled, 1e-10)
        return {
            "total_filled": total_filled,
            "total_notional": total_notional,
            "vwap": vwap,
            "venues": [o.venue for o in orders],
            "venue_fills": {o.venue: o.filled for o in orders},
        }

    def venue_scores(self) -> Dict[str, float]:
        return {v: s.score(self._weights) for v, s in self._stats.items()}
