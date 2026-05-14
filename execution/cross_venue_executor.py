"""
Cross-Venue Atomic Execution — Order Splitting Across Exchanges.

Pure-computation module that optimises how a single order should be split
across multiple venues to minimise execution cost.  It does **not** send
real orders; downstream executors consume the ``VenueOrder`` list.

Usage::

    executor = CrossVenueExecutor()
    orders = executor.split_order("BTC/AUD", size=1.5, venues=["kraken", "coinbase"],
                                  venue_prices={"kraken": 98_450, "coinbase": 98_480})
    best = executor.get_best_split("BTC/AUD", 1.5, venue_books)
    savings = executor.estimate_savings_bps(naive, best)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VenueOrder:
    """A child order destined for a single venue."""

    venue: str
    symbol: str
    size: float
    expected_price: float
    side: str                     # "buy" or "sell"
    fee_bps: float = 0.0         # expected fee in basis points


@dataclass
class OrderBookLevel:
    """A single price level from a venue order book."""

    price: float
    size: float


@dataclass
class VenueBook:
    """Simplified order-book snapshot for one venue."""

    venue: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    fee_bps: float = 0.0         # taker fee in bps for this venue


class CrossVenueExecutor:
    """Optimise order splitting across multiple trading venues.

    The optimiser walks aggregated order-book depth across venues and
    allocates size to the cheapest available liquidity first (greedy
    price-priority algorithm).

    Parameters
    ----------
    default_fee_bps : float
        Assumed taker fee if not provided per venue.
    min_split_size : float
        Minimum order size per venue (below this, consolidate to one venue).
    """

    def __init__(
        self,
        default_fee_bps: float = 10.0,
        min_split_size: float = 0.001,
    ) -> None:
        self._default_fee_bps = default_fee_bps
        self._min_split_size = min_split_size
        logger.info("CrossVenueExecutor initialised — default_fee=%.1f bps min_split=%.4f",
                     default_fee_bps, min_split_size)

    # ------------------------------------------------------------------
    # Simple price-based split (no orderbook)
    # ------------------------------------------------------------------

    def split_order(
        self,
        symbol: str,
        size: float,
        venues: List[str],
        venue_prices: Dict[str, float],
        side: str = "buy",
        venue_fees: Optional[Dict[str, float]] = None,
    ) -> List[VenueOrder]:
        """Split an order across *venues* based on price alone.

        Allocates more size to venues with better prices.  For buys,
        lower price is better; for sells, higher price is better.

        Parameters
        ----------
        symbol : str
            Trading pair.
        size : float
            Total base-currency quantity.
        venues : list[str]
            Available venue names.
        venue_prices : dict[str, float]
            Best price per venue.
        side : str
            ``"buy"`` or ``"sell"``.
        venue_fees : dict[str, float] | None
            Per-venue fee in basis points.  Defaults to ``default_fee_bps``.

        Returns
        -------
        list[VenueOrder]
        """
        if not venues or size <= 0:
            return []

        fees = venue_fees or {}

        # Effective price includes fee impact.
        def effective(v: str) -> float:
            p = venue_prices.get(v, float("inf"))
            fee = fees.get(v, self._default_fee_bps) / 10_000.0
            if side == "buy":
                return p * (1 + fee)
            return p * (1 - fee)

        # Sort venues: best effective price first.
        ranked = sorted(
            [v for v in venues if v in venue_prices],
            key=effective,
            reverse=(side == "sell"),
        )
        if not ranked:
            return []

        # Weight by inverse distance from worst price.
        eff_prices = {v: effective(v) for v in ranked}
        worst = max(eff_prices.values()) if side == "buy" else min(eff_prices.values())
        weights: Dict[str, float] = {}
        for v in ranked:
            diff = abs(worst - eff_prices[v])
            weights[v] = diff + 1e-9   # small epsilon so worst venue gets tiny allocation

        w_sum = sum(weights.values())
        orders: List[VenueOrder] = []
        allocated = 0.0

        for i, v in enumerate(ranked):
            if i == len(ranked) - 1:
                alloc = size - allocated  # remainder to avoid floating-point drift
            else:
                alloc = size * (weights[v] / w_sum)
            alloc = max(0.0, alloc)
            if alloc < self._min_split_size and len(ranked) > 1:
                continue
            allocated += alloc
            orders.append(VenueOrder(
                venue=v,
                symbol=symbol,
                size=alloc,
                expected_price=venue_prices[v],
                side=side,
                fee_bps=fees.get(v, self._default_fee_bps),
            ))

        # If rounding caused under-allocation, add remainder to best venue.
        if orders and allocated < size:
            orders[0].size += size - allocated

        logger.info("Split order %s %.6f across %d venues (%s)",
                     symbol, size, len(orders), side)
        return orders

    # ------------------------------------------------------------------
    # Order-book-aware split
    # ------------------------------------------------------------------

    def get_best_split(
        self,
        symbol: str,
        size: float,
        venue_books: Dict[str, VenueBook],
        side: str = "buy",
    ) -> List[VenueOrder]:
        """Optimally split *size* across venues using order-book depth.

        Walks aggregated depth across all venues, consuming the cheapest
        liquidity first (price-priority across venues).

        Parameters
        ----------
        symbol : str
            Trading pair.
        size : float
            Total base-currency quantity.
        venue_books : dict[str, VenueBook]
            Order-book snapshots keyed by venue name.
        side : str
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        list[VenueOrder]
        """
        if size <= 0 or not venue_books:
            return []

        # Build merged level list: (effective_price, venue, raw_price, available_size)
        merged: List[Tuple[float, str, float, float]] = []
        for venue, book in venue_books.items():
            levels = book.asks if side == "buy" else book.bids
            fee_mult = 1 + book.fee_bps / 10_000.0 if side == "buy" else 1 - book.fee_bps / 10_000.0
            for lvl in levels:
                eff = lvl.price * fee_mult
                merged.append((eff, venue, lvl.price, lvl.size))

        # Sort: buy → cheapest effective first; sell → most expensive first.
        merged.sort(key=lambda x: x[0], reverse=(side == "sell"))

        venue_alloc: Dict[str, Tuple[float, float]] = {}  # venue -> (total_size, wavg_price_num)
        remaining = size

        for eff, venue, raw_price, avail in merged:
            if remaining <= 0:
                break
            take = min(remaining, avail)
            if venue in venue_alloc:
                prev_size, prev_pv = venue_alloc[venue]
                venue_alloc[venue] = (prev_size + take, prev_pv + take * raw_price)
            else:
                venue_alloc[venue] = (take, take * raw_price)
            remaining -= take

        orders: List[VenueOrder] = []
        for venue, (alloc_size, pv) in venue_alloc.items():
            if alloc_size < self._min_split_size:
                continue
            wavg = pv / alloc_size if alloc_size > 0 else 0.0
            book = venue_books[venue]
            orders.append(VenueOrder(
                venue=venue,
                symbol=symbol,
                size=alloc_size,
                expected_price=wavg,
                side=side,
                fee_bps=book.fee_bps,
            ))

        # If remaining > 0, we exhausted all books.
        if remaining > 0:
            logger.warning("Insufficient liquidity: %.6f of %.6f unfilled across %d venues",
                           remaining, size, len(venue_books))

        orders.sort(key=lambda o: o.size, reverse=True)
        logger.info("Best split %s %.6f: %d venue orders, side=%s",
                     symbol, size, len(orders), side)
        return orders

    # ------------------------------------------------------------------
    # Savings estimation
    # ------------------------------------------------------------------

    def estimate_savings_bps(
        self,
        naive_single_venue: List[VenueOrder],
        optimized_split: List[VenueOrder],
    ) -> float:
        """Estimate cost savings (in basis points) of optimised split vs naive.

        Compares the volume-weighted average effective price (price + fee)
        between the two order lists.

        Parameters
        ----------
        naive_single_venue : list[VenueOrder]
            Baseline: all size on one venue.
        optimized_split : list[VenueOrder]
            Optimised multi-venue split.

        Returns
        -------
        float
            Savings in basis points (positive = optimised is cheaper for buys
            or gets better price for sells).
        """
        def wavg_eff(orders: List[VenueOrder]) -> float:
            total_size = sum(o.size for o in orders)
            if total_size <= 0:
                return 0.0
            pv = sum(o.size * o.expected_price * (1 + o.fee_bps / 10_000.0) for o in orders)
            return pv / total_size

        if not naive_single_venue or not optimized_split:
            return 0.0

        naive_eff = wavg_eff(naive_single_venue)
        opt_eff = wavg_eff(optimized_split)

        if naive_eff <= 0:
            return 0.0

        # For buys: lower effective price is better → positive savings.
        savings = (naive_eff - opt_eff) / naive_eff * 10_000.0
        logger.debug("Savings estimate: naive_eff=%.4f opt_eff=%.4f savings=%.2f bps",
                      naive_eff, opt_eff, savings)
        return savings
