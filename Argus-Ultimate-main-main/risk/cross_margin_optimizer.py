#!/usr/bin/env python3
"""
Cross-Margin Position Optimizer — identifies netting and margin efficiency gains.

When positions are spread across multiple venues (Kraken, Bybit, Coinbase),
margin is posted redundantly.  This module detects offsetting positions that
could be netted and suggests inter-venue capital transfers to free up margin.

Features:
- ``add_position(venue, symbol, side, size, margin_used)`` — register a position
- ``get_netting_opportunities()`` → list of NettingOpportunity
- ``get_total_margin_efficiency()`` → float (0–1)
- ``suggest_transfers(max_transfer_usd=500)`` → list of transfer recommendations

Usage::

    opt = CrossMarginOptimizer()
    opt.add_position("kraken", "BTC/USD", "long", 0.05, 500.0)
    opt.add_position("bybit", "BTC/USD", "short", 0.03, 300.0)
    opps = opt.get_netting_opportunities()
    for opp in opps:
        logger.info(opp.symbol, opp.margin_saved_usd)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A single position on a specific venue."""

    venue: str
    symbol: str
    side: str          # "long" or "short"
    size: float        # in base units
    margin_used: float  # USD posted as margin


@dataclass
class NettingOpportunity:
    """An opportunity to net offsetting positions across venues."""

    venue_a: str
    venue_b: str
    symbol: str
    size_to_net: float       # base units that can be netted
    margin_saved_usd: float  # estimated margin freed


@dataclass
class TransferRecommendation:
    """A suggested inter-venue capital transfer."""

    from_venue: str
    to_venue: str
    amount_usd: float
    reason: str
    margin_freed_usd: float


# ---------------------------------------------------------------------------
# Cross-Margin Optimizer
# ---------------------------------------------------------------------------


class CrossMarginOptimizer:
    """Identifies cross-venue netting opportunities and margin inefficiencies.

    Parameters
    ----------
    margin_rate : float
        Assumed margin rate (fraction of notional).  Default 0.10 (10x leverage).
    """

    def __init__(self, margin_rate: float = 0.10) -> None:
        self._positions: List[Position] = []
        self._margin_rate = margin_rate
        logger.info(
            "CrossMarginOptimizer initialised — margin_rate=%.2f", margin_rate,
        )

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(
        self,
        venue: str,
        symbol: str,
        side: str,
        size: float,
        margin_used: float,
    ) -> None:
        """Register a position for analysis.

        Parameters
        ----------
        venue : str
            Exchange name (e.g. ``"kraken"``).
        symbol : str
            Trading pair (e.g. ``"BTC/USD"``).
        side : str
            ``"long"`` or ``"short"``.
        size : float
            Position size in base units.
        margin_used : float
            USD margin posted for this position.
        """
        pos = Position(
            venue=venue.lower(),
            symbol=symbol.upper(),
            side=side.lower(),
            size=abs(size),
            margin_used=abs(margin_used),
        )
        self._positions.append(pos)
        logger.debug(
            "CrossMarginOptimizer: added %s %s %s %.4f (margin $%.2f)",
            pos.venue, pos.symbol, pos.side, pos.size, pos.margin_used,
        )

    def clear_positions(self) -> None:
        """Clear all registered positions (call at start of each cycle)."""
        self._positions.clear()
        logger.debug("CrossMarginOptimizer: positions cleared")

    # ------------------------------------------------------------------
    # Netting
    # ------------------------------------------------------------------

    def get_netting_opportunities(self) -> List[NettingOpportunity]:
        """Find offsetting positions across different venues.

        For each symbol, if venue A has a long and venue B has a short (or
        vice-versa), the overlapping size can theoretically be netted, freeing
        the margin on both sides for the netted amount.

        Returns
        -------
        list of NettingOpportunity
            Sorted by margin_saved_usd descending.
        """
        # Group positions by symbol
        by_symbol: Dict[str, List[Position]] = {}
        for pos in self._positions:
            by_symbol.setdefault(pos.symbol, []).append(pos)

        opportunities: List[NettingOpportunity] = []

        for symbol, positions in by_symbol.items():
            longs = [p for p in positions if p.side == "long"]
            shorts = [p for p in positions if p.side == "short"]

            # Try all cross-venue long/short pairs
            for long_pos in longs:
                for short_pos in shorts:
                    if long_pos.venue == short_pos.venue:
                        continue  # same venue — exchange handles netting

                    nettable = min(long_pos.size, short_pos.size)
                    if nettable <= 0:
                        continue

                    # Margin saved: both sides free margin proportional to netted size
                    long_margin_per_unit = (
                        long_pos.margin_used / long_pos.size
                        if long_pos.size > 0
                        else 0
                    )
                    short_margin_per_unit = (
                        short_pos.margin_used / short_pos.size
                        if short_pos.size > 0
                        else 0
                    )
                    margin_saved = nettable * (long_margin_per_unit + short_margin_per_unit)

                    opportunities.append(
                        NettingOpportunity(
                            venue_a=long_pos.venue,
                            venue_b=short_pos.venue,
                            symbol=symbol,
                            size_to_net=round(nettable, 8),
                            margin_saved_usd=round(margin_saved, 2),
                        )
                    )

        opportunities.sort(key=lambda o: -o.margin_saved_usd)
        if opportunities:
            logger.info(
                "CrossMarginOptimizer: found %d netting opportunities "
                "(total margin saving $%.2f)",
                len(opportunities),
                sum(o.margin_saved_usd for o in opportunities),
            )
        return opportunities

    # ------------------------------------------------------------------
    # Efficiency metric
    # ------------------------------------------------------------------

    def get_total_margin_efficiency(self) -> float:
        """Calculate overall margin efficiency (0–1, where 1 = perfect).

        Efficiency = theoretical_minimum_margin / actual_total_margin.
        Theoretical minimum assumes perfect netting of all offsetting positions.

        Returns
        -------
        float
            Efficiency ratio.  1.0 means no redundant margin; 0.5 means half
            the margin is redundant.
        """
        if not self._positions:
            return 1.0

        actual_total = sum(p.margin_used for p in self._positions)
        if actual_total <= 0:
            return 1.0

        # Theoretical: net positions per symbol, then sum margin needed
        by_symbol: Dict[str, float] = {}
        by_symbol_margin_rate: Dict[str, float] = {}

        for pos in self._positions:
            sign = 1.0 if pos.side == "long" else -1.0
            by_symbol[pos.symbol] = by_symbol.get(pos.symbol, 0.0) + sign * pos.size
            # Track margin rate per unit across all positions for this symbol
            if pos.size > 0:
                rate = pos.margin_used / pos.size
                existing = by_symbol_margin_rate.get(pos.symbol, rate)
                by_symbol_margin_rate[pos.symbol] = (existing + rate) / 2.0

        theoretical_total = 0.0
        for symbol, net_size in by_symbol.items():
            rate = by_symbol_margin_rate.get(symbol, self._margin_rate)
            theoretical_total += abs(net_size) * rate

        efficiency = min(1.0, theoretical_total / actual_total) if actual_total > 0 else 1.0
        logger.info(
            "CrossMarginOptimizer: margin efficiency %.1f%% "
            "(actual $%.2f, theoretical $%.2f)",
            efficiency * 100, actual_total, theoretical_total,
        )
        return round(efficiency, 4)

    # ------------------------------------------------------------------
    # Transfer suggestions
    # ------------------------------------------------------------------

    def suggest_transfers(
        self, max_transfer_usd: float = 500.0
    ) -> List[TransferRecommendation]:
        """Suggest inter-venue transfers to improve margin utilisation.

        Transfers move excess margin from over-margined venues to
        under-margined venues.

        Parameters
        ----------
        max_transfer_usd : float
            Maximum single transfer amount.

        Returns
        -------
        list of TransferRecommendation
            Sorted by margin_freed_usd descending.
        """
        # Calculate per-venue margin usage and excess
        venue_margin: Dict[str, float] = {}
        venue_notional: Dict[str, float] = {}

        for pos in self._positions:
            venue_margin[pos.venue] = venue_margin.get(pos.venue, 0.0) + pos.margin_used
            # Estimate notional using margin_rate
            notional = pos.margin_used / self._margin_rate if self._margin_rate > 0 else 0
            venue_notional[pos.venue] = venue_notional.get(pos.venue, 0.0) + notional

        if len(venue_margin) < 2:
            return []

        # Calculate ideal margin per venue (proportional to notional)
        total_margin = sum(venue_margin.values())
        total_notional = sum(venue_notional.values())
        if total_notional <= 0 or total_margin <= 0:
            return []

        ideal_margin: Dict[str, float] = {}
        for venue, notional in venue_notional.items():
            ideal_margin[venue] = total_margin * (notional / total_notional)

        # Find surplus and deficit venues
        surplus: List[Tuple[str, float]] = []
        deficit: List[Tuple[str, float]] = []

        for venue in venue_margin:
            diff = venue_margin[venue] - ideal_margin.get(venue, 0.0)
            if diff > 10.0:  # Only worth transferring if >$10
                surplus.append((venue, diff))
            elif diff < -10.0:
                deficit.append((venue, abs(diff)))

        transfers: List[TransferRecommendation] = []
        for from_venue, excess in sorted(surplus, key=lambda x: -x[1]):
            for to_venue, needed in sorted(deficit, key=lambda x: -x[1]):
                amount = min(excess, needed, max_transfer_usd)
                if amount < 10.0:
                    continue
                transfers.append(
                    TransferRecommendation(
                        from_venue=from_venue,
                        to_venue=to_venue,
                        amount_usd=round(amount, 2),
                        reason=f"Rebalance: {from_venue} over-margined by ${excess:.0f}, "
                               f"{to_venue} under-margined by ${needed:.0f}",
                        margin_freed_usd=round(amount * 0.5, 2),  # conservative estimate
                    )
                )

        transfers.sort(key=lambda t: -t.margin_freed_usd)
        if transfers:
            logger.info(
                "CrossMarginOptimizer: %d transfer suggestions, "
                "total margin freed $%.2f",
                len(transfers),
                sum(t.margin_freed_usd for t in transfers),
            )
        return transfers
