#!/usr/bin/env python3
"""
Maker Rebate Optimizer — maximise maker rebates through smart limit placement.

Builds on top of ``FeeOptimizer`` (execution/fee_optimizer.py) to provide
actionable limit-price placement, queue-priority scoring, rebate simulation,
and tier-upgrade ROI calculations.

Classes
-------
RebateOpportunity     — rebate opportunity descriptor (dataclass)
MakerRebateOptimizer  — core optimiser

Usage::

    opt = MakerRebateOptimizer()
    price = opt.optimal_limit_price("kraken", "buy", 30_000.0, spread_bps=5.0, urgency=0.2)
    rebate = opt.expected_rebate_bps("kraken", size_usd=10_000)
    split  = opt.should_split_for_rebate(total_size_usd=50_000, spread_bps=4.0)
    roi    = opt.tier_upgrade_roi("kraken", extra_volume_usd=200_000)
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from execution.fee_optimizer import FeeOptimizer, FeeTier, _FEE_SCHEDULES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RebateOpportunity
# ---------------------------------------------------------------------------

@dataclass
class RebateOpportunity:
    """Description of a single maker rebate opportunity.

    Attributes
    ----------
    venue : str
        Exchange name.
    symbol : str
        Instrument.
    side : str
        ``"buy"`` or ``"sell"``.
    estimated_rebate_bps : float
        Expected rebate in basis points (negative = fee, positive = rebate).
    queue_position_estimate : float
        Estimated fractional queue position (0 = front, 1 = back).
    fill_probability : float
        Estimated probability of fill in the next N minutes [0-1].
    expected_value_bps : float
        Risk-adjusted expected value: ``rebate_bps * fill_probability``.
    """

    venue:                  str
    symbol:                 str
    side:                   str
    estimated_rebate_bps:   float
    queue_position_estimate: float
    fill_probability:       float
    expected_value_bps:     float

    def __post_init__(self) -> None:
        # Ensure consistency
        if self.expected_value_bps == 0.0:
            self.expected_value_bps = self.estimated_rebate_bps * self.fill_probability


# ---------------------------------------------------------------------------
# Rolling 30-day volume tracker
# ---------------------------------------------------------------------------

@dataclass
class _VolumeRecord:
    timestamp: float
    volume_usd: float


class _RollingVolume:
    """Rolling 30-day USD volume tracker backed by a deque."""

    _WINDOW_SECONDS = 30 * 24 * 3600  # 30 days

    def __init__(self) -> None:
        self._records: Deque[_VolumeRecord] = deque()

    def add(self, volume_usd: float, ts: Optional[float] = None) -> None:
        """Record a trade."""
        self._records.append(_VolumeRecord(ts or time.time(), volume_usd))

    def total(self, as_of: Optional[float] = None) -> float:
        """Sum of volume in the rolling 30-day window."""
        now = as_of or time.time()
        cutoff = now - self._WINDOW_SECONDS
        # Purge stale records
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()
        return sum(r.volume_usd for r in self._records)


# ---------------------------------------------------------------------------
# MakerRebateOptimizer
# ---------------------------------------------------------------------------

class MakerRebateOptimizer:
    """Maker rebate optimisation engine.

    Maintains a per-venue rolling 30-day volume tracker to determine the
    current fee tier, then provides optimal limit-price placement, queue
    scoring, and rebate simulation.

    Parameters
    ----------
    fee_optimizer : FeeOptimizer | None
        Underlying fee schedule engine.  A default instance is created if
        ``None`` is passed.
    initial_volumes : dict[str, float] | None
        Optional pre-seed of 30d volumes per venue (e.g. from persistence).
    """

    # Minimum spread in bps below which splitting is not worthwhile
    _MIN_SPREAD_FOR_SPLIT_BPS = 3.0
    # Minimum trade size (USD) below which overhead of splitting dominates
    _MIN_SPLIT_SIZE_USD = 10_000.0
    # Rebate threshold for split decision (bps)
    _SPLIT_REBATE_THRESHOLD_BPS = 2.0

    def __init__(
        self,
        fee_optimizer: Optional[FeeOptimizer] = None,
        initial_volumes: Optional[Dict[str, float]] = None,
    ) -> None:
        self._fee_opt  = fee_optimizer or FeeOptimizer()
        self._volumes: Dict[str, _RollingVolume] = {}
        if initial_volumes:
            for venue, vol in initial_volumes.items():
                rv = _RollingVolume()
                rv.add(vol)
                self._volumes[venue] = rv

    # ------------------------------------------------------------------
    # Volume tracking
    # ------------------------------------------------------------------

    def record_trade(self, venue: str, size_usd: float) -> None:
        """Record an executed trade for rolling-volume tracking."""
        venue = venue.lower()
        if venue not in self._volumes:
            self._volumes[venue] = _RollingVolume()
        self._volumes[venue].add(size_usd)

    def _get_volume(self, venue: str) -> float:
        v = self._volumes.get(venue.lower())
        return v.total() if v else 0.0

    # ------------------------------------------------------------------
    # Fee tier (current)
    # ------------------------------------------------------------------

    def _current_tier(self, venue: str) -> FeeTier:
        return self._fee_opt.get_fee_tier(venue, self._get_volume(venue))

    # ------------------------------------------------------------------
    # Optimal limit price placement
    # ------------------------------------------------------------------

    def optimal_limit_price(
        self,
        venue:      str,
        side:       str,
        best_price: float,
        spread_bps: float,
        urgency:    float,
    ) -> float:
        """Compute the optimal limit price to maximise maker rebate probability.

        Strategy
        --------
        * ``urgency < 0.3``  — 1–2 bps inside the spread to jump the queue
          while remaining a maker (i.e. not crossing).
        * ``urgency 0.3–0.7`` — join the best bid/ask exactly.
        * ``urgency > 0.7``  — cross the spread (taker); return the far side.

        Parameters
        ----------
        venue : str
            Exchange name.
        side : str
            ``"buy"`` or ``"sell"``.
        best_price : float
            Current best bid (if buying) or best ask (if selling).
        spread_bps : float
            Current bid-ask spread in basis points.
        urgency : float
            Urgency score [0, 1].

        Returns
        -------
        float
            Recommended limit price.
        """
        urgency = max(0.0, min(1.0, urgency))
        side    = side.lower()
        tick    = best_price * spread_bps / 10_000.0   # 1 bps in price units

        if urgency < 0.3:
            # Queue-jump: 1 bps inside spread, stay maker
            improvement = tick * 1.5  # 1.5 bps
            if side == "buy":
                price = best_price + improvement  # improve bid
            else:
                price = best_price - improvement  # improve ask
            logger.debug("optimal_limit_price LOW URGENCY %.6f -> %.6f", best_price, price)

        elif urgency <= 0.7:
            # Join best bid/ask exactly
            price = best_price
            logger.debug("optimal_limit_price MID URGENCY join %.6f", price)

        else:
            # Cross spread (taker)
            half_spread = tick * spread_bps / 2.0
            if side == "buy":
                price = best_price + half_spread   # lift the ask
            else:
                price = best_price - half_spread   # hit the bid
            logger.debug("optimal_limit_price HIGH URGENCY cross %.6f -> %.6f", best_price, price)

        return round(price, 8)

    # ------------------------------------------------------------------
    # Expected rebate
    # ------------------------------------------------------------------

    def expected_rebate_bps(self, venue: str, size_usd: float) -> float:
        """Return the expected maker rebate in basis points for *venue*.

        A negative value means the exchange charges a maker fee (not rebate).
        Positive values (e.g. Binance BNB discounts, some DeFi venues) mean a
        true rebate.

        Based purely on the current fee tier's maker_bps; caller should
        negate the sign convention for rebate arithmetic.

        Parameters
        ----------
        venue : str
            Exchange name.
        size_usd : float
            Trade notional in USD (used to project whether this trade would
            push volume into a higher tier intra-period).

        Returns
        -------
        float
            Maker fee in bps (positive = cost, negative = true rebate).
        """
        tier = self._current_tier(venue)
        _ = size_usd  # may be used for projected-tier logic in future
        return tier.maker_bps

    # ------------------------------------------------------------------
    # Should split for rebate
    # ------------------------------------------------------------------

    def should_split_for_rebate(
        self,
        total_size_usd: float,
        spread_bps:     float,
    ) -> bool:
        """Decide whether to split a large order into maker slices.

        Splitting is worthwhile when:
        1. Total size is large enough to absorb the operational overhead.
        2. The spread is wide enough that the maker-taker differential
           meaningfully exceeds the expected slippage from splitting.

        Parameters
        ----------
        total_size_usd : float
            Total order notional in USD.
        spread_bps : float
            Current bid-ask spread in basis points.

        Returns
        -------
        bool
        """
        if total_size_usd < self._MIN_SPLIT_SIZE_USD:
            return False
        if spread_bps < self._MIN_SPREAD_FOR_SPLIT_BPS:
            return False
        # Rough maker-taker saving vs taker cost
        # Assume kraken-level differential as a conservative baseline
        taker_bps = 26.0
        maker_bps = 16.0
        saving_bps = taker_bps - maker_bps   # 10 bps
        # Cost of splitting: additional spread exposure ≈ half spread per slice
        n_slices   = max(2, int(total_size_usd / 50_000))
        slice_cost = spread_bps * 0.5 * (n_slices - 1) / n_slices
        net_saving = saving_bps - slice_cost
        result     = net_saving > self._SPLIT_REBATE_THRESHOLD_BPS
        logger.debug(
            "should_split_for_rebate: size=$%.0f spread=%.1fbps n_slices=%d "
            "saving=%.1f slice_cost=%.1f net=%.1f -> %s",
            total_size_usd, spread_bps, n_slices, saving_bps, slice_cost, net_saving, result,
        )
        return result

    # ------------------------------------------------------------------
    # Queue priority score
    # ------------------------------------------------------------------

    def queue_priority_score(
        self,
        venue:      str,
        symbol:     str,
        our_price:  float,
        book_depth: List[Tuple[float, float]],
    ) -> float:
        """Estimate our queue position as a fraction [0 = front, 1 = back].

        Uses book depth to estimate how much volume is ahead of our price
        level.  Returns 0 if our price is better than the best bid/ask.

        Parameters
        ----------
        venue : str
            Exchange name.
        symbol : str
            Instrument.
        our_price : float
            Our limit price.
        book_depth : list of (price, qty) tuples
            Bid or ask side of the order book, sorted best-first.

        Returns
        -------
        float
            Estimated queue position fraction [0, 1].
        """
        if not book_depth:
            return 0.5  # unknown

        best = book_depth[0][0]
        total_qty_ahead = 0.0
        total_qty       = 0.0

        for px, qty in book_depth:
            total_qty += qty
            # For a bid, volume at prices better than ours is ahead in queue
            if abs(px - our_price) < 1e-10:
                # Same price level — we are at the back of this level
                total_qty_ahead += qty * 0.9   # conservative: 90% ahead
            elif px > our_price:
                total_qty_ahead += qty         # strictly better bid → ahead

        if total_qty <= 0:
            return 0.0
        score = min(1.0, total_qty_ahead / total_qty)
        logger.debug("queue_priority_score: %s %s our=%.4f score=%.3f", venue, symbol, our_price, score)
        return score

    # ------------------------------------------------------------------
    # Rebate PnL simulation
    # ------------------------------------------------------------------

    def simulate_rebate_pnl(
        self,
        strategy_trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Simulate what maker rebates would have earned on historical trades.

        Parameters
        ----------
        strategy_trades : list of dict
            Each dict must have keys: ``venue``, ``size_usd``, ``side``,
            ``order_type`` (``"maker"`` or ``"taker"``).

        Returns
        -------
        dict
            Summary with keys ``total_rebate_usd``, ``total_fee_usd``,
            ``net_usd``, ``by_venue``, ``maker_fill_rate``.
        """
        by_venue: Dict[str, Dict[str, float]] = {}
        total_rebate_usd = 0.0
        total_fee_usd    = 0.0
        maker_count      = 0
        total_count      = len(strategy_trades)

        for trade in strategy_trades:
            venue    = trade.get("venue", "kraken").lower()
            size_usd = float(trade.get("size_usd", 0.0))
            ot       = trade.get("order_type", "taker")

            if venue not in by_venue:
                by_venue[venue] = {"rebate_usd": 0.0, "fee_usd": 0.0, "maker_trades": 0, "taker_trades": 0}

            tier = self._fee_opt.get_fee_tier(venue, self._get_volume(venue))

            if ot == "maker":
                fee_usd = size_usd * tier.maker_bps / 10_000.0
                total_rebate_usd += fee_usd  # treated as savings vs taker
                by_venue[venue]["rebate_usd"] += fee_usd
                by_venue[venue]["maker_trades"] += 1
                maker_count += 1
                self.record_trade(venue, size_usd)
            else:
                fee_usd = size_usd * tier.taker_bps / 10_000.0
                total_fee_usd += fee_usd
                by_venue[venue]["fee_usd"] += fee_usd
                by_venue[venue]["taker_trades"] += 1

        maker_fill_rate = maker_count / total_count if total_count > 0 else 0.0
        net_usd         = total_fee_usd - total_rebate_usd  # lower is better

        return {
            "total_rebate_usd": total_rebate_usd,
            "total_fee_usd":    total_fee_usd,
            "net_usd":          net_usd,
            "by_venue":         by_venue,
            "maker_fill_rate":  maker_fill_rate,
            "total_trades":     total_count,
        }

    # ------------------------------------------------------------------
    # Tier upgrade ROI
    # ------------------------------------------------------------------

    def tier_upgrade_roi(self, venue: str, extra_volume_usd: float) -> float:
        """Return the expected improvement in maker rebate per dollar of extra volume.

        Calculates the marginal reduction in maker fee (bps) if
        ``extra_volume_usd`` additional 30d volume were added, and
        expresses it as a fractional rebate rate.

        Parameters
        ----------
        venue : str
            Exchange name.
        extra_volume_usd : float
            Hypothetical extra trading volume in USD over 30 days.

        Returns
        -------
        float
            Basis points improvement in maker rate per $1 of extra volume
            (scaled by 1e6 for readability).  Returns 0.0 if already at top tier.
        """
        current_vol  = self._get_volume(venue)
        current_tier = self._fee_opt.get_fee_tier(venue, current_vol)
        future_tier  = self._fee_opt.get_fee_tier(venue, current_vol + extra_volume_usd)

        bps_improvement = current_tier.maker_bps - future_tier.maker_bps
        if extra_volume_usd <= 0:
            return 0.0

        roi = bps_improvement / extra_volume_usd  # bps per USD
        logger.info(
            "tier_upgrade_roi: %s current=%s future=%s improvement=%.2f bps roi=%.6e bps/USD",
            venue, current_tier.tier_name, future_tier.tier_name, bps_improvement, roi,
        )
        return roi

    # ------------------------------------------------------------------
    # Opportunity generator
    # ------------------------------------------------------------------

    def find_rebate_opportunities(
        self,
        venues:     List[str],
        symbol:     str,
        side:       str,
        size_usd:   float,
        book_data:  Optional[Dict[str, Dict]] = None,
    ) -> List[RebateOpportunity]:
        """Generate and rank maker rebate opportunities across venues.

        Parameters
        ----------
        venues : list of str
            Candidate venues.
        symbol : str
            Instrument.
        side : str
            ``"buy"`` or ``"sell"``.
        size_usd : float
            Trade notional in USD.
        book_data : dict | None
            Optional dict keyed by venue with ``"spread_bps"`` and
            ``"book_depth"`` keys for richer scoring.

        Returns
        -------
        list of RebateOpportunity
            Sorted by expected_value_bps descending.
        """
        opportunities: List[RebateOpportunity] = []

        for venue in venues:
            try:
                rebate_bps  = self.expected_rebate_bps(venue, size_usd)
                spread_bps  = (book_data or {}).get(venue, {}).get("spread_bps", 5.0)
                book_depth  = (book_data or {}).get(venue, {}).get("book_depth", [])
                best_price  = (book_data or {}).get(venue, {}).get("best_price", 1.0)
                queue_score = self.queue_priority_score(venue, symbol, best_price, book_depth)

                # Estimate fill probability: inversely proportional to queue position and spread
                fill_prob = max(0.0, 1.0 - queue_score * 0.5) * min(1.0, spread_bps / 10.0)
                ev_bps    = rebate_bps * fill_prob  # crude but directionally correct

                opp = RebateOpportunity(
                    venue                  = venue,
                    symbol                 = symbol,
                    side                   = side,
                    estimated_rebate_bps   = rebate_bps,
                    queue_position_estimate = queue_score,
                    fill_probability       = fill_prob,
                    expected_value_bps     = ev_bps,
                )
                opportunities.append(opp)
            except ValueError:
                logger.warning("Skipping unknown venue: %s", venue)

        opportunities.sort(key=lambda o: o.expected_value_bps)
        return opportunities
