"""
Cross-Venue Arbitrage Signal Pipeline.

Real-time arbitrage signal generator supporting:
  - Direct cross-venue (two-leg) arbitrage
  - Triangular arbitrage (A→B→C→A on a single venue)
  - Spot-perpetual funding-rate arbitrage

Each venue's order book is represented as a VenueBook snapshot.
Opportunities are ranked by net expected value after all fees.

Usage
-----
    pipeline = CrossVenueArbPipeline()
    pipeline.update_venue(VenueBook("binance", "BTC/USDT", ...))
    pipeline.update_venue(VenueBook("okx",     "BTC/USDT", ...))
    opps = pipeline.scan_arbitrage(min_net_spread_bps=0.5)
    funding = pipeline.scan_funding_arb(venues)
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VenueBook:
    """Snapshot of a venue's best bid/ask for one symbol."""
    venue: str
    symbol: str
    best_bid: float           # highest buyer bid (we sell here)
    best_ask: float           # lowest seller ask (we buy here)
    bid_size: float           # available quantity at best bid
    ask_size: float           # available quantity at best ask
    timestamp_ns: int         # nanosecond timestamp
    fee_maker_bps: float      # maker rebate/fee in bps (negative = rebate)
    fee_taker_bps: float      # taker fee in bps (positive = cost)

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def market_spread_bps(self) -> float:
        if self.mid <= 0:
            return 0.0
        return (self.best_ask - self.best_bid) / self.mid * 10_000


@dataclass
class ArbOpportunity:
    """A detected cross-venue arbitrage opportunity."""
    buy_venue: str
    sell_venue: str
    symbol: str
    gross_spread_bps: float       # raw price difference in bps
    net_spread_bps: float         # after both legs as taker
    maker_net_spread_bps: float   # buy as maker on buy_venue
    fill_probability: float       # P(both legs fill before opp disappears)
    max_size_usd: float           # depth-constrained, position-limit-adjusted
    latency_budget_us: float      # estimated opportunity lifetime in µs
    expected_value_usd: float     # net_spread_bps * max_size_usd / 10_000

    def __repr__(self) -> str:
        return (
            f"ArbOpportunity({self.buy_venue}→{self.sell_venue} "
            f"{self.symbol} net={self.net_spread_bps:.2f}bps "
            f"EV=${self.expected_value_usd:.4f})"
        )


@dataclass
class TriangularArbOpportunity:
    """A detected triangular arbitrage opportunity on a single venue."""
    venue: str
    path: List[str]              # e.g. ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
    gross_profit_bps: float
    net_profit_bps: float        # after 3 × taker_fee
    max_size_usd: float
    expected_value_usd: float

    def __repr__(self) -> str:
        path_str = " → ".join(self.path)
        return (
            f"TriArbOpportunity({self.venue} {path_str} "
            f"net={self.net_profit_bps:.2f}bps EV=${self.expected_value_usd:.4f})"
        )


@dataclass
class FundingRateSignal:
    """Funding-rate arbitrage signal between spot and perpetual."""
    symbol: str
    venue: str
    rate_8h: float              # 8-hour funding rate (positive = longs pay shorts)
    annualised_rate: float      # annualised yield = rate_8h * 3 * 365
    direction: str              # "LONG_SPOT_SHORT_PERP" or "SHORT_SPOT_LONG_PERP"
    expected_value_bps: float   # net edge per position after fees

    def __repr__(self) -> str:
        return (
            f"FundingRateSignal({self.venue} {self.symbol} "
            f"ann={self.annualised_rate:.2%} {self.direction})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Funding rate helpers  (module-level pure functions)
# ─────────────────────────────────────────────────────────────────────────────

def spot_perp_arb(
    spot_price: float,
    perp_price: float,
    funding_rate_8h: float,
    fee_bps: float,
) -> float:
    """
    Estimate expected value in bps of a spot-perpetual arbitrage position.

    Strategy: if funding_rate_8h > 0 (longs pay shorts):
        → Buy spot + Short perpetual, collect funding payment.
      Net edge = funding_rate_8h - round_trip_fee_bps / 10_000

    If funding_rate_8h < 0 (shorts pay longs):
        → Short spot + Long perpetual, collect funding payment.

    The basis (perp_price - spot_price) also contributes edge at convergence.

    Args:
        spot_price:       current spot mid price
        perp_price:       current perpetual contract mid price
        funding_rate_8h:  8-hour funding rate (signed decimal, e.g. 0.0001 = 1bp)
        fee_bps:          round-trip taker fee in bps (both legs)

    Returns:
        Expected value in bps per 8-hour period (can be negative).
    """
    if spot_price <= 0 or perp_price <= 0:
        return 0.0

    # Basis contribution (realises at funding/settlement)
    basis_bps = (perp_price - spot_price) / spot_price * 10_000

    # Funding edge (in bps) per 8h
    funding_bps = funding_rate_8h * 10_000

    if funding_rate_8h >= 0:
        # Collect positive funding by being short perp
        net_edge_bps = funding_bps - basis_bps - fee_bps
    else:
        # Collect negative funding (shorts pay) by being long perp
        net_edge_bps = abs(funding_bps) + basis_bps - fee_bps

    return float(net_edge_bps)


def scan_funding_arb(
    venues: List[Dict[str, Any]],
    min_annualised_yield: float = 0.05,
) -> List[FundingRateSignal]:
    """
    Scan a list of venue/symbol/funding-rate dicts for profitable funding arb.

    Args:
        venues: list of dicts with keys:
            - "venue": str
            - "symbol": str
            - "spot_price": float
            - "perp_price": float
            - "funding_rate_8h": float
            - "fee_bps": float   (round-trip taker fee, both legs)
        min_annualised_yield: minimum annualised yield to include (default 5%)

    Returns:
        List of FundingRateSignal sorted descending by annualised_rate.
    """
    signals: List[FundingRateSignal] = []

    for v in venues:
        try:
            rate_8h = float(v.get("funding_rate_8h", 0.0))
            spot    = float(v.get("spot_price", 0.0))
            perp    = float(v.get("perp_price", 0.0))
            fee_bps = float(v.get("fee_bps", 10.0))
            symbol  = str(v.get("symbol", ""))
            venue   = str(v.get("venue", ""))

            if spot <= 0 or perp <= 0:
                continue

            ev_bps = spot_perp_arb(spot, perp, rate_8h, fee_bps)

            # Annualise: 3 funding periods per day × 365
            ann_rate = rate_8h * 3 * 365

            if abs(ann_rate) < min_annualised_yield:
                continue

            direction = (
                "LONG_SPOT_SHORT_PERP" if rate_8h >= 0
                else "SHORT_SPOT_LONG_PERP"
            )

            signals.append(FundingRateSignal(
                symbol=symbol,
                venue=venue,
                rate_8h=rate_8h,
                annualised_rate=ann_rate,
                direction=direction,
                expected_value_bps=round(ev_bps, 4),
            ))

        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("scan_funding_arb: bad venue entry %s: %s", v, exc)

    return sorted(signals, key=lambda s: abs(s.annualised_rate), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class CrossVenueArbPipeline:
    """
    Real-time cross-venue arbitrage signal generator.

    Maintains a live view of multiple VenueBook snapshots and emits
    arbitrage opportunities whenever profitable after fees.

    Parameters
    ----------
    position_limit_usd:
        Maximum USD notional per arb leg.
    max_age_ns:
        Discard venue snapshots older than this (nanoseconds). Default 5s.
    decay_halflife_s:
        Half-life used when estimating opportunity lifetime from spread
        decay history. Default 0.5 s.
    """

    def __init__(
        self,
        position_limit_usd: float = 10_000.0,
        max_age_ns: int = 5_000_000_000,   # 5 seconds
        decay_halflife_s: float = 0.5,
    ):
        self.position_limit_usd = position_limit_usd
        self.max_age_ns = max_age_ns
        self.decay_halflife_s = decay_halflife_s

        # venue → symbol → VenueBook
        self._books: Dict[str, Dict[str, VenueBook]] = defaultdict(dict)

        # (symbol, venue_pair) → deque of (timestamp_ns, spread_bps)
        self._spread_history: Dict[Tuple[str, str], Deque[Tuple[int, float]]] = (
            defaultdict(lambda: deque(maxlen=500))
        )

        # Fill history for learning
        self._fill_log: List[Dict[str, Any]] = []
        self._opportunity_log: List[ArbOpportunity] = []

    # ── Book management ───────────────────────────────────────────────────────

    def update_venue(self, venue_book: VenueBook) -> None:
        """
        Ingest a fresh VenueBook snapshot.

        Also records the current spread into the history buffer so
        historical_decay_rate() can estimate opportunity lifetime.

        Args:
            venue_book: fresh snapshot from the venue.
        """
        venue = venue_book.venue
        symbol = venue_book.symbol
        self._books[venue][symbol] = venue_book

        # Record spread for decay estimation
        key = (symbol, venue)
        self._spread_history[key].append(
            (venue_book.timestamp_ns, venue_book.market_spread_bps)
        )

        logger.debug(
            "update_venue: %s %s bid=%.4f ask=%.4f",
            venue, symbol, venue_book.best_bid, venue_book.best_ask,
        )

    def _stale(self, book: VenueBook) -> bool:
        """Return True if the book snapshot is older than max_age_ns."""
        now_ns = time.time_ns()
        return (now_ns - book.timestamp_ns) > self.max_age_ns

    # ── Cross-venue arbitrage scan ────────────────────────────────────────────

    def scan_arbitrage(
        self,
        min_net_spread_bps: float = 0.5,
    ) -> List[ArbOpportunity]:
        """
        Find all profitable cross-venue arbitrage opportunities.

        Checks every pair of venues for each shared symbol. An opportunity
        exists when: buy at venue A's ask < sell at venue B's bid (net of fees).

        Args:
            min_net_spread_bps: minimum net spread (after taker fees on both
                                legs) required to emit a signal. Default 0.5bps.

        Returns:
            List of ArbOpportunity sorted descending by expected_value_usd.
        """
        opportunities: List[ArbOpportunity] = []

        # Build symbol → list of (venue, book) mapping
        symbol_venues: Dict[str, List[Tuple[str, VenueBook]]] = defaultdict(list)
        for venue, books in self._books.items():
            for symbol, book in books.items():
                if not self._stale(book):
                    symbol_venues[symbol].append((venue, book))

        for symbol, venue_books in symbol_venues.items():
            if len(venue_books) < 2:
                continue

            for (venue_a, book_a), (venue_b, book_b) in combinations(venue_books, 2):
                # Try A as buy venue, B as sell venue
                for buy_book, sell_book, buy_venue, sell_venue in [
                    (book_a, book_b, venue_a, venue_b),
                    (book_b, book_a, venue_b, venue_a),
                ]:
                    opp = self._evaluate_arb(
                        buy_venue=buy_venue,
                        sell_venue=sell_venue,
                        buy_book=buy_book,
                        sell_book=sell_book,
                        symbol=symbol,
                        min_net_spread_bps=min_net_spread_bps,
                    )
                    if opp is not None:
                        opportunities.append(opp)

        opportunities.sort(key=lambda o: o.expected_value_usd, reverse=True)
        self._opportunity_log.extend(opportunities)
        return opportunities

    def _evaluate_arb(
        self,
        buy_venue: str,
        sell_venue: str,
        buy_book: VenueBook,
        sell_book: VenueBook,
        symbol: str,
        min_net_spread_bps: float,
    ) -> Optional[ArbOpportunity]:
        """
        Evaluate whether buying on buy_venue and selling on sell_venue is
        profitable after fees.

        We use taker fees on both legs (worst case for speed-sensitive arb).
        Also computes the maker scenario where we post a limit buy.

        Returns:
            ArbOpportunity or None if below threshold.
        """
        buy_price  = buy_book.best_ask    # we lift the ask
        sell_price = sell_book.best_bid   # we hit the bid

        if buy_price <= 0 or sell_price <= 0:
            return None

        mid_ref = (buy_price + sell_price) / 2.0

        # Gross spread
        gross_bps = (sell_price - buy_price) / mid_ref * 10_000

        # Taker fees on both legs
        total_fee_bps = buy_book.fee_taker_bps + sell_book.fee_taker_bps
        net_spread_bps = gross_bps - total_fee_bps

        # Maker-taker scenario: post limit buy, take sell
        maker_fee_bps = buy_book.fee_maker_bps + sell_book.fee_taker_bps
        maker_net_spread_bps = gross_bps - maker_fee_bps

        if net_spread_bps < min_net_spread_bps:
            return None

        # Size constrained by depth and position limit
        max_qty_by_depth = min(buy_book.ask_size, sell_book.bid_size)
        max_usd_by_depth = max_qty_by_depth * buy_price
        max_size_usd = min(max_usd_by_depth, self.position_limit_usd)

        # Fill probability: rough estimate from spread decay
        # Higher gross spread → opportunity more persistent → higher fill prob
        p_fill = float(np.clip(1.0 - math.exp(-gross_bps / 5.0), 0.01, 0.99))

        # Latency budget: time until the spread likely collapses
        decay_rate = self.historical_decay_rate(symbol, f"{buy_venue}_{sell_venue}")
        if decay_rate > 0:
            latency_us = (1.0 / decay_rate) * 1e6  # seconds → µs
        else:
            latency_us = self.decay_halflife_s * 1e6  # fallback

        # Expected value
        ev_usd = net_spread_bps * max_size_usd / 10_000 * p_fill

        return ArbOpportunity(
            buy_venue=buy_venue,
            sell_venue=sell_venue,
            symbol=symbol,
            gross_spread_bps=round(gross_bps, 4),
            net_spread_bps=round(net_spread_bps, 4),
            maker_net_spread_bps=round(maker_net_spread_bps, 4),
            fill_probability=round(p_fill, 4),
            max_size_usd=round(max_size_usd, 2),
            latency_budget_us=round(latency_us, 1),
            expected_value_usd=round(ev_usd, 6),
        )

    # ── Triangular arbitrage ──────────────────────────────────────────────────

    def scan_triangular(
        self,
        symbols: List[str],
        min_net_profit_bps: float = 1.0,
    ) -> List[TriangularArbOpportunity]:
        """
        Search for triangular arbitrage on each venue.

        A triangular arb completes a cycle A→B→C→A using three currency
        pairs where each pair's base/quote forms part of the triangle.

        We express each step as:
          - "BUY"  means: pay quote, receive base   (lift ask)
          - "SELL" means: receive quote, pay base    (hit bid)

        Profit in bps = (product_of_effective_rates − 1) × 10_000

        Args:
            symbols: list of symbol strings present in the venue books,
                     e.g. ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
            min_net_profit_bps: minimum net profit after 3 taker fees.

        Returns:
            List of TriangularArbOpportunity sorted by expected_value_usd.
        """
        results: List[TriangularArbOpportunity] = []
        seen: set = set()

        for venue, books in self._books.items():
            venue_symbols = {s: b for s, b in books.items()
                             if s in symbols and not self._stale(b)}
            if len(venue_symbols) < 3:
                continue

            # Build currency graph
            # pair (base, quote) → (ask, bid, ask_size, bid_size, fee_taker)
            pairs: Dict[Tuple[str, str], Tuple[float, float, float, float, float]] = {}
            for sym, book in venue_symbols.items():
                parts = sym.split("/")
                if len(parts) != 2:
                    continue
                base, quote = parts
                pairs[(base, quote)] = (
                    book.best_ask, book.best_bid,
                    book.ask_size, book.bid_size,
                    book.fee_taker_bps,
                )

            currencies = list({c for pair in pairs for c in pair})

            # Try every ordered triple of distinct currencies
            for A in currencies:
                for B in currencies:
                    if B == A:
                        continue
                    for C in currencies:
                        if C == A or C == B:
                            continue
                        key = (venue, A, B, C)
                        if key in seen:
                            continue
                        seen.add(key)

                        opp = self._eval_triangular_abc(
                            venue=venue,
                            pairs=pairs,
                            A=A, B=B, C=C,
                            min_net_profit_bps=min_net_profit_bps,
                        )
                        if opp is not None:
                            results.append(opp)

        results.sort(key=lambda o: o.expected_value_usd, reverse=True)
        return results

    def _exchange_rate(
        self,
        pairs: Dict[Tuple[str, str], Tuple[float, float, float, float, float]],
        from_curr: str,
        to_curr: str,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get the effective taker rate to convert from_curr → to_curr.

        Returns (effective_rate, depth_usd, fee_bps) or None if no pair exists.
        The rate is the amount of to_curr received per unit of from_curr.
        """
        if (from_curr, to_curr) in pairs:
            # SELL from_curr, receive to_curr at bid
            ask, bid, ask_sz, bid_sz, fee = pairs[(from_curr, to_curr)]
            if bid <= 0:
                return None
            depth_usd = bid_sz * bid
            return (bid, depth_usd, fee)   # rate = bid price in to_curr per from_curr

        if (to_curr, from_curr) in pairs:
            # BUY to_curr using from_curr: pay ask price in from_curr per to_curr
            ask, bid, ask_sz, bid_sz, fee = pairs[(to_curr, from_curr)]
            if ask <= 0:
                return None
            depth_usd = ask_sz * ask
            return (1.0 / ask, depth_usd, fee)  # rate = 1/ask in to_curr per from_curr

        return None

    def _eval_triangular_abc(
        self,
        venue: str,
        pairs: Dict[Tuple[str, str], Tuple[float, float, float, float, float]],
        A: str, B: str, C: str,
        min_net_profit_bps: float,
    ) -> Optional[TriangularArbOpportunity]:
        """
        Evaluate the cycle A→B→C→A.

        Starting with 1 unit of currency A:
          step1: convert A → B  (leg rate r1)
          step2: convert B → C  (leg rate r2)
          step3: convert C → A  (leg rate r3)
        product = r1 * r2 * r3  → profit if product > 1

        Returns TriangularArbOpportunity or None.
        """
        r1 = self._exchange_rate(pairs, A, B)
        r2 = self._exchange_rate(pairs, B, C)
        r3 = self._exchange_rate(pairs, C, A)

        if r1 is None or r2 is None or r3 is None:
            return None

        rate1, depth1, fee1 = r1
        rate2, depth2, fee2 = r2
        rate3, depth3, fee3 = r3

        product = rate1 * rate2 * rate3
        gross_bps = (product - 1.0) * 10_000
        total_fee_bps = fee1 + fee2 + fee3
        net_bps = gross_bps - total_fee_bps

        if net_bps < min_net_profit_bps:
            return None

        min_depth_usd = min(depth1, depth2, depth3)
        max_size_usd = min(min_depth_usd, 10_000.0)
        ev_usd = net_bps * max_size_usd / 10_000

        path_str = [f"{A}→{B}", f"{B}→{C}", f"{C}→{A}"]

        return TriangularArbOpportunity(
            venue=venue,
            path=path_str,
            gross_profit_bps=round(gross_bps, 4),
            net_profit_bps=round(net_bps, 4),
            max_size_usd=round(max_size_usd, 2),
            expected_value_usd=round(ev_usd, 6),
        )

    # ── Historical spread decay ───────────────────────────────────────────────

    def historical_decay_rate(self, symbol: str, venue_pair: str) -> float:
        """
        Estimate the rate at which cross-venue spread collapses.

        Uses the cross-venue spread history to fit an exponential decay:
            spread(t) = spread_0 * exp(−λ * t)

        Returns the decay rate λ (s⁻¹). Higher λ → faster collapse.

        Args:
            symbol:     trading symbol
            venue_pair: string like "binance_okx"

        Returns:
            Decay rate in s⁻¹. Returns reciprocal of decay_halflife_s if
            insufficient history exists.
        """
        key = (symbol, venue_pair)
        history = self._spread_history.get(key)
        if not history or len(history) < 5:
            return 1.0 / self.decay_halflife_s  # default

        ts_arr = np.array([h[0] for h in history], dtype=float)
        sp_arr = np.array([h[1] for h in history], dtype=float)

        if sp_arr.max() <= 0:
            return 1.0 / self.decay_halflife_s

        # Normalise time to seconds
        ts_sec = (ts_arr - ts_arr[0]) / 1e9

        # Robust log-linear fit: log(spread) = log(A) - λ * t
        positive_mask = sp_arr > 0
        if positive_mask.sum() < 3:
            return 1.0 / self.decay_halflife_s

        try:
            log_spread = np.log(sp_arr[positive_mask])
            t_vals = ts_sec[positive_mask]
            coeffs = np.polyfit(t_vals, log_spread, 1)
            decay_rate = float(-coeffs[0])  # λ = -slope
            return max(decay_rate, 0.001)   # never negative
        except (np.linalg.LinAlgError, ValueError):
            return 1.0 / self.decay_halflife_s

    # ── Signal summary ────────────────────────────────────────────────────────

    def signal_summary(self) -> Dict[str, Any]:
        """
        Return a high-level summary of pipeline activity.

        Returns
        -------
        Dict containing:
            top_opportunities: list of top-5 ArbOpportunity dicts
            total_opportunity_count: cumulative count since pipeline start
            avg_spread_bps: average net spread across all logged opportunities
            daily_pnl_estimate_usd: simple extrapolation assuming opportunities
                repeat 8h/day at observed frequency
        """
        all_opps = self._opportunity_log
        n = len(all_opps)

        if n == 0:
            return {
                "top_opportunities": [],
                "total_opportunity_count": 0,
                "avg_spread_bps": 0.0,
                "daily_pnl_estimate_usd": 0.0,
            }

        # Top-5 by EV
        top5 = sorted(all_opps, key=lambda o: o.expected_value_usd, reverse=True)[:5]
        top5_dicts = [
            {
                "buy_venue": o.buy_venue,
                "sell_venue": o.sell_venue,
                "symbol": o.symbol,
                "net_spread_bps": o.net_spread_bps,
                "expected_value_usd": o.expected_value_usd,
            }
            for o in top5
        ]

        avg_spread = float(np.mean([o.net_spread_bps for o in all_opps]))
        avg_ev = float(np.mean([o.expected_value_usd for o in all_opps]))

        # Simple daily estimate: assume we see N opps per observation window
        # and the pipeline runs 8 active hours/day
        ACTIVE_HOURS = 8
        daily_pnl_estimate = avg_ev * n * ACTIVE_HOURS  # crude upper bound

        return {
            "top_opportunities": top5_dicts,
            "total_opportunity_count": n,
            "avg_spread_bps": round(avg_spread, 4),
            "daily_pnl_estimate_usd": round(daily_pnl_estimate, 2),
        }
