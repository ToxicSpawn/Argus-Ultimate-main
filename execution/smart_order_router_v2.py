#!/usr/bin/env python3
"""
Smart Order Router V2 — venue-optimal order routing with learning.

Compares across Kraken, Bybit, Coinbase on price, spread, depth, fees,
and historical fill rate.  Learns from execution outcomes (expected vs actual
cost) to improve future routing decisions.

Features:
- ``get_best_venue(symbol, side, size_usd, venue_books)`` — single-venue pick
- ``split_across_venues(symbol, side, size_usd, venue_books)`` — multi-venue split
- ``record_execution(venue, symbol, expected_cost_bps, actual_cost_bps)`` — learn

Persistence: SQLite at ``data/smart_router.db``.

Usage::

    router = SmartOrderRouterV2()
    rec = router.get_best_venue("BTC/USD", "buy", 500.0, venue_books)
    logger.info(rec.venue, rec.total_cost_bps)
    splits = router.split_across_venues("BTC/USD", "buy", 2000.0, venue_books)
    for order in splits:
        logger.info(order.venue, order.size_usd)
"""
from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "smart_router.db"

# Default fee schedules in basis points (maker/taker)
_DEFAULT_FEES_BPS: Dict[str, Dict[str, float]] = {
    "kraken":   {"maker": 16.0, "taker": 26.0},
    "bybit":    {"maker": 10.0, "taker": 6.0},
    "coinbase": {"maker": 40.0, "taker": 60.0},
}

# Minimum number of historical executions before venue-level bias is used
_MIN_HISTORY_FOR_BIAS = 10

# Maximum portion of available depth to consume in a single fill (80%)
_MAX_DEPTH_UTILISATION = 0.80

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VenueRecommendation:
    """Result of single-venue routing decision."""

    venue: str
    expected_price: float
    expected_slippage_bps: float
    expected_fees_bps: float
    total_cost_bps: float
    confidence: float  # 0.0–1.0


@dataclass
class VenueOrder:
    """A single leg of a multi-venue split."""

    venue: str
    size_usd: float
    expected_price: float
    expected_slippage_bps: float
    expected_fees_bps: float
    total_cost_bps: float


@dataclass
class VenueBook:
    """Snapshot of a venue's order-book state for a single symbol.

    Callers populate this from live L2 data or ticker snapshots.
    """

    best_bid: float
    best_ask: float
    bid_depth_usd: float = 0.0   # total bid-side depth in USD within 20 bps
    ask_depth_usd: float = 0.0   # total ask-side depth in USD within 20 bps
    mid_price: float = 0.0       # (best_bid + best_ask) / 2
    last_trade_ts: float = 0.0   # epoch seconds of most recent fill on this venue

    def __post_init__(self) -> None:
        if self.mid_price == 0.0 and self.best_bid > 0 and self.best_ask > 0:
            self.mid_price = (self.best_bid + self.best_ask) / 2.0


# ---------------------------------------------------------------------------
# Smart Order Router V2
# ---------------------------------------------------------------------------


class SmartOrderRouterV2:
    """Venue-optimal order routing with outcome-based learning.

    Parameters
    ----------
    db_path : str or Path, optional
        SQLite database path.  Defaults to ``data/smart_router.db``.
    fee_overrides : dict, optional
        Per-venue fee overrides ``{"venue": {"maker": bps, "taker": bps}}``.
    min_depth_usd : float
        Minimum depth required on a venue to consider it routable.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        fee_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        min_depth_usd: float = 50.0,
    ) -> None:
        if db_path is None:
            self._db_path = _DEFAULT_DB_DIR / _DEFAULT_DB_NAME
        else:
            self._db_path = Path(db_path)

        self._fees = dict(_DEFAULT_FEES_BPS)
        if fee_overrides:
            for venue, schedule in fee_overrides.items():
                self._fees[venue.lower()] = schedule

        self._min_depth_usd = min_depth_usd
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(
            "SmartOrderRouterV2 initialised — db=%s, venues=%s",
            self._db_path,
            sorted(self._fees.keys()),
        )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS execution_history (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts           REAL    NOT NULL,
                        venue        TEXT    NOT NULL,
                        symbol       TEXT    NOT NULL,
                        side         TEXT    NOT NULL,
                        size_usd     REAL    NOT NULL DEFAULT 0,
                        expected_bps REAL    NOT NULL,
                        actual_bps   REAL    NOT NULL,
                        bias_bps     REAL    NOT NULL DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_exec_venue_symbol
                    ON execution_history (venue, symbol)
                """)
                conn.commit()
            finally:
                conn.close()

    def _get_venue_bias(self, venue: str, symbol: str) -> float:
        """Return average (actual - expected) cost in bps for a venue/symbol.

        Positive bias means the venue historically costs more than predicted.
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                """
                SELECT AVG(actual_bps - expected_bps), COUNT(*)
                FROM execution_history
                WHERE venue = ? AND symbol = ?
                ORDER BY ts DESC
                LIMIT 200
                """,
                (venue.lower(), symbol),
            ).fetchone()
            if row and row[1] >= _MIN_HISTORY_FOR_BIAS:
                return float(row[0] or 0.0)
            return 0.0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Venue scoring
    # ------------------------------------------------------------------

    def _score_venue(
        self,
        venue: str,
        side: str,
        size_usd: float,
        book: VenueBook,
        symbol: str,
    ) -> Optional[VenueRecommendation]:
        """Score a single venue and return a VenueRecommendation (or None if not routable)."""
        venue_lower = venue.lower()

        # Determine relevant depth
        depth = book.ask_depth_usd if side == "buy" else book.bid_depth_usd
        if depth < self._min_depth_usd:
            logger.debug(
                "SmartOrderRouterV2: %s depth %.1f USD < min %.1f — skipping",
                venue, depth, self._min_depth_usd,
            )
            return None

        # Spread in bps
        if book.mid_price <= 0:
            return None
        spread_bps = ((book.best_ask - book.best_bid) / book.mid_price) * 10_000.0

        # Slippage estimate: proportional to size / depth
        depth_ratio = min(size_usd / max(depth, 1.0), 1.0)
        # Square-root impact model (Kyle's lambda)
        raw_slippage_bps = spread_bps * 0.5 + 10.0 * math.sqrt(depth_ratio)

        # Fee
        fee_schedule = self._fees.get(venue_lower, {"maker": 25.0, "taker": 40.0})
        # Assume taker for market orders; maker for limit
        fee_bps = fee_schedule["taker"]

        # Historical bias adjustment
        bias = self._get_venue_bias(venue_lower, symbol)
        adjusted_slippage_bps = max(0.0, raw_slippage_bps + bias)

        total_cost_bps = adjusted_slippage_bps + fee_bps

        # Confidence: higher when depth is adequate and we have history
        depth_conf = min(1.0, depth / max(size_usd * 3.0, 1.0))
        staleness_penalty = 0.0
        if book.last_trade_ts > 0:
            age_s = time.time() - book.last_trade_ts
            if age_s > 60:
                staleness_penalty = min(0.3, age_s / 600.0)
        confidence = max(0.05, depth_conf - staleness_penalty)

        expected_price = book.best_ask if side == "buy" else book.best_bid

        return VenueRecommendation(
            venue=venue_lower,
            expected_price=expected_price,
            expected_slippage_bps=round(adjusted_slippage_bps, 2),
            expected_fees_bps=round(fee_bps, 2),
            total_cost_bps=round(total_cost_bps, 2),
            confidence=round(confidence, 4),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_best_venue(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        venue_books: Dict[str, VenueBook],
    ) -> VenueRecommendation:
        """Select the single best venue for an order.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/USD"``.
        side : str
            ``"buy"`` or ``"sell"``.
        size_usd : float
            Notional order size in USD.
        venue_books : dict
            Mapping of venue name → :class:`VenueBook`.

        Returns
        -------
        VenueRecommendation
            The lowest total-cost venue.  If no venue qualifies, returns a
            fallback recommendation with ``confidence=0``.
        """
        side = side.lower()
        candidates: List[VenueRecommendation] = []

        for venue_name, book in venue_books.items():
            rec = self._score_venue(venue_name, side, size_usd, book, symbol)
            if rec is not None:
                candidates.append(rec)

        if not candidates:
            logger.warning(
                "SmartOrderRouterV2: no routable venue for %s %s %.1f USD",
                symbol, side, size_usd,
            )
            return VenueRecommendation(
                venue="none",
                expected_price=0.0,
                expected_slippage_bps=0.0,
                expected_fees_bps=0.0,
                total_cost_bps=0.0,
                confidence=0.0,
            )

        # Sort by total cost (ascending), break ties by confidence (descending)
        candidates.sort(key=lambda r: (r.total_cost_bps, -r.confidence))
        best = candidates[0]
        logger.info(
            "SmartOrderRouterV2: best venue for %s %s %.1f USD → %s "
            "(cost %.1f bps, conf %.2f)",
            symbol, side, size_usd, best.venue, best.total_cost_bps, best.confidence,
        )
        return best

    def split_across_venues(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        venue_books: Dict[str, VenueBook],
    ) -> List[VenueOrder]:
        """Split an order across multiple venues to minimise total cost.

        Uses inverse-cost weighting: venues with lower expected cost get a
        larger share.  Each venue's allocation is capped by available depth.

        Parameters
        ----------
        symbol : str
            Trading pair.
        side : str
            ``"buy"`` or ``"sell"``.
        size_usd : float
            Total notional in USD.
        venue_books : dict
            Mapping of venue name → :class:`VenueBook`.

        Returns
        -------
        list of VenueOrder
            Ordered by size descending.  May contain a single venue if others
            are not routable.
        """
        side = side.lower()
        scored: List[Tuple[VenueRecommendation, float]] = []

        for venue_name, book in venue_books.items():
            rec = self._score_venue(venue_name, side, size_usd, book, symbol)
            if rec is None:
                continue
            depth = book.ask_depth_usd if side == "buy" else book.bid_depth_usd
            max_fill = depth * _MAX_DEPTH_UTILISATION
            scored.append((rec, max_fill))

        if not scored:
            logger.warning(
                "SmartOrderRouterV2: no venues for split — %s %s %.1f USD",
                symbol, side, size_usd,
            )
            return []

        # Inverse-cost weighting
        total_inv_cost = sum(
            1.0 / max(r.total_cost_bps, 0.1) for r, _ in scored
        )
        remaining = size_usd
        orders: List[VenueOrder] = []

        for rec, max_fill in scored:
            if remaining <= 0:
                break
            weight = (1.0 / max(rec.total_cost_bps, 0.1)) / total_inv_cost
            ideal = size_usd * weight
            allocated = min(ideal, max_fill, remaining)
            if allocated < 1.0:
                continue
            orders.append(
                VenueOrder(
                    venue=rec.venue,
                    size_usd=round(allocated, 2),
                    expected_price=rec.expected_price,
                    expected_slippage_bps=rec.expected_slippage_bps,
                    expected_fees_bps=rec.expected_fees_bps,
                    total_cost_bps=rec.total_cost_bps,
                )
            )
            remaining -= allocated

        # If there's remaining size (depth-limited), push overflow to cheapest
        if remaining > 1.0 and orders:
            orders[0] = VenueOrder(
                venue=orders[0].venue,
                size_usd=round(orders[0].size_usd + remaining, 2),
                expected_price=orders[0].expected_price,
                expected_slippage_bps=orders[0].expected_slippage_bps,
                expected_fees_bps=orders[0].expected_fees_bps,
                total_cost_bps=orders[0].total_cost_bps,
            )

        orders.sort(key=lambda o: -o.size_usd)
        logger.info(
            "SmartOrderRouterV2: split %s %s %.1f USD across %d venues: %s",
            symbol, side, size_usd, len(orders),
            [(o.venue, o.size_usd) for o in orders],
        )
        return orders

    def record_execution(
        self,
        venue: str,
        symbol: str,
        expected_cost_bps: float,
        actual_cost_bps: float,
        side: str = "buy",
        size_usd: float = 0.0,
    ) -> None:
        """Record an execution outcome for learning.

        Parameters
        ----------
        venue : str
            Exchange venue name.
        symbol : str
            Trading pair.
        expected_cost_bps : float
            The cost the router predicted.
        actual_cost_bps : float
            The realised cost from fill data.
        side : str
            ``"buy"`` or ``"sell"``.
        size_usd : float
            Notional filled.
        """
        bias = actual_cost_bps - expected_cost_bps
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute(
                    """
                    INSERT INTO execution_history
                        (ts, venue, symbol, side, size_usd, expected_bps, actual_bps, bias_bps)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        venue.lower(),
                        symbol,
                        side.lower(),
                        size_usd,
                        expected_cost_bps,
                        actual_cost_bps,
                        bias,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        logger.info(
            "SmartOrderRouterV2: recorded execution %s/%s — "
            "expected=%.1f bps, actual=%.1f bps, bias=%+.1f bps",
            venue, symbol, expected_cost_bps, actual_cost_bps, bias,
        )

    def get_venue_stats(self, venue: str, symbol: str, lookback: int = 100) -> Dict[str, Any]:
        """Return summary statistics for a venue/symbol pair.

        Returns
        -------
        dict
            Keys: ``count``, ``avg_expected_bps``, ``avg_actual_bps``,
            ``avg_bias_bps``, ``bias_std_bps``.
        """
        conn = sqlite3.connect(str(self._db_path))
        try:
            rows = conn.execute(
                """
                SELECT expected_bps, actual_bps, bias_bps
                FROM execution_history
                WHERE venue = ? AND symbol = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (venue.lower(), symbol, lookback),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "count": 0,
                "avg_expected_bps": 0.0,
                "avg_actual_bps": 0.0,
                "avg_bias_bps": 0.0,
                "bias_std_bps": 0.0,
            }

        expected = [r[0] for r in rows]
        actual = [r[1] for r in rows]
        biases = [r[2] for r in rows]
        n = len(rows)
        avg_bias = sum(biases) / n
        variance = sum((b - avg_bias) ** 2 for b in biases) / max(n - 1, 1)

        return {
            "count": n,
            "avg_expected_bps": round(sum(expected) / n, 2),
            "avg_actual_bps": round(sum(actual) / n, 2),
            "avg_bias_bps": round(avg_bias, 2),
            "bias_std_bps": round(math.sqrt(variance), 2),
        }
