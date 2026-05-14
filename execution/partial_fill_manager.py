"""
Intelligent Partial Fill Manager — decides what to do when a limit order is
only partially filled: wait, cancel-and-replace at a better price, or sweep
the remaining quantity at market.

Decision logic uses fill percentage, elapsed time, and urgency to choose
the optimal action.  Outcomes are persisted to SQLite for venue fill-rate
analytics.

Usage:
    manager = PartialFillManager()
    action = manager.on_partial_fill(
        order_id="abc-123", symbol="BTC/USD", side="buy",
        filled_qty=0.85, remaining_qty=0.15, fill_price=65010,
        elapsed_ms=2500,
    )
    if action.action == "cancel_replace":
        # re-submit at action.new_price
        ...
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DB_DIR = Path("data")
_DEFAULT_DB_PATH = _DB_DIR / "partial_fills.db"

# Decision thresholds (defaults — overridable via constructor)
_WAIT_FILL_PCT = 0.80          # >80% filled → likely to complete, just wait
_WAIT_MAX_ELAPSED_MS = 5_000   # within 5 seconds → still patient
_CANCEL_FILL_PCT = 0.50        # <50% filled → re-price
_CANCEL_MIN_ELAPSED_MS = 30_000  # after 30s → aggressive re-price
_SWEEP_URGENCY_THRESHOLD = 0.8  # urgency >= 0.8 → market sweep remainder


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PartialFillAction:
    """Recommended action for a partial fill."""
    action: str             # "wait" | "cancel_replace" | "market_sweep"
    new_price: Optional[float] = None   # suggested new limit price (cancel_replace)
    urgency: float = 0.0   # 0-1 urgency score
    reason: str = ""        # human-readable explanation


@dataclass
class _FillRecord:
    """Internal record of a partial fill event."""
    order_id: str
    symbol: str
    side: str
    filled_qty: float
    remaining_qty: float
    fill_price: float
    elapsed_ms: float
    action_taken: str
    exchange: Optional[str]
    timestamp: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PartialFillManager:
    """
    Intelligent handler for partial order fills.

    Parameters
    ----------
    db_path : str or Path
        SQLite persistence path (default ``data/partial_fills.db``).
    wait_fill_pct : float
        Fill fraction above which we simply wait (default 0.80).
    wait_max_elapsed_ms : float
        Maximum elapsed ms where waiting is still the default (default 5000).
    cancel_fill_pct : float
        Fill fraction below which we cancel and replace (default 0.50).
    cancel_min_elapsed_ms : float
        Elapsed ms after which we become more aggressive (default 30000).
    sweep_urgency_threshold : float
        Urgency value (0-1) above which we sweep remainder at market (default 0.8).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        wait_fill_pct: float = _WAIT_FILL_PCT,
        wait_max_elapsed_ms: float = _WAIT_MAX_ELAPSED_MS,
        cancel_fill_pct: float = _CANCEL_FILL_PCT,
        cancel_min_elapsed_ms: float = _CANCEL_MIN_ELAPSED_MS,
        sweep_urgency_threshold: float = _SWEEP_URGENCY_THRESHOLD,
    ) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._wait_fill_pct = wait_fill_pct
        self._wait_max_elapsed_ms = wait_max_elapsed_ms
        self._cancel_fill_pct = cancel_fill_pct
        self._cancel_min_elapsed_ms = cancel_min_elapsed_ms
        self._sweep_urgency = sweep_urgency_threshold
        self._lock = threading.Lock()

        self._init_db()
        logger.info(
            "PartialFillManager initialised  db=%s  wait_pct=%.2f  cancel_pct=%.2f  "
            "sweep_urgency=%.2f",
            self._db_path, wait_fill_pct, cancel_fill_pct, sweep_urgency_threshold,
        )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS partial_fills (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              REAL    NOT NULL,
                    order_id        TEXT    NOT NULL,
                    symbol          TEXT    NOT NULL,
                    side            TEXT    NOT NULL,
                    filled_qty      REAL    NOT NULL,
                    remaining_qty   REAL    NOT NULL,
                    fill_price      REAL    NOT NULL,
                    elapsed_ms      REAL    NOT NULL,
                    fill_pct        REAL    NOT NULL,
                    action_taken    TEXT    NOT NULL,
                    exchange        TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pf_symbol ON partial_fills(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pf_exchange ON partial_fills(exchange)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_partial_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        filled_qty: float,
        remaining_qty: float,
        fill_price: float,
        elapsed_ms: float,
        urgency: float = 0.5,
        exchange: Optional[str] = None,
    ) -> PartialFillAction:
        """
        Decide what to do with a partially filled order.

        Parameters
        ----------
        order_id : str
        symbol : str
        side : str          "buy" or "sell"
        filled_qty : float  Quantity already filled.
        remaining_qty : float  Quantity still outstanding.
        fill_price : float  Average fill price so far.
        elapsed_ms : float  Time since order was placed (milliseconds).
        urgency : float     0-1 urgency score (higher = more aggressive).
        exchange : str      Exchange name (optional, for venue analytics).

        Returns
        -------
        PartialFillAction
        """
        side = side.lower()
        total_qty = filled_qty + remaining_qty
        fill_pct = filled_qty / total_qty if total_qty > 0 else 0.0

        # Clamp urgency
        urgency = max(0.0, min(1.0, urgency))

        # Decision logic
        action = self._decide(fill_pct, elapsed_ms, urgency, side, fill_price)

        # Persist
        now = time.time()
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO partial_fills
                       (ts, order_id, symbol, side, filled_qty, remaining_qty,
                        fill_price, elapsed_ms, fill_pct, action_taken, exchange)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (now, order_id, symbol, side, filled_qty, remaining_qty,
                     fill_price, elapsed_ms, fill_pct, action.action, exchange),
                )
                conn.commit()

        logger.info(
            "Partial fill decision  order=%s  symbol=%s  side=%s  fill_pct=%.1f%%  "
            "elapsed=%dms  urgency=%.2f  action=%s  reason=%s",
            order_id, symbol, side, fill_pct * 100, elapsed_ms,
            urgency, action.action, action.reason,
        )
        return action

    def get_fill_rate_by_venue(self, exchange: str) -> float:
        """
        Return the percentage of orders fully filled (fill_pct > 0.99) at a venue.

        Parameters
        ----------
        exchange : str

        Returns
        -------
        float
            Fraction 0-1 of orders considered "fully filled".
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN fill_pct >= 0.99 THEN 1 ELSE 0 END) as full_fills
                   FROM partial_fills WHERE exchange = ?""",
                (exchange,),
            ).fetchone()

        if not row or row[0] == 0:
            return 0.0
        return round(row[1] / row[0], 4)

    def get_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Return summary statistics for partial fills.

        Returns
        -------
        dict
            total_events, avg_fill_pct, avg_elapsed_ms,
            action_counts (dict of action -> count).
        """
        query = "SELECT fill_pct, elapsed_ms, action_taken FROM partial_fills"
        params: list = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return {
                "total_events": 0,
                "avg_fill_pct": 0.0,
                "avg_elapsed_ms": 0.0,
                "action_counts": {},
            }

        fill_pcts = [r[0] for r in rows]
        elapsed = [r[1] for r in rows]
        actions = [r[2] for r in rows]

        action_counts: Dict[str, int] = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        return {
            "total_events": len(rows),
            "avg_fill_pct": round(sum(fill_pcts) / len(fill_pcts), 4),
            "avg_elapsed_ms": round(sum(elapsed) / len(elapsed), 1),
            "action_counts": action_counts,
        }

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        fill_pct: float,
        elapsed_ms: float,
        urgency: float,
        side: str,
        fill_price: float,
    ) -> PartialFillAction:
        """
        Core decision engine.

        Rules (evaluated in priority order):
        1. If urgency >= sweep_threshold → market sweep
        2. If fill_pct >= wait_pct AND elapsed < wait_max → wait
        3. If fill_pct < cancel_pct AND elapsed > cancel_min → cancel_replace
        4. If elapsed > cancel_min (regardless of fill_pct) → cancel_replace
        5. Otherwise → wait (give more time)
        """
        # Rule 1: High urgency — sweep immediately
        if urgency >= self._sweep_urgency:
            return PartialFillAction(
                action="market_sweep",
                urgency=urgency,
                reason=f"urgency {urgency:.2f} >= {self._sweep_urgency:.2f} threshold",
            )

        # Rule 2: Nearly complete and still early — wait
        if fill_pct >= self._wait_fill_pct and elapsed_ms <= self._wait_max_elapsed_ms:
            return PartialFillAction(
                action="wait",
                urgency=urgency,
                reason=f"fill {fill_pct:.0%} >= {self._wait_fill_pct:.0%} and "
                       f"elapsed {elapsed_ms:.0f}ms <= {self._wait_max_elapsed_ms:.0f}ms",
            )

        # Rule 3: Low fill after long time — cancel and replace
        if fill_pct < self._cancel_fill_pct and elapsed_ms > self._cancel_min_elapsed_ms:
            # Suggest a more aggressive price: move 2-5 bps towards market
            improvement_bps = 2.0 + urgency * 3.0
            if side == "buy":
                new_price = fill_price * (1.0 + improvement_bps / 10_000)
            else:
                new_price = fill_price * (1.0 - improvement_bps / 10_000)
            return PartialFillAction(
                action="cancel_replace",
                new_price=round(new_price, 8),
                urgency=urgency,
                reason=f"fill {fill_pct:.0%} < {self._cancel_fill_pct:.0%} and "
                       f"elapsed {elapsed_ms:.0f}ms > {self._cancel_min_elapsed_ms:.0f}ms "
                       f"— improved {improvement_bps:.1f}bps",
            )

        # Rule 4: Stale order regardless of fill percentage
        if elapsed_ms > self._cancel_min_elapsed_ms:
            improvement_bps = 1.0 + urgency * 2.0
            if side == "buy":
                new_price = fill_price * (1.0 + improvement_bps / 10_000)
            else:
                new_price = fill_price * (1.0 - improvement_bps / 10_000)
            return PartialFillAction(
                action="cancel_replace",
                new_price=round(new_price, 8),
                urgency=urgency,
                reason=f"elapsed {elapsed_ms:.0f}ms > {self._cancel_min_elapsed_ms:.0f}ms "
                       f"— refreshing with {improvement_bps:.1f}bps improvement",
            )

        # Rule 5: Default — wait
        return PartialFillAction(
            action="wait",
            urgency=urgency,
            reason=f"fill {fill_pct:.0%}, elapsed {elapsed_ms:.0f}ms — giving more time",
        )
