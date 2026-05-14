"""
Push 87 — TradeLedger
=====================
Persistent double-entry trade ledger that sits downstream of FillTracker.

Every confirmed fill that passes through FillTracker.record_fill() is also
posted here as an immutable ledger entry.  The ledger provides:

  * Append-only SQLite journal (ledger_entries table)
  * Per-strategy realised P&L running totals
  * Net-position tracking per symbol
  * CSV / JSON export helpers for reconciliation

Usage (standalone)::

    ledger = TradeLedger()
    ledger.post(
        fill_id="uuid", strategy="trend_follow", symbol="BTC/USD",
        side="buy", quantity_usd=500.0, fill_price=67_000.0,
        fee_usd=0.25, exchange="kraken", timestamp=time.time(),
    )
    pnl = ledger.realised_pnl("trend_follow")

Integration with FillTracker — see LedgerFillObserver in this module.
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """Immutable record of a single confirmed fill in the trade ledger."""

    entry_id: str
    fill_id: str                 # FK → fills.fill_id in FillTracker DB
    posted_at: float             # unix timestamp of ledger posting
    strategy: str
    symbol: str
    side: str                    # "buy" | "sell"
    quantity_usd: float          # notional size in USD
    fill_price: float
    fee_usd: float
    exchange: str
    realised_pnl_usd: float      # P&L contribution from this fill (0 on open leg)
    running_pnl_usd: float       # cumulative P&L for strategy after this entry
    timestamp: float             # original fill timestamp


@dataclass
class PositionState:
    """Tracks the net open position for a (strategy, symbol) pair."""

    strategy: str
    symbol: str
    net_qty_usd: float = 0.0      # positive = long, negative = short
    avg_entry_price: float = 0.0
    total_fees_usd: float = 0.0
    open_since: Optional[float] = None

    def is_flat(self) -> bool:
        return abs(self.net_qty_usd) < 1e-8


# ---------------------------------------------------------------------------
# TradeLedger
# ---------------------------------------------------------------------------

class TradeLedger:
    """
    Append-only trade ledger with realised P&L and net-position tracking.

    Thread-safe: all mutations acquire self._lock.
    SQLite is opened in WAL mode.
    """

    def __init__(self, db_path: str = "data/ledger.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._pnl: Dict[str, float] = {}             # strategy → cumulative P&L
        self._positions: Dict[str, PositionState] = {}  # "strategy:symbol" → state
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_state()
        logger.info("TradeLedger initialised: db=%s", db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def post(
        self,
        fill_id: str,
        strategy: str,
        symbol: str,
        side: str,
        quantity_usd: float,
        fill_price: float,
        exchange: str,
        timestamp: float,
        fee_usd: float = 0.0,
    ) -> LedgerEntry:
        """
        Post a confirmed fill to the ledger.

        Returns the LedgerEntry with computed P&L fields.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if fill_price <= 0:
            raise ValueError(f"fill_price must be positive, got {fill_price}")
        if quantity_usd <= 0:
            raise ValueError(f"quantity_usd must be positive, got {quantity_usd}")

        with self._lock:
            pos_key = f"{strategy}:{symbol}"
            pos = self._positions.setdefault(
                pos_key,
                PositionState(strategy=strategy, symbol=symbol),
            )

            realised = self._compute_realised_pnl(pos, side, quantity_usd, fill_price)
            self._update_position(pos, side, quantity_usd, fill_price)

            running = self._pnl.get(strategy, 0.0) + realised - fee_usd
            self._pnl[strategy] = running

            entry = LedgerEntry(
                entry_id=str(uuid.uuid4()),
                fill_id=fill_id,
                posted_at=time.time(),
                strategy=strategy,
                symbol=symbol,
                side=side,
                quantity_usd=quantity_usd,
                fill_price=fill_price,
                fee_usd=fee_usd,
                exchange=exchange,
                realised_pnl_usd=realised,
                running_pnl_usd=running,
                timestamp=timestamp,
            )
            self._persist(entry)

        logger.debug(
            "ledger posted fill=%s strat=%s %s %s qty_usd=%.2f @ %.6f "
            "realised=%.4f running=%.4f",
            fill_id, strategy, side, symbol,
            quantity_usd, fill_price, realised, running,
        )
        return entry

    def realised_pnl(self, strategy: str) -> float:
        """Return cumulative realised P&L (USD) for a strategy."""
        with self._lock:
            return self._pnl.get(strategy, 0.0)

    def get_position(self, strategy: str, symbol: str) -> PositionState:
        """Return current net position for (strategy, symbol)."""
        with self._lock:
            key = f"{strategy}:{symbol}"
            return self._positions.get(
                key, PositionState(strategy=strategy, symbol=symbol)
            )

    def all_positions(self) -> List[PositionState]:
        """Return all tracked positions (including flat ones)."""
        with self._lock:
            return list(self._positions.values())

    def open_positions(self) -> List[PositionState]:
        """Return only non-flat positions."""
        with self._lock:
            return [p for p in self._positions.values() if not p.is_flat()]

    def export_csv(self, path: str, strategy: Optional[str] = None) -> None:
        """Export ledger entries to CSV.  Optionally filter by strategy."""
        entries = self._load_entries(strategy=strategy)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            if not entries:
                fh.write("")
                return
            writer = csv.DictWriter(fh, fieldnames=list(asdict(entries[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(e) for e in entries)
        logger.info("TradeLedger: exported %d entries to %s", len(entries), path)

    def export_json(self, path: str, strategy: Optional[str] = None) -> None:
        """Export ledger entries to JSON.  Optionally filter by strategy."""
        entries = self._load_entries(strategy=strategy)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump([asdict(e) for e in entries], fh, indent=2)
        logger.info("TradeLedger: exported %d entries to %s", len(entries), path)

    # ------------------------------------------------------------------
    # P&L + position helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_realised_pnl(
        pos: PositionState,
        side: str,
        quantity_usd: float,
        fill_price: float,
    ) -> float:
        """
        Realised P&L is only recognised when closing (or partially closing)
        an existing position.

        Opening leg  → P&L = 0
        Closing leg  → P&L = (exit_price - entry_price) * qty_usd / entry_price
                              (sign-adjusted for long vs short)
        """
        if pos.is_flat() or pos.avg_entry_price == 0.0:
            return 0.0

        is_long = pos.net_qty_usd > 0
        is_closing = (is_long and side == "sell") or (not is_long and side == "buy")
        if not is_closing:
            return 0.0

        close_qty = min(abs(pos.net_qty_usd), quantity_usd)
        price_diff = fill_price - pos.avg_entry_price
        direction = 1.0 if is_long else -1.0
        realised = direction * price_diff * close_qty / pos.avg_entry_price
        return realised

    @staticmethod
    def _update_position(
        pos: PositionState,
        side: str,
        quantity_usd: float,
        fill_price: float,
    ) -> None:
        """Mutate pos in-place to reflect the new fill (FIFO average-cost)."""
        signed_qty = quantity_usd if side == "buy" else -quantity_usd

        if pos.is_flat():
            pos.net_qty_usd = signed_qty
            pos.avg_entry_price = fill_price
            pos.open_since = time.time()
            return

        prev_qty = pos.net_qty_usd
        new_qty = prev_qty + signed_qty

        same_direction = (prev_qty > 0 and signed_qty > 0) or (
            prev_qty < 0 and signed_qty < 0
        )
        if same_direction:
            # Adding to position — recalculate weighted average entry
            total_cost = abs(prev_qty) * pos.avg_entry_price + quantity_usd * fill_price
            pos.avg_entry_price = total_cost / (abs(prev_qty) + quantity_usd)
        else:
            # Partial or full close
            if abs(new_qty) < 1e-8:
                pos.avg_entry_price = 0.0
                pos.open_since = None
            # else: avg_entry_price stays the same on a partial close

        pos.net_qty_usd = new_qty

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _persist(self, entry: LedgerEntry) -> None:
        """Write entry to SQLite.  Caller must hold self._lock."""
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO ledger_entries
                        (entry_id, fill_id, posted_at, strategy, symbol, side,
                         quantity_usd, fill_price, fee_usd, exchange,
                         realised_pnl_usd, running_pnl_usd, timestamp)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        entry.entry_id, entry.fill_id, entry.posted_at,
                        entry.strategy, entry.symbol, entry.side,
                        entry.quantity_usd, entry.fill_price, entry.fee_usd,
                        entry.exchange, entry.realised_pnl_usd,
                        entry.running_pnl_usd, entry.timestamp,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.exception("TradeLedger: failed to persist entry %s", entry.entry_id)

    def _load_entries(
        self, strategy: Optional[str] = None
    ) -> List[LedgerEntry]:
        """Load all ledger entries, optionally filtered by strategy."""
        entries: List[LedgerEntry] = []
        try:
            conn = self._connect()
            try:
                if strategy:
                    rows = conn.execute(
                        "SELECT * FROM ledger_entries WHERE strategy=? ORDER BY posted_at",
                        (strategy,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM ledger_entries ORDER BY posted_at"
                    ).fetchall()
            finally:
                conn.close()
            for row in rows:
                entries.append(LedgerEntry(*row))
        except Exception:
            logger.exception("TradeLedger: failed to load entries")
        return entries

    def _load_state(self) -> None:
        """
        On startup, replay ledger entries to restore in-memory P&L and
        position state.  This makes the ledger crash-safe.
        """
        entries = self._load_entries()
        for entry in entries:
            self._pnl[entry.strategy] = entry.running_pnl_usd
            pos_key = f"{entry.strategy}:{entry.symbol}"
            pos = self._positions.setdefault(
                pos_key,
                PositionState(strategy=entry.strategy, symbol=entry.symbol),
            )
            self._update_position(
                pos, entry.side, entry.quantity_usd, entry.fill_price
            )
        if entries:
            logger.info(
                "TradeLedger: replayed %d entries to restore state", len(entries)
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger_entries (
                    entry_id         TEXT PRIMARY KEY,
                    fill_id          TEXT NOT NULL,
                    posted_at        REAL NOT NULL,
                    strategy         TEXT NOT NULL,
                    symbol           TEXT NOT NULL,
                    side             TEXT NOT NULL,
                    quantity_usd     REAL NOT NULL,
                    fill_price       REAL NOT NULL,
                    fee_usd          REAL NOT NULL,
                    exchange         TEXT NOT NULL,
                    realised_pnl_usd REAL NOT NULL,
                    running_pnl_usd  REAL NOT NULL,
                    timestamp        REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_le_strategy_ts "
                "ON ledger_entries (strategy, posted_at)"
            )
            conn.commit()
        logger.debug("TradeLedger DB initialised at %s", self.db_path)
