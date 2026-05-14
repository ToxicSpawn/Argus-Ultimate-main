"""
Position Registry — single source of truth for all open positions across strategies.

Solves the double-exposure problem: when TrendFollowing AND StatArb both want
to buy BTC, the registry prevents the combined exposure from exceeding limits.

Features:
  - Tracks positions per (symbol, strategy) pair
  - Computes net exposure per symbol across all strategies
  - Enforces max_exposure_per_symbol and max_total_exposure limits
  - Persists to SQLite for restart recovery
  - Thread-safe via threading.RLock

Usage:
    registry = PositionRegistry(max_exposure_per_symbol_usd=500.0)

    # Before opening a trade:
    ok, reason = registry.can_open("BTC/USD", "trend_follow", side="buy", usd=100.0)
    if not ok:
        logger.warning("Blocked: %s", reason)

    # Record the open:
    pos_id = registry.open_position("BTC/USD", "trend_follow", side="buy",
                                     price=65000.0, quantity=0.001, usd=65.0)

    # Record the close:
    registry.close_position(pos_id, exit_price=66000.0)

    # Query:
    net = registry.net_exposure("BTC/USD")  # signed USD exposure
    total = registry.total_exposure_usd()   # sum of |exposure| across all symbols
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """A single tracked position (open or closed)."""

    position_id: str  # uuid4
    symbol: str
    strategy: str
    side: str  # "buy" or "sell"
    entry_price: float
    quantity: float
    usd_value: float
    open_time: float
    close_time: Optional[float] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    status: str = "open"  # "open", "closed", "partially_closed"
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.side = self.side.lower()
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if self.usd_value < 0:
            raise ValueError("usd_value must be non-negative")

    @property
    def signed_exposure(self) -> float:
        """Signed USD exposure: positive for buy, negative for sell."""
        return self.usd_value if self.side == "buy" else -self.usd_value


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class PositionRegistry:
    """
    Thread-safe, SQLite-backed position registry.

    Uses threading.RLock (reentrant) so that can_open() and open_position()
    may be called from the same thread in sequence without deadlock.
    """

    def __init__(
        self,
        max_exposure_per_symbol_usd: float = 500.0,
        max_total_exposure_usd: float = 800.0,
        max_positions_per_strategy: int = 3,
        max_total_positions: int = 10,
        db_path: str = "data/position_registry.db",
        persist: bool = True,
    ) -> None:
        self.max_exposure_per_symbol_usd = float(max_exposure_per_symbol_usd)
        self.max_total_exposure_usd = float(max_total_exposure_usd)
        self.max_positions_per_strategy = int(max_positions_per_strategy)
        self.max_total_positions = int(max_total_positions)
        self.db_path = db_path
        self.persist = persist

        self._lock = threading.RLock()
        # In-memory store: position_id -> Position (open only)
        self._open: Dict[str, Position] = {}

        if self.persist:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._load_open_positions()

        logger.info(
            "PositionRegistry initialized: max_symbol=%.0f max_total=%.0f "
            "max_per_strategy=%d max_total=%d persist=%s db=%s",
            self.max_exposure_per_symbol_usd,
            self.max_total_exposure_usd,
            self.max_positions_per_strategy,
            self.max_total_positions,
            self.persist,
            db_path,
        )

    # ------------------------------------------------------------------
    # Exposure checks
    # ------------------------------------------------------------------

    def can_open(
        self, symbol: str, strategy: str, side: str, usd: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened. Returns (allowed, reason).

        Checks:
          1. Symbol gross exposure + usd <= max_exposure_per_symbol_usd
          2. Total gross exposure + usd <= max_total_exposure_usd
          3. Strategy open positions < max_positions_per_strategy
          4. Total open positions < max_total_positions
          5. No opposite-side position already open for same (symbol, strategy)
        """
        side = side.lower()
        usd = float(usd)

        with self._lock:
            # 1. Symbol gross exposure
            sym_gross = self.gross_exposure(symbol)
            if sym_gross + usd > self.max_exposure_per_symbol_usd:
                return (
                    False,
                    f"Symbol exposure limit: {sym_gross + usd:.2f} > "
                    f"{self.max_exposure_per_symbol_usd:.2f} USD for {symbol}",
                )

            # 2. Total gross exposure
            total_gross = self.total_exposure_usd()
            if total_gross + usd > self.max_total_exposure_usd:
                return (
                    False,
                    f"Total exposure limit: {total_gross + usd:.2f} > "
                    f"{self.max_total_exposure_usd:.2f} USD",
                )

            # 3. Strategy position count
            strategy_positions = [
                p for p in self._open.values() if p.strategy == strategy
            ]
            if len(strategy_positions) >= self.max_positions_per_strategy:
                return (
                    False,
                    f"Strategy position limit: {strategy} already has "
                    f"{len(strategy_positions)} open positions "
                    f"(max {self.max_positions_per_strategy})",
                )

            # 4. Total position count
            if len(self._open) >= self.max_total_positions:
                return (
                    False,
                    f"Total position limit: {len(self._open)} open positions "
                    f"(max {self.max_total_positions})",
                )

            # 5. Opposite-side conflict for same (symbol, strategy)
            opposite = "sell" if side == "buy" else "buy"
            for p in self._open.values():
                if p.symbol == symbol and p.strategy == strategy and p.side == opposite:
                    return (
                        False,
                        f"Opposite-side conflict: {strategy} already has a {opposite} "
                        f"position in {symbol} (id={p.position_id})",
                    )

            return True, "ok"

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        strategy: str,
        side: str,
        price: float,
        quantity: float,
        usd: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Record a new open position. Returns position_id."""
        pos = Position(
            position_id=str(uuid.uuid4()),
            symbol=symbol,
            strategy=strategy,
            side=side,
            entry_price=float(price),
            quantity=float(quantity),
            usd_value=float(usd),
            open_time=time.time(),
            tags=tags or {},
        )

        with self._lock:
            self._open[pos.position_id] = pos
            if self.persist:
                self._persist_open(pos)

        logger.info(
            "Position opened: id=%s %s %s %s qty=%.6f @ %.4f usd=%.2f",
            pos.position_id,
            strategy,
            symbol,
            side,
            quantity,
            price,
            usd,
        )
        return pos.position_id

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        quantity: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Close (or partially close) a position. Returns the updated Position.

        If quantity is None or >= position quantity, the position is fully closed.
        If quantity < position quantity, the position becomes partially_closed and
        a new closed record is written for the partial amount.
        """
        with self._lock:
            pos = self._open.get(position_id)
            if pos is None:
                logger.warning("close_position: id=%s not found in open positions", position_id)
                return None

            exit_price = float(exit_price)
            close_qty = float(quantity) if quantity is not None else pos.quantity

            if close_qty >= pos.quantity:
                # Full close
                pnl = _calc_pnl(pos.side, pos.entry_price, exit_price, pos.quantity)
                pos.exit_price = exit_price
                pos.close_time = time.time()
                pos.realized_pnl = pnl
                pos.status = "closed"
                del self._open[position_id]
                if self.persist:
                    self._persist_close(pos)
                logger.info(
                    "Position closed: id=%s %s %s pnl=%.4f",
                    position_id,
                    pos.symbol,
                    pos.strategy,
                    pnl,
                )
                return pos
            else:
                # Partial close — write a synthetic closed record, reduce open qty
                pnl = _calc_pnl(pos.side, pos.entry_price, exit_price, close_qty)
                partial_usd = pos.usd_value * (close_qty / pos.quantity)
                partial = Position(
                    position_id=str(uuid.uuid4()),
                    symbol=pos.symbol,
                    strategy=pos.strategy,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    quantity=close_qty,
                    usd_value=partial_usd,
                    open_time=pos.open_time,
                    close_time=time.time(),
                    exit_price=exit_price,
                    realized_pnl=pnl,
                    status="closed",
                    tags=dict(pos.tags, partial_of=position_id),
                )
                # Reduce the original position
                pos.quantity -= close_qty
                pos.usd_value -= partial_usd
                pos.status = "partially_closed"

                if self.persist:
                    self._persist_open(partial)  # insert the partial record
                    self._persist_close(partial)  # immediately mark it closed
                    self._update_open_in_db(pos)  # update remaining open qty

                logger.info(
                    "Partial close: id=%s %s qty=%.6f pnl=%.4f remaining=%.6f",
                    position_id,
                    pos.symbol,
                    close_qty,
                    pnl,
                    pos.quantity,
                )
                return partial

    # ------------------------------------------------------------------
    # Mark / price updates
    # ------------------------------------------------------------------

    def update_mark(self, symbol: str, mark_price: float) -> None:
        """Update mark price for all open positions in a symbol (stored in tags)."""
        mark_price = float(mark_price)
        with self._lock:
            for pos in self._open.values():
                if pos.symbol == symbol:
                    pos.tags["mark_price"] = str(mark_price)

    # ------------------------------------------------------------------
    # Exposure queries
    # ------------------------------------------------------------------

    def net_exposure(self, symbol: str) -> float:
        """Net signed USD exposure for a symbol (buy=positive, sell=negative)."""
        with self._lock:
            return sum(
                p.signed_exposure for p in self._open.values() if p.symbol == symbol
            )

    def gross_exposure(self, symbol: str) -> float:
        """Gross USD exposure for a symbol (sum of absolute values)."""
        with self._lock:
            return sum(
                p.usd_value for p in self._open.values() if p.symbol == symbol
            )

    def total_exposure_usd(self) -> float:
        """Total gross USD exposure across all open positions."""
        with self._lock:
            return sum(p.usd_value for p in self._open.values())

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> List[Position]:
        """Query open positions with optional filters."""
        with self._lock:
            results = list(self._open.values())
        if symbol is not None:
            results = [p for p in results if p.symbol == symbol]
        if strategy is not None:
            results = [p for p in results if p.strategy == strategy]
        return results

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific open position by ID."""
        with self._lock:
            return self._open.get(position_id)

    def get_strategy_exposure(self) -> Dict[str, float]:
        """Gross USD exposure per strategy."""
        out: Dict[str, float] = {}
        with self._lock:
            for p in self._open.values():
                out[p.strategy] = out.get(p.strategy, 0.0) + p.usd_value
        return out

    def get_symbol_exposure(self) -> Dict[str, float]:
        """Net USD exposure per symbol."""
        out: Dict[str, float] = {}
        with self._lock:
            for p in self._open.values():
                out[p.symbol] = out.get(p.symbol, 0.0) + p.signed_exposure
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Full state snapshot for monitoring/dashboards."""
        with self._lock:
            positions_list = [
                {
                    "position_id": p.position_id,
                    "symbol": p.symbol,
                    "strategy": p.strategy,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "usd_value": p.usd_value,
                    "open_time": p.open_time,
                    "status": p.status,
                    "tags": p.tags,
                }
                for p in self._open.values()
            ]
            return {
                "timestamp": time.time(),
                "open_count": len(self._open),
                "total_exposure_usd": self.total_exposure_usd(),
                "symbol_exposure": self.get_symbol_exposure(),
                "strategy_exposure": self.get_strategy_exposure(),
                "max_exposure_per_symbol_usd": self.max_exposure_per_symbol_usd,
                "max_total_exposure_usd": self.max_total_exposure_usd,
                "positions": positions_list,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _init_db(self) -> None:
        """Create tables if not exists."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol      TEXT NOT NULL,
                    strategy    TEXT NOT NULL,
                    side        TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity    REAL NOT NULL,
                    usd_value   REAL NOT NULL,
                    open_time   REAL NOT NULL,
                    close_time  REAL,
                    exit_price  REAL,
                    realized_pnl REAL,
                    status      TEXT NOT NULL DEFAULT 'open',
                    tags_json   TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pos_symbol   ON positions(symbol)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pos_strategy ON positions(strategy)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pos_status   ON positions(status)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pos_open_time ON positions(open_time)"
            )
            conn.commit()

    def _persist_open(self, pos: Position) -> None:
        """Write new position to SQLite."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions
                        (position_id, symbol, strategy, side, entry_price, quantity,
                         usd_value, open_time, close_time, exit_price, realized_pnl,
                         status, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pos.position_id,
                        pos.symbol,
                        pos.strategy,
                        pos.side,
                        pos.entry_price,
                        pos.quantity,
                        pos.usd_value,
                        pos.open_time,
                        pos.close_time,
                        pos.exit_price,
                        pos.realized_pnl,
                        pos.status,
                        json.dumps(pos.tags),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("_persist_open failed for %s: %s", pos.position_id, exc)

    def _persist_close(self, pos: Position) -> None:
        """Update closed position in SQLite."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE positions
                    SET close_time=?, exit_price=?, realized_pnl=?, status=?
                    WHERE position_id=?
                    """,
                    (
                        pos.close_time,
                        pos.exit_price,
                        pos.realized_pnl,
                        pos.status,
                        pos.position_id,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("_persist_close failed for %s: %s", pos.position_id, exc)

    def _update_open_in_db(self, pos: Position) -> None:
        """Update quantity/usd_value/status for a partially closed position."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE positions
                    SET quantity=?, usd_value=?, status=?, tags_json=?
                    WHERE position_id=?
                    """,
                    (
                        pos.quantity,
                        pos.usd_value,
                        pos.status,
                        json.dumps(pos.tags),
                        pos.position_id,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error("_update_open_in_db failed for %s: %s", pos.position_id, exc)

    def _load_open_positions(self) -> None:
        """Restore open positions from SQLite on startup."""
        db_path = Path(self.db_path)
        if not db_path.exists():
            return

        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM positions WHERE status IN ('open', 'partially_closed')"
                )
                rows = cur.fetchall()

            loaded = 0
            for row in rows:
                try:
                    tags: Dict[str, str] = {}
                    try:
                        tags = json.loads(row["tags_json"] or "{}")
                    except Exception as _e:
                        logger.debug("position_registry error: %s", _e)

                    pos = Position(
                        position_id=str(row["position_id"]),
                        symbol=str(row["symbol"]),
                        strategy=str(row["strategy"]),
                        side=str(row["side"]),
                        entry_price=float(row["entry_price"]),
                        quantity=float(row["quantity"]),
                        usd_value=float(row["usd_value"]),
                        open_time=float(row["open_time"]),
                        close_time=float(row["close_time"]) if row["close_time"] is not None else None,
                        exit_price=float(row["exit_price"]) if row["exit_price"] is not None else None,
                        realized_pnl=float(row["realized_pnl"]) if row["realized_pnl"] is not None else None,
                        status=str(row["status"]),
                        tags=tags,
                    )
                    self._open[pos.position_id] = pos
                    loaded += 1
                except Exception as row_exc:
                    logger.warning(
                        "Skipping malformed position row id=%s: %s",
                        row["position_id"] if "position_id" in row.keys() else "?",
                        row_exc,
                    )

            if loaded:
                logger.info(
                    "Restored %d open position(s) from %s (total_exposure=%.2f USD)",
                    loaded,
                    self.db_path,
                    self.total_exposure_usd(),
                )

        except Exception as exc:
            logger.error("_load_open_positions failed: %s", exc)


# ---------------------------------------------------------------------------
# PnL helper
# ---------------------------------------------------------------------------


def _calc_pnl(side: str, entry: float, exit_price: float, qty: float) -> float:
    """Compute realized PnL for a position close."""
    if side == "buy":
        return (exit_price - entry) * qty
    else:  # sell / short
        return (entry - exit_price) * qty


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_registry: Optional[PositionRegistry] = None
_registry_lock = threading.Lock()


def get_registry(**kwargs: Any) -> PositionRegistry:
    """Get or create the default global registry."""
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = PositionRegistry(**kwargs)
    return _default_registry
