"""
Fill Tracker & Slippage Budget — tracks actual vs expected fill prices per strategy.

Every order that goes through ARGUS should call record_fill() with both the
expected price (from MarketImpactModel.adjust_price()) and the actual fill.
This module accumulates slippage statistics and enforces a daily per-strategy
slippage budget — when a strategy has burned its budget, it is paused until
reset (daily at midnight UTC).

Usage:
    tracker = FillTracker()

    # Before placing order:
    expected = impact_model.adjust_price(mid, "buy", qty_usd)

    # After fill confirmed:
    tracker.record_fill(
        strategy="trend_follow", symbol="BTC/USD", side="buy",
        expected_price=expected, actual_price=fill_price,
        quantity_usd=qty_usd, exchange="kraken"
    )

    # Check if strategy is within budget:
    if not tracker.is_within_budget("trend_follow"):
        logger.warning("trend_follow slippage budget exhausted — pausing")
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

_MIDNIGHT_UTC_SECONDS = 86400.0  # seconds in a day


@dataclass
class FillRecord:
    """Single completed fill with slippage attribution."""

    fill_id: str
    timestamp: float
    strategy: str
    symbol: str
    side: str                  # "buy" or "sell"
    expected_price: float
    actual_price: float
    quantity_usd: float
    exchange: str
    slippage_bps: float        # positive = worse than expected
    slippage_usd: float        # positive = money lost to slippage


@dataclass
class SlippageBudget:
    """Per-strategy daily slippage allowance."""

    strategy: str
    daily_limit_bps: float = 20.0   # 20 bps total allowed slippage per day
    daily_limit_usd: float = 50.0   # $50 max slippage per day
    used_bps: float = 0.0
    used_usd: float = 0.0
    last_reset: float = field(default_factory=time.time)
    paused: bool = False

    @property
    def remaining_bps(self) -> float:
        return max(0.0, self.daily_limit_bps - self.used_bps)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.daily_limit_usd - self.used_usd)

    def within_budget(self) -> bool:
        return not self.paused and self.used_bps < self.daily_limit_bps and self.used_usd < self.daily_limit_usd


# ---------------------------------------------------------------------------
# FillTracker
# ---------------------------------------------------------------------------

class FillTracker:
    """
    Records fills, computes slippage vs expected, and enforces per-strategy
    daily slippage budgets.

    Thread-safe: all public methods acquire self._lock before any mutation.
    SQLite is opened in WAL mode for concurrent read access.
    """

    def __init__(
        self,
        db_path: str = "data/fills.db",
        daily_limit_bps: float = 20.0,
        daily_limit_usd: float = 50.0,
    ) -> None:
        self.db_path = db_path
        self.daily_limit_bps = daily_limit_bps
        self.daily_limit_usd = daily_limit_usd

        self._lock = threading.Lock()
        self._budgets: Dict[str, SlippageBudget] = {}

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("FillTracker initialised: db=%s bps_limit=%.1f usd_limit=%.2f",
                    db_path, daily_limit_bps, daily_limit_usd)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_fill(
        self,
        strategy: str,
        symbol: str,
        side: str,
        expected_price: float,
        actual_price: float,
        quantity_usd: float,
        exchange: str = "kraken",
    ) -> FillRecord:
        """
        Record a completed fill and update the strategy's slippage budget.

        Slippage convention (positive = adverse / money lost):
          buy  side: actual > expected is bad  (+)
          sell side: actual < expected is bad  (+)

        Returns the FillRecord with computed slippage fields.
        """
        if expected_price <= 0:
            raise ValueError(f"expected_price must be positive, got {expected_price}")
        if actual_price <= 0:
            raise ValueError(f"actual_price must be positive, got {actual_price}")
        if quantity_usd <= 0:
            raise ValueError(f"quantity_usd must be positive, got {quantity_usd}")

        slippage_bps, slippage_usd = self._compute_slippage_bps(
            side, expected_price, actual_price, quantity_usd
        )

        record = FillRecord(
            fill_id=str(uuid.uuid4()),
            timestamp=time.time(),
            strategy=strategy,
            symbol=symbol,
            side=side.lower(),
            expected_price=expected_price,
            actual_price=actual_price,
            quantity_usd=quantity_usd,
            exchange=exchange,
            slippage_bps=slippage_bps,
            slippage_usd=slippage_usd,
        )

        with self._lock:
            budget = self._get_or_create_budget(strategy)
            # Only accumulate adverse slippage against the budget
            if slippage_bps > 0:
                budget.used_bps += slippage_bps
                budget.used_usd += slippage_usd
                if not budget.within_budget():
                    if not budget.paused:
                        logger.warning(
                            "Strategy %s slippage budget exhausted "
                            "(used=%.2f bps / $%.2f) — pausing",
                            strategy, budget.used_bps, budget.used_usd,
                        )
                    budget.paused = True
            self._persist(record)

        logger.debug(
            "fill recorded strategy=%s symbol=%s side=%s "
            "expected=%.6f actual=%.6f slip=%.2f bps ($%.4f)",
            strategy, symbol, side,
            expected_price, actual_price, slippage_bps, slippage_usd,
        )
        return record

    def is_within_budget(self, strategy: str) -> bool:
        """True if strategy has remaining slippage budget for today."""
        with self._lock:
            budget = self._get_or_create_budget(strategy)
            return budget.within_budget()

    def get_budget(self, strategy: str) -> SlippageBudget:
        """Get (or create) budget for a strategy. Auto-resets if a new UTC day."""
        with self._lock:
            return self._get_or_create_budget(strategy)

    def reset_budgets(self) -> None:
        """
        Reset all strategy budgets.  Call at midnight UTC (or from a scheduler).
        """
        now = time.time()
        with self._lock:
            for budget in self._budgets.values():
                budget.used_bps = 0.0
                budget.used_usd = 0.0
                budget.paused = False
                budget.last_reset = now
            logger.info("Slippage budgets reset for %d strategies", len(self._budgets))

    def get_strategy_stats(
        self, strategy: str, lookback_hours: float = 24
    ) -> Dict[str, float]:
        """
        Compute slippage statistics for a single strategy over the past
        ``lookback_hours`` hours.

        Returns a dict with keys:
            avg_slippage_bps, total_slippage_bps, total_slippage_usd,
            fill_count, worst_fill_bps, best_fill_bps
        """
        since_ts = time.time() - lookback_hours * 3600.0
        with self._lock:
            fills = self._load_fills(strategy, since_ts)

        if not fills:
            return {
                "avg_slippage_bps": 0.0,
                "total_slippage_bps": 0.0,
                "total_slippage_usd": 0.0,
                "fill_count": 0,
                "worst_fill_bps": 0.0,
                "best_fill_bps": 0.0,
            }

        slippages = [f.slippage_bps for f in fills]
        return {
            "avg_slippage_bps": sum(slippages) / len(slippages),
            "total_slippage_bps": sum(slippages),
            "total_slippage_usd": sum(f.slippage_usd for f in fills),
            "fill_count": len(fills),
            "worst_fill_bps": max(slippages),
            "best_fill_bps": min(slippages),
        }

    def get_all_stats(self, lookback_hours: float = 24) -> Dict[str, Dict]:
        """
        Stats for all strategies that have fills in the lookback window.
        Returns a mapping of strategy name → stats dict.
        """
        since_ts = time.time() - lookback_hours * 3600.0
        strategies: set = set()

        with self._lock:
            strategies = set(self._budgets.keys())
            # Also discover any strategies that exist in the DB but not in memory
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        "SELECT DISTINCT strategy FROM fills WHERE timestamp >= ?",
                        (since_ts,),
                    ).fetchall()
                    strategies.update(row[0] for row in rows)
                finally:
                    conn.close()
            except Exception:
                logger.exception("get_all_stats: failed to query strategies from DB")

        return {
            strategy: self.get_strategy_stats(strategy, lookback_hours)
            for strategy in sorted(strategies)
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_slippage_bps(
        self,
        side: str,
        expected: float,
        actual: float,
        qty_usd: float,
    ) -> Tuple[float, float]:
        """
        Compute (slippage_bps, slippage_usd).

        Sign convention — positive means adverse (cost to the strategy):
          buy:  (actual - expected) / expected * 10000   (paid more → positive)
          sell: (expected - actual) / expected * 10000   (received less → positive)
        """
        if side.lower() == "buy":
            slippage_bps = (actual - expected) / expected * 10_000.0
        else:
            slippage_bps = (expected - actual) / expected * 10_000.0

        # Dollar cost: fraction of notional value
        slippage_usd = qty_usd * slippage_bps / 10_000.0
        return slippage_bps, slippage_usd

    def _get_or_create_budget(self, strategy: str) -> SlippageBudget:
        """
        Return existing budget, creating a fresh one if absent.
        Also auto-resets if a new UTC day has started since last_reset.
        Caller must hold self._lock.
        """
        now = time.time()
        if strategy not in self._budgets:
            self._budgets[strategy] = SlippageBudget(
                strategy=strategy,
                daily_limit_bps=self.daily_limit_bps,
                daily_limit_usd=self.daily_limit_usd,
                last_reset=now,
            )
            return self._budgets[strategy]

        budget = self._budgets[strategy]
        # Determine the UTC midnight that separates last_reset from now
        # If now is in a different UTC day, reset
        last_day = int(budget.last_reset // _MIDNIGHT_UTC_SECONDS)
        current_day = int(now // _MIDNIGHT_UTC_SECONDS)
        if current_day > last_day:
            logger.info(
                "Auto-resetting slippage budget for strategy %s (new UTC day)", strategy
            )
            budget.used_bps = 0.0
            budget.used_usd = 0.0
            budget.paused = False
            budget.last_reset = now

        return budget

    def _persist(self, record: FillRecord) -> None:
        """
        Write a FillRecord to the SQLite fills table.
        Caller must hold self._lock (connection is opened and closed inline).
        """
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO fills
                        (fill_id, timestamp, strategy, symbol, side,
                         expected_price, actual_price, quantity_usd, exchange,
                         slippage_bps, slippage_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.fill_id,
                        record.timestamp,
                        record.strategy,
                        record.symbol,
                        record.side,
                        record.expected_price,
                        record.actual_price,
                        record.quantity_usd,
                        record.exchange,
                        record.slippage_bps,
                        record.slippage_usd,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.exception("FillTracker: failed to persist fill %s", record.fill_id)

    def _load_fills(self, strategy: str, since_ts: float) -> List[FillRecord]:
        """
        Load fills from the DB for a given strategy since ``since_ts``.
        Caller must hold self._lock.
        """
        fills: List[FillRecord] = []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT fill_id, timestamp, strategy, symbol, side,
                           expected_price, actual_price, quantity_usd, exchange,
                           slippage_bps, slippage_usd
                    FROM fills
                    WHERE strategy = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                    """,
                    (strategy, since_ts),
                ).fetchall()
            finally:
                conn.close()

            for row in rows:
                fills.append(
                    FillRecord(
                        fill_id=row[0],
                        timestamp=row[1],
                        strategy=row[2],
                        symbol=row[3],
                        side=row[4],
                        expected_price=row[5],
                        actual_price=row[6],
                        quantity_usd=row[7],
                        exchange=row[8],
                        slippage_bps=row[9],
                        slippage_usd=row[10],
                    )
                )
        except Exception:
            logger.exception(
                "FillTracker: failed to load fills for strategy %s", strategy
            )
        return fills

    def _connect(self) -> sqlite3.Connection:
        """Open a WAL-mode SQLite connection."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _init_db(self) -> None:
        """Create the fills table if it does not exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    fill_id         TEXT    NOT NULL UNIQUE,
                    timestamp       REAL    NOT NULL,
                    strategy        TEXT    NOT NULL,
                    symbol          TEXT    NOT NULL,
                    side            TEXT    NOT NULL,
                    expected_price  REAL    NOT NULL,
                    actual_price    REAL    NOT NULL,
                    quantity_usd    REAL    NOT NULL,
                    exchange        TEXT    NOT NULL,
                    slippage_bps    REAL    NOT NULL,
                    slippage_usd    REAL    NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fills_strategy_ts "
                "ON fills (strategy, timestamp)"
            )
            conn.commit()
        logger.debug("FillTracker DB initialised at %s", self.db_path)
