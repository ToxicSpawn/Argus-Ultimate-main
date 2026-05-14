"""
ml/trade_outcome_labeler.py — Automatic Trade Outcome Labeling

Labels every trade as "good", "bad", or "neutral" by measuring post-trade
price movement relative to trade direction.  Stores labels in a thread-safe
SQLite database and provides helpers for strategy accuracy computation
and ML dataset construction.

Classes
-------
TradeLabel      Immutable dataclass for a single labelled trade.
TradeOutcomeLabeler
    Main labeler — call ``label_trade()`` for each fill, then query
    ``get_strategy_accuracy()`` or ``build_classifier_dataset()``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradeLabel:
    """Immutable record of a labelled trade."""
    symbol: str
    side: str                       # "buy" or "sell"
    entry_price: float
    label: str                      # "good", "bad", or "neutral"
    post_trade_return_pct: float    # signed pct move in trade direction
    max_favorable_pct: float        # best move in trade direction
    max_adverse_pct: float          # worst move against trade direction
    horizon_minutes: int
    timestamp: str                  # ISO-8601 UTC
    strategy: str = ""
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS trade_labels (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    entry_price   REAL    NOT NULL,
    exit_price    REAL,
    exit_time     TEXT,
    label         TEXT    NOT NULL,
    post_trade_return_pct  REAL NOT NULL,
    max_favorable_pct      REAL NOT NULL,
    max_adverse_pct        REAL NOT NULL,
    horizon_minutes        INTEGER NOT NULL,
    strategy      TEXT    NOT NULL DEFAULT '',
    timestamp     TEXT    NOT NULL
);
"""

_IDX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tl_symbol ON trade_labels(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_tl_strategy ON trade_labels(strategy);",
    "CREATE INDEX IF NOT EXISTS idx_tl_timestamp ON trade_labels(timestamp);",
]


class TradeOutcomeLabeler:
    """Label trades as good / bad / neutral and persist to SQLite.

    Parameters
    ----------
    db_path : str, optional
        Override the default ``data/trade_labels.db`` path.
    good_threshold_pct : float
        Minimum post-trade return (%) in trade direction for a *good* label.
    bad_threshold_pct : float
        Maximum adverse excursion (%) for a *bad* label.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        good_threshold_pct: float = 0.5,
        bad_threshold_pct: float = 1.0,
    ) -> None:
        self._db_path = db_path or os.path.join(_DB_DIR, "trade_labels.db")
        self._good_threshold = good_threshold_pct
        self._bad_threshold = bad_threshold_pct
        self._lock = threading.Lock()
        self._ensure_schema()
        log.info(
            "TradeOutcomeLabeler initialised  db=%s  good>=%.2f%%  bad>=%.2f%%",
            self._db_path, self._good_threshold, self._bad_threshold,
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with self._lock:
            con = sqlite3.connect(self._db_path)
            try:
                con.execute(_CREATE_SQL)
                for idx in _IDX_SQL:
                    con.execute(idx)
                con.commit()
            finally:
                con.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        horizon_minutes: int = 60,
        strategy: str = "",
        *,
        observed_prices: Optional[List[float]] = None,
    ) -> TradeLabel:
        """Classify a single trade and persist the label.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/USD"``.
        side : str
            ``"buy"`` or ``"sell"``.
        entry_price : float
            Fill price at entry.
        entry_time : datetime, optional
            When the trade was entered (defaults to now UTC).
        exit_price : float, optional
            If the trade has already been closed.
        exit_time : datetime, optional
            When the trade was exited.
        horizon_minutes : int
            Evaluation horizon in minutes (used for metadata).
        strategy : str
            Strategy name for later accuracy queries.
        observed_prices : list[float], optional
            Intra-horizon price samples (high-frequency ticks or OHLC
            closes).  Used to compute MFE / MAE.  If *None*, only
            ``exit_price`` is used; ``max_favorable_pct`` and
            ``max_adverse_pct`` equal the post-trade return.

        Returns
        -------
        TradeLabel
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        ts = (entry_time or datetime.now(timezone.utc)).isoformat()

        # Compute return in trade direction
        if exit_price is not None and exit_price > 0:
            if side == "buy":
                post_return_pct = ((exit_price - entry_price) / entry_price) * 100.0
            else:
                post_return_pct = ((entry_price - exit_price) / entry_price) * 100.0
        else:
            post_return_pct = 0.0

        # Compute MFE / MAE from observed prices
        max_favorable_pct = 0.0
        max_adverse_pct = 0.0
        if observed_prices:
            for p in observed_prices:
                if side == "buy":
                    move = ((p - entry_price) / entry_price) * 100.0
                else:
                    move = ((entry_price - p) / entry_price) * 100.0
                if move > max_favorable_pct:
                    max_favorable_pct = move
                if move < 0 and abs(move) > max_adverse_pct:
                    max_adverse_pct = abs(move)
        else:
            # Fallback: use exit price only
            if post_return_pct > 0:
                max_favorable_pct = post_return_pct
            elif post_return_pct < 0:
                max_adverse_pct = abs(post_return_pct)

        # Classification
        if post_return_pct >= self._good_threshold:
            label = "good"
        elif max_adverse_pct >= self._bad_threshold:
            label = "bad"
        else:
            label = "neutral"

        exit_time_str = exit_time.isoformat() if exit_time else None

        result = TradeLabel(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            label=label,
            post_trade_return_pct=round(post_return_pct, 6),
            max_favorable_pct=round(max_favorable_pct, 6),
            max_adverse_pct=round(max_adverse_pct, 6),
            horizon_minutes=horizon_minutes,
            timestamp=ts,
            strategy=strategy,
            exit_price=exit_price,
            exit_time=exit_time_str,
        )

        self._persist(result)
        log.debug("Labelled %s %s %s → %s (ret=%.4f%%)", symbol, side, entry_price, label, post_return_pct)
        return result

    def get_strategy_accuracy(
        self,
        strategy: str,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """Return accuracy statistics for a strategy over the lookback window.

        Returns
        -------
        dict
            Keys: ``win_rate``, ``avg_return``, ``good_count``, ``bad_count``,
            ``neutral_count``, ``total``.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT label, post_trade_return_pct FROM trade_labels "
                    "WHERE strategy = ? AND timestamp >= ?",
                    (strategy, cutoff_iso),
                ).fetchall()
            finally:
                con.close()

        good = sum(1 for r in rows if r[0] == "good")
        bad = sum(1 for r in rows if r[0] == "bad")
        neutral = sum(1 for r in rows if r[0] == "neutral")
        total = len(rows)
        avg_ret = sum(r[1] for r in rows) / total if total else 0.0
        win_rate = good / total if total else 0.0

        return {
            "win_rate": round(win_rate, 4),
            "avg_return": round(avg_ret, 6),
            "good_count": good,
            "bad_count": bad,
            "neutral_count": neutral,
            "total": total,
        }

    def build_classifier_dataset(
        self,
        lookback_days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Build a flat feature-dict list suitable for ML training.

        Each row contains trade metadata plus the label.  Downstream code
        can enrich these rows with additional features (orderbook, regime,
        sentiment, etc.) before training.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        with self._lock:
            con = self._connect()
            try:
                con.row_factory = sqlite3.Row
                rows = con.execute(
                    "SELECT * FROM trade_labels WHERE timestamp >= ? ORDER BY timestamp",
                    (cutoff_iso,),
                ).fetchall()
            finally:
                con.close()

        dataset: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            # Encode side numerically for ML convenience
            d["side_numeric"] = 1 if d.get("side") == "buy" else -1
            # Label encoding: good=1, neutral=0, bad=-1
            label_map = {"good": 1, "neutral": 0, "bad": -1}
            d["label_numeric"] = label_map.get(d.get("label", "neutral"), 0)
            dataset.append(d)

        log.info("Built classifier dataset: %d rows over %d-day lookback", len(dataset), lookback_days)
        return dataset

    def get_all_labels(self, limit: int = 1000) -> List[TradeLabel]:
        """Return the most recent labels from the database."""
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT symbol, side, entry_price, label, post_trade_return_pct, "
                    "max_favorable_pct, max_adverse_pct, horizon_minutes, timestamp, "
                    "strategy, exit_price, exit_time "
                    "FROM trade_labels ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                con.close()

        return [
            TradeLabel(
                symbol=r[0], side=r[1], entry_price=r[2], label=r[3],
                post_trade_return_pct=r[4], max_favorable_pct=r[5],
                max_adverse_pct=r[6], horizon_minutes=r[7], timestamp=r[8],
                strategy=r[9], exit_price=r[10], exit_time=r[11],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _persist(self, tl: TradeLabel) -> None:
        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    "INSERT INTO trade_labels "
                    "(symbol, side, entry_price, exit_price, exit_time, label, "
                    " post_trade_return_pct, max_favorable_pct, max_adverse_pct, "
                    " horizon_minutes, strategy, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        tl.symbol, tl.side, tl.entry_price, tl.exit_price,
                        tl.exit_time, tl.label, tl.post_trade_return_pct,
                        tl.max_favorable_pct, tl.max_adverse_pct,
                        tl.horizon_minutes, tl.strategy, tl.timestamp,
                    ),
                )
                con.commit()
            finally:
                con.close()
