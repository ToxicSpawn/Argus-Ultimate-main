"""
Drawdown DNA Analysis — classifies drawdowns by root cause so the system can
learn which types of drawdowns are recurring and adapt.

Each drawdown is stored with its equity curve, associated trades, and metadata.
The classifier uses heuristics based on drawdown depth, duration, trade count,
and clustering to assign a cause label.

Classifications:
    * ``regime_change`` — drawdown coincides with a detected regime shift
    * ``execution_failure`` — high slippage / fill rate issues during drawdown
    * ``model_drift`` — gradual degradation (long duration, many trades)
    * ``black_swan`` — rapid, deep drawdown (< few trades, > large pct)
    * ``normal_variance`` — within expected statistical range
    * ``strategy_decay`` — monotonic equity decline over extended period

Persistence: SQLite at ``data/drawdown_dna.db``.
"""

from __future__ import annotations

import logging
import sqlite3
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "drawdown_dna.db"


@dataclass
class DrawdownClassification:
    """Classification result for a single drawdown episode."""

    drawdown_id: int
    cause: str
    depth_pct: float
    duration_hours: float
    trade_count: int
    confidence: float = 0.0
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"DrawdownClassification(id={self.drawdown_id}, cause={self.cause!r}, "
            f"depth={self.depth_pct:.2f}%, dur={self.duration_hours:.1f}h, "
            f"trades={self.trade_count}, conf={self.confidence:.2f})"
        )


class DrawdownDNA:
    """Records and classifies drawdown episodes for post-mortem analysis.

    Parameters
    ----------
    db_path : str | None
        SQLite database path.  Defaults to ``data/drawdown_dna.db``.
    black_swan_depth_pct : float
        Minimum depth (%) to consider a drawdown a black-swan candidate.
    black_swan_max_hours : float
        Maximum duration (hours) for black-swan classification.
    decay_min_hours : float
        Minimum duration (hours) to consider strategy decay.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        black_swan_depth_pct: float = 10.0,
        black_swan_max_hours: float = 4.0,
        decay_min_hours: float = 168.0,
    ) -> None:
        if db_path is None:
            db_dir = _DEFAULT_DB_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_dir / _DEFAULT_DB_NAME)
        else:
            self._db_path = str(db_path)

        self._black_swan_depth = black_swan_depth_pct
        self._black_swan_max_hours = black_swan_max_hours
        self._decay_min_hours = decay_min_hours
        self._lock = threading.Lock()
        self._init_db()
        logger.info("DrawdownDNA initialised, db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drawdowns (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_equity    REAL    NOT NULL,
                        trough_equity   REAL    NOT NULL,
                        end_equity      REAL    NOT NULL,
                        start_time      TEXT    NOT NULL,
                        end_time        TEXT    NOT NULL,
                        trades_json     TEXT    NOT NULL DEFAULT '[]',
                        cause           TEXT    DEFAULT NULL,
                        depth_pct       REAL    DEFAULT 0.0,
                        duration_hours  REAL    DEFAULT 0.0,
                        confidence      REAL    DEFAULT 0.0,
                        notes           TEXT    DEFAULT ''
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_drawdown(
        self,
        start_equity: float,
        trough_equity: float,
        end_equity: float,
        start_time: datetime,
        end_time: datetime,
        trades_during: Optional[List[dict]] = None,
    ) -> int:
        """Record a drawdown episode.

        Parameters
        ----------
        start_equity : float
            Portfolio equity at drawdown start.
        trough_equity : float
            Lowest equity during the drawdown.
        end_equity : float
            Equity when the drawdown ended (recovered or stabilised).
        start_time : datetime
            UTC timestamp of drawdown start.
        end_time : datetime
            UTC timestamp of drawdown end.
        trades_during : list[dict] | None
            List of trade dicts that occurred during the drawdown.
            Each dict may contain ``symbol``, ``strategy``, ``pnl``, ``slippage_bps``.

        Returns
        -------
        int
            The database row ID of the recorded drawdown.
        """
        if trades_during is None:
            trades_during = []

        depth_pct = 0.0
        if start_equity > 0:
            depth_pct = ((start_equity - trough_equity) / start_equity) * 100.0

        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        trades_json = json.dumps(trades_during)

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """INSERT INTO drawdowns
                       (start_equity, trough_equity, end_equity, start_time, end_time,
                        trades_json, depth_pct, duration_hours)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (start_equity, trough_equity, end_equity,
                     start_time.isoformat(), end_time.isoformat(),
                     trades_json, depth_pct, duration_hours),
                )
                conn.commit()
                dd_id = cursor.lastrowid
                logger.info(
                    "Recorded drawdown id=%d depth=%.2f%% dur=%.1fh trades=%d",
                    dd_id, depth_pct, duration_hours, len(trades_during),
                )
                return dd_id
            finally:
                conn.close()

    def classify(self, drawdown_id: int) -> DrawdownClassification:
        """Classify a drawdown by its root cause.

        Parameters
        ----------
        drawdown_id : int
            Row ID of the drawdown to classify.

        Returns
        -------
        DrawdownClassification
            The classification result.
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM drawdowns WHERE id = ?", (drawdown_id,)
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            logger.warning("classify: drawdown_id=%d not found", drawdown_id)
            return DrawdownClassification(
                drawdown_id=drawdown_id, cause="unknown",
                depth_pct=0.0, duration_hours=0.0, trade_count=0,
            )

        depth_pct = row["depth_pct"]
        duration_hours = row["duration_hours"]
        trades = json.loads(row["trades_json"])
        trade_count = len(trades)

        # Compute aggregate slippage from trades
        total_slippage = 0.0
        for t in trades:
            total_slippage += abs(t.get("slippage_bps", 0.0))
        avg_slippage = total_slippage / trade_count if trade_count > 0 else 0.0

        cause, confidence, notes = self._classify_heuristic(
            depth_pct, duration_hours, trade_count, avg_slippage,
        )

        # Persist classification
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE drawdowns SET cause = ?, confidence = ?, notes = ? WHERE id = ?",
                    (cause, confidence, notes, drawdown_id),
                )
                conn.commit()
            finally:
                conn.close()

        result = DrawdownClassification(
            drawdown_id=drawdown_id,
            cause=cause,
            depth_pct=depth_pct,
            duration_hours=duration_hours,
            trade_count=trade_count,
            confidence=confidence,
            notes=notes,
        )
        logger.info("Classified drawdown: %s", result)
        return result

    def _classify_heuristic(
        self,
        depth_pct: float,
        duration_hours: float,
        trade_count: int,
        avg_slippage_bps: float,
    ) -> tuple:
        """Apply rule-based classification heuristics.

        Returns (cause, confidence, notes).
        """
        # Black swan: deep and fast
        if depth_pct >= self._black_swan_depth and duration_hours <= self._black_swan_max_hours:
            return (
                "black_swan",
                0.85,
                f"Rapid {depth_pct:.1f}% drop in {duration_hours:.1f}h",
            )

        # Execution failure: high average slippage
        if avg_slippage_bps > 20.0 and trade_count > 0:
            return (
                "execution_failure",
                0.75,
                f"High avg slippage {avg_slippage_bps:.1f}bps over {trade_count} trades",
            )

        # Strategy decay: long, slow bleed
        if duration_hours >= self._decay_min_hours and depth_pct < self._black_swan_depth:
            return (
                "strategy_decay",
                0.70,
                f"Slow bleed over {duration_hours:.0f}h ({depth_pct:.1f}%)",
            )

        # Model drift: moderate duration with many trades
        if trade_count >= 10 and duration_hours >= 24.0:
            return (
                "model_drift",
                0.65,
                f"{trade_count} trades over {duration_hours:.0f}h — possible model degradation",
            )

        # Regime change: moderate depth, few trades (market moved against positions)
        if depth_pct >= 3.0 and trade_count <= 3:
            return (
                "regime_change",
                0.60,
                f"{depth_pct:.1f}% drop with only {trade_count} trades — likely regime shift",
            )

        # Default: normal variance
        return (
            "normal_variance",
            0.50,
            f"Within expected range: {depth_pct:.1f}% over {duration_hours:.1f}h",
        )

    def get_drawdown_history(
        self, lookback_days: int = 90
    ) -> List[DrawdownClassification]:
        """Return classified drawdowns within the lookback window.

        Parameters
        ----------
        lookback_days : int
            Number of days to look back.

        Returns
        -------
        list[DrawdownClassification]
            Classified drawdowns, most recent first.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM drawdowns WHERE start_time >= ? ORDER BY start_time DESC",
                    (cutoff,),
                ).fetchall()
            finally:
                conn.close()

        results = []
        for row in rows:
            trades = json.loads(row["trades_json"])
            cause = row["cause"] or "unclassified"
            results.append(DrawdownClassification(
                drawdown_id=row["id"],
                cause=cause,
                depth_pct=row["depth_pct"],
                duration_hours=row["duration_hours"],
                trade_count=len(trades),
                confidence=row["confidence"] or 0.0,
                notes=row["notes"] or "",
            ))
        return results

    def get_common_causes(self) -> Dict[str, int]:
        """Return a frequency count of drawdown causes.

        Returns
        -------
        dict[str, int]
            Mapping of cause → count, sorted by frequency descending.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT cause, COUNT(*) as cnt FROM drawdowns WHERE cause IS NOT NULL GROUP BY cause ORDER BY cnt DESC"
                ).fetchall()
            finally:
                conn.close()

        result = {row["cause"]: row["cnt"] for row in rows}
        logger.info("Common drawdown causes: %s", result)
        return result
