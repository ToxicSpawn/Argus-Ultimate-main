"""
OutcomeCorrelator — records trade conditions at entry and outcome at close.

Learns per-condition win rates so MomentReadiness can adapt its entry thresholds
based on observed reality rather than static heuristics.

Storage: SQLite at data/outcome_correlations.db (in-memory fallback if unavailable).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trade_conditions (
    trade_id          TEXT PRIMARY KEY,
    health_label      TEXT,
    regime            TEXT,
    timing_action     TEXT,
    signal_conviction REAL,
    readiness_score   REAL,
    strategy_name     TEXT,
    entry_cycle       INTEGER,
    won               INTEGER,
    pnl               REAL,
    closed_cycle      INTEGER
);
"""


@dataclass
class TradeConditionRecord:
    trade_id: str
    health_label: str = "UNKNOWN"
    regime: str = "UNKNOWN"
    timing_action: str = "UNKNOWN"
    signal_conviction: float = 0.5
    readiness_score: float = 50.0
    strategy_name: str = ""
    entry_cycle: int = 0
    won: Optional[bool] = None
    pnl: Optional[float] = None
    closed_cycle: Optional[int] = None


class OutcomeCorrelator:
    """Records trade conditions and learns per-condition win rates."""

    def __init__(
        self,
        db_path: str = "data/outcome_correlations.db",
        min_samples: int = 10,
    ) -> None:
        self.db_path = db_path
        self.min_samples = min_samples

        # In-memory store — always available even when SQLite is unavailable
        self._records: Dict[str, TradeConditionRecord] = {}

        # In-memory condition stats cache: condition_key → (wins, total)
        self._condition_stats: Dict[str, list] = {}  # [wins, total]

        self._db_ok = self._init_db()
        if self._db_ok:
            self._load_from_db()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record_entry(
        self,
        trade_id: str,
        advisory: Dict[str, Any],
        strategy_name: str,
        cycle: int,
    ) -> None:
        """Record the conditions present at trade entry."""
        if not trade_id:
            return

        # Extract health label
        hs = advisory.get("health_score") or {}
        health_label = str(hs.get("label") or "UNKNOWN")

        # Extract regime
        regime_adv = (
            advisory.get("regime_parameters")
            or advisory.get("strategy_regime_matrix")
            or {}
        )
        regime = str(regime_adv.get("regime") or "UNKNOWN")
        if regime == "UNKNOWN":
            # Try strategy_engine path
            se = advisory.get("strategy_engine") or {}
            regime = str(se.get("regime") or "UNKNOWN")

        # Extract timing action
        ti = advisory.get("timing_intelligence") or {}
        timing_action = str(ti.get("action") or "UNKNOWN").upper()
        if timing_action not in ("IDEAL", "OK", "DEFER", "BLOCK"):
            timing_action = "OK"

        # Extract signal conviction
        sig_intel = advisory.get("signal_intelligence") or {}
        signal_conviction = float(sig_intel.get("conviction", 0.5) or 0.5)

        # Extract readiness score (if already computed)
        mr_adv = advisory.get("moment_readiness") or {}
        readiness_score = float(mr_adv.get("score", 50.0) or 50.0)

        rec = TradeConditionRecord(
            trade_id=trade_id,
            health_label=health_label,
            regime=regime,
            timing_action=timing_action,
            signal_conviction=signal_conviction,
            readiness_score=readiness_score,
            strategy_name=strategy_name,
            entry_cycle=cycle,
        )
        self._records[trade_id] = rec

        if self._db_ok:
            self._db_insert(rec)

        logger.debug(
            "OutcomeCorrelator: entry recorded trade_id=%s health=%s regime=%s timing=%s",
            trade_id,
            health_label,
            regime,
            timing_action,
        )

    def record_close(self, trade_id: str, pnl: float, cycle: int) -> None:
        """Update outcome once the trade is closed."""
        if not trade_id:
            return

        rec = self._records.get(trade_id)
        if rec is None:
            # Trade opened before correlator started — create a minimal record
            rec = TradeConditionRecord(trade_id=trade_id, entry_cycle=cycle)
            self._records[trade_id] = rec

        rec.pnl = float(pnl)
        rec.won = pnl > 0.0
        rec.closed_cycle = cycle

        # Update in-memory condition stats
        key = self._condition_key(rec.health_label, rec.regime, rec.timing_action)
        if key not in self._condition_stats:
            self._condition_stats[key] = [0, 0]  # [wins, total]
        self._condition_stats[key][1] += 1
        if rec.won:
            self._condition_stats[key][0] += 1

        if self._db_ok:
            self._db_update_close(trade_id, pnl, pnl > 0.0, cycle)

        logger.debug(
            "OutcomeCorrelator: close recorded trade_id=%s pnl=%.4f won=%s",
            trade_id,
            pnl,
            rec.won,
        )

    def win_rate_for_conditions(
        self,
        health_label: str,
        regime: str,
        timing_action: str,
    ) -> Optional[float]:
        """
        Return the win rate for a given condition combination.

        Returns None if fewer than min_samples closed trades exist for this
        combination — callers should treat None as "not enough data".
        """
        key = self._condition_key(health_label, regime, timing_action)

        # Prefer DB query for freshness if DB is available
        if self._db_ok:
            return self._db_win_rate(key)

        # Fall back to in-memory stats
        stats = self._condition_stats.get(key)
        if stats is None or stats[1] < self.min_samples:
            return None
        return stats[0] / stats[1]

    def top_conditions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return conditions ranked by win rate (min min_samples)."""
        return self._ranked_conditions(reverse=True, n=n)

    def worst_conditions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return conditions ranked by win rate ascending (worst first)."""
        return self._ranked_conditions(reverse=False, n=n)

    def snapshot(self) -> Dict[str, Any]:
        """Return a summary dict suitable for the advisory."""
        total_records = len(self._records)
        closed = sum(1 for r in self._records.values() if r.won is not None)
        learned = sum(
            1
            for stats in self._condition_stats.values()
            if stats[1] >= self.min_samples
        )

        top = self.top_conditions(1)
        worst = self.worst_conditions(1)

        return {
            "total_records": total_records,
            "closed_trades": closed,
            "learned_conditions": learned,
            "top_condition": top[0]["key"] if top else "",
            "worst_condition": worst[0]["key"] if worst else "",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _condition_key(health_label: str, regime: str, timing_action: str) -> str:
        return f"{health_label}:{regime}:{timing_action}"

    def _ranked_conditions(self, reverse: bool, n: int) -> List[Dict[str, Any]]:
        """Build ranked condition list from in-memory stats."""
        qualified = []
        for key, (wins, total) in self._condition_stats.items():
            if total >= self.min_samples:
                qualified.append(
                    {"key": key, "win_rate": wins / total, "sample_count": total}
                )
        qualified.sort(key=lambda x: x["win_rate"], reverse=reverse)
        return qualified[:n]

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> bool:
        """Initialise the SQLite DB. Returns True if successful."""
        import os

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            try:
                os.makedirs(db_dir, exist_ok=True)
            except OSError:
                pass  # Will fail on connect below

        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(_CREATE_TABLE)
            conn.commit()
            conn.close()
            return True
        except Exception as exc:
            logger.debug("OutcomeCorrelator: DB init failed (%s) — using memory", exc)
            return False

    def _load_from_db(self) -> None:
        """Bootstrap in-memory condition stats from existing closed trades."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            cutoff = time.time() - 7 * 86400  # last 7 days
            cur = conn.execute(
                "SELECT * FROM trade_conditions "
                "WHERE won IS NOT NULL AND entry_cycle > ?",
                (int(cutoff),),
            )
            rows = cur.fetchall()
            conn.close()

            for row in rows:
                d = dict(row)
                key = self._condition_key(
                    str(d.get("health_label") or "UNKNOWN"),
                    str(d.get("regime") or "UNKNOWN"),
                    str(d.get("timing_action") or "UNKNOWN"),
                )
                if key not in self._condition_stats:
                    self._condition_stats[key] = [0, 0]
                self._condition_stats[key][1] += 1
                if d.get("won"):
                    self._condition_stats[key][0] += 1

                # Also populate _records
                trade_id = str(d.get("trade_id") or "")
                if trade_id:
                    self._records[trade_id] = TradeConditionRecord(
                        trade_id=trade_id,
                        health_label=str(d.get("health_label") or "UNKNOWN"),
                        regime=str(d.get("regime") or "UNKNOWN"),
                        timing_action=str(d.get("timing_action") or "UNKNOWN"),
                        signal_conviction=float(d.get("signal_conviction") or 0.5),
                        readiness_score=float(d.get("readiness_score") or 50.0),
                        strategy_name=str(d.get("strategy_name") or ""),
                        entry_cycle=int(d.get("entry_cycle") or 0),
                        won=bool(d.get("won")) if d.get("won") is not None else None,
                        pnl=float(d.get("pnl")) if d.get("pnl") is not None else None,
                        closed_cycle=int(d.get("closed_cycle"))
                        if d.get("closed_cycle") is not None
                        else None,
                    )
        except Exception as exc:
            logger.debug("OutcomeCorrelator: DB load failed: %s", exc)

    def _db_insert(self, rec: TradeConditionRecord) -> None:
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(
                "INSERT OR IGNORE INTO trade_conditions "
                "(trade_id, health_label, regime, timing_action, signal_conviction, "
                "readiness_score, strategy_name, entry_cycle) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    rec.trade_id,
                    rec.health_label,
                    rec.regime,
                    rec.timing_action,
                    rec.signal_conviction,
                    rec.readiness_score,
                    rec.strategy_name,
                    rec.entry_cycle,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.debug("OutcomeCorrelator._db_insert failed: %s", exc)

    def _db_update_close(
        self, trade_id: str, pnl: float, won: bool, cycle: int
    ) -> None:
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(
                "UPDATE trade_conditions SET pnl=?, won=?, closed_cycle=? "
                "WHERE trade_id=?",
                (pnl, int(won), cycle, trade_id),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.debug("OutcomeCorrelator._db_update_close failed: %s", exc)

    def _db_win_rate(self, condition_key: str) -> Optional[float]:
        """Query the DB for win rate of a given condition key."""
        try:
            parts = condition_key.split(":", 2)
            if len(parts) != 3:
                return None
            health_label, regime, timing_action = parts

            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cur = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) "
                "FROM trade_conditions "
                "WHERE health_label=? AND regime=? AND timing_action=? "
                "AND won IS NOT NULL",
                (health_label, regime, timing_action),
            )
            row = cur.fetchone()
            conn.close()

            if row is None or row[0] is None or row[0] < self.min_samples:
                return None
            total = int(row[0])
            wins = int(row[1] or 0)
            return wins / total if total > 0 else None
        except Exception as exc:
            logger.debug("OutcomeCorrelator._db_win_rate failed: %s", exc)
            # Fall back to in-memory
            stats = self._condition_stats.get(condition_key)
            if stats is None or stats[1] < self.min_samples:
                return None
            return stats[0] / stats[1]
