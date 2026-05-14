"""Decision audit trail — persists every sizing/gating decision to SQLite."""
from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionRecord:
    order_id: str
    symbol: str
    side: str
    strategy: str
    initial_size_pct: float
    final_size_pct: float
    gates_applied: List[str]
    advisory_keys_used: List[str]
    reason: str
    timestamp: float = field(default_factory=time.time)


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS decision_audit (
    order_id         TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    side             TEXT NOT NULL,
    strategy         TEXT NOT NULL,
    initial_size     REAL NOT NULL,
    final_size       REAL NOT NULL,
    gates_json       TEXT NOT NULL,
    advisory_keys_json TEXT NOT NULL,
    reason           TEXT NOT NULL,
    timestamp        REAL NOT NULL
)
"""

_CREATE_IDX_SYMBOL = (
    "CREATE INDEX IF NOT EXISTS idx_da_symbol ON decision_audit(symbol)"
)
_CREATE_IDX_STRATEGY = (
    "CREATE INDEX IF NOT EXISTS idx_da_strategy ON decision_audit(strategy)"
)
_CREATE_IDX_TS = (
    "CREATE INDEX IF NOT EXISTS idx_da_ts ON decision_audit(timestamp)"
)


class DecisionAuditTrail:
    """SQLite-backed audit trail for every order-sizing decision."""

    def __init__(self, db_path: str = "data/decision_audit.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = self._connect()
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_IDX_SYMBOL)
            conn.execute(_CREATE_IDX_STRATEGY)
            conn.execute(_CREATE_IDX_TS)
            conn.commit()
        finally:
            conn.close()
        logger.info("DecisionAuditTrail initialised — db=%s", db_path)

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    # ------------------------------------------------------------------
    def record(self, rec: DecisionRecord) -> None:
        """Persist a single decision record."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO decision_audit "
                    "(order_id, symbol, side, strategy, initial_size, final_size, "
                    "gates_json, advisory_keys_json, reason, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec.order_id,
                        rec.symbol,
                        rec.side,
                        rec.strategy,
                        rec.initial_size_pct,
                        rec.final_size_pct,
                        json.dumps(rec.gates_applied),
                        json.dumps(rec.advisory_keys_used),
                        rec.reason,
                        rec.timestamp,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        logger.debug("Recorded decision %s %s %s", rec.order_id, rec.symbol, rec.side)

    # ------------------------------------------------------------------
    def query(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        since_seconds: Optional[float] = None,
        limit: int = 100,
    ) -> List[DecisionRecord]:
        """Query decisions with optional filters."""
        clauses: List[str] = []
        params: List[object] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if strategy is not None:
            clauses.append("strategy = ?")
            params.append(strategy)
        if since_seconds is not None:
            clauses.append("timestamp >= ?")
            params.append(time.time() - since_seconds)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT order_id, symbol, side, strategy, initial_size, final_size, "
            "gates_json, advisory_keys_json, reason, timestamp "
            f"FROM decision_audit{where} ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        return [
            DecisionRecord(
                order_id=r[0],
                symbol=r[1],
                side=r[2],
                strategy=r[3],
                initial_size_pct=r[4],
                final_size_pct=r[5],
                gates_applied=json.loads(r[6]),
                advisory_keys_used=json.loads(r[7]),
                reason=r[8],
                timestamp=r[9],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    def export_csv(self, path: str) -> Path:
        """Export all records to CSV. Returns the output path."""
        out = Path(path)
        os.makedirs(out.parent, exist_ok=True)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT order_id, symbol, side, strategy, initial_size, final_size, "
                "gates_json, advisory_keys_json, reason, timestamp "
                "FROM decision_audit ORDER BY timestamp"
            ).fetchall()
        finally:
            conn.close()

        headers = [
            "order_id", "symbol", "side", "strategy",
            "initial_size_pct", "final_size_pct",
            "gates_applied", "advisory_keys_used", "reason", "timestamp",
        ]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for r in rows:
                writer.writerow(r)
        logger.info("Exported %d decision records to %s", len(rows), out)
        return out

    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        """Return aggregate statistics over all recorded decisions."""
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM decision_audit").fetchone()[0]
            rows = conn.execute(
                "SELECT gates_json FROM decision_audit"
            ).fetchall()
        finally:
            conn.close()

        if total == 0:
            return {
                "total_decisions": 0,
                "avg_gates_per_decision": 0.0,
                "most_common_gates": [],
            }

        gate_counts: Dict[str, int] = {}
        total_gates = 0
        for (gates_json,) in rows:
            gates = json.loads(gates_json)
            total_gates += len(gates)
            for g in gates:
                gate_counts[g] = gate_counts.get(g, 0) + 1

        sorted_gates = sorted(gate_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_decisions": total,
            "avg_gates_per_decision": round(total_gates / total, 2),
            "most_common_gates": sorted_gates[:10],
        }
