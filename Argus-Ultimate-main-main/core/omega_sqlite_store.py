#!/usr/bin/env python3
"""
core/omega_sqlite_store.py
==========================
OmegaSQLiteStore — extracted from unified_trading_system.py.

Minimal Ω-spine persistence layer:
  - decision_snapshots : full audit trail of every risk gate decision
  - order_intents      : lifecycle tracking for every order from CREATED → FILLED/CANCELED
  - system_health_snapshots : periodic health metrics (latency, RSS, event-loop delay)

SQLite-only (WAL mode) so Windows paper runs are immediately auditable
without any extra services.

Backward-compat re-export: core/__init__.py exposes OmegaSQLiteStore.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OmegaSQLiteStore:
    """Minimal Ω spine persistence: decision snapshots + order intents.

    Intentionally lightweight and SQLite-only so Windows paper runs are
    immediately auditable without extra services.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30.0)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def _table_columns(self, con: sqlite3.Connection, table_name: str) -> set[str]:
        try:
            rows = con.execute(f"PRAGMA table_info({table_name})").fetchall()
            return {str(r[1]) for r in rows if len(r) > 1 and r[1]}
        except Exception:
            return set()

    def init_schema(self) -> None:
        con = self.connect()
        try:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS decision_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                run_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                correlation_id TEXT,
                symbol TEXT,
                strategy TEXT,
                side TEXT,
                signal_score REAL,
                allowed INTEGER NOT NULL,
                reason_code TEXT NOT NULL,
                details_json TEXT,
                cost_json TEXT,
                exec_plan_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_decision_trace ON decision_snapshots(trace_id);
            CREATE INDEX IF NOT EXISTS idx_decision_cycle ON decision_snapshots(cycle_id);

            CREATE TABLE IF NOT EXISTS order_intents (
                intent_id TEXT PRIMARY KEY,
                ts_utc TEXT NOT NULL,
                run_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                correlation_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT,
                amount REAL,
                price REAL,
                status TEXT NOT NULL,
                exchange_order_id TEXT,
                exec_plan_json TEXT,
                meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_intent_cycle ON order_intents(cycle_id);

            CREATE TABLE IF NOT EXISTS system_health_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycles_completed INTEGER NOT NULL,
                avg_latency_ms REAL NOT NULL,
                errors_last_hour INTEGER NOT NULL,
                warnings_last_hour INTEGER NOT NULL,
                event_loop_delay_ms REAL NOT NULL,
                memory_rss_mb REAL NOT NULL DEFAULT 0.0,
                memory_python_mb REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health_snapshots(timestamp);
            """)
            # Backward-compatible migration for existing DBs.
            try:
                cols = self._table_columns(con, "system_health_snapshots")
                if "memory_rss_mb" not in cols:
                    con.execute("ALTER TABLE system_health_snapshots ADD COLUMN memory_rss_mb REAL NOT NULL DEFAULT 0.0")
                if "memory_python_mb" not in cols:
                    con.execute("ALTER TABLE system_health_snapshots ADD COLUMN memory_python_mb REAL NOT NULL DEFAULT 0.0")
            except Exception:
                pass  # Never block runtime on optional metric columns.
            con.commit()
        finally:
            con.close()

    @staticmethod
    def _normalize_exec_plan(details: dict | None, exec_plan: dict | None, *, allowed: bool) -> dict:
        """Guarantee deterministic execution-plan payload for audit rows."""
        plan = dict(exec_plan or {})
        details_d = dict(details or {})
        if not plan:
            for key in (
                "order_type", "planned_order_size", "slice_count",
                "expected_slippage_bps", "expected_fill_probability",
                "fallback_after_seconds", "priority_score", "reason_codes",
            ):
                if key in details_d and details_d.get(key) is not None:
                    plan[key] = details_d.get(key)
        if plan.get("order_type") is None:
            plan["order_type"] = "none" if not allowed else "unspecified"
        return plan

    def record_decision(
        self,
        *,
        run_id: str,
        trace_id: str,
        cycle_id: int,
        correlation_id: str | None,
        symbol: str | None,
        strategy: str | None,
        side: str | None,
        signal_score: float | None,
        allowed: bool,
        reason_code: str,
        details: dict | None = None,
        cost: dict | None = None,
        exec_plan: dict | None = None,
    ) -> None:
        norm_details = dict(details or {})
        norm_exec_plan = self._normalize_exec_plan(norm_details, exec_plan, allowed=bool(allowed))
        payload_details = json.dumps(norm_details, ensure_ascii=True, default=str)
        payload_cost = json.dumps(cost or {}, ensure_ascii=True, default=str)
        payload_plan = json.dumps(norm_exec_plan, ensure_ascii=True, default=str)
        con = self.connect()
        try:
            table_cols = self._table_columns(con, "decision_snapshots")
            row = {
                "ts_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "timestamp": float(time.time()),
                "run_id": str(run_id),
                "trace_id": str(trace_id),
                "cycle_id": int(cycle_id),
                "correlation_id": correlation_id,
                "symbol": symbol,
                "strategy": strategy,
                "side": side,
                "signal_score": float(signal_score) if signal_score is not None else None,
                "allowed": 1 if allowed else 0,
                "reason_code": str(reason_code),
                "details_json": payload_details,
                "cost_json": payload_cost,
                "exec_plan_json": payload_plan,
                "execution_plan_json": payload_plan,
            }
            cols = [k for k in row.keys() if k in table_cols]
            if not cols:
                raise sqlite3.OperationalError("decision_snapshots table has no compatible columns")
            placeholders = ", ".join(["?"] * len(cols))
            sql = f"INSERT INTO decision_snapshots ({', '.join(cols)}) VALUES ({placeholders})"
            con.execute(sql, tuple(row[c] for c in cols))
            con.commit()
        finally:
            con.close()

    def create_intent(
        self,
        *,
        intent_id: str,
        run_id: str,
        trace_id: str,
        cycle_id: int,
        correlation_id: str | None,
        symbol: str,
        side: str,
        order_type: str | None,
        amount: float | None,
        price: float | None,
        status: str = "CREATED",
        exec_plan: dict | None = None,
        meta: dict | None = None,
    ) -> None:
        con = self.connect()
        try:
            con.execute(
                """
                INSERT OR IGNORE INTO order_intents
                (intent_id, ts_utc, run_id, trace_id, cycle_id, correlation_id,
                 symbol, side, order_type, amount, price, status, exec_plan_json, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent_id,
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    run_id, trace_id, int(cycle_id), correlation_id,
                    symbol, side, order_type, amount, price,
                    status,
                    json.dumps(exec_plan or {}, ensure_ascii=True, default=str),
                    json.dumps(meta or {}, ensure_ascii=True, default=str),
                ),
            )
            con.commit()
        finally:
            con.close()

    def update_intent(
        self,
        intent_id: str,
        *,
        status: str,
        exchange_order_id: str | None = None,
        meta: dict | None = None,
    ) -> None:
        con = self.connect()
        try:
            con.execute(
                """
                UPDATE order_intents
                SET status = ?,
                    exchange_order_id = COALESCE(?, exchange_order_id),
                    meta_json = COALESCE(?, meta_json)
                WHERE intent_id = ?
                """,
                (
                    str(status),
                    exchange_order_id,
                    json.dumps(meta, ensure_ascii=True, default=str) if meta is not None else None,
                    intent_id,
                ),
            )
            con.commit()
        finally:
            con.close()

    def record_system_health_snapshot(self, snapshot: Dict[str, Any]) -> None:
        con = self.connect()
        try:
            cols = self._table_columns(con, "system_health_snapshots")
            row = {
                "timestamp": str(snapshot.get("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")),
                "cycles_completed": int(snapshot.get("cycles_completed", 0) or 0),
                "avg_latency_ms": float(snapshot.get("avg_latency_ms", 0.0) or 0.0),
                "errors_last_hour": int(snapshot.get("errors_last_hour", 0) or 0),
                "warnings_last_hour": int(snapshot.get("warnings_last_hour", 0) or 0),
                "event_loop_delay_ms": float(snapshot.get("event_loop_delay_ms", 0.0) or 0.0),
                "memory_rss_mb": float(snapshot.get("memory_rss_mb", 0.0) or 0.0),
                "memory_python_mb": float(snapshot.get("memory_python_mb", 0.0) or 0.0),
            }
            insert_cols = [k for k in row.keys() if k in cols]
            if not insert_cols:
                return
            placeholders = ", ".join(["?"] * len(insert_cols))
            sql = (
                f"INSERT INTO system_health_snapshots ({', '.join(insert_cols)}) "
                f"VALUES ({placeholders})"
            )
            con.execute(sql, tuple(row[c] for c in insert_cols))
            con.commit()
        finally:
            con.close()
