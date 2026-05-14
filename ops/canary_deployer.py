#!/usr/bin/env python3
"""
Canary Deployment for Strategies — traffic-split testing of parameter changes.

Allows gradual rollout of new strategy parameters by routing a configurable
percentage of traffic to the canary variant while the incumbent continues to
handle the majority.  Metrics are tracked per-canary so operators can make
data-driven promote/rollback decisions.

Persistence: SQLite at ``data/canary_deployments.db`` (configurable).

Usage::

    deployer = CanaryDeployer()
    cid = deployer.create_canary("momentum", {"fast_period": 8}, traffic_pct=10)

    if deployer.should_route_to_canary("momentum"):
        # Execute with canary params
        ...

    status = deployer.get_canary_status(cid)
    if status.pnl > 0 and status.trades > 20:
        deployer.promote_canary(cid)
    else:
        deployer.rollback_canary(cid)
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CanaryStatus:
    """Snapshot of a canary deployment's state."""

    canary_id: str
    strategy: str
    traffic_pct: float
    old_params: Dict[str, Any]
    new_params: Dict[str, Any]
    trades: int
    pnl: float
    started_at: float
    health: str  # "healthy" | "degraded" | "failed" | "promoted" | "rolled_back"


# ---------------------------------------------------------------------------
# Canary Deployer
# ---------------------------------------------------------------------------


class CanaryDeployer:
    """Manages canary deployments for strategy parameter changes.

    Parameters
    ----------
    db_path:
        Path to SQLite database.  Defaults to ``data/canary_deployments.db``
        relative to the repository root.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            db_path = str(repo_root / "data" / "canary_deployments.db")

        self._db_path = db_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = self._connect()
        self._create_tables()
        logger.info("CanaryDeployer initialised — db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS canaries (
                    canary_id   TEXT PRIMARY KEY,
                    strategy    TEXT NOT NULL,
                    old_params  TEXT NOT NULL DEFAULT '{}',
                    new_params  TEXT NOT NULL,
                    traffic_pct REAL NOT NULL DEFAULT 5.0,
                    trades      INTEGER NOT NULL DEFAULT 0,
                    pnl         REAL NOT NULL DEFAULT 0.0,
                    started_at  REAL NOT NULL,
                    health      TEXT NOT NULL DEFAULT 'healthy',
                    active      INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_canaries_strategy
                    ON canaries (strategy, active);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_canary(
        self,
        strategy_name: str,
        new_params: Dict[str, Any],
        traffic_pct: float = 5.0,
        old_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new canary deployment for *strategy_name*.

        Parameters
        ----------
        strategy_name:
            Name of the strategy being canary-tested.
        new_params:
            The parameter dictionary to trial.
        traffic_pct:
            Percentage of traffic (0-100) to route to the canary.
        old_params:
            The current (incumbent) parameters.  Stored for rollback reference.

        Returns
        -------
        str
            A unique ``canary_id``.
        """
        canary_id = str(uuid.uuid4())[:12]
        now = time.time()

        with self._lock:
            self._conn.execute(
                "INSERT INTO canaries "
                "(canary_id, strategy, old_params, new_params, traffic_pct, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    canary_id,
                    strategy_name,
                    json.dumps(old_params or {}),
                    json.dumps(new_params),
                    traffic_pct,
                    now,
                ),
            )
            self._conn.commit()

        logger.info(
            "CanaryDeployer created canary=%s strategy=%s traffic_pct=%.1f%%",
            canary_id,
            strategy_name,
            traffic_pct,
        )
        return canary_id

    def get_canary_status(self, canary_id: str) -> Optional[CanaryStatus]:
        """Return the current status of a canary deployment.

        Returns ``None`` if *canary_id* does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT canary_id, strategy, old_params, new_params, traffic_pct, "
                "trades, pnl, started_at, health "
                "FROM canaries WHERE canary_id = ?",
                (canary_id,),
            ).fetchone()

        if row is None:
            return None

        return CanaryStatus(
            canary_id=row[0],
            strategy=row[1],
            old_params=json.loads(row[2]),
            new_params=json.loads(row[3]),
            traffic_pct=row[4],
            trades=row[5],
            pnl=row[6],
            started_at=row[7],
            health=row[8],
        )

    def promote_canary(self, canary_id: str) -> bool:
        """Promote the canary: mark it as the new incumbent.

        Returns ``True`` if the canary existed and was active, ``False`` otherwise.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE canaries SET health = 'promoted', active = 0 "
                "WHERE canary_id = ? AND active = 1",
                (canary_id,),
            )
            self._conn.commit()
            changed = cur.rowcount > 0

        if changed:
            logger.info("CanaryDeployer promoted canary=%s", canary_id)
        else:
            logger.warning(
                "CanaryDeployer promote failed — canary=%s not found or inactive",
                canary_id,
            )
        return changed

    def rollback_canary(self, canary_id: str) -> bool:
        """Roll back the canary: deactivate it and revert to incumbent params.

        Returns ``True`` if the canary existed and was active, ``False`` otherwise.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE canaries SET health = 'rolled_back', active = 0 "
                "WHERE canary_id = ? AND active = 1",
                (canary_id,),
            )
            self._conn.commit()
            changed = cur.rowcount > 0

        if changed:
            logger.info("CanaryDeployer rolled back canary=%s", canary_id)
        else:
            logger.warning(
                "CanaryDeployer rollback failed — canary=%s not found or inactive",
                canary_id,
            )
        return changed

    def should_route_to_canary(self, strategy_name: str) -> bool:
        """Probabilistically decide whether to route to the canary for *strategy_name*.

        Returns ``False`` if no active canary exists for the strategy.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT traffic_pct FROM canaries "
                "WHERE strategy = ? AND active = 1 ORDER BY started_at DESC LIMIT 1",
                (strategy_name,),
            ).fetchone()

        if row is None:
            return False

        traffic_pct: float = row[0]
        return random.random() * 100.0 < traffic_pct

    def record_trade(
        self, canary_id: str, pnl: float, *, health: Optional[str] = None
    ) -> None:
        """Record a trade result against the canary.

        Parameters
        ----------
        canary_id:
            The canary that executed the trade.
        pnl:
            Profit/loss of the trade (AUD or USD).
        health:
            Optional health override (e.g. ``"degraded"`` if fill quality is poor).
        """
        with self._lock:
            if health:
                self._conn.execute(
                    "UPDATE canaries SET trades = trades + 1, pnl = pnl + ?, health = ? "
                    "WHERE canary_id = ? AND active = 1",
                    (pnl, health, canary_id),
                )
            else:
                self._conn.execute(
                    "UPDATE canaries SET trades = trades + 1, pnl = pnl + ? "
                    "WHERE canary_id = ? AND active = 1",
                    (pnl, canary_id),
                )
            self._conn.commit()

    def get_active_canaries(self, strategy_name: Optional[str] = None) -> List[CanaryStatus]:
        """Return all active canaries, optionally filtered by strategy."""
        with self._lock:
            if strategy_name:
                rows = self._conn.execute(
                    "SELECT canary_id, strategy, old_params, new_params, traffic_pct, "
                    "trades, pnl, started_at, health "
                    "FROM canaries WHERE active = 1 AND strategy = ? "
                    "ORDER BY started_at DESC",
                    (strategy_name,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT canary_id, strategy, old_params, new_params, traffic_pct, "
                    "trades, pnl, started_at, health "
                    "FROM canaries WHERE active = 1 ORDER BY started_at DESC"
                ).fetchall()

        return [
            CanaryStatus(
                canary_id=r[0],
                strategy=r[1],
                old_params=json.loads(r[2]),
                new_params=json.loads(r[3]),
                traffic_pct=r[4],
                trades=r[5],
                pnl=r[6],
                started_at=r[7],
                health=r[8],
            )
            for r in rows
        ]

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
        logger.info("CanaryDeployer closed — db=%s", self._db_path)
