"""
Dynamic Drawdown Controller — adjusts position sizing with a convex reduction
curve as drawdown deepens, and fully halts trading beyond a configurable
maximum drawdown threshold.

Sizing curve (default):
    DD%     Multiplier
    0%      1.0
    5%      0.8
    10%     0.5
    15%     0.2
    20%+    0.0  (halted)

The curve is interpolated linearly between control points and persisted to
SQLite so that equity state survives restarts.

Thread-safe: all state mutations are guarded by a lock.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default convex curve: (drawdown_pct, multiplier)
# ---------------------------------------------------------------------------

DEFAULT_CURVE: List[Tuple[float, float]] = [
    (0.0, 1.0),
    (5.0, 0.8),
    (10.0, 0.5),
    (15.0, 0.2),
    (20.0, 0.0),
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DrawdownState:
    """Snapshot of drawdown controller state."""
    current_equity: float
    peak_equity: float
    drawdown_pct: float
    position_multiplier: float
    is_halted: bool
    timestamp: float


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drawdown_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    peak_equity REAL NOT NULL,
    current_equity REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class _DrawdownDB:
    """Thin SQLite wrapper for drawdown state persistence."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def load(self) -> Optional[Tuple[float, float]]:
        """Load (peak_equity, current_equity) or None if empty."""
        row = self._conn.execute(
            "SELECT peak_equity, current_equity FROM drawdown_state WHERE id = 1"
        ).fetchone()
        return (row[0], row[1]) if row else None

    def save(self, peak: float, current: float) -> None:
        """Upsert state."""
        self._conn.execute(
            "INSERT INTO drawdown_state (id, peak_equity, current_equity, updated_at) "
            "VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET peak_equity=excluded.peak_equity, "
            "current_equity=excluded.current_equity, updated_at=excluded.updated_at",
            (peak, current, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DynamicDrawdownController:
    """
    Adjusts position multiplier on a convex curve as drawdown deepens.

    Parameters
    ----------
    db_path : str
        SQLite database path for state persistence.
    curve : list of (drawdown_pct, multiplier) tuples, optional
        Custom sizing curve.  Must be sorted ascending by drawdown_pct,
        with multiplier decreasing.  Defaults to ``DEFAULT_CURVE``.
    initial_equity : float
        Starting equity (used only when no persisted state exists).
    """

    def __init__(self, db_path: str = "data/drawdown_controller.db",
                 curve: Optional[List[Tuple[float, float]]] = None,
                 initial_equity: float = 1000.0) -> None:
        self._curve = sorted(curve or DEFAULT_CURVE, key=lambda t: t[0])
        self._lock = threading.Lock()
        self._halt_threshold = self._curve[-1][0] if self._curve else 20.0

        # Persistence
        self._db = _DrawdownDB(db_path)

        # Restore or initialise
        saved = self._db.load()
        if saved:
            self._peak_equity = saved[0]
            self._current_equity = saved[1]
            logger.info("DynamicDrawdownController restored: peak=%.2f current=%.2f",
                        self._peak_equity, self._current_equity)
        else:
            self._peak_equity = initial_equity
            self._current_equity = initial_equity
            self._db.save(self._peak_equity, self._current_equity)
            logger.info("DynamicDrawdownController initialised: equity=%.2f", initial_equity)

    # ------------------------------------------------------------------
    # Equity updates
    # ------------------------------------------------------------------

    def update(self, current_equity: float, peak_equity: Optional[float] = None) -> DrawdownState:
        """
        Update the current equity (and optionally override peak).

        Parameters
        ----------
        current_equity : float
        peak_equity : float, optional
            Override peak equity; otherwise the controller auto-tracks
            the high-water mark.

        Returns
        -------
        DrawdownState
        """
        with self._lock:
            self._current_equity = current_equity
            if peak_equity is not None:
                self._peak_equity = max(peak_equity, self._peak_equity)
            else:
                self._peak_equity = max(self._current_equity, self._peak_equity)

            self._db.save(self._peak_equity, self._current_equity)

        state = self.get_state()
        if state.is_halted:
            logger.warning("DRAWDOWN HALT: equity=%.2f peak=%.2f dd=%.1f%%",
                           current_equity, self._peak_equity, state.drawdown_pct)
        return state

    # ------------------------------------------------------------------
    # Position multiplier
    # ------------------------------------------------------------------

    def get_position_multiplier(self) -> float:
        """
        Current position size multiplier in [0.0, 1.0] based on drawdown depth.

        Interpolated linearly between curve control points.
        """
        dd = self.get_drawdown_pct()
        return self._interpolate(dd)

    def _interpolate(self, dd_pct: float) -> float:
        """Linearly interpolate the sizing curve at *dd_pct*."""
        if dd_pct <= self._curve[0][0]:
            return self._curve[0][1]
        if dd_pct >= self._curve[-1][0]:
            return self._curve[-1][1]

        for i in range(len(self._curve) - 1):
            d0, m0 = self._curve[i]
            d1, m1 = self._curve[i + 1]
            if d0 <= dd_pct <= d1:
                frac = (dd_pct - d0) / (d1 - d0) if d1 > d0 else 0.0
                return m0 + frac * (m1 - m0)

        return 0.0

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    def get_drawdown_pct(self) -> float:
        """Current drawdown as a positive percentage (0 = at peak)."""
        with self._lock:
            peak = self._peak_equity
            current = self._current_equity
        if peak <= 0:
            return 0.0
        dd = (peak - current) / peak * 100.0
        return max(0.0, dd)

    def is_halted(self) -> bool:
        """True when drawdown exceeds the halt threshold."""
        return self.get_drawdown_pct() >= self._halt_threshold

    # ------------------------------------------------------------------
    # Recovery estimate
    # ------------------------------------------------------------------

    def get_recovery_estimate(self, avg_daily_return_pct: float) -> int:
        """
        Estimated trading days to recover from current drawdown.

        Parameters
        ----------
        avg_daily_return_pct : float
            Average daily portfolio return (e.g. 0.5 for 0.5%/day).

        Returns
        -------
        int
            Days needed.  Returns 0 if no drawdown, -1 if recovery impossible
            (avg return <= 0).
        """
        dd_pct = self.get_drawdown_pct()
        if dd_pct <= 0:
            return 0
        if avg_daily_return_pct <= 0:
            return -1

        # Compound growth: peak = current * (1 + r)^n  →  n = log(peak/current) / log(1+r)
        with self._lock:
            ratio = self._peak_equity / max(self._current_equity, 1e-12)
        r = avg_daily_return_pct / 100.0
        if r <= 0:
            return -1
        days = math.log(ratio) / math.log(1.0 + r)
        return max(1, math.ceil(days))

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> DrawdownState:
        """Full state snapshot."""
        dd = self.get_drawdown_pct()
        return DrawdownState(
            current_equity=self._current_equity,
            peak_equity=self._peak_equity,
            drawdown_pct=round(dd, 4),
            position_multiplier=round(self.get_position_multiplier(), 4),
            is_halted=self.is_halted(),
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
