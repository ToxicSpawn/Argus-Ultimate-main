"""
Win/Loss Streak Analyzer — tracks consecutive outcomes per strategy and applies
the Wald–Wolfowitz runs test to detect non-random streaks.

A strategy that is "on tilt" (extended losing streak) may warrant reduced
allocation or temporary disabling.  Conversely, abnormally long winning
streaks may indicate curve-fitting rather than genuine edge.

Persistence: SQLite at ``data/streaks.db``.

Usage::

    sa = StreakAnalyzer()
    sa.record_outcome("momentum", won=True, pnl=42.0)
    sa.record_outcome("momentum", won=False, pnl=-15.0)
    streak = sa.get_current_streak("momentum")
    logger.info(streak)
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "streaks.db"


@dataclass
class Streak:
    """Current streak state for a strategy."""

    type: str  # "win" or "loss"
    length: int
    total_pnl: float

    def __repr__(self) -> str:
        return f"Streak(type={self.type!r}, length={self.length}, pnl={self.total_pnl:.4f})"


@dataclass
class RunsTestResult:
    """Result of the Wald-Wolfowitz runs test for randomness."""

    z_score: float
    p_value: float
    random: bool
    interpretation: str

    def __repr__(self) -> str:
        return (
            f"RunsTestResult(z={self.z_score:.3f}, p={self.p_value:.4f}, "
            f"random={self.random}, interp={self.interpretation!r})"
        )


class StreakAnalyzer:
    """Track win/loss streaks and apply statistical tests for randomness.

    Parameters
    ----------
    db_path : str | None
        SQLite database path.  Defaults to ``data/streaks.db``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_dir = _DEFAULT_DB_DIR
            db_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_dir / _DEFAULT_DB_NAME)
        else:
            self._db_path = str(db_path)

        self._lock = threading.Lock()
        self._init_db()
        logger.info("StreakAnalyzer initialised, db=%s", self._db_path)

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
                    CREATE TABLE IF NOT EXISTS outcomes (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy    TEXT    NOT NULL,
                        won         INTEGER NOT NULL,
                        pnl         REAL    NOT NULL DEFAULT 0.0,
                        recorded_at TEXT    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_outcomes_strategy
                    ON outcomes(strategy)
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        strategy: str,
        won: bool,
        pnl: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a trade outcome.

        Parameters
        ----------
        strategy : str
            Strategy name.
        won : bool
            Whether the trade was a winner.
        pnl : float
            Realised P&L of the trade.
        timestamp : datetime | None
            UTC timestamp.  Defaults to now.
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO outcomes (strategy, won, pnl, recorded_at) VALUES (?, ?, ?, ?)",
                    (strategy, int(won), pnl, ts),
                )
                conn.commit()
            finally:
                conn.close()
        logger.debug("Recorded outcome: %s won=%s pnl=%.4f", strategy, won, pnl)

    def get_current_streak(self, strategy: str) -> Streak:
        """Return the current win or loss streak for a strategy.

        Parameters
        ----------
        strategy : str
            Strategy name.

        Returns
        -------
        Streak
            Current streak with type, length, and cumulative P&L.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT won, pnl FROM outcomes WHERE strategy = ? ORDER BY id DESC",
                    (strategy,),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return Streak(type="win", length=0, total_pnl=0.0)

        streak_type = "win" if rows[0]["won"] else "loss"
        length = 0
        total_pnl = 0.0

        for row in rows:
            current = "win" if row["won"] else "loss"
            if current != streak_type:
                break
            length += 1
            total_pnl += row["pnl"]

        return Streak(type=streak_type, length=length, total_pnl=total_pnl)

    def runs_test(self, strategy: str, lookback: int = 100) -> RunsTestResult:
        """Apply the Wald-Wolfowitz runs test for randomness.

        The runs test checks whether the sequence of wins and losses is
        consistent with a random (independent) sequence.  A significantly
        low number of runs suggests clustering (momentum/streakiness);
        a high number suggests mean reversion.

        Parameters
        ----------
        strategy : str
            Strategy name.
        lookback : int
            Number of most recent outcomes to analyse.

        Returns
        -------
        RunsTestResult
            Z-score, p-value, whether the sequence looks random, and
            a human-readable interpretation.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT won FROM outcomes WHERE strategy = ? ORDER BY id DESC LIMIT ?",
                    (strategy, lookback),
                ).fetchall()
            finally:
                conn.close()

        if len(rows) < 10:
            return RunsTestResult(
                z_score=0.0, p_value=1.0, random=True,
                interpretation="Insufficient data (need >= 10 outcomes)",
            )

        # Reverse to chronological order
        sequence = [r["won"] for r in reversed(rows)]
        n = len(sequence)
        n_wins = sum(sequence)
        n_losses = n - n_wins

        if n_wins == 0 or n_losses == 0:
            return RunsTestResult(
                z_score=0.0, p_value=1.0, random=False,
                interpretation="All outcomes are the same — no variance to test",
            )

        # Count runs
        runs = 1
        for i in range(1, n):
            if sequence[i] != sequence[i - 1]:
                runs += 1

        # Expected runs and variance under null hypothesis (random)
        expected_runs = (2.0 * n_wins * n_losses) / n + 1.0
        numer = 2.0 * n_wins * n_losses * (2.0 * n_wins * n_losses - n)
        denom = n * n * (n - 1.0)
        variance = numer / denom if denom > 0 else 0.0

        if variance <= 0:
            return RunsTestResult(
                z_score=0.0, p_value=1.0, random=True,
                interpretation="Variance is zero — cannot compute z-score",
            )

        z_score = (runs - expected_runs) / math.sqrt(variance)

        # Two-tailed p-value approximation using standard normal CDF
        p_value = 2.0 * (1.0 - self._normal_cdf(abs(z_score)))
        is_random = p_value > 0.05

        if is_random:
            interpretation = f"Sequence appears random (z={z_score:.2f}, p={p_value:.3f})"
        elif z_score < 0:
            interpretation = (
                f"Streaky — fewer runs than expected (z={z_score:.2f}, p={p_value:.3f}). "
                "Outcomes cluster: wins follow wins, losses follow losses."
            )
        else:
            interpretation = (
                f"Mean-reverting — more runs than expected (z={z_score:.2f}, p={p_value:.3f}). "
                "Wins and losses alternate more than random."
            )

        return RunsTestResult(
            z_score=z_score,
            p_value=p_value,
            random=is_random,
            interpretation=interpretation,
        )

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate standard normal CDF using the Abramowitz & Stegun formula."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def get_streak_stats(self, strategy: str) -> Dict[str, float]:
        """Compute streak statistics for a strategy.

        Returns
        -------
        dict
            Keys: ``longest_win``, ``longest_loss``, ``avg_win_streak``,
            ``avg_loss_streak``.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT won FROM outcomes WHERE strategy = ? ORDER BY id ASC",
                    (strategy,),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return {
                "longest_win": 0,
                "longest_loss": 0,
                "avg_win_streak": 0.0,
                "avg_loss_streak": 0.0,
            }

        win_streaks: List[int] = []
        loss_streaks: List[int] = []
        current_len = 1
        current_type = rows[0]["won"]

        for i in range(1, len(rows)):
            if rows[i]["won"] == current_type:
                current_len += 1
            else:
                if current_type:
                    win_streaks.append(current_len)
                else:
                    loss_streaks.append(current_len)
                current_len = 1
                current_type = rows[i]["won"]

        # Append the final streak
        if current_type:
            win_streaks.append(current_len)
        else:
            loss_streaks.append(current_len)

        return {
            "longest_win": max(win_streaks) if win_streaks else 0,
            "longest_loss": max(loss_streaks) if loss_streaks else 0,
            "avg_win_streak": (sum(win_streaks) / len(win_streaks)) if win_streaks else 0.0,
            "avg_loss_streak": (sum(loss_streaks) / len(loss_streaks)) if loss_streaks else 0.0,
        }

    def is_on_tilt(self, strategy: str, loss_streak_threshold: int = 5) -> bool:
        """Check if a strategy is on tilt (extended losing streak).

        Parameters
        ----------
        strategy : str
            Strategy name.
        loss_streak_threshold : int
            Number of consecutive losses to trigger tilt.

        Returns
        -------
        bool
            ``True`` if the strategy is currently on a losing streak of at
            least ``loss_streak_threshold`` trades.
        """
        streak = self.get_current_streak(strategy)
        on_tilt = streak.type == "loss" and streak.length >= loss_streak_threshold
        if on_tilt:
            logger.warning(
                "Strategy %s is ON TILT: %d consecutive losses (pnl=%.4f)",
                strategy, streak.length, streak.total_pnl,
            )
        return on_tilt
