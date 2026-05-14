"""
risk/kelly_position_sizer.py — Kelly Criterion Position Sizing

Implements the Kelly criterion for optimal position sizing based on
historical win rate and average win/loss ratio per strategy.

Half-Kelly (the default recommendation) halves the theoretical optimal
fraction to reduce variance — widely regarded as best practice for
live trading.

Requires a minimum of 20 trades before Kelly kicks in; until then a
conservative default fraction is returned.

Persistence: SQLite at ``data/kelly_outcomes.db``.

Usage::

    sizer = KellyPositionSizer()
    sizer.update_outcome("momentum", won=True, return_pct=1.2)
    sizer.update_outcome("momentum", won=False, return_pct=-0.8)
    frac = sizer.get_half_kelly("momentum")
    size = sizer.get_position_size_usd("momentum", capital_usd=10000)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.path.join("data", "kelly_outcomes.db")
_MIN_TRADES = 20                # minimum observations before Kelly activates
_CONSERVATIVE_DEFAULT = 0.01    # 1% of capital when insufficient data
_MAX_KELLY = 0.25              # hard cap on raw Kelly fraction


class KellyPositionSizer:
    """Kelly-criterion position sizer with SQLite persistence.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database for outcome storage.
    min_trades : int
        Minimum trade count before Kelly fraction is used.
    conservative_default : float
        Fraction returned when fewer than *min_trades* observations exist.
    max_kelly : float
        Hard cap on the raw Kelly fraction (guards against estimation error).
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        min_trades: int = _MIN_TRADES,
        conservative_default: float = _CONSERVATIVE_DEFAULT,
        max_kelly: float = _MAX_KELLY,
    ) -> None:
        self._db_path = db_path
        self._min_trades = min_trades
        self._conservative_default = conservative_default
        self._max_kelly = max_kelly
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        log.info(
            "KellyPositionSizer initialised — db=%s min_trades=%d",
            db_path,
            min_trades,
        )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outcomes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL    NOT NULL,
                    strategy    TEXT    NOT NULL,
                    won         INTEGER NOT NULL,
                    return_pct  REAL    NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcomes_strat ON outcomes(strategy)"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Record outcomes
    # ------------------------------------------------------------------

    def update_outcome(
        self,
        strategy: str,
        won: bool,
        return_pct: float,
    ) -> None:
        """Record a single trade outcome.

        Parameters
        ----------
        strategy : str
            Strategy name.
        won : bool
            Whether the trade was profitable.
        return_pct : float
            Return expressed as a percentage (e.g. ``1.5`` for +1.5%,
            ``-0.8`` for -0.8%).
        """
        ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO outcomes (ts, strategy, won, return_pct) VALUES (?, ?, ?, ?)",
                (ts, strategy, int(won), return_pct),
            )
            self._conn.commit()
        log.debug(
            "Kelly outcome: strategy=%s won=%s return=%.2f%%",
            strategy,
            won,
            return_pct,
        )

    # ------------------------------------------------------------------
    # Kelly computation
    # ------------------------------------------------------------------

    def _get_stats(self, strategy: str) -> Tuple[int, float, float, float]:
        """Return (n_trades, win_rate, avg_win_pct, avg_loss_pct) for a strategy.

        ``avg_loss_pct`` is returned as a positive number (absolute value).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT won, return_pct FROM outcomes WHERE strategy = ?",
                (strategy,),
            ).fetchall()

        if not rows:
            return 0, 0.0, 0.0, 0.0

        wins = [r[1] for r in rows if r[0] == 1]
        losses = [r[1] for r in rows if r[0] == 0]
        n = len(rows)
        win_rate = len(wins) / n if n > 0 else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        return n, win_rate, avg_win, avg_loss

    def get_kelly_fraction(self, strategy: str) -> float:
        """Return the full Kelly fraction for *strategy*.

        The Kelly formula is::

            f* = p / l  -  q / w

        where *p* = win probability, *q* = 1-p, *w* = avg win (ratio),
        *l* = avg loss (ratio).

        If there are fewer than *min_trades* observations, returns the
        conservative default.

        Returns
        -------
        float
            Kelly fraction in ``[0, max_kelly]``.  Returns ``0.0`` if edge
            is negative.
        """
        n, win_rate, avg_win, avg_loss = self._get_stats(strategy)

        if n < self._min_trades:
            log.debug(
                "Kelly: strategy '%s' has %d trades (< %d) — using conservative %.2f%%",
                strategy,
                n,
                self._min_trades,
                self._conservative_default * 100,
            )
            return self._conservative_default

        if avg_win <= 0 or avg_loss <= 0:
            log.debug("Kelly: strategy '%s' avg_win or avg_loss is 0 — returning default", strategy)
            return self._conservative_default

        # Kelly formula: f* = W/L - (1-W)/G
        # Where W = win_rate, L = avg_loss (as ratio), G = avg_win (as ratio)
        # Using percent returns: convert to ratios
        w_ratio = avg_win / 100.0  # e.g. 1.5% → 0.015
        l_ratio = avg_loss / 100.0

        if l_ratio == 0:
            return self._conservative_default

        kelly = win_rate / l_ratio - (1.0 - win_rate) / w_ratio

        if kelly <= 0:
            log.info(
                "Kelly: strategy '%s' has negative edge (kelly=%.4f) — returning 0",
                strategy,
                kelly,
            )
            return 0.0

        capped = min(kelly, self._max_kelly)
        log.debug(
            "Kelly: strategy '%s' raw=%.4f capped=%.4f  "
            "(n=%d wr=%.2f avgW=%.2f%% avgL=%.2f%%)",
            strategy,
            kelly,
            capped,
            n,
            win_rate,
            avg_win,
            avg_loss,
        )
        return capped

    def get_half_kelly(self, strategy: str) -> float:
        """Return half-Kelly fraction — the recommended sizing for live use.

        Half-Kelly achieves ~75% of the optimal growth rate with ~50% of
        the variance.

        Returns
        -------
        float
            Half of the full Kelly fraction.
        """
        return self.get_kelly_fraction(strategy) / 2.0

    def get_position_size_usd(
        self,
        strategy: str,
        capital_usd: float,
        max_pct: float = 0.05,
    ) -> float:
        """Return the dollar size for the next trade.

        Uses half-Kelly, capped at *max_pct* of capital.

        Parameters
        ----------
        strategy : str
            Strategy name.
        capital_usd : float
            Current account equity in USD.
        max_pct : float
            Maximum fraction of capital per trade (default 5%).

        Returns
        -------
        float
            Position size in USD.
        """
        frac = self.get_half_kelly(strategy)
        capped_frac = min(frac, max_pct)
        size = capital_usd * capped_frac
        log.debug(
            "Kelly size: strategy=%s capital=$%.0f half_kelly=%.4f "
            "capped=%.4f size=$%.2f",
            strategy,
            capital_usd,
            frac,
            capped_frac,
            size,
        )
        return size

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_trade_count(self, strategy: str) -> int:
        """Return number of recorded outcomes for *strategy*."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM outcomes WHERE strategy = ?",
                (strategy,),
            ).fetchone()
        return row[0] if row else 0

    def get_all_strategies(self) -> List[str]:
        """Return list of all strategies with recorded outcomes."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT strategy FROM outcomes ORDER BY strategy"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()
        log.info("KellyPositionSizer: database closed")
