"""
P&L Attribution Engine — decomposes realised P&L into alpha, beta, timing,
execution cost, slippage, and fee components.

Each recorded trade is tagged with a market-return benchmark so that the
engine can separate skill (alpha) from passive exposure (beta).

Persistence: SQLite at ``data/attribution.db``.

Usage::

    engine = AttributionEngine()
    engine.record_trade("BTC/AUD", "momentum", 60000, 61200, 0.1,
                        market_return_pct=1.5, slippage_bps=3.0, fees_usd=1.2)
    attr = engine.decompose(lookback_days=30)
    logger.info(attr.alpha_pnl, attr.beta_pnl)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "attribution.db"


@dataclass
class Attribution:
    """Decomposed P&L attribution result."""

    total_pnl: float = 0.0
    alpha_pnl: float = 0.0
    beta_pnl: float = 0.0
    timing_pnl: float = 0.0
    execution_cost: float = 0.0
    slippage_cost: float = 0.0
    fee_cost: float = 0.0
    residual: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"Attribution(total={self.total_pnl:.4f}, alpha={self.alpha_pnl:.4f}, "
            f"beta={self.beta_pnl:.4f}, timing={self.timing_pnl:.4f}, "
            f"exec_cost={self.execution_cost:.4f}, slippage={self.slippage_cost:.4f}, "
            f"fees={self.fee_cost:.4f}, residual={self.residual:.4f})"
        )


class AttributionEngine:
    """Records trades and decomposes P&L into factor-based attribution buckets.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the SQLite database.  Defaults to ``data/attribution.db``.
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
        logger.info("AttributionEngine initialised, db=%s", self._db_path)

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL mode enabled."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol          TEXT    NOT NULL,
                        strategy        TEXT    NOT NULL,
                        entry_price     REAL    NOT NULL,
                        exit_price      REAL    NOT NULL,
                        size            REAL    NOT NULL,
                        market_return_pct REAL  NOT NULL DEFAULT 0.0,
                        slippage_bps    REAL    NOT NULL DEFAULT 0.0,
                        fees_usd        REAL    NOT NULL DEFAULT 0.0,
                        trade_pnl       REAL    NOT NULL DEFAULT 0.0,
                        recorded_at     TEXT    NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_strategy
                    ON trades(strategy)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trades_recorded_at
                    ON trades(recorded_at)
                """)
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        exit_price: float,
        size: float,
        market_return_pct: float = 0.0,
        slippage_bps: float = 0.0,
        fees_usd: float = 0.0,
    ) -> None:
        """Record a completed trade for later attribution analysis.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTC/AUD"``.
        strategy : str
            Name of the strategy that generated the signal.
        entry_price : float
            Entry fill price.
        exit_price : float
            Exit fill price.
        size : float
            Absolute position size in base units.
        market_return_pct : float
            Benchmark market return over the trade's holding period (%).
        slippage_bps : float
            Observed slippage in basis points.
        fees_usd : float
            Total fees paid in USD.
        """
        trade_pnl = (exit_price - entry_price) * size
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO trades
                       (symbol, strategy, entry_price, exit_price, size,
                        market_return_pct, slippage_bps, fees_usd, trade_pnl, recorded_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, strategy, entry_price, exit_price, size,
                     market_return_pct, slippage_bps, fees_usd, trade_pnl, now),
                )
                conn.commit()
                logger.debug(
                    "Recorded trade: %s %s pnl=%.4f", symbol, strategy, trade_pnl,
                )
            finally:
                conn.close()

    def decompose(self, lookback_days: int = 30) -> Attribution:
        """Decompose aggregate P&L into attribution components.

        Parameters
        ----------
        lookback_days : int
            Number of days to look back from now.

        Returns
        -------
        Attribution
            Decomposed P&L attribution.

        The decomposition works as follows:

        * **total_pnl** = sum of all ``(exit - entry) * size``
        * **beta_pnl** = sum of ``entry_price * size * market_return_pct / 100`` — the
          P&L you would have earned from passive market exposure.
        * **slippage_cost** = sum of ``entry_price * size * slippage_bps / 10_000``
        * **fee_cost** = sum of ``fees_usd``
        * **execution_cost** = slippage_cost + fee_cost
        * **timing_pnl** = estimated timing effect (residual correlation proxy)
        * **alpha_pnl** = total_pnl - beta_pnl - timing_pnl + execution_cost
        * **residual** = total_pnl - alpha_pnl - beta_pnl - timing_pnl + execution_cost
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE recorded_at >= ? ORDER BY recorded_at",
                    (cutoff,),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            logger.info("decompose: no trades in lookback window (%d days)", lookback_days)
            return Attribution()

        total_pnl = 0.0
        beta_pnl = 0.0
        slippage_cost = 0.0
        fee_cost = 0.0
        timing_deltas: list[float] = []

        for row in rows:
            pnl = row["trade_pnl"]
            total_pnl += pnl

            # Beta component: what the position would have earned from market movement
            notional = row["entry_price"] * row["size"]
            beta_component = notional * row["market_return_pct"] / 100.0
            beta_pnl += beta_component

            # Slippage cost
            slip = notional * row["slippage_bps"] / 10_000.0
            slippage_cost += slip

            # Fees
            fee_cost += row["fees_usd"]

            # Timing effect: difference between actual P&L direction and beta direction
            # Positive timing means good entry/exit timing
            if abs(beta_component) > 1e-12:
                timing_deltas.append(pnl - beta_component)

        execution_cost = slippage_cost + fee_cost

        # Timing P&L: portion of excess return attributable to entry/exit timing
        # We estimate this as a fraction of the timing deltas (conservative 20% attribution)
        timing_pnl = sum(timing_deltas) * 0.2 if timing_deltas else 0.0

        # Alpha = what's left after removing beta, timing, and adding back costs
        alpha_pnl = total_pnl - beta_pnl - timing_pnl + execution_cost

        # Residual = rounding / unattributed
        residual = total_pnl - alpha_pnl - beta_pnl - timing_pnl + execution_cost

        attr = Attribution(
            total_pnl=total_pnl,
            alpha_pnl=alpha_pnl,
            beta_pnl=beta_pnl,
            timing_pnl=timing_pnl,
            execution_cost=execution_cost,
            slippage_cost=slippage_cost,
            fee_cost=fee_cost,
            residual=residual,
            timestamp=datetime.now(timezone.utc),
        )
        logger.info("decompose: %s", attr)
        return attr

    def get_strategy_attribution(self, strategy: str) -> Attribution:
        """Decompose P&L for a single strategy.

        Parameters
        ----------
        strategy : str
            Strategy name to filter on.

        Returns
        -------
        Attribution
            Attribution scoped to the given strategy.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE strategy = ? ORDER BY recorded_at",
                    (strategy,),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            logger.info("get_strategy_attribution: no trades for strategy=%s", strategy)
            return Attribution()

        total_pnl = 0.0
        beta_pnl = 0.0
        slippage_cost = 0.0
        fee_cost = 0.0

        for row in rows:
            total_pnl += row["trade_pnl"]
            notional = row["entry_price"] * row["size"]
            beta_pnl += notional * row["market_return_pct"] / 100.0
            slippage_cost += notional * row["slippage_bps"] / 10_000.0
            fee_cost += row["fees_usd"]

        execution_cost = slippage_cost + fee_cost
        alpha_pnl = total_pnl - beta_pnl + execution_cost
        residual = 0.0

        return Attribution(
            total_pnl=total_pnl,
            alpha_pnl=alpha_pnl,
            beta_pnl=beta_pnl,
            timing_pnl=0.0,
            execution_cost=execution_cost,
            slippage_cost=slippage_cost,
            fee_cost=fee_cost,
            residual=residual,
            timestamp=datetime.now(timezone.utc),
        )

    def get_best_alpha_source(self) -> str:
        """Return the strategy name with the highest alpha contribution.

        Returns
        -------
        str
            Strategy name.  Returns ``""`` if no trades exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT strategy FROM trades"
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            logger.info("get_best_alpha_source: no strategies found")
            return ""

        best_strategy = ""
        best_alpha = float("-inf")

        for row in rows:
            strat = row["strategy"]
            attr = self.get_strategy_attribution(strat)
            if attr.alpha_pnl > best_alpha:
                best_alpha = attr.alpha_pnl
                best_strategy = strat

        logger.info(
            "get_best_alpha_source: %s (alpha=%.4f)", best_strategy, best_alpha
        )
        return best_strategy
