"""
adaptive/strategy_decay_detector.py — Strategy Alpha Decay Detection

Monitors per-strategy trade P&L streams, computes rolling Sharpe ratios,
and detects when alpha is decaying (declining Sharpe trend).  Provides
allocation multipliers so the capital allocator can automatically
reduce exposure to weakening strategies.

Classes
-------
DecayReport     Immutable dataclass for a decay analysis result.
StrategyDecayDetector
    Record trades, detect decay, recommend allocation adjustments.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecayReport:
    """Result of a strategy decay analysis."""
    strategy: str
    decaying: bool
    sharpe_trend: float          # slope of rolling Sharpe (negative = decaying)
    current_sharpe: float
    peak_sharpe: float
    drawdown_from_peak_pct: float
    recommendation: str          # "maintain", "reduce", or "disable"
    timestamp: str               # ISO-8601 UTC


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS strategy_trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy   TEXT    NOT NULL,
    pnl        REAL    NOT NULL,
    timestamp  TEXT    NOT NULL
);
"""

_IDX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_st_strategy ON strategy_trades(strategy);",
    "CREATE INDEX IF NOT EXISTS idx_st_ts ON strategy_trades(timestamp);",
]


class StrategyDecayDetector:
    """Detect alpha decay in strategies using rolling Sharpe analysis.

    Parameters
    ----------
    db_path : str, optional
        Override the default ``data/strategy_decay.db`` path.
    decay_slope_threshold : float
        Sharpe slope below this triggers a *decaying* flag.
    disable_sharpe : float
        If current Sharpe drops below this, recommend "disable".
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        decay_slope_threshold: float = -0.1,
        disable_sharpe: float = -0.5,
    ) -> None:
        self._db_path = db_path or os.path.join(_DB_DIR, "strategy_decay.db")
        self._decay_slope = decay_slope_threshold
        self._disable_sharpe = disable_sharpe
        self._lock = threading.Lock()
        self._ensure_schema()
        log.info(
            "StrategyDecayDetector initialised  db=%s  slope_threshold=%.2f",
            self._db_path, self._decay_slope,
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

    def record_trade(
        self,
        strategy: str,
        pnl: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a completed trade's P&L for a strategy.

        Parameters
        ----------
        strategy : str
            Strategy identifier (e.g. ``"momentum_cross"``).
        pnl : float
            Realised profit/loss in base currency.
        timestamp : datetime, optional
            Trade close time; defaults to now UTC.
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        with self._lock:
            con = self._connect()
            try:
                con.execute(
                    "INSERT INTO strategy_trades (strategy, pnl, timestamp) VALUES (?, ?, ?)",
                    (strategy, pnl, ts),
                )
                con.commit()
            finally:
                con.close()
        log.debug("Recorded trade for %s: pnl=%.4f", strategy, pnl)

    def detect_decay(
        self,
        strategy: str,
        window_trades: int = 50,
        min_trades: int = 20,
    ) -> DecayReport:
        """Analyse alpha decay for a strategy.

        Uses the last *window_trades* trades.  A rolling 20-trade Sharpe
        is computed and its linear slope determines whether the strategy
        is decaying.

        Parameters
        ----------
        strategy : str
        window_trades : int
            Number of recent trades to analyse.
        min_trades : int
            Minimum trades required; below this, returns a *healthy* report.

        Returns
        -------
        DecayReport
        """
        pnl_list = self._fetch_recent_pnl(strategy, window_trades)
        ts_now = datetime.now(timezone.utc).isoformat()

        if len(pnl_list) < min_trades:
            log.debug("Insufficient trades (%d/%d) for decay detection on %s", len(pnl_list), min_trades, strategy)
            return DecayReport(
                strategy=strategy,
                decaying=False,
                sharpe_trend=0.0,
                current_sharpe=0.0,
                peak_sharpe=0.0,
                drawdown_from_peak_pct=0.0,
                recommendation="maintain",
                timestamp=ts_now,
            )

        # Compute rolling 20-trade Sharpe ratios
        rolling_window = 20
        sharpes: List[float] = []
        for i in range(len(pnl_list) - rolling_window + 1):
            window = pnl_list[i : i + rolling_window]
            sharpes.append(self._sharpe(window))

        if not sharpes:
            return DecayReport(
                strategy=strategy, decaying=False, sharpe_trend=0.0,
                current_sharpe=0.0, peak_sharpe=0.0,
                drawdown_from_peak_pct=0.0, recommendation="maintain",
                timestamp=ts_now,
            )

        # Linear slope of rolling Sharpe
        slope = self._linear_slope(sharpes)
        current_sharpe = sharpes[-1]
        peak_sharpe = max(sharpes)
        dd_pct = ((peak_sharpe - current_sharpe) / abs(peak_sharpe) * 100.0) if peak_sharpe != 0 else 0.0

        decaying = slope < self._decay_slope

        # Recommendation
        if current_sharpe < self._disable_sharpe:
            recommendation = "disable"
        elif decaying:
            recommendation = "reduce"
        else:
            recommendation = "maintain"

        report = DecayReport(
            strategy=strategy,
            decaying=decaying,
            sharpe_trend=round(slope, 6),
            current_sharpe=round(current_sharpe, 4),
            peak_sharpe=round(peak_sharpe, 4),
            drawdown_from_peak_pct=round(dd_pct, 2),
            recommendation=recommendation,
            timestamp=ts_now,
        )

        if decaying:
            log.warning(
                "Alpha decay detected for %s: slope=%.4f  sharpe=%.3f→%.3f  rec=%s",
                strategy, slope, peak_sharpe, current_sharpe, recommendation,
            )
        else:
            log.debug("Strategy %s healthy: slope=%.4f  sharpe=%.3f", strategy, slope, current_sharpe)

        return report

    def get_allocation_multiplier(self, strategy: str) -> float:
        """Return a 0.0–1.0 multiplier for position sizing.

        - 1.0 = healthy (maintain full allocation)
        - 0.5 = decaying (reduce allocation)
        - 0.0 = disabled (zero allocation)
        """
        report = self.detect_decay(strategy)
        if report.recommendation == "disable":
            return 0.0
        elif report.recommendation == "reduce":
            return 0.5
        return 1.0

    def get_all_strategies_health(self) -> Dict[str, DecayReport]:
        """Return a DecayReport for every known strategy."""
        strategies = self._get_all_strategy_names()
        return {s: self.detect_decay(s) for s in strategies}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_recent_pnl(self, strategy: str, limit: int) -> List[float]:
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT pnl FROM strategy_trades WHERE strategy = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (strategy, limit),
                ).fetchall()
            finally:
                con.close()
        # Reverse so oldest first
        return [r[0] for r in reversed(rows)]

    def _get_all_strategy_names(self) -> List[str]:
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT DISTINCT strategy FROM strategy_trades ORDER BY strategy"
                ).fetchall()
            finally:
                con.close()
        return [r[0] for r in rows]

    @staticmethod
    def _sharpe(returns: List[float]) -> float:
        """Compute Sharpe ratio (annualised not needed here — relative scale)."""
        if not returns:
            return 0.0
        n = len(returns)
        mean = sum(returns) / n
        var = sum((r - mean) ** 2 for r in returns) / n
        std = math.sqrt(var) if var > 0 else 1e-9
        return mean / std

    @staticmethod
    def _linear_slope(values: List[float]) -> float:
        """Compute the OLS slope of *values* against index."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0
