"""
Nightly Continuous Backtesting — automated strategy health monitoring.

Runs every strategy against recent OHLCV data using a walk-forward window,
persists results to SQLite, and identifies strategies that should be demoted
(failing pass criteria for N consecutive nights) or promoted (passing after
a previous demotion).

Pass criteria (all must hold):
  - Sharpe ratio > 0.3
  - Max drawdown < 25%
  - Profit factor > 1.1

Usage:
    from backtesting.continuous_backtester import ContinuousBacktester

    bt = ContinuousBacktester()
    report = bt.run_nightly(
        strategies=["momentum_v2", "mean_reversion"],
        ohlcv_data={"BTC/USD": [{"t": ..., "o": ..., "h": ..., "l": ..., "c": ..., "v": ...}, ...]},
        walk_forward_days=30,
    )
    print(report.demoted)    # strategies to remove from live
    print(report.promoted)   # strategies passing again after demotion
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Performance metrics for a single strategy backtest run."""

    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    profit_factor: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BacktestMetrics":
        return cls(
            sharpe=float(d.get("sharpe", 0.0)),
            max_drawdown_pct=float(d.get("max_drawdown_pct", 0.0)),
            win_rate=float(d.get("win_rate", 0.0)),
            trade_count=int(d.get("trade_count", 0)),
            profit_factor=float(d.get("profit_factor", 0.0)),
            passed=bool(d.get("passed", False)),
        )


@dataclass
class NightlyReport:
    """Result of a nightly continuous backtest run."""

    timestamp: str = ""
    strategy_results: Dict[str, BacktestMetrics] = field(default_factory=dict)
    demoted: List[str] = field(default_factory=list)
    promoted: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pass criteria defaults
# ---------------------------------------------------------------------------

_DEFAULT_MIN_SHARPE = 0.3
_DEFAULT_MAX_DRAWDOWN_PCT = 25.0
_DEFAULT_MIN_PROFIT_FACTOR = 1.1
_DEFAULT_DEMOTION_CONSECUTIVE = 3


# ---------------------------------------------------------------------------
# ContinuousBacktester
# ---------------------------------------------------------------------------

class ContinuousBacktester:
    """Nightly automated strategy backtester with SQLite persistence.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database.  Created automatically if absent.
    min_sharpe : float
        Minimum Sharpe ratio to pass.
    max_drawdown_pct : float
        Maximum drawdown percentage to pass.
    min_profit_factor : float
        Minimum profit factor to pass.
    demotion_consecutive : int
        Number of consecutive failing nights before demotion.
    """

    def __init__(
        self,
        db_path: str = "data/continuous_backtest.db",
        *,
        min_sharpe: float = _DEFAULT_MIN_SHARPE,
        max_drawdown_pct: float = _DEFAULT_MAX_DRAWDOWN_PCT,
        min_profit_factor: float = _DEFAULT_MIN_PROFIT_FACTOR,
        demotion_consecutive: int = _DEFAULT_DEMOTION_CONSECUTIVE,
    ) -> None:
        self.db_path = Path(db_path)
        self.min_sharpe = min_sharpe
        self.max_drawdown_pct = max_drawdown_pct
        self.min_profit_factor = min_profit_factor
        self.demotion_consecutive = demotion_consecutive

        self._lock = threading.Lock()
        self._ensure_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create SQLite tables if they do not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy    TEXT    NOT NULL,
                    run_ts      TEXT    NOT NULL,
                    sharpe      REAL    NOT NULL,
                    max_dd_pct  REAL    NOT NULL,
                    win_rate    REAL    NOT NULL,
                    trade_count INTEGER NOT NULL,
                    profit_factor REAL  NOT NULL,
                    passed      INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bt_strategy_ts
                ON backtest_runs(strategy, run_ts)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS demotion_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy    TEXT    NOT NULL,
                    action      TEXT    NOT NULL,
                    ts          TEXT    NOT NULL,
                    reason      TEXT    DEFAULT ''
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ------------------------------------------------------------------
    # Core backtest logic
    # ------------------------------------------------------------------

    def _evaluate_strategy(
        self,
        strategy_name: str,
        ohlcv_data: Dict[str, List[Dict[str, Any]]],
        walk_forward_days: int,
    ) -> BacktestMetrics:
        """Run a simple walk-forward backtest for a single strategy.

        This uses a momentum-style evaluation over the most recent
        *walk_forward_days* bars of each symbol's data.  Subclass and
        override this method to plug in a custom strategy runner.

        Parameters
        ----------
        strategy_name : str
            Strategy identifier (used for logging).
        ohlcv_data : dict
            Mapping of symbol -> list of OHLCV dicts.
        walk_forward_days : int
            Number of recent bars to evaluate.

        Returns
        -------
        BacktestMetrics
        """
        all_returns: List[float] = []
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0

        for symbol, bars in ohlcv_data.items():
            if not bars:
                continue
            window = bars[-walk_forward_days:] if len(bars) > walk_forward_days else bars
            for i in range(1, len(window)):
                prev_close = float(window[i - 1].get("c", window[i - 1].get("close", 0)))
                curr_close = float(window[i].get("c", window[i].get("close", 0)))
                if prev_close <= 0:
                    continue
                ret = (curr_close - prev_close) / prev_close
                all_returns.append(ret)
                if ret > 0:
                    wins += 1
                    gross_profit += ret
                elif ret < 0:
                    losses += 1
                    gross_loss += abs(ret)

        trade_count = wins + losses
        if trade_count == 0:
            return BacktestMetrics(
                sharpe=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                trade_count=0,
                profit_factor=0.0,
                passed=False,
            )

        # Sharpe ratio (annualised assuming daily returns)
        mean_ret = sum(all_returns) / len(all_returns)
        std_ret = (sum((r - mean_ret) ** 2 for r in all_returns) / len(all_returns)) ** 0.5
        sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 1e-12 else 0.0

        # Max drawdown
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in all_returns:
            equity *= (1 + r)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        win_rate = wins / trade_count if trade_count > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        passed = (
            sharpe > self.min_sharpe
            and max_dd < self.max_drawdown_pct
            and profit_factor > self.min_profit_factor
        )

        return BacktestMetrics(
            sharpe=round(sharpe, 4),
            max_drawdown_pct=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            trade_count=trade_count,
            profit_factor=round(profit_factor, 4),
            passed=passed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_nightly(
        self,
        strategies: List[str],
        ohlcv_data: Dict[str, List[Dict[str, Any]]],
        walk_forward_days: int = 30,
    ) -> NightlyReport:
        """Execute a full nightly backtest run for all strategies.

        Parameters
        ----------
        strategies : list of str
            Strategy names to evaluate.
        ohlcv_data : dict
            Mapping of symbol -> list of OHLCV dicts (keys: t/o/h/l/c/v or
            timestamp/open/high/low/close/volume).
        walk_forward_days : int
            Number of recent bars used in the walk-forward window.

        Returns
        -------
        NightlyReport
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        report = NightlyReport(timestamp=now_ts)

        if not strategies:
            report.warnings.append("No strategies provided")
            logger.warning("ContinuousBacktester: no strategies provided")
            return report

        if not ohlcv_data:
            report.warnings.append("No OHLCV data provided")
            logger.warning("ContinuousBacktester: no OHLCV data provided")
            return report

        logger.info(
            "ContinuousBacktester: running nightly for %d strategies, %d symbols, window=%d",
            len(strategies), len(ohlcv_data), walk_forward_days,
        )

        for strat in strategies:
            try:
                metrics = self._evaluate_strategy(strat, ohlcv_data, walk_forward_days)
                report.strategy_results[strat] = metrics
                self._persist_run(strat, now_ts, metrics)
                if not metrics.passed:
                    report.warnings.append(
                        f"{strat} failed pass criteria: sharpe={metrics.sharpe}, "
                        f"dd={metrics.max_drawdown_pct}%, pf={metrics.profit_factor}"
                    )
                logger.info(
                    "ContinuousBacktester: %s — sharpe=%.4f dd=%.2f%% pf=%.4f passed=%s",
                    strat, metrics.sharpe, metrics.max_drawdown_pct, metrics.profit_factor, metrics.passed,
                )
            except Exception:
                logger.exception("ContinuousBacktester: error evaluating %s", strat)
                report.warnings.append(f"Error evaluating {strat}")

        # Demotion / promotion logic
        report.demoted = self.get_demotion_candidates()
        report.promoted = self._get_promotion_candidates(strategies)

        for s in report.demoted:
            self._log_demotion_action(s, "demoted", "Failed pass criteria for consecutive nights")
        for s in report.promoted:
            self._log_demotion_action(s, "promoted", "Now passing after previous demotion")

        logger.info(
            "ContinuousBacktester: nightly complete — demoted=%s promoted=%s warnings=%d",
            report.demoted, report.promoted, len(report.warnings),
        )
        return report

    def get_demotion_candidates(self) -> List[str]:
        """Return strategies failing pass criteria for N consecutive nights.

        Returns
        -------
        list of str
            Strategy names that should be demoted.
        """
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT strategy FROM backtest_runs"
                )
                all_strategies = [row[0] for row in cursor.fetchall()]

        demoted: List[str] = []
        for strat in all_strategies:
            history = self.get_history(strat, lookback_days=self.demotion_consecutive)
            if len(history) < self.demotion_consecutive:
                continue
            recent = history[-self.demotion_consecutive:]
            if all(not m.passed for m in recent):
                demoted.append(strat)

        return demoted

    def _get_promotion_candidates(self, strategies: List[str]) -> List[str]:
        """Find strategies that were previously demoted but now pass.

        Parameters
        ----------
        strategies : list of str
            Strategies evaluated in this run.

        Returns
        -------
        list of str
        """
        promoted: List[str] = []
        with self._lock:
            with self._connect() as conn:
                for strat in strategies:
                    cursor = conn.execute(
                        "SELECT action FROM demotion_log WHERE strategy = ? ORDER BY ts DESC LIMIT 1",
                        (strat,),
                    )
                    row = cursor.fetchone()
                    if row and row[0] == "demoted":
                        # Check if latest run passed
                        hist = self.get_history(strat, lookback_days=1)
                        if hist and hist[-1].passed:
                            promoted.append(strat)
        return promoted

    def get_history(self, strategy: str, lookback_days: int = 30) -> List[BacktestMetrics]:
        """Get backtest history for a strategy.

        Parameters
        ----------
        strategy : str
            Strategy name.
        lookback_days : int
            Number of most recent runs to return.

        Returns
        -------
        list of BacktestMetrics
            Ordered oldest-first.
        """
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    SELECT sharpe, max_dd_pct, win_rate, trade_count, profit_factor, passed
                    FROM backtest_runs
                    WHERE strategy = ?
                    ORDER BY run_ts DESC
                    LIMIT ?
                    """,
                    (strategy, lookback_days),
                )
                rows = cursor.fetchall()

        # Reverse to oldest-first
        rows.reverse()
        return [
            BacktestMetrics(
                sharpe=r[0],
                max_drawdown_pct=r[1],
                win_rate=r[2],
                trade_count=r[3],
                profit_factor=r[4],
                passed=bool(r[5]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_run(self, strategy: str, run_ts: str, m: BacktestMetrics) -> None:
        """Write a single backtest run to SQLite."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO backtest_runs
                        (strategy, run_ts, sharpe, max_dd_pct, win_rate, trade_count, profit_factor, passed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (strategy, run_ts, m.sharpe, m.max_drawdown_pct, m.win_rate,
                     m.trade_count, m.profit_factor, int(m.passed)),
                )

    def _log_demotion_action(self, strategy: str, action: str, reason: str) -> None:
        """Record a demotion or promotion event."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO demotion_log (strategy, action, ts, reason) VALUES (?, ?, ?, ?)",
                    (strategy, action, ts, reason),
                )
        logger.info("ContinuousBacktester: %s %s — %s", strategy, action, reason)
