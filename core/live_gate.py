"""core/live_gate.py — Live-trading gate with hard GraduationError raise (M27).

Previously GraduationError only issued a warning; it now raises so callers
cannot silently proceed with under-qualified strategies.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class GraduationError(Exception):
    """Raised when a strategy fails the graduation requirements for live trading.

    M27: This used to be a warning only — it is now a hard raise so no caller
    can silently promote an unqualified strategy to live.
    """


@dataclass
class GraduationCriteria:
    """Minimum requirements a strategy must meet before live deployment."""

    min_paper_days: int = 30
    min_sharpe: float = 1.5
    min_win_rate: float = 0.52
    max_drawdown: float = 0.15  # 15 %
    min_trades: int = 100


@dataclass
class StrategyRecord:
    """Runtime record tracked by LiveGate."""

    name: str
    paper_days: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    promoted_at: float | None = None
    _extra: dict[str, Any] = field(default_factory=dict)


class LiveGate:
    """Controls promotion of strategies from paper → live trading.

    Raises
    ------
    GraduationError
        When ``promote()`` is called for a strategy that fails criteria.
        **This is a hard raise — no silent fallthrough (M27).**
    """

    def __init__(self, criteria: GraduationCriteria | None = None) -> None:
        self._criteria = criteria or GraduationCriteria()
        self._records: dict[str, StrategyRecord] = {}
        self._lock = threading.Lock()

    # ── Record management ────────────────────────────────────────────────────

    def register(self, name: str) -> StrategyRecord:
        """Register a strategy for tracking."""
        with self._lock:
            if name not in self._records:
                self._records[name] = StrategyRecord(name=name)
            return self._records[name]

    def update(
        self,
        name: str,
        *,
        paper_days: float | None = None,
        sharpe: float | None = None,
        win_rate: float | None = None,
        max_drawdown: float | None = None,
        trade_count: int | None = None,
    ) -> None:
        """Update performance metrics for a registered strategy."""
        with self._lock:
            rec = self._records.setdefault(name, StrategyRecord(name=name))
            if paper_days is not None:
                rec.paper_days = paper_days
            if sharpe is not None:
                rec.sharpe = sharpe
            if win_rate is not None:
                rec.win_rate = win_rate
            if max_drawdown is not None:
                rec.max_drawdown = max_drawdown
            if trade_count is not None:
                rec.trade_count = trade_count

    # ── Graduation check ─────────────────────────────────────────────────────

    def check(self, name: str) -> list[str]:
        """Return list of unmet criteria strings (empty list = passes)."""
        with self._lock:
            rec = self._records.get(name)
        if rec is None:
            return [f"strategy '{name}' not registered"]

        c = self._criteria
        failures: list[str] = []
        if rec.paper_days < c.min_paper_days:
            failures.append(
                f"paper_days={rec.paper_days:.1f} < {c.min_paper_days}"
            )
        if rec.sharpe < c.min_sharpe:
            failures.append(f"sharpe={rec.sharpe:.2f} < {c.min_sharpe}")
        if rec.win_rate < c.min_win_rate:
            failures.append(f"win_rate={rec.win_rate:.3f} < {c.min_win_rate}")
        if rec.max_drawdown > c.max_drawdown:
            failures.append(
                f"max_drawdown={rec.max_drawdown:.3f} > {c.max_drawdown}"
            )
        if rec.trade_count < c.min_trades:
            failures.append(f"trade_count={rec.trade_count} < {c.min_trades}")
        return failures

    def promote(self, name: str) -> StrategyRecord:
        """Promote *name* to live.  Raises ``GraduationError`` if any criteria fail.

        M27 fix: was previously a ``logger.warning()`` only.
        """
        failures = self.check(name)
        if failures:
            msg = (
                f"Strategy '{name}' failed graduation criteria: "
                + "; ".join(failures)
            )
            logger.error(msg)
            raise GraduationError(msg)  # ← hard raise (was warning-only before M27)

        with self._lock:
            rec = self._records[name]
            rec.promoted_at = time.time()

        logger.info("Strategy '%s' promoted to live trading ✓", name)
        return rec

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return a snapshot of all tracked strategies."""
        with self._lock:
            return {
                name: {
                    "paper_days": r.paper_days,
                    "sharpe": r.sharpe,
                    "win_rate": r.win_rate,
                    "max_drawdown": r.max_drawdown,
                    "trade_count": r.trade_count,
                    "promoted": r.promoted_at is not None,
                    "failures": self.check(name),
                }
                for name, r in self._records.items()
            }
