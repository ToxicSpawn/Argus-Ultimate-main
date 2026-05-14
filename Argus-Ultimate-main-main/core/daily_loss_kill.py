"""Daily Loss Kill Switch — timezone-aware session reset.

Improvements over the original:
1. Configurable session timezone — resets at midnight in YOUR timezone
   (default: Australia/Sydney / AEST), not midnight UTC.
2. Auto-reset scheduler — background thread checks once per minute and
   auto-resets when the session boundary crosses.
3. Tiered warning levels — emits WARNING at 50% and 80% of threshold.
4. Per-session history — keeps a rolling log of last 30 session P&Ls.
5. Thread-safe; all original get_stats() / assert_trading_allowed() API
   preserved for backward compatibility.

Bug fix (2026-04 Codex):
  - pct_used now uses max(0, -current_pnl) / abs(threshold) so profitable
    days never trigger false loss warnings. Previously abs(pnl/threshold)
    treated +PnL as loss consumption.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
    def _make_tz(tz_name: str):
        return ZoneInfo(tz_name)
    def _now_in_tz(tz) -> datetime:
        return datetime.now(tz)
except ImportError:
    try:
        import pytz
        def _make_tz(tz_name: str):
            return pytz.timezone(tz_name)
        def _now_in_tz(tz) -> datetime:
            return datetime.now(pytz.utc).astimezone(tz)
    except ImportError:
        _make_tz = None
        _now_in_tz = None
        logger.warning(
            "DailyLossKill: neither zoneinfo nor pytz available — "
            "falling back to UTC for session resets."
        )

DEFAULT_TZ = "Australia/Sydney"


@dataclass(frozen=True)
class DailyLossStatus:
    current_pnl:   float
    threshold:     float
    triggered:     bool
    warning_level: int    # 0=ok, 1=50% used, 2=80% used, 3=triggered
    session_date:  str
    reason:        str


class DailyLossKill:
    """
    Kill switch that halts trading when daily loss exceeds threshold.
    Resets automatically at midnight in the configured timezone.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.03,
        initial_capital:    float = 1000.0,
        session_tz:         str   = DEFAULT_TZ,
        auto_reset:         bool  = True,
    ) -> None:
        self._max_pct         = max_daily_loss_pct
        self._initial_capital = initial_capital
        self._threshold       = -(abs(max_daily_loss_pct) * initial_capital)
        self._session_tz_str  = session_tz
        self._lock            = threading.Lock()

        self._current_pnl     = 0.0
        self._triggered       = False
        self._trigger_count   = 0
        self._session_date    = self._current_session_date()
        self._session_history: Deque[Dict] = deque(maxlen=30)

        if _make_tz is not None:
            try:
                self._tz = _make_tz(session_tz)
            except Exception:
                logger.warning(
                    "DailyLossKill: unknown timezone '%s', falling back to UTC",
                    session_tz,
                )
                self._tz = None
        else:
            self._tz = None

        if auto_reset:
            self._stop_event = threading.Event()
            self._reset_thread = threading.Thread(
                target=self._auto_reset_loop,
                daemon=True,
                name="DailyLossKill-AutoReset",
            )
            self._reset_thread.start()
            logger.info(
                "DailyLossKill: auto-reset enabled, session timezone='%s'",
                session_tz,
            )
        else:
            self._stop_event = None
            self._reset_thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_pnl(self, pnl: float) -> None:
        """Update the running daily P&L value (absolute, not delta)."""
        with self._lock:
            self._current_pnl = pnl
            self._check_trigger()

    def record_trade_pnl(self, delta_pnl: float) -> None:
        """Accumulate a trade P&L delta."""
        with self._lock:
            self._current_pnl += delta_pnl
            self._check_trigger()

    def check(self) -> DailyLossStatus:
        with self._lock:
            return self._build_status()

    def assert_trading_allowed(self) -> None:
        with self._lock:
            status = self._build_status()
        if status.triggered:
            raise RuntimeError(status.reason)

    def reset(self) -> None:
        with self._lock:
            self._do_reset(reason="manual")

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "max_daily_loss_pct": self._max_pct,
                "initial_capital":    self._initial_capital,
                "threshold":          self._threshold,
                "current_pnl":        self._current_pnl,
                "triggered":          self._triggered,
                "trigger_count":      self._trigger_count,
                "session_date":       self._session_date,
                "session_tz":         self._session_tz_str,
                "session_history":    list(self._session_history),
                "warning_level":      self._warning_level(),
            }

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _loss_pct_used(current_pnl: float, threshold: float) -> float:
        """Fraction of daily loss limit consumed.

        Only counts downside exposure: a profitable day returns 0.0.
        threshold is negative (e.g. -30.0).
        """
        if threshold == 0:
            return 0.0
        downside = max(0.0, -current_pnl)   # 0 on profitable days
        return downside / abs(threshold)

    def _check_trigger(self) -> None:
        """Must be called under self._lock."""
        if not self._triggered and self._current_pnl <= self._threshold:
            self._triggered = True
            self._trigger_count += 1
            logger.critical(
                "DailyLossKill TRIGGERED [%s]: PnL %.2f <= threshold %.2f (%.1f%% of %.2f)",
                self._session_date,
                self._current_pnl,
                self._threshold,
                self._max_pct * 100,
                self._initial_capital,
            )
        # Tiered warnings — only on actual downside exposure.
        pct_used = self._loss_pct_used(self._current_pnl, self._threshold)
        if not self._triggered:
            if pct_used >= 0.80:
                logger.warning(
                    "DailyLossKill WARNING: 80%% of daily loss limit used "
                    "(PnL %.2f / threshold %.2f)",
                    self._current_pnl, self._threshold,
                )
            elif pct_used >= 0.50:
                logger.info(
                    "DailyLossKill NOTICE: 50%% of daily loss limit used "
                    "(PnL %.2f / threshold %.2f)",
                    self._current_pnl, self._threshold,
                )

    def _warning_level(self) -> int:
        """Must be called under self._lock."""
        if self._triggered:
            return 3
        pct_used = self._loss_pct_used(self._current_pnl, self._threshold)
        if pct_used >= 0.80:
            return 2
        if pct_used >= 0.50:
            return 1
        return 0

    def _build_status(self) -> DailyLossStatus:
        """Must be called under self._lock."""
        wl = self._warning_level()
        if self._triggered:
            reason = (
                f"KILL ACTIVE [{self._session_date}] — "
                f"PnL {self._current_pnl:.2f} breached threshold "
                f"{self._threshold:.2f} ({self._max_pct:.1%} of "
                f"{self._initial_capital:.2f})"
            )
        else:
            reason = (
                f"[{self._session_date}] PnL {self._current_pnl:.2f}, "
                f"threshold {self._threshold:.2f} — trading allowed"
            )
        return DailyLossStatus(
            current_pnl   = self._current_pnl,
            threshold     = self._threshold,
            triggered     = self._triggered,
            warning_level = wl,
            session_date  = self._session_date,
            reason        = reason,
        )

    def _do_reset(self, reason: str = "auto") -> None:
        """Internal reset — must be called under self._lock."""
        logger.info(
            "DailyLossKill reset (%s): session=%s final_pnl=%.2f",
            reason, self._session_date, self._current_pnl,
        )
        self._session_history.append({
            "date":      self._session_date,
            "final_pnl": self._current_pnl,
            "triggered": self._triggered,
            "reason":    reason,
        })
        self._current_pnl  = 0.0
        self._triggered    = False
        self._session_date = self._current_session_date()

    def _current_session_date(self) -> str:
        try:
            if self._tz is not None and _now_in_tz is not None:
                return _now_in_tz(self._tz).strftime("%Y-%m-%d")
        except Exception:
            pass
        return datetime.utcnow().strftime("%Y-%m-%d")

    def _auto_reset_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                new_date = self._current_session_date()
                with self._lock:
                    if new_date != self._session_date:
                        logger.info(
                            "DailyLossKill: new session detected (%s -> %s), auto-resetting",
                            self._session_date, new_date,
                        )
                        self._do_reset(reason="auto-session-rollover")
            except Exception as exc:
                logger.error("DailyLossKill auto-reset error: %s", exc)
            self._stop_event.wait(timeout=60)
