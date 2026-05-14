"""
ARGUS Capital Migration Monitor
================================
Monitors capital migration readiness and sends alerts when the system is ready
to advance to the next deployment stage (PAPER → MICRO → SEED → LIVE).

Usage::

    from ops.capital_migration import CapitalMigration
    from monitoring.capital_migration_monitor import CapitalMigrationMonitor

    migration = CapitalMigration()
    monitor = CapitalMigrationMonitor(
        migration=migration,
        discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL", ""),
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
    )

    # Call periodically (e.g., every 4 hours from component_registry heartbeat):
    stats = get_portfolio_stats_from_system(trading_system)
    result = monitor.check_and_alert(stats)
    if result["can_advance"]:
        logger.info(f"Ready to advance to {result['next_stage']}")
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Alert rate-limit: one alert per stage per 4 hours
_ALERT_COOLDOWN_SECONDS = 4 * 3600


# ---------------------------------------------------------------------------
# Helper: build a PerformanceSnapshot from portfolio_stats dict
# ---------------------------------------------------------------------------

def _build_perf_snapshot(portfolio_stats: Dict[str, Any]) -> Any:
    """
    Convert a portfolio_stats dict → PerformanceSnapshot.
    Returns None if the import fails.
    """
    try:
        from ops.capital_migration import PerformanceSnapshot, Stage  # type: ignore

        stage_raw = str(portfolio_stats.get("current_stage") or "paper").lower()
        try:
            current_stage = Stage(stage_raw)
        except ValueError:
            current_stage = Stage.PAPER

        return PerformanceSnapshot(
            days_at_stage=int(portfolio_stats.get("days_at_stage") or 0),
            sharpe_annualised=float(portfolio_stats.get("sharpe_ratio") or 0.0),
            max_drawdown_pct=float(portfolio_stats.get("max_drawdown_pct") or 0.0),
            circuit_breaks_7d=int(portfolio_stats.get("circuit_breaker_count") or 0),
            total_trades=int(portfolio_stats.get("total_trades") or 0),
            current_stage=current_stage,
        )
    except Exception as exc:
        logger.debug("CapitalMigrationMonitor: failed to build PerformanceSnapshot: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CapitalMigrationMonitor:
    """
    Wraps CapitalMigration.assess() with alerting and rate-limiting.

    Parameters
    ----------
    migration:
        An initialised CapitalMigration instance.
    discord_webhook_url:
        Discord webhook URL for advancement alerts. Falls back to
        DISCORD_WEBHOOK_URL env var when empty.
    telegram_token:
        Telegram Bot token. Falls back to TELEGRAM_BOT_TOKEN env var.
    telegram_chat_id:
        Telegram chat ID. Falls back to TELEGRAM_CHAT_ID env var.
    """

    def __init__(
        self,
        migration: Any,
        discord_webhook_url: str = "",
        telegram_token: str = "",
        telegram_chat_id: str = "",
    ) -> None:
        self.migration = migration
        self.discord_webhook_url = (
            discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        )
        self.telegram_token = telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

        # Rate-limit tracking: stage_value → last alert timestamp
        self._last_alert_time: Dict[str, float] = {}
        self._alert_count = 0
        self._check_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_alert(self, portfolio_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate migration readiness and alert if ready to advance.

        Parameters
        ----------
        portfolio_stats:
            Dict with keys: sharpe_ratio, max_drawdown_pct, total_trades,
            days_at_stage, circuit_breaker_count, daily_pnl_aud,
            current_stage (optional, defaults to migration.current_stage).

        Returns
        -------
        dict with keys:
            current_stage (str), can_advance (bool), next_stage (str|None),
            missing_requirements (list[str]), recommendation (str)
        """
        self._check_count += 1

        # Fill in current_stage if not provided
        if "current_stage" not in portfolio_stats:
            try:
                portfolio_stats = dict(portfolio_stats)
                portfolio_stats["current_stage"] = self.migration.current_stage.value
            except Exception as _e:
                logger.debug("capital_migration_monitor error: %s", _e)

        # Build snapshot and call assess()
        assessment = None
        try:
            perf = _build_perf_snapshot(portfolio_stats)
            if perf is not None:
                assessment = self.migration.assess(perf)
        except Exception as exc:
            logger.debug("CapitalMigrationMonitor: assess() failed: %s", exc)

        if assessment is None:
            return {
                "current_stage": str(portfolio_stats.get("current_stage") or "unknown"),
                "can_advance": False,
                "next_stage": None,
                "missing_requirements": ["assessment failed — check logs"],
                "recommendation": "Could not evaluate migration readiness.",
            }

        # Build the missing-requirements list
        missing: List[str] = [
            f"{c.requirement}: need {c.required}, got {c.actual}"
            for c in (assessment.checks or [])
            if not c.passed
        ]

        result: Dict[str, Any] = {
            "current_stage": assessment.current_stage.value,
            "can_advance": assessment.can_advance,
            "next_stage": assessment.next_stage.value if assessment.next_stage else None,
            "missing_requirements": missing,
            "recommendation": assessment.recommendation,
        }

        # Alert if can_advance and not rate-limited
        if assessment.can_advance:
            stage_key = assessment.current_stage.value
            last_alert = self._last_alert_time.get(stage_key, 0.0)
            if time.time() - last_alert >= _ALERT_COOLDOWN_SECONDS:
                self._last_alert_time[stage_key] = time.time()
                self._alert_count += 1
                msg = self.format_alert_message(result)
                self._send_discord(msg, next_stage=result["next_stage"])
                self._send_telegram(msg)
            else:
                age_h = (time.time() - last_alert) / 3600
                logger.debug(
                    "CapitalMigrationMonitor: rate-limited alert for stage %s (last %.1fh ago)",
                    stage_key, age_h,
                )

        return result

    def format_alert_message(self, result: Dict[str, Any]) -> str:
        """Format a human-readable alert message for the given check result."""
        current = str(result.get("current_stage") or "").upper()
        nxt = str(result.get("next_stage") or "").upper()
        rec = str(result.get("recommendation") or "")
        lines = [
            f"ARGUS Capital Migration Alert",
            f"Stage: {current} -> {nxt}",
            f"Status: Ready to advance!",
            f"Recommendation: {rec}",
        ]
        missing = result.get("missing_requirements") or []
        if missing:
            lines.append("Missing requirements:")
            for m in missing:
                lines.append(f"  - {m}")
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, Any]:
        """Return monitor state and statistics."""
        try:
            current_stage = self.migration.current_stage.value
        except Exception:
            current_stage = "unknown"

        return {
            "current_stage": current_stage,
            "check_count": self._check_count,
            "alert_count": self._alert_count,
            "last_alert_times": dict(self._last_alert_time),
            "discord_configured": bool(self.discord_webhook_url),
            "telegram_configured": bool(self.telegram_token and self.telegram_chat_id),
        }

    # ------------------------------------------------------------------
    # Alert senders (best-effort, synchronous, never raise)
    # ------------------------------------------------------------------

    def _send_discord(self, message: str, next_stage: Optional[str] = None) -> None:
        """POST a Discord embed. Best-effort."""
        if not self.discord_webhook_url:
            return
        try:
            nxt = (next_stage or "NEXT STAGE").upper()
            embed = {
                "title": f"ARGUS Capital Migration: Ready to advance to {nxt}!",
                "description": message,
                "color": 0x2ECC71,  # green
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "footer": {"text": "Check ops/capital_migration.py to advance"},
            }
            payload = json.dumps({"username": "ARGUS Bot", "embeds": [embed]}).encode("utf-8")
            req = urllib.request.Request(
                self.discord_webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status not in (200, 204):
                    logger.debug("CapitalMigrationMonitor: Discord HTTP %d", resp.status)
                else:
                    logger.info("CapitalMigrationMonitor: Discord alert sent")
        except urllib.error.HTTPError as exc:
            logger.debug("CapitalMigrationMonitor: Discord HTTP error %d: %s", exc.code, exc.reason)
        except Exception as exc:
            logger.debug("CapitalMigrationMonitor: Discord send failed: %s", exc)

    def _send_telegram(self, message: str) -> None:
        """POST a Telegram message. Best-effort."""
        if not (self.telegram_token and self.telegram_chat_id):
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            # Escape special markdown characters for Telegram MarkdownV2
            safe_msg = message.replace("_", "\\_").replace("*", "\\*").replace("[", "\\[")
            payload = json.dumps({
                "chat_id": self.telegram_chat_id,
                "text": safe_msg,
                "parse_mode": "MarkdownV2",
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    logger.info("CapitalMigrationMonitor: Telegram alert sent")
                else:
                    logger.debug("CapitalMigrationMonitor: Telegram HTTP %d", resp.status)
        except urllib.error.HTTPError as exc:
            logger.debug(
                "CapitalMigrationMonitor: Telegram HTTP error %d: %s", exc.code, exc.reason
            )
        except Exception as exc:
            logger.debug("CapitalMigrationMonitor: Telegram send failed: %s", exc)


# ---------------------------------------------------------------------------
# Standalone helper: extract portfolio stats from a UnifiedTradingSystem
# ---------------------------------------------------------------------------

def get_portfolio_stats_from_system(trading_system: Any) -> Dict[str, Any]:
    """
    Extract the dict of portfolio stats needed by CapitalMigrationMonitor
    from a UnifiedTradingSystem instance.

    Missing attributes are handled gracefully (defaulting to zero/None).

    Returns
    -------
    dict with keys matching check_and_alert() expectations:
        sharpe_ratio, max_drawdown_pct, total_trades, days_at_stage,
        circuit_breaker_count, daily_pnl_aud, current_stage
    """
    def _getf(obj: Any, *attrs: str, default: float = 0.0) -> float:
        for attr in attrs:
            try:
                v = getattr(obj, attr, None)
                if v is None:
                    v = (obj.get(attr) if isinstance(obj, dict) else None)
                if v is not None:
                    return float(v)
            except Exception as _e:
                logger.debug("capital_migration_monitor error: %s", _e)
        return default

    def _geti(obj: Any, *attrs: str, default: int = 0) -> int:
        return int(_getf(obj, *attrs, default=float(default)))

    stats: Dict[str, Any] = {}

    # Realized P&L and trade count
    stats["daily_pnl_aud"] = _getf(trading_system, "daily_pnl", "realized_pnl", "total_pnl")
    stats["total_trades"] = _geti(trading_system, "total_trades", "trade_count", "fills_count")

    # Drawdown
    stats["max_drawdown_pct"] = _getf(
        trading_system, "max_drawdown_pct", "current_drawdown_pct", "drawdown_pct"
    )

    # Sharpe (may live on risk manager or performance tracker)
    sharpe = _getf(trading_system, "sharpe_ratio", "sharpe", default=-999.0)
    if sharpe == -999.0:
        # Try risk manager
        try:
            rm = getattr(trading_system, "risk_manager", None)
            if rm is not None:
                sharpe = _getf(rm, "sharpe_ratio", "sharpe", default=0.0)
            else:
                sharpe = 0.0
        except Exception:
            sharpe = 0.0
    stats["sharpe_ratio"] = sharpe

    # Circuit breaker count
    stats["circuit_breaker_count"] = _geti(
        trading_system, "circuit_break_count", "circuit_breaker_count", "circuit_breaks"
    )

    # Days running (use start_time if available)
    days = 0
    try:
        start_ts = getattr(trading_system, "start_time", None)
        if start_ts is not None:
            if isinstance(start_ts, datetime):
                elapsed = (datetime.now(tz=timezone.utc) - start_ts).total_seconds()
            else:
                elapsed = time.time() - float(start_ts)
            days = int(elapsed / 86400)
    except Exception as _e:
        logger.debug("capital_migration_monitor error: %s", _e)
    stats["days_at_stage"] = days

    # Current capital migration stage
    try:
        cr = getattr(trading_system, "component_registry", None)
        if cr is not None:
            cm = getattr(cr, "capital_migration", None)
            if cm is not None:
                stats["current_stage"] = cm.current_stage.value
    except Exception as _e:
        logger.debug("capital_migration_monitor error: %s", _e)

    return stats
