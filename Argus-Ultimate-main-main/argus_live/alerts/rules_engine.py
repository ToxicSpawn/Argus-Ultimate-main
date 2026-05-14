from __future__ import annotations

import logging

from argus_live.alerts.anomaly_detectors import (
    detect_drawdown_breach,
    detect_execution_alpha_drop,
    detect_reject_rate_spike,
    detect_replay_failure,
    detect_slippage_spike,
)
from argus_live.alerts.models import AlertEvent
from argus_live.alerts.severity import AlertSeverity

logger = logging.getLogger(__name__)


class AlertRulesEngine:
    """Translates anomaly signals into typed AlertEvent instances."""

    # ------------------------------------------------------------------
    # Execution alpha
    # ------------------------------------------------------------------
    def evaluate_execution_alpha(
        self,
        execution_alpha_bps: float,
        threshold: float = -1.0,
    ) -> AlertEvent | None:
        sig = detect_execution_alpha_drop(execution_alpha_bps, threshold)
        if not sig.detected:
            return None
        severity = AlertSeverity.CRITICAL if sig.score > 3.0 else AlertSeverity.MAJOR
        return AlertEvent(
            alert_type="EXECUTION_ALPHA_DROP",
            severity=severity,
            title="Execution alpha drop detected",
            message=sig.reason,
            payload={"execution_alpha_bps": execution_alpha_bps, "score": sig.score},
        )

    # ------------------------------------------------------------------
    # Slippage
    # ------------------------------------------------------------------
    def evaluate_slippage(
        self,
        realized_slippage_bps: float,
        baseline_slippage_bps: float,
        multiplier: float = 2.0,
    ) -> AlertEvent | None:
        sig = detect_slippage_spike(realized_slippage_bps, baseline_slippage_bps, multiplier)
        if not sig.detected:
            return None
        severity = AlertSeverity.CRITICAL if sig.score > 4.0 else AlertSeverity.WARNING
        return AlertEvent(
            alert_type="SLIPPAGE_SPIKE",
            severity=severity,
            title="Slippage spike detected",
            message=sig.reason,
            payload={
                "realized_slippage_bps": realized_slippage_bps,
                "baseline_slippage_bps": baseline_slippage_bps,
                "score": sig.score,
            },
        )

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------
    def evaluate_drawdown(
        self,
        current_drawdown_pct: float,
        max_drawdown_pct: float,
    ) -> AlertEvent | None:
        sig = detect_drawdown_breach(current_drawdown_pct, max_drawdown_pct)
        if not sig.detected:
            return None
        severity = AlertSeverity.CRITICAL if current_drawdown_pct > max_drawdown_pct * 1.2 else AlertSeverity.MAJOR
        return AlertEvent(
            alert_type="DRAWDOWN_BREACH",
            severity=severity,
            title="Drawdown breach detected",
            message=sig.reason,
            payload={
                "current_drawdown_pct": current_drawdown_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "score": sig.score,
            },
        )

    # ------------------------------------------------------------------
    # Reject rate
    # ------------------------------------------------------------------
    def evaluate_reject_rate(
        self,
        reject_rate: float,
        baseline_reject_rate: float,
        multiplier: float = 2.0,
    ) -> AlertEvent | None:
        sig = detect_reject_rate_spike(reject_rate, baseline_reject_rate, multiplier)
        if not sig.detected:
            return None
        severity = AlertSeverity.MAJOR if sig.score > 3.0 else AlertSeverity.WARNING
        return AlertEvent(
            alert_type="REJECT_RATE_SPIKE",
            severity=severity,
            title="Order reject rate spike",
            message=sig.reason,
            payload={"reject_rate": reject_rate, "score": sig.score},
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def evaluate_replay(self, replay_passed: bool) -> AlertEvent | None:
        sig = detect_replay_failure(replay_passed)
        if not sig.detected:
            return None
        return AlertEvent(
            alert_type="REPLAY_FAILURE",
            severity=AlertSeverity.CRITICAL,
            title="Replay verification failed",
            message=sig.reason,
            payload={"replay_passed": replay_passed},
        )
