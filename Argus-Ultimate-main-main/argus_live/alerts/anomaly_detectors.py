from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AnomalySignal:
    """Result of an anomaly detection check."""

    detected: bool
    score: float
    reason: str


def detect_execution_alpha_drop(
    execution_alpha_bps: float,
    threshold: float = -1.0,
) -> AnomalySignal:
    """Return an anomaly when execution alpha drops below *threshold* bps."""
    detected = execution_alpha_bps < threshold
    return AnomalySignal(
        detected=detected,
        score=abs(execution_alpha_bps - threshold) if detected else 0.0,
        reason=f"Execution alpha {execution_alpha_bps:.2f} bps < {threshold:.2f} bps"
        if detected
        else "",
    )


def detect_slippage_spike(
    realized_slippage_bps: float,
    baseline_slippage_bps: float,
    multiplier: float = 2.0,
) -> AnomalySignal:
    """Return an anomaly when realised slippage exceeds *multiplier* x baseline."""
    threshold = baseline_slippage_bps * multiplier
    detected = realized_slippage_bps > threshold
    return AnomalySignal(
        detected=detected,
        score=realized_slippage_bps / max(baseline_slippage_bps, 1e-9) if detected else 0.0,
        reason=f"Slippage {realized_slippage_bps:.2f} bps > {multiplier}x baseline ({threshold:.2f} bps)"
        if detected
        else "",
    )


def detect_drawdown_breach(
    current_drawdown_pct: float,
    max_drawdown_pct: float,
) -> AnomalySignal:
    """Return an anomaly when current drawdown exceeds the allowed maximum."""
    detected = current_drawdown_pct > max_drawdown_pct
    return AnomalySignal(
        detected=detected,
        score=current_drawdown_pct / max(max_drawdown_pct, 1e-9) if detected else 0.0,
        reason=f"Drawdown {current_drawdown_pct:.2f}% > max {max_drawdown_pct:.2f}%"
        if detected
        else "",
    )


def detect_reject_rate_spike(
    reject_rate: float,
    baseline_reject_rate: float,
    multiplier: float = 2.0,
) -> AnomalySignal:
    """Return an anomaly when the order reject rate spikes above *multiplier* x baseline."""
    threshold = baseline_reject_rate * multiplier
    detected = reject_rate > threshold
    return AnomalySignal(
        detected=detected,
        score=reject_rate / max(baseline_reject_rate, 1e-9) if detected else 0.0,
        reason=f"Reject rate {reject_rate:.4f} > {multiplier}x baseline ({threshold:.4f})"
        if detected
        else "",
    )


def detect_replay_failure(replay_passed: bool) -> AnomalySignal:
    """Return an anomaly when the replay check did not pass."""
    detected = not replay_passed
    return AnomalySignal(
        detected=detected,
        score=1.0 if detected else 0.0,
        reason="Replay verification failed" if detected else "",
    )
