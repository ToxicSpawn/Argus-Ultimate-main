from __future__ import annotations

from argus_live.alerts.anomaly_detectors import (
    detect_drawdown_breach,
    detect_execution_alpha_drop,
    detect_reject_rate_spike,
    detect_replay_failure,
    detect_slippage_spike,
)


def test_execution_alpha_drop_detected() -> None:
    sig = detect_execution_alpha_drop(-2.5, threshold=-1.0)
    assert sig.detected is True
    assert sig.score > 0.0
    assert "alpha" in sig.reason.lower()


def test_slippage_spike_detected() -> None:
    sig = detect_slippage_spike(10.0, 3.0, multiplier=2.0)
    assert sig.detected is True
    assert sig.score > 0.0


def test_drawdown_breach_detected() -> None:
    sig = detect_drawdown_breach(12.0, 10.0)
    assert sig.detected is True
    assert "12.00%" in sig.reason


def test_reject_rate_spike_detected() -> None:
    sig = detect_reject_rate_spike(0.20, 0.05, multiplier=2.0)
    assert sig.detected is True
    assert sig.score > 0.0


def test_replay_failure_detected() -> None:
    sig = detect_replay_failure(replay_passed=False)
    assert sig.detected is True
    assert sig.score == 1.0
    assert "failed" in sig.reason.lower()
