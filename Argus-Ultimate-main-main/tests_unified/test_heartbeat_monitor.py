"""
tests_unified/test_heartbeat_monitor.py
========================================
Unit tests for infra/heartbeat_monitor.py

Run with:
    pytest tests_unified/test_heartbeat_monitor.py -v
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
import sys
import os

# Make sure the repo root is on sys.path so we can import infra.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from infra.heartbeat_monitor import (
    HeartbeatConfig,
    HeartbeatMonitor,
    UptimeTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(**kwargs) -> HeartbeatMonitor:
    """Return a HeartbeatMonitor with a quick interval and no real HTTP server."""
    defaults = dict(
        heartbeat_interval_s=1.0,
        alert_after_missed=3,
        alert_channels=["log", "file"],
        metrics_port=0,          # port 0 = let OS choose (avoids conflicts)
        check_strategies=["micro_mm", "funding_arb"],
    )
    defaults.update(kwargs)
    cfg = HeartbeatConfig(**defaults)
    mon = HeartbeatMonitor(cfg)
    # Prevent the metrics HTTP server from binding a real socket in unit tests
    mon._start_metrics_server = lambda: None  # type: ignore[method-assign]
    return mon


# ---------------------------------------------------------------------------
# test_heartbeat_record
# ---------------------------------------------------------------------------

def test_heartbeat_record():
    """record_heartbeat() must update _last_seen for the given strategy."""
    mon = _make_monitor()
    before = time.monotonic()
    mon.record_heartbeat("micro_mm")
    after = time.monotonic()

    assert "micro_mm" in mon._last_seen
    assert before <= mon._last_seen["micro_mm"] <= after


# ---------------------------------------------------------------------------
# test_heartbeat_check_alive
# ---------------------------------------------------------------------------

def test_heartbeat_check_alive():
    """A heartbeat recorded within 2× interval should report alive=True."""
    mon = _make_monitor()
    mon.record_heartbeat("micro_mm")
    result = mon.check_heartbeats()
    assert result["micro_mm"] is True


# ---------------------------------------------------------------------------
# test_heartbeat_check_dead
# ---------------------------------------------------------------------------

def test_heartbeat_check_dead():
    """
    A strategy that has NOT sent a heartbeat within 2× interval should be False.
    We simulate staleness by manually backdating _last_seen.
    """
    mon = _make_monitor()
    # Inject a last_seen timestamp 10× the interval in the past
    stale_time = time.monotonic() - (mon.config.heartbeat_interval_s * 10)
    mon._last_seen["micro_mm"] = stale_time
    result = mon.check_heartbeats()
    assert result["micro_mm"] is False


# ---------------------------------------------------------------------------
# test_uptime_tracker_100pct
# ---------------------------------------------------------------------------

def test_uptime_tracker_100pct():
    """All record_alive events → uptime = 1.0."""
    tracker = UptimeTracker()
    now_ns = time.time_ns()
    for i in range(100):
        tracker.record_alive("micro_mm", timestamp_ns=now_ns - i * 1_000_000_000)
    pct = tracker.get_uptime_pct("micro_mm")
    assert pct == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# test_uptime_tracker_partial
# ---------------------------------------------------------------------------

def test_uptime_tracker_partial():
    """
    50 alive + 50 dead events within the window → uptime ≈ 0.5.
    We allow ±0.05 tolerance to account for any pruning edge effects.
    """
    tracker = UptimeTracker(window_s=86400.0)
    now_ns = time.time_ns()
    for i in range(50):
        # alive events: spaced 1 second apart starting from now-100s
        tracker.record_alive("funding_arb", timestamp_ns=now_ns - (100 + i * 2) * 1_000_000_000)
    for i in range(50):
        # dead events: interleaved
        tracker.record_dead("funding_arb", timestamp_ns=now_ns - (99 + i * 2) * 1_000_000_000)

    pct = tracker.get_uptime_pct("funding_arb")
    assert 0.45 <= pct <= 0.55, f"Expected ~0.5, got {pct}"


# ---------------------------------------------------------------------------
# test_alert_handler_registered
# ---------------------------------------------------------------------------

def test_alert_handler_registered():
    """
    A custom alert handler must be called when _alert() is invoked.
    """
    mon = _make_monitor(alert_channels=["custom"])
    handler_mock = MagicMock()
    mon.register_alert_handler("custom", handler_mock)

    # Fire the alert synchronously via asyncio.run
    asyncio.run(mon._alert("micro_mm", "Test alert message"))

    handler_mock.assert_called_once_with("micro_mm", "Test alert message")


# ---------------------------------------------------------------------------
# test_get_uptime_stats_keys
# ---------------------------------------------------------------------------

def test_get_uptime_stats_keys():
    """
    get_uptime_stats() must return all expected top-level and per-strategy keys.
    """
    mon = _make_monitor()
    # Seed _start_time so system_uptime_s is non-zero
    mon._start_time = time.monotonic() - 10.0

    # Record a heartbeat so per-strategy fields are populated
    mon.record_heartbeat("micro_mm")
    mon.record_heartbeat("funding_arb")

    stats = mon.get_uptime_stats()

    # Top-level keys
    assert "strategies" in stats
    assert "total_alerts" in stats
    assert "last_alert_time" in stats
    assert "system_uptime_s" in stats

    assert stats["system_uptime_s"] >= 10.0

    # Per-strategy keys
    for strategy in mon.config.check_strategies:
        assert strategy in stats["strategies"], f"Missing strategy '{strategy}' in stats"
        s = stats["strategies"][strategy]
        assert "uptime_pct" in s
        assert "consecutive_misses" in s
        assert "last_seen_s" in s
        assert "alive" in s
        # uptime_pct should be in valid range
        assert 0.0 <= s["uptime_pct"] <= 1.0
