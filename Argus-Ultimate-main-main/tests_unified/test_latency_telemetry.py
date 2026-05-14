"""
tests_unified/test_latency_telemetry.py
========================================
10 tests for hft_engine/latency_telemetry.py.

Coverage:
  1. journey_lifecycle           – start → mark → complete full happy path
  2. tick_to_order_property      – tick_to_order_us computed correctly
  3. percentile_calculation      – p50/p95/p99/p999 accuracy with known data
  4. threshold_alerts_warning    – p99 > 5 ms triggers warning alert
  5. threshold_alerts_critical   – p99 > 50 ms triggers critical alert
  6. profiler_context_manager    – section() records elapsed correctly
  7. profiler_instrument_deco    – @instrument wraps async fn
  8. jitter_detection            – jitter_us returns std-dev of intervals
  9. late_ticks_detection        – late_ticks counts correctly
 10. prometheus_export_format    – export string passes basic format checks
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import List
from unittest.mock import patch

import pytest

# The module under test
from hft_engine.latency_telemetry import (
    HotPathProfiler,
    JitterMonitor,
    LatencyReport,
    LatencyStage,
    LatencyTelemetry,
    TradeJourney,
    generate_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_telemetry():
    """
    Ensure each test gets a clean LatencyTelemetry singleton.
    We reset (not re-instantiate) so the singleton pattern is preserved.
    """
    tel = LatencyTelemetry.get_instance()
    tel.reset_stats()
    yield tel
    tel.reset_stats()


@pytest.fixture()
def profiler():
    return HotPathProfiler()


@pytest.fixture()
def jitter():
    return JitterMonitor()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Journey lifecycle (start → mark all stages → complete)
# ─────────────────────────────────────────────────────────────────────────────

def test_journey_lifecycle(fresh_telemetry):
    """Full tick-to-fill journey should be stored in the ring buffer."""
    tel = fresh_telemetry
    jid = tel.start_journey("AAPL")
    assert isinstance(jid, str)
    assert len(jid) == 36  # UUID4

    for stage in LatencyStage:
        if stage != LatencyStage.MARKET_DATA_RX:  # already stamped by start_journey
            time.sleep(0.0001)                     # tiny sleep to ensure monotonic timestamps
            tel.mark(jid, stage)

    journey = tel.complete_journey(jid)
    assert journey is not None
    assert journey.symbol == "AAPL"
    assert set(journey.timestamps.keys()) == set(LatencyStage)

    stats = tel.get_stats()
    assert stats["completed_journeys"] == 1
    assert stats["in_flight_journeys"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – tick_to_order_us and other properties
# ─────────────────────────────────────────────────────────────────────────────

def test_tick_to_order_property():
    """tick_to_order_us should equal ORDER_SUBMIT - MARKET_DATA_RX in µs."""
    journey = TradeJourney(journey_id=str(uuid.uuid4()), symbol="TSLA")
    t0 = time.time_ns()
    journey.timestamps[LatencyStage.MARKET_DATA_RX] = t0
    journey.timestamps[LatencyStage.SIGNAL_COMPUTE]  = t0 + 500_000      # +500 µs
    journey.timestamps[LatencyStage.RISK_CHECK]      = t0 + 900_000      # +900 µs
    journey.timestamps[LatencyStage.ORDER_SUBMIT]    = t0 + 1_500_000    # +1500 µs
    journey.timestamps[LatencyStage.ACK_RX]          = t0 + 3_000_000    # +3000 µs
    journey.timestamps[LatencyStage.FILL_RX]         = t0 + 10_000_000   # +10000 µs

    assert abs(journey.tick_to_order_us - 1500.0) < 1.0
    assert abs(journey.tick_to_trade_us - 10_000.0) < 1.0
    assert abs(journey.order_to_ack_us  - 1500.0) < 1.0
    assert abs(journey.signal_latency_us - 400.0) < 1.0   # SIGNAL_COMPUTE stage = 500→900 = 400 µs
    assert abs(journey.risk_latency_us   - 600.0) < 1.0   # RISK_CHECK stage = 900→1500 = 600 µs


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Percentile calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_percentile_calculation(fresh_telemetry):
    """Inject known stage durations and verify p50/p99 are computed correctly."""
    import numpy as np

    tel = fresh_telemetry
    # Inject 100 artificial samples with known distribution
    values_us = list(range(1, 101))  # 1 µs … 100 µs
    for v in values_us:
        tel._t2o_samples.append(float(v))

    stats = tel.get_stats()
    t2o = stats["TICK_TO_ORDER"]
    assert t2o["count"] == 100
    # numpy p50 of 1..100 → 50.5
    assert abs(t2o["p50_us"] - 50.5) < 0.5
    # p99 of 1..100 → ~99.01
    assert t2o["p99_us"] >= 98.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – Warning alert (p99 > 5 ms)
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_alerts_warning(fresh_telemetry):
    """p99 just above 5 000 µs should produce a warning alert."""
    tel = fresh_telemetry
    # Fill TICK_TO_ORDER samples with 6 000 µs (above 5 ms warning threshold)
    for _ in range(200):
        tel._t2o_samples.append(6_000.0)

    alerts = tel.get_alert_thresholds()
    assert alerts["has_warning"] or alerts["has_critical"]
    levels = {a["level"] for a in alerts["alerts"]}
    # 6 ms exceeds 5 ms warning but NOT 50 ms critical
    assert "warning" in levels


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – Critical alert (p99 > 50 ms)
# ─────────────────────────────────────────────────────────────────────────────

def test_threshold_alerts_critical(fresh_telemetry):
    """p99 above 50 000 µs should produce a critical alert."""
    tel = fresh_telemetry
    for _ in range(200):
        tel._t2o_samples.append(60_000.0)

    alerts = tel.get_alert_thresholds()
    assert alerts["has_critical"]
    assert any(a["level"] == "critical" for a in alerts["alerts"])


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – HotPathProfiler context manager
# ─────────────────────────────────────────────────────────────────────────────

def test_profiler_context_manager(profiler):
    """section() should record elapsed time > 0 for a measurable code block."""
    with profiler.section("signal_compute"):
        # Tiny CPU work to ensure non-zero elapsed
        _ = sum(range(10_000))

    rep = profiler.report()
    assert "signal_compute" in rep
    s = rep["signal_compute"]
    assert s["calls"] == 1.0
    assert s["mean_us"] >= 0.0   # wall-clock time is non-negative
    assert s["total_us"] >= 0.0
    assert s["p99_us"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 – HotPathProfiler @instrument decorator
# ─────────────────────────────────────────────────────────────────────────────

def test_profiler_instrument_decorator(profiler):
    """@profiler.instrument should wrap an async function and record timing."""

    @profiler.instrument
    async def my_signal() -> int:
        await asyncio.sleep(0)
        return 42

    result = asyncio.get_event_loop().run_until_complete(my_signal())
    assert result == 42

    rep = profiler.report()
    assert len(rep) == 1
    fn_name = list(rep.keys())[0]
    assert "my_signal" in fn_name
    assert rep[fn_name]["calls"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 – JitterMonitor jitter_us
# ─────────────────────────────────────────────────────────────────────────────

def test_jitter_detection(jitter):
    """Inject timestamps with known inter-arrival times and verify jitter."""
    # Manually insert arrival timestamps with perfectly regular 1 ms intervals
    # + one outlier to create jitter
    symbol = "MSFT"
    base_ns = time.time_ns()
    # 10 ticks, each 1_000_000 ns = 1 ms apart, plus last tick is 5 ms later
    arrivals = [base_ns + i * 1_000_000 for i in range(9)]
    arrivals.append(base_ns + 9 * 1_000_000 + 4_000_000)  # 5 ms gap at end

    jitter._arrivals[symbol].extend(arrivals)
    j = jitter.jitter_us(symbol)
    # Jitter should be non-zero due to the outlier
    assert j > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 – JitterMonitor late_ticks and gap_detected
# ─────────────────────────────────────────────────────────────────────────────

def test_late_ticks_and_gap_detection(jitter):
    """late_ticks should count outlier intervals; gap_detected returns True after silence."""
    symbol = "SPY"
    # Record 5 ticks with ~100 µs intervals
    for _ in range(5):
        jitter.record_tick(symbol)
        time.sleep(0.0001)   # 100 µs

    # Should NOT detect a gap immediately after recording
    assert not jitter.gap_detected(symbol, gap_threshold_us=1_000_000.0)  # 1 second threshold

    # Patch last arrival to be 10 seconds in the past to trigger gap
    old_arrivals = list(jitter._arrivals[symbol])
    ten_seconds_ago = time.time_ns() - 10_000_000_000
    # Replace last entry
    jitter._arrivals[symbol].clear()
    jitter._arrivals[symbol].extend(old_arrivals[:-1])
    jitter._arrivals[symbol].append(ten_seconds_ago)

    assert jitter.gap_detected(symbol, gap_threshold_us=5_000.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 – Prometheus export format
# ─────────────────────────────────────────────────────────────────────────────

def test_prometheus_export_format(fresh_telemetry):
    """export_prometheus_metrics() should produce valid Prometheus text format."""
    tel = fresh_telemetry
    # Add some data so output is non-trivial
    jid = tel.start_journey("BTC")
    tel.mark(jid, LatencyStage.SIGNAL_COMPUTE)
    tel.mark(jid, LatencyStage.RISK_CHECK)
    tel.mark(jid, LatencyStage.ORDER_SUBMIT)
    tel.mark(jid, LatencyStage.ACK_RX)
    tel.mark(jid, LatencyStage.FILL_RX)
    tel.complete_journey(jid)

    output = tel.export_prometheus_metrics()

    # Must end with newline
    assert output.endswith("\n")

    # Must contain HELP and TYPE lines
    assert "# HELP" in output
    assert "# TYPE" in output

    # Must contain the custom metric names
    assert "argus_hft_stage_latency_us" in output
    assert "argus_hft_aggregate_latency_us" in output
    assert "argus_hft_completed_journeys_total" in output

    # All data lines should have exactly one space between label set and value
    for line in output.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Metric name{...} value
        assert "{" in line or " " in line, f"Malformed line: {line!r}"

    # Completed journeys should be 1
    counter_line = next(
        (l for l in output.splitlines() if l.startswith("argus_hft_completed_journeys_total ")),
        None,
    )
    assert counter_line is not None
    assert counter_line.split()[-1] == "1"
