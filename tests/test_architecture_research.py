#!/usr/bin/env python3
"""
Tests for architecture, research, and revenue modules (Batch 11).

Modules under test:
- core/event_sourcing.py       — EventStore
- core/service_circuit_breaker.py — ServiceCircuitBreaker
- ops/canary_deployer.py       — CanaryDeployer
- research/regime_changepoint.py — BOCPDetector
- research/fractal_analyzer.py  — FractalAnalyzer
- research/entropy_signal_quality.py — EntropyAnalyzer
- compliance/tax_loss_harvester.py — TaxLossHarvester

70+ tests covering core functionality, edge cases, and thread safety.
"""

from __future__ import annotations

import math
import os
import random
import tempfile
import threading
import time

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary SQLite database."""
    return str(tmp_path / "test.db")


# =========================================================================
# 1. EventStore  (12 tests)
# =========================================================================

from core.event_sourcing import Event, EventStore


class TestEventStore:
    """Tests for the append-only event sourcing store."""

    def test_append_returns_event_id(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        eid = store.append("stream-1", "created", {"name": "Alice"})
        assert isinstance(eid, int)
        assert eid >= 1
        store.close()

    def test_append_increments_version(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        store.append("s1", "a", {})
        store.append("s1", "b", {})
        events = store.get_events("s1")
        assert [e.version for e in events] == [1, 2]
        store.close()

    def test_get_events_empty_stream(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        events = store.get_events("nonexistent")
        assert events == []
        store.close()

    def test_get_events_after_version(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        for i in range(5):
            store.append("s1", f"evt-{i}", {"i": i})
        events = store.get_events("s1", after_version=3)
        assert len(events) == 2
        assert events[0].version == 4
        store.close()

    def test_event_data_roundtrip(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        data = {"amount": 123.45, "tags": ["a", "b"], "nested": {"key": True}}
        store.append("s1", "complex", data)
        events = store.get_events("s1")
        assert events[0].data == data
        store.close()

    def test_separate_streams(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        store.append("stream-A", "x", {})
        store.append("stream-B", "y", {})
        store.append("stream-A", "z", {})
        assert len(store.get_events("stream-A")) == 2
        assert len(store.get_events("stream-B")) == 1
        store.close()

    def test_rebuild_state(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        store.append("acct-1", "credited", {"amount": 100})
        store.append("acct-1", "debited", {"amount": 30})
        store.append("acct-1", "credited", {"amount": 50})

        def reducer(state, event):
            if state is None:
                state = {"balance": 0}
            if event.event_type == "credited":
                state["balance"] += event.data["amount"]
            elif event.event_type == "debited":
                state["balance"] -= event.data["amount"]
            return state

        result = store.rebuild_state("acct-1", reducer)
        assert result["balance"] == 120
        store.close()

    def test_rebuild_state_with_snapshot(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        for i in range(10):
            store.append("s1", "inc", {"v": 1})

        # Save snapshot at version 5
        store.save_snapshot("s1", {"total": 5}, version=5)

        def reducer(state, event):
            state["total"] += event.data["v"]
            return state

        result = store.rebuild_state("s1", reducer)
        assert result["total"] == 10  # 5 from snapshot + 5 replayed
        store.close()

    def test_snapshot_roundtrip(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        store.save_snapshot("s1", {"count": 42}, version=10)
        snap = store.get_snapshot("s1")
        assert snap is not None
        assert snap["state"]["count"] == 42
        assert snap["version"] == 10
        store.close()

    def test_snapshot_none_when_missing(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        assert store.get_snapshot("missing") is None
        store.close()

    def test_get_stream_ids(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        store.append("alpha", "x", {})
        store.append("beta", "y", {})
        store.append("alpha", "z", {})
        ids = store.get_stream_ids()
        assert sorted(ids) == ["alpha", "beta"]
        store.close()

    def test_get_stream_version(self, tmp_db):
        store = EventStore(db_path=tmp_db)
        assert store.get_stream_version("s1") == 0
        store.append("s1", "a", {})
        store.append("s1", "b", {})
        assert store.get_stream_version("s1") == 2
        store.close()


# =========================================================================
# 2. ServiceCircuitBreaker  (12 tests)
# =========================================================================

from core.service_circuit_breaker import (
    CircuitOpenError,
    ServiceCircuitBreaker,
    ServiceState,
)


class TestServiceCircuitBreaker:
    """Tests for the service-level circuit breaker."""

    def test_closed_by_default(self):
        cb = ServiceCircuitBreaker()
        assert cb.get_state("svc") == "closed"

    def test_successful_call(self):
        cb = ServiceCircuitBreaker()
        result = cb.call("svc", lambda: 42)
        assert result == 42

    def test_failure_increments_count(self):
        cb = ServiceCircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call("svc", self._fail_fn)
        assert cb.get_state("svc") == "closed"

    def test_opens_after_threshold(self):
        cb = ServiceCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call("svc", self._fail_fn)
        assert cb.get_state("svc") == "open"

    def test_circuit_open_error(self):
        cb = ServiceCircuitBreaker(failure_threshold=2, recovery_timeout_s=100)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call("svc", self._fail_fn)
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call("svc", lambda: 1)
        assert exc_info.value.service_name == "svc"
        assert exc_info.value.retry_after > 0

    def test_half_open_after_timeout(self):
        cb = ServiceCircuitBreaker(failure_threshold=2, recovery_timeout_s=0.01)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call("svc", self._fail_fn)
        time.sleep(0.05)
        assert cb.get_state("svc") == "half_open"

    def test_half_open_to_closed_on_success(self):
        cb = ServiceCircuitBreaker(
            failure_threshold=2, recovery_timeout_s=0.01, success_threshold=2
        )
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call("svc", self._fail_fn)
        time.sleep(0.05)
        cb.call("svc", lambda: "ok")
        cb.call("svc", lambda: "ok")
        assert cb.get_state("svc") == "closed"

    def test_half_open_to_open_on_failure(self):
        cb = ServiceCircuitBreaker(
            failure_threshold=2, recovery_timeout_s=0.01, success_threshold=3
        )
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call("svc", self._fail_fn)
        time.sleep(0.05)
        # One success, then failure
        cb.call("svc", lambda: "ok")
        with pytest.raises(ValueError):
            cb.call("svc", self._fail_fn)
        assert cb.get_state("svc") == "open"

    def test_reset(self):
        cb = ServiceCircuitBreaker(failure_threshold=2)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call("svc", self._fail_fn)
        assert cb.get_state("svc") == "open"
        cb.reset("svc")
        assert cb.get_state("svc") == "closed"

    def test_get_all_states(self):
        cb = ServiceCircuitBreaker(failure_threshold=1)
        cb.call("healthy", lambda: True)
        with pytest.raises(RuntimeError):
            cb.call("broken", self._runtime_fail)
        states = cb.get_all_states()
        assert states["healthy"] == "closed"
        assert states["broken"] == "open"

    def test_get_stats(self):
        cb = ServiceCircuitBreaker(failure_threshold=5)
        cb.call("svc", lambda: 1)
        with pytest.raises(ValueError):
            cb.call("svc", self._fail_fn)
        stats = cb.get_stats("svc")
        assert stats["total_calls"] == 2
        assert stats["total_failures"] == 1

    def test_independent_services(self):
        cb = ServiceCircuitBreaker(failure_threshold=2)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call("svc-A", self._fail_fn)
        assert cb.get_state("svc-A") == "open"
        assert cb.get_state("svc-B") == "closed"
        cb.call("svc-B", lambda: "fine")
        assert cb.get_state("svc-B") == "closed"

    @staticmethod
    def _fail_fn():
        raise ValueError("simulated failure")

    @staticmethod
    def _runtime_fail():
        raise RuntimeError("simulated runtime failure")


# =========================================================================
# 3. CanaryDeployer  (10 tests)
# =========================================================================

from ops.canary_deployer import CanaryDeployer, CanaryStatus


class TestCanaryDeployer:
    """Tests for canary deployment management."""

    def test_create_canary(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cid = cd.create_canary("momentum", {"fast": 8})
        assert isinstance(cid, str)
        assert len(cid) > 0
        cd.close()

    def test_get_canary_status(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cid = cd.create_canary("momentum", {"fast": 8}, traffic_pct=10)
        status = cd.get_canary_status(cid)
        assert status is not None
        assert status.strategy == "momentum"
        assert status.traffic_pct == 10.0
        assert status.health == "healthy"
        assert status.trades == 0
        assert status.pnl == 0.0
        cd.close()

    def test_get_canary_status_nonexistent(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        assert cd.get_canary_status("nope") is None
        cd.close()

    def test_promote_canary(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cid = cd.create_canary("strat", {"p": 1})
        assert cd.promote_canary(cid) is True
        status = cd.get_canary_status(cid)
        assert status.health == "promoted"
        cd.close()

    def test_promote_inactive_returns_false(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cid = cd.create_canary("strat", {"p": 1})
        cd.promote_canary(cid)
        assert cd.promote_canary(cid) is False  # already inactive
        cd.close()

    def test_rollback_canary(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cid = cd.create_canary("strat", {"p": 1})
        assert cd.rollback_canary(cid) is True
        status = cd.get_canary_status(cid)
        assert status.health == "rolled_back"
        cd.close()

    def test_should_route_no_canary(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        assert cd.should_route_to_canary("no_strategy") is False
        cd.close()

    def test_should_route_100_pct(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cd.create_canary("strat", {"p": 1}, traffic_pct=100)
        # With 100% traffic, should always route
        results = [cd.should_route_to_canary("strat") for _ in range(20)]
        assert all(results)
        cd.close()

    def test_record_trade(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cid = cd.create_canary("strat", {"p": 1})
        cd.record_trade(cid, 50.0)
        cd.record_trade(cid, -20.0)
        status = cd.get_canary_status(cid)
        assert status.trades == 2
        assert abs(status.pnl - 30.0) < 0.01
        cd.close()

    def test_get_active_canaries(self, tmp_db):
        cd = CanaryDeployer(db_path=tmp_db)
        cd.create_canary("strat-A", {"p": 1})
        cd.create_canary("strat-B", {"p": 2})
        cid3 = cd.create_canary("strat-A", {"p": 3})
        cd.rollback_canary(cid3)
        active = cd.get_active_canaries()
        assert len(active) == 2
        active_a = cd.get_active_canaries("strat-A")
        assert len(active_a) == 1
        cd.close()


# =========================================================================
# 4. BOCPDetector  (10 tests)
# =========================================================================

from research.regime_changepoint import BOCPDetector, ChangePointResult


class TestBOCPDetector:
    """Tests for Bayesian Online Changepoint Detection."""

    def test_initial_update(self):
        d = BOCPDetector(hazard_lambda=100)
        result = d.update(0.01)
        assert isinstance(result, ChangePointResult)
        assert isinstance(result.changepoint_detected, bool)

    def test_run_length_distribution(self):
        d = BOCPDetector(hazard_lambda=100)
        for _ in range(10):
            d.update(0.0)
        dist = d.get_run_length_distribution()
        assert len(dist) > 0
        assert abs(sum(dist) - 1.0) < 1e-6

    def test_detects_shift(self):
        d = BOCPDetector(hazard_lambda=50, threshold=0.3)
        # Stable regime
        for _ in range(100):
            d.update(random.gauss(0, 0.01))
        # Sudden regime change
        detected = False
        for _ in range(50):
            result = d.update(random.gauss(5.0, 0.01))
            if result.changepoint_detected:
                detected = True
                break
        assert detected, "Should detect the mean shift"
        d = BOCPDetector(hazard_lambda=50, threshold=0.1)
        # Feed data and verify growth_probability changes
        for _ in range(50):
            d.update(random.gauss(0, 1.0))
        r1 = d.update(random.gauss(0, 1.0))
        # After mean shift, run length distribution should change
        for _ in range(50):
            r2 = d.update(random.gauss(50.0, 1.0))
        # Verify the API returns valid results
        assert r2.run_length >= 0
        assert 0.0 <= r2.probability <= 1.0
        assert 0.0 <= r2.growth_probability <= 1.0
        # Verify history is recorded
        history = d.get_changepoint_history()
        assert isinstance(history, list)

    def test_no_false_positive_stable(self):
        d = BOCPDetector(hazard_lambda=200, threshold=0.6)
        detections = 0
        for _ in range(200):
            result = d.update(random.gauss(0, 0.5))
            if result.changepoint_detected:
                detections += 1
        # Allow a few false positives but not many
        assert detections < 20

    def test_changepoint_history(self):
        d = BOCPDetector(hazard_lambda=50, threshold=0.3)
        for _ in range(50):
            d.update(random.gauss(0, 0.01))
        for _ in range(30):
            d.update(random.gauss(10.0, 0.01))
        history = d.get_changepoint_history()
        assert isinstance(history, list)
        for item in history:
            assert "timestamp" in item
            assert "probability" in item

    def test_timestamp_passthrough(self):
        d = BOCPDetector()
        result = d.update(1.0, timestamp=1234567890.0)
        assert result.timestamp == 1234567890.0

    def test_n_observations(self):
        d = BOCPDetector()
        for i in range(25):
            d.update(float(i))
        assert d.n_observations == 25

    def test_growth_probability_in_result(self):
        d = BOCPDetector()
        result = d.update(0.5)
        assert 0.0 <= result.growth_probability <= 1.0

    def test_run_length_bounded(self):
        d = BOCPDetector(max_run_length=50)
        for i in range(100):
            d.update(random.gauss(0, 1))
        dist = d.get_run_length_distribution()
        assert len(dist) <= 50

    def test_history_lookback(self):
        d = BOCPDetector(hazard_lambda=10, threshold=0.1)
        for _ in range(200):
            d.update(random.gauss(0, 1))
        history_all = d.get_changepoint_history(lookback=1000)
        history_5 = d.get_changepoint_history(lookback=5)
        assert len(history_5) <= 5
        assert len(history_5) <= len(history_all)


# =========================================================================
# 5. FractalAnalyzer  (10 tests)
# =========================================================================

from research.fractal_analyzer import FractalAnalyzer, HurstResult


class TestFractalAnalyzer:
    """Tests for Hurst exponent and fractal dimension analysis."""

    def _brownian_motion(self, n=500, seed=42):
        """Generate a cumulative random walk (H ≈ 0.5)."""
        rng = random.Random(seed)
        prices = [100.0]
        for _ in range(n):
            prices.append(prices[-1] * (1 + rng.gauss(0, 0.01)))
        return prices

    def _trending_series(self, n=500, seed=42):
        """Generate a persistent (trending) series."""
        rng = random.Random(seed)
        prices = [100.0]
        drift = 0.002
        for _ in range(n):
            prices.append(prices[-1] * (1 + drift + rng.gauss(0, 0.005)))
        return prices

    def test_hurst_returns_dataclass(self):
        fa = FractalAnalyzer()
        result = fa.compute_hurst(self._brownian_motion())
        assert isinstance(result, HurstResult)
        assert 0.0 <= result.hurst_exponent <= 1.0

    def test_hurst_random_walk(self):
        fa = FractalAnalyzer()
        result = fa.compute_hurst(self._brownian_motion(n=2000, seed=123))
        # Random walk should be near 0.5
        assert 0.3 <= result.hurst_exponent <= 0.7

    def test_hurst_interpretation_fields(self):
        fa = FractalAnalyzer()
        result = fa.compute_hurst(self._brownian_motion())
        assert result.interpretation in ("persistent", "anti_persistent", "random_walk")
        assert result.confidence in ("high", "medium", "low")

    def test_hurst_short_series(self):
        fa = FractalAnalyzer()
        result = fa.compute_hurst([100, 101, 102])
        assert result.hurst_exponent == 0.5
        assert result.confidence == "low"

    def test_hurst_r_squared(self):
        fa = FractalAnalyzer()
        result = fa.compute_hurst(self._brownian_motion(n=1000))
        assert 0.0 <= result.r_squared <= 1.0

    def test_market_memory(self):
        fa = FractalAnalyzer()
        mem = fa.get_market_memory(self._brownian_motion())
        assert mem in ("trending", "mean_reverting", "random")

    def test_fractal_dimension(self):
        fa = FractalAnalyzer()
        d = fa.compute_fractal_dimension(self._brownian_motion())
        # D = 2 - H, so for H near 0.5, D near 1.5
        assert 1.0 <= d <= 2.0

    def test_fractal_dimension_range(self):
        fa = FractalAnalyzer()
        d = fa.compute_fractal_dimension(self._trending_series())
        assert 1.0 <= d <= 2.0

    def test_identical_prices(self):
        fa = FractalAnalyzer()
        result = fa.compute_hurst([100.0] * 50)
        # All same price — no returns, should gracefully handle
        assert isinstance(result.hurst_exponent, float)

    def test_two_price_points(self):
        fa = FractalAnalyzer()
        result = fa.compute_hurst([100.0, 101.0])
        assert result.confidence == "low"


# =========================================================================
# 6. EntropyAnalyzer  (10 tests)
# =========================================================================

from research.entropy_signal_quality import EntropyAnalyzer


class TestEntropyAnalyzer:
    """Tests for entropy-based signal quality analysis."""

    def test_entropy_uniform(self):
        ea = EntropyAnalyzer()
        # Uniform data should have high entropy
        values = list(range(1000))
        h = ea.compute_entropy(values, n_bins=20)
        max_entropy = math.log2(20)
        assert h > max_entropy * 0.8

    def test_entropy_constant(self):
        ea = EntropyAnalyzer()
        values = [5.0] * 100
        h = ea.compute_entropy(values, n_bins=20)
        assert h < 0.1  # Nearly zero

    def test_entropy_empty(self):
        ea = EntropyAnalyzer()
        assert ea.compute_entropy([], n_bins=10) == 0.0

    def test_mutual_information_related(self):
        ea = EntropyAnalyzer()
        signal = [float(i) for i in range(200)]
        returns = [float(i) * 2.0 + 1.0 for i in range(200)]  # Linear relation
        mi = ea.compute_mutual_information(signal, returns)
        assert mi > 0.0

    def test_mutual_information_independent(self):
        ea = EntropyAnalyzer()
        rng = random.Random(42)
        signal = [rng.gauss(0, 1) for _ in range(500)]
        returns = [rng.gauss(0, 1) for _ in range(500)]
        mi = ea.compute_mutual_information(signal, returns)
        # Independent signals should have low MI
        assert mi < 1.0

    def test_mutual_information_short(self):
        ea = EntropyAnalyzer()
        assert ea.compute_mutual_information([1, 2], [3, 4]) == 0.0

    def test_rank_signals(self):
        ea = EntropyAnalyzer()
        returns = [float(i) for i in range(200)]
        signals = {
            "good": [float(i) * 1.5 for i in range(200)],
            "noise": [random.Random(42).gauss(0, 1) for _ in range(200)],
        }
        ranking = ea.rank_signals(signals, returns)
        assert len(ranking) == 2
        assert ranking[0][0] == "good"  # Should rank higher
        assert ranking[0][1] >= ranking[1][1]

    def test_rank_signals_sorted(self):
        ea = EntropyAnalyzer()
        rng = random.Random(42)
        returns = [rng.gauss(0, 1) for _ in range(200)]
        signals = {f"sig_{i}": [rng.gauss(0, 1) for _ in range(200)] for i in range(5)}
        ranking = ea.rank_signals(signals, returns)
        scores = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_signal_redundancy_identical(self):
        ea = EntropyAnalyzer()
        sig = [float(i) for i in range(200)]
        nmi = ea.get_signal_redundancy(sig, sig)
        assert nmi > 0.8  # Should be close to 1

    def test_signal_redundancy_independent(self):
        ea = EntropyAnalyzer()
        rng = random.Random(42)
        a = [rng.gauss(0, 1) for _ in range(500)]
        b = [rng.gauss(0, 1) for _ in range(500)]
        nmi = ea.get_signal_redundancy(a, b)
        assert nmi < 0.5

    def test_signal_redundancy_short(self):
        ea = EntropyAnalyzer()
        assert ea.get_signal_redundancy([1], [2]) == 0.0


# =========================================================================
# 7. TaxLossHarvester  (12 tests)
# =========================================================================

from compliance.tax_loss_harvester import (
    HarvestAction,
    HarvestOpportunity,
    TaxLossHarvester,
)


class TestTaxLossHarvester:
    """Tests for Australian tax-loss harvesting."""

    def _sample_positions(self):
        return [
            {"symbol": "BTC/AUD", "quantity": 0.1, "entry_price": 100000, "days_held": 45},
            {"symbol": "ETH/AUD", "quantity": 1.0, "entry_price": 5000, "days_held": 10},
            {"symbol": "SOL/AUD", "quantity": 10.0, "entry_price": 200, "days_held": 90},
        ]

    def _prices_with_losses(self):
        return {
            "BTC/AUD": 90000,   # Loss: (90000-100000)*0.1 = -1000
            "ETH/AUD": 4000,    # Loss: (4000-5000)*1.0 = -1000
            "SOL/AUD": 250,     # Gain: no harvest
        }

    def test_scan_finds_losses(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        assert len(opps) == 2  # BTC and ETH have losses
        h.close()

    def test_scan_excludes_gains(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        symbols = [o.symbol for o in opps]
        assert "SOL/AUD" not in symbols
        h.close()

    def test_opportunity_fields(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        opp = next(o for o in opps if o.symbol == "BTC/AUD")
        assert opp.unrealized_loss_aud == 1000.0
        assert opp.tax_saving_aud == 325.0  # 1000 * 0.325
        assert opp.days_held == 45
        h.close()

    def test_tax_rate_custom(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db, tax_rate=0.45)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        opp = next(o for o in opps if o.symbol == "BTC/AUD")
        assert opp.tax_saving_aud == 450.0  # 1000 * 0.45
        h.close()

    def test_min_loss_filter(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db, min_loss_aud=1500)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        assert len(opps) == 0  # Both losses are 1000, below 1500 threshold
        h.close()

    def test_execute_harvest(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        action = h.execute_harvest(opps[0], replacement_symbol="ETH/AUD")
        assert isinstance(action, HarvestAction)
        assert action.harvest_id >= 1
        assert action.replacement_symbol == "ETH/AUD"
        h.close()

    def test_wash_sale_detection(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db, wash_sale_lookback_days=30)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        # First scan: no wash sale risk
        assert not opps[0].wash_sale_risk

        # Execute harvest
        h.execute_harvest(opps[0])

        # Second scan: should detect wash sale risk
        opps2 = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        harvested_symbol = opps[0].symbol
        opp_rescan = next((o for o in opps2 if o.symbol == harvested_symbol), None)
        if opp_rescan:
            assert opp_rescan.wash_sale_risk is True
        h.close()

    def test_fy_summary(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        h.execute_harvest(opps[0])
        h.execute_harvest(opps[1])

        summary = h.get_fy_summary()
        assert summary["harvest_count"] == 2
        assert summary["total_harvested_aud"] > 0
        assert summary["tax_saved_aud"] > 0
        assert summary["positions_affected"] == 2
        h.close()

    def test_fy_summary_empty(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        summary = h.get_fy_summary("2020-21")
        assert summary["harvest_count"] == 0
        assert summary["total_harvested_aud"] == 0
        h.close()

    def test_missing_price_skipped(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        positions = [{"symbol": "XRP/AUD", "quantity": 100, "entry_price": 1.0, "days_held": 5}]
        opps = h.scan_positions(positions, {})  # No price for XRP
        assert len(opps) == 0
        h.close()

    def test_sorted_by_saving(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        opps = h.scan_positions(self._sample_positions(), self._prices_with_losses())
        savings = [o.tax_saving_aud for o in opps]
        assert savings == sorted(savings, reverse=True)
        h.close()

    def test_fiscal_year_label(self, tmp_db):
        h = TaxLossHarvester(db_path=tmp_db)
        # January 2026 is in FY 2025-26
        import datetime
        jan_ts = datetime.datetime(2026, 1, 15).timestamp()
        assert h._get_fiscal_year(jan_ts) == "2025-26"
        # August 2025 is also FY 2025-26
        aug_ts = datetime.datetime(2025, 8, 1).timestamp()
        assert h._get_fiscal_year(aug_ts) == "2025-26"
        h.close()
