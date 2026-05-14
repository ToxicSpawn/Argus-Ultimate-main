"""Push 95 — Tests for RegimeHistoryWiring and /alert-rules endpoints."""
from __future__ import annotations

import time
import threading
import pytest

from core.regime_history_buffer import RegimeHistoryBuffer
from core.regime_history_wiring import (
    wire_regime_history,
    SnapshotPoller,
    RegimeTransitionCallback,
    _default_context_extractor,
)


# ---------------------------------------------------------------------------
# Fake detector helpers
# ---------------------------------------------------------------------------

class FakeDetectorWithHook:
    """Has register_transition_callback — Strategy 1."""
    def __init__(self):
        self._cbs = []
    def register_transition_callback(self, cb):
        self._cbs.append(cb)
    def fire(self, regime, ctx=None):
        for cb in self._cbs:
            cb(regime, ctx or {})


class FakeDetectorWithAttr:
    """Has on_transition attribute — Strategy 2."""
    def __init__(self):
        self.on_transition = None
    def fire(self, regime, ctx=None):
        if callable(self.on_transition):
            self.on_transition(regime, ctx or {})


class FakeDetectorSnapshot:
    """Only has snapshot() — Strategy 3 (poller fallback)."""
    def __init__(self, regime="UNKNOWN"):
        self._regime = regime
        self._vol    = 1.0
    def snapshot(self):
        return {"regime": self._regime, "vol_ratio": self._vol}
    def set_regime(self, r):
        self._regime = r


class FakeAppContext:
    regime_history = None
    alert_config   = None


# ---------------------------------------------------------------------------
# wire_regime_history — Strategy 1
# ---------------------------------------------------------------------------

class TestWiringStrategy1:

    def test_buf_attached_to_context(self):
        ctx = FakeAppContext()
        det = FakeDetectorWithHook()
        buf = wire_regime_history(ctx, det)
        assert ctx.regime_history is buf

    def test_transition_recorded_via_callback(self):
        ctx = FakeAppContext()
        det = FakeDetectorWithHook()
        buf = wire_regime_history(ctx, det)
        det.fire("HIGH_VOL", {"vol_ratio": 2.5})
        det.fire("TRENDING", {"vol_ratio": 0.8})
        assert len(buf) == 2
        assert buf.latest.to_regime == "TRENDING"

    def test_dedup_via_callback(self):
        ctx = FakeAppContext()
        det = FakeDetectorWithHook()
        buf = wire_regime_history(ctx, det)
        det.fire("HIGH_VOL")
        det.fire("HIGH_VOL")  # duplicate
        assert len(buf) == 1

    def test_custom_buf_used(self):
        ctx  = FakeAppContext()
        det  = FakeDetectorWithHook()
        existing = RegimeHistoryBuffer(maxlen=10)
        buf  = wire_regime_history(ctx, det, buf=existing)
        assert buf is existing
        assert ctx.regime_history is existing


# ---------------------------------------------------------------------------
# wire_regime_history — Strategy 2
# ---------------------------------------------------------------------------

class TestWiringStrategy2:

    def test_on_transition_patched(self):
        ctx = FakeAppContext()
        det = FakeDetectorWithAttr()
        buf = wire_regime_history(ctx, det)
        det.fire("RANGING")
        assert len(buf) == 1
        assert buf.latest.to_regime == "RANGING"

    def test_original_callback_chained(self):
        ctx  = FakeAppContext()
        det  = FakeDetectorWithAttr()
        fired = []
        det.on_transition = lambda r, c: fired.append(r)
        wire_regime_history(ctx, det)
        det.fire("HIGH_VOL")
        assert "HIGH_VOL" in fired  # original still called
        assert len(ctx.regime_history) == 1


# ---------------------------------------------------------------------------
# wire_regime_history — Strategy 3 (SnapshotPoller)
# ---------------------------------------------------------------------------

class TestSnapshotPoller:

    def test_poller_records_initial_regime(self):
        det = FakeDetectorSnapshot(regime="QUIET")
        buf = RegimeHistoryBuffer(maxlen=100)
        poller = SnapshotPoller(
            detector=det, buf=buf,
            context_extractor=_default_context_extractor,
            poll_interval=0.05,
        )
        poller.start()
        time.sleep(0.15)
        poller.stop()
        assert len(buf) >= 1
        assert buf.transitions[0].to_regime == "QUIET"

    def test_poller_records_transition(self):
        det = FakeDetectorSnapshot(regime="QUIET")
        buf = RegimeHistoryBuffer(maxlen=100)
        poller = SnapshotPoller(
            detector=det, buf=buf,
            context_extractor=_default_context_extractor,
            poll_interval=0.05,
        )
        poller.start()
        time.sleep(0.12)
        det.set_regime("HIGH_VOL")
        time.sleep(0.12)
        poller.stop()
        regimes = [t.to_regime for t in buf.transitions]
        assert "QUIET" in regimes
        assert "HIGH_VOL" in regimes

    def test_wiring_strategy3_attaches_poller(self):
        ctx = FakeAppContext()
        det = FakeDetectorSnapshot(regime="TRENDING")
        buf = wire_regime_history(ctx, det, poll_interval=0.05) if False else wire_regime_history(ctx, det)
        # poller stored on detector to prevent GC
        assert hasattr(det, "_history_poller")
        assert isinstance(det._history_poller, SnapshotPoller)
        det._history_poller.stop()


# ---------------------------------------------------------------------------
# RegimeTransitionCallback
# ---------------------------------------------------------------------------

class TestRegimeTransitionCallback:

    def test_call_records(self):
        buf = RegimeHistoryBuffer()
        cb  = RegimeTransitionCallback(buf)
        cb("HIGH_VOL", {"vol_ratio": 2.1})
        assert len(buf) == 1
        assert buf.latest.context["vol_ratio"] == 2.1

    def test_from_snapshot_dict(self):
        buf = RegimeHistoryBuffer()
        cb  = RegimeTransitionCallback(buf)
        cb.from_snapshot({"regime": "RANGING", "vol_ratio": 0.9, "confidence": 0.7})
        assert buf.latest.to_regime == "RANGING"
        assert buf.latest.context["confidence"] == 0.7


# ---------------------------------------------------------------------------
# /alert-rules endpoint tests
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    from core.api.app import create_app, AppContext
    _TC = True
except ImportError:
    _TC = False


@pytest.mark.skipif(not _TC, reason="fastapi not installed")
class TestAlertRulesEndpoints:

    def _client(self, alert_config=None):
        ctx = AppContext(alert_config=alert_config)
        return TestClient(create_app(ctx))

    def test_get_default_rules(self):
        client = self._client()
        r = client.get("/alert-rules")
        assert r.status_code == 200
        body = r.json()
        assert body["alerts_wired"] is True
        assert body["count"] == 6  # 6 defaults
        names = [rule["name"] for rule in body["rules"]]
        assert "vol_spike_ratio" in names
        assert "regime_dwell_min_secs" in names

    def test_upsert_new_rule(self):
        client = self._client()
        r = client.post("/alert-rules", json={"name": "custom_rule", "value": 99.0, "enabled": True})
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "custom_rule"
        assert body["value"] == 99.0

    def test_upsert_updates_existing(self):
        client = self._client()
        client.post("/alert-rules", json={"name": "vol_spike_ratio", "value": 5.0, "enabled": True})
        r = client.get("/alert-rules")
        rules = {rule["name"]: rule for rule in r.json()["rules"]}
        assert rules["vol_spike_ratio"]["value"] == 5.0

    def test_delete_rule(self):
        client = self._client()
        r = client.delete("/alert-rules/vol_spike_ratio")
        assert r.status_code == 200
        body = r.json()
        assert body["deleted"] is True
        r2 = client.get("/alert-rules")
        names = [rule["name"] for rule in r2.json()["rules"]]
        assert "vol_spike_ratio" not in names

    def test_delete_nonexistent_rule(self):
        client = self._client()
        r = client.delete("/alert-rules/does_not_exist")
        assert r.status_code == 200
        assert r.json()["deleted"] is False

    def test_disable_rule(self):
        client = self._client()
        client.post("/alert-rules", json={"name": "kill_switch_auto_pct", "value": 10.0, "enabled": False})
        r = client.get("/alert-rules")
        rules = {rule["name"]: rule for rule in r.json()["rules"]}
        assert rules["kill_switch_auto_pct"]["enabled"] is False

    def test_version_bump(self):
        client = self._client()
        body = client.get("/health").json()
        assert body["version"] == "8.31.0"
        assert body["codename"] == "AlertRules"
