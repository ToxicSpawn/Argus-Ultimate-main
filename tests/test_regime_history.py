"""Push 94 — Tests for RegimeHistoryBuffer and /regime/history + /regime/stats endpoints."""
from __future__ import annotations

import time
import pytest

from core.regime_history_buffer import RegimeHistoryBuffer, RegimeTransition, RegimeHistoryStats


# ---------------------------------------------------------------------------
# RegimeHistoryBuffer unit tests
# ---------------------------------------------------------------------------

class TestRegimeHistoryBuffer:

    def test_empty_buffer(self):
        buf = RegimeHistoryBuffer(maxlen=10)
        assert len(buf) == 0
        assert buf.latest is None
        assert buf.transitions == []

    def test_first_record(self):
        buf = RegimeHistoryBuffer()
        t = buf.record("HIGH_VOL")
        assert t is not None
        assert t.seq == 1
        assert t.from_regime is None
        assert t.to_regime == "HIGH_VOL"
        assert t.duration_secs is None
        assert len(buf) == 1

    def test_no_duplicate_transition(self):
        buf = RegimeHistoryBuffer()
        buf.record("HIGH_VOL")
        result = buf.record("HIGH_VOL")  # same regime — should be ignored
        assert result is None
        assert len(buf) == 1

    def test_transition_recorded(self):
        buf = RegimeHistoryBuffer()
        buf.record("HIGH_VOL", timestamp=1000.0)
        t2 = buf.record("TRENDING", timestamp=1060.0)
        assert t2 is not None
        assert t2.from_regime == "HIGH_VOL"
        assert t2.to_regime == "TRENDING"
        assert t2.duration_secs == 60.0
        assert t2.seq == 2

    def test_maxlen_ring_behaviour(self):
        buf = RegimeHistoryBuffer(maxlen=3)
        regimes = ["A", "B", "C", "D", "E"]
        ts = 1000.0
        for r in regimes:
            buf.record(r, timestamp=ts)
            ts += 10.0
        assert len(buf) == 3
        assert [t.to_regime for t in buf.transitions] == ["C", "D", "E"]

    def test_since_filter(self):
        buf = RegimeHistoryBuffer()
        buf.record("A", timestamp=1000.0)
        buf.record("B", timestamp=1100.0)
        buf.record("C", timestamp=1200.0)
        result = buf.since(1050.0)
        assert len(result) == 2
        assert result[0].to_regime == "B"

    def test_last_n(self):
        buf = RegimeHistoryBuffer()
        for i, r in enumerate(["A", "B", "C", "D", "E"]):
            buf.record(r, timestamp=float(1000 + i * 10))
        assert [t.to_regime for t in buf.last_n(3)] == ["C", "D", "E"]

    def test_clear(self):
        buf = RegimeHistoryBuffer()
        buf.record("A")
        buf.record("B")
        buf.clear()
        assert len(buf) == 0
        assert buf.latest is None

    def test_stats_empty(self):
        buf = RegimeHistoryBuffer()
        s = buf.stats()
        assert s.total_transitions == 0
        assert s.current_regime is None

    def test_stats_populated(self):
        buf = RegimeHistoryBuffer()
        buf.record("HIGH_VOL", timestamp=1000.0)
        buf.record("TRENDING",  timestamp=1060.0)  # 60 s in HIGH_VOL
        buf.record("QUIET",     timestamp=1120.0)  # 60 s in TRENDING
        s = buf.stats()
        assert s.total_transitions == 3
        assert s.current_regime == "QUIET"
        assert "HIGH_VOL" in s.regime_counts
        assert s.avg_duration_secs == 60.0
        assert s.min_duration_secs == 60.0
        assert s.max_duration_secs == 60.0

    def test_to_dict_fields(self):
        buf = RegimeHistoryBuffer()
        buf.record("HIGH_VOL", timestamp=1000.0, context={"vol": 0.42})
        t = buf.latest
        d = t.to_dict()
        assert d["seq"] == 1
        assert d["to_regime"] == "HIGH_VOL"
        assert d["context"]["vol"] == 0.42
        assert "iso" in d

    def test_iso_format(self):
        buf = RegimeHistoryBuffer()
        buf.record("X", timestamp=0.0)
        assert buf.latest.iso.endswith("Z")

    def test_repr(self):
        buf = RegimeHistoryBuffer(maxlen=50)
        assert "maxlen=50" in repr(buf)

    def test_invalid_maxlen(self):
        with pytest.raises(ValueError):
            RegimeHistoryBuffer(maxlen=0)

    def test_context_stored(self):
        buf = RegimeHistoryBuffer()
        ctx = {"vol_ratio": 1.5, "trend_score": 0.7}
        t = buf.record("HIGH_VOL", context=ctx)
        assert t.context["vol_ratio"] == 1.5

    def test_thread_safety_basic(self):
        import threading
        buf = RegimeHistoryBuffer(maxlen=500)
        regimes = ["A", "B", "C", "D", "E"]
        errors = []

        def writer(r: str, base_ts: float):
            try:
                for i in range(20):
                    buf.record(r, timestamp=base_ts + i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(r, float(i * 1000))) for i, r in enumerate(regimes)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        assert len(buf) > 0


# ---------------------------------------------------------------------------
# API endpoint tests (no FastAPI server required — direct route call)
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    from core.api.app import create_app, AppContext
    _TC = True
except ImportError:
    _TC = False


@pytest.mark.skipif(not _TC, reason="fastapi not installed")
class TestRegimeHistoryEndpoints:

    def _make_client(self, buf=None):
        ctx = AppContext(regime_history=buf)
        app = create_app(ctx)
        return TestClient(app)

    def test_history_unwired(self):
        client = self._make_client(buf=None)
        r = client.get("/regime/history")
        assert r.status_code == 200
        body = r.json()
        assert body["history_wired"] is False
        assert body["transitions"] == []

    def test_history_empty_buffer(self):
        buf = RegimeHistoryBuffer()
        client = self._make_client(buf=buf)
        r = client.get("/regime/history")
        assert r.status_code == 200
        body = r.json()
        assert body["history_wired"] is True
        assert body["count"] == 0

    def test_history_populated(self):
        buf = RegimeHistoryBuffer()
        buf.record("HIGH_VOL", timestamp=1000.0)
        buf.record("TRENDING",  timestamp=1060.0)
        buf.record("QUIET",     timestamp=1120.0)
        client = self._make_client(buf=buf)
        r = client.get("/regime/history?limit=10")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        assert body["transitions"][0]["to_regime"] == "HIGH_VOL"
        assert body["transitions"][2]["to_regime"] == "QUIET"

    def test_history_limit(self):
        buf = RegimeHistoryBuffer()
        ts = 1000.0
        for r in ["A", "B", "C", "D", "E"]:
            buf.record(r, timestamp=ts)
            ts += 10.0
        client = self._make_client(buf=buf)
        r = client.get("/regime/history?limit=2")
        body = r.json()
        assert body["count"] == 2
        assert body["transitions"][1]["to_regime"] == "E"

    def test_history_since_filter(self):
        buf = RegimeHistoryBuffer()
        buf.record("A", timestamp=1000.0)
        buf.record("B", timestamp=1100.0)
        buf.record("C", timestamp=1200.0)
        client = self._make_client(buf=buf)
        r = client.get("/regime/history?since=1050.0")
        body = r.json()
        assert body["count"] == 2
        assert body["transitions"][0]["to_regime"] == "B"

    def test_stats_unwired(self):
        client = self._make_client(buf=None)
        r = client.get("/regime/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["history_wired"] is False
        assert body["total_transitions"] == 0

    def test_stats_populated(self):
        buf = RegimeHistoryBuffer()
        buf.record("HIGH_VOL", timestamp=1000.0)
        buf.record("TRENDING",  timestamp=1060.0)
        buf.record("QUIET",     timestamp=1120.0)
        client = self._make_client(buf=buf)
        r = client.get("/regime/stats")
        body = r.json()
        assert body["history_wired"] is True
        assert body["total_transitions"] == 3
        assert body["current_regime"] == "QUIET"
        assert body["avg_duration_secs"] == 60.0

    def test_transition_model_fields(self):
        buf = RegimeHistoryBuffer()
        buf.record("HIGH_VOL", timestamp=1000.0, context={"vol": 0.5})
        buf.record("QUIET",    timestamp=1060.0)
        client = self._make_client(buf=buf)
        body = client.get("/regime/history").json()
        t1 = body["transitions"][0]
        assert "seq" in t1
        assert "iso" in t1
        assert "from_regime" in t1
        assert "duration_secs" in t1
        assert "context" in t1

    def test_buffer_maxlen_exposed(self):
        buf = RegimeHistoryBuffer(maxlen=77)
        client = self._make_client(buf=buf)
        body = client.get("/regime/history").json()
        assert body["buffer_maxlen"] == 77

    def test_version_bump(self):
        client = self._make_client()
        body = client.get("/health").json()
        assert body["version"] == "8.30.0"
        assert body["codename"] == "RegimeHistory"
