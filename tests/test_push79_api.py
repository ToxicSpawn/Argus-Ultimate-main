"""Push 79 — Tests: PrometheusRegistry, ConnectionManager,
AppContext, FastAPI endpoints. 28 tests.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# PrometheusRegistry (8)
# ---------------------------------------------------------------------------

class TestPrometheusRegistry:
    def _reg(self):
        from core.api.prometheus import PrometheusRegistry
        return PrometheusRegistry()

    def test_instantiates(self):
        r = self._reg()
        assert r is not None

    def test_text_exposition_nonempty(self):
        r = self._reg()
        text = r.text_exposition()
        assert len(text) > 100

    def test_text_has_help_lines(self):
        r = self._reg()
        text = r.text_exposition()
        assert "# HELP" in text

    def test_text_has_type_lines(self):
        r = self._reg()
        text = r.text_exposition()
        assert "# TYPE" in text

    def test_gauge_set(self):
        r = self._reg()
        r.equity.set(12345.67)
        assert r.equity.value == pytest.approx(12345.67)

    def test_counter_inc(self):
        r = self._reg()
        r.signals_total.inc(v=5)
        assert r.signals_total.total() == pytest.approx(5)

    def test_histogram_observe(self):
        r = self._reg()
        r.order_latency.observe(75.0)
        assert r.order_latency._total == 1
        assert r.order_latency._sum == pytest.approx(75.0)

    def test_update_from_risk(self):
        r = self._reg()
        r.update_from_risk({"equity": 9999, "portfolio_heat": 0.45, "kill_switch": True})
        assert r.equity.value == pytest.approx(9999)
        assert r.kill_switch.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ConnectionManager (6)
# ---------------------------------------------------------------------------

class TestConnectionManager:
    def test_instantiates(self):
        from core.api.ws_feed import ConnectionManager
        cm = ConnectionManager()
        assert cm is not None

    def test_initial_count_zero(self):
        from core.api.ws_feed import ConnectionManager
        cm = ConnectionManager()
        assert cm.get_connection_count("prices") == 0

    def test_stats_empty(self):
        from core.api.ws_feed import ConnectionManager
        cm = ConnectionManager()
        s = cm.stats
        assert "connections" in s

    def test_signal_payload(self):
        from core.api.ws_feed import signal_to_ws_payload
        from core.strategy.signal import Signal, SignalSide
        sig = Signal(symbol="BTCUSDT", side=SignalSide.LONG, strength=0.8)
        payload = signal_to_ws_payload(sig)
        assert payload["type"] == "signal"
        assert payload["symbol"] == "BTCUSDT"

    def test_risk_event_payload(self):
        from core.api.ws_feed import risk_event_to_ws_payload
        from core.risk.risk_event import RiskEvent, RiskEventType
        ev = RiskEvent(RiskEventType.KILL_SWITCH, "test")
        payload = risk_event_to_ws_payload(ev)
        assert payload["type"] == "risk_event"
        assert payload["event_type"] == "KILL_SWITCH"

    def test_price_tick_payload(self):
        from core.api.ws_feed import price_tick_to_ws_payload
        p = price_tick_to_ws_payload("BTCUSDT", 55000.0)
        assert p["price"] == 55000.0
        assert p["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# AppContext (3)
# ---------------------------------------------------------------------------

class TestAppContext:
    def test_instantiates(self):
        from core.api.app import AppContext
        ctx = AppContext()
        assert ctx is not None

    def test_has_registry(self):
        from core.api.app import AppContext
        ctx = AppContext()
        assert ctx.registry is not None

    def test_has_ws_manager(self):
        from core.api.app import AppContext
        ctx = AppContext()
        assert ctx.ws_manager is not None


# ---------------------------------------------------------------------------
# FastAPI endpoints via TestClient (11) — skip if fastapi/httpx not installed
# ---------------------------------------------------------------------------

try:
    from fastapi.testclient import TestClient
    import httpx
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi/httpx not installed")
class TestFastAPIEndpoints:
    def _client(self):
        from core.api.app import create_app, AppContext
        from core.api.prometheus import PrometheusRegistry
        from core.api.ws_feed import ConnectionManager
        from core.execution.order_manager import OrderManager
        from core.risk.risk_manager import RiskManager
        from core.strategy.signal_bus import AsyncSignalBus

        ctx = AppContext(
            order_manager=OrderManager(),
            risk_manager=RiskManager(),
            signal_bus=AsyncSignalBus(),
            registry=PrometheusRegistry(),
            ws_manager=ConnectionManager(),
        )
        app = create_app(context=ctx)
        return TestClient(app)

    def test_health_200(self):
        c = self._client()
        r = c.get("/health")
        assert r.status_code == 200

    def test_health_has_version(self):
        c = self._client()
        r = c.get("/health")
        assert r.json()["version"] == "8.15.0"

    def test_status_200(self):
        c = self._client()
        r = c.get("/status")
        assert r.status_code == 200

    def test_status_has_keys(self):
        c = self._client()
        r = c.get("/status")
        data = r.json()
        assert "engine" in data and "risk" in data

    def test_metrics_200(self):
        c = self._client()
        r = c.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type(self):
        c = self._client()
        r = c.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_positions_200(self):
        c = self._client()
        r = c.get("/positions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_orders_200(self):
        c = self._client()
        r = c.get("/orders")
        assert r.status_code == 200

    def test_signals_200(self):
        c = self._client()
        r = c.get("/signals")
        assert r.status_code == 200

    def test_kill_switch_activate(self):
        c = self._client()
        r = c.post("/kill-switch", json={"action": "activate", "reason": "test"})
        assert r.status_code == 200
        assert r.json()["kill_switch_active"] is True

    def test_kill_switch_reset(self):
        c = self._client()
        c.post("/kill-switch", json={"action": "activate"})
        r = c.post("/kill-switch", json={"action": "reset"})
        assert r.json()["kill_switch_active"] is False
