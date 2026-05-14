"""
Tests for the ARGUS web dashboard API endpoints.

Covers the new /api/* endpoints added to api/dashboard.py:
  - POST /api/kill-switch
  - DELETE /api/kill-switch
  - GET  /api/kill-switch
  - GET  /api/trades?limit=N
  - GET  /api/signals?limit=N
  - GET  /api/strategies
  - GET  /api/state
Also validates existing endpoints still work, and that the static
web/ directory contains the expected files.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import the dashboard module
# ---------------------------------------------------------------------------
from api.dashboard import (
    ArgusAPIServer,
    _DEFAULT_STATE,
    _make_fastapi_app,
    _trades_to_csv,
    _signals_to_csv,
)

# Check if FastAPI is available for async tests
try:
    from fastapi.testclient import TestClient
    import fastapi
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_and_lock():
    """Return a fresh (state_dict, lock) pair."""
    state = dict(_DEFAULT_STATE)
    state["status"] = "running"
    state["mode"] = "paper"
    state["cycle"] = 42
    state["capital_aud"] = 1000.0
    state["pnl_aud"] = 12.50
    state["pnl_pct"] = 1.25
    state["drawdown_pct"] = 0.5
    state["var_95"] = 0.0234
    state["circuit_breaker"] = False
    state["regime"] = "NORMAL"
    state["ensemble_bias"] = 0.15
    state["active_strategies"] = ["momentum", "mean_reversion"]
    state["top_signals"] = [
        {"symbol": "BTC/AUD", "action": "buy", "confidence": 0.85, "strategy": "momentum", "status": "executed", "timestamp": time.time()},
        {"symbol": "ETH/AUD", "action": "sell", "confidence": 0.62, "strategy": "mean_reversion", "status": "pending", "timestamp": time.time()},
    ]
    state["trades"] = [
        {"timestamp": "2026-03-18T10:00:00", "symbol": "BTC/AUD", "side": "buy", "qty": 0.001, "price": 140000.0, "pnl": 5.20, "strategy": "momentum", "order_type": "limit"},
        {"timestamp": "2026-03-18T10:05:00", "symbol": "ETH/AUD", "side": "sell", "qty": 0.01, "price": 5600.0, "pnl": -2.10, "strategy": "mean_reversion", "order_type": "market"},
        {"timestamp": "2026-03-18T10:10:00", "symbol": "BTC/AUD", "side": "buy", "qty": 0.002, "price": 140100.0, "pnl": 8.00, "strategy": "momentum", "order_type": "limit"},
    ]
    state["positions"] = {
        "BTC/AUD": {"qty": 0.003, "entry_price": 140050.0, "current_price": 140200.0, "unrealised_pnl": 4.50, "side": "LONG"},
    }
    state["components"] = {"components": {"momentum": True, "risk_manager": True, "exchange_manager": True}}
    state["models"] = {"regime_classifier": "loaded", "rl_agent": "loaded"}
    state["strategy_stats"] = {
        "momentum": {"trades": 15, "win_rate": 0.67, "total_pnl": 45.0, "sharpe": 1.8, "status": "active"},
        "mean_reversion": {"trades": 8, "win_rate": 0.50, "total_pnl": -5.0, "sharpe": 0.3, "status": "cooldown"},
    }
    lock = threading.Lock()
    return state, lock


@pytest.fixture
def client(state_and_lock):
    """Return a FastAPI TestClient for the dashboard app."""
    if not _FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")
    state, lock = state_and_lock
    app = _make_fastapi_app(state, lock)
    return TestClient(app)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporarily override the data directory for kill-switch tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


# ---------------------------------------------------------------------------
# Test: Static web files exist
# ---------------------------------------------------------------------------

class TestWebStaticFiles:
    """Verify the web/ directory contains the expected SPA files."""

    def test_index_html_exists(self):
        root = Path(__file__).resolve().parent.parent / "web" / "index.html"
        assert root.exists(), f"web/index.html not found at {root}"

    def test_serve_py_exists(self):
        serve = Path(__file__).resolve().parent.parent / "web" / "serve.py"
        assert serve.exists(), f"web/serve.py not found at {serve}"

    def test_index_html_contains_dashboard_elements(self):
        index = Path(__file__).resolve().parent.parent / "web" / "index.html"
        content = index.read_text(encoding="utf-8")
        assert "ARGUS" in content
        assert "WebSocket" in content or "websocket" in content.lower() or "ws://" in content
        assert "positionsBody" in content
        assert "tradesBody" in content
        assert "signalFeed" in content
        assert "killBtn" in content
        assert "riskGauge" in content or "riskScore" in content

    def test_index_html_has_dark_theme(self):
        index = Path(__file__).resolve().parent.parent / "web" / "index.html"
        content = index.read_text(encoding="utf-8")
        # Check for dark background color in CSS
        assert "#0d1117" in content or "0d1117" in content


# ---------------------------------------------------------------------------
# Test: Existing endpoints
# ---------------------------------------------------------------------------

class TestExistingEndpoints:
    """Confirm the pre-existing REST endpoints still work."""

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"
        assert data["cycle"] == 42

    def test_portfolio_endpoint(self, client):
        r = client.get("/portfolio")
        assert r.status_code == 200
        data = r.json()
        assert data["capital_aud"] == 1000.0
        assert data["pnl_aud"] == 12.50

    def test_risk_endpoint(self, client):
        r = client.get("/risk")
        assert r.status_code == 200
        data = r.json()
        assert data["var_95"] == 0.0234
        assert data["circuit_breaker"] is False

    def test_signals_endpoint(self, client):
        r = client.get("/signals")
        assert r.status_code == 200
        data = r.json()
        assert data["regime"] == "NORMAL"
        assert len(data["active_strategies"]) == 2

    def test_trades_endpoint(self, client):
        r = client.get("/trades")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 3


# ---------------------------------------------------------------------------
# Test: New /api/* endpoints
# ---------------------------------------------------------------------------

class TestKillSwitchAPI:
    """Test the kill switch endpoints."""

    def test_kill_switch_activate(self, client, tmp_data_dir):
        """POST /api/kill-switch creates the KILL_SWITCH file."""
        # The endpoint writes to data/KILL_SWITCH relative to the project root.
        # Just verify it returns 200 (file creation uses the real project path).
        r = client.post("/api/kill-switch")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "activated"
        # Clean up
        kill_path = data.get("path", "")
        if kill_path and os.path.exists(kill_path):
            os.remove(kill_path)

    def test_kill_switch_deactivate_not_active(self, client):
        """DELETE /api/kill-switch when no file exists returns not_active."""
        with mock.patch("api.dashboard.os.path.exists", return_value=False):
            r = client.delete("/api/kill-switch")
            assert r.status_code == 200
            data = r.json()
            assert data["status"] == "not_active"

    def test_kill_switch_status_inactive(self, client):
        """GET /api/kill-switch returns active=false when file absent."""
        with mock.patch("api.dashboard.os.path.exists", return_value=False):
            r = client.get("/api/kill-switch")
            assert r.status_code == 200
            assert r.json()["active"] is False

    def test_kill_switch_status_active(self, client):
        """GET /api/kill-switch returns active=true when file present."""
        with mock.patch("api.dashboard.os.path.exists", return_value=True):
            r = client.get("/api/kill-switch")
            assert r.status_code == 200
            assert r.json()["active"] is True


class TestAPITradesEndpoint:
    """Test GET /api/trades."""

    def test_trades_returns_all(self, client):
        r = client.get("/api/trades")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_trades_with_limit(self, client):
        r = client.get("/api/trades?limit=2")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2

    def test_trades_default_limit(self, client):
        r = client.get("/api/trades?limit=50")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3  # Only 3 trades in state


class TestAPISignalsEndpoint:
    """Test GET /api/signals."""

    def test_signals_returns_all(self, client):
        r = client.get("/api/signals")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_signals_with_limit(self, client):
        r = client.get("/api/signals?limit=1")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1

    def test_signal_has_expected_fields(self, client):
        r = client.get("/api/signals")
        data = r.json()
        sig = data[0]
        assert "symbol" in sig
        assert "action" in sig
        assert "confidence" in sig


class TestAPIStrategiesEndpoint:
    """Test GET /api/strategies."""

    def test_strategies_returns_list(self, client):
        r = client.get("/api/strategies")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_strategy_has_expected_fields(self, client):
        r = client.get("/api/strategies")
        data = r.json()
        strat = data[0]
        assert "name" in strat
        assert "trades" in strat
        assert "win_rate" in strat
        assert "total_pnl" in strat
        assert "sharpe" in strat
        assert "status" in strat

    def test_strategy_stats_are_correct(self, client):
        r = client.get("/api/strategies")
        data = r.json()
        momentum = next((s for s in data if s["name"] == "momentum"), None)
        assert momentum is not None
        assert momentum["trades"] == 15
        assert momentum["win_rate"] == 0.67
        assert momentum["total_pnl"] == 45.0
        assert momentum["sharpe"] == 1.8

    def test_strategies_fallback_from_trades(self, state_and_lock):
        """When no active_strategies or strategy_stats, build from trades."""
        if not _FASTAPI_AVAILABLE:
            pytest.skip("FastAPI not installed")
        state, lock = state_and_lock
        state["active_strategies"] = []
        state["strategy_stats"] = {}
        app = _make_fastapi_app(state, lock)
        c = TestClient(app)
        r = c.get("/api/strategies")
        assert r.status_code == 200
        data = r.json()
        # Should build from trade data
        assert isinstance(data, list)
        assert len(data) == 2  # momentum + mean_reversion from trades
        names = {s["name"] for s in data}
        assert "momentum" in names
        assert "mean_reversion" in names


class TestAPIFullState:
    """Test GET /api/state."""

    def test_full_state_returns_dict(self, client):
        r = client.get("/api/state")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert "capital_aud" in data
        assert "trades" in data
        assert "positions" in data
        assert "regime" in data


# ---------------------------------------------------------------------------
# Test: CSV helpers
# ---------------------------------------------------------------------------

class TestCSVHelpers:
    """Test CSV export helper functions."""

    def test_trades_to_csv_empty(self):
        csv = _trades_to_csv([])
        assert "timestamp,symbol,side,qty,price,pnl" in csv

    def test_trades_to_csv_with_data(self):
        trades = [{"timestamp": "2026-01-01", "symbol": "BTC/AUD", "side": "buy", "qty": 0.1, "price": 100000, "pnl": 50}]
        csv = _trades_to_csv(trades)
        assert "BTC/AUD" in csv
        assert "buy" in csv

    def test_signals_to_csv_empty(self):
        csv = _signals_to_csv([])
        assert "index,signal" in csv

    def test_signals_to_csv_with_dicts(self):
        signals = [{"symbol": "BTC", "action": "buy"}]
        csv = _signals_to_csv(signals)
        assert "BTC" in csv


# ---------------------------------------------------------------------------
# Test: ArgusAPIServer class
# ---------------------------------------------------------------------------

class TestArgusAPIServerUnit:
    """Test ArgusAPIServer state management (no server start)."""

    def test_update_state(self):
        server = ArgusAPIServer(port=19999)
        server.update_state("capital_aud", 5000.0)
        assert server.get_state("capital_aud") == 5000.0

    def test_update_states_bulk(self):
        server = ArgusAPIServer(port=19999)
        server.update_states({"capital_aud": 2000.0, "regime": "CRISIS"})
        assert server.get_state("capital_aud") == 2000.0
        assert server.get_state("regime") == "CRISIS"

    def test_trades_capped_at_20(self):
        server = ArgusAPIServer(port=19999)
        for i in range(30):
            server.update_state("trades", {"symbol": f"T{i}", "pnl": i})
        trades = server.get_state("trades")
        assert len(trades) == 20
        # Should keep the last 20
        assert trades[-1]["symbol"] == "T29"

    def test_last_updated_auto_stamped(self):
        server = ArgusAPIServer(port=19999)
        assert server.get_state("last_updated") is None
        server.update_state("cycle", 1)
        assert server.get_state("last_updated") is not None
        assert isinstance(server.get_state("last_updated"), float)

    def test_mark_started_stopped(self):
        server = ArgusAPIServer(port=19999)
        server.mark_started("live")
        assert server.get_state("status") == "running"
        assert server.get_state("mode") == "live"
        server.mark_stopped()
        assert server.get_state("status") == "stopped"
