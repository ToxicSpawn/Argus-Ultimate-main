"""Tests for api/dashboard.py — new P&L history, strategy comparison, and trade replay endpoints."""
from __future__ import annotations

import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# We test the FastAPI app factory directly (no server needed)
try:
    from fastapi.testclient import TestClient
    _TESTCLIENT_AVAILABLE = True
except ImportError:
    _TESTCLIENT_AVAILABLE = False

from api.dashboard import _DEFAULT_STATE, _make_fastapi_app, ArgusAPIServer
import threading


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state():
    """Return a mutable copy of the default dashboard state."""
    s = dict(_DEFAULT_STATE)
    s["status"] = "running"
    s["mode"] = "paper"
    s["capital_aud"] = 1000.0
    s["pnl_aud"] = 42.5
    s["pnl_pct"] = 4.25
    s["last_updated"] = time.time()
    return s


@pytest.fixture
def lock():
    return threading.Lock()


@pytest.fixture
def app(state, lock):
    """Create a FastAPI test app."""
    return _make_fastapi_app(state, lock)


@pytest.fixture
def client(app):
    if not _TESTCLIENT_AVAILABLE:
        pytest.skip("fastapi[testclient] not installed")
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/pnl_history tests
# ---------------------------------------------------------------------------

class TestPnLHistory:
    def test_pnl_history_empty_returns_single_point(self, client, state):
        """With no accumulated history, endpoint synthesises a single point."""
        resp = client.get("/api/pnl_history")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "timestamp" in data[0]
        assert "equity" in data[0]
        assert "pnl" in data[0]

    def test_pnl_history_with_accumulated_data(self, client, state):
        """Pre-populated history should be returned."""
        state["pnl_history"] = [
            {"timestamp": time.time() - 60, "equity": 1000.0, "pnl": 0.0},
            {"timestamp": time.time() - 30, "equity": 1010.0, "pnl": 10.0},
            {"timestamp": time.time(), "equity": 1042.5, "pnl": 42.5},
        ]
        resp = client.get("/api/pnl_history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_pnl_history_capped_at_500(self, client, state):
        """History should be capped at 500 entries."""
        state["pnl_history"] = [
            {"timestamp": time.time(), "equity": 1000.0 + i, "pnl": float(i)}
            for i in range(600)
        ]
        resp = client.get("/api/pnl_history")
        data = resp.json()
        assert len(data) == 500


# ---------------------------------------------------------------------------
# /api/strategy_comparison tests
# ---------------------------------------------------------------------------

class TestStrategyComparison:
    def test_empty_returns_empty_list(self, client):
        resp = client.get("/api/strategy_comparison")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_with_active_strategies(self, client, state):
        state["active_strategies"] = ["momentum", "mean_reversion"]
        state["strategy_stats"] = {
            "momentum": {"trades": 10, "total_pnl": 50.0, "sharpe": 1.2},
            "mean_reversion": {"trades": 5, "total_pnl": -10.0, "sharpe": -0.3},
        }
        resp = client.get("/api/strategy_comparison")
        data = resp.json()
        assert len(data) == 2
        names = [d["name"] for d in data]
        assert "momentum" in names
        assert "mean_reversion" in names
        # Check return_pct calculation: 50 / 1000 * 100 = 5.0%
        momentum = next(d for d in data if d["name"] == "momentum")
        assert momentum["return_pct"] == pytest.approx(5.0, rel=0.01)
        assert momentum["trades"] == 10

    def test_fallback_to_trade_derived(self, client, state):
        """When no active_strategies, derive from trades."""
        state["active_strategies"] = []
        state["trades"] = [
            {"strategy": "arb", "pnl": 20.0},
            {"strategy": "arb", "pnl": -5.0},
            {"strategy": "dca", "pnl": 10.0},
        ]
        resp = client.get("/api/strategy_comparison")
        data = resp.json()
        assert len(data) == 2

    def test_sharpe_included(self, client, state):
        state["active_strategies"] = ["test_strat"]
        state["strategy_stats"] = {
            "test_strat": {"trades": 3, "total_pnl": 30.0, "sharpe": 2.5},
        }
        resp = client.get("/api/strategy_comparison")
        data = resp.json()
        assert data[0]["sharpe"] == 2.5


# ---------------------------------------------------------------------------
# /api/trade_replay tests
# ---------------------------------------------------------------------------

class TestTradeReplay:
    def test_empty_trades(self, client):
        resp = client.get("/api/trade_replay")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_with_trades(self, client, state):
        state["trades"] = [
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "price": 60000.0,
                "qty": 0.001,
                "timestamp": "2026-03-23T10:00:00Z",
                "confidence": 0.85,
                "pnl": 5.0,
                "strategy": "momentum",
            },
        ]
        resp = client.get("/api/trade_replay")
        data = resp.json()
        assert len(data) == 1
        trade = data[0]
        assert trade["symbol"] == "BTC/USD"
        assert trade["side"] == "buy"
        assert trade["price"] == 60000.0
        assert trade["quantity"] == 0.001
        assert trade["confidence"] == 0.85
        assert trade["pnl"] == 5.0

    def test_trade_replay_capped_at_50(self, client, state):
        state["trades"] = [
            {"symbol": f"SYM{i}", "side": "buy", "price": 100.0, "qty": 1.0,
             "timestamp": f"2026-03-23T{i:02d}:00:00Z", "pnl": 0.0}
            for i in range(60)
        ]
        resp = client.get("/api/trade_replay")
        data = resp.json()
        assert len(data) == 50

    def test_regime_falls_back_to_state(self, client, state):
        """If trade has no regime, should use state regime."""
        state["regime"] = "TRENDING"
        state["trades"] = [
            {"symbol": "ETH/USD", "side": "sell", "price": 3000.0, "qty": 0.5,
             "timestamp": "2026-03-23T12:00:00Z", "pnl": -2.0},
        ]
        resp = client.get("/api/trade_replay")
        data = resp.json()
        assert data[0]["regime"] == "TRENDING"


# ---------------------------------------------------------------------------
# P&L history accumulation in ArgusAPIServer
# ---------------------------------------------------------------------------

class TestPnLAccumulation:
    def test_update_state_accumulates_pnl(self):
        """Updating capital_aud should add to pnl_history."""
        server = ArgusAPIServer(port=19999)
        server.update_state("capital_aud", 1050.0)
        server.update_state("pnl_aud", 50.0)
        history = server.get_state("pnl_history", [])
        assert len(history) >= 1
        assert history[-1]["equity"] == 1050.0

    def test_pnl_history_capped_at_1000(self):
        """History should not grow beyond 1000 entries."""
        server = ArgusAPIServer(port=19998)
        for i in range(1100):
            server.update_state("capital_aud", 1000.0 + i)
        history = server.get_state("pnl_history", [])
        assert len(history) <= 1000
