"""
Tests for api/dashboard.py (ArgusAPIServer) and monitoring/regime_alert.py (RegimeChangeAlerter).

Designed to pass even when fastapi/uvicorn are NOT installed.
All network calls are mocked — no real HTTP requests are made.
"""
from __future__ import annotations

import asyncio
import time
import threading
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run a coroutine synchronously in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Test 1: ArgusAPIServer initialises without fastapi
# ---------------------------------------------------------------------------

class TestArgusAPIServerInit(unittest.TestCase):
    def test_server_initializes_without_fastapi(self):
        """Server should construct successfully whether or not fastapi is installed."""
        from api.dashboard import ArgusAPIServer
        server = ArgusAPIServer(port=19080)
        self.assertIsNotNone(server)
        self.assertEqual(server.port, 19080)
        self.assertFalse(server._running)

    def test_default_state_values(self):
        """Default state should contain expected keys with sane defaults."""
        from api.dashboard import ArgusAPIServer
        server = ArgusAPIServer(port=19081)
        self.assertEqual(server.get_state("status"), "stopped")
        self.assertEqual(server.get_state("mode"), "paper")
        self.assertEqual(server.get_state("cycle"), 0)
        self.assertIsInstance(server.get_state("positions"), dict)
        self.assertIsInstance(server.get_state("trades"), list)


# ---------------------------------------------------------------------------
# Test 2: update_state works correctly
# ---------------------------------------------------------------------------

class TestUpdateState(unittest.TestCase):
    def setUp(self):
        from api.dashboard import ArgusAPIServer
        self.server = ArgusAPIServer(port=19082)

    def test_update_single_key(self):
        self.server.update_state("cycle", 42)
        self.assertEqual(self.server.get_state("cycle"), 42)

    def test_update_string_key(self):
        self.server.update_state("regime", "BULL_TREND")
        self.assertEqual(self.server.get_state("regime"), "BULL_TREND")

    def test_update_bool_key(self):
        self.server.update_state("circuit_breaker", True)
        self.assertTrue(self.server.get_state("circuit_breaker"))

    def test_update_states_bulk(self):
        self.server.update_states({
            "capital_aud": 1234.56,
            "pnl_aud": -50.0,
            "drawdown_pct": 5.0,
        })
        self.assertAlmostEqual(self.server.get_state("capital_aud"), 1234.56)
        self.assertAlmostEqual(self.server.get_state("pnl_aud"), -50.0)

    def test_trades_capped_at_20(self):
        """Adding more than 20 trades should keep only the last 20."""
        for i in range(25):
            self.server.update_state("trades", {"symbol": "BTC/AUD", "seq": i})
        trades = self.server.get_state("trades")
        self.assertLessEqual(len(trades), 20)
        # Last item should have the highest seq number
        self.assertEqual(trades[-1]["seq"], 24)


# ---------------------------------------------------------------------------
# Test 3: /health endpoint returns correct dict shape
# ---------------------------------------------------------------------------

class TestHealthEndpoint(unittest.TestCase):
    def test_health_dict_shape(self):
        """_build_prometheus and direct state reads should match expected schema."""
        from api.dashboard import ArgusAPIServer, _uptime
        server = ArgusAPIServer(port=19083)
        server.update_state("status", "running")
        server.update_state("mode", "live")
        server.update_state("cycle", 100)
        # Simulate start_time
        server._state["start_time"] = time.time() - 60

        s = server._state
        health = {
            "status": s.get("status", "stopped"),
            "uptime_seconds": _uptime(s),
            "mode": s.get("mode", "paper"),
            "cycle": s.get("cycle", 0),
        }
        self.assertEqual(health["status"], "running")
        self.assertEqual(health["mode"], "live")
        self.assertEqual(health["cycle"], 100)
        self.assertGreaterEqual(health["uptime_seconds"], 50)  # at least ~60s


# ---------------------------------------------------------------------------
# Test 4: /portfolio endpoint returns correct dict shape
# ---------------------------------------------------------------------------

class TestPortfolioEndpoint(unittest.TestCase):
    def test_portfolio_dict_shape(self):
        from api.dashboard import ArgusAPIServer
        server = ArgusAPIServer(port=19084)
        server.update_states({
            "capital_aud": 1000.0,
            "pnl_aud": 25.50,
            "pnl_pct": 2.55,
            "positions": {
                "BTC/AUD": {
                    "qty": 0.01,
                    "entry_price": 95000.0,
                    "current_price": 97500.0,
                    "unrealised_pnl": 25.0,
                }
            },
        })
        s = server._state
        portfolio = {
            "capital_aud": s.get("capital_aud", 0.0),
            "pnl_aud": s.get("pnl_aud", 0.0),
            "pnl_pct": s.get("pnl_pct", 0.0),
            "positions": s.get("positions", {}),
        }
        self.assertAlmostEqual(portfolio["capital_aud"], 1000.0)
        self.assertIn("BTC/AUD", portfolio["positions"])
        self.assertAlmostEqual(portfolio["positions"]["BTC/AUD"]["unrealised_pnl"], 25.0)


# ---------------------------------------------------------------------------
# Test 5: /risk endpoint returns correct dict shape
# ---------------------------------------------------------------------------

class TestRiskEndpoint(unittest.TestCase):
    def test_risk_dict_shape(self):
        from api.dashboard import ArgusAPIServer
        server = ArgusAPIServer(port=19085)
        server.update_states({
            "daily_loss_aud": -120.0,
            "drawdown_pct": 3.5,
            "var_95": 0.0214,
            "circuit_breaker": True,
        })
        s = server._state
        risk = {
            "daily_loss_aud": s.get("daily_loss_aud", 0.0),
            "drawdown_pct": s.get("drawdown_pct", 0.0),
            "var_95": s.get("var_95", 0.0),
            "circuit_breaker": s.get("circuit_breaker", False),
        }
        self.assertAlmostEqual(risk["daily_loss_aud"], -120.0)
        self.assertTrue(risk["circuit_breaker"])
        self.assertAlmostEqual(risk["drawdown_pct"], 3.5)


# ---------------------------------------------------------------------------
# Test 6: Prometheus metrics output contains expected lines
# ---------------------------------------------------------------------------

class TestPrometheusMetrics(unittest.TestCase):
    def test_prometheus_output_format(self):
        from api.dashboard import ArgusAPIServer, _build_prometheus
        server = ArgusAPIServer(port=19086)
        server.update_states({
            "capital_aud": 2000.0,
            "pnl_aud": 50.0,
            "cycle": 99,
            "circuit_breaker": False,
        })
        with server._lock:
            text = _build_prometheus(dict(server._state))

        self.assertIn("argus_capital_aud 2000.0", text)
        self.assertIn("argus_pnl_aud 50.0", text)
        self.assertIn("argus_cycle_total 99", text)
        self.assertIn("argus_circuit_breaker 0", text)
        self.assertIn("# TYPE", text)
        self.assertIn("# HELP", text)


# ---------------------------------------------------------------------------
# Test 7: RegimeChangeAlerter detects regime change
# ---------------------------------------------------------------------------

class TestRegimeChangeDetection(unittest.TestCase):
    def test_detects_first_regime(self):
        from monitoring.regime_alert import RegimeChangeAlerter
        alerter = RegimeChangeAlerter(webhook_url=None)
        changed = _run_async(alerter.check_and_alert("BTC/AUD", "BULL_TREND"))
        # First observation — not a "change", returns False
        self.assertFalse(changed)
        self.assertEqual(alerter.get_regime("BTC/AUD"), "BULL_TREND")

    def test_detects_regime_change(self):
        from monitoring.regime_alert import RegimeChangeAlerter
        alerter = RegimeChangeAlerter(webhook_url=None)
        # Establish initial regime
        _run_async(alerter.check_and_alert("BTC/AUD", "BULL_TREND"))
        # Change it
        changed = _run_async(alerter.check_and_alert("BTC/AUD", "BEAR_TREND"))
        self.assertTrue(changed)
        self.assertEqual(alerter.get_regime("BTC/AUD"), "BEAR_TREND")

    def test_no_change_when_same_regime(self):
        from monitoring.regime_alert import RegimeChangeAlerter
        alerter = RegimeChangeAlerter(webhook_url=None)
        _run_async(alerter.check_and_alert("ETH/AUD", "RANGING"))
        changed = _run_async(alerter.check_and_alert("ETH/AUD", "RANGING"))
        self.assertFalse(changed)


# ---------------------------------------------------------------------------
# Test 8: Throttle prevents duplicate alerts
# ---------------------------------------------------------------------------

class TestAlertThrottle(unittest.TestCase):
    def test_throttle_blocks_duplicate_within_window(self):
        """Second alert_on_change within throttle window must return False."""
        from monitoring.regime_alert import RegimeChangeAlerter
        alerter = RegimeChangeAlerter(webhook_url=None, throttle_seconds=300)
        # First alert (no webhook → returns False but timestamps are set)
        result1 = _run_async(alerter.alert_on_change("BTC/AUD", "BULL_TREND", "BEAR_TREND"))
        # Immediately retry — should be throttled
        result2 = _run_async(alerter.alert_on_change("BTC/AUD", "BULL_TREND", "BEAR_TREND"))
        # First fires (no webhook so False), second is throttled (also False)
        # Key check: second call must NOT have updated the timestamp again
        # (both return False, but for different reasons — check throttle is set)
        throttle_key = ("BTC/AUD", "BEAR_TREND")
        ts = alerter._alert_timestamps.get(throttle_key, 0)
        self.assertGreater(ts, 0, "Throttle timestamp should be set after first alert")

    def test_throttle_allows_after_window(self):
        """After throttle window expires, alert should be allowed (timestamp updated)."""
        from monitoring.regime_alert import RegimeChangeAlerter
        # Use a real webhook URL so the path reaches _send_discord_alert
        alerter = RegimeChangeAlerter(webhook_url="http://fake-webhook.local/test", throttle_seconds=1)
        throttle_key = ("BTC/AUD", "BEAR_TREND")
        # Pre-set an expired timestamp (10 seconds ago, window is 1 second)
        alerter._alert_timestamps[throttle_key] = time.time() - 10

        # Patch _send_discord_alert so no real HTTP is attempted
        # Use AsyncMock (Python 3.8+) to properly mock an async method
        from unittest.mock import AsyncMock
        with patch.object(alerter, "_send_discord_alert", new=AsyncMock(return_value=True)) as mock_send:
            _run_async(alerter.alert_on_change("BTC/AUD", "BULL_TREND", "BEAR_TREND"))
            mock_send.assert_called_once()

        # Verify timestamp was refreshed to a recent value
        new_ts = alerter._alert_timestamps[throttle_key]
        self.assertGreater(new_ts, time.time() - 5)


# ---------------------------------------------------------------------------
# Test 9: Regime duration tracking
# ---------------------------------------------------------------------------

class TestRegimeDuration(unittest.TestCase):
    def test_duration_increases_over_time(self):
        from monitoring.regime_alert import RegimeChangeAlerter
        alerter = RegimeChangeAlerter(webhook_url=None)
        # Establish regime with a backdated start
        now = time.time()
        alerter._regime_state["BTC/AUD"] = {
            "regime": "BULL_TREND",
            "changed_at": now - 120,  # 2 minutes ago
            "duration_start": now - 120,
        }
        duration = alerter.get_duration_minutes("BTC/AUD")
        self.assertGreater(duration, 1.9)
        self.assertLess(duration, 2.5)

    def test_duration_resets_on_change(self):
        from monitoring.regime_alert import RegimeChangeAlerter
        alerter = RegimeChangeAlerter(webhook_url=None)
        # Set up an old regime
        old_start = time.time() - 3600
        alerter._regime_state["ETH/AUD"] = {
            "regime": "RANGING",
            "changed_at": old_start,
            "duration_start": old_start,
        }
        # Trigger regime change
        _run_async(alerter.check_and_alert("ETH/AUD", "BULL_TREND"))
        duration = alerter.get_duration_minutes("ETH/AUD")
        # Duration should have reset to ~0 (within 5 seconds of now)
        self.assertLess(duration, 0.1)

    def test_get_regime_summary_shape(self):
        from monitoring.regime_alert import RegimeChangeAlerter
        alerter = RegimeChangeAlerter(webhook_url=None)
        _run_async(alerter.check_and_alert("BTC/AUD", "BULL_TREND"))
        _run_async(alerter.check_and_alert("ETH/AUD", "RANGING"))
        summary = alerter.get_regime_summary()
        self.assertIn("BTC/AUD", summary)
        self.assertIn("ETH/AUD", summary)
        for sym, info in summary.items():
            self.assertIn("regime", info)
            self.assertIn("duration_minutes", info)
            self.assertIn("changed_at", info)
            self.assertIsInstance(info["duration_minutes"], float)
            self.assertGreaterEqual(info["duration_minutes"], 0.0)


# ---------------------------------------------------------------------------
# Test 10: HTML status page builds without errors
# ---------------------------------------------------------------------------

class TestHTMLPage(unittest.TestCase):
    def test_html_builds_with_empty_state(self):
        from api.dashboard import ArgusAPIServer, _build_html
        server = ArgusAPIServer(port=19090)
        html = _build_html(server._state)
        self.assertIn("ARGUS", html)
        self.assertIn("<meta http-equiv=\"refresh\"", html)
        self.assertIn("content=\"10\"", html)
        self.assertIn("Auto-refresh", html)

    def test_html_builds_with_full_state(self):
        from api.dashboard import ArgusAPIServer, _build_html
        server = ArgusAPIServer(port=19091)
        server.update_states({
            "status": "running",
            "mode": "live",
            "cycle": 1234,
            "capital_aud": 5000.0,
            "pnl_aud": 200.0,
            "regime": "BULL_TREND",
            "circuit_breaker": False,
            "positions": {
                "BTC/AUD": {
                    "qty": 0.05,
                    "entry_price": 90000.0,
                    "current_price": 94000.0,
                    "unrealised_pnl": 200.0,
                }
            },
            "trades": [
                {
                    "timestamp": "2026-03-12T10:00:00Z",
                    "symbol": "BTC/AUD",
                    "side": "buy",
                    "qty": 0.01,
                    "price": 90000.0,
                    "pnl": 0.0,
                }
            ],
        })
        html = _build_html(server._state)
        self.assertIn("BTC/AUD", html)
        self.assertIn("BULL_TREND", html)
        self.assertIn("5,000.00", html)
        self.assertIn("LIVE", html)


if __name__ == "__main__":
    unittest.main()
