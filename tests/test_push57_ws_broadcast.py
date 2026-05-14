"""Push 57 — WebSocket broadcast + live dashboard feed: 25 tests."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# WsMessage tests (6)
# ---------------------------------------------------------------------------
from core.broadcast.ws_message import WsMessage, MessageType


class TestWsMessage:
    def test_to_json_contains_type(self):
        msg = WsMessage(type=MessageType.HEARTBEAT, data={})
        j = json.loads(msg.to_json())
        assert j["type"] == "heartbeat"

    def test_to_json_contains_ts(self):
        msg = WsMessage(type=MessageType.TICK, data={})
        j = json.loads(msg.to_json())
        assert "ts" in j

    def test_from_json_roundtrip(self):
        msg = WsMessage(type=MessageType.FILL, data={"price": 65000.0})
        msg2 = WsMessage.from_json(msg.to_json())
        assert msg2.type == MessageType.FILL
        assert msg2.data["price"] == 65000.0

    def test_heartbeat_factory(self):
        hb = WsMessage.heartbeat()
        assert hb.type == MessageType.HEARTBEAT
        assert hb.data["ping"] == "pong"

    def test_alert_factory(self):
        a = WsMessage.alert("warning", "drawdown exceeded")
        assert a.type == MessageType.ALERT
        assert a.data["level"] == "warning"

    def test_tick_factory(self):
        t = WsMessage.tick("BTCUSDT", 65000.0, 64990.0, 65010.0)
        assert t.type == MessageType.TICK
        assert t.data["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# WsHub tests (9)
# ---------------------------------------------------------------------------
from core.broadcast.ws_hub import WsHub


class TestWsHub:
    def _hub(self) -> WsHub:
        return WsHub()

    def _mock_ws(self, fail: bool = False) -> MagicMock:
        ws = MagicMock()
        if fail:
            ws.send_text = AsyncMock(side_effect=Exception("disconnected"))
        else:
            ws.send_text = AsyncMock()
        return ws

    def test_register_increases_count(self):
        hub = self._hub()
        ws = self._mock_ws()
        asyncio.get_event_loop().run_until_complete(hub.register(ws))
        assert hub.client_count == 1

    def test_unregister_decreases_count(self):
        hub = self._hub()
        ws = self._mock_ws()
        asyncio.get_event_loop().run_until_complete(hub.register(ws))
        asyncio.get_event_loop().run_until_complete(hub.unregister(ws))
        assert hub.client_count == 0

    def test_broadcast_sends_to_all(self):
        hub = self._hub()
        ws1, ws2 = self._mock_ws(), self._mock_ws()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hub.register(ws1))
        loop.run_until_complete(hub.register(ws2))
        msg = WsMessage(type=MessageType.HEARTBEAT, data={})
        sent = loop.run_until_complete(hub.broadcast(msg))
        assert sent == 2

    def test_broadcast_removes_dead_client(self):
        hub = self._hub()
        ws_dead = self._mock_ws(fail=True)
        ws_ok = self._mock_ws()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hub.register(ws_dead))
        loop.run_until_complete(hub.register(ws_ok))
        loop.run_until_complete(hub.broadcast(WsMessage.heartbeat()))
        assert hub.client_count == 1  # dead removed

    def test_broadcast_empty_returns_zero(self):
        hub = self._hub()
        sent = asyncio.get_event_loop().run_until_complete(
            hub.broadcast(WsMessage.heartbeat())
        )
        assert sent == 0

    def test_broadcast_increments_counter(self):
        hub = self._hub()
        ws = self._mock_ws()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hub.register(ws))
        loop.run_until_complete(hub.broadcast(WsMessage.heartbeat()))
        assert hub.broadcast_count == 1

    def test_send_to_single_client(self):
        hub = self._hub()
        ws = self._mock_ws()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hub.register(ws))
        result = loop.run_until_complete(hub.send_to(ws, WsMessage.heartbeat()))
        assert result is True

    def test_status_dict_keys(self):
        hub = self._hub()
        s = hub.status()
        assert "clients" in s and "broadcasts" in s and "errors" in s

    def test_broadcast_json_convenience(self):
        hub = self._hub()
        ws = self._mock_ws()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(hub.register(ws))
        sent = loop.run_until_complete(
            hub.broadcast_json({"equity": 10000.0}, MessageType.PNL_SNAPSHOT)
        )
        assert sent == 1


# ---------------------------------------------------------------------------
# DashboardFeed tests (10)
# ---------------------------------------------------------------------------
from core.broadcast.dashboard_feed import DashboardFeed


class TestDashboardFeed:
    def _hub_with_client(self):
        hub = WsHub()
        ws = MagicMock()
        ws.send_text = AsyncMock()
        asyncio.get_event_loop().run_until_complete(hub.register(ws))
        return hub, ws

    def test_tick_broadcast(self):
        hub, ws = self._hub_with_client()
        feed = DashboardFeed(hub=hub)
        asyncio.get_event_loop().run_until_complete(
            feed.on_tick("BTCUSDT", 65000.0, 64990.0, 65010.0)
        )
        assert feed.tick_count == 1
        ws.send_text.assert_called_once()

    def test_multiple_ticks_increment_counter(self):
        hub, _ = self._hub_with_client()
        feed = DashboardFeed(hub=hub)
        loop = asyncio.get_event_loop()
        for _ in range(5):
            loop.run_until_complete(feed.on_tick("ETHUSDT", 3000.0))
        assert feed.tick_count == 5

    def test_pnl_snapshot_empty_without_tracker(self):
        hub = WsHub()
        feed = DashboardFeed(hub=hub)
        assert feed._pnl_snapshot() == {}

    def test_risk_snapshot_empty_without_manager(self):
        hub = WsHub()
        feed = DashboardFeed(hub=hub)
        assert feed._risk_snapshot() == {}

    def test_full_snapshot_has_keys(self):
        hub = WsHub()
        feed = DashboardFeed(hub=hub)
        snap = feed.full_snapshot()
        assert "pnl" in snap and "risk" in snap and "hub" in snap and "ts" in snap

    def test_on_fill_broadcasts_three_messages(self):
        hub, ws = self._hub_with_client()
        feed = DashboardFeed(hub=hub)
        fill = MagicMock()
        fill.order_id = "abc"
        fill.symbol = "BTCUSDT"
        fill.side = MagicMock(value="buy")
        fill.price = 65000.0
        fill.qty = 1.0
        fill.fee = 1.3
        fill.venue = "sim"
        order = MagicMock()
        asyncio.get_event_loop().run_until_complete(feed._on_fill(order, fill))
        assert ws.send_text.call_count == 3  # FILL + PNL + RISK

    def test_attach_registers_callback(self):
        hub = WsHub()
        engine = MagicMock()
        engine.add_fill_callback = MagicMock()
        feed = DashboardFeed(hub=hub, engine=engine)
        feed.attach()
        engine.add_fill_callback.assert_called_once_with(feed._on_fill)

    def test_pnl_snapshot_with_tracker(self):
        from core.pnl.pnl_tracker import PnLTracker
        pnl = PnLTracker()
        hub = WsHub()
        feed = DashboardFeed(hub=hub, pnl=pnl)
        snap = feed._pnl_snapshot()
        assert "n_trades" in snap

    def test_risk_snapshot_with_manager(self):
        from core.risk.risk_manager import RiskManager
        from core.risk.risk_config import RiskConfig
        rm = RiskManager(RiskConfig())
        hub = WsHub()
        feed = DashboardFeed(hub=hub, risk=rm)
        snap = feed._risk_snapshot()
        assert "halted" in snap

    def test_fill_count_increments(self):
        hub, ws = self._hub_with_client()
        feed = DashboardFeed(hub=hub)
        fill = MagicMock()
        fill.order_id = "x"
        fill.symbol = "X"
        fill.side = MagicMock(value="buy")
        fill.price = 100.0
        fill.qty = 1.0
        fill.fee = 0.0
        fill.venue = "sim"
        asyncio.get_event_loop().run_until_complete(feed._on_fill(MagicMock(), fill))
        assert feed.fill_count == 1
