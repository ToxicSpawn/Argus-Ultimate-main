"""Push 71 — Tests: BybitSigner, AsyncRateLimiter, BybitV5Client (stub),
LiveOrderManager (stub), PositionSyncManager. 26 tests.
All mock-based — no real API calls.
"""
from __future__ import annotations
import asyncio
import sys
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# BybitSigner (5)
# ---------------------------------------------------------------------------

class TestBybitSigner:
    def _signer(self):
        from core.live.signing import BybitSigner
        return BybitSigner(api_key="testkey", api_secret="testsecret")

    def test_sign_post_returns_headers_and_body(self):
        s = self._signer()
        headers, body = s.sign_post({"symbol": "BTCUSDT", "qty": "0.1"})
        assert "X-BAPI-SIGN" in headers
        assert "X-BAPI-API-KEY" in headers
        assert "BTCUSDT" in body

    def test_sign_get_returns_headers(self):
        s = self._signer()
        headers, params = s.sign_get({"symbol": "BTCUSDT"})
        assert "X-BAPI-SIGN" in headers
        assert params["symbol"] == "BTCUSDT"

    def test_signature_is_hex_string(self):
        s = self._signer()
        headers, _ = s.sign_post({"test": 1})
        sig = headers["X-BAPI-SIGN"]
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_different_payloads_different_sigs(self):
        s = self._signer()
        h1, _ = s.sign_post({"a": 1}, timestamp=1000000)
        h2, _ = s.sign_post({"b": 2}, timestamp=1000000)
        assert h1["X-BAPI-SIGN"] != h2["X-BAPI-SIGN"]

    def test_validate_signature_roundtrip(self):
        s = self._signer()
        ts = 1700000000000
        payload = '{"symbol":"BTCUSDT"}'
        headers, _ = s.sign_post({"symbol": "BTCUSDT"}, timestamp=ts)
        sig = headers["X-BAPI-SIGN"]
        # Recompute manually
        import hmac, hashlib
        expected = hmac.new(
            b"testsecret",
            f"{ts}testkey5000{payload}".encode(),
            hashlib.sha256
        ).hexdigest()
        assert sig == expected


# ---------------------------------------------------------------------------
# AsyncRateLimiter (5)
# ---------------------------------------------------------------------------

class TestAsyncRateLimiter:
    def test_instantiates_with_defaults(self):
        from core.live.rate_limiter import AsyncRateLimiter
        rl = AsyncRateLimiter()
        assert rl.available_tokens("order") > 0

    def test_try_acquire_succeeds_when_tokens_available(self):
        from core.live.rate_limiter import AsyncRateLimiter
        rl = AsyncRateLimiter()
        assert rl.try_acquire("order") is True

    def test_try_acquire_fails_when_exhausted(self):
        from core.live.rate_limiter import AsyncRateLimiter, BucketConfig
        rl = AsyncRateLimiter({"test": BucketConfig(max_tokens=1, refill_rate=0.01)})
        rl.try_acquire("test")   # consume the one token
        assert rl.try_acquire("test") is False

    def test_wait_completes_async(self):
        from core.live.rate_limiter import AsyncRateLimiter
        rl = AsyncRateLimiter()
        async def run():
            await rl.wait("order")
        asyncio.get_event_loop().run_until_complete(run())

    def test_add_custom_category(self):
        from core.live.rate_limiter import AsyncRateLimiter
        rl = AsyncRateLimiter()
        rl.add_category("custom", max_tokens=5, refill_rate=5)
        assert rl.available_tokens("custom") > 0


# ---------------------------------------------------------------------------
# BybitV5Client stub (4)
# ---------------------------------------------------------------------------

class TestBybitV5ClientStub:
    """All tests use stub mode (no aiohttp installed required)."""
    def _client(self):
        from core.live.bybit_client import BybitV5Client
        return BybitV5Client(api_key="k", api_secret="s", testnet=True)

    def test_instantiates(self):
        c = self._client()
        assert c is not None

    def test_place_order_stub(self):
        from core.live.bybit_client import BybitV5Client, OrderRequest
        c = self._client()
        req = OrderRequest(symbol="BTCUSDT", side="Buy",
                            order_type="Market", qty="0.001")
        async def run():
            return await c.place_order(req)
        resp = asyncio.get_event_loop().run_until_complete(run())
        assert resp.get("retCode") == 0

    def test_get_position_stub(self):
        from core.live.bybit_client import BybitV5Client
        c = self._client()
        async def run():
            return await c.get_position("BTCUSDT")
        resp = asyncio.get_event_loop().run_until_complete(run())
        assert "result" in resp

    def test_order_request_to_dict(self):
        from core.live.bybit_client import OrderRequest
        req = OrderRequest(symbol="ETHUSDT", side="Sell",
                            order_type="Limit", qty="1.0", price="3000")
        d = req.to_dict()
        assert d["symbol"] == "ETHUSDT"
        assert d["price"] == "3000"


# ---------------------------------------------------------------------------
# LiveOrderManager (6)
# ---------------------------------------------------------------------------

class TestLiveOrderManager:
    def _manager(self):
        from core.live.bybit_client import BybitV5Client
        from core.live.live_order_manager import LiveOrderManager
        client = BybitV5Client(api_key="k", api_secret="s", testnet=True)
        return LiveOrderManager(client=client, poll_interval_secs=0.05)

    def test_submit_order_creates_live_order(self):
        mgr = self._manager()
        async def run():
            return await mgr.submit_order("BTCUSDT", "Buy", "Market", 0.001)
        order = asyncio.get_event_loop().run_until_complete(run())
        assert order is not None
        assert order.symbol == "BTCUSDT"

    def test_submit_sets_state_open_or_filled(self):
        from core.live.live_order_manager import LiveOrderState
        mgr = self._manager()
        async def run():
            return await mgr.submit_order("BTCUSDT", "Buy", "Market", 0.001)
        order = asyncio.get_event_loop().run_until_complete(run())
        assert order.state in (LiveOrderState.OPEN, LiveOrderState.FILLED,
                                LiveOrderState.REJECTED)

    def test_cancel_nonexistent_returns_false(self):
        mgr = self._manager()
        async def run():
            return await mgr.cancel_order("nonexistent")
        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is False

    def test_cancel_all_returns_count(self):
        mgr = self._manager()
        async def run():
            return await mgr.cancel_all()
        count = asyncio.get_event_loop().run_until_complete(run())
        assert count == 0  # no open orders yet

    def test_start_stop_lifecycle(self):
        mgr = self._manager()
        async def run():
            await mgr.start()
            await asyncio.sleep(0.1)
            await mgr.stop()
        asyncio.get_event_loop().run_until_complete(run())

    def test_on_fill_callback_wired(self):
        from core.live.bybit_client import BybitV5Client
        from core.live.live_order_manager import LiveOrderManager, LiveOrderState
        fills = []
        client = BybitV5Client(api_key="k", api_secret="s", testnet=True)
        mgr = LiveOrderManager(client=client, on_fill=fills.append)
        order_data = {"orderStatus": "Filled", "cumExecQty": "0.001",
                       "avgPrice": "50000", "cumExecFee": "0.5"}
        from core.live.live_order_manager import LiveOrder
        o = LiveOrder(local_id="x", exchange_id="y", symbol="BTCUSDT",
                       side="Buy", order_type="Market", qty=0.001, price=None)
        mgr._apply_exchange_status(o, order_data)
        assert len(fills) == 1
        assert o.state == LiveOrderState.FILLED


# ---------------------------------------------------------------------------
# PositionSyncManager (6)
# ---------------------------------------------------------------------------

class TestPositionSyncManager:
    def _sync_mgr(self, internal_fn=None):
        from core.live.bybit_client import BybitV5Client
        from core.live.position_sync import PositionSyncManager
        client = BybitV5Client(api_key="k", api_secret="s", testnet=True)
        return PositionSyncManager(
            client=client,
            internal_positions_fn=internal_fn or (lambda: {}),
            sync_interval_secs=0.05,
        )

    def test_instantiates(self):
        s = self._sync_mgr()
        assert s is not None

    def test_initial_status_unknown(self):
        from core.live.position_sync import SyncStatus
        s = self._sync_mgr()
        assert s.status == SyncStatus.UNKNOWN

    def test_sync_once_returns_synced_when_no_positions(self):
        from core.live.position_sync import SyncStatus
        s = self._sync_mgr()
        async def run():
            return await s.sync_once()
        status = asyncio.get_event_loop().run_until_complete(run())
        assert status == SyncStatus.SYNCED

    def test_divergence_detected_side_mismatch(self):
        from core.live.position_sync import PositionSyncManager, ExchangePosition, SyncStatus
        from core.live.bybit_client import BybitV5Client
        from unittest.mock import MagicMock
        internal_pos = MagicMock()
        internal_pos.qty = 0.1
        internal_pos.side = "long"
        s = self._sync_mgr(internal_fn=lambda: {"BTCUSDT": internal_pos})
        exchange = {"BTCUSDT": ExchangePosition(
            symbol="BTCUSDT", side="Sell",
            size=0.1, avg_price=50000,
            unrealised_pnl=0, leverage=1,
        )}
        divs = s._reconcile({"BTCUSDT": internal_pos}, exchange)
        assert len(divs) == 1
        assert divs[0].is_side_mismatch

    def test_no_divergence_when_flat(self):
        s = self._sync_mgr()
        divs = s._reconcile({}, {})
        assert divs == []

    def test_start_stop_lifecycle(self):
        s = self._sync_mgr()
        async def run():
            await s.start()
            await asyncio.sleep(0.15)
            await s.stop()
        asyncio.get_event_loop().run_until_complete(run())
        assert s.sync_count >= 1
