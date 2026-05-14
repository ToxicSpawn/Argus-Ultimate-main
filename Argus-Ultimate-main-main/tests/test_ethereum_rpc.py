"""Tests for core.connectors.ethereum_rpc — EthereumRPCClient."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.connectors.ethereum_rpc import EthereumRPCClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _make_client(url: str = "http://localhost:8545") -> EthereumRPCClient:
    return EthereumRPCClient(rpc_url=url, timeout=2, max_retries=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEthereumRPCClientInit:
    """Initialisation and fallback behaviour."""

    def test_init_with_url(self):
        client = EthereumRPCClient(rpc_url="http://localhost:8545")
        assert client.rpc_url == "http://localhost:8545"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("ETHEREUM_RPC_URL", "http://env-rpc:8545")
        client = EthereumRPCClient()
        assert client.rpc_url == "http://env-rpc:8545"

    def test_init_no_url(self, monkeypatch):
        monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)
        client = EthereumRPCClient()
        assert client.rpc_url is None

    def test_graceful_no_url(self, monkeypatch):
        monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)
        client = EthereumRPCClient()
        result = _run(client.get_block_number())
        assert result is None

    def test_graceful_no_url_balance(self, monkeypatch):
        monkeypatch.delenv("ETHEREUM_RPC_URL", raising=False)
        client = EthereumRPCClient()
        result = _run(client.get_balance("0x0000000000000000000000000000000000000000"))
        assert result is None


class TestEthereumRPCMocked:
    """RPC calls with mocked aiohttp responses."""

    @patch("core.connectors.ethereum_rpc.aiohttp")
    def test_get_block_number(self, mock_aiohttp):
        """get_block_number should parse hex block number."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"jsonrpc": "2.0", "id": 1, "result": "0x10"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        client = _make_client()
        result = _run(client.get_block_number())
        assert result == 16  # 0x10 = 16

    @patch("core.connectors.ethereum_rpc.aiohttp")
    def test_get_balance(self, mock_aiohttp):
        """get_balance should convert wei to ETH."""
        wei_1eth = hex(10 ** 18)
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"jsonrpc": "2.0", "id": 1, "result": wei_1eth})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        client = _make_client()
        result = _run(client.get_balance("0xabc"))
        assert result == pytest.approx(1.0)

    @patch("core.connectors.ethereum_rpc.aiohttp")
    def test_get_gas_price(self, mock_aiohttp):
        """get_gas_price should return int wei."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"jsonrpc": "2.0", "id": 1, "result": "0x3b9aca00"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        client = _make_client()
        result = _run(client.get_gas_price())
        assert result == 1_000_000_000  # 1 gwei

    @patch("core.connectors.ethereum_rpc.aiohttp")
    def test_call_contract(self, mock_aiohttp):
        """call_contract should return raw hex result."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "jsonrpc": "2.0", "id": 1,
            "result": "0x00000000000000000000000000000000000000000000000000000000000000ff",
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        client = _make_client()
        result = _run(client.call_contract("0xdead", "0x70a08231"))
        assert result is not None
        assert result.startswith("0x")

    @patch("core.connectors.ethereum_rpc.aiohttp")
    def test_get_token_balance(self, mock_aiohttp):
        """get_token_balance should parse hex return from balanceOf."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "jsonrpc": "2.0", "id": 1,
            "result": "0x" + hex(1000000)[2:].zfill(64),
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        client = _make_client()
        result = _run(client.get_token_balance("0xtoken", "0xwallet"))
        assert result == 1_000_000

    def test_rpc_error_returns_none(self):
        """RPC-level error should return None gracefully."""
        client = _make_client()

        async def _mock_call(*a, **kw):
            return None

        client._call = _mock_call  # type: ignore[assignment]
        result = _run(client.get_block_number())
        assert result is None
