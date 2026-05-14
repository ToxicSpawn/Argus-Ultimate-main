"""Tests for core.connectors.solana_rpc — SolanaRPCClient."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.connectors.solana_rpc import SolanaRPCClient


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


def _make_client(url: str = "http://localhost:8899") -> SolanaRPCClient:
    return SolanaRPCClient(rpc_url=url, timeout=2, max_retries=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSolanaRPCClientInit:
    """Initialisation and fallback behaviour."""

    def test_init_with_url(self):
        client = SolanaRPCClient(rpc_url="http://localhost:8899")
        assert client.rpc_url == "http://localhost:8899"

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("SOLANA_RPC_URL", "http://env-rpc:8899")
        client = SolanaRPCClient()
        assert client.rpc_url == "http://env-rpc:8899"

    def test_init_no_url(self, monkeypatch):
        monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
        client = SolanaRPCClient()
        assert client.rpc_url is None

    def test_graceful_no_url_slot(self, monkeypatch):
        monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
        client = SolanaRPCClient()
        result = _run(client.get_slot())
        assert result is None

    def test_graceful_no_url_blockhash(self, monkeypatch):
        monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
        client = SolanaRPCClient()
        result = _run(client.get_recent_blockhash())
        assert result is None


class TestSolanaRPCMocked:
    """RPC calls with mocked aiohttp responses."""

    @patch("core.connectors.solana_rpc.aiohttp")
    def test_get_slot(self, mock_aiohttp):
        """get_slot should return integer slot."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"jsonrpc": "2.0", "id": 1, "result": 123456789})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        client = _make_client()
        result = _run(client.get_slot())
        assert result == 123456789

    @patch("core.connectors.solana_rpc.aiohttp")
    def test_get_recent_blockhash(self, mock_aiohttp):
        """get_recent_blockhash should return hash string."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "context": {"slot": 1},
                "value": {"blockhash": "7xKXYW...abc", "lastValidBlockHeight": 100},
            },
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
        result = _run(client.get_recent_blockhash())
        assert result == "7xKXYW...abc"

    @patch("core.connectors.solana_rpc.aiohttp")
    def test_get_account_info(self, mock_aiohttp):
        """get_account_info should extract value from response."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "context": {"slot": 1},
                "value": {"data": ["base64data", "base64"], "lamports": 999},
            },
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
        result = _run(client.get_account_info("someAddress"))
        assert result is not None
        assert result["lamports"] == 999

    @patch("core.connectors.solana_rpc.aiohttp")
    def test_get_token_accounts(self, mock_aiohttp):
        """get_token_accounts should return list from value."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "context": {"slot": 1},
                "value": [{"pubkey": "tok1"}, {"pubkey": "tok2"}],
            },
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
        result = _run(client.get_token_accounts("wallet"))
        assert len(result) == 2

    def test_rpc_error_returns_none(self):
        """RPC-level error should return None gracefully."""
        client = _make_client()

        async def _mock_call(*a, **kw):
            return None

        client._call = _mock_call  # type: ignore[assignment]
        result = _run(client.get_slot())
        assert result is None

    def test_empty_token_accounts_on_none(self):
        """get_token_accounts returns [] when _call returns None."""
        client = _make_client()

        async def _mock_call(*a, **kw):
            return None

        client._call = _mock_call  # type: ignore[assignment]
        result = _run(client.get_token_accounts("wallet"))
        assert result == []
