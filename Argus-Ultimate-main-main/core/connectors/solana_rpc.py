"""
ARGUS Solana JSON-RPC client — lightweight HTTP wrapper with no heavy SDK.

Uses raw ``aiohttp`` over JSON-RPC 2.0 for the Solana RPC API.
All methods degrade gracefully when no RPC URL is configured (returns None).

Usage::

    from core.connectors.solana_rpc import SolanaRPCClient
    client = SolanaRPCClient("https://api.mainnet-beta.solana.com")
    slot = await client.get_slot()
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    _AIOHTTP_AVAILABLE = False

_DEFAULT_TIMEOUT = 10
_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5


class SolanaRPCClient:
    """Lightweight async Solana JSON-RPC 2.0 client.

    Parameters
    ----------
    rpc_url : str, optional
        HTTP(S) endpoint of a Solana RPC node.  Falls back to the
        ``SOLANA_RPC_URL`` environment variable when not supplied.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Number of retry attempts with exponential back-off.
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self.rpc_url: Optional[str] = rpc_url or os.environ.get("SOLANA_RPC_URL")
        self.timeout = timeout
        self.max_retries = max_retries
        self._request_id = 0

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _call(self, method: str, params: list | None = None) -> Optional[Any]:
        """Send a JSON-RPC 2.0 request with retry + exponential back-off."""
        if not self.rpc_url:
            logger.debug("SolanaRPCClient: no RPC URL configured, returning None")
            return None
        if not _AIOHTTP_AVAILABLE:
            logger.warning("SolanaRPCClient: aiohttp not installed")
            return None

        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": self._next_id(),
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.rpc_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(
                                "SolanaRPC %s returned HTTP %d (attempt %d/%d)",
                                method, resp.status, attempt, self.max_retries,
                            )
                            last_error = Exception(f"HTTP {resp.status}")
                        else:
                            data = await resp.json()
                            if "error" in data:
                                logger.warning(
                                    "SolanaRPC %s error: %s",
                                    method, data["error"],
                                )
                                return None
                            return data.get("result")
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "SolanaRPC %s failed (attempt %d/%d): %s",
                    method, attempt, self.max_retries, exc,
                )

            if attempt < self.max_retries:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))

        logger.warning(
            "SolanaRPC %s exhausted retries: %s", method, last_error,
        )
        return None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def get_account_info(self, address: str) -> Optional[dict]:
        """Return account info for *address*, or None on failure."""
        result = await self._call("getAccountInfo", [address, {"encoding": "jsonParsed"}])
        if result is None:
            return None
        # result is {"context": ..., "value": {...}} — extract value
        if isinstance(result, dict):
            return result.get("value")
        return result

    async def get_token_accounts(self, wallet: str) -> List[dict]:
        """Return SPL token accounts owned by *wallet*."""
        result = await self._call(
            "getTokenAccountsByOwner",
            [
                wallet,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"},
            ],
        )
        if result is None:
            return []
        if isinstance(result, dict):
            return result.get("value", [])
        return []

    async def get_slot(self) -> Optional[int]:
        """Return the current slot number, or None on failure."""
        result = await self._call("getSlot")
        if result is None:
            return None
        try:
            return int(result)
        except (ValueError, TypeError):
            return None

    async def get_recent_blockhash(self) -> Optional[str]:
        """Return a recent blockhash string, or None on failure."""
        result = await self._call("getLatestBlockhash")
        if result is None:
            return None
        if isinstance(result, dict):
            value = result.get("value", {})
            if isinstance(value, dict):
                return value.get("blockhash")
        return None
