"""
ARGUS Ethereum JSON-RPC client — thin async wrapper with no web3.py dependency.

Uses raw ``aiohttp`` over JSON-RPC 2.0 to interact with any EVM-compatible node.
All methods degrade gracefully when no RPC URL is configured (returns None).

Usage::

    from core.connectors.ethereum_rpc import EthereumRPCClient
    client = EthereumRPCClient("https://eth-mainnet.g.alchemy.com/v2/<KEY>")
    block = await client.get_block_number()
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

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


class EthereumRPCClient:
    """Lightweight async Ethereum JSON-RPC 2.0 client.

    Parameters
    ----------
    rpc_url : str, optional
        HTTP(S) endpoint of an Ethereum node.  Falls back to the
        ``ETHEREUM_RPC_URL`` environment variable when not supplied.
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
        self.rpc_url: Optional[str] = rpc_url or os.environ.get("ETHEREUM_RPC_URL")
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
            logger.debug("EthereumRPCClient: no RPC URL configured, returning None")
            return None
        if not _AIOHTTP_AVAILABLE:
            logger.warning("EthereumRPCClient: aiohttp not installed")
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
                                "EthereumRPC %s returned HTTP %d (attempt %d/%d)",
                                method, resp.status, attempt, self.max_retries,
                            )
                            last_error = Exception(f"HTTP {resp.status}")
                        else:
                            data = await resp.json()
                            if "error" in data:
                                logger.warning(
                                    "EthereumRPC %s error: %s",
                                    method, data["error"],
                                )
                                return None
                            return data.get("result")
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "EthereumRPC %s failed (attempt %d/%d): %s",
                    method, attempt, self.max_retries, exc,
                )

            if attempt < self.max_retries:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))

        logger.warning(
            "EthereumRPC %s exhausted retries: %s", method, last_error,
        )
        return None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def get_block_number(self) -> Optional[int]:
        """Return the latest block number, or None on failure."""
        result = await self._call("eth_blockNumber")
        if result is None:
            return None
        try:
            return int(result, 16)
        except (ValueError, TypeError):
            return None

    async def get_balance(self, address: str) -> Optional[float]:
        """Return balance in ETH for *address*, or None on failure."""
        result = await self._call("eth_getBalance", [address, "latest"])
        if result is None:
            return None
        try:
            wei = int(result, 16)
            return wei / 1e18
        except (ValueError, TypeError):
            return None

    async def call_contract(self, to: str, data: str) -> Optional[str]:
        """Execute ``eth_call`` and return the hex-encoded result."""
        result = await self._call("eth_call", [{"to": to, "data": data}, "latest"])
        return result

    async def get_gas_price(self) -> Optional[int]:
        """Return current gas price in wei, or None on failure."""
        result = await self._call("eth_gasPrice")
        if result is None:
            return None
        try:
            return int(result, 16)
        except (ValueError, TypeError):
            return None

    async def get_token_balance(self, token_contract: str, wallet: str) -> Optional[int]:
        """Return ERC-20 token balance (raw units) for *wallet*.

        Calls ``balanceOf(address)`` on the token contract.
        """
        # ERC-20 balanceOf(address) selector: 0x70a08231
        padded_wallet = wallet.lower().replace("0x", "").zfill(64)
        data = "0x70a08231" + padded_wallet
        result = await self.call_contract(token_contract, data)
        if result is None:
            return None
        try:
            return int(result, 16)
        except (ValueError, TypeError):
            return None
