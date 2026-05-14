"""
DEX Connector Base — abstract interface for decentralised exchange connections.

All DEX connectors (Uniswap V3, SushiSwap, Curve, etc.) inherit from this
base class.  Concrete implementations use raw RPC calls via aiohttp — no
web3.py dependency to keep the footprint lightweight.

Private keys are NEVER stored in code or config files.  They are always
read from environment variables at connection time.

All DEX features are disabled by default and opt-in via ``unified_config.yaml``.
"""

from __future__ import annotations

import abc
import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DEXConnectorError(Exception):
    """Raised when a DEX RPC call or transaction fails."""

    def __init__(self, message: str, rpc_error: Optional[Dict] = None):
        self.rpc_error = rpc_error
        super().__init__(message)


class DEXConnector(abc.ABC):
    """
    Abstract base class for decentralised exchange connections.

    Parameters
    ----------
    rpc_url : str
        HTTP(S) or WSS URL for the chain's RPC endpoint.
    private_key_env_var : str
        Name of the environment variable holding the wallet private key.
        The key is read lazily — only when a write transaction is needed.
    chain_id : int
        EVM chain ID (1 = Ethereum, 42161 = Arbitrum, 8453 = Base).
    """

    # Common chain IDs for reference
    CHAIN_ETHEREUM = 1
    CHAIN_ARBITRUM = 42161
    CHAIN_BASE = 8453
    CHAIN_POLYGON = 137
    CHAIN_OPTIMISM = 10

    def __init__(
        self,
        rpc_url: str,
        private_key_env_var: str = "DEX_PRIVATE_KEY",
        chain_id: int = CHAIN_ARBITRUM,
    ) -> None:
        if not rpc_url:
            raise ValueError("rpc_url must not be empty")
        self.rpc_url = rpc_url
        self._private_key_env_var = private_key_env_var
        self.chain_id = chain_id
        self._session: Any = None  # aiohttp.ClientSession — created lazily
        self._rpc_id: int = 0

        logger.info(
            "DEXConnector initialised: chain_id=%d rpc=%s key_env=%s",
            chain_id,
            rpc_url[:40] + "..." if len(rpc_url) > 40 else rpc_url,
            private_key_env_var,
        )

    # ------------------------------------------------------------------
    # Private key access — never stored, always read from env
    # ------------------------------------------------------------------

    def _get_private_key(self) -> str:
        """Read private key from environment variable.  Raises if missing."""
        key = os.environ.get(self._private_key_env_var)
        if not key:
            raise DEXConnectorError(
                f"Private key env var '{self._private_key_env_var}' not set. "
                "Set it before executing write transactions."
            )
        return key

    def has_private_key(self) -> bool:
        """Return True if the private key env var is set (without exposing it)."""
        return bool(os.environ.get(self._private_key_env_var))

    # ------------------------------------------------------------------
    # RPC helpers
    # ------------------------------------------------------------------

    def _next_rpc_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    async def _ensure_session(self) -> Any:
        """Lazily create an aiohttp session."""
        if self._session is None or getattr(self._session, "closed", True):
            try:
                import aiohttp
                self._session = aiohttp.ClientSession()
            except ImportError:
                raise DEXConnectorError(
                    "aiohttp is required for DEX connectors. "
                    "Install with: pip install aiohttp"
                )
        return self._session

    async def _rpc_call(self, method: str, params: List[Any]) -> Any:
        """Execute a JSON-RPC call against the configured RPC endpoint."""
        session = await self._ensure_session()
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._next_rpc_id(),
        }
        try:
            async with session.post(self.rpc_url, json=payload) as resp:
                data = await resp.json()
                if "error" in data and data["error"]:
                    raise DEXConnectorError(
                        f"RPC error: {data['error']}",
                        rpc_error=data["error"],
                    )
                return data.get("result")
        except DEXConnectorError:
            raise
        except Exception as exc:
            raise DEXConnectorError(f"RPC call failed: {exc}") from exc

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not getattr(self._session, "closed", True):
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Abstract interface — implemented by concrete connectors
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def get_pool_price(self, pool_address: str) -> float:
        """Get the current price from a liquidity pool."""
        ...

    @abc.abstractmethod
    async def get_pool_reserves(self, pool_address: str) -> Dict[str, Any]:
        """
        Get pool reserves and fee info.

        Returns
        -------
        dict
            {token0_reserve, token1_reserve, fee_tier}
        """
        ...

    @abc.abstractmethod
    async def submit_swap(
        self,
        pool: str,
        token_in: str,
        token_out: str,
        amount: float,
        min_amount_out: float,
        deadline_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Submit a swap transaction to the DEX.

        Returns
        -------
        dict
            {tx_hash, status, amount_out, gas_used}
        """
        ...

    @abc.abstractmethod
    async def get_pending_txs(self, pool_address: str) -> List[Dict[str, Any]]:
        """Get pending (mempool) transactions related to a pool."""
        ...

    @abc.abstractmethod
    async def get_gas_price(self) -> Dict[str, float]:
        """
        Get current gas price information.

        Returns
        -------
        dict
            {base_fee, priority_fee, estimated_cost_usd}
        """
        ...

    @abc.abstractmethod
    async def get_token_balance(self, token_address: str) -> float:
        """Get the balance of a specific token in the connected wallet."""
        ...

    @abc.abstractmethod
    async def approve_token(
        self,
        token_address: str,
        spender: str,
        amount: float,
    ) -> str:
        """
        Approve a spender to transfer tokens.

        Returns
        -------
        str
            Transaction hash.
        """
        ...

    # ------------------------------------------------------------------
    # Pure math helpers — no RPC needed
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_price_impact(
        reserves: Dict[str, float],
        amount_in: float,
        fee_bps: int = 30,
    ) -> float:
        """
        Estimate the price impact of a swap using the constant-product formula.

        Uses x * y = k with fee adjustment.

        Parameters
        ----------
        reserves : dict
            Must contain ``token0_reserve`` and ``token1_reserve``.
        amount_in : float
            Amount of token0 being swapped in.
        fee_bps : int
            Pool fee in basis points (e.g. 30 = 0.3%).

        Returns
        -------
        float
            Price impact as a fraction (e.g. 0.01 = 1% impact).
        """
        x = reserves.get("token0_reserve", 0.0)
        y = reserves.get("token1_reserve", 0.0)

        if x <= 0 or y <= 0 or amount_in <= 0:
            return 0.0

        fee_multiplier = 1.0 - fee_bps / 10_000.0
        amount_in_after_fee = amount_in * fee_multiplier

        # Constant product: new_y = k / (x + amount_in_after_fee)
        new_y = (x * y) / (x + amount_in_after_fee)
        amount_out = y - new_y

        # Spot price before trade
        spot_price = y / x
        # Effective price after trade
        effective_price = amount_out / amount_in if amount_in > 0 else 0

        if spot_price == 0:
            return 0.0

        impact = 1.0 - (effective_price / spot_price)
        return max(0.0, impact)

    @staticmethod
    def calculate_optimal_amount(
        reserves_a: Dict[str, float],
        reserves_b: Dict[str, float],
        fee_bps: int = 30,
    ) -> float:
        """
        Calculate the optimal arbitrage amount between two pools.

        Uses the formula for maximum profitable trade size when pool A is
        cheaper than pool B (both constant-product AMMs).

        Parameters
        ----------
        reserves_a : dict
            Reserves of the cheaper pool {token0_reserve, token1_reserve}.
        reserves_b : dict
            Reserves of the more expensive pool {token0_reserve, token1_reserve}.
        fee_bps : int
            Pool fee in basis points (applied on each leg).

        Returns
        -------
        float
            Optimal input amount in token0 units for maximum profit.
            Returns 0.0 if no profitable arbitrage exists.
        """
        xa = reserves_a.get("token0_reserve", 0.0)
        ya = reserves_a.get("token1_reserve", 0.0)
        xb = reserves_b.get("token0_reserve", 0.0)
        yb = reserves_b.get("token1_reserve", 0.0)

        if xa <= 0 or ya <= 0 or xb <= 0 or yb <= 0:
            return 0.0

        fee_mult = 1.0 - fee_bps / 10_000.0

        # Price on pool A (token1 per token0)
        price_a = ya / xa
        # Price on pool B
        price_b = yb / xb

        # Pool A must be cheaper (lower price_a means token0 is cheaper there)
        if price_a >= price_b:
            return 0.0

        # Optimal amount formula for constant-product AMM arbitrage:
        # optimal = sqrt(xa * ya * fee^2 * price_b) - xa
        # Simplified from maximising profit function
        try:
            optimal = math.sqrt(xa * ya * fee_mult * fee_mult * (price_b / price_a)) - xa
        except (ValueError, ZeroDivisionError):
            return 0.0

        return max(0.0, optimal)
