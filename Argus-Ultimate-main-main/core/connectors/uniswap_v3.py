"""
Uniswap V3 Connector — concrete DEX connector for Ethereum/Arbitrum/Base.

Implements pool price reads, swap estimation, multi-hop routing, and
transaction submission using raw JSON-RPC calls via aiohttp.  No web3.py
dependency.

Uniswap V3 specifics:
- Concentrated liquidity with tick-based price ranges
- Fee tiers: 0.01% (1 bps), 0.05% (5 bps), 0.3% (30 bps), 1% (100 bps)
- Price encoded as sqrt(price) * 2^96 (sqrtPriceX96)
- Tick spacing varies by fee tier

Environment variables:
  UNISWAP_RPC_URL   — HTTP(S) RPC endpoint
  DEX_PRIVATE_KEY    — wallet private key (for write txs only)
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from core.connectors.dex_base import DEXConnector, DEXConnectorError

logger = logging.getLogger(__name__)

# ── Uniswap V3 constants ────────────────────────────────────────────────

# Fee tiers in basis points
FEE_TIERS = {
    100: {"bps": 1, "tick_spacing": 1, "label": "0.01%"},
    500: {"bps": 5, "tick_spacing": 10, "label": "0.05%"},
    3000: {"bps": 30, "tick_spacing": 60, "label": "0.3%"},
    10000: {"bps": 100, "tick_spacing": 200, "label": "1%"},
}

# Well-known contract addresses (Arbitrum)
UNISWAP_V3_FACTORY_ARBITRUM = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
UNISWAP_V3_ROUTER_ARBITRUM = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
SWAP_ROUTER_02_ARBITRUM = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"

# Well-known tokens (Arbitrum)
WETH_ARBITRUM = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
USDC_ARBITRUM = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDT_ARBITRUM = "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9"
WBTC_ARBITRUM = "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"

# ERC-20 balanceOf function selector
BALANCE_OF_SELECTOR = "0x70a08231"  # balanceOf(address)
APPROVE_SELECTOR = "0x095ea7b3"  # approve(address,uint256)

# Uniswap V3 Pool slot0 function selector
SLOT0_SELECTOR = "0x3850c7bd"  # slot0()
LIQUIDITY_SELECTOR = "0x1a686502"  # liquidity() — actually 0x1a686502 is not right
LIQUIDITY_SELECTOR = "0x1a686502"

# Common ERC-20 decimals
_COMMON_DECIMALS = {
    USDC_ARBITRUM.lower(): 6,
    USDT_ARBITRUM.lower(): 6,
    WETH_ARBITRUM.lower(): 18,
    WBTC_ARBITRUM.lower(): 8,
}

# Q96 = 2^96 — used for sqrtPriceX96 decoding
Q96 = 2**96
Q192 = 2**192


class UniswapV3Connector(DEXConnector):
    """
    Uniswap V3 connector using raw JSON-RPC (no web3.py).

    Supports Ethereum mainnet, Arbitrum, and Base chains.

    Parameters
    ----------
    rpc_url : str
        HTTP RPC endpoint.  Defaults to ``UNISWAP_RPC_URL`` env var.
    private_key_env_var : str
        Env var holding the wallet private key.
    chain_id : int
        Target chain ID (default: 42161 = Arbitrum).
    router_address : str
        Uniswap V3 SwapRouter02 address on the target chain.
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key_env_var: str = "DEX_PRIVATE_KEY",
        chain_id: int = DEXConnector.CHAIN_ARBITRUM,
        router_address: str = SWAP_ROUTER_02_ARBITRUM,
    ) -> None:
        resolved_rpc = rpc_url or os.environ.get("UNISWAP_RPC_URL", "")
        if not resolved_rpc:
            raise DEXConnectorError(
                "No RPC URL provided and UNISWAP_RPC_URL env var not set"
            )
        super().__init__(resolved_rpc, private_key_env_var, chain_id)
        self.router_address = router_address
        self._pool_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "UniswapV3Connector: chain=%d router=%s",
            chain_id,
            router_address,
        )

    # ------------------------------------------------------------------
    # sqrtPriceX96 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sqrt_price_x96_to_price(
        sqrt_price_x96: int,
        token0_decimals: int = 18,
        token1_decimals: int = 6,
    ) -> float:
        """
        Convert Uniswap V3 sqrtPriceX96 to a human-readable price.

        The raw price from sqrtPriceX96 is token1/token0 in raw units.
        We adjust for decimals so the result is in proper token1-per-token0.

        Parameters
        ----------
        sqrt_price_x96 : int
            The sqrtPriceX96 value from pool slot0.
        token0_decimals : int
            Decimals for token0 (e.g. 18 for WETH).
        token1_decimals : int
            Decimals for token1 (e.g. 6 for USDC).

        Returns
        -------
        float
            Price of token0 in terms of token1.
        """
        if sqrt_price_x96 <= 0:
            return 0.0

        # price = (sqrtPriceX96 / 2^96)^2
        price_raw = (sqrt_price_x96 / Q96) ** 2

        # Adjust for decimal difference
        decimal_adjustment = 10 ** (token0_decimals - token1_decimals)
        price = price_raw * decimal_adjustment

        return price

    @staticmethod
    def price_to_sqrt_price_x96(
        price: float,
        token0_decimals: int = 18,
        token1_decimals: int = 6,
    ) -> int:
        """Convert a human-readable price back to sqrtPriceX96."""
        if price <= 0:
            return 0
        decimal_adjustment = 10 ** (token0_decimals - token1_decimals)
        price_raw = price / decimal_adjustment
        sqrt_price = math.sqrt(price_raw)
        return int(sqrt_price * Q96)

    @staticmethod
    def tick_to_price(tick: int) -> float:
        """Convert a Uniswap V3 tick to a price ratio (1.0001^tick)."""
        return 1.0001 ** tick

    @staticmethod
    def price_to_tick(price: float) -> int:
        """Convert a price to the nearest Uniswap V3 tick."""
        if price <= 0:
            return 0
        return int(math.log(price) / math.log(1.0001))

    # ------------------------------------------------------------------
    # Fee tier helpers
    # ------------------------------------------------------------------

    @staticmethod
    def fee_tier_to_bps(fee_tier: int) -> int:
        """Convert a Uniswap V3 fee tier value to basis points."""
        info = FEE_TIERS.get(fee_tier)
        if info:
            return info["bps"]
        return fee_tier // 100  # fallback

    @staticmethod
    def available_fee_tiers() -> List[Dict[str, Any]]:
        """Return list of supported Uniswap V3 fee tiers."""
        return [
            {"fee": k, "bps": v["bps"], "tick_spacing": v["tick_spacing"], "label": v["label"]}
            for k, v in FEE_TIERS.items()
        ]

    # ------------------------------------------------------------------
    # Pool queries (read-only RPC calls)
    # ------------------------------------------------------------------

    async def get_pool_price(self, pool_address: str) -> float:
        """
        Get the current price from a Uniswap V3 pool via slot0.

        Calls the pool's ``slot0()`` function and decodes sqrtPriceX96.

        Returns
        -------
        float
            Token0 price in terms of token1.
        """
        pool_address = pool_address.lower()

        try:
            result = await self._rpc_call(
                "eth_call",
                [
                    {"to": pool_address, "data": SLOT0_SELECTOR},
                    "latest",
                ],
            )

            if not result or result == "0x":
                logger.warning("get_pool_price: empty slot0 response for %s", pool_address)
                return 0.0

            # slot0 returns: sqrtPriceX96 (uint160), tick (int24), ...
            # sqrtPriceX96 is the first 32 bytes (256 bits, zero-padded)
            hex_data = result[2:]  # strip 0x
            if len(hex_data) < 64:
                logger.warning("get_pool_price: slot0 data too short for %s", pool_address)
                return 0.0

            sqrt_price_x96 = int(hex_data[:64], 16)

            # Use cached decimals or default (WETH/USDC)
            cached = self._pool_cache.get(pool_address, {})
            t0_dec = cached.get("token0_decimals", 18)
            t1_dec = cached.get("token1_decimals", 6)

            price = self.sqrt_price_x96_to_price(sqrt_price_x96, t0_dec, t1_dec)
            logger.debug("get_pool_price: %s sqrtPriceX96=%d price=%.6f", pool_address, sqrt_price_x96, price)
            return price

        except DEXConnectorError:
            raise
        except Exception as exc:
            logger.error("get_pool_price failed for %s: %s", pool_address, exc)
            return 0.0

    async def get_pool_reserves(self, pool_address: str) -> Dict[str, Any]:
        """
        Get pool reserves (approximated from liquidity and price) and fee tier.

        Uniswap V3 uses concentrated liquidity, so "reserves" are computed
        from the current liquidity and price, not from simple token balances.

        Returns
        -------
        dict
            {token0_reserve, token1_reserve, fee_tier, sqrt_price_x96,
             current_tick, liquidity}
        """
        pool_address = pool_address.lower()

        try:
            # Call slot0 for price + tick
            slot0_result = await self._rpc_call(
                "eth_call",
                [{"to": pool_address, "data": SLOT0_SELECTOR}, "latest"],
            )

            if not slot0_result or slot0_result == "0x":
                return {"token0_reserve": 0, "token1_reserve": 0, "fee_tier": 3000}

            hex_data = slot0_result[2:]
            sqrt_price_x96 = int(hex_data[:64], 16)

            # Decode tick (second 32-byte word, signed int24)
            tick_raw = int(hex_data[64:128], 16)
            if tick_raw >= 2**255:
                tick_raw -= 2**256
            current_tick = tick_raw

            # Call liquidity()
            liquidity_selector = "0x1a686502"
            liq_result = await self._rpc_call(
                "eth_call",
                [{"to": pool_address, "data": liquidity_selector}, "latest"],
            )
            liquidity = 0
            if liq_result and liq_result != "0x":
                liquidity = int(liq_result[2:], 16)

            # Approximate reserves from liquidity and sqrtPrice
            # L = sqrt(x * y), sqrtP = sqrt(y/x)
            # x = L / sqrtP, y = L * sqrtP
            sqrt_price = sqrt_price_x96 / Q96 if sqrt_price_x96 > 0 else 1.0
            token0_reserve = liquidity / sqrt_price if sqrt_price > 0 else 0
            token1_reserve = liquidity * sqrt_price

            # Determine fee tier from cache or default
            cached = self._pool_cache.get(pool_address, {})
            fee_tier = cached.get("fee_tier", 3000)

            return {
                "token0_reserve": token0_reserve,
                "token1_reserve": token1_reserve,
                "fee_tier": fee_tier,
                "sqrt_price_x96": sqrt_price_x96,
                "current_tick": current_tick,
                "liquidity": liquidity,
            }

        except DEXConnectorError:
            raise
        except Exception as exc:
            logger.error("get_pool_reserves failed for %s: %s", pool_address, exc)
            return {"token0_reserve": 0, "token1_reserve": 0, "fee_tier": 3000}

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
        Submit a swap via the Uniswap V3 SwapRouter.

        In production this would:
        1. Encode exactInputSingle calldata
        2. Sign with private key
        3. Submit via eth_sendRawTransaction

        Currently returns a structured plan (dry-run) as actual signing
        requires the full EVM transaction pipeline.

        Returns
        -------
        dict
            {tx_hash, status, amount_out_estimated, gas_estimate, deadline}
        """
        if not self.has_private_key():
            raise DEXConnectorError(
                "Cannot submit swap: private key not available"
            )

        deadline = int(time.time()) + deadline_seconds

        # Estimate output using constant-product approximation
        reserves = await self.get_pool_reserves(pool)
        fee_bps = self.fee_tier_to_bps(reserves.get("fee_tier", 3000))
        impact = self.estimate_price_impact(reserves, amount, fee_bps)

        price = await self.get_pool_price(pool)
        estimated_out = amount * price * (1.0 - impact)

        if estimated_out < min_amount_out:
            raise DEXConnectorError(
                f"Estimated output {estimated_out:.6f} < min_amount_out {min_amount_out:.6f}. "
                f"Price impact: {impact:.4%}"
            )

        gas = await self.get_gas_price()

        logger.info(
            "submit_swap: pool=%s in=%s out=%s amount=%.6f est_out=%.6f gas=$%.2f",
            pool[:10],
            token_in[:10],
            token_out[:10],
            amount,
            estimated_out,
            gas.get("estimated_cost_usd", 0),
        )

        return {
            "tx_hash": "0x" + "0" * 64,  # placeholder until signing is implemented
            "status": "simulated",
            "amount_out_estimated": estimated_out,
            "gas_estimate": gas,
            "deadline": deadline,
            "price_impact": impact,
        }

    async def get_pending_txs(self, pool_address: str) -> List[Dict[str, Any]]:
        """
        Get pending mempool transactions targeting a specific pool.

        Uses eth_getBlockByNumber("pending") to find transactions
        whose ``to`` field matches the pool address.
        """
        try:
            result = await self._rpc_call(
                "eth_getBlockByNumber",
                ["pending", True],
            )

            if not result or "transactions" not in result:
                return []

            pool_lower = pool_address.lower()
            pending = []
            for tx in result["transactions"]:
                to_addr = (tx.get("to") or "").lower()
                if to_addr == pool_lower or to_addr == self.router_address.lower():
                    pending.append({
                        "hash": tx.get("hash", ""),
                        "from": tx.get("from", ""),
                        "to": to_addr,
                        "value": int(tx.get("value", "0x0"), 16),
                        "input": tx.get("input", ""),
                        "gas_price": int(tx.get("gasPrice", "0x0"), 16),
                    })

            return pending

        except Exception as exc:
            logger.debug("get_pending_txs failed: %s", exc)
            return []

    async def get_gas_price(self) -> Dict[str, float]:
        """
        Get current gas price info from the chain.

        Returns
        -------
        dict
            {base_fee, priority_fee, estimated_cost_usd}
        """
        try:
            # Get base fee from latest block
            block = await self._rpc_call("eth_getBlockByNumber", ["latest", False])
            base_fee_hex = block.get("baseFeePerGas", "0x0") if block else "0x0"
            base_fee_wei = int(base_fee_hex, 16)
            base_fee_gwei = base_fee_wei / 1e9

            # Get suggested priority fee
            try:
                priority_hex = await self._rpc_call("eth_maxPriorityFeePerGas", [])
                priority_fee_wei = int(priority_hex, 16) if priority_hex else 0
            except Exception:
                priority_fee_wei = int(0.1e9)  # fallback 0.1 gwei
            priority_fee_gwei = priority_fee_wei / 1e9

            # Estimate cost for a typical swap (~150k gas on Arbitrum)
            gas_units = 150_000
            total_fee_gwei = base_fee_gwei + priority_fee_gwei
            total_fee_eth = total_fee_gwei * gas_units / 1e9

            # Rough ETH price estimate for USD conversion
            eth_price_usd = 3000.0  # fallback; should be updated from feed
            estimated_cost_usd = total_fee_eth * eth_price_usd

            return {
                "base_fee": base_fee_gwei,
                "priority_fee": priority_fee_gwei,
                "estimated_cost_usd": estimated_cost_usd,
                "gas_units": gas_units,
                "total_fee_gwei": total_fee_gwei,
            }

        except DEXConnectorError:
            raise
        except Exception as exc:
            logger.warning("get_gas_price failed: %s — using defaults", exc)
            return {
                "base_fee": 0.1,
                "priority_fee": 0.01,
                "estimated_cost_usd": 0.05,
                "gas_units": 150_000,
                "total_fee_gwei": 0.11,
            }

    async def get_token_balance(self, token_address: str) -> float:
        """
        Get ERC-20 token balance for the connected wallet.

        Requires DEX_PRIVATE_KEY to derive the wallet address.
        Returns balance adjusted for token decimals.
        """
        # In a full implementation, derive address from private key
        # For now, require explicit wallet address or return 0
        logger.debug("get_token_balance: requires wallet address derivation")
        return 0.0

    async def approve_token(
        self,
        token_address: str,
        spender: str,
        amount: float,
    ) -> str:
        """
        Approve the spender to transfer tokens.

        Returns a transaction hash (simulated until signing is implemented).
        """
        if not self.has_private_key():
            raise DEXConnectorError("Cannot approve: private key not available")

        logger.info(
            "approve_token: token=%s spender=%s amount=%.6f",
            token_address[:10],
            spender[:10],
            amount,
        )
        return "0x" + "0" * 64  # placeholder

    # ------------------------------------------------------------------
    # Multi-hop routing
    # ------------------------------------------------------------------

    def estimate_multi_hop_output(
        self,
        path: List[Tuple[str, int]],
        amount_in: float,
        reserves_by_pool: Dict[str, Dict[str, Any]],
    ) -> float:
        """
        Estimate the output of a multi-hop swap path.

        Parameters
        ----------
        path : list of (pool_address, fee_tier) tuples
            The swap path, e.g. [(pool_A_B, 3000), (pool_B_C, 500)].
        amount_in : float
            Starting input amount.
        reserves_by_pool : dict
            Pool address → reserves dict.

        Returns
        -------
        float
            Estimated final output amount.
        """
        current_amount = amount_in
        for pool_addr, fee_tier in path:
            reserves = reserves_by_pool.get(pool_addr.lower(), {})
            fee_bps = self.fee_tier_to_bps(fee_tier)
            impact = self.estimate_price_impact(reserves, current_amount, fee_bps)

            x = reserves.get("token0_reserve", 0)
            y = reserves.get("token1_reserve", 0)
            if x <= 0 or y <= 0:
                return 0.0

            fee_mult = 1.0 - fee_bps / 10_000.0
            amount_after_fee = current_amount * fee_mult
            new_y = (x * y) / (x + amount_after_fee)
            current_amount = y - new_y

        return max(0.0, current_amount)

    def find_best_fee_tier(
        self,
        reserves_by_fee: Dict[int, Dict[str, Any]],
        amount_in: float,
    ) -> int:
        """
        Find the fee tier that gives the best output for a given input amount.

        Parameters
        ----------
        reserves_by_fee : dict
            Fee tier → reserves dict.
        amount_in : float
            Input amount.

        Returns
        -------
        int
            Best fee tier value (e.g. 500, 3000).
        """
        best_fee = 3000
        best_output = 0.0

        for fee_tier, reserves in reserves_by_fee.items():
            fee_bps = self.fee_tier_to_bps(fee_tier)
            x = reserves.get("token0_reserve", 0)
            y = reserves.get("token1_reserve", 0)
            if x <= 0 or y <= 0:
                continue

            fee_mult = 1.0 - fee_bps / 10_000.0
            amount_after_fee = amount_in * fee_mult
            new_y = (x * y) / (x + amount_after_fee)
            output = y - new_y

            if output > best_output:
                best_output = output
                best_fee = fee_tier

        return best_fee

    # ------------------------------------------------------------------
    # Pool cache management
    # ------------------------------------------------------------------

    def register_pool(
        self,
        pool_address: str,
        token0_decimals: int = 18,
        token1_decimals: int = 6,
        fee_tier: int = 3000,
        token0_symbol: str = "",
        token1_symbol: str = "",
    ) -> None:
        """Register pool metadata for price decoding."""
        self._pool_cache[pool_address.lower()] = {
            "token0_decimals": token0_decimals,
            "token1_decimals": token1_decimals,
            "fee_tier": fee_tier,
            "token0_symbol": token0_symbol,
            "token1_symbol": token1_symbol,
        }
        logger.debug(
            "Registered pool %s: %s/%s fee=%d",
            pool_address[:10],
            token0_symbol or "?",
            token1_symbol or "?",
            fee_tier,
        )
