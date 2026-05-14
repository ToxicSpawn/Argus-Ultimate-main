"""DeFi protocol adapter for quoting swaps and liquidation opportunities."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

QuoteResult = dict[str, str | float | int]


@dataclass
class DexPoolSnapshot:
    dex: str
    pair: str
    token0: str
    token1: str
    reserve0: float
    reserve1: float
    fee_pct: float
    chain: str = "ethereum"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def price(self) -> float:
        return self.reserve1 / self.reserve0 if self.reserve0 > 0 else 0.0

    @property
    def liquidity_usd(self) -> float:
        return self.reserve0 + self.reserve1


@dataclass
class LiquidationCandidate:
    protocol: str
    borrower: str
    collateral_asset: str
    debt_asset: str
    health_factor: float
    debt_value_usd: float
    liquidation_bonus_pct: float
    chain: str = "ethereum"
    timestamp: float = field(default_factory=time.time)


class GenericDexInterface(Protocol):
    async def get_quote(self, token_in: str, token_out: str, amount_in: float) -> QuoteResult:
        ...


class DeFiAdapter:
    """Best-effort integration layer for DEX and lending protocols."""

    def __init__(self, chain: str = "ethereum") -> None:
        self.chain: str = chain.lower()
        self._pools: dict[str, DexPoolSnapshot] = {}
        self._lending_positions: list[LiquidationCandidate] = []

    def update_pool(self, snapshot: DexPoolSnapshot) -> None:
        self._pools[f"{snapshot.chain}:{snapshot.dex}:{snapshot.pair}"] = snapshot

    def add_liquidation_candidate(self, candidate: LiquidationCandidate) -> None:
        self._lending_positions.append(candidate)

    async def quote_uniswap_v2(self, token_in: str, token_out: str, amount_in: float) -> QuoteResult | None:
        pool = self._find_pool("uniswap_v2", token_in, token_out)
        if pool is None:
            return None
        return self._constant_product_quote(pool, token_in, token_out, amount_in)

    async def quote_uniswap_v3(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        fee_tier: float = 0.0005,
    ) -> QuoteResult | None:
        pool = self._find_pool("uniswap_v3", token_in, token_out, fee_tier=fee_tier)
        if pool is None:
            return None
        quote = self._constant_product_quote(pool, token_in, token_out, amount_in)
        if quote is None:
            return None
        return {
            **quote,
            "concentrated_liquidity_bonus": 0.0015,
            "amount_out": float(quote["amount_out"]) * 1.0015,
        }

    async def quote_curve_pool(self, token_in: str, token_out: str, amount_in: float) -> QuoteResult | None:
        pool = self._find_pool("curve", token_in, token_out)
        if pool is None:
            return None
        quote = self._constant_product_quote(pool, token_in, token_out, amount_in)
        if quote is None:
            return None
        return {
            **quote,
            "stable_swap_efficiency": 0.002,
            "amount_out": float(quote["amount_out"]) * 1.002,
        }

    async def find_aave_liquidations(self, max_health_factor: float = 1.0) -> list[LiquidationCandidate]:
        return [candidate for candidate in self._lending_positions if candidate.protocol == "aave" and candidate.health_factor <= max_health_factor]

    async def find_compound_liquidations(self, max_health_factor: float = 1.0) -> list[LiquidationCandidate]:
        return [candidate for candidate in self._lending_positions if candidate.protocol == "compound" and candidate.health_factor <= max_health_factor]

    async def get_best_quote(self, token_in: str, token_out: str, amount_in: float) -> QuoteResult | None:
        candidates: list[QuoteResult] = []
        for dex in (self.quote_uniswap_v2, self.quote_uniswap_v3, self.quote_curve_pool):
            try:
                quote = await dex(token_in, token_out, amount_in)
            except Exception as exc:
                logger.warning("DEX quote failed: %s", exc)
                quote = None
            if quote is not None:
                candidates.append(quote)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.get("amount_out", 0.0))

    def _find_pool(
        self,
        dex: str,
        token_in: str,
        token_out: str,
        fee_tier: float | None = None,
    ) -> DexPoolSnapshot | None:
        for pool in self._pools.values():
            if pool.chain != self.chain or pool.dex != dex:
                continue
            tokens = {pool.token0.upper(), pool.token1.upper()}
            if {token_in.upper(), token_out.upper()} != tokens:
                continue
            if fee_tier is not None and abs(pool.fee_pct - fee_tier) > 1e-9:
                continue
            return pool
        return None

    def _constant_product_quote(
        self,
        pool: DexPoolSnapshot,
        token_in: str,
        token_out: str,
        amount_in: float,
    ) -> QuoteResult | None:
        if amount_in <= 0:
            return None

        if token_in.upper() == pool.token0.upper() and token_out.upper() == pool.token1.upper():
            reserve_in, reserve_out = pool.reserve0, pool.reserve1
        elif token_in.upper() == pool.token1.upper() and token_out.upper() == pool.token0.upper():
            reserve_in, reserve_out = pool.reserve1, pool.reserve0
        else:
            return None

        amount_in_after_fee = amount_in * (1 - pool.fee_pct)
        denominator = reserve_in + amount_in_after_fee
        if denominator <= 0:
            return None
        amount_out = reserve_out * amount_in_after_fee / denominator
        spot_price = reserve_out / max(reserve_in, 1e-9)
        execution_price = amount_out / max(amount_in, 1e-9)
        slippage_pct = max(0.0, (spot_price - execution_price) / max(spot_price, 1e-9) * 100)

        return {
            "dex": pool.dex,
            "pair": pool.pair,
            "amount_in": amount_in,
            "amount_out": amount_out,
            "slippage_pct": slippage_pct,
            "gas_estimate": 160000 if pool.dex.startswith("uniswap") else 220000,
            "liquidity_usd": pool.liquidity_usd,
        }
