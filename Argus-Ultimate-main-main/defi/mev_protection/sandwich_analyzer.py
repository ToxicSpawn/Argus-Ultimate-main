"""Sandwich attack analysis and vulnerability estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .mempool_monitor import MempoolTransaction

logger = logging.getLogger(__name__)


@dataclass
class SandwichAnalysis:
    victim_tx_hash: str
    victim_slippage_pct: float
    optimal_front_run_size_usd: float
    optimal_back_run_size_usd: float
    estimated_gross_profit_usd: float
    estimated_gas_cost_usd: float
    estimated_net_profit_usd: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandwichAnalyzer:
    """Analyze pending swaps for sandwich profitability and risk."""

    def __init__(
        self,
        default_pool_liquidity_usd: float = 5_000_000.0,
        max_victim_slippage_pct: float = 5.0,
    ) -> None:
        self.default_pool_liquidity_usd = default_pool_liquidity_usd
        self.max_victim_slippage_pct = max_victim_slippage_pct

    def identify_vulnerable_transaction(
        self,
        transaction: MempoolTransaction,
        pool_liquidity_usd: Optional[float] = None,
        user_slippage_tolerance_pct: Optional[float] = None,
    ) -> bool:
        if not transaction.is_swap:
            return False

        liquidity = max(pool_liquidity_usd or self.default_pool_liquidity_usd, 1.0)
        estimated_slippage = self.estimate_victim_slippage(transaction.amount_in_usd, liquidity)
        slippage_limit = user_slippage_tolerance_pct or self.max_victim_slippage_pct

        vulnerable = (
            transaction.amount_in_usd >= 25_000
            and estimated_slippage >= 0.2
            and estimated_slippage <= slippage_limit
        )
        return vulnerable

    def estimate_victim_slippage(self, amount_in_usd: float, pool_liquidity_usd: float) -> float:
        liquidity = max(pool_liquidity_usd, 1.0)
        trade_ratio = amount_in_usd / liquidity
        return min(self.max_victim_slippage_pct, trade_ratio * 100 * 1.6)

    def calculate_optimal_sandwich_parameters(
        self,
        transaction: MempoolTransaction,
        pool_liquidity_usd: Optional[float] = None,
        gas_cost_usd: float = 30.0,
    ) -> Optional[SandwichAnalysis]:
        liquidity = pool_liquidity_usd or self.default_pool_liquidity_usd
        if not self.identify_vulnerable_transaction(transaction, liquidity):
            return None

        victim_slippage = self.estimate_victim_slippage(transaction.amount_in_usd, liquidity)
        front_run_size = min(transaction.amount_in_usd * 0.35, liquidity * 0.04)
        back_run_size = front_run_size
        price_impact = self._constant_product_price_impact(front_run_size, liquidity)
        gross_profit = (transaction.amount_in_usd * price_impact * 0.55) + (front_run_size * price_impact * 0.15)
        gas_cost = max(gas_cost_usd, transaction.effective_gas_gwei * 0.45)
        net_profit = gross_profit - gas_cost
        confidence = min(0.95, 0.45 + (victim_slippage / max(self.max_victim_slippage_pct, 0.1)) * 0.4)

        if net_profit <= 0:
            return None

        return SandwichAnalysis(
            victim_tx_hash=transaction.tx_hash,
            victim_slippage_pct=victim_slippage,
            optimal_front_run_size_usd=front_run_size,
            optimal_back_run_size_usd=back_run_size,
            estimated_gross_profit_usd=gross_profit,
            estimated_gas_cost_usd=gas_cost,
            estimated_net_profit_usd=net_profit,
            confidence=confidence,
            metadata={
                "pool_liquidity_usd": liquidity,
                "chain": transaction.chain,
                "dex": transaction.dex,
            },
        )

    def calculate_profit_after_gas(
        self,
        front_run_size_usd: float,
        victim_size_usd: float,
        pool_liquidity_usd: float,
        gas_cost_usd: float,
    ) -> float:
        price_impact = self._constant_product_price_impact(front_run_size_usd, pool_liquidity_usd)
        gross = victim_size_usd * price_impact * 0.55
        return gross - gas_cost_usd

    @staticmethod
    def _constant_product_price_impact(amount_in_usd: float, pool_liquidity_usd: float) -> float:
        liquidity = max(pool_liquidity_usd, 1.0)
        ratio = amount_in_usd / liquidity
        return min(0.08, ratio / (1 + ratio))
