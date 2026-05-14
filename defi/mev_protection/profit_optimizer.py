"""Profit optimization for MEV execution and private bundles."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional

from .mev_detector import MEVOpportunity

logger = logging.getLogger(__name__)


@dataclass
class BundlePerformance:
    bundle_hash: str
    opportunity_type: str
    estimated_profit_usd: float
    realized_profit_usd: float
    gas_cost_usd: float
    included: bool
    chain: str = "ethereum"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProfitOptimizer:
    """Optimize gas, ordering, and risk-adjusted return for MEV flows."""

    def __init__(self) -> None:
        self._history: List[BundlePerformance] = []

    def optimize_gas_price(
        self,
        expected_profit_usd: float,
        base_fee_gwei: float,
        competition_level: float = 0.5,
        chain: str = "ethereum",
    ) -> Dict[str, float]:
        competition = min(max(competition_level, 0.0), 1.0)
        chain_multiplier = {"ethereum": 1.0, "arbitrum": 0.35, "optimism": 0.3}.get(chain, 0.5)
        tip_budget = min(expected_profit_usd * 0.15, expected_profit_usd / 25)
        max_priority = max(0.05, tip_budget * chain_multiplier * (0.5 + competition))
        max_fee = base_fee_gwei + max_priority * 2
        return {
            "base_fee_gwei": base_fee_gwei,
            "max_priority_fee_gwei": round(max_priority, 4),
            "max_fee_per_gas_gwei": round(max_fee, 4),
        }

    def optimize_bundle_ordering(self, opportunities: List[MEVOpportunity]) -> List[MEVOpportunity]:
        return sorted(
            opportunities,
            key=lambda item: (item.profit_estimate * item.confidence) - (item.risk * 100),
            reverse=True,
        )

    def risk_adjusted_profit(self, opportunity: MEVOpportunity, inclusion_probability: Optional[float] = None) -> float:
        probability = inclusion_probability if inclusion_probability is not None else opportunity.confidence
        probability = min(max(probability, 0.0), 1.0)
        return opportunity.profit_estimate * probability * (1 - opportunity.risk)

    def track_performance(self, performance: BundlePerformance) -> None:
        self._history.append(performance)
        self._history = self._history[-1000:]

    def historical_performance(self, opportunity_type: Optional[str] = None, chain: Optional[str] = None) -> Dict[str, float]:
        records = self._history
        if opportunity_type is not None:
            records = [item for item in records if item.opportunity_type == opportunity_type]
        if chain is not None:
            records = [item for item in records if item.chain == chain]
        if not records:
            return {
                "count": 0,
                "avg_realized_profit_usd": 0.0,
                "inclusion_rate": 0.0,
                "avg_gas_cost_usd": 0.0,
            }
        return {
            "count": float(len(records)),
            "avg_realized_profit_usd": mean(item.realized_profit_usd for item in records),
            "inclusion_rate": mean(1.0 if item.included else 0.0 for item in records),
            "avg_gas_cost_usd": mean(item.gas_cost_usd for item in records),
        }
