"""MEV opportunity detection driven by mempool and DeFi state."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .arbitrage_scanner import ArbitrageRoute, ArbitrageScanner
from .mempool_monitor import MempoolTransaction
from .sandwich_analyzer import SandwichAnalyzer

logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    FRONT_RUN = "front_run"
    BACK_RUN = "back_run"
    SANDWICH = "sandwich"
    LIQUIDATION = "liquidation"
    ARBITRAGE = "arbitrage"


@dataclass
class MEVOpportunity:
    opportunity_type: OpportunityType
    transaction_hash: str
    chain: str
    target_protocol: str
    profit_estimate: float
    risk: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MEVDetector:
    """Detect front-run, back-run, sandwich, arbitrage, and liquidation setups."""

    def __init__(
        self,
        sandwich_analyzer: Optional[SandwichAnalyzer] = None,
        arbitrage_scanner: Optional[ArbitrageScanner] = None,
        min_profit_usd: float = 25.0,
    ) -> None:
        self.sandwich_analyzer = sandwich_analyzer or SandwichAnalyzer()
        self.arbitrage_scanner = arbitrage_scanner or ArbitrageScanner(min_profit_usd=min_profit_usd)
        self.min_profit_usd = min_profit_usd

    def detect_opportunities(
        self,
        transaction: MempoolTransaction,
        recent_transactions: Optional[List[MempoolTransaction]] = None,
        liquidation_context: Optional[List[Dict[str, Any]]] = None,
    ) -> List[MEVOpportunity]:
        opportunities: List[MEVOpportunity] = []
        recent_transactions = recent_transactions or []

        front_run = self.detect_front_running(transaction)
        if front_run is not None:
            opportunities.append(front_run)

        sandwich = self.detect_sandwich(transaction)
        if sandwich is not None:
            opportunities.append(sandwich)

        back_runs = self.detect_back_running(transaction)
        opportunities.extend(back_runs)

        liquidation = self.detect_liquidation_opportunity(liquidation_context or [])
        if liquidation is not None:
            opportunities.append(liquidation)

        opportunities.extend(self.detect_related_flow_patterns(transaction, recent_transactions))
        return sorted(opportunities, key=lambda item: item.profit_estimate, reverse=True)

    def detect_front_running(self, transaction: MempoolTransaction) -> Optional[MEVOpportunity]:
        if not transaction.is_swap or not transaction.is_large_swap:
            return None
        if transaction.amount_in_usd < 100_000:
            return None

        profit = transaction.amount_in_usd * 0.0015
        if profit < self.min_profit_usd:
            return None
        return MEVOpportunity(
            opportunity_type=OpportunityType.FRONT_RUN,
            transaction_hash=transaction.tx_hash,
            chain=transaction.chain,
            target_protocol=transaction.dex or "unknown",
            profit_estimate=profit,
            risk=self._risk_score(transaction, competition_multiplier=1.25),
            confidence=0.68,
            details={"reason": "large pending swap with likely short-term price impact"},
        )

    def detect_back_running(self, transaction: MempoolTransaction) -> List[MEVOpportunity]:
        if not transaction.is_swap:
            return []
        profit = max(0.0, transaction.amount_in_usd * 0.0009)
        if profit < self.min_profit_usd:
            return []
        return [
            MEVOpportunity(
                opportunity_type=OpportunityType.BACK_RUN,
                transaction_hash=transaction.tx_hash,
                chain=transaction.chain,
                target_protocol=transaction.dex or "unknown",
                profit_estimate=profit,
                risk=self._risk_score(transaction, competition_multiplier=1.1),
                confidence=0.61,
                details={"reason": "post-trade arbitrage or inventory rebalance opportunity"},
            )
        ]

    def detect_sandwich(self, transaction: MempoolTransaction) -> Optional[MEVOpportunity]:
        analysis = self.sandwich_analyzer.calculate_optimal_sandwich_parameters(transaction)
        if analysis is None or analysis.estimated_net_profit_usd < self.min_profit_usd:
            return None
        return MEVOpportunity(
            opportunity_type=OpportunityType.SANDWICH,
            transaction_hash=transaction.tx_hash,
            chain=transaction.chain,
            target_protocol=transaction.dex or "unknown",
            profit_estimate=analysis.estimated_net_profit_usd,
            risk=max(0.25, 1.0 - analysis.confidence),
            confidence=analysis.confidence,
            details={
                "victim_slippage_pct": analysis.victim_slippage_pct,
                "front_run_size_usd": analysis.optimal_front_run_size_usd,
            },
        )

    def detect_liquidation_opportunity(self, liquidation_context: List[Dict[str, Any]]) -> Optional[MEVOpportunity]:
        best: Optional[MEVOpportunity] = None
        for candidate in liquidation_context:
            health_factor = float(candidate.get("health_factor", 2.0))
            debt_value = float(candidate.get("debt_value_usd", 0.0))
            bonus = float(candidate.get("liquidation_bonus_pct", 0.0))
            if health_factor > 1.0 or debt_value <= 0:
                continue
            profit = debt_value * (bonus / 100)
            if profit < self.min_profit_usd:
                continue
            current = MEVOpportunity(
                opportunity_type=OpportunityType.LIQUIDATION,
                transaction_hash=str(candidate.get("borrower", "liquidation")),
                chain=str(candidate.get("chain", "ethereum")),
                target_protocol=str(candidate.get("protocol", "aave")),
                profit_estimate=profit,
                risk=min(0.8, max(0.1, health_factor)),
                confidence=0.72,
                details=dict(candidate),
            )
            if best is None or current.profit_estimate > best.profit_estimate:
                best = current
        return best

    def detect_related_flow_patterns(
        self,
        transaction: MempoolTransaction,
        recent_transactions: List[MempoolTransaction],
    ) -> List[MEVOpportunity]:
        if not recent_transactions or not transaction.is_swap:
            return []

        same_pair = [
            tx for tx in recent_transactions
            if tx.token_in == transaction.token_in and tx.token_out == transaction.token_out and tx.tx_hash != transaction.tx_hash
        ]
        if len(same_pair) < 2:
            return []

        cumulative_size = sum(tx.amount_in_usd for tx in same_pair[:5]) + transaction.amount_in_usd
        profit = cumulative_size * 0.0005
        if profit < self.min_profit_usd:
            return []

        return [
            MEVOpportunity(
                opportunity_type=OpportunityType.ARBITRAGE,
                transaction_hash=transaction.tx_hash,
                chain=transaction.chain,
                target_protocol=transaction.dex or "unknown",
                profit_estimate=profit,
                risk=0.35,
                confidence=0.57,
                details={"reason": "clustered order flow indicates transient pricing imbalance"},
            )
        ]

    def arbitrage_opportunities(self, pair: str) -> List[MEVOpportunity]:
        routes: List[ArbitrageRoute] = self.arbitrage_scanner.detect_price_discrepancies(pair)
        opportunities = []
        for route in routes:
            opportunities.append(
                MEVOpportunity(
                    opportunity_type=OpportunityType.ARBITRAGE,
                    transaction_hash=route.route_id,
                    chain=route.chain,
                    target_protocol=route.dexes[0] if route.dexes else "unknown",
                    profit_estimate=route.expected_profit_usd,
                    risk=max(0.1, 1 - route.confidence),
                    confidence=route.confidence,
                    details={"dexes": route.dexes, "hops": route.hops},
                )
            )
        return opportunities

    @staticmethod
    def _risk_score(transaction: MempoolTransaction, competition_multiplier: float) -> float:
        gas_pressure = min(1.0, transaction.effective_gas_gwei / 150)
        size_pressure = min(1.0, transaction.amount_in_usd / 1_000_000)
        return min(0.95, 0.2 + gas_pressure * 0.3 + size_pressure * 0.2 * competition_multiplier)
