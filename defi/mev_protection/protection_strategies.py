"""Strategies for protecting user flow against adverse MEV."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .mempool_monitor import MempoolTransaction
from .mev_detector import MEVOpportunity, OpportunityType

logger = logging.getLogger(__name__)


@dataclass
class ProtectionStrategy:
    name: str
    description: str
    chain: str
    priority: int
    estimated_cost_usd: float
    risk_reduction: float
    parameters: Dict[str, Any] = field(default_factory=dict)


class ProtectionStrategies:
    """Recommend and construct MEV protection flows."""

    def __init__(self, flashbots_relay_url: str = "https://relay.flashbots.net") -> None:
        self.flashbots_relay_url = flashbots_relay_url

    def recommend(
        self,
        transaction: MempoolTransaction,
        opportunities: Optional[List[MEVOpportunity]] = None,
    ) -> List[ProtectionStrategy]:
        opportunities = opportunities or []
        strategies: List[ProtectionStrategy] = []

        if any(item.opportunity_type in {OpportunityType.FRONT_RUN, OpportunityType.SANDWICH} for item in opportunities):
            strategies.append(self.private_transaction_submission(transaction.chain))
            strategies.append(self.slippage_optimization(transaction))

        if any(item.opportunity_type == OpportunityType.SANDWICH for item in opportunities):
            strategies.append(self.commit_reveal_scheme(transaction))

        if transaction.chain in {"arbitrum", "optimism"}:
            strategies.append(self.time_delayed_execution(transaction.chain, delay_seconds=4))

        if not strategies:
            strategies.append(self.slippage_optimization(transaction))

        return sorted(strategies, key=lambda item: item.priority)

    def private_transaction_submission(self, chain: str) -> ProtectionStrategy:
        relay = self.flashbots_relay_url if chain == "ethereum" else f"private-{chain}-relay"
        return ProtectionStrategy(
            name="private_submission",
            description="Send the transaction through a private relay to avoid public mempool exposure.",
            chain=chain,
            priority=1,
            estimated_cost_usd=5.0,
            risk_reduction=0.75,
            parameters={"relay": relay},
        )

    def commit_reveal_scheme(self, transaction: MempoolTransaction) -> ProtectionStrategy:
        commit_hash = hashlib.sha256(f"{transaction.tx_hash}:{transaction.nonce}".encode("utf-8")).hexdigest()
        return ProtectionStrategy(
            name="commit_reveal",
            description="Split intent publication from execution to reduce actionable information leakage.",
            chain=transaction.chain,
            priority=2,
            estimated_cost_usd=12.0,
            risk_reduction=0.65,
            parameters={"commit_hash": commit_hash},
        )

    def time_delayed_execution(self, chain: str, delay_seconds: int = 12) -> ProtectionStrategy:
        return ProtectionStrategy(
            name="time_delay",
            description="Delay execution to avoid deterministic ordering around volatile pending flow.",
            chain=chain,
            priority=3,
            estimated_cost_usd=1.0,
            risk_reduction=0.3,
            parameters={"delay_seconds": delay_seconds},
        )

    def slippage_optimization(self, transaction: MempoolTransaction, baseline_slippage_pct: float = 1.0) -> ProtectionStrategy:
        optimized = max(0.1, min(baseline_slippage_pct, transaction.amount_in_usd / 250_000))
        return ProtectionStrategy(
            name="slippage_optimization",
            description="Tighten slippage bounds to reduce sandwichable price movement.",
            chain=transaction.chain,
            priority=4,
            estimated_cost_usd=0.0,
            risk_reduction=0.45,
            parameters={"slippage_pct": round(optimized, 3)},
        )

    def build_private_submission_payload(self, raw_tx: Dict[str, Any], chain: str) -> Dict[str, Any]:
        return {
            "chain": chain,
            "relay": self.flashbots_relay_url if chain == "ethereum" else f"private-{chain}-relay",
            "tx": raw_tx,
            "submitted_at": time.time(),
        }

    def build_commit_reveal_payload(self, intent: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        serialized = repr(sorted(intent.items()))
        commit_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return {
            "commit": {"hash": commit_hash, "intent_hint": intent.get("action", "swap")},
            "reveal": dict(intent),
        }
