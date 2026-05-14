"""
Strategy Bridge — converts existing strategy signals into target proposals.

Existing strategies produce TradingSignal objects with action/confidence/strength.
This bridge converts them into TargetProposal objects that the sealed runtime
can accept or reject through promotion gates.

Strategies CANNOT execute orders. They can only propose targets.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetProposal:
    """A target proposal from a strategy — must pass promotion to become live."""
    strategy: str
    symbol: str
    target_weight_pct: float    # desired portfolio weight (0.0 to 1.0)
    direction: str              # "long", "short", "flat"
    confidence: float           # 0.0 to 1.0
    time_horizon_hours: float   # expected hold time
    execution_hint: str         # "market", "limit", "twap", "patient"
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class PromotionDecision:
    """Result of a promotion gate check."""
    proposal: TargetProposal
    approved: bool
    reason: str
    adjusted_weight_pct: float  # may be reduced by promotion gate


class StrategyBridge:
    """
    Converts existing TradingSignal objects into governed TargetProposals.

    Promotion rules:
    - Strategy must be in the approved list
    - Confidence must exceed minimum threshold
    - Strategy must not be disabled by PerformanceScorecard
    - Weight is capped by constitution family caps
    """

    def __init__(
        self,
        approved_strategies: Optional[List[str]] = None,
        min_confidence: float = 0.40,
        max_weight_pct: float = 0.15,
        family_caps: Optional[Dict[str, float]] = None,
        disabled_strategies: Optional[set] = None,
    ):
        self._approved = set(approved_strategies or [
            "mean_reversion", "momentum", "funding_rate_harvester",
        ])
        self._min_confidence = min_confidence
        self._max_weight = max_weight_pct
        self._family_caps = family_caps or {}
        self._disabled = disabled_strategies or set()
        self._total_proposals = 0
        self._total_approved = 0
        self._total_rejected = 0

    def convert_signal(self, signal: Any) -> Optional[TargetProposal]:
        """Convert a TradingSignal into a TargetProposal."""
        try:
            symbol = str(getattr(signal, "symbol", "") or "")
            action = str(getattr(signal, "action", "") or "").upper()
            confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
            strength = float(getattr(signal, "strength", 0.0) or 0.0)
            strategy = str(getattr(signal, "strategy", getattr(signal, "source_strategy", "unknown")) or "unknown")
            reasoning = str(getattr(signal, "reasoning", "") or "")

            if action not in ("BUY", "SELL"):
                return None

            direction = "long" if action == "BUY" else "short"
            target_weight = min(strength * confidence * self._max_weight, self._max_weight)

            return TargetProposal(
                strategy=strategy,
                symbol=symbol,
                target_weight_pct=target_weight,
                direction=direction,
                confidence=confidence,
                time_horizon_hours=1.0,
                execution_hint="limit" if confidence > 0.7 else "market",
                reasoning=reasoning,
            )
        except Exception as e:
            logger.debug("StrategyBridge: failed to convert signal: %s", e)
            return None

    def promote(self, proposal: TargetProposal) -> PromotionDecision:
        """Check if a proposal passes promotion gates."""
        self._total_proposals += 1

        # Gate 1: Strategy must be approved
        if proposal.strategy not in self._approved:
            self._total_rejected += 1
            return PromotionDecision(
                proposal=proposal, approved=False,
                reason=f"strategy '{proposal.strategy}' not in approved list",
                adjusted_weight_pct=0.0,
            )

        # Gate 2: Strategy must not be disabled
        if proposal.strategy in self._disabled:
            self._total_rejected += 1
            return PromotionDecision(
                proposal=proposal, approved=False,
                reason=f"strategy '{proposal.strategy}' disabled by scorecard",
                adjusted_weight_pct=0.0,
            )

        # Gate 3: Minimum confidence
        if proposal.confidence < self._min_confidence:
            self._total_rejected += 1
            return PromotionDecision(
                proposal=proposal, approved=False,
                reason=f"confidence {proposal.confidence:.2f} < {self._min_confidence:.2f}",
                adjusted_weight_pct=0.0,
            )

        # Gate 4: Family cap
        family_cap = self._family_caps.get(proposal.strategy, self._max_weight)
        adjusted_weight = min(proposal.target_weight_pct, family_cap)

        self._total_approved += 1
        return PromotionDecision(
            proposal=proposal, approved=True,
            reason="all gates passed",
            adjusted_weight_pct=adjusted_weight,
        )

    def update_disabled(self, disabled: set) -> None:
        """Update the set of disabled strategies (from PerformanceScorecard)."""
        self._disabled = disabled

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_proposals": self._total_proposals,
            "approved": self._total_approved,
            "rejected": self._total_rejected,
            "approved_strategies": sorted(self._approved),
            "disabled_strategies": sorted(self._disabled),
        }
