#!/usr/bin/env python3
"""
Portfolio Governor - S+ Tier
Dynamically allocates capital across strategies based on performance and risk metrics.

Key Features:
- Uses performance stability as primary signal.
- Applies regime-aware penalties to reduce allocation in RANGE/CHAOS/ILLIQUID.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class ScoreResult:
    """Result of strategy scoring"""

    score: float
    reason: str


class PortfolioGovernor:
    """
    Portfolio Governor - S+ Tier

    Dynamically allocates capital across strategies based on performance and risk metrics.

    Key Features:
    - Uses performance stability as primary signal.
    - Applies regime-aware penalties to reduce allocation in RANGE/CHAOS/ILLIQUID.
    """

    def score(self, pnls: List[float], current_regime: Optional[str] = None) -> ScoreResult:
        """Score a strategy based on its performance history"""
        if len(pnls) < 2:
            return ScoreResult(0.0, "NO_DATA")

        m = sum(pnls) / len(pnls)
        v = sum((x - m) ** 2 for x in pnls) / (len(pnls) - 1)
        sd = math.sqrt(max(v, 0.0))
        base = float(m / (sd + 1e-9))

        reg = (current_regime or "unknown").lower()
        penalty = 0.0

        if reg == "range":
            penalty = 0.3
        elif reg == "chaos":
            penalty = 0.5
        elif reg == "illiquid":
            penalty = 0.7

        final_score = base * (1.0 - penalty)

        if final_score > 2.0:
            reason = "EXCELLENT"
        elif final_score > 1.0:
            reason = "GOOD"
        elif final_score > 0.0:
            reason = "OK"
        elif final_score > -1.0:
            reason = "POOR"
        else:
            reason = "TERRIBLE"

        return ScoreResult(final_score, reason)

    def allocate(self, strategy_scores: Dict[str, ScoreResult], total_capital: float) -> Dict[str, float]:
        """
        Allocate capital across strategies based on their scores.

        Args:
            strategy_scores: Dictionary mapping strategy names to their scores
            total_capital: Total capital available for allocation

        Returns:
            Dictionary mapping strategy names to allocated capital
        """
        if not strategy_scores:
            return {}

        # Convert scores to allocation weights
        weights = {}
        total_weight = 0.0

        for strategy, score_result in strategy_scores.items():
            # Only allocate to strategies with positive scores
            if score_result.score > 0:
                # Use exponential weighting to favor better strategies
                weight = math.exp(score_result.score * 0.5)
                weights[strategy] = weight
                total_weight += weight

        # Normalize weights and calculate allocations
        allocations = {}
        for strategy, weight in weights.items():
            allocation = (weight / total_weight) * total_capital
            allocations[strategy] = allocation

        return allocations

    def rebalance_signal(
        self,
        current_allocations: Dict[str, float],
        new_allocations: Dict[str, float],
        threshold: float = 0.05,
    ) -> bool:
        """
        Determine if a rebalance is needed based on allocation drift.

        Args:
            current_allocations: Current capital allocations
            new_allocations: Proposed new allocations
            threshold: Minimum drift percentage to trigger rebalance

        Returns:
            True if rebalance is needed, False otherwise
        """
        total_current = sum(current_allocations.values())
        total_new = sum(new_allocations.values())

        if abs(total_current - total_new) > threshold * total_current:
            return True

        # Check individual strategy drift
        all_strategies = set(current_allocations.keys()) | set(new_allocations.keys())

        for strategy in all_strategies:
            current = current_allocations.get(strategy, 0)
            new = new_allocations.get(strategy, 0)

            if total_current > 0:
                current_pct = current / total_current
                new_pct = new / total_new if total_new > 0 else 0

                if abs(current_pct - new_pct) > threshold:
                    return True

        return False

    def emergency_stop(self, strategy_scores: Dict[str, ScoreResult], emergency_threshold: float = -2.0) -> List[str]:
        """
        Identify strategies that should be emergency stopped.

        Args:
            strategy_scores: Current strategy scores
            emergency_threshold: Score threshold below which strategies are stopped

        Returns:
            List of strategy names to emergency stop
        """
        to_stop = []
        for strategy, score_result in strategy_scores.items():
            if score_result.score < emergency_threshold:
                to_stop.append(strategy)

        return to_stop
