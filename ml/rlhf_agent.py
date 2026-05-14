"""
Reinforcement Learning with Human Feedback (RLHF) for Argus
===========================================================
Extends Thompson Sampling Bandit with human feedback for:
1. Strategy selection
2. Parameter tuning
3. Continuous learning

Dependencies:
- numpy
- scipy
- stable-baselines3 (optional for advanced RL)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of human feedback."""
    GOOD = "good"
    BAD = "bad"
    NEUTRAL = "neutral"
    EXCELLENT = "excellent"
    TERRIBLE = "terrible"


@dataclass
class TradeOutcome:
    """Outcome of a trade for RLHF."""
    strategy: str
    pnl: float
    win: bool
    feedback: Optional[FeedbackType] = None
    human_score: Optional[float] = None  # 0-1 scale


@dataclass
class StrategyStats:
    """Statistics for a strategy."""
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    human_feedback: Dict[FeedbackType, int] = field(default_factory=lambda: defaultdict(int))
    avg_human_score: float = 0.5


class RLHFAgent:
    """
    RLHF Agent for Argus.
    Combines:
    1. Thompson Sampling (for exploration/exploitation)
    2. Human Feedback (for bias correction)
    3. Continuous Learning (for adaptation)
    """
    def __init__(
        self,
        strategies: List[str],
        exploration_weight: float = 0.5,
        learning_rate: float = 0.1,
        feedback_weight: float = 0.3,
    ):
        self.strategies = strategies
        self.exploration_weight = exploration_weight
        self.learning_rate = learning_rate
        self.feedback_weight = feedback_weight

        # Strategy statistics
        self.stats: Dict[str, StrategyStats] = {
            strategy: StrategyStats() for strategy in strategies
        }

        # Thompson Sampling parameters
        self.success_counts: Dict[str, float] = {s: 1.0 for s in strategies}
        self.failure_counts: Dict[str, float] = {s: 1.0 for s in strategies}

        # Human feedback
        self.feedback_history: Dict[str, deque] = {
            strategy: deque(maxlen=100) for strategy in strategies
        }

        # Current best strategy
        self.current_strategy: Optional[str] = None

        logger.info(f"RLHFAgent initialized with strategies: {strategies}")

    def select_strategy(self) -> str:
        """
        Select a strategy using Thompson Sampling + Human Feedback.
        Returns:
            Selected strategy name
        """
        # Thompson Sampling: Sample from Beta distribution
        samples = {}
        for strategy in self.strategies:
            alpha = self.success_counts[strategy]
            beta = self.failure_counts[strategy]
            samples[strategy] = np.random.beta(alpha, beta)

        # Human Feedback Adjustment
        for strategy in self.strategies:
            # Get average human score (0-1)
            feedbacks = self.feedback_history[strategy]
            if feedbacks:
                avg_score = np.mean([f.human_score for f in feedbacks if f.human_score is not None])
                # Adjust sample by human feedback
                samples[strategy] *= (1 + self.feedback_weight * (avg_score - 0.5))

        # Select strategy with highest adjusted sample
        self.current_strategy = max(samples, key=samples.get)
        return self.current_strategy

    def record_outcome(self, outcome: TradeOutcome):
        """
        Record the outcome of a trade.
        Args:
            outcome: TradeOutcome with PnL, win/loss, and optional feedback
        """
        strategy = outcome.strategy
        if strategy not in self.stats:
            logger.warning(f"Unknown strategy: {strategy}")
            return

        # Update basic stats
        self.stats[strategy].wins += 1 if outcome.win else 0
        self.stats[strategy].losses += 1 if not outcome.win else 0
        self.stats[strategy].total_pnl += outcome.pnl

        # Update Thompson Sampling counts
        if outcome.win:
            self.success_counts[strategy] += 1
        else:
            self.failure_counts[strategy] += 1

        # Update human feedback
        if outcome.feedback:
            self.stats[strategy].human_feedback[outcome.feedback] += 1
            if outcome.human_score is not None:
                self.feedback_history[strategy].append(outcome)

        # Update average human score
        feedbacks = self.feedback_history[strategy]
        if feedbacks:
            scores = [f.human_score for f in feedbacks if f.human_score is not None]
            if scores:
                self.stats[strategy].avg_human_score = float(np.mean(scores))

    def add_feedback(
        self,
        strategy: str,
        feedback: FeedbackType,
        score: Optional[float] = None,
    ):
        """
        Add human feedback for a strategy.
        Args:
            strategy: Strategy name
            feedback: FeedbackType (GOOD, BAD, etc.)
            score: Optional numeric score (0-1)
        """
        if strategy not in self.stats:
            logger.warning(f"Unknown strategy: {strategy}")
            return

        # Create a dummy outcome for feedback
        outcome = TradeOutcome(
            strategy=strategy,
            pnl=0.0,
            win=True,  # Neutral for feedback
            feedback=feedback,
            human_score=score,
        )
        self.record_outcome(outcome)

    def get_strategy_stats(self, strategy: str) -> Dict[str, Any]:
        """
        Get statistics for a strategy.
        Args:
            strategy: Strategy name
        Returns:
            Dict with strategy stats
        """
        if strategy not in self.stats:
            return {"error": f"Unknown strategy: {strategy}"}

        stats = self.stats[strategy]
        total_trades = stats.wins + stats.losses
        win_rate = stats.wins / total_trades if total_trades > 0 else 0.0

        return {
            "strategy": strategy,
            "wins": stats.wins,
            "losses": stats.losses,
            "win_rate": win_rate,
            "total_pnl": stats.total_pnl,
            "avg_pnl": stats.total_pnl / total_trades if total_trades > 0 else 0.0,
            "human_feedback": dict(stats.human_feedback),
            "avg_human_score": stats.avg_human_score,
            "success_counts": self.success_counts[strategy],
            "failure_counts": self.failure_counts[strategy],
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all strategies."""
        return {s: self.get_strategy_stats(s) for s in self.strategies}

    def get_best_strategy(self) -> Optional[str]:
        """
        Get the best strategy based on combined metrics.
        Returns:
            Best strategy name or None
        """
        if not self.strategies:
            return None

        best_strategy = None
        best_score = -np.inf

        for strategy in self.strategies:
            stats = self.get_strategy_stats(strategy)
            if "error" in stats:
                continue

            # Combined score: win rate + human feedback + PnL
            win_rate = stats["win_rate"]
            human_score = stats["avg_human_score"]
            avg_pnl = stats["avg_pnl"]

            # Normalize and combine
            score = (
                0.4 * win_rate +
                0.3 * human_score +
                0.3 * (avg_pnl / (abs(avg_pnl) + 1e-6))  # Normalized PnL
            )

            if score > best_score:
                best_score = score
                best_strategy = strategy

        return best_strategy

    def reset(self):
        """Reset all statistics."""
        for strategy in self.strategies:
            self.stats[strategy] = StrategyStats()
            self.success_counts[strategy] = 1.0
            self.failure_counts[strategy] = 1.0
            self.feedback_history[strategy].clear()
        self.current_strategy = None
        logger.info("RLHFAgent reset")
