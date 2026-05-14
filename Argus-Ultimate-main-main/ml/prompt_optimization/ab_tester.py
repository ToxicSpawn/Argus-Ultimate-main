"""
A/B testing framework for prompt variations.

Based on ATLAS-GIC pattern:
- 54 prompt modifications attempted
- 16 kept (30% survival rate)
- Statistical significance testing
- Automatic winner selection
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.prompt_optimization.prompt_tracker import PromptTracker

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test."""
    experiment_id: str
    prompt_a_id: str
    prompt_b_id: str
    min_trades_per_variant: int = 30
    confidence_level: float = 0.95
    max_duration_hours: int = 24
    allocation_ratio: float = 0.5  # 50/50 split


@dataclass
class ExperimentResult:
    """Result of an A/B test."""
    experiment_id: str
    prompt_a_id: str
    prompt_b_id: str
    
    # Performance
    a_trades: int = 0
    b_trades: int = 0
    a_sharpe: float = 0.0
    b_sharpe: float = 0.0
    a_win_rate: float = 0.0
    b_win_rate: float = 0.0
    
    # Statistical significance
    p_value: float = 1.0
    is_significant: bool = False
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Winner
    winner: Optional[str] = None  # "A", "B", or None (inconclusive)
    improvement_pct: float = 0.0
    
    # Timing
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    @property
    def duration_hours(self) -> float:
        if not self.started_at or not self.ended_at:
            return 0.0
        return (self.ended_at - self.started_at).total_seconds() / 3600


class ABTester:
    """A/B testing framework for prompt variations.
    
    Usage:
        tester = ABTester(tracker)
        
        # Start experiment
        exp_id = tester.start_experiment(
            prompt_a_id="prompt_abc123",
            prompt_b_id="prompt_def456",
        )
        
        # Record outcomes
        tester.record_outcome(exp_id, variant="A", pnl=100.0, is_win=True)
        
        # Check results
        result = tester.get_result(exp_id)
        if result.is_significant:
            print(f"Winner: {result.winner}")
    """
    
    def __init__(self, tracker: PromptTracker) -> None:
        self._tracker = tracker
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._results: Dict[str, ExperimentResult] = {}
        self._variant_assignments: Dict[str, Dict[str, str]] = {}  # exp_id -> {prompt_id -> variant}
    
    def start_experiment(
        self,
        prompt_a_id: str,
        prompt_b_id: str,
        min_trades: int = 30,
        max_hours: int = 24,
    ) -> str:
        """Start a new A/B experiment."""
        experiment_id = f"exp_{len(self._experiments) + 1:04d}"
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            prompt_a_id=prompt_a_id,
            prompt_b_id=prompt_b_id,
            min_trades_per_variant=min_trades,
            max_duration_hours=max_hours,
        )
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            prompt_a_id=prompt_a_id,
            prompt_b_id=prompt_b_id,
            started_at=datetime.now(),
        )
        
        self._experiments[experiment_id] = config
        self._results[experiment_id] = result
        self._variant_assignments[experiment_id] = {}
        
        logger.info(
            "Started A/B experiment %s: A=%s vs B=%s",
            experiment_id, prompt_a_id, prompt_b_id,
        )
        
        return experiment_id
    
    def assign_variant(self, experiment_id: str, prompt_id: str) -> str:
        """Assign a variant (A or B) for a prompt in an experiment."""
        if experiment_id not in self._experiments:
            return random.choice(["A", "B"])
        
        if prompt_id in self._variant_assignments[experiment_id]:
            return self._variant_assignments[experiment_id][prompt_id]
        
        # Weighted random assignment
        variant = "A" if random.random() < 0.5 else "B"
        self._variant_assignments[experiment_id][prompt_id] = variant
        
        return variant
    
    def record_outcome(
        self,
        experiment_id: str,
        variant: str,
        pnl: float,
        is_win: bool,
        sharpe_contribution: float = 0.0,
    ) -> None:
        """Record a trade outcome for a variant."""
        if experiment_id not in self._results:
            return
        
        result = self._results[experiment_id]
        
        if variant == "A":
            result.a_trades += 1
            if is_win:
                result.a_win_rate = (result.a_win_rate * (result.a_trades - 1) + 1) / result.a_trades
            else:
                result.a_win_rate = (result.a_win_rate * (result.a_trades - 1)) / result.a_trades
            result.a_sharpe += sharpe_contribution
        else:
            result.b_trades += 1
            if is_win:
                result.b_win_rate = (result.b_win_rate * (result.b_trades - 1) + 1) / result.b_trades
            else:
                result.b_win_rate = (result.b_win_rate * (result.b_trades - 1)) / result.b_trades
            result.b_sharpe += sharpe_contribution
        
        # Check if experiment is complete
        self._check_experiment_complete(experiment_id)
    
    def _check_experiment_complete(self, experiment_id: str) -> None:
        """Check if experiment has enough data to conclude."""
        config = self._experiments[experiment_id]
        result = self._results[experiment_id]
        
        # Check minimum trades
        if result.a_trades < config.min_trades_per_variant:
            return
        if result.b_trades < config.min_trades_per_variant:
            return
        
        # Check duration
        if result.duration_hours < config.max_duration_hours:
            return
        
        # Calculate statistical significance
        self._calculate_significance(experiment_id)
    
    def _calculate_significance(self, experiment_id: str) -> None:
        """Calculate statistical significance of experiment results."""
        result = self._results[experiment_id]
        
        # Simplified t-test approximation for Sharpe ratio comparison
        # In production, use scipy.stats.ttest_ind or bootstrap
        n_a = result.a_trades
        n_b = result.b_trades
        
        if n_a < 2 or n_b < 2:
            return
        
        # Approximate p-value based on Sharpe difference
        sharpe_diff = abs(result.a_sharpe - result.b_sharpe)
        pooled_std = np.sqrt(2.0 / min(n_a, n_b))  # Simplified
        
        if pooled_std > 0:
            z_score = sharpe_diff / pooled_std
            # Approximate p-value from z-score
            result.p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
        
        # Check significance
        result.is_significant = result.p_value < (1 - 0.95)
        
        if result.is_significant:
            if result.a_sharpe > result.b_sharpe:
                result.winner = "A"
                result.improvement_pct = (
                    (result.a_sharpe - result.b_sharpe) / abs(result.b_sharpe) * 100
                    if result.b_sharpe != 0 else 0
                )
            else:
                result.winner = "B"
                result.improvement_pct = (
                    (result.b_sharpe - result.a_sharpe) / abs(result.a_sharpe) * 100
                    if result.a_sharpe != 0 else 0
                )
        
        result.ended_at = datetime.now()
        
        logger.info(
            "Experiment %s complete: winner=%s, p=%.4f, improvement=%.1f%%",
            experiment_id, result.winner, result.p_value, result.improvement_pct,
        )
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
    
    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment result."""
        return self._results.get(experiment_id)
    
    def get_active_experiments(self) -> List[str]:
        """Get IDs of active (incomplete) experiments."""
        active = []
        for exp_id, result in self._results.items():
            if result.winner is None:
                active.append(exp_id)
        return active
    
    def get_completed_experiments(self) -> List[ExperimentResult]:
        """Get all completed experiments."""
        return [r for r in self._results.values() if r.winner is not None]
    
    def get_winning_prompts(self) -> List[str]:
        """Get all prompt IDs that won experiments."""
        winners = []
        for result in self._results.values():
            if result.winner == "A":
                winners.append(result.prompt_a_id)
            elif result.winner == "B":
                winners.append(result.prompt_b_id)
        return winners
    
    def get_statistics(self) -> dict:
        """Get overall A/B testing statistics."""
        completed = self.get_completed_experiments()
        significant = [r for r in completed if r.is_significant]
        
        return {
            "total_experiments": len(self._experiments),
            "completed": len(completed),
            "active": len(self.get_active_experiments()),
            "significant_results": len(significant),
            "significance_rate": len(significant) / max(len(completed), 1),
            "avg_improvement": np.mean([r.improvement_pct for r in significant])
            if significant else 0.0,
        }
