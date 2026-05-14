"""
Feedback loop for self-improving trading agents.

Based on ATLAS-GIC:
- Agents improve their own prompts from market feedback
- Prompt modifications: 54 attempted, 16 kept (30% survival)
- Deployment return: +22% in 173 days
- Best pick: AVGO +128%
- 9 agents spawned autonomously for new knowledge gaps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeedbackConfig:
    """Configuration for the feedback loop."""
    min_trades_for_feedback: int = 10
    feedback_interval_hours: int = 1
    auto_mutate: bool = True
    auto_ab_test: bool = True
    max_concurrent_experiments: int = 3
    improvement_threshold: float = 0.05  # 5% improvement required


@dataclass
class FeedbackEvent:
    """A feedback event from the market."""
    timestamp: datetime
    prompt_id: str
    regime: str
    pnl: float
    return_pct: float
    is_win: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackLoop:
    """Self-improving feedback loop for trading agents.
    
    Flow:
    1. Agent uses prompt to make trading decisions
    2. Market outcomes are recorded
    3. Feedback loop analyzes performance
    4. Underperforming prompts are mutated
    5. A/B tests validate improvements
    6. Best prompts are promoted
    
    Based on ATLAS-GIC pattern:
    - Agents discover their own weaknesses
    - Autonomous spawning for knowledge gaps
    - Prompt self-improvement from market feedback
    """
    
    def __init__(
        self,
        config: Optional[FeedbackConfig] = None,
    ) -> None:
        # Lazy imports to avoid circular dependencies
        from ml.prompt_optimization.ab_tester import ABTester
        from ml.prompt_optimization.prompt_optimizer import PromptOptimizer
        from ml.prompt_optimization.prompt_tracker import PromptTracker
        
        self._config = config or FeedbackConfig()
        self._tracker = PromptTracker()
        self._optimizer = PromptOptimizer(self._tracker)
        self._ab_tester = ABTester(self._tracker)
        
        # Feedback buffer
        self._feedback_buffer: List[FeedbackEvent] = []
        self._last_feedback_time: Optional[datetime] = None
        
        # Agent spawning
        self._spawned_agents: List[Dict[str, Any]] = []
        self._knowledge_gaps: List[str] = []
        
        # Statistics
        self._total_mutations = 0
        self._successful_mutations = 0
        self._feedback_cycles = 0
    
    @property
    def tracker(self) -> "PromptTracker":
        return self._tracker
    
    @property
    def optimizer(self) -> "PromptOptimizer":
        return self._optimizer
    
    @property
    def ab_tester(self) -> "ABTester":
        return self._ab_tester
    
    def register_prompt(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register an initial prompt."""
        return self._tracker.register_prompt(content=content, metadata=metadata)
    
    def record_feedback(self, event: FeedbackEvent) -> None:
        """Record a feedback event from the market."""
        self._feedback_buffer.append(event)
        
        # Also record in tracker
        self._tracker.record_trade(
            prompt_id=event.prompt_id,
            pnl=event.pnl,
            return_pct=event.return_pct,
            regime=event.regime,
            is_win=event.is_win,
        )
    
    def run_feedback_cycle(self) -> Dict[str, Any]:
        """Run one feedback cycle.
        
        1. Analyze recent performance
        2. Identify underperforming prompts
        3. Mutate or A/B test
        4. Spawn new agents if knowledge gaps detected
        
        Returns summary of actions taken.
        """
        self._feedback_cycles += 1
        actions: Dict[str, Any] = {
            "mutations": 0,
            "ab_tests_started": 0,
            "agents_spawned": 0,
            "knowledge_gaps_detected": [],
        }
        
        # Get worst performing prompts
        worst_prompts = self._tracker.get_worst_prompts(n=5, min_trades=10)
        
        for record in worst_prompts:
            if record.performance is None:
                continue
            
            # Check if significantly underperforming
            if record.performance.sharpe_ratio < -0.5:
                # Mutate the prompt
                mutated, mutation_type, new_id = self._optimizer.mutate_prompt(
                    prompt=record.content,
                    parent_id=record.prompt_id,
                )
                actions["mutations"] += 1
                self._total_mutations += 1
                
                logger.info(
                    "Mutated underperforming prompt %s (sharpe=%.2f) -> %s via %s",
                    record.prompt_id, record.performance.sharpe_ratio,
                    new_id, mutation_type,
                )
        
        # Check for A/B test opportunities
        best_prompts = self._tracker.get_best_prompts(n=3, min_trades=30)
        if len(best_prompts) >= 2:
            # Start A/B test between top 2
            active_tests = self._ab_tester.get_active_experiments()
            if len(active_tests) < self._config.max_concurrent_experiments:
                exp_id = self._ab_tester.start_experiment(
                    prompt_a_id=best_prompts[0].prompt_id,
                    prompt_b_id=best_prompts[1].prompt_id,
                )
                actions["ab_tests_started"] += 1
                logger.info("Started A/B test: %s", exp_id)
        
        # Detect knowledge gaps (regimes with poor performance)
        knowledge_gaps = self._detect_knowledge_gaps()
        actions["knowledge_gaps_detected"] = knowledge_gaps
        
        # Spawn agents for knowledge gaps
        for gap in knowledge_gaps:
            agent = self._spawn_agent_for_gap(gap)
            if agent:
                actions["agents_spawned"] += 1
                self._spawned_agents.append(agent)
        
        # Process completed A/B tests
        self._process_completed_experiments()
        
        # Update optimizer based on feedback
        self._update_optimizer()
        
        return actions
    
    def _detect_knowledge_gaps(self) -> List[str]:
        """Detect regimes or conditions where agents perform poorly."""
        gaps = []
        
        # Check regime-specific performance
        regime_performance = {}
        for record in self._tracker._prompts.values():
            if record.performance and record.performance.regime_performance:
                for regime, pnl in record.performance.regime_performance.items():
                    if regime not in regime_performance:
                        regime_performance[regime] = []
                    regime_performance[regime].append(pnl)
        
        # Identify underperforming regimes
        for regime, pnls in regime_performance.items():
            avg_pnl = sum(pnls) / len(pnls) if pnls else 0
            if avg_pnl < -0.02 and len(pnls) >= 10:  # -2% average with sufficient samples
                gaps.append(f"regime:{regime}")
        
        return gaps
    
    def _spawn_agent_for_gap(self, gap: str) -> Optional[Dict[str, Any]]:
        """Spawn a new agent to address a knowledge gap."""
        if gap.startswith("regime:"):
            regime = gap.split(":")[1]
            
            agent = {
                "id": f"agent_{len(self._spawned_agents) + 1}",
                "type": "regime_specialist",
                "regime": regime,
                "spawned_at": datetime.now().isoformat(),
                "status": "active",
                "focus": f"Improve performance in {regime} regime",
            }
            
            logger.info("Spawned agent %s for regime gap: %s", agent["id"], regime)
            return agent
        
        return None
    
    def _process_completed_experiments(self) -> None:
        """Process completed A/B experiments and learn from results."""
        completed = self._ab_tester.get_completed_experiments()
        
        for result in completed:
            if not result.is_significant:
                continue
            
            # Determine which prompt improved
            if result.winner == "A":
                improved_id = result.prompt_a_id
                baseline_id = result.prompt_b_id
            else:
                improved_id = result.prompt_b_id
                baseline_id = result.prompt_a_id
            
            # Learn from the improvement
            improved = result.improvement_pct > self._config.improvement_threshold * 100
            
            # Find mutation type
            record = self._tracker.get_prompt(improved_id)
            mutation_type = record.mutation_type if record else "unknown"
            
            self._optimizer.learn_from_outcome(
                prompt_id=improved_id,
                mutation_type=mutation_type or "unknown",
                improved=improved,
            )
            
            if improved:
                self._successful_mutations += 1
    
    def _update_optimizer(self) -> None:
        """Update optimizer based on accumulated feedback."""
        # Adjust mutation rate based on success rate
        if self._total_mutations > 10:
            success_rate = self._successful_mutations / self._total_mutations
            if success_rate < 0.2:
                # Too aggressive, reduce mutation rate
                self._optimizer._mutation_rate *= 0.9
            elif success_rate > 0.5:
                # Good success rate, can be more aggressive
                self._optimizer._mutation_rate = min(
                    self._optimizer._mutation_rate * 1.1,
                    0.5,
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback loop statistics."""
        tracker_stats = self._tracker.get_statistics()
        ab_stats = self._ab_tester.get_statistics()
        strategy_stats = self._optimizer.get_strategy_stats()
        
        return {
            "feedback_cycles": self._feedback_cycles,
            "total_mutations": self._total_mutations,
            "successful_mutations": self._successful_mutations,
            "mutation_success_rate": (
                self._successful_mutations / max(self._total_mutations, 1)
            ),
            "spawned_agents": len(self._spawned_agents),
            "tracker": tracker_stats,
            "ab_testing": ab_stats,
            "strategies": strategy_stats,
            "recommendations": self._optimizer.get_recommendations(),
        }
    
    def get_spawned_agents(self) -> List[Dict[str, Any]]:
        """Get list of spawned agents."""
        return self._spawned_agents.copy()
    
    def get_best_prompt(self, regime: Optional[str] = None) -> Optional[str]:
        """Get the best prompt, optionally for a specific regime."""
        if regime:
            prompts = self._tracker.get_regime_best_prompts(regime, n=1)
        else:
            prompts = self._tracker.get_best_prompts(n=1, min_trades=30)
        
        if prompts:
            return prompts[0].content
        return None
