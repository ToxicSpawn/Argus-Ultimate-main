"""
Prompt optimizer using RL-based feedback.

Based on ATLAS-GIC:
- 54 prompt modifications attempted
- 16 kept (30% survival rate)
- 22% returns in 173 days deployment
- Agents discover their own weaknesses and downweight
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ml.prompt_optimization.prompt_tracker import PromptTracker

logger = logging.getLogger(__name__)


@dataclass
class MutationStrategy:
    """Strategy for mutating prompts."""
    name: str
    weight: float = 1.0
    success_rate: float = 0.0
    applications: int = 0
    
    def update_success(self, success: bool) -> None:
        self.applications += 1
        if success:
            self.success_rate = (self.success_rate * (self.applications - 1) + 1) / self.applications
        else:
            self.success_rate = (self.success_rate * (self.applications - 1)) / self.applications


class PromptOptimizer:
    """Optimize prompts using reinforcement learning feedback.
    
    Strategies:
    - Rewrite: Complete rephrasing
    - Tweak: Minor adjustments
    - Regime Adapt: Market regime-specific modifications
    - Combine: Merge successful elements from multiple prompts
    - Prune: Remove underperforming sections
    """
    
    def __init__(
        self,
        tracker: PromptTracker,
        mutation_rate: float = 0.3,
        exploration_rate: float = 0.2,
    ) -> None:
        self._tracker = tracker
        self._mutation_rate = mutation_rate
        self._exploration_rate = exploration_rate
        
        # Mutation strategies
        self._strategies: Dict[str, MutationStrategy] = {
            "rewrite": MutationStrategy("rewrite", weight=1.0),
            "tweak": MutationStrategy("tweak", weight=1.5),
            "regime_adapt": MutationStrategy("regime_adapt", weight=1.2),
            "combine": MutationStrategy("combine", weight=0.8),
            "prune": MutationStrategy("prune", weight=1.0),
        }
        
        # Known successful patterns
        self._successful_patterns: List[str] = []
        self._failed_patterns: List[str] = []
    
    def mutate_prompt(
        self,
        prompt: str,
        parent_id: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Mutate a prompt using learned strategies.
        
        Returns:
            (mutated_prompt, mutation_type, parent_id)
        """
        # Decide mutation strategy
        if random.random() < self._exploration_rate:
            # Explore: random strategy
            mutation_type = random.choice(list(self._strategies.keys()))
        else:
            # Exploit: weighted by success rate
            weights = [s.weight * (s.success_rate + 0.1) for s in self._strategies.values()]
            total = sum(weights)
            if total > 0:
                probs = [w / total for w in weights]
                mutation_type = random.choices(
                    list(self._strategies.keys()),
                    weights=probs,
                )[0]
            else:
                mutation_type = "tweak"
        
        # Apply mutation
        if mutation_type == "rewrite":
            mutated = self._rewrite_prompt(prompt)
        elif mutation_type == "tweak":
            mutated = self._tweak_prompt(prompt)
        elif mutation_type == "regime_adapt" and regime:
            mutated = self._regime_adapt_prompt(prompt, regime)
        elif mutation_type == "combine":
            mutated = self._combine_prompts(prompt)
        elif mutation_type == "prune":
            mutated = self._prune_prompt(prompt)
        else:
            mutated = self._tweak_prompt(prompt)
            mutation_type = "tweak"
        
        # Register the new prompt
        new_id = self._tracker.register_prompt(
            content=mutated,
            parent_id=parent_id,
            mutation_type=mutation_type,
        )
        
        return mutated, mutation_type, new_id
    
    def _rewrite_prompt(self, prompt: str) -> str:
        """Complete rephrasing of the prompt."""
        # In production, this would use an LLM to rewrite
        # For now, apply structural changes
        lines = prompt.strip().split("\n")
        
        # Shuffle non-critical lines
        if len(lines) > 3:
            # Keep first and last line, shuffle middle
            middle = lines[1:-1]
            random.shuffle(middle)
            lines = [lines[0]] + middle + [lines[-1]]
        
        return "\n".join(lines)
    
    def _tweak_prompt(self, prompt: str) -> str:
        """Minor adjustments to the prompt."""
        tweaks = [
            self._add_emphasis,
            self._adjust_temperature_hint,
            self._add_confidence_threshold,
            self._reorder_instructions,
        ]
        
        tweak = random.choice(tweaks)
        return tweak(prompt)
    
    def _regime_adapt_prompt(self, prompt: str, regime: str) -> str:
        """Adapt prompt for specific market regime."""
        regime_instructions = {
            "trending": "Focus on momentum and trend-following signals. Prioritize directional conviction.",
            "ranging": "Focus on mean-reversion and support/resistance. Reduce position sizes.",
            "volatile": "Increase stop-loss distances. Reduce position sizes by 50%. Wait for confirmation.",
            "crisis": "Preserve capital. Only take high-conviction trades with clear risk-defined setups.",
        }
        
        instruction = regime_instructions.get(regime, "")
        if instruction:
            return f"{prompt}\n\n[Regime: {regime.upper()}]\n{instruction}"
        return prompt
    
    def _combine_prompts(self, prompt: str) -> str:
        """Combine elements from successful prompts."""
        if not self._successful_patterns:
            return prompt
        
        # Get a successful pattern
        pattern = random.choice(self._successful_patterns)
        
        # Insert at random position
        lines = prompt.strip().split("\n")
        insert_pos = random.randint(0, len(lines))
        lines.insert(insert_pos, pattern)
        
        return "\n".join(lines)
    
    def _prune_prompt(self, prompt: str) -> str:
        """Remove underperforming sections."""
        lines = prompt.strip().split("\n")
        
        if len(lines) <= 2:
            return prompt
        
        # Remove a random middle line
        if len(lines) > 3:
            remove_pos = random.randint(1, len(lines) - 2)
            lines.pop(remove_pos)
        
        return "\n".join(lines)
    
    def _add_emphasis(self, prompt: str) -> str:
        """Add emphasis to key instructions."""
        lines = prompt.strip().split("\n")
        if lines:
            # Add emphasis to first instruction
            lines[0] = f"IMPORTANT: {lines[0]}"
        return "\n".join(lines)
    
    def _adjust_temperature_hint(self, prompt: str) -> str:
        """Add temperature guidance."""
        if "temperature" not in prompt.lower():
            return f"{prompt}\n\nUse moderate temperature (0.3-0.5) for balanced decisions."
        return prompt
    
    def _add_confidence_threshold(self, prompt: str) -> str:
        """Add confidence threshold guidance."""
        if "confidence" not in prompt.lower():
            return f"{prompt}\n\nOnly act when confidence exceeds 0.7."
        return prompt
    
    def _reorder_instructions(self, prompt: str) -> str:
        """Reorder instructions for emphasis."""
        lines = prompt.strip().split("\n")
        if len(lines) > 2:
            # Move last line to second position
            last = lines.pop()
            lines.insert(1, last)
        return "\n".join(lines)
    
    def learn_from_outcome(
        self,
        prompt_id: str,
        mutation_type: str,
        improved: bool,
    ) -> None:
        """Learn from prompt outcome to improve future mutations."""
        if mutation_type in self._strategies:
            self._strategies[mutation_type].update_success(improved)
        
        if improved:
            # Extract successful pattern
            record = self._tracker.get_prompt(prompt_id)
            if record:
                self._successful_patterns.append(record.content[:200])
                # Keep only recent patterns
                if len(self._successful_patterns) > 50:
                    self._successful_patterns = self._successful_patterns[-50:]
        else:
            record = self._tracker.get_prompt(prompt_id)
            if record:
                self._failed_patterns.append(record.content[:200])
                if len(self._failed_patterns) > 50:
                    self._failed_patterns = self._failed_patterns[-50:]
    
    def get_strategy_stats(self) -> Dict[str, dict]:
        """Get statistics for each mutation strategy."""
        return {
            name: {
                "success_rate": s.success_rate,
                "applications": s.applications,
                "weight": s.weight,
            }
            for name, s in self._strategies.items()
        }
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for prompt improvement."""
        stats = self._tracker.get_statistics()
        recommendations = []
        
        if stats["survival_rate"] < 0.2:
            recommendations.append(
                "Low survival rate. Consider more conservative mutations."
            )
        
        if stats["total_prompts"] > 100 and stats["surviving_prompts"] < 5:
            recommendations.append(
                "Many prompts failing. Review base prompt structure."
            )
        
        # Check best strategy
        best_strategy = max(self._strategies.values(), key=lambda s: s.success_rate)
        if best_strategy.applications > 10:
            recommendations.append(
                f"'{best_strategy.name}' strategy has {best_strategy.success_rate:.1%} success rate. "
                f"Consider increasing its weight."
            )
        
        return recommendations
