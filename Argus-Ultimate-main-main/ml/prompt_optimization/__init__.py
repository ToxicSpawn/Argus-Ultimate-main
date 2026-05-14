"""
Self-Improving Prompt Optimization for Trading Agents.

Based on ATLAS-GIC pattern:
- Track prompt performance over time
- A/B test prompt variations
- Auto-optimize based on market feedback
- Agents improve their own prompts from market outcomes

Performance: 30% prompt survival rate, 22% returns in 173 days (ATLAS-GIC)
"""

from ml.prompt_optimization.prompt_tracker import PromptTracker, PromptPerformance
from ml.prompt_optimization.prompt_optimizer import PromptOptimizer
from ml.prompt_optimization.ab_tester import ABTester, ExperimentResult
from ml.prompt_optimization.feedback_loop import FeedbackLoop

__all__ = [
    "PromptTracker",
    "PromptPerformance",
    "PromptOptimizer",
    "ABTester",
    "ExperimentResult",
    "FeedbackLoop",
]
