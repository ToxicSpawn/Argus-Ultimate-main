"""Strategy package.

Batch 4 contract:
- strategies are advisory-only
- strategies must emit target proposals, not direct orders
"""

from .target_contract import StrategyTargetProposal, TargetOnlyStrategyMixin

__all__ = ["StrategyTargetProposal", "TargetOnlyStrategyMixin"]
