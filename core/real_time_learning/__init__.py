"""
Real-Time Learning System - Core Infrastructure

This package provides the foundation for all adaptive components in Argus.
Components can learn from:
- Market data (every tick/bar)
- Trade executions
- Performance metrics
- External signals

All adaptations are:
1. Validated by safety rules
2. Logged in audit trail
3. Tested in paper mode before promotion

Available Components:
- AdaptiveStrategyAllocator: Dynamically adjusts strategy weights
- DynamicCorrelationMatrix: Continuously updates asset correlations
- SmartOrderRouter: Dynamically routes orders to optimal venues
- RegimeSpecificParameters: Adjusts parameters based on market regime
- DynamicPositionSizer: Adapts position sizes based on performance
- SmartExecutionEngine: Optimizes execution parameters
"""

# Avoid circular imports - import at module level
# Users should import directly from core.real_time_learning.orchestrator

__all__ = [
    "RealTimeLearningOrchestrator",
    "LearningComponent",
    "SafetyValidator",
    "LearningAudit",
    "AdaptiveStrategyAllocator",
    "DynamicCorrelationMatrix",
    "SmartOrderRouter",
    "RegimeSpecificParameters"
]