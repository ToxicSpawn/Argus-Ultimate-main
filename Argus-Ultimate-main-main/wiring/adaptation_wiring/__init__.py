"""
Adaptation Wiring Package
Wires all adaptive systems together for 100% connectivity
"""

from .strategy_learning_wiring import (
    StrategyLearningWiring,
    get_strategy_learning_wiring,
    wire_all_strategy_learning
)

from .complete_adaptation_wiring import (
    CompleteAdaptationWiring,
    get_complete_adaptation_wiring,
    wire_all_adaptation_systems
)

__all__ = [
    'StrategyLearningWiring',
    'get_strategy_learning_wiring',
    'wire_all_strategy_learning',
    'CompleteAdaptationWiring',
    'get_complete_adaptation_wiring',
    'wire_all_adaptation_systems',
]
