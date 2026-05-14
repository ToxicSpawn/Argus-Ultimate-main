"""
Argus Evolution Module - Genetic Algorithm Strategy Optimization

Continuously evolves strategy parameters to maximize Sharpe ratio
and adapt to changing market conditions.

V2.1 ULTIMATE Features:
- TRUE Rolling K-Fold Walk-Forward Validation
- Transaction Costs & Slippage
- Short Position Support
- Multi-Timeframe Signal Integration (actually used)
- NSGA-II Pareto Optimization (real Pareto front)
- CMA-ES with Evolution Paths & Step-Size Adaptation
- Seeded Robustness Testing
- Regime Detection & Signal Weighting
- ATR-Based Dynamic Stops
- Volume Confirmation
- Z-Score Mean Reversion
- Volatility-Scaled Position Sizing
- Early Stopping
"""

from .godmode_evolution import (
    GodmodeEvolutionEngine,
    GodmodeGAConfig,
    evolve_godmode_params,
    load_evolved_params,
)

from .godmode_evolution_v2 import (
    UltimateEvolutionEngine,
    UltimateGAConfig,
    UltimateEvolutionResult,
    FitnessComponents,
    LRUFitnessCache,
    CMAState,
    MarketRegime,
    ULTIMATE_PARAM_BOUNDS,
    load_ultimate_evolved_params,
)

__all__ = [
    # V1 (Original)
    "GodmodeEvolutionEngine",
    "GodmodeGAConfig",
    "evolve_godmode_params",
    "load_evolved_params",
    # V2.1 (Ultimate)
    "UltimateEvolutionEngine",
    "UltimateGAConfig",
    "UltimateEvolutionResult",
    "FitnessComponents",
    "LRUFitnessCache",
    "CMAState",
    "MarketRegime",
    "ULTIMATE_PARAM_BOUNDS",
    "load_ultimate_evolved_params",
]
