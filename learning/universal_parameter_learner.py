# pyright: reportMissingImports=false
"""
Universal Parameter Learning Engine
=====================================
Continuously learns optimal values for ALL parameters across Argus.

This system:
1. TRACKS every hard-coded parameter in the codebase
2. RECORDS outcomes when parameters are used
3. LEARNS optimal values per regime/asset/time-of-day
4. UPDATES parameters automatically via hot-reload
5. LOGS all changes for auditability
6. VALIDATES learned values before applying

Target: 200+ parameters across:
- Signal weights (40+)
- Confidence thresholds (50+)
- Risk parameters (30+)
- Learning rates (30+)
- Strategy parameters (99+ strategies)
- Execution parameters (20+)
- Volatility parameters (15+)

Architecture:
- ParameterRegistry: Stores all parameter definitions
- ParameterLearner: Core learning algorithm (Bayesian optimization)
- ParameterStore: Thread-safe parameter storage with versioning
- ParameterApplier: Hot-reload parameter updates
- ParameterValidator: Validates learned values before apply
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of learnable parameters."""
    WEIGHT = auto()           # Signal/model weights (0-1)
    THRESHOLD = auto()        # Decision thresholds
    MULTIPLIER = auto()       # Position sizing multipliers
    LEARNING_RATE = auto()    # ML learning rates
    PERCENTAGE = auto()       # Percentage values (0-100)
    INTEGER = auto()          # Integer parameters
    CATEGORICAL = auto()      # Category selection


class ParameterCategory(Enum):
    """Categories of parameters."""
    SIGNAL = auto()           # Signal generation
    RISK = auto()             # Risk management
    EXECUTION = auto()        # Order execution
    STRATEGY = auto()         # Strategy-specific
    ML = auto()               # Machine learning
    ENSEMBLE = auto()          # Ensemble/weighting
    REGIME = auto()           # Regime detection
    MARKET_MAKING = auto()    # Market making
    VOLATILITY = auto()       # Volatility models


@dataclass
class ParameterDefinition:
    """Definition of a learnable parameter."""
    name: str
    full_path: str            # Full Python path (e.g., "risk.stop_loss_pct")
    parameter_type: ParameterType
    category: ParameterCategory
    default_value: float      # Original hard-coded value
    current_value: float      # Current (potentially learned) value
    min_value: float          # Minimum allowed value
    max_value: float          # Maximum allowed value
    learning_rate: float = 0.01  # How fast to adapt
    exploration_rate: float = 0.1  # Chance of trying new values
    min_samples: int = 20     # Minimum samples before learning
    file_path: Optional[str] = None  # File to update
    line_number: Optional[int] = None  # Line to update


@dataclass
class ParameterObservation:
    """A single observation of parameter performance."""
    timestamp: datetime
    parameter_name: str
    parameter_value: float
    regime: str
    asset: str
    outcome: float           # PnL or other metric
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterLearningResult:
    """Result of parameter learning."""
    parameter_name: str
    old_value: float
    new_value: float
    improvement_estimate: float
    confidence: float
    samples_used: int
    reason: str


class ParameterLearner:
    """
    Core learning algorithm for individual parameters.
    
    Uses a combination of:
    1. Thompson Sampling for exploration/exploitation
    2. Bayesian updating for uncertainty
    3. Contextual bandits for regime-aware learning
    """
    
    def __init__(
        self,
        definition: ParameterDefinition,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0
    ):
        self.definition = definition
        self.alpha = alpha_prior  # Beta distribution alpha (successes)
        self.beta = beta_prior    # Beta distribution beta (failures)
        
        # Context-aware learning (regime -> performance)
        self.regime_performance: Dict[str, List[float]] = defaultdict(list)
        self.asset_performance: Dict[str, List[float]] = defaultdict(list)
        self.time_performance: Dict[int, List[float]] = defaultdict(list)  # Hour -> performance
        
        # Observation history
        self.observations: Deque[ParameterObservation] = deque(maxlen=10000)
        
        # Value testing
        self.value_performance: Dict[float, List[float]] = defaultdict(list)
    
    def observe(self, observation: ParameterObservation) -> None:
        """Record a new observation."""
        self.observations.append(observation)
        
        # Update global statistics
        if observation.outcome > 0:
            self.alpha += 1
        else:
            self.beta += 1
        
        # Update contextual statistics
        self.regime_performance[observation.regime].append(observation.outcome)
        self.asset_performance[observation.asset].append(observation.outcome)
        
        hour = observation.timestamp.hour
        self.time_performance[hour].append(observation.outcome)
        
        # Track value-specific performance
        rounded_value = round(observation.parameter_value, 4)
        self.value_performance[rounded_value].append(observation.outcome)
    
    def sample_value(self, regime: Optional[str] = None) -> float:
        """
        Sample a value to try using Thompson Sampling.
        
        Returns:
            A parameter value to test
        """
        # With probability exploration_rate, try a random value
        if np.random.random() < self.definition.exploration_rate:
            return self._random_value()
        
        # Get contextual values if regime provided
        if regime and regime in self.regime_performance:
            regime_vals = self._get_best_regime_value(regime)
            if regime_vals is not None:
                return regime_vals
        
        # Use Thompson Sampling on observed values
        if self.value_performance:
            best_value = self._thompson_sample()
            if best_value is not None:
                return best_value
        
        # Default: sample from Beta distribution
        sampled_rate = np.random.beta(self.alpha, self.beta)
        
        # Map rate to parameter range
        return self.definition.min_value + (
            sampled_rate * (self.definition.max_value - self.definition.min_value)
        )
    
    def _random_value(self) -> float:
        """Generate a random value within bounds."""
        return np.random.uniform(self.definition.min_value, self.definition.max_value)
    
    def _thompson_sample(self) -> Optional[float]:
        """Thompson sampling over observed values."""
        if not self.value_performance:
            return None
        
        best_value = None
        best_sample = -float('inf')
        
        for value, outcomes in self.value_performance.items():
            if len(outcomes) < 3:
                continue
            
            # Calculate success rate
            successes = sum(1 for o in outcomes if o > 0)
            failures = len(outcomes) - successes
            
            # Sample from Beta distribution
            sample = np.random.beta(successes + 1, failures + 1)
            
            if sample > best_sample:
                best_sample = sample
                best_value = value
        
        return best_value
    
    def _get_best_regime_value(self, regime: str) -> Optional[float]:
        """Get best value for a specific regime."""
        if regime not in self.regime_performance:
            return None
        
        outcomes = self.regime_performance[regime]
        if len(outcomes) < 5:
            return None
        
        # Simple: if regime performance is positive, keep current direction
        # If negative, try changing
        avg_outcome = np.mean(outcomes[-20:]) if len(outcomes) >= 20 else np.mean(outcomes)
        
        if avg_outcome > 0:
            return self.definition.current_value  # Keep doing what works
        else:
            # Try a different value
            return self._random_value()
    
    def get_confidence(self) -> float:
        """Get confidence in current optimal value."""
        total_observations = len(self.observations)
        if total_observations < self.definition.min_samples:
            return 0.0
        
        # Confidence based on sample size and consistency
        sample_confidence = min(1.0, total_observations / (self.definition.min_samples * 5))
        
        # Consistency bonus
        if self.value_performance:
            best_value = max(self.value_performance.keys(), 
                           key=lambda k: np.mean(self.value_performance[k]))
            best_outcomes = self.value_performance[best_value]
            if len(best_outcomes) >= 5:
                consistency = np.std(best_outcomes) / (abs(np.mean(best_outcomes)) + 1e-6)
                consistency = max(0, 1 - consistency)
            else:
                consistency = 0.5
        else:
            consistency = 0.5
        
        return sample_confidence * 0.6 + consistency * 0.4
    
    def get_learned_value(
        self, 
        regime: Optional[str] = None,
        asset: Optional[str] = None
    ) -> Optional[float]:
        """Get the learned optimal value, optionally specific to regime and/or asset."""
        if len(self.observations) < self.definition.min_samples:
            return None
        
        # Check asset+regime-specific value first (most specific)
        asset_regime_key = f"{asset}:{regime}" if asset and regime else None
        if asset_regime_key:
            # Track combined asset+regime performance
            key = (asset, regime)
            if key in self._get_asset_regime_performance():
                outcomes = self._get_asset_regime_performance()[key]
                if len(outcomes) >= self.definition.min_samples:
                    return self._get_best_value_for_outcomes(outcomes)
        
        # Check asset-specific value
        if asset and asset in self.asset_performance:
            outcomes = self.asset_performance[asset]
            if len(outcomes) >= self.definition.min_samples:
                return self._get_best_value_for_outcomes(outcomes)
        
        # Check regime-specific value
        if regime and regime in self.regime_performance:
            outcomes = self.regime_performance[regime]
            if len(outcomes) >= self.definition.min_samples:
                return self._get_best_value_for_outcomes(outcomes)
        
        # Return best observed value (global)
        if self.value_performance:
            best_value = max(self.value_performance.keys(),
                           key=lambda k: np.mean(self.value_performance[k]))
            return best_value
        
        return None
    
    def _get_asset_regime_performance(self) -> Dict[Tuple[str, str], List[float]]:
        """Get asset+regime combined performance (computed on demand)."""
        # This is computed from observations for better context-specific learning
        # Convert to list first to avoid "deque mutated during iteration"
        result: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for obs in list(self.observations):
            key = (obs.asset, obs.regime)
            result[key].append(obs.outcome)
        return result
    
    def _get_best_value_for_outcomes(self, outcomes: List[float]) -> float:
        """Get the best parameter value based on observed outcomes."""
        # Find the value with highest mean outcome in this context
        if self.value_performance:
            best_value = max(self.value_performance.keys(),
                           key=lambda k: np.mean(self.value_performance[k]))
            return best_value
        return self.definition.current_value
    
    def get_asset_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get per-asset learning statistics."""
        stats = {}
        for asset, outcomes in self.asset_performance.items():
            if len(outcomes) > 0:
                stats[asset] = {
                    "observations": len(outcomes),
                    "mean_outcome": float(np.mean(outcomes)),
                    "std_outcome": float(np.std(outcomes)) if len(outcomes) > 1 else 0.0,
                    "positive_rate": sum(1 for o in outcomes if o > 0) / len(outcomes),
                }
        return stats
    
    def get_regime_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get per-regime learning statistics."""
        stats = {}
        for regime, outcomes in self.regime_performance.items():
            if len(outcomes) > 0:
                stats[regime] = {
                    "observations": len(outcomes),
                    "mean_outcome": float(np.mean(outcomes)),
                    "std_outcome": float(np.std(outcomes)) if len(outcomes) > 1 else 0.0,
                    "positive_rate": sum(1 for o in outcomes if o > 0) / len(outcomes),
                }
        return stats


class ParameterRegistry:
    """
    Registry of all learnable parameters in Argus.
    
    Target: 200+ parameters across all trading systems.
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterDefinition] = {}
        self.learners: Dict[str, ParameterLearner] = {}
        
        # Register all known parameters
        self._register_signal_parameters()
        self._register_signal_smoothing_parameters()
        self._register_threshold_parameters()
        self._register_risk_parameters()
        self._register_risk_position_sizing_parameters()
        self._register_risk_stop_parameters()
        self._register_var_parameters()
        self._register_ensemble_parameters()
        self._register_execution_parameters()
        self._register_execution_latency_parameters()
        self._register_latency_tier_parameters()
        self._register_strategy_parameters()
        self._register_strategy_market_making_parameters()
        self._register_strategy_trend_parameters()
        self._register_strategy_mean_reversion_parameters()
        self._register_ml_model_parameters()
        self._register_ml_training_parameters()
        self._register_ml_deep_learning_parameters()
        self._register_volatility_parameters()
        self._register_regime_detection_parameters()
        self._register_adaptive_learning_parameters()
        self._register_microstructure_parameters()
        self._register_alpha_parameters()
        self._register_portfolio_parameters()
        self._register_circuit_breaker_parameters()
        self._register_defi_parameters()
        self._register_on_chain_parameters()
        self._register_backtest_parameters()
        
        logger.info("Parameter Registry initialized with %d parameters", len(self.parameters))
    
    def _register_signal_parameters(self) -> None:
        """Register signal weight parameters."""
        signal_params = [
            ParameterDefinition(
                name="whale_tracking_weight",
                full_path="alpha.signal_sources.whale_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.15,
                current_value=0.15,
                min_value=0.0,
                max_value=0.5,
                file_path="alpha/signal_sources.py",
            ),
            ParameterDefinition(
                name="exchange_flow_weight",
                full_path="alpha.signal_sources.flow_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.20,
                current_value=0.20,
                min_value=0.0,
                max_value=0.5,
                file_path="alpha/signal_sources.py",
            ),
            ParameterDefinition(
                name="social_sentiment_weight",
                full_path="alpha.signal_sources.social_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.15,
                current_value=0.15,
                min_value=0.0,
                max_value=0.5,
                file_path="alpha/signal_sources.py",
            ),
            ParameterDefinition(
                name="news_sentiment_weight",
                full_path="alpha.signal_sources.news_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.25,
                current_value=0.25,
                min_value=0.0,
                max_value=0.5,
                file_path="alpha/signal_sources.py",
            ),
            ParameterDefinition(
                name="derivatives_weight",
                full_path="alpha.signal_sources.derivatives_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.25,
                current_value=0.25,
                min_value=0.0,
                max_value=0.5,
                file_path="alpha/signal_sources.py",
            ),
            ParameterDefinition(
                name="news_weight_alt",
                full_path="analytics.alt_data_fusion.news_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.6,
                current_value=0.6,
                min_value=0.0,
                max_value=1.0,
                file_path="analytics/alt_data_fusion.py",
            ),
            ParameterDefinition(
                name="social_weight_alt",
                full_path="analytics.alt_data_fusion.social_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.4,
                current_value=0.4,
                min_value=0.0,
                max_value=1.0,
                file_path="analytics/alt_data_fusion.py",
            ),
            ParameterDefinition(
                name="lob_weight",
                full_path="alpha.microstructure.lob_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.5,
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                file_path="alpha/microstructure/live_ofi_stream.py",
            ),
        ]
        
        for param in signal_params:
            self.register_parameter(param)
    
    def _register_threshold_parameters(self) -> None:
        """Register threshold parameters."""
        threshold_params = [
            ParameterDefinition(
                name="confidence_threshold",
                full_path="adaptive.confidence_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.7,
                current_value=0.7,
                min_value=0.3,
                max_value=0.95,
                file_path="adaptive/adaptive_learning_engine.py",
            ),
            ParameterDefinition(
                name="warning_threshold",
                full_path="unified.warning_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.65,
                current_value=0.65,
                min_value=0.5,
                max_value=0.85,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="danger_threshold",
                full_path="unified.danger_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.82,
                current_value=0.82,
                min_value=0.7,
                max_value=0.95,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="quality_threshold",
                full_path="unified.quality_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.75,
                current_value=0.75,
                min_value=0.5,
                max_value=0.95,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="rebalance_threshold",
                full_path="unified.rebalance_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=0.05,
                current_value=0.05,
                min_value=0.01,
                max_value=0.20,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="high_correlation_threshold",
                full_path="unified.high_correlation_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.7,
                current_value=0.7,
                min_value=0.5,
                max_value=0.95,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="large_trade_threshold",
                full_path="unified.large_trade_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.EXECUTION,
                default_value=10000.0,
                current_value=10000.0,
                min_value=1000.0,
                max_value=100000.0,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="rebalance_drift_threshold",
                full_path="adaptive.rebalance_drift_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=0.10,
                current_value=0.10,
                min_value=0.05,
                max_value=0.25,
                file_path="adaptive/auto_capital_allocator.py",
            ),
            ParameterDefinition(
                name="confidence_threshold_adaptive",
                full_path="adaptive.omega_confidence_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.3,
                current_value=0.3,
                min_value=0.1,
                max_value=0.8,
                file_path="adaptive/omega_adaptive.py",
            ),
            ParameterDefinition(
                name="importance_threshold",
                full_path="argus.importance_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=0.3,
                current_value=0.3,
                min_value=0.1,
                max_value=0.8,
                file_path="argus_ultimate.py",
            ),
            ParameterDefinition(
                name="scale_up_threshold",
                full_path="unified.scale_up_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=0.05,
                current_value=0.05,
                min_value=0.01,
                max_value=0.15,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="impact_threshold_bps",
                full_path="unified.impact_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.EXECUTION,
                default_value=15.0,
                current_value=15.0,
                min_value=5.0,
                max_value=50.0,
                file_path="unified_trading_system.py",
            ),
        ]
        
        for param in threshold_params:
            self.register_parameter(param)
    
    def _register_risk_parameters(self) -> None:
        """Register risk management parameters."""
        risk_params = [
            ParameterDefinition(
                name="quantum_weight",
                full_path="unified.quantum_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.RISK,
                default_value=0.3,
                current_value=0.3,
                min_value=0.0,
                max_value=1.0,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="position_size_multiplier",
                full_path="risk.position_size_multiplier",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=1.0,
                current_value=1.0,
                min_value=0.1,
                max_value=2.0,
                file_path="risk/adaptive_risk_manager.py",
            ),
            ParameterDefinition(
                name="stop_loss_multiplier",
                full_path="risk.stop_loss_multiplier",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=1.2,
                current_value=1.2,
                min_value=0.5,
                max_value=3.0,
                file_path="risk/adaptive_risk_manager.py",
            ),
            ParameterDefinition(
                name="leverage_threshold",
                full_path="unified.leverage_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=8.0,
                current_value=8.0,
                min_value=1.0,
                max_value=20.0,
                file_path="unified_trading_system.py",
            ),
            ParameterDefinition(
                name="funding_threshold",
                full_path="unified.funding_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.005,
                current_value=0.005,
                min_value=0.001,
                max_value=0.02,
                file_path="unified_trading_system.py",
            ),
        ]
        
        for param in risk_params:
            self.register_parameter(param)
    
    def _register_ensemble_parameters(self) -> None:
        """Register ensemble/weighting parameters."""
        ensemble_params = [
            ParameterDefinition(
                name="ensemble_voter_weight",
                full_path="ml.ensemble_voter_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.10,
                current_value=0.10,
                min_value=0.0,
                max_value=0.5,
                file_path="activate_ml_ensemble.py",
            ),
            ParameterDefinition(
                name="signal_stacker_weight",
                full_path="ml.signal_stacker_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.08,
                current_value=0.08,
                min_value=0.0,
                max_value=0.5,
                file_path="activate_ml_ensemble.py",
            ),
        ]
        
        for param in ensemble_params:
            self.register_parameter(param)
    
    def _register_execution_parameters(self) -> None:
        """Register execution parameters."""
        execution_params = [
            ParameterDefinition(
                name="expected_slippage_bps",
                full_path="execution.expected_slippage",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.EXECUTION,
                default_value=5.0,
                current_value=5.0,
                min_value=1.0,
                max_value=50.0,
                file_path="execution/microstructure_adapter.py",
            ),
        ]
        
        for param in execution_params:
            self.register_parameter(param)
    
    def _register_strategy_parameters(self) -> None:
        """Register strategy-specific parameters."""
        # Market making parameters
        strategy_params = [
            ParameterDefinition(
                name="base_spread_bps",
                full_path="strategy.market_maker.base_spread",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=10.0,
                current_value=10.0,
                min_value=1.0,
                max_value=50.0,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="min_apy_threshold",
                full_path="strategy.funding_arb.min_apy",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.10,
                current_value=0.10,
                min_value=0.02,
                max_value=0.50,
                file_path="strategies/funding_rate_arb.py",
            ),
            ParameterDefinition(
                name="max_basis_pct",
                full_path="strategy.funding_arb.max_basis",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.003,
                current_value=0.003,
                min_value=0.001,
                max_value=0.01,
                file_path="strategies/funding_rate_arb.py",
            ),
        ]
        
        for param in strategy_params:
            self.register_parameter(param)
    
    def _register_signal_smoothing_parameters(self) -> None:
        """Register signal smoothing/EMA parameters."""
        smoothing_params = [
            ParameterDefinition(
                name="ema_short_period",
                full_path="signals.ema_short_period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=9.0,
                current_value=9.0,
                min_value=3.0,
                max_value=20.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="ema_medium_period",
                full_path="signals.ema_medium_period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=21.0,
                current_value=21.0,
                min_value=10.0,
                max_value=50.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="ema_long_period",
                full_path="signals.ema_long_period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=50.0,
                current_value=50.0,
                min_value=20.0,
                max_value=100.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="rsi_period",
                full_path="signals.rsi_period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=14.0,
                current_value=14.0,
                min_value=7.0,
                max_value=30.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="rsi_overbought",
                full_path="signals.rsi_overbought",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=70.0,
                current_value=70.0,
                min_value=60.0,
                max_value=85.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="rsi_oversold",
                full_path="signals.rsi_oversold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=30.0,
                current_value=30.0,
                min_value=15.0,
                max_value=40.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="macd_fast_period",
                full_path="signals.macd_fast",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=12.0,
                current_value=12.0,
                min_value=8.0,
                max_value=20.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="macd_slow_period",
                full_path="signals.macd_slow",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=26.0,
                current_value=26.0,
                min_value=15.0,
                max_value=40.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="macd_signal_period",
                full_path="signals.macd_signal",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=9.0,
                current_value=9.0,
                min_value=5.0,
                max_value=15.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="bollinger_period",
                full_path="signals.bollinger_period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=20.0,
                current_value=20.0,
                min_value=10.0,
                max_value=40.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="bollinger_std",
                full_path="signals.bollinger_std",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.SIGNAL,
                default_value=2.0,
                current_value=2.0,
                min_value=1.5,
                max_value=3.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="atr_period",
                full_path="signals.atr_period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=14.0,
                current_value=14.0,
                min_value=7.0,
                max_value=30.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="stochastic_k_period",
                full_path="signals.stoch_k",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=14.0,
                current_value=14.0,
                min_value=5.0,
                max_value=30.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="stochastic_d_period",
                full_path="signals.stoch_d",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=3.0,
                current_value=3.0,
                min_value=1.0,
                max_value=7.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="adx_period",
                full_path="signals.adx_period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=14.0,
                current_value=14.0,
                min_value=7.0,
                max_value=30.0,
                file_path="signals/technical_indicators.py",
            ),
            ParameterDefinition(
                name="adx_threshold",
                full_path="signals.adx_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=25.0,
                current_value=25.0,
                min_value=15.0,
                max_value=40.0,
                file_path="signals/technical_indicators.py",
            ),
        ]
        
        for param in smoothing_params:
            self.register_parameter(param)
    
    def _register_risk_position_sizing_parameters(self) -> None:
        """Register risk position sizing parameters."""
        sizing_params = [
            ParameterDefinition(
                name="kelly_fraction",
                full_path="risk.kelly_fraction",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=0.25,
                current_value=0.25,
                min_value=0.1,
                max_value=1.0,
                file_path="risk/kelly_criterion.py",
            ),
            ParameterDefinition(
                name="volatility_scaling_factor",
                full_path="risk.vol_scaling",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=1.0,
                current_value=1.0,
                min_value=0.5,
                max_value=2.0,
                file_path="risk/volatility_sizing.py",
            ),
            ParameterDefinition(
                name="max_position_pct",
                full_path="risk.max_position_pct",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.05,
                current_value=0.05,
                min_value=0.01,
                max_value=0.20,
                file_path="risk/position_manager.py",
            ),
            ParameterDefinition(
                name="risk_per_trade_pct",
                full_path="risk.risk_per_trade",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.01,
                current_value=0.01,
                min_value=0.005,
                max_value=0.05,
                file_path="risk/position_manager.py",
            ),
            ParameterDefinition(
                name="correlation_adjustment_factor",
                full_path="risk.corr_adj_factor",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=0.8,
                current_value=0.8,
                min_value=0.5,
                max_value=1.0,
                file_path="risk/correlation_manager.py",
            ),
            ParameterDefinition(
                name="sector_concentration_limit",
                full_path="risk.sector_limit",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.30,
                current_value=0.30,
                min_value=0.10,
                max_value=0.50,
                file_path="risk/concentration_manager.py",
            ),
            ParameterDefinition(
                name="single_asset_limit_pct",
                full_path="risk.single_asset_limit",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.15,
                current_value=0.15,
                min_value=0.05,
                max_value=0.30,
                file_path="risk/concentration_manager.py",
            ),
            ParameterDefinition(
                name="drawdown_position_scale",
                full_path="risk.dd_position_scale",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=0.5,
                current_value=0.5,
                min_value=0.1,
                max_value=1.0,
                file_path="risk/drawdown_manager.py",
            ),
            ParameterDefinition(
                name="win_streak_scale_factor",
                full_path="risk.win_streak_scale",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=1.2,
                current_value=1.2,
                min_value=1.0,
                max_value=2.0,
                file_path="risk/streak_manager.py",
            ),
            ParameterDefinition(
                name="loss_streak_scale_factor",
                full_path="risk.loss_streak_scale",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=0.7,
                current_value=0.7,
                min_value=0.3,
                max_value=1.0,
                file_path="risk/streak_manager.py",
            ),
        ]
        
        for param in sizing_params:
            self.register_parameter(param)
    
    def _register_risk_stop_parameters(self) -> None:
        """Register risk stop loss parameters."""
        stop_params = [
            ParameterDefinition(
                name="atr_stop_multiplier",
                full_path="risk.atr_stop_mult",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=2.0,
                current_value=2.0,
                min_value=1.0,
                max_value=4.0,
                file_path="risk/stop_manager.py",
            ),
            ParameterDefinition(
                name="trailing_stop_activation_pct",
                full_path="risk.trailing_stop_activation",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.02,
                current_value=0.02,
                min_value=0.005,
                max_value=0.05,
                file_path="risk/stop_manager.py",
            ),
            ParameterDefinition(
                name="trailing_stop_distance_pct",
                full_path="risk.trailing_stop_distance",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.01,
                current_value=0.01,
                min_value=0.005,
                max_value=0.03,
                file_path="risk/stop_manager.py",
            ),
            ParameterDefinition(
                name="breakeven_trigger_pct",
                full_path="risk.breakeven_trigger",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.015,
                current_value=0.015,
                min_value=0.005,
                max_value=0.03,
                file_path="risk/stop_manager.py",
            ),
            ParameterDefinition(
                name="time_stop_hours",
                full_path="risk.time_stop_hours",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.RISK,
                default_value=4.0,
                current_value=4.0,
                min_value=1.0,
                max_value=24.0,
                file_path="risk/stop_manager.py",
            ),
            ParameterDefinition(
                name="max_hold_time_hours",
                full_path="risk.max_hold_time",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.RISK,
                default_value=24.0,
                current_value=24.0,
                min_value=4.0,
                max_value=72.0,
                file_path="risk/stop_manager.py",
            ),
            ParameterDefinition(
                name="partial_exit_1_pct",
                full_path="risk.partial_exit_1",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.02,
                current_value=0.02,
                min_value=0.005,
                max_value=0.05,
                file_path="risk/partial_exits.py",
            ),
            ParameterDefinition(
                name="partial_exit_2_pct",
                full_path="risk.partial_exit_2",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.04,
                current_value=0.04,
                min_value=0.02,
                max_value=0.10,
                file_path="risk/partial_exits.py",
            ),
            ParameterDefinition(
                name="partial_exit_1_size_pct",
                full_path="risk.partial_exit_1_size",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.25,
                current_value=0.25,
                min_value=0.10,
                max_value=0.50,
                file_path="risk/partial_exits.py",
            ),
            ParameterDefinition(
                name="partial_exit_2_size_pct",
                full_path="risk.partial_exit_2_size",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.35,
                current_value=0.35,
                min_value=0.20,
                max_value=0.60,
                file_path="risk/partial_exits.py",
            ),
        ]
        
        for param in stop_params:
            self.register_parameter(param)
    
    def _register_execution_latency_parameters(self) -> None:
        """Register execution latency parameters."""
        latency_params = [
            ParameterDefinition(
                name="slippage_model_aggressiveness",
                full_path="execution.slippage_aggression",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.EXECUTION,
                default_value=0.5,
                current_value=0.5,
                min_value=0.1,
                max_value=1.0,
                file_path="execution/slippage_model.py",
            ),
            ParameterDefinition(
                name="latency_compensation_ms",
                full_path="execution.latency_comp_ms",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=50.0,
                current_value=50.0,
                min_value=10.0,
                max_value=200.0,
                file_path="execution/latency_compensator.py",
            ),
            ParameterDefinition(
                name="order_timeout_ms",
                full_path="execution.order_timeout",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=5000.0,
                current_value=5000.0,
                min_value=1000.0,
                max_value=30000.0,
                file_path="execution/order_manager.py",
            ),
            ParameterDefinition(
                name="retry_delay_ms",
                full_path="execution.retry_delay",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=100.0,
                current_value=100.0,
                min_value=50.0,
                max_value=500.0,
                file_path="execution/order_manager.py",
            ),
            ParameterDefinition(
                name="max_retries",
                full_path="execution.max_retries",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=3.0,
                current_value=3.0,
                min_value=1.0,
                max_value=10.0,
                file_path="execution/order_manager.py",
            ),
            ParameterDefinition(
                name="iceberg_order_size_pct",
                full_path="execution.iceberg_size",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.EXECUTION,
                default_value=0.10,
                current_value=0.10,
                min_value=0.01,
                max_value=0.50,
                file_path="execution/iceberg_executor.py",
            ),
            ParameterDefinition(
                name="twap_interval_ms",
                full_path="execution.twap_interval",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=1000.0,
                current_value=1000.0,
                min_value=100.0,
                max_value=10000.0,
                file_path="execution/twap_executor.py",
            ),
            ParameterDefinition(
                name="vwap_lookback_minutes",
                full_path="execution.vwap_lookback",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=30.0,
                current_value=30.0,
                min_value=5.0,
                max_value=120.0,
                file_path="execution/vwap_executor.py",
            ),
        ]
        
        for param in latency_params:
            self.register_parameter(param)
    
    def _register_strategy_market_making_parameters(self) -> None:
        """Register market making strategy parameters."""
        mm_params = [
            ParameterDefinition(
                name="mm_spread_multiplier",
                full_path="strategy.mm.spread_mult",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.STRATEGY,
                default_value=1.0,
                current_value=1.0,
                min_value=0.5,
                max_value=3.0,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="mm_inventory_skew_factor",
                full_path="strategy.mm.inventory_skew",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.STRATEGY,
                default_value=0.5,
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="mm_max_inventory_pct",
                full_path="strategy.mm.max_inventory",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.10,
                current_value=0.10,
                min_value=0.02,
                max_value=0.30,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="mm_order_refresh_seconds",
                full_path="strategy.mm.refresh_sec",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.STRATEGY,
                default_value=5.0,
                current_value=5.0,
                min_value=1.0,
                max_value=30.0,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="mm_cancel_replace_threshold_bps",
                full_path="strategy.mm.cancel_replace_bps",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=2.0,
                current_value=2.0,
                min_value=0.5,
                max_value=10.0,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="mm_max_spread_bps",
                full_path="strategy.mm.max_spread",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=20.0,
                current_value=20.0,
                min_value=5.0,
                max_value=50.0,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="mm_min_spread_bps",
                full_path="strategy.mm.min_spread",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=2.0,
                current_value=2.0,
                min_value=0.5,
                max_value=10.0,
                file_path="strategies/market_maker.py",
            ),
            ParameterDefinition(
                name="mm_quote_size_usd",
                full_path="strategy.mm.quote_size",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=100.0,
                current_value=100.0,
                min_value=10.0,
                max_value=1000.0,
                file_path="strategies/market_maker.py",
            ),
        ]
        
        for param in mm_params:
            self.register_parameter(param)
    
    def _register_strategy_trend_parameters(self) -> None:
        """Register trend following strategy parameters."""
        trend_params = [
            ParameterDefinition(
                name="trend_ma_fast_period",
                full_path="strategy.trend.ma_fast",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.STRATEGY,
                default_value=10.0,
                current_value=10.0,
                min_value=5.0,
                max_value=30.0,
                file_path="strategies/trend_following.py",
            ),
            ParameterDefinition(
                name="trend_ma_slow_period",
                full_path="strategy.trend.ma_slow",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.STRATEGY,
                default_value=50.0,
                current_value=50.0,
                min_value=20.0,
                max_value=100.0,
                file_path="strategies/trend_following.py",
            ),
            ParameterDefinition(
                name="trend_confirmation_bars",
                full_path="strategy.trend.confirm_bars",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.STRATEGY,
                default_value=3.0,
                current_value=3.0,
                min_value=1.0,
                max_value=10.0,
                file_path="strategies/trend_following.py",
            ),
            ParameterDefinition(
                name="trend_min_momentum_pct",
                full_path="strategy.trend.min_momentum",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.01,
                current_value=0.01,
                min_value=0.001,
                max_value=0.05,
                file_path="strategies/trend_following.py",
            ),
            ParameterDefinition(
                name="trend_trailing_stop_atr_mult",
                full_path="strategy.trend.trailing_stop_mult",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.STRATEGY,
                default_value=3.0,
                current_value=3.0,
                min_value=1.5,
                max_value=5.0,
                file_path="strategies/trend_following.py",
            ),
        ]
        
        for param in trend_params:
            self.register_parameter(param)
    
    def _register_strategy_mean_reversion_parameters(self) -> None:
        """Register mean reversion strategy parameters."""
        mr_params = [
            ParameterDefinition(
                name="mr_zscore_entry",
                full_path="strategy.mr.zscore_entry",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=2.0,
                current_value=2.0,
                min_value=1.5,
                max_value=3.0,
                file_path="strategies/mean_reversion.py",
            ),
            ParameterDefinition(
                name="mr_zscore_exit",
                full_path="strategy.mr.zscore_exit",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=0.5,
                current_value=0.5,
                min_value=0.0,
                max_value=1.0,
                file_path="strategies/mean_reversion.py",
            ),
            ParameterDefinition(
                name="mr_lookback_period",
                full_path="strategy.mr.lookback",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.STRATEGY,
                default_value=20.0,
                current_value=20.0,
                min_value=10.0,
                max_value=50.0,
                file_path="strategies/mean_reversion.py",
            ),
            ParameterDefinition(
                name="mr_max_holding_period",
                full_path="strategy.mr.max_hold",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.STRATEGY,
                default_value=10.0,
                current_value=10.0,
                min_value=3.0,
                max_value=30.0,
                file_path="strategies/mean_reversion.py",
            ),
            ParameterDefinition(
                name="mr_min_spread_pct",
                full_path="strategy.mr.min_spread",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.02,
                current_value=0.02,
                min_value=0.005,
                max_value=0.05,
                file_path="strategies/mean_reversion.py",
            ),
        ]
        
        for param in mr_params:
            self.register_parameter(param)
    
    def _register_ml_model_parameters(self) -> None:
        """Register ML model parameters."""
        ml_params = [
            ParameterDefinition(
                name="lstm_hidden_size",
                full_path="ml.lstm.hidden_size",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=128.0,
                current_value=128.0,
                min_value=32.0,
                max_value=512.0,
                file_path="ml/lstm_predictor.py",
            ),
            ParameterDefinition(
                name="lstm_num_layers",
                full_path="ml.lstm.num_layers",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=2.0,
                current_value=2.0,
                min_value=1.0,
                max_value=4.0,
                file_path="ml/lstm_predictor.py",
            ),
            ParameterDefinition(
                name="lstm_dropout",
                full_path="ml.lstm.dropout",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ML,
                default_value=0.2,
                current_value=0.2,
                min_value=0.0,
                max_value=0.5,
                file_path="ml/lstm_predictor.py",
            ),
            ParameterDefinition(
                name="transformer_heads",
                full_path="ml.transformer.heads",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=8.0,
                current_value=8.0,
                min_value=4.0,
                max_value=16.0,
                file_path="ml/transformer_model.py",
            ),
            ParameterDefinition(
                name="transformer_layers",
                full_path="ml.transformer.layers",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=6.0,
                current_value=6.0,
                min_value=2.0,
                max_value=12.0,
                file_path="ml/transformer_model.py",
            ),
            ParameterDefinition(
                name="transformer_d_model",
                full_path="ml.transformer.d_model",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=256.0,
                current_value=256.0,
                min_value=64.0,
                max_value=512.0,
                file_path="ml/transformer_model.py",
            ),
            ParameterDefinition(
                name="xgboost_max_depth",
                full_path="ml.xgboost.max_depth",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=6.0,
                current_value=6.0,
                min_value=3.0,
                max_value=10.0,
                file_path="ml/xgboost_model.py",
            ),
            ParameterDefinition(
                name="xgboost_n_estimators",
                full_path="ml.xgboost.n_estimators",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=100.0,
                current_value=100.0,
                min_value=50.0,
                max_value=500.0,
                file_path="ml/xgboost_model.py",
            ),
            ParameterDefinition(
                name="gnn_hidden_channels",
                full_path="ml.gnn.hidden_channels",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=64.0,
                current_value=64.0,
                min_value=32.0,
                max_value=256.0,
                file_path="ml/gnn_model.py",
            ),
            ParameterDefinition(
                name="gnn_num_layers",
                full_path="ml.gnn.num_layers",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=3.0,
                current_value=3.0,
                min_value=1.0,
                max_value=6.0,
                file_path="ml/gnn_model.py",
            ),
        ]
        
        for param in ml_params:
            self.register_parameter(param)
    
    def _register_ml_training_parameters(self) -> None:
        """Register ML training parameters."""
        training_params = [
            ParameterDefinition(
                name="ml_learning_rate",
                full_path="ml.training.learning_rate",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ML,
                default_value=0.001,
                current_value=0.001,
                min_value=0.0001,
                max_value=0.01,
                file_path="ml/training_config.py",
            ),
            ParameterDefinition(
                name="ml_batch_size",
                full_path="ml.training.batch_size",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=32.0,
                current_value=32.0,
                min_value=16.0,
                max_value=128.0,
                file_path="ml/training_config.py",
            ),
            ParameterDefinition(
                name="ml_weight_decay",
                full_path="ml.training.weight_decay",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ML,
                default_value=0.0001,
                current_value=0.0001,
                min_value=0.0,
                max_value=0.01,
                file_path="ml/training_config.py",
            ),
            ParameterDefinition(
                name="ml_gradient_clip",
                full_path="ml.training.grad_clip",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ML,
                default_value=1.0,
                current_value=1.0,
                min_value=0.1,
                max_value=10.0,
                file_path="ml/training_config.py",
            ),
            ParameterDefinition(
                name="ml_early_stop_patience",
                full_path="ml.training.early_stop",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=10.0,
                current_value=10.0,
                min_value=3.0,
                max_value=30.0,
                file_path="ml/training_config.py",
            ),
            ParameterDefinition(
                name="ml_warmup_epochs",
                full_path="ml.training.warmup_epochs",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=5.0,
                current_value=5.0,
                min_value=1.0,
                max_value=20.0,
                file_path="ml/training_config.py",
            ),
            ParameterDefinition(
                name="ml_ensemble_weight_rf",
                full_path="ml.ensemble.rf_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.ML,
                default_value=0.20,
                current_value=0.20,
                min_value=0.0,
                max_value=0.5,
                file_path="ml/ensemble.py",
            ),
            ParameterDefinition(
                name="ml_ensemble_weight_xgb",
                full_path="ml.ensemble.xgb_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.ML,
                default_value=0.25,
                current_value=0.25,
                min_value=0.0,
                max_value=0.5,
                file_path="ml/ensemble.py",
            ),
            ParameterDefinition(
                name="ml_ensemble_weight_nn",
                full_path="ml.ensemble.nn_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.ML,
                default_value=0.30,
                current_value=0.30,
                min_value=0.0,
                max_value=0.5,
                file_path="ml/ensemble.py",
            ),
            ParameterDefinition(
                name="ml_ensemble_weight_gbm",
                full_path="ml.ensemble.gbm_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.ML,
                default_value=0.25,
                current_value=0.25,
                min_value=0.0,
                max_value=0.5,
                file_path="ml/ensemble.py",
            ),
        ]
        
        for param in training_params:
            self.register_parameter(param)
    
    def _register_volatility_parameters(self) -> None:
        """Register volatility model parameters."""
        vol_params = [
            ParameterDefinition(
                name="garch_omega",
                full_path="vol.garch.omega",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.0001,
                current_value=0.0001,
                min_value=0.00001,
                max_value=0.001,
                file_path="volatility/garch.py",
            ),
            ParameterDefinition(
                name="garch_alpha",
                full_path="vol.garch.alpha",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.1,
                current_value=0.1,
                min_value=0.01,
                max_value=0.3,
                file_path="volatility/garch.py",
            ),
            ParameterDefinition(
                name="garch_beta",
                full_path="vol.garch.beta",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.85,
                current_value=0.85,
                min_value=0.5,
                max_value=0.99,
                file_path="volatility/garch.py",
            ),
            ParameterDefinition(
                name="realized_vol_period",
                full_path="vol.realized.period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.VOLATILITY,
                default_value=20.0,
                current_value=20.0,
                min_value=5.0,
                max_value=60.0,
                file_path="volatility/realized.py",
            ),
            ParameterDefinition(
                name="implied_vol_smile_slope",
                full_path="vol.implied.smile_slope",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.3,
                current_value=0.3,
                min_value=0.1,
                max_value=0.8,
                file_path="volatility/implied.py",
            ),
            ParameterDefinition(
                name="vol_regime_high_threshold",
                full_path="vol.regime.high_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.03,
                current_value=0.03,
                min_value=0.01,
                max_value=0.08,
                file_path="volatility/regime.py",
            ),
            ParameterDefinition(
                name="vol_regime_low_threshold",
                full_path="vol.regime.low_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.01,
                current_value=0.01,
                min_value=0.005,
                max_value=0.02,
                file_path="volatility/regime.py",
            ),
            ParameterDefinition(
                name="vol_scaling_target",
                full_path="vol.scaling.target",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.15,
                current_value=0.15,
                min_value=0.05,
                max_value=0.30,
                file_path="volatility/scaling.py",
            ),
            ParameterDefinition(
                name="vol_forecast_horizon_hours",
                full_path="vol.forecast.horizon",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.VOLATILITY,
                default_value=24.0,
                current_value=24.0,
                min_value=1.0,
                max_value=168.0,
                file_path="volatility/forecast.py",
            ),
            ParameterDefinition(
                name="vol_mean_reversion_speed",
                full_path="vol.mean_rev.speed",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.VOLATILITY,
                default_value=0.1,
                current_value=0.1,
                min_value=0.01,
                max_value=0.5,
                file_path="volatility/mean_reversion.py",
            ),
        ]
        
        for param in vol_params:
            self.register_parameter(param)
    
    def _register_regime_detection_parameters(self) -> None:
        """Register regime detection parameters."""
        regime_params = [
            ParameterDefinition(
                name="regime_n_states",
                full_path="regime.n_states",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.REGIME,
                default_value=4.0,
                current_value=4.0,
                min_value=2.0,
                max_value=6.0,
                file_path="regime/hmm_detector.py",
            ),
            ParameterDefinition(
                name="regime_lookback_days",
                full_path="regime.lookback_days",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.REGIME,
                default_value=30.0,
                current_value=30.0,
                min_value=7.0,
                max_value=90.0,
                file_path="regime/hmm_detector.py",
            ),
            ParameterDefinition(
                name="regime_min_duration_hours",
                full_path="regime.min_duration",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.REGIME,
                default_value=4.0,
                current_value=4.0,
                min_value=1.0,
                max_value=24.0,
                file_path="regime/detector.py",
            ),
            ParameterDefinition(
                name="regime_vol_threshold_high",
                full_path="regime.vol_high",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.REGIME,
                default_value=0.025,
                current_value=0.025,
                min_value=0.01,
                max_value=0.05,
                file_path="regime/vol_regime.py",
            ),
            ParameterDefinition(
                name="regime_vol_threshold_low",
                full_path="regime.vol_low",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.REGIME,
                default_value=0.01,
                current_value=0.01,
                min_value=0.005,
                max_value=0.02,
                file_path="regime/vol_regime.py",
            ),
            ParameterDefinition(
                name="regime_trend_threshold",
                full_path="regime.trend_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.REGIME,
                default_value=0.02,
                current_value=0.02,
                min_value=0.005,
                max_value=0.05,
                file_path="regime/trend_regime.py",
            ),
            ParameterDefinition(
                name="regime_confidence_min",
                full_path="regime.confidence_min",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.REGIME,
                default_value=0.6,
                current_value=0.6,
                min_value=0.4,
                max_value=0.9,
                file_path="regime/detector.py",
            ),
            ParameterDefinition(
                name="regime_transition_cost_weight",
                full_path="regime.transition_cost",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.REGIME,
                default_value=0.1,
                current_value=0.1,
                min_value=0.0,
                max_value=0.5,
                file_path="regime/transition.py",
            ),
        ]
        
        for param in regime_params:
            self.register_parameter(param)
    
    def _register_adaptive_learning_parameters(self) -> None:
        """Register adaptive learning parameters."""
        adaptive_params = [
            ParameterDefinition(
                name="meta_learning_rate",
                full_path="adaptive.meta_lr",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.01,
                current_value=0.01,
                min_value=0.001,
                max_value=0.1,
                file_path="adaptive/meta_learner.py",
            ),
            ParameterDefinition(
                name="forgetting_factor",
                full_path="adaptive.forgetting_factor",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.99,
                current_value=0.99,
                min_value=0.9,
                max_value=0.999,
                file_path="adaptive/forgetting.py",
            ),
            ParameterDefinition(
                name="exploration_decay_rate",
                full_path="adaptive.exploration_decay",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.995,
                current_value=0.995,
                min_value=0.98,
                max_value=0.999,
                file_path="adaptive/exploration.py",
            ),
            ParameterDefinition(
                name="min_exploration_rate",
                full_path="adaptive.min_exploration",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.01,
                current_value=0.01,
                min_value=0.001,
                max_value=0.1,
                file_path="adaptive/exploration.py",
            ),
            ParameterDefinition(
                name="ensemble_temperature",
                full_path="adaptive.ensemble_temp",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=1.0,
                current_value=1.0,
                min_value=0.1,
                max_value=5.0,
                file_path="adaptive/ensemble.py",
            ),
            ParameterDefinition(
                name="confidence_decay_hours",
                full_path="adaptive.confidence_decay",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ENSEMBLE,
                default_value=24.0,
                current_value=24.0,
                min_value=1.0,
                max_value=168.0,
                file_path="adaptive/confidence.py",
            ),
            ParameterDefinition(
                name="strategy_decay_warning_pct",
                full_path="adaptive.decay_warning",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.10,
                current_value=0.10,
                min_value=0.05,
                max_value=0.20,
                file_path="adaptive/decay_detector.py",
            ),
            ParameterDefinition(
                name="strategy_decay_retire_pct",
                full_path="adaptive.decay_retire",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.25,
                current_value=0.25,
                min_value=0.15,
                max_value=0.40,
                file_path="adaptive/decay_detector.py",
            ),
        ]
        
        for param in adaptive_params:
            self.register_parameter(param)
    
    def _register_microstructure_parameters(self) -> None:
        """Register market microstructure parameters."""
        micro_params = [
            ParameterDefinition(
                name="ofi_imbalance_threshold",
                full_path="micro.ofi_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=0.3,
                current_value=0.3,
                min_value=0.1,
                max_value=0.7,
                file_path="alpha/microstructure.py",
            ),
            ParameterDefinition(
                name="spread_threshold_bps",
                full_path="micro.spread_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=5.0,
                current_value=5.0,
                min_value=1.0,
                max_value=20.0,
                file_path="alpha/microstructure.py",
            ),
            ParameterDefinition(
                name="depth_imbalance_threshold",
                full_path="micro.depth_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=0.4,
                current_value=0.4,
                min_value=0.2,
                max_value=0.8,
                file_path="alpha/microstructure.py",
            ),
            ParameterDefinition(
                name="order_flow_toxicity_threshold",
                full_path="micro.vpin_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=0.5,
                current_value=0.5,
                min_value=0.2,
                max_value=0.9,
                file_path="alpha/microstructure.py",
            ),
            ParameterDefinition(
                name="trade_size_outlier_pct",
                full_path="micro.size_outlier",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.SIGNAL,
                default_value=0.01,
                current_value=0.01,
                min_value=0.001,
                max_value=0.05,
                file_path="alpha/microstructure.py",
            ),
            ParameterDefinition(
                name="quote_imbalance_weight",
                full_path="micro.quote_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.3,
                current_value=0.3,
                min_value=0.1,
                max_value=0.6,
                file_path="alpha/microstructure.py",
            ),
        ]
        
        for param in micro_params:
            self.register_parameter(param)
    
    def _register_portfolio_parameters(self) -> None:
        """Register portfolio optimization parameters."""
        portfolio_params = [
            ParameterDefinition(
                name="portfolio_rebalance_threshold",
                full_path="portfolio.rebalance_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.05,
                current_value=0.05,
                min_value=0.01,
                max_value=0.15,
                file_path="portfolio/optimizer.py",
            ),
            ParameterDefinition(
                name="portfolio_target_volatility",
                full_path="portfolio.target_vol",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.15,
                current_value=0.15,
                min_value=0.05,
                max_value=0.30,
                file_path="portfolio/vol_target.py",
            ),
            ParameterDefinition(
                name="portfolio_max_correlation",
                full_path="portfolio.max_corr",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.7,
                current_value=0.7,
                min_value=0.3,
                max_value=0.9,
                file_path="portfolio/diversification.py",
            ),
            ParameterDefinition(
                name="risk_parity_target_risk",
                full_path="portfolio.rp_target_risk",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.10,
                current_value=0.10,
                min_value=0.05,
                max_value=0.20,
                file_path="portfolio/risk_parity.py",
            ),
            ParameterDefinition(
                name="black_litterman_tau",
                full_path="portfolio.bl_tau",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.05,
                current_value=0.05,
                min_value=0.01,
                max_value=0.20,
                file_path="portfolio/black_litterman.py",
            ),
            ParameterDefinition(
                name="hrp_clustering_threshold",
                full_path="portfolio.hrp_cluster",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ENSEMBLE,
                default_value=0.5,
                current_value=0.5,
                min_value=0.2,
                max_value=0.8,
                file_path="portfolio/hrp.py",
            ),
        ]
        
        for param in portfolio_params:
            self.register_parameter(param)
    
    def _register_circuit_breaker_parameters(self) -> None:
        """Register circuit breaker parameters."""
        cb_params = [
            ParameterDefinition(
                name="circuit_breaker_drawdown_pct",
                full_path="risk.cb.drawdown_pct",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.10,
                current_value=0.10,
                min_value=0.05,
                max_value=0.20,
                file_path="risk/circuit_breakers.py",
            ),
            ParameterDefinition(
                name="circuit_breaker_daily_loss_pct",
                full_path="risk.cb.daily_loss_pct",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.03,
                current_value=0.03,
                min_value=0.01,
                max_value=0.08,
                file_path="risk/circuit_breakers.py",
            ),
            ParameterDefinition(
                name="circuit_breaker_cooldown_minutes",
                full_path="risk.cb.cooldown_min",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.RISK,
                default_value=30.0,
                current_value=30.0,
                min_value=5.0,
                max_value=120.0,
                file_path="risk/circuit_breakers.py",
            ),
            ParameterDefinition(
                name="circuit_breaker_max_consecutive_losses",
                full_path="risk.cb.max_consec_losses",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.RISK,
                default_value=5.0,
                current_value=5.0,
                min_value=3.0,
                max_value=10.0,
                file_path="risk/circuit_breakers.py",
            ),
            ParameterDefinition(
                name="circuit_breaker_vol_spike_threshold",
                full_path="risk.cb.vol_spike",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=2.0,
                current_value=2.0,
                min_value=1.5,
                max_value=4.0,
                file_path="risk/circuit_breakers.py",
            ),
            ParameterDefinition(
                name="circuit_breaker_spread_threshold_bps",
                full_path="risk.cb.spread_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=50.0,
                current_value=50.0,
                min_value=20.0,
                max_value=100.0,
                file_path="risk/circuit_breakers.py",
            ),
        ]
        
        for param in cb_params:
            self.register_parameter(param)
    
    def _register_defi_parameters(self) -> None:
        """Register DeFi strategy parameters."""
        defi_params = [
            ParameterDefinition(
                name="defi_min_apy_threshold",
                full_path="defi.min_apy",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.05,
                current_value=0.05,
                min_value=0.02,
                max_value=0.20,
                file_path="defi/yield_optimizer.py",
            ),
            ParameterDefinition(
                name="defi_max_il_pct",
                full_path="defi.max_il",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.05,
                current_value=0.05,
                min_value=0.01,
                max_value=0.15,
                file_path="defi/yield_optimizer.py",
            ),
            ParameterDefinition(
                name="defi_min_liquidity_usd",
                full_path="defi.min_liquidity",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=1000000.0,
                current_value=1000000.0,
                min_value=100000.0,
                max_value=10000000.0,
                file_path="defi/yield_optimizer.py",
            ),
            ParameterDefinition(
                name="defi_rebalance_threshold_pct",
                full_path="defi.rebalance_threshold",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.STRATEGY,
                default_value=0.02,
                current_value=0.02,
                min_value=0.005,
                max_value=0.05,
                file_path="defi/yield_optimizer.py",
            ),
            ParameterDefinition(
                name="defi_max_gas_gwei",
                full_path="defi.max_gas",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.STRATEGY,
                default_value=50.0,
                current_value=50.0,
                min_value=10.0,
                max_value=200.0,
                file_path="defi/yield_optimizer.py",
            ),
            ParameterDefinition(
                name="defi_compound_frequency_hours",
                full_path="defi.compound_freq",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.STRATEGY,
                default_value=24.0,
                current_value=24.0,
                min_value=1.0,
                max_value=168.0,
                file_path="defi/yield_optimizer.py",
            ),
        ]
        
        for param in defi_params:
            self.register_parameter(param)
    
    def _register_on_chain_parameters(self) -> None:
        """Register on-chain analytics parameters."""
        onchain_params = [
            ParameterDefinition(
                name="whale_tx_threshold_usd",
                full_path="onchain.whale_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=1000000.0,
                current_value=1000000.0,
                min_value=100000.0,
                max_value=10000000.0,
                file_path="onchain/whale_tracker.py",
            ),
            ParameterDefinition(
                name="exchange_flow_threshold_usd",
                full_path="onchain.flow_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=5000000.0,
                current_value=5000000.0,
                min_value=1000000.0,
                max_value=50000000.0,
                file_path="onchain/exchange_flow.py",
            ),
            ParameterDefinition(
                name="miner_selling_threshold_pct",
                full_path="onchain.miner_selling",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.SIGNAL,
                default_value=0.10,
                current_value=0.10,
                min_value=0.05,
                max_value=0.30,
                file_path="onchain/miner_tracker.py",
            ),
            ParameterDefinition(
                name="long_term_holder_threshold_days",
                full_path="onchain.lth_threshold",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=155.0,
                current_value=155.0,
                min_value=90.0,
                max_value=365.0,
                file_path="onchain/holder_analysis.py",
            ),
            ParameterDefinition(
                name="nvt_ratio_threshold",
                full_path="onchain.nvt_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=50.0,
                current_value=50.0,
                min_value=20.0,
                max_value=100.0,
                file_path="onchain/nvt.py",
            ),
        ]
        
        for param in onchain_params:
            self.register_parameter(param)
    
    def _register_ml_deep_learning_parameters(self) -> None:
        """Register deep learning model parameters."""
        dl_params = [
            ParameterDefinition(
                name="cnn_kernel_size",
                full_path="ml.cnn.kernel_size",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=3.0,
                current_value=3.0,
                min_value=1.0,
                max_value=7.0,
                file_path="ml/cnn_model.py",
            ),
            ParameterDefinition(
                name="cnn_num_filters",
                full_path="ml.cnn.num_filters",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=32.0,
                current_value=32.0,
                min_value=16.0,
                max_value=128.0,
                file_path="ml/cnn_model.py",
            ),
            ParameterDefinition(
                name="autoencoder_bottleneck",
                full_path="ml.ae.bottleneck",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=16.0,
                current_value=16.0,
                min_value=4.0,
                max_value=64.0,
                file_path="ml/autoencoder.py",
            ),
            ParameterDefinition(
                name="vae_latent_dim",
                full_path="ml.vae.latent_dim",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=32.0,
                current_value=32.0,
                min_value=8.0,
                max_value=128.0,
                file_path="ml/vae.py",
            ),
            ParameterDefinition(
                name="vae_kl_weight",
                full_path="ml.vae.kl_weight",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ML,
                default_value=0.001,
                current_value=0.001,
                min_value=0.0001,
                max_value=0.01,
                file_path="ml/vae.py",
            ),
            ParameterDefinition(
                name="diffusion_timesteps",
                full_path="ml.diffusion.timesteps",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=1000.0,
                current_value=1000.0,
                min_value=100.0,
                max_value=2000.0,
                file_path="ml/diffusion.py",
            ),
            ParameterDefinition(
                name="diffusion_beta_start",
                full_path="ml.diffusion.beta_start",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ML,
                default_value=0.0001,
                current_value=0.0001,
                min_value=0.00001,
                max_value=0.001,
                file_path="ml/diffusion.py",
            ),
            ParameterDefinition(
                name="diffusion_beta_end",
                full_path="ml.diffusion.beta_end",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.ML,
                default_value=0.02,
                current_value=0.02,
                min_value=0.005,
                max_value=0.05,
                file_path="ml/diffusion.py",
            ),
            ParameterDefinition(
                name="attention_heads",
                full_path="ml.attention.heads",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=8.0,
                current_value=8.0,
                min_value=1.0,
                max_value=16.0,
                file_path="ml/attention.py",
            ),
            ParameterDefinition(
                name="attention_dim",
                full_path="ml.attention.dim",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=64.0,
                current_value=64.0,
                min_value=16.0,
                max_value=256.0,
                file_path="ml/attention.py",
            ),
            ParameterDefinition(
                name="gru_hidden_size",
                full_path="ml.gru.hidden_size",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=64.0,
                current_value=64.0,
                min_value=16.0,
                max_value=256.0,
                file_path="ml/gru_model.py",
            ),
            ParameterDefinition(
                name="mlp_hidden_layers",
                full_path="ml.mlp.hidden_layers",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=2.0,
                current_value=2.0,
                min_value=1.0,
                max_value=5.0,
                file_path="ml/mlp_model.py",
            ),
            ParameterDefinition(
                name="mlp_hidden_units",
                full_path="ml.mlp.hidden_units",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.ML,
                default_value=128.0,
                current_value=128.0,
                min_value=32.0,
                max_value=512.0,
                file_path="ml/mlp_model.py",
            ),
        ]
        
        for param in dl_params:
            self.register_parameter(param)
    
    def _register_var_parameters(self) -> None:
        """Register VaR/CVaR risk parameters."""
        var_params = [
            ParameterDefinition(
                name="var_confidence_level",
                full_path="risk.var.confidence",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.99,
                current_value=0.99,
                min_value=0.95,
                max_value=0.999,
                file_path="risk/var_calculator.py",
            ),
            ParameterDefinition(
                name="var_lookback_days",
                full_path="risk.var.lookback",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.RISK,
                default_value=252.0,
                current_value=252.0,
                min_value=60.0,
                max_value=500.0,
                file_path="risk/var_calculator.py",
            ),
            ParameterDefinition(
                name="var_cornish_fisher_adjustment",
                full_path="risk.var.cf_adjust",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=1.0,
                current_value=1.0,
                min_value=0.0,
                max_value=2.0,
                file_path="risk/var_calculator.py",
            ),
            ParameterDefinition(
                name="stress_test_drop_pct",
                full_path="risk.stress.drop_pct",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.20,
                current_value=0.20,
                min_value=0.10,
                max_value=0.50,
                file_path="risk/stress_tester.py",
            ),
            ParameterDefinition(
                name="stress_test_vol_mult",
                full_path="risk.stress.vol_mult",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.RISK,
                default_value=2.0,
                current_value=2.0,
                min_value=1.5,
                max_value=5.0,
                file_path="risk/stress_tester.py",
            ),
            ParameterDefinition(
                name="tail_risk_hedge_ratio",
                full_path="risk.tail.hedge_ratio",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.10,
                current_value=0.10,
                min_value=0.05,
                max_value=0.30,
                file_path="risk/tail_hedger.py",
            ),
            ParameterDefinition(
                name="tail_risk_option_dte",
                full_path="risk.tail.option_dte",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.RISK,
                default_value=30.0,
                current_value=30.0,
                min_value=7.0,
                max_value=90.0,
                file_path="risk/tail_hedger.py",
            ),
            ParameterDefinition(
                name="max_portfolio_var_pct",
                full_path="risk.max_var_pct",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.02,
                current_value=0.02,
                min_value=0.01,
                max_value=0.05,
                file_path="risk/var_manager.py",
            ),
            ParameterDefinition(
                name="max_portfolio_cvar_pct",
                full_path="risk.max_cvar_pct",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.RISK,
                default_value=0.03,
                current_value=0.03,
                min_value=0.015,
                max_value=0.08,
                file_path="risk/var_manager.py",
            ),
            ParameterDefinition(
                name="correlation_stress_threshold",
                full_path="risk.corr_stress_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.8,
                current_value=0.8,
                min_value=0.5,
                max_value=0.95,
                file_path="risk/correlation_manager.py",
            ),
            ParameterDefinition(
                name="liquidity_risk_threshold",
                full_path="risk.liq_threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.RISK,
                default_value=0.5,
                current_value=0.5,
                min_value=0.1,
                max_value=0.9,
                file_path="risk/liquidity_manager.py",
            ),
        ]
        
        for param in var_params:
            self.register_parameter(param)
    
    def _register_alpha_parameters(self) -> None:
        """Register alpha factor parameters."""
        alpha_params = [
            ParameterDefinition(
                name="momentum_lookback_days",
                full_path="alpha.momentum.lookback",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=20.0,
                current_value=20.0,
                min_value=5.0,
                max_value=60.0,
                file_path="alpha/momentum.py",
            ),
            ParameterDefinition(
                name="momentum_skip_days",
                full_path="alpha.momentum.skip",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=1.0,
                current_value=1.0,
                min_value=0.0,
                max_value=5.0,
                file_path="alpha/momentum.py",
            ),
            ParameterDefinition(
                name="reversion_half_life_hours",
                full_path="alpha.reversion.half_life",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=12.0,
                current_value=12.0,
                min_value=1.0,
                max_value=48.0,
                file_path="alpha/reversion.py",
            ),
            ParameterDefinition(
                name="carry_return_annualized_pct",
                full_path="alpha.carry.annualized",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.SIGNAL,
                default_value=0.10,
                current_value=0.10,
                min_value=0.02,
                max_value=0.30,
                file_path="alpha/carry.py",
            ),
            ParameterDefinition(
                name="value_composite_weight",
                full_path="alpha.value.composite",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.3,
                current_value=0.3,
                min_value=0.1,
                max_value=0.6,
                file_path="alpha/value.py",
            ),
            ParameterDefinition(
                name="quality_score_threshold",
                full_path="alpha.quality.threshold",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.SIGNAL,
                default_value=0.6,
                current_value=0.6,
                min_value=0.3,
                max_value=0.9,
                file_path="alpha/quality.py",
            ),
            ParameterDefinition(
                name="sentiment_decay_hours",
                full_path="alpha.sentiment.decay",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=24.0,
                current_value=24.0,
                min_value=1.0,
                max_value=72.0,
                file_path="alpha/sentiment.py",
            ),
            ParameterDefinition(
                name="fundamental_weight_pe",
                full_path="alpha.fundamental.pe_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.25,
                current_value=0.25,
                min_value=0.0,
                max_value=0.5,
                file_path="alpha/fundamental.py",
            ),
            ParameterDefinition(
                name="fundamental_weight_roe",
                full_path="alpha.fundamental.roe_weight",
                parameter_type=ParameterType.WEIGHT,
                category=ParameterCategory.SIGNAL,
                default_value=0.25,
                current_value=0.25,
                min_value=0.0,
                max_value=0.5,
                file_path="alpha/fundamental.py",
            ),
            ParameterDefinition(
                name="flow_momentum_period_hours",
                full_path="alpha.flow.period",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.SIGNAL,
                default_value=4.0,
                current_value=4.0,
                min_value=1.0,
                max_value=24.0,
                file_path="alpha/flow.py",
            ),
        ]
        
        for param in alpha_params:
            self.register_parameter(param)
    
    def _register_backtest_parameters(self) -> None:
        """Register backtesting parameters."""
        bt_params = [
            ParameterDefinition(
                name="backtest_commission_pct",
                full_path="backtest.commission",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.EXECUTION,
                default_value=0.001,
                current_value=0.001,
                min_value=0.0001,
                max_value=0.01,
                file_path="backtest/engine.py",
            ),
            ParameterDefinition(
                name="backtest_slippage_pct",
                full_path="backtest.slippage",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.EXECUTION,
                default_value=0.0005,
                current_value=0.0005,
                min_value=0.0001,
                max_value=0.005,
                file_path="backtest/engine.py",
            ),
            ParameterDefinition(
                name="backtest_spread_pct",
                full_path="backtest.spread",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.EXECUTION,
                default_value=0.0002,
                current_value=0.0002,
                min_value=0.00005,
                max_value=0.001,
                file_path="backtest/engine.py",
            ),
            ParameterDefinition(
                name="backtest_lookahead_bias_check",
                full_path="backtest.lookahead_check",
                parameter_type=ParameterType.THRESHOLD,
                category=ParameterCategory.EXECUTION,
                default_value=1.0,
                current_value=1.0,
                min_value=0.0,
                max_value=1.0,
                file_path="backtest/validation.py",
            ),
            ParameterDefinition(
                name="walk_forward_window_days",
                full_path="backtest.wf_window",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=90.0,
                current_value=90.0,
                min_value=30.0,
                max_value=365.0,
                file_path="backtest/walk_forward.py",
            ),
            ParameterDefinition(
                name="walk_forward_train_pct",
                full_path="backtest.wf_train_pct",
                parameter_type=ParameterType.PERCENTAGE,
                category=ParameterCategory.EXECUTION,
                default_value=0.7,
                current_value=0.7,
                min_value=0.5,
                max_value=0.9,
                file_path="backtest/walk_forward.py",
            ),
        ]
        
        for param in bt_params:
            self.register_parameter(param)
    
    def _register_latency_tier_parameters(self) -> None:
        """Register latency tier parameters."""
        tier_params = [
            ParameterDefinition(
                name="latency_tier_1_max_ms",
                full_path="latency.tier1_max",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=50.0,
                current_value=50.0,
                min_value=10.0,
                max_value=100.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_2_max_ms",
                full_path="latency.tier2_max",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=200.0,
                current_value=200.0,
                min_value=100.0,
                max_value=500.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_3_max_ms",
                full_path="latency.tier3_max",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=500.0,
                current_value=500.0,
                min_value=200.0,
                max_value=1000.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_size_mult_1",
                full_path="latency.tier1_size_mult",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.EXECUTION,
                default_value=1.0,
                current_value=1.0,
                min_value=0.5,
                max_value=2.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_size_mult_2",
                full_path="latency.tier2_size_mult",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.EXECUTION,
                default_value=0.75,
                current_value=0.75,
                min_value=0.3,
                max_value=1.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_size_mult_3",
                full_path="latency.tier3_size_mult",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.EXECUTION,
                default_value=0.5,
                current_value=0.5,
                min_value=0.1,
                max_value=0.8,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_stop_widen_1",
                full_path="latency.tier1_stop_widen",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.EXECUTION,
                default_value=1.0,
                current_value=1.0,
                min_value=0.5,
                max_value=2.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_stop_widen_2",
                full_path="latency.tier2_stop_widen",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.EXECUTION,
                default_value=1.25,
                current_value=1.25,
                min_value=1.0,
                max_value=2.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_tier_stop_widen_3",
                full_path="latency.tier3_stop_widen",
                parameter_type=ParameterType.MULTIPLIER,
                category=ParameterCategory.EXECUTION,
                default_value=1.5,
                current_value=1.5,
                min_value=1.0,
                max_value=3.0,
                file_path="execution/latency_tiers.py",
            ),
            ParameterDefinition(
                name="latency_check_interval_ms",
                full_path="latency.check_interval",
                parameter_type=ParameterType.INTEGER,
                category=ParameterCategory.EXECUTION,
                default_value=1000.0,
                current_value=1000.0,
                min_value=100.0,
                max_value=5000.0,
                file_path="execution/latency_monitor.py",
            ),
        ]
        
        for param in tier_params:
            self.register_parameter(param)
    
    def register_parameter(self, definition: ParameterDefinition) -> None:
        """Register a new learnable parameter."""
        self.parameters[definition.name] = definition
        self.learners[definition.name] = ParameterLearner(definition)
    
    def get_parameter(self, name: str) -> Optional[ParameterDefinition]:
        """Get a parameter definition."""
        return self.parameters.get(name)
    
    def get_learner(self, name: str) -> Optional[ParameterLearner]:
        """Get a parameter learner."""
        return self.learners.get(name)
    
    def get_parameters_by_category(self, category: ParameterCategory) -> List[ParameterDefinition]:
        """Get all parameters in a category."""
        return [p for p in self.parameters.values() if p.category == category]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_category = defaultdict(int)
        by_type = defaultdict(int)
        
        for param in self.parameters.values():
            by_category[param.category.name] += 1
            by_type[param.parameter_type.name] += 1
        
        return {
            "total_parameters": len(self.parameters),
            "by_category": dict(by_category),
            "by_type": dict(by_type),
            "parameters_with_data": sum(
                1 for learner in self.learners.values()
                if len(learner.observations) > 0
            ),
        }


class UniversalParameterLearningEngine:
    """
    Master engine that learns ALL parameters in Argus.
    
    This is the "brain" that:
    1. Observes outcomes from every trade/decision
    2. Attributes outcomes to parameter values
    3. Learns optimal parameters per context
    4. Applies learned parameters via hot-reload
    5. Tracks improvement over baseline
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.registry = ParameterRegistry()
        
        # Current context
        self.current_regime: str = "unknown"
        self.current_asset: str = "BTC"
        
        # Observation buffer
        self.observation_buffer: Deque[Dict[str, Any]] = deque(maxlen=10000)
        
        # Learning state
        self.total_observations: int = 0
        self.total_parameter_updates: int = 0
        self.learning_enabled: bool = True
        
        # Performance tracking
        self.baseline_performance: Dict[str, float] = {}
        self.learned_performance: Dict[str, float] = {}
        
        # Update history
        self.update_history: List[ParameterLearningResult] = []
        
        logger.info("Universal Parameter Learning Engine initialized")
        logger.info("Tracking %d parameters across %d categories",
                    len(self.registry.parameters),
                    len(set(p.category for p in self.registry.parameters.values())))
    
    def update_context(self, regime: str, asset: str) -> None:
        """Update the current trading context."""
        self.current_regime = regime
        self.current_asset = asset
    
    def record_outcome(
        self,
        parameter_values: Dict[str, float],
        outcome: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an outcome for parameter learning.
        
        Args:
            parameter_values: Dict of parameter_name -> value used
            outcome: PnL or other performance metric
            metadata: Additional context
        """
        timestamp = datetime.now()
        
        for param_name, value in parameter_values.items():
            if param_name not in self.registry.parameters:
                continue
            
            learner = self.registry.get_learner(param_name)
            if learner is None:
                continue
            
            observation = ParameterObservation(
                timestamp=timestamp,
                parameter_name=param_name,
                parameter_value=value,
                regime=self.current_regime,
                asset=self.current_asset,
                outcome=outcome,
                metadata=metadata or {}
            )
            
            learner.observe(observation)
        
        self.total_observations += 1
        self.observation_buffer.append({
            "timestamp": timestamp,
            "parameters": parameter_values,
            "outcome": outcome,
            "regime": self.current_regime,
            "asset": self.current_asset,
        })
    
    def learn_parameters(self, min_confidence: float = 0.6) -> List[ParameterLearningResult]:
        """
        Run learning on all parameters and return updates.
        
        Args:
            min_confidence: Minimum confidence to apply learned value
            
        Returns:
            List of parameter updates
        """
        if not self.learning_enabled:
            return []
        
        updates = []
        
        for param_name, learner in self.registry.learners.items():
            # Check if we have enough data
            if len(learner.observations) < learner.definition.min_samples:
                continue
            
            # Get learned value (with regime and asset context)
            learned_value = learner.get_learned_value(
                regime=self.current_regime,
                asset=self.current_asset
            )
            if learned_value is None:
                continue
            
            # Check confidence
            confidence = learner.get_confidence()
            if confidence < min_confidence:
                continue
            
            # Check if different enough from current
            current_value = learner.definition.current_value
            if abs(learned_value - current_value) / (abs(current_value) + 1e-6) < 0.05:
                continue  # Less than 5% change, skip
            
            # Calculate improvement estimate
            improvement = self._estimate_improvement(learner, learned_value)
            
            # Create update
            result = ParameterLearningResult(
                parameter_name=param_name,
                old_value=current_value,
                new_value=learned_value,
                improvement_estimate=improvement,
                confidence=confidence,
                samples_used=len(learner.observations),
                reason=f"Learned from {len(learner.observations)} samples "
                       f"with {confidence:.1%} confidence"
            )
            
            updates.append(result)
            
            # Apply update
            learner.definition.current_value = learned_value
            self.total_parameter_updates += 1
            
            logger.info("Parameter update: %s %.4f → %.4f (improvement: %.2f%%) [%s/%s]",
                       param_name, current_value, learned_value, improvement * 100,
                       self.current_regime, self.current_asset)
        
        self.update_history.extend(updates)
        return updates
    
    def _estimate_improvement(
        self, 
        learner: ParameterLearner, 
        new_value: float
    ) -> float:
        """Estimate improvement from using learned value."""
        # Compare performance of new value vs current
        current_value = learner.definition.current_value
        
        current_outcomes = learner.value_performance.get(
            round(current_value, 4), []
        )
        new_outcomes = learner.value_performance.get(
            round(new_value, 4), []
        )
        
        if not current_outcomes or not new_outcomes:
            return 0.0
        
        current_mean = np.mean(current_outcomes)
        new_mean = np.mean(new_outcomes)
        
        if abs(current_mean) < 1e-6:
            return 0.0
        
        return (new_mean - current_mean) / abs(current_mean)
    
    def get_parameter_value(self, name: str, use_learned: bool = True) -> float:
        """
        Get the current value for a parameter.
        
        Args:
            name: Parameter name
            use_learned: Whether to use learned value if available
            
        Returns:
            Parameter value
        """
        param = self.registry.get_parameter(name)
        if param is None:
            return 0.0
        
        if use_learned:
            learner = self.registry.get_learner(name)
            if learner and len(learner.observations) >= param.min_samples:
                learned = learner.get_learned_value(
                    regime=self.current_regime,
                    asset=self.current_asset
                )
                if learned is not None:
                    return learned
        
        return param.current_value
    
    def get_all_learned_values(self) -> Dict[str, float]:
        """Get all current learned parameter values."""
        values = {}
        
        for name in self.registry.parameters:
            values[name] = self.get_parameter_value(name, use_learned=True)
        
        return values
    
    def get_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report."""
        registry_stats = self.registry.get_statistics()
        
        # Parameters with significant learning
        learned_params = []
        for name, learner in self.registry.learners.items():
            if len(learner.observations) >= learner.definition.min_samples:
                confidence = learner.get_confidence()
                learned_value = learner.get_learned_value(
                    regime=self.current_regime,
                    asset=self.current_asset
                )
                
                if learned_value is not None:
                    learned_params.append({
                        "name": name,
                        "current_value": learner.definition.current_value,
                        "learned_value": learned_value,
                        "confidence": confidence,
                        "observations": len(learner.observations),
                        "asset_stats": learner.get_asset_statistics(),
                        "regime_stats": learner.get_regime_statistics(),
                    })
        
        # Sort by confidence
        learned_params.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "learning_enabled": self.learning_enabled,
            "total_observations": self.total_observations,
            "total_updates": self.total_parameter_updates,
            "current_regime": self.current_regime,
            "current_asset": self.current_asset,
            "registry": registry_stats,
            "learned_parameters": learned_params[:20],  # Top 20
            "recent_updates": [
                {
                    "parameter": r.parameter_name,
                    "old": r.old_value,
                    "new": r.new_value,
                    "improvement": r.improvement_estimate,
                    "confidence": r.confidence,
                }
                for r in self.update_history[-10:]  # Last 10
            ],
        }
    
    def reset_parameter(self, name: str) -> None:
        """Reset a parameter to its default value."""
        param = self.registry.get_parameter(name)
        if param:
            param.current_value = param.default_value
            logger.info("Reset %s to default: %.4f", name, param.default_value)
    
    def reset_all_parameters(self) -> None:
        """Reset all parameters to defaults."""
        for param in self.registry.parameters.values():
            param.current_value = param.default_value
        logger.info("Reset all parameters to defaults")
    
    def export_learned_parameters(self, filepath: str) -> None:
        """Export learned parameters and learner state to a JSON file."""
        data = {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "engine_state": {
                "total_observations": self.total_observations,
                "total_parameter_updates": self.total_parameter_updates,
                "current_regime": self.current_regime,
                "current_asset": self.current_asset,
            },
            "parameters": {},
            "learners": {}
        }
        
        # Export parameter definitions
        for name, param in self.registry.parameters.items():
            data["parameters"][name] = {
                "default_value": param.default_value,
                "current_value": param.current_value,
                "min_value": param.min_value,
                "max_value": param.max_value,
                "category": param.category.name,
                "type": param.parameter_type.name,
            }
        
        # Export learner state (alpha, beta, performance data)
        for name, learner in self.registry.learners.items():
            data["learners"][name] = {
                "alpha": learner.alpha,
                "beta": learner.beta,
                "value_performance": {
                    str(k): v for k, v in learner.value_performance.items()
                },
                "regime_performance": dict(learner.regime_performance),
                "asset_performance": dict(learner.asset_performance),
                "observation_count": len(learner.observations),
                "confidence": learner.get_confidence(),
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Exported %d parameters and %d learner states to %s", 
                    len(data["parameters"]), len(data["learners"]), filepath)
    
    def import_learned_parameters(self, filepath: str) -> int:
        """Import learned parameters and learner state from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        version = data.get("version", "1.0")
        count = 0
        
        # Import parameter values
        for name, param_data in data.get("parameters", {}).items():
            param = self.registry.get_parameter(name)
            if param:
                param.current_value = param_data["current_value"]
                count += 1
        
        # Import learner state (if v2.0+)
        if version >= "2.0":
            for name, learner_data in data.get("learners", {}).items():
                learner = self.registry.get_learner(name)
                if learner:
                    learner.alpha = learner_data.get("alpha", 1.0)
                    learner.beta = learner_data.get("beta", 1.0)
                    
                    # Restore value_performance
                    vp = learner_data.get("value_performance", {})
                    learner.value_performance = {
                        float(k): v for k, v in vp.items()
                    }
                    
                    # Restore regime_performance
                    learner.regime_performance.update(
                        learner_data.get("regime_performance", {})
                    )
                    
                    # Restore asset_performance
                    learner.asset_performance.update(
                        learner_data.get("asset_performance", {})
                    )
        
        # Restore engine state
        engine_state = data.get("engine_state", {})
        self.total_observations = engine_state.get("total_observations", 0)
        self.total_parameter_updates = engine_state.get("total_parameter_updates", 0)
        
        logger.info("Imported %d parameters with full learner state from %s", count, filepath)
        return count
    
    def auto_save(self, save_dir: str = "data/learned_parameters") -> str:
        """Automatically save learned parameters with timestamp."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"learned_params_{timestamp}.json"
        filepath = Path(save_dir) / filename
        
        self.export_learned_parameters(str(filepath))
        
        # Also save a "latest" symlink/copy
        latest_path = Path(save_dir) / "learned_params_latest.json"
        self.export_learned_parameters(str(latest_path))
        
        return str(filepath)
    
    def auto_load(self, save_dir: str = "data/learned_parameters") -> bool:
        """Automatically load the latest learned parameters if available."""
        latest_path = Path(save_dir) / "learned_params_latest.json"
        
        if latest_path.exists():
            try:
                count = self.import_learned_parameters(str(latest_path))
                logger.info("Auto-loaded %d parameters from %s", count, latest_path)
                return True
            except Exception as e:
                logger.error("Failed to auto-load parameters: %s", e)
                return False
        
        logger.info("No saved parameters found at %s", latest_path)
        return False


# Singleton instance
_engine: Optional[UniversalParameterLearningEngine] = None


def get_parameter_learning_engine(
    config: Optional[Dict[str, Any]] = None
) -> UniversalParameterLearningEngine:
    """Get or create the Universal Parameter Learning Engine singleton."""
    global _engine
    if _engine is None:
        _engine = UniversalParameterLearningEngine(config)
    return _engine


def reset_parameter_learning_engine() -> None:
    """Reset the singleton engine (for testing)."""
    global _engine
    _engine = None


__all__ = [
    "UniversalParameterLearningEngine",
    "ParameterRegistry",
    "ParameterLearner",
    "ParameterDefinition",
    "ParameterObservation",
    "ParameterLearningResult",
    "ParameterType",
    "ParameterCategory",
    "get_parameter_learning_engine",
]
