"""
Continuous Real-Time Evolution Engine
=====================================

Ultra-fast self-improvement that operates at 0.5-second market speed.
Every tick triggers micro-improvements, creating continuous evolution.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
from copy import deepcopy
import random

from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from core.cache_manager import cache
from core.unified_config import config

logger = logging.getLogger(__name__)


@dataclass
class MicroEvolutionState:
    """State for continuous micro-evolution at tick level."""
    strategy_type: str
    parameters: Dict[str, float]
    fitness_score: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    pnl_accumulated: float = 0.0
    last_updated: float = field(default_factory=time.time)
    generation: int = 0
    mutation_history: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class TickLevelAdaptation:
    """Adaptation applied at every 0.5s tick."""
    parameter_deltas: Dict[str, float]
    confidence: float
    regime: str
    timestamp: float


class RealTimeStrategyEvolver:
    """
    Evolves strategies in real-time at every market tick.
    Uses lightweight genetic operations optimized for 0.5s cycle time.
    """
    
    def __init__(self, max_micro_generations: int = 5):
        self.max_micro_generations = max_micro_generations
        
        # Micro-populations per strategy type (small for speed)
        self.micro_populations: Dict[str, List[MicroEvolutionState]] = {
            'momentum': [],
            'mean_reversion': [],
            'breakout': []
        }
        
        # Current best parameters (what's actually used for trading)
        self.live_parameters: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking per parameter set
        self.performance_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # Adaptation velocity (how fast parameters change)
        self.adaptation_velocity: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Initialize populations
        self._initialize_micro_populations()
        
        logger.info("RealTimeStrategyEvolver initialized for 0.5s evolution")
    
    def _initialize_micro_populations(self):
        """Create small, fast-evolving populations."""
        # Momentum strategies - 3 variants for speed
        for i in range(3):
            self.micro_populations['momentum'].append(MicroEvolutionState(
                strategy_type='momentum',
                parameters={
                    'short_window': 8 + i * 2,  # 8, 10, 12
                    'long_window': 30 + i * 10,  # 30, 40, 50
                    'min_strength': 0.0015 + i * 0.0005,  # 0.0015, 0.002, 0.0025
                },
                generation=0
            ))
        
        # Mean reversion strategies - 3 variants
        for i in range(3):
            self.micro_populations['mean_reversion'].append(MicroEvolutionState(
                strategy_type='mean_reversion',
                parameters={
                    'lookback': 40 + i * 15,  # 40, 55, 70
                    'base_threshold': 1.3 + i * 0.2,  # 1.3, 1.5, 1.7
                    'vol_scale': 0.8 + i * 0.2,  # 0.8, 1.0, 1.2
                },
                generation=0
            ))
        
        # Set initial live parameters to middle variant
        self.live_parameters['momentum'] = self.micro_populations['momentum'][1].parameters.copy()
        self.live_parameters['mean_reversion'] = self.micro_populations['mean_reversion'][1].parameters.copy()
    
    def evolve_on_tick(self, tick_data: Dict, current_regime: str) -> Dict[str, float]:
        """
        Perform one micro-evolution step on every 0.5s tick.
        Returns optimized parameters for current regime.
        
        Total latency target: <20ms
        """
        start_time = time.time()
        
        improvements = {}
        
        # Evolve each strategy type (parallel processing)
        for strategy_type in ['momentum', 'mean_reversion']:
            if strategy_type not in self.micro_populations:
                continue
            
            # 1. Score current population based on recent performance
            self._score_population(strategy_type, tick_data)
            
            # 2. Select best performer
            best = max(self.micro_populations[strategy_type], 
                      key=lambda x: x.fitness_score)
            
            # 3. Create micro-mutation of best (lightweight)
            mutated = self._micro_mutate(best, current_regime)
            
            # 4. Replace worst performer with mutation
            worst_idx = min(range(len(self.micro_populations[strategy_type])),
                          key=lambda i: self.micro_populations[strategy_type][i].fitness_score)
            self.micro_populations[strategy_type][worst_idx] = mutated
            
            # 5. Update live parameters if best improved
            if best.fitness_score > 0.6:  # Threshold for deployment
                old_params = self.live_parameters.get(strategy_type, {})
                self.live_parameters[strategy_type] = self._blend_parameters(
                    old_params, best.parameters, blend_factor=0.1  # Gradual transition
                )
                
                improvements[strategy_type] = best.fitness_score
                
                # Log significant improvements
                if best.generation % 100 == 0:
                    logger.info(f"[{strategy_type}] Gen {best.generation}: "
                               f"fitness={best.fitness_score:.3f}, "
                               f"params={best.parameters}")
        
        elapsed = (time.time() - start_time) * 1000
        if elapsed > 20:
            logger.warning(f"Evolution took {elapsed:.1f}ms (target <20ms)")
        
        return improvements
    
    def _score_population(self, strategy_type: str, tick_data: Dict):
        """Score population based on recent tick data and performance."""
        for state in self.micro_populations[strategy_type]:
            # Calculate fitness from performance buffer
            if len(self.performance_buffer[strategy_type]) > 0:
                recent_perf = list(self.performance_buffer[strategy_type])[-10:]
                
                win_rate = sum(1 for p in recent_perf if p > 0) / len(recent_perf)
                avg_pnl = sum(recent_perf) / len(recent_perf)
                consistency = 1.0 - np.std(recent_perf) / (abs(np.mean(recent_perf)) + 0.001)
                
                # Fitness formula
                state.fitness_score = (
                    0.4 * win_rate +
                    0.3 * min(max(avg_pnl * 100, -1), 1) +
                    0.2 * max(0, consistency) +
                    0.1 * (1.0 / (1.0 + state.generation * 0.001))  # Prefer younger
                )
    
    def _micro_mutate(self, parent: MicroEvolutionState, regime: str) -> MicroEvolutionState:
        """
        Create lightweight mutation for real-time evolution.
        Ultra-fast: <1ms
        """
        child = MicroEvolutionState(
            strategy_type=parent.strategy_type,
            parameters=parent.parameters.copy(),
            generation=parent.generation + 1
        )
        
        # Adaptive mutation size based on regime
        if 'high_vol' in regime:
            mutation_scale = 0.15  # Larger mutations in volatile markets
        elif 'ranging' in regime:
            mutation_scale = 0.05  # Smaller mutations in stable markets
        else:
            mutation_scale = 0.10
        
        # Mutate one parameter at a time (faster)
        param_to_mutate = random.choice(list(child.parameters.keys()))
        current_val = child.parameters[param_to_mutate]
        
        # Gaussian mutation
        mutation = random.gauss(0, current_val * mutation_scale)
        new_val = current_val + mutation
        
        # Enforce bounds
        if 'window' in param_to_mutate or 'lookback' in param_to_mutate:
            new_val = max(5, min(100, int(new_val)))
        elif 'threshold' in param_to_mutate or 'strength' in param_to_mutate:
            new_val = max(0.0001, min(0.1, new_val))
        else:
            new_val = max(0.1, min(2.0, new_val))
        
        child.parameters[param_to_mutate] = new_val
        child.mutation_history.append({
            'param': param_to_mutate,
            'from': current_val,
            'to': new_val,
            'regime': regime
        })
        
        return child
    
    def _blend_parameters(self, old: Dict, new: Dict, blend_factor: float) -> Dict:
        """Blend old and new parameters for smooth transitions."""
        result = {}
        for key in new:
            old_val = old.get(key, new[key])
            result[key] = old_val * (1 - blend_factor) + new[key] * blend_factor
            
            # Integer parameters
            if 'window' in key or 'lookback' in key:
                result[key] = int(round(result[key]))
        
        return result
    
    def record_trade_outcome(self, strategy_type: str, pnl: float, regime: str):
        """Record trade outcome for fitness calculation."""
        self.performance_buffer[strategy_type].append(pnl)
        
        # Also track by regime
        regime_key = f"{strategy_type}_{regime}"
        self.performance_buffer[regime_key].append(pnl)
    
    def get_live_parameters(self, strategy_type: str) -> Dict[str, float]:
        """Get current live-evolved parameters."""
        return self.live_parameters.get(strategy_type, {})


class ContinuousFeatureDiscoverer:
    """
    Discovers new features at every 0.5s tick.
    Uses incremental feature evaluation for speed.
    """
    
    def __init__(self, max_features: int = 50):
        self.max_features = max_features
        self.active_features: Dict[str, Any] = {}
        self.feature_candidates: deque = deque(maxlen=20)
        self.feature_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Base features (always active)
        self.base_features = ['returns', 'volatility', 'sma_ratio', 'momentum']
        
        # Candidate feature pool
        self.candidate_pool = [
            'returns_squared',
            'returns_cubed',
            'volatility_of_returns',
            'price_velocity',
            'price_acceleration',
            'volume_weighted_returns',
            'high_low_range',
            'gap_size',
        ]
        
        self.discovery_interval = 10  # Try new feature every 10 ticks (5 seconds)
        self.tick_counter = 0
        
        logger.info("ContinuousFeatureDiscoverer initialized")
    
    def discover_on_tick(self, tick_data: Dict, prices: List[float]) -> Optional[str]:
        """
        Attempt feature discovery every 10 ticks (5 seconds).
        Ultra-lightweight: <5ms
        """
        self.tick_counter += 1
        
        if self.tick_counter % self.discovery_interval != 0:
            return None
        
        if len(self.active_features) >= self.max_features:
            return None
        
        # Pick random candidate
        if not self.candidate_pool:
            return None
        
        candidate = random.choice(self.candidate_pool)
        
        if candidate in self.active_features:
            return None
        
        # Quick correlation test (last 20 prices)
        if len(prices) < 20:
            return None
        
        try:
            # Calculate candidate feature values
            if candidate == 'returns_squared':
                values = [((prices[i] - prices[i-1]) / prices[i-1]) ** 2 
                         for i in range(1, len(prices))]
            elif candidate == 'price_velocity':
                values = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            elif candidate == 'price_acceleration':
                velocities = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                values = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]
            else:
                return None
            
            if len(values) < 10:
                return None
            
            # Test correlation with future returns
            future_returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                            for i in range(2, min(len(prices), len(values) + 2))]
            
            if len(values) != len(future_returns):
                values = values[-len(future_returns):]
            
            correlation = np.corrcoef(values, future_returns)[0, 1]
            
            if abs(correlation) > 0.15:  # Significant correlation threshold
                self.active_features[candidate] = {
                    'discovery_time': time.time(),
                    'correlation': correlation,
                    'values': values[-10:]  # Keep last 10 for incremental updates
                }
                
                logger.info(f"[FEATURE] Activated: {candidate} (corr={correlation:.3f})")
                return candidate
                
        except Exception as e:
            pass
        
        return None
    
    def get_feature_vector(self, prices: List[float]) -> np.ndarray:
        """Get current feature vector for ML models."""
        features = []
        
        # Base features
        if len(prices) >= 2:
            returns = (prices[-1] - prices[-2]) / prices[-2]
            features.append(returns)
            
            if len(prices) >= 20:
                volatility = np.std([(prices[i] - prices[i-1]) / prices[i-1] 
                                   for i in range(-20, 0)]) if len(prices) >= 20 else 0
                features.append(volatility)
                
                sma_short = np.mean(prices[-10:])
                sma_long = np.mean(prices[-50:])
                features.append(sma_short / sma_long if sma_long > 0 else 1.0)
                
                momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
                features.append(momentum)
        
        # Discovered features
        for name, feature_data in self.active_features.items():
            try:
                # Get last computed value
                if feature_data.get('values'):
                    features.append(feature_data['values'][-1])
                else:
                    features.append(0.0)
            except:
                features.append(0.0)
        
        return np.array(features)
    
    def update_feature_performance(self, feature_name: str, prediction_accuracy: float):
        """Update performance tracking for feature pruning."""
        self.feature_performance[feature_name].append(prediction_accuracy)
        
        # Prune underperforming features
        if len(self.feature_performance[feature_name]) > 50:
            avg_perf = np.mean(list(self.feature_performance[feature_name]))
            if avg_perf < 0.5 and feature_name in self.active_features:
                del self.active_features[feature_name]
                logger.info(f"[FEATURE] Pruned: {feature_name} (low performance)")


class HyperparamContinuousTuner:
    """
    Continuously tunes hyperparameters at every tick.
    Uses gradient-like descent on performance metrics.
    """
    
    def __init__(self):
        # Current hyperparameters
        self.hyperparams = {
            'learning_rate': 0.01,
            'adaptation_speed': 0.1,
            'exploration_rate': 0.2,
            'momentum_factor': 0.9,
        }
        
        # Gradient estimates
        self.gradients = {k: 0.0 for k in self.hyperparams}
        
        # Performance history for gradient estimation
        self.performance_history: deque = deque(maxlen=20)
        
        # Update frequency (every 20 ticks = 10 seconds)
        self.update_interval = 20
        self.tick_count = 0
        
        logger.info("HyperparamContinuousTuner initialized")
    
    def tune_on_tick(self, current_performance: float) -> Optional[Dict]:
        """
        Tune hyperparameters every 20 ticks (10 seconds).
        Lightweight gradient descent on performance.
        """
        self.tick_count += 1
        self.performance_history.append(current_performance)
        
        if self.tick_count % self.update_interval != 0:
            return None
        
        if len(self.performance_history) < 10:
            return None
        
        # Estimate gradients
        recent_perf = list(self.performance_history)[-10:]
        perf_trend = (recent_perf[-1] - recent_perf[0]) / len(recent_perf)
        
        # Update hyperparams based on performance trend
        if perf_trend > 0:
            # Performance improving - increase adaptation speed slightly
            self.hyperparams['adaptation_speed'] = min(
                0.3, 
                self.hyperparams['adaptation_speed'] * 1.05
            )
            self.hyperparams['learning_rate'] = min(
                0.05,
                self.hyperparams['learning_rate'] * 1.02
            )
        else:
            # Performance degrading - decrease adaptation speed
            self.hyperparams['adaptation_speed'] = max(
                0.01,
                self.hyperparams['adaptation_speed'] * 0.95
            )
            self.hyperparams['learning_rate'] = max(
                0.001,
                self.hyperparams['learning_rate'] * 0.98
            )
        
        return self.hyperparams.copy()
    
    def get_current_hyperparams(self) -> Dict[str, float]:
        """Get current hyperparameter values."""
        return self.hyperparams.copy()


class ContinuousEvolutionEngine:
    """
    Master controller for continuous 0.5s evolution.
    Orchestrates all real-time improvement systems.
    """
    
    def __init__(self):
        self.strategy_evolver = RealTimeStrategyEvolver(max_micro_generations=5)
        self.feature_discoverer = ContinuousFeatureDiscoverer(max_features=50)
        self.hyperparam_tuner = HyperparamContinuousTuner()
        
        # Evolution statistics
        self.tick_count = 0
        self.evolution_cycles = 0
        self.features_discovered = 0
        self.last_improvement_time = time.time()
        
        # Performance tracking
        self.current_performance = 0.5
        
        logger.info("ContinuousEvolutionEngine initialized - 0.5s evolution ready")
    
    async def evolve_every_tick(self, tick_data: Dict) -> Dict:
        """
        Main entry point - called every 0.5 seconds.
        Performs all evolution steps in <30ms.
        
        Args:
            tick_data: Current market tick data
            
        Returns:
            Dict with evolution results for this tick
        """
        start_time = time.time()
        
        self.tick_count += 1
        
        results = {
            'tick': self.tick_count,
            'timestamp': time.time(),
            'strategy_improvements': {},
            'new_features': None,
            'hyperparam_updates': None,
            'latency_ms': 0.0
        }
        
        # Extract data
        prices = tick_data.get('price_history', [])
        current_regime = tick_data.get('regime', 'unknown')
        
        # 1. Evolve strategies (<20ms)
        strategy_improvements = self.strategy_evolver.evolve_on_tick(
            tick_data, current_regime
        )
        results['strategy_improvements'] = strategy_improvements
        
        # 2. Discover features (<5ms, every 10 ticks)
        if len(prices) > 20:
            new_feature = self.feature_discoverer.discover_on_tick(tick_data, prices)
            if new_feature:
                results['new_features'] = new_feature
                self.features_discovered += 1
        
        # 3. Tune hyperparameters (<1ms, every 20 ticks)
        hyperparam_updates = self.hyperparam_tuner.tune_on_tick(
            self.current_performance
        )
        if hyperparam_updates:
            results['hyperparam_updates'] = hyperparam_updates
        
        # 4. Get evolved parameters for strategies
        results['live_parameters'] = {
            'momentum': self.strategy_evolver.get_live_parameters('momentum'),
            'mean_reversion': self.strategy_evolver.get_live_parameters('mean_reversion'),
            'hyperparams': self.hyperparam_tuner.get_current_hyperparams()
        }
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        results['latency_ms'] = latency_ms
        
        if self.tick_count % 100 == 0:  # Log every 100 ticks (50 seconds)
            logger.info(f"[EVOLUTION] Tick {self.tick_count}: "
                       f"strategies={len(strategy_improvements)}, "
                       f"features={self.features_discovered}, "
                       f"latency={latency_ms:.1f}ms")
        
        return results
    
    def record_trade(self, strategy_type: str, pnl: float, regime: str):
        """Record trade outcome for evolution scoring."""
        self.strategy_evolver.record_trade_outcome(strategy_type, pnl, regime)
        
        # Update performance metric
        self.current_performance = 0.9 * self.current_performance + 0.1 * (1.0 if pnl > 0 else 0.0)
    
    def get_evolved_strategy(self, strategy_type: str) -> Dict:
        """Get strategy with evolved parameters."""
        params = self.strategy_evolver.get_live_parameters(strategy_type)
        
        if strategy_type == 'momentum':
            return {
                'type': 'momentum',
                'short_window': int(params.get('short_window', 10)),
                'long_window': int(params.get('long_window', 40)),
                'min_strength': params.get('min_strength', 0.002),
                'generation': self.tick_count
            }
        elif strategy_type == 'mean_reversion':
            return {
                'type': 'mean_reversion',
                'lookback': int(params.get('lookback', 50)),
                'base_threshold': params.get('base_threshold', 1.5),
                'vol_scale': params.get('vol_scale', 1.0),
                'generation': self.tick_count
            }
        
        return {}
    
    def get_status(self) -> Dict:
        """Get current evolution status."""
        return {
            'tick_count': self.tick_count,
            'evolution_cycles': self.evolution_cycles,
            'features_discovered': self.features_discovered,
            'current_performance': self.current_performance,
            'live_parameters': {
                'momentum': self.strategy_evolver.get_live_parameters('momentum'),
                'mean_reversion': self.strategy_evolver.get_live_parameters('mean_reversion'),
                'hyperparams': self.hyperparam_tuner.get_current_hyperparams()
            },
            'active_features': list(self.feature_discoverer.active_features.keys())
        }


# Singleton instance
_continuous_engine: Optional[ContinuousEvolutionEngine] = None

def get_continuous_evolution_engine() -> ContinuousEvolutionEngine:
    """Get or create the singleton ContinuousEvolutionEngine."""
    global _continuous_engine
    if _continuous_engine is None:
        _continuous_engine = ContinuousEvolutionEngine()
    return _continuous_engine
