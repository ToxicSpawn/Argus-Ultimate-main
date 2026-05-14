"""
Meta-Improvement Engine - Argus Ultimate Self-Evolution
=======================================================

The highest level of self-improvement: Argus improves HOW it improves.
This system evolves the learning, adaptation, and prediction mechanisms themselves.

Capabilities:
1. Evolutionary Learning Optimization - Evolves learning algorithms
2. Auto-Strategy Discovery - Creates new strategies automatically
3. Hyper-Parameter Meta-Optimization - Optimizes the optimizers
4. Feature Engineering Automation - Discovers new predictive features
5. Strategy Composition Synthesis - Combines strategies into super-strategies
6. Competitive Evolution - Strategies compete, best survive
7. Self-Modifying Parameters - Parameters that adapt their own adaptation rates
"""

import asyncio
import logging
import time
import numpy as np
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from copy import deepcopy
import json
import hashlib

from strategies.momentum import MomentumStrategy, MomentumConfig
from strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from ml.online_learning import OnlineLearner, AdaptiveLearningManager
from adaptive.enhanced_adaptation import NeuralRegimeDetector
from core.unified_config import config

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """Result of an evolutionary optimization cycle."""
    generation: int
    best_fitness: float
    avg_fitness: float
    improvements: List[str]
    mutations_applied: int
    strategies_evolved: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StrategyGenome:
    """Genetic representation of a strategy for evolution."""
    strategy_type: str
    parameters: Dict[str, float]
    fitness: float = 0.0
    age: int = 0
    mutations: int = 0
    wins: int = 0
    losses: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'type': self.strategy_type,
            'params': self.parameters,
            'fitness': self.fitness,
            'age': self.age,
            'mutations': self.mutations,
            'win_rate': self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0
        }


@dataclass
class LearningConfiguration:
    """Configuration for learning algorithms - evolved over time."""
    learning_rate: float = 0.01
    forgetting_factor: float = 0.99
    regularization: float = 0.001
    adaptation_speed: float = 0.1
    exploration_rate: float = 0.2
    
    # Meta-parameters (how fast these parameters themselves change)
    meta_learning_rate: float = 0.001
    meta_adaptation_speed: float = 0.05


class EvolutionaryStrategyOptimizer:
    """
    Genetic algorithm for evolving strategy parameters.
    Strategies compete, best performers reproduce, worst die off.
    """
    
    def __init__(self, population_size: int = 50):
        self.population_size = population_size
        self.population: List[StrategyGenome] = []
        self.generation = 0
        self.elite_count = 5
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        
        # Performance history
        self.fitness_history: deque = deque(maxlen=100)
        self.evolution_log: List[EvolutionResult] = []
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"EvolutionaryOptimizer initialized with {population_size} genomes")
    
    def _initialize_population(self):
        """Create initial population of strategy genomes."""
        for i in range(self.population_size):
            if i % 2 == 0:
                # Momentum strategies
                genome = StrategyGenome(
                    strategy_type='momentum',
                    parameters={
                        'short_window': random.randint(5, 20),
                        'long_window': random.randint(30, 60),
                        'min_strength': random.uniform(0.001, 0.005),
                        'acceleration_threshold': random.uniform(-0.001, 0.001)
                    }
                )
            else:
                # Mean reversion strategies
                genome = StrategyGenome(
                    strategy_type='mean_reversion',
                    parameters={
                        'lookback': random.randint(30, 100),
                        'base_threshold': random.uniform(1.0, 2.5),
                        'vol_scale': random.uniform(0.5, 1.5)
                    }
                )
            
            self.population.append(genome)
    
    def evaluate_fitness(self, genome: StrategyGenome, 
                        market_data: List[Dict]) -> float:
        """
        Evaluate fitness of a strategy genome on historical data.
        Uses backtesting simulation.
        """
        # Create strategy instance
        if genome.strategy_type == 'momentum':
            strategy = MomentumStrategy(
                short_window=int(genome.parameters['short_window']),
                long_window=int(genome.parameters['long_window']),
                min_strength=genome.parameters['min_strength']
            )
        else:
            strategy = MeanReversionStrategy(
                lookback=int(genome.parameters['lookback']),
                base_threshold=genome.parameters['base_threshold'],
                vol_scale=genome.parameters['vol_scale']
            )
        
        # Simulate trading
        pnl = 0.0
        wins = 0
        losses = 0
        
        for i in range(100, len(market_data)):
            prices = [d['price'] for d in market_data[i-100:i]]
            
            try:
                if genome.strategy_type == 'momentum':
                    signal = strategy.generate(prices)
                else:
                    volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
                    signal = strategy.generate(prices, volatility)
                
                # Simulate trade
                if signal.signal in ['buy', 'sell']:
                    future_return = (market_data[i+1]['price'] - market_data[i]['price']) / market_data[i]['price']
                    
                    if signal.signal == 'buy':
                        trade_pnl = future_return
                    else:
                        trade_pnl = -future_return
                    
                    pnl += trade_pnl
                    if trade_pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                        
            except Exception as e:
                continue
        
        # Calculate fitness
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        profit_factor = abs(pnl) / (wins + losses) if (wins + losses) > 0 else 0
        
        # Fitness = weighted combination
        fitness = (0.4 * win_rate + 0.4 * min(pnl * 100, 10) + 0.2 * min(profit_factor, 5))
        
        genome.fitness = fitness
        genome.wins = wins
        genome.losses = losses
        
        return fitness
    
    def evolve_generation(self, market_data: List[Dict]) -> EvolutionResult:
        """
        Evolve one generation of strategies.
        """
        start_time = time.time()
        
        # Evaluate fitness of all genomes
        for genome in self.population:
            self.evaluate_fitness(genome, market_data)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep elite performers
        elite = self.population[:self.elite_count]
        
        # Create next generation
        new_population = elite.copy()
        
        mutations_applied = 0
        
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = deepcopy(parent1)
            
            # Mutation
            if random.random() < self.mutation_rate:
                self._mutate(child)
                mutations_applied += 1
            
            child.age = 0
            new_population.append(child)
        
        # Age all genomes
        for genome in self.population:
            genome.age += 1
        
        self.population = new_population
        self.generation += 1
        
        # Log results
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([g.fitness for g in self.population])
        
        improvements = self._identify_improvements()
        
        result = EvolutionResult(
            generation=self.generation,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            improvements=improvements,
            mutations_applied=mutations_applied,
            strategies_evolved=len(self.population)
        )
        
        self.evolution_log.append(result)
        self.fitness_history.append(best_fitness)
        
        logger.info(f"Generation {self.generation}: Best={best_fitness:.3f}, "
                   f"Avg={avg_fitness:.3f}, Mutations={mutations_applied}")
        
        return result
    
    def _tournament_select(self, tournament_size: int = 3) -> StrategyGenome:
        """Tournament selection for breeding."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Combine two parent genomes into a child."""
        child = StrategyGenome(
            strategy_type=parent1.strategy_type,
            parameters={}
        )
        
        # Random parameter mixing
        for key in parent1.parameters:
            if key in parent2.parameters:
                # Blend parameters
                alpha = random.random()
                child.parameters[key] = alpha * parent1.parameters[key] + (1 - alpha) * parent2.parameters[key]
            else:
                child.parameters[key] = parent1.parameters[key]
        
        return child
    
    def _mutate(self, genome: StrategyGenome):
        """Apply random mutations to a genome."""
        for key in genome.parameters:
            if random.random() < 0.3:  # 30% chance per parameter
                # Gaussian mutation
                current_value = genome.parameters[key]
                
                if 'window' in key or 'lookback' in key:
                    # Integer parameter
                    mutation = random.gauss(0, 2)
                    new_value = max(5, int(current_value + mutation))
                else:
                    # Float parameter
                    mutation = random.gauss(0, current_value * 0.1)  # 10% std dev
                    new_value = current_value + mutation
                
                genome.parameters[key] = new_value
                genome.mutations += 1
    
    def _identify_improvements(self) -> List[str]:
        """Identify what improved in this generation."""
        improvements = []
        
        if len(self.fitness_history) > 1:
            if self.fitness_history[-1] > self.fitness_history[-2]:
                improvements.append("fitness_increased")
        
        # Check for new best strategy
        best = self.population[0]
        if best.fitness > 0.8:
            improvements.append(f"high_fitness_{best.strategy_type}")
        
        # Check for diversity
        types = set(g.strategy_type for g in self.population)
        if len(types) > 1:
            improvements.append("strategy_diversity_maintained")
        
        return improvements
    
    def get_best_strategies(self, n: int = 5) -> List[StrategyGenome]:
        """Get the top N performing strategies."""
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n]


class AutoFeatureEngineer:
    """
    Automatically discovers new predictive features from market data.
    Uses genetic programming to evolve feature combinations.
    """
    
    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self.features: Dict[str, Callable] = {}
        self.feature_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.discovered_features: List[str] = []
        
        # Base feature set
        self._initialize_base_features()
        
        logger.info(f"AutoFeatureEngineer initialized")
    
    def _initialize_base_features(self):
        """Initialize with basic technical indicators."""
        self.features['returns'] = lambda prices: np.diff(prices) / prices[:-1]
        self.features['volatility'] = lambda prices: np.std(prices)
        self.features['sma_ratio'] = lambda prices: np.mean(prices[-10:]) / np.mean(prices[-50:])
        self.features['momentum'] = lambda prices: (prices[-1] - prices[-10]) / prices[-10]
        self.features['zscore'] = lambda prices: (prices[-1] - np.mean(prices)) / np.std(prices)
    
    def discover_new_features(self, market_data: List[Dict]) -> List[str]:
        """
        Automatically discover new predictive features.
        Uses combinations of existing features and mathematical operations.
        """
        new_features = []
        
        # Feature combination operators
        operators = [
            ('add', lambda a, b: a + b),
            ('subtract', lambda a, b: a - b),
            ('multiply', lambda a, b: a * b),
            ('divide', lambda a, b: a / (b + 1e-9)),
            ('power', lambda a, b: np.power(np.abs(a), b)),
            ('lag', lambda a, b: np.roll(a, int(b))),
        ]
        
        base_feature_names = list(self.features.keys())
        
        # Try random combinations
        for _ in range(20):  # Try 20 new combinations
            feat1 = random.choice(base_feature_names)
            feat2 = random.choice(base_feature_names)
            op_name, operator = random.choice(operators)
            
            # Create feature name
            feature_name = f"{op_name}_{feat1}_{feat2}_{int(time.time()*1000)%1000}"
            
            try:
                # Test the feature
                prices = np.array([d['price'] for d in market_data[-100:]])
                
                feat1_values = self.features[feat1](prices)
                feat2_values = self.features[feat2](prices)
                
                # Ensure arrays are compatible
                min_len = min(len(feat1_values), len(feat2_values))
                combined = operator(feat1_values[-min_len:], feat2_values[-min_len:])
                
                # Check if feature has predictive power
                correlation = self._test_predictive_power(combined, market_data)
                
                if abs(correlation) > 0.1:  # Significant correlation
                    self.features[feature_name] = lambda p, f1=feat1, f2=feat2, op=operator: \
                        op(self.features[f1](p), self.features[f2](p))
                    
                    self.discovered_features.append(feature_name)
                    new_features.append(feature_name)
                    
                    logger.info(f"Discovered new feature: {feature_name} (corr={correlation:.3f})")
                    
            except Exception as e:
                continue
        
        return new_features
    
    def _test_predictive_power(self, feature_values: np.ndarray, 
                                market_data: List[Dict]) -> float:
        """Test if a feature correlates with future returns."""
        try:
            # Future returns
            future_returns = []
            for i in range(len(feature_values)):
                if i < len(market_data) - 1:
                    ret = (market_data[i+1]['price'] - market_data[i]['price']) / market_data[i]['price']
                    future_returns.append(ret)
            
            if len(future_returns) == len(feature_values) and len(feature_values) > 10:
                correlation = np.corrcoef(feature_values[:len(future_returns)], future_returns)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
        except:
            pass
        
        return 0.0
    
    def get_feature_vector(self, market_data: List[Dict]) -> np.ndarray:
        """Get current feature vector for ML models."""
        prices = np.array([d['price'] for d in market_data[-100:]])
        
        features = []
        for name, func in self.features.items():
            try:
                value = func(prices)
                if isinstance(value, np.ndarray):
                    features.append(value[-1] if len(value) > 0 else 0)
                else:
                    features.append(value)
            except:
                features.append(0)
        
        return np.array(features)


class HyperParameterMetaOptimizer:
    """
    Optimizes the learning and adaptation hyper-parameters themselves.
    The system learns how to learn better.
    """
    
    def __init__(self):
        self.learning_configs: List[LearningConfiguration] = []
        self.config_performance: Dict[int, deque] = defaultdict(lambda: deque(maxlen=50))
        self.best_config_index = 0
        
        # Initialize diverse configurations
        self._initialize_configs()
        
        logger.info("HyperParameterMetaOptimizer initialized")
    
    def _initialize_configs(self):
        """Create diverse initial configurations."""
        for i in range(10):
            config = LearningConfiguration(
                learning_rate=10 ** random.uniform(-4, -1),
                forgetting_factor=random.uniform(0.95, 0.999),
                regularization=10 ** random.uniform(-5, -2),
                adaptation_speed=random.uniform(0.01, 0.5),
                exploration_rate=random.uniform(0.05, 0.5),
                meta_learning_rate=10 ** random.uniform(-5, -2),
                meta_adaptation_speed=random.uniform(0.01, 0.2)
            )
            self.learning_configs.append(config)
    
    def evaluate_config_performance(self, config_index: int, 
                                   recent_performance: float) -> float:
        """Evaluate how well a learning configuration is performing."""
        self.config_performance[config_index].append(recent_performance)
        
        # Average recent performance
        avg_perf = np.mean(list(self.config_performance[config_index]))
        return avg_perf
    
    def evolve_configs(self) -> LearningConfiguration:
        """Evolve learning configurations based on performance."""
        # Rank configurations by performance
        ranked = sorted(
            range(len(self.learning_configs)),
            key=lambda i: np.mean(list(self.config_performance[i])) if self.config_performance[i] else 0,
            reverse=True
        )
        
        # Keep top performers
        top_configs = [self.learning_configs[i] for i in ranked[:3]]
        
        # Create new configurations by blending top performers
        new_configs = top_configs.copy()
        
        while len(new_configs) < 10:
            parent1, parent2 = random.sample(top_configs, 2)
            
            child = LearningConfiguration(
                learning_rate=random.choice([parent1.learning_rate, parent2.learning_rate]),
                forgetting_factor=(parent1.forgetting_factor + parent2.forgetting_factor) / 2,
                regularization=(parent1.regularization + parent2.regularization) / 2,
                adaptation_speed=(parent1.adaptation_speed + parent2.adaptation_speed) / 2,
                exploration_rate=(parent1.exploration_rate + parent2.exploration_rate) / 2,
            )
            
            # Add small mutation
            child.learning_rate *= random.uniform(0.9, 1.1)
            child.forgetting_factor = np.clip(child.forgetting_factor + random.gauss(0, 0.001), 0.9, 0.9999)
            
            new_configs.append(child)
        
        self.learning_configs = new_configs
        self.best_config_index = 0
        
        return self.learning_configs[0]
    
    def get_optimal_config(self) -> LearningConfiguration:
        """Get the currently best performing configuration."""
        return self.learning_configs[self.best_config_index]


class StrategyComposer:
    """
    Automatically composes new strategies by combining existing ones.
    Creates "super-strategies" that blend multiple approaches.
    """
    
    def __init__(self):
        self.composite_strategies: Dict[str, Dict] = {}
        self.composition_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("StrategyComposer initialized")
    
    def compose_strategy(self, base_strategies: List[str], 
                        market_regime: str) -> Optional[Dict]:
        """
        Create a new composite strategy for a specific market regime.
        """
        if len(base_strategies) < 2:
            return None
        
        composition_name = f"composite_{'_'.join(base_strategies)}_{market_regime}_{int(time.time())}"
        
        # Define composition weights (will be learned)
        weights = {name: 1.0 / len(base_strategies) for name in base_strategies}
        
        composition = {
            'name': composition_name,
            'components': base_strategies,
            'weights': weights,
            'regime': market_regime,
            'activation_logic': 'weighted_sum',
            'created': datetime.utcnow().isoformat()
        }
        
        self.composite_strategies[composition_name] = composition
        
        logger.info(f"Created composite strategy: {composition_name}")
        
        return composition
    
    def optimize_weights(self, composition_name: str, 
                        performance_history: List[Dict]):
        """
        Optimize the weights of a composite strategy based on performance.
        """
        if composition_name not in self.composite_strategies:
            return
        
        comp = self.composite_strategies[composition_name]
        
        # Calculate each component's contribution to profit
        component_pnl = defaultdict(float)
        
        for trade in performance_history:
            for component in comp['components']:
                if component in trade.get('signals', {}):
                    contribution = trade['pnl'] * trade['signals'][component]
                    component_pnl[component] += contribution
        
        # Update weights proportional to performance
        total_pnl = sum(abs(pnl) for pnl in component_pnl.values())
        
        if total_pnl > 0:
            for component in comp['components']:
                pnl = component_pnl[component]
                # Weight proportional to profit contribution
                new_weight = max(0.1, (pnl + total_pnl / len(comp['components'])) / (2 * total_pnl))
                comp['weights'][component] = new_weight
            
            # Normalize weights to sum to 1
            total_weight = sum(comp['weights'].values())
            for component in comp['weights']:
                comp['weights'][component] /= total_weight
        
        logger.info(f"Optimized weights for {composition_name}: {comp['weights']}")


class MetaImprovementEngine:
    """
    The master controller for all meta-improvement systems.
    Orchestrates evolution, feature discovery, and strategy composition.
    """
    
    def __init__(self):
        self.evolutionary_optimizer = EvolutionaryStrategyOptimizer(population_size=50)
        self.feature_engineer = AutoFeatureEngineer(max_features=100)
        self.hyper_optimizer = HyperParameterMetaOptimizer()
        self.strategy_composer = StrategyComposer()
        
        # Meta-learning state
        self.improvement_cycles = 0
        self.last_improvement_time = time.time()
        self.improvement_interval = 300  # 5 minutes between major improvements
        
        # Performance tracking
        self.system_performance: deque = deque(maxlen=1000)
        
        logger.info("MetaImprovementEngine initialized - Argus can now improve how it improves!")
    
    async def run_improvement_cycle(self, market_data: List[Dict], 
                                   current_performance: float):
        """
        Run one complete meta-improvement cycle.
        Called every 5 minutes or after significant events.
        """
        current_time = time.time()
        
        if current_time - self.last_improvement_time < self.improvement_interval:
            return None
        
        self.last_improvement_time = current_time
        self.improvement_cycles += 1
        
        logger.info(f"=== META-IMPROVEMENT CYCLE #{self.improvement_cycles} ===")
        
        results = {
            'cycle': self.improvement_cycles,
            'timestamp': datetime.utcnow().isoformat(),
            'evolution': None,
            'features': None,
            'hyperparams': None,
            'compositions': []
        }
        
        # 1. Evolve strategies
        try:
            evolution_result = self.evolutionary_optimizer.evolve_generation(market_data)
            results['evolution'] = evolution_result
            logger.info(f"Evolved strategies: Gen {evolution_result.generation}, "
                       f"Best Fitness: {evolution_result.best_fitness:.3f}")
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
        
        # 2. Discover new features
        try:
            new_features = self.feature_engineer.discover_new_features(market_data)
            results['features'] = new_features
            if new_features:
                logger.info(f"Discovered {len(new_features)} new predictive features")
        except Exception as e:
            logger.error(f"Feature discovery failed: {e}")
        
        # 3. Evolve learning hyper-parameters
        try:
            if self.improvement_cycles % 6 == 0:  # Every 30 minutes
                new_config = self.hyper_optimizer.evolve_configs()
                results['hyperparams'] = {
                    'learning_rate': new_config.learning_rate,
                    'adaptation_speed': new_config.adaptation_speed
                }
                logger.info(f"Evolved learning hyperparameters: lr={new_config.learning_rate:.5f}")
        except Exception as e:
            logger.error(f"Hyperparameter evolution failed: {e}")
        
        # 4. Create composite strategies for strong regimes
        try:
            # Find best performing strategies
            best_strategies = self.evolutionary_optimizer.get_best_strategies(3)
            strategy_names = [s.strategy_type for s in best_strategies]
            
            # Create composite
            composite = self.strategy_composer.compose_strategy(
                strategy_names, 
                market_data[-1].get('regime', 'unknown')
            )
            if composite:
                results['compositions'].append(composite['name'])
        except Exception as e:
            logger.error(f"Strategy composition failed: {e}")
        
        # Log results
        self.system_performance.append({
            'cycle': self.improvement_cycles,
            'performance': current_performance,
            'improvements': results
        })
        
        logger.info(f"=== META-IMPROVEMENT CYCLE #{self.improvement_cycles} COMPLETE ===")
        
        return results
    
    def get_best_evolved_strategy(self) -> Optional[StrategyGenome]:
        """Get the best performing evolved strategy."""
        best = self.evolutionary_optimizer.get_best_strategies(1)
        return best[0] if best else None
    
    def get_discovered_features(self) -> List[str]:
        """Get list of automatically discovered features."""
        return self.feature_engineer.discovered_features
    
    def get_optimal_learning_config(self) -> LearningConfiguration:
        """Get the optimal learning configuration."""
        return self.hyper_optimizer.get_optimal_config()
    
    def export_improvements(self) -> Dict:
        """Export all improvements for persistence."""
        return {
            'cycles': self.improvement_cycles,
            'evolution_log': [e.to_dict() if hasattr(e, 'to_dict') else str(e) 
                            for e in self.evolutionary_optimizer.evolution_log],
            'discovered_features': self.feature_engineer.discovered_features,
            'composite_strategies': list(self.strategy_composer.composite_strategies.keys()),
            'timestamp': datetime.utcnow().isoformat()
        }


# Singleton instance
_meta_engine: Optional[MetaImprovementEngine] = None

def get_meta_improvement_engine() -> MetaImprovementEngine:
    """Get or create the singleton MetaImprovementEngine instance."""
    global _meta_engine
    if _meta_engine is None:
        _meta_engine = MetaImprovementEngine()
    return _meta_engine
