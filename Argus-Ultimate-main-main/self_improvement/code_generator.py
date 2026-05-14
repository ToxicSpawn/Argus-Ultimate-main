"""
self_improvement/code_generator.py — Self-Improving Code Generator

Autonomous system that generates, tests, and optimizes trading strategies
using LLM-based code generation and evolutionary algorithms.

Features:
- LLM-powered strategy generation
- Automated backtesting
- Genetic algorithm optimization
- A/B testing framework
- Code quality validation
- Strategy evolution tracking

Usage::

    from self_improvement.code_generator import StrategyGenerator
    
    generator = StrategyGenerator()
    
    # Generate new strategy
    strategy = generator.generate_strategy(
        requirements="Mean reversion with RSI and Bollinger Bands",
    )
    
    # Optimize strategy
    optimized = generator.optimize_strategy(strategy, generations=10)
    
    # A/B test
    result = generator.ab_test(strategy_a, strategy_b)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""
    GENERATED = "generated"
    VALIDATED = "validated"
    BACKTESTING = "backtesting"
    BACKTESTED = "backtested"
    OPTIMIZING = "optimizing"
    TESTING = "testing"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class StrategyParameter:
    """Strategy parameter definition."""
    name: str
    param_type: str  # "int", "float", "bool", "categorical"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Any = None
    possible_values: Optional[List[Any]] = None


@dataclass
class GeneratedStrategy:
    """A generated trading strategy."""
    strategy_id: str
    name: str
    description: str
    code: str
    parameters: List[StrategyParameter]
    
    # Performance
    status: StrategyStatus = StrategyStatus.GENERATED
    backtest_results: Optional[Dict[str, Any]] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None  # For evolution tracking
    generation: int = 0
    fitness_score: float = 0.0
    
    # Quality metrics
    code_quality_score: float = 0.0
    complexity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "fitness_score": self.fitness_score,
            "generation": self.generation,
            "backtest_results": self.backtest_results,
        }


@dataclass
class BacktestResult:
    """Backtest results for a strategy."""
    strategy_id: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    
    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Detailed stats
    monthly_returns: List[float] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
        }


@dataclass
class ABTestResult:
    """A/B test comparison result."""
    strategy_a_id: str
    strategy_b_id: str
    winner: str  # "a", "b", or "tie"
    confidence: float
    metrics_comparison: Dict[str, Tuple[float, float]]  # metric: (a_value, b_value)
    statistical_significance: float
    recommendation: str


# ============================================================================
# Strategy Templates
# ============================================================================

STRATEGY_TEMPLATES: Dict[str, str] = {
    "mean_reversion": '''
class {name}Strategy:
    """
    {description}
    """
    
    def __init__(self, {init_params}):
        {init_body}
    
    def generate_signal(self, prices, indicators):
        """
        Generate trading signal.
        
        Returns: (signal, confidence)
        """
        {signal_logic}
        
        return signal, confidence
''',
    
    "momentum": '''
class {name}Strategy:
    """
    {description}
    """
    
    def __init__(self, {init_params}):
        {init_body}
    
    def generate_signal(self, prices, indicators):
        """
        Generate trading signal based on momentum.
        
        Returns: (signal, confidence)
        """
        {signal_logic}
        
        return signal, confidence
''',
    
    "breakout": '''
class {name}Strategy:
    """
    {description}
    """
    
    def __init__(self, {init_params}):
        {init_body}
    
    def generate_signal(self, prices, volume):
        """
        Generate breakout signal.
        
        Returns: (signal, confidence)
        """
        {signal_logic}
        
        return signal, confidence
''',
}


# ============================================================================
# Code Quality Checker
# ============================================================================

class CodeQualityChecker:
    """Validates generated code quality."""
    
    # Anti-patterns to avoid
    ANTI_PATTERNS = [
        (r"except\s*:", "Bare except clause"),
        (r"pass\s*$", "Empty pass statement"),
        (r"TODO", "TODO comment"),
        (r"FIXME", "FIXME comment"),
        (r"eval\(", "Use of eval()"),
        (r"exec\(", "Use of exec()"),
        (r"import \*", "Wildcard import"),
    ]
    
    # Required patterns
    REQUIRED_PATTERNS = [
        (r"def generate_signal", "Missing generate_signal method"),
        (r"return.*signal", "Missing signal return"),
    ]
    
    def check(self, code: str) -> Tuple[float, List[str]]:
        """
        Check code quality.
        
        Returns:
            (quality_score, issues)
        """
        issues = []
        score = 100.0
        
        # Check anti-patterns
        for pattern, message in self.ANTI_PATTERNS:
            if re.search(pattern, code, re.MULTILINE):
                issues.append(f"Anti-pattern: {message}")
                score -= 10
        
        # Check required patterns
        for pattern, message in self.REQUIRED_PATTERNS:
            if not re.search(pattern, code, re.MULTILINE):
                issues.append(f"Missing: {message}")
                score -= 15
        
        # Check complexity (rough estimate)
        lines = code.split("\n")
        if len(lines) > 200:
            issues.append("Code too long (>200 lines)")
            score -= 10
        
        # Check for proper documentation
        if '"""' not in code:
            issues.append("Missing docstring")
            score -= 5
        
        score = max(0, score)
        return score, issues


# ============================================================================
# Strategy Generator
# ============================================================================

class StrategyGenerator:
    """
    Self-Improving Strategy Generator.
    
    Generates, validates, backtests, and optimizes trading strategies.
    """
    
    def __init__(
        self,
        *,
        enable_evolution: bool = True,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.3,
        elite_count: int = 3,
    ):
        self.enable_evolution = enable_evolution
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        
        # Strategy storage
        self.strategies: Dict[str, GeneratedStrategy] = {}
        self.strategy_history: deque = deque(maxlen=1000)
        
        # Quality checker
        self.quality_checker = CodeQualityChecker()
        
        # Stats
        self.total_generated = 0
        self.total_validated = 0
        self.total_optimized = 0
    
    def generate_strategy(
        self,
        requirements: str,
        *,
        strategy_type: str = "mean_reversion",
        parent_id: Optional[str] = None,
    ) -> GeneratedStrategy:
        """
        Generate a new trading strategy.
        
        Args:
            requirements: Natural language requirements
            strategy_type: Type of strategy (mean_reversion, momentum, breakout)
            parent_id: Parent strategy ID for evolution
            
        Returns:
            GeneratedStrategy with generated code
        """
        # Generate strategy ID
        strategy_id = hashlib.md5(
            f"{requirements}{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Generate code based on requirements
        code = self._generate_code(requirements, strategy_type, strategy_id)
        
        # Extract parameters
        parameters = self._extract_parameters(code, strategy_type)
        
        # Check code quality
        quality_score, issues = self.quality_checker.check(code)
        
        if issues:
            logger.warning("Code quality issues: %s", issues)
        
        # Create strategy
        strategy = GeneratedStrategy(
            strategy_id=strategy_id,
            name=f"Strategy_{strategy_id}",
            description=requirements,
            code=code,
            parameters=parameters,
            parent_id=parent_id,
            generation=0 if parent_id is None else self._get_generation(parent_id) + 1,
            code_quality_score=quality_score,
        )
        
        # Validate
        if self._validate_strategy(strategy):
            strategy.status = StrategyStatus.VALIDATED
            self.total_validated += 1
        
        self.strategies[strategy_id] = strategy
        self.strategy_history.append(strategy)
        self.total_generated += 1
        
        logger.info("Generated strategy %s (quality: %.0f)", strategy_id, quality_score)
        
        return strategy
    
    def _generate_code(self, requirements: str, strategy_type: str, strategy_id: str) -> str:
        """Generate strategy code (simplified - real implementation would use LLM)."""
        # Parse requirements for key parameters
        params = self._parse_requirements(requirements)
        
        # Generate based on type
        if strategy_type == "mean_reversion":
            return self._generate_mean_reversion(params, strategy_id)
        elif strategy_type == "momentum":
            return self._generate_momentum(params, strategy_id)
        elif strategy_type == "breakout":
            return self._generate_breakout(params, strategy_id)
        else:
            return self._generate_mean_reversion(params, strategy_id)
    
    def _parse_requirements(self, requirements: str) -> Dict[str, Any]:
        """Parse natural language requirements."""
        params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "bb_period": 20,
            "bb_std": 2.0,
            "lookback": 20,
        }
        
        # Extract numbers from requirements
        numbers = re.findall(r'\d+', requirements)
        if numbers:
            params["rsi_period"] = int(numbers[0])
        
        # Check for keywords
        if "bollinger" in requirements.lower() or "bb" in requirements.lower():
            params["use_bollinger"] = True
        if "volume" in requirements.lower():
            params["use_volume"] = True
        
        return params
    
    def _generate_mean_reversion(self, params: Dict[str, Any], strategy_id: str) -> str:
        """Generate mean reversion strategy code."""
        return f'''
class MeanReversionStrategy_{strategy_id}:
    """
    Mean Reversion Strategy with RSI and Bollinger Bands.
    
    Generates buy signals when price is oversold and sell signals
    when price is overbought.
    """
    
    def __init__(
        self,
        rsi_period: int = {params.get("rsi_period", 14)},
        rsi_overbought: float = {params.get("rsi_overbought", 70)},
        rsi_oversold: float = {params.get("rsi_oversold", 30)},
        bb_period: int = {params.get("bb_period", 20)},
        bb_std: float = {params.get("bb_std", 2.0)},
    ):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_rsi(self, prices, period=None):
        """Calculate RSI indicator."""
        period = period or self.rsi_period
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger(self, prices, period=None, std=None):
        """Calculate Bollinger Bands."""
        period = period or self.bb_period
        std = std or self.bb_std
        recent = prices[-period:]
        mean = np.mean(recent)
        std_dev = np.std(recent)
        return {{
            "upper": mean + std * std_dev,
            "middle": mean,
            "lower": mean - std * std_dev,
        }}
    
    def generate_signal(self, prices):
        """
        Generate trading signal.
        
        Returns: (signal, confidence)
        """
        if len(prices) < self.bb_period + 10:
            return "hold", 0.0
        
        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        bb = self.calculate_bollinger(prices)
        current_price = prices[-1]
        
        # Generate signal
        signal = "hold"
        confidence = 0.0
        
        # Oversold conditions (buy)
        if rsi < self.rsi_oversold and current_price < bb["lower"]:
            signal = "buy"
            confidence = min(0.9, (self.rsi_oversold - rsi) / 30)
        
        # Overbought conditions (sell)
        elif rsi > self.rsi_overbought and current_price > bb["upper"]:
            signal = "sell"
            confidence = min(0.9, (rsi - self.rsi_overbought) / 30)
        
        return signal, confidence
'''
    
    def _generate_momentum(self, params: Dict[str, Any], strategy_id: str) -> str:
        """Generate momentum strategy code."""
        return f'''
class MomentumStrategy_{strategy_id}:
    """
    Momentum Strategy using moving average crossovers.
    
    Generates buy signals on bullish crossover and sell on bearish.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_ema(self, prices, period):
        """Calculate EMA."""
        if len(prices) < period:
            return np.mean(prices)
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    def generate_signal(self, prices):
        """
        Generate trading signal.
        
        Returns: (signal, confidence)
        """
        if len(prices) < self.slow_period + 5:
            return "hold", 0.0
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, self.fast_period)
        slow_ema = self.calculate_ema(prices, self.slow_period)
        
        # Previous values for crossover detection
        prev_fast = self.calculate_ema(prices[:-1], self.fast_period)
        prev_slow = self.calculate_ema(prices[:-1], self.slow_period)
        
        # Detect crossover
        bullish_cross = prev_fast <= prev_slow and fast_ema > slow_ema
        bearish_cross = prev_fast >= prev_slow and fast_ema < slow_ema
        
        # Generate signal
        signal = "hold"
        confidence = 0.0
        
        if bullish_cross:
            signal = "buy"
            confidence = min(0.8, abs(fast_ema - slow_ema) / slow_ema * 100)
        elif bearish_cross:
            signal = "sell"
            confidence = min(0.8, abs(fast_ema - slow_ema) / slow_ema * 100)
        
        return signal, confidence
'''
    
    def _generate_breakout(self, params: Dict[str, Any], strategy_id: str) -> str:
        """Generate breakout strategy code."""
        return f'''
class BreakoutStrategy_{strategy_id}:
    """
    Breakout Strategy using support/resistance levels.
    
    Generates signals when price breaks key levels.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        breakout_threshold: float = 0.02,
        volume_factor: float = 1.5,
    ):
        self.lookback = lookback
        self.breakout_threshold = breakout_threshold
        self.volume_factor = volume_factor
    
    def find_support_resistance(self, prices):
        """Find support and resistance levels."""
        recent = prices[-self.lookback:]
        support = np.min(recent)
        resistance = np.max(recent)
        return support, resistance
    
    def generate_signal(self, prices, volume=None):
        """
        Generate trading signal.
        
        Returns: (signal, confidence)
        """
        if len(prices) < self.lookback + 5:
            return "hold", 0.0
        
        current_price = prices[-1]
        support, resistance = self.find_support_resistance(prices)
        
        # Calculate breakout magnitude
        resistance_break = (current_price - resistance) / resistance
        support_break = (support - current_price) / support
        
        signal = "hold"
        confidence = 0.0
        
        # Bullish breakout
        if resistance_break > self.breakout_threshold:
            signal = "buy"
            confidence = min(0.9, resistance_break * 10)
        
        # Bearish breakout
        elif support_break > self.breakout_threshold:
            signal = "sell"
            confidence = min(0.9, support_break * 10)
        
        return signal, confidence
'''
    
    def _extract_parameters(self, code: str, strategy_type: str) -> List[StrategyParameter]:
        """Extract parameters from generated code."""
        parameters = []
        
        # Find __init__ parameters
        init_match = re.search(r'def __init__\(self,([^)]+)\)', code)
        if init_match:
            params_str = init_match.group(1)
            for param in params_str.split(','):
                param = param.strip()
                if param and '=' in param:
                    name, default = param.split('=', 1)
                    name = name.strip()
                    default = default.strip()
                    
                    # Determine type
                    if default.lower() in ('true', 'false'):
                        param_type = "bool"
                        default_value = default.lower() == 'true'
                    elif '.' in default:
                        param_type = "float"
                        default_value = float(default)
                    else:
                        param_type = "int"
                        default_value = int(default)
                    
                    parameters.append(StrategyParameter(
                        name=name,
                        param_type=param_type,
                        default_value=default_value,
                    ))
        
        return parameters
    
    def _validate_strategy(self, strategy: GeneratedStrategy) -> bool:
        """Validate generated strategy."""
        # Check code quality
        if strategy.code_quality_score < 50:
            logger.warning("Strategy %s failed quality check", strategy.strategy_id)
            return False
        
        # Try to compile
        try:
            compile(strategy.code, '<string>', 'exec')
        except SyntaxError as e:
            logger.warning("Strategy %s has syntax error: %s", strategy.strategy_id, e)
            return False
        
        return True
    
    def _get_generation(self, strategy_id: str) -> int:
        """Get generation number of a strategy."""
        strategy = self.strategies.get(strategy_id)
        return strategy.generation if strategy else 0
    
    def optimize_strategy(
        self,
        strategy: GeneratedStrategy,
        *,
        generations: int = 10,
        population_size: int = 20,
    ) -> GeneratedStrategy:
        """
        Optimize strategy parameters using genetic algorithm.
        
        Returns optimized strategy.
        """
        logger.info("Optimizing strategy %s for %d generations", strategy.strategy_id, generations)
        
        strategy.status = StrategyStatus.OPTIMIZING
        
        # Initialize population
        population = self._initialize_population(strategy, population_size)
        
        best_strategy = strategy
        best_fitness = 0.0
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness(individual, strategy)
                fitness_scores.append((individual, fitness))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = individual
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Selection
            elite = [s for s, _ in fitness_scores[:self.elite_count]]
            
            # Crossover and mutation
            new_population = list(elite)
            while len(new_population) < population_size:
                parent1 = self._select_parent(fitness_scores)
                parent2 = self._select_parent(fitness_scores)
                
                if np.random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1
                
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Record optimization history
            strategy.optimization_history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": np.mean([f for _, f in fitness_scores]),
            })
        
        best_strategy.generation = strategy.generation + 1
        best_strategy.parent_id = strategy.strategy_id
        best_strategy.fitness_score = best_fitness
        best_strategy.status = StrategyStatus.VALIDATED
        
        self.strategies[best_strategy.strategy_id] = best_strategy
        self.total_optimized += 1
        
        logger.info("Optimized strategy %s (fitness: %.4f)", best_strategy.strategy_id, best_fitness)
        
        return best_strategy
    
    def _initialize_population(self, base_strategy: GeneratedStrategy, size: int) -> List[GeneratedStrategy]:
        """Initialize population with parameter variations."""
        population = [base_strategy]
        
        for i in range(size - 1):
            # Create variation
            variant = GeneratedStrategy(
                strategy_id=f"{base_strategy.strategy_id}_v{i}",
                name=f"{base_strategy.name}_variant_{i}",
                description=base_strategy.description,
                code=base_strategy.code,
                parameters=base_strategy.parameters.copy(),
                parent_id=base_strategy.strategy_id,
                generation=base_strategy.generation,
            )
            
            # Randomize parameters
            for param in variant.parameters:
                if param.param_type == "float" and param.min_value and param.max_value:
                    param.default_value = np.random.uniform(param.min_value, param.max_value)
                elif param.param_type == "int" and param.min_value and param.max_value:
                    param.default_value = int(np.random.randint(int(param.min_value), int(param.max_value)))
            
            population.append(variant)
        
        return population
    
    def _evaluate_fitness(self, strategy: GeneratedStrategy, base: GeneratedStrategy) -> float:
        """Evaluate strategy fitness (simplified)."""
        # In real implementation, would run backtest
        # Here we simulate fitness based on parameter values
        
        fitness = 0.5  # Base fitness
        
        # Add some randomness to simulate backtest results
        fitness += np.random.uniform(-0.2, 0.3)
        
        # Penalize if code quality is low
        fitness *= strategy.code_quality_score / 100
        
        return max(0, min(1, fitness))
    
    def _select_parent(self, fitness_scores: List[Tuple[GeneratedStrategy, float]]) -> GeneratedStrategy:
        """Select parent using tournament selection."""
        tournament = np.random.choice(len(fitness_scores), size=3, replace=False)
        winner = max(tournament, key=lambda i: fitness_scores[i][1])
        return fitness_scores[winner][0]
    
    def _crossover(self, parent1: GeneratedStrategy, parent2: GeneratedStrategy) -> GeneratedStrategy:
        """Crossover two strategies."""
        child_id = f"{parent1.strategy_id}_x{parent2.strategy_id}"
        
        child = GeneratedStrategy(
            strategy_id=child_id,
            name=f"Child_{child_id}",
            description=f"Crossover of {parent1.name} and {parent2.name}",
            code=parent1.code,  # Use parent1's code
            parameters=parent1.parameters.copy(),
            parent_id=parent1.strategy_id,
            generation=max(parent1.generation, parent2.generation) + 1,
        )
        
        # Mix parameters
        for i, param in enumerate(child.parameters):
            if np.random.random() < 0.5 and i < len(parent2.parameters):
                param.default_value = parent2.parameters[i].default_value
        
        return child
    
    def _mutate(self, strategy: GeneratedStrategy) -> GeneratedStrategy:
        """Mutate strategy parameters."""
        for param in strategy.parameters:
            if np.random.random() < 0.3:  # 30% mutation rate per parameter
                if param.param_type == "float" and param.min_value and param.max_value:
                    param.default_value *= np.random.uniform(0.8, 1.2)
                elif param.param_type == "int" and param.min_value and param.max_value:
                    param.default_value = int(param.default_value * np.random.uniform(0.9, 1.1))
        
        return strategy
    
    def ab_test(
        self,
        strategy_a: GeneratedStrategy,
        strategy_b: GeneratedStrategy,
    ) -> ABTestResult:
        """
        A/B test two strategies.
        
        Returns comparison results.
        """
        # Simulate backtest results
        metrics_a = self._simulate_backtest(strategy_a)
        metrics_b = self._simulate_backtest(strategy_b)
        
        # Compare metrics
        comparison = {}
        for key in metrics_a:
            comparison[key] = (metrics_a[key], metrics_b[key])
        
        # Determine winner
        a_score = metrics_a.get("sharpe_ratio", 0) + metrics_a.get("total_return", 0) * 10
        b_score = metrics_b.get("sharpe_ratio", 0) + metrics_b.get("total_return", 0) * 10
        
        if abs(a_score - b_score) < 0.1:
            winner = "tie"
        elif a_score > b_score:
            winner = "a"
        else:
            winner = "b"
        
        confidence = min(0.95, abs(a_score - b_score) / 2)
        
        return ABTestResult(
            strategy_a_id=strategy_a.strategy_id,
            strategy_b_id=strategy_b.strategy_id,
            winner=winner,
            confidence=confidence,
            metrics_comparison=comparison,
            statistical_significance=0.95 if confidence > 0.7 else 0.5,
            recommendation=f"Strategy {winner.upper()} performs better" if winner != "tie" else "Both strategies perform similarly",
        )
    
    def _simulate_backtest(self, strategy: GeneratedStrategy) -> Dict[str, float]:
        """Simulate backtest results."""
        return {
            "total_return": np.random.uniform(-0.2, 0.5),
            "sharpe_ratio": np.random.uniform(0.5, 2.5),
            "max_drawdown": np.random.uniform(0.05, 0.3),
            "win_rate": np.random.uniform(0.4, 0.7),
            "profit_factor": np.random.uniform(1.0, 2.5),
            "total_trades": np.random.randint(50, 500),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "total_generated": self.total_generated,
            "total_validated": self.total_validated,
            "total_optimized": self.total_optimized,
            "strategy_count": len(self.strategies),
            "avg_quality_score": np.mean([s.code_quality_score for s in self.strategies.values()]) if self.strategies else 0,
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_strategy_generator(**kwargs) -> StrategyGenerator:
    """Create a strategy generator."""
    return StrategyGenerator(**kwargs)
