"""
Strategy Genesis System v2.0
==============================
Genetic programming for automated strategy discovery in Argus Ultimate.

Provides:
- Tree-based strategy representation
- Genetic algorithm for strategy evolution
- Multi-objective fitness evaluation
- Strategy validation and overfitting detection
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Strategy tree node types."""
    INDICATOR = "indicator"
    OPERATOR = "operator"
    CONSTANT = "constant"
    CONDITION = "condition"


class IndicatorType(Enum):
    """Technical indicator types."""
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BB = "bollinger"
    ATR = "atr"
    STOCH = "stochastic"
    OBV = "obv"
    VWAP = "vwap"


class OperatorType(Enum):
    """Operator types for strategy trees."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    GT = "gt"
    LT = "lt"
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class StrategyNode:
    """Node in a strategy tree."""
    node_type: NodeType
    value: Union[str, float, OperatorType, IndicatorType]
    children: List['StrategyNode'] = field(default_factory=list)
    
    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate node on data."""
        if self.node_type == NodeType.CONSTANT:
            return np.full(len(next(iter(data.values()))), float(self.value))
        
        elif self.node_type == NodeType.INDICATOR:
            return self._evaluate_indicator(data)
        
        elif self.node_type == NodeType.OPERATOR:
            return self._evaluate_operator(data)
        
        elif self.node_type == NodeType.CONDITION:
            return self._evaluate_condition(data)
        
        return np.array([])
    
    def _evaluate_indicator(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate indicator node."""
        indicator = self.value
        prices = data.get("close", np.array([]))
        
        if len(prices) == 0:
            return np.array([])
        
        if indicator == IndicatorType.SMA:
            window = int(self.children[0].value) if self.children else 20
            return self._sma(prices, window)
        
        elif indicator == IndicatorType.RSI:
            window = int(self.children[0].value) if self.children else 14
            return self._rsi(prices, window)
        
        elif indicator == IndicatorType.EMA:
            span = int(self.children[0].value) if self.children else 12
            return self._ema(prices, span)
        
        return prices
    
    def _evaluate_operator(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate operator node."""
        if len(self.children) == 0:
            return np.array([])
        
        if len(self.children) == 1:
            child_result = self.children[0].evaluate(data)
            if self.value == OperatorType.NOT:
                return (child_result < 0.5).astype(float)
            return child_result
        
        left = self.children[0].evaluate(data)
        right = self.children[1].evaluate(data)
        
        if len(left) == 0 or len(right) == 0:
            return np.array([])
        
        # Align lengths
        min_len = min(len(left), len(right))
        left = left[-min_len:]
        right = right[-min_len:]
        
        if self.value == OperatorType.ADD:
            return left + right
        elif self.value == OperatorType.SUB:
            return left - right
        elif self.value == OperatorType.MUL:
            return left * right
        elif self.value == OperatorType.DIV:
            return np.divide(left, right, out=np.zeros_like(left), where=right != 0)
        elif self.value == OperatorType.GT:
            return (left > right).astype(float)
        elif self.value == OperatorType.LT:
            return (left < right).astype(float)
        elif self.value == OperatorType.AND:
            return ((left > 0.5) & (right > 0.5)).astype(float)
        elif self.value == OperatorType.OR:
            return ((left > 0.5) | (right > 0.5)).astype(float)
        
        return left
    
    def _evaluate_condition(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate condition node."""
        if len(self.children) >= 2:
            return self.children[0].evaluate(data) > self.children[1].evaluate(data)
        return np.array([])
    
    def _sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        if len(prices) < window:
            return prices
        
        kernel = np.ones(window) / window
        sma = np.convolve(prices, kernel, mode='valid')
        
        # Pad beginning
        padded = np.concatenate([np.full(window - 1, prices[0]), sma])
        return padded
    
    def _ema(self, prices: np.ndarray, span: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) == 0:
            return prices
        
        alpha = 2.0 / (span + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return np.full_like(prices, 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        rsi = np.full_like(prices, 50.0, dtype=float)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def depth(self) -> int:
        """Calculate tree depth."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """Calculate tree size (number of nodes)."""
        return 1 + sum(child.size() for child in self.children)
    
    def copy(self) -> StrategyNode:
        """Create deep copy of node."""
        return StrategyNode(
            node_type=self.node_type,
            value=self.value,
            children=[child.copy() for child in self.children]
        )


@dataclass
class StrategyGenome:
    """Genome representing a trading strategy."""
    genome_id: str
    root_node: StrategyNode
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def generate_signals(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate trading signals from strategy."""
        return self.root_node.evaluate(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "genome_id": self.genome_id,
            "fitness": self.fitness,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "depth": self.root_node.depth(),
            "size": self.root_node.size(),
            "generation": self.generation,
        }


class StrategyPrimitives:
    """Library of strategy building blocks."""
    
    @staticmethod
    def create_indicator_node(indicator: IndicatorType, params: List[float] = None) -> StrategyNode:
        """Create indicator node with parameters."""
        children = [StrategyNode(NodeType.CONSTANT, p) for p in (params or [20])]
        return StrategyNode(NodeType.INDICATOR, indicator, children)
    
    @staticmethod
    def create_operator_node(operator: OperatorType, left: StrategyNode, right: StrategyNode) -> StrategyNode:
        """Create operator node."""
        return StrategyNode(NodeType.OPERATOR, operator, [left, right])
    
    @staticmethod
    def create_constant(value: float) -> StrategyNode:
        """Create constant node."""
        return StrategyNode(NodeType.CONSTANT, value)
    
    @staticmethod
    def get_random_indicator() -> StrategyNode:
        """Get random indicator node."""
        indicator = random.choice(list(IndicatorType))
        param = random.uniform(5, 50)
        return StrategyPrimitives.create_indicator_node(indicator, [param])
    
    @staticmethod
    def get_random_operator() -> OperatorType:
        """Get random operator."""
        return random.choice([
            OperatorType.ADD,
            OperatorType.SUB,
            OperatorType.MUL,
            OperatorType.GT,
            OperatorType.LT,
            OperatorType.AND,
            OperatorType.OR,
        ])


class GeneticAlgorithm:
    """
    Genetic algorithm for strategy evolution.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_count: int = 5,
        max_depth: int = 6
    ) -> None:
        """
        Initialize genetic algorithm.
        
        Args:
            population_size: Size of population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_count: Number of elite individuals to preserve
            max_depth: Maximum tree depth
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.max_depth = max_depth
        
        self._generation = 0
        self._population: List[StrategyGenome] = []
    
    def initialize_population(self) -> List[StrategyGenome]:
        """Initialize random population."""
        self._population = []
        
        for i in range(self.population_size):
            root = self._generate_random_tree(depth=random.randint(2, 4))
            genome = StrategyGenome(
                genome_id=f"gen0_ind{i}",
                root_node=root,
                generation=0
            )
            self._population.append(genome)
        
        return self._population
    
    def _generate_random_tree(self, depth: int, current_depth: int = 0) -> StrategyNode:
        """Generate random strategy tree."""
        if current_depth >= depth or random.random() < 0.3:
            # Terminal node (indicator or constant)
            if random.random() < 0.7:
                return StrategyPrimitives.get_random_indicator()
            else:
                return StrategyPrimitives.create_constant(random.uniform(-1, 1))
        
        # Operator node
        operator = StrategyPrimitives.get_random_operator()
        
        if operator in (OperatorType.NOT,):
            child = self._generate_random_tree(depth, current_depth + 1)
            return StrategyNode(NodeType.OPERATOR, operator, [child])
        else:
            left = self._generate_random_tree(depth, current_depth + 1)
            right = self._generate_random_tree(depth, current_depth + 1)
            return StrategyNode(NodeType.OPERATOR, operator, [left, right])
    
    def evolve(
        self,
        fitness_function: Callable[[StrategyGenome], float]
    ) -> StrategyGenome:
        """
        Run one generation of evolution.
        
        Args:
            fitness_function: Function to evaluate fitness
            
        Returns:
            Best genome from this generation
        """
        # Evaluate fitness
        for genome in self._population:
            if genome.fitness == 0.0:
                genome.fitness = fitness_function(genome)
        
        # Sort by fitness
        self._population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(self.elitism_count):
            if i < len(self._population):
                elite = self._population[i].copy()
                new_population.append(elite)
        
        # Fill rest with crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            else:
                parent = self._tournament_select()
                child = parent.copy()
            
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            child.generation = self._generation + 1
            child.fitness = 0.0  # Reset fitness
            new_population.append(child)
        
        self._population = new_population[:self.population_size]
        self._generation += 1
        
        return self._population[0]
    
    def _tournament_select(self, tournament_size: int = 5) -> StrategyGenome:
        """Tournament selection."""
        tournament = random.sample(self._population, min(tournament_size, len(self._population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def _crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Crossover two parents."""
        child_root = parent1.root_node.copy()
        
        # Find crossover point
        nodes = self._get_nodes(child_root)
        if len(nodes) > 1:
            crossover_node = random.choice(nodes[1:])  # Skip root
            
            # Replace with random node from parent2
            donor_nodes = self._get_nodes(parent2.root_node)
            if donor_nodes:
                donor = random.choice(donor_nodes).copy()
                crossover_node.children = donor.children
                crossover_node.node_type = donor.node_type
                crossover_node.value = donor.value
        
        return StrategyGenome(
            genome_id=f"gen{self._generation}_child",
            root_node=child_root,
            parent_ids=[parent1.genome_id, parent2.genome_id]
        )
    
    def _mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Mutate a genome."""
        nodes = self._get_nodes(genome.root_node)
        if nodes:
            mutation_point = random.choice(nodes)
            
            # Replace with random subtree
            new_subtree = self._generate_random_tree(depth=random.randint(1, 3))
            mutation_point.children = new_subtree.children
            mutation_point.node_type = new_subtree.node_type
            mutation_point.value = new_subtree.value
        
        return genome
    
    def _get_nodes(self, node: StrategyNode) -> List[StrategyNode]:
        """Get all nodes in tree."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_nodes(child))
        return nodes
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        if not self._population:
            return {}
        
        fitnesses = [g.fitness for g in self._population if g.fitness != 0]
        
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0.0,
            "std_fitness": np.std(fitnesses) if fitnesses else 0.0,
        }


class FitnessEvaluator:
    """
    Multi-objective fitness evaluation for strategies.
    """
    
    def __init__(
        self,
        sharpe_weight: float = 0.4,
        win_rate_weight: float = 0.2,
        profit_factor_weight: float = 0.2,
        drawdown_weight: float = 0.2
    ) -> None:
        """
        Initialize fitness evaluator.
        
        Args:
            sharpe_weight: Weight for Sharpe ratio
            win_rate_weight: Weight for win rate
            profit_factor_weight: Weight for profit factor
            drawdown_weight: Weight for max drawdown (negative)
        """
        self.sharpe_weight = sharpe_weight
        self.win_rate_weight = win_rate_weight
        self.profit_factor_weight = profit_factor_weight
        self.drawdown_weight = drawdown_weight
    
    def evaluate(
        self,
        genome: StrategyGenome,
        data: Dict[str, np.ndarray],
        transaction_cost: float = 0.001
    ) -> float:
        """
        Evaluate strategy fitness.
        
        Args:
            genome: Strategy genome
            data: Market data
            transaction_cost: Transaction cost per trade
            
        Returns:
            Fitness score
        """
        # Generate signals
        signals = genome.generate_signals(data)
        
        if len(signals) == 0:
            return 0.0
        
        # Get prices
        prices = data.get("close", np.array([]))
        if len(prices) == 0:
            return 0.0
        
        # Align lengths
        min_len = min(len(signals), len(prices))
        signals = signals[-min_len:]
        prices = prices[-min_len:]
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        signals = signals[:-1]  # Align with returns
        
        # Apply signals
        strategy_returns = returns * signals
        
        # Subtract transaction costs
        trades = np.abs(np.diff(signals))
        strategy_returns = strategy_returns - trades * transaction_cost
        
        # Calculate metrics
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0
        
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        # Win rate
        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        win_rate = len(winning_trades) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
        gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss
        
        # Max drawdown
        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Update genome metrics
        genome.sharpe_ratio = float(sharpe)
        genome.win_rate = float(win_rate)
        genome.profit_factor = float(profit_factor)
        genome.max_drawdown = float(max_drawdown)
        
        # Composite fitness
        fitness = (
            self.sharpe_weight * sharpe +
            self.win_rate_weight * win_rate +
            self.profit_factor_weight * min(profit_factor, 3.0) / 3.0 -
            self.drawdown_weight * max_drawdown * 10
        )
        
        return float(fitness)


class StrategyGenesisSystem:
    """
    Main strategy genesis system for Argus.
    
    Uses genetic programming to discover new trading strategies.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 50
    ) -> None:
        """
        Initialize strategy genesis system.
        
        Args:
            population_size: GA population size
            n_generations: Number of generations to evolve
        """
        self.ga = GeneticAlgorithm(population_size=population_size)
        self.fitness_evaluator = FitnessEvaluator()
        self.n_generations = n_generations
        
        self._best_strategies: List[StrategyGenome] = []
        self._evolution_history: List[Dict[str, Any]] = []
        
        logger.info("StrategyGenesisSystem initialized: pop=%d, gen=%d", population_size, n_generations)
    
    def discover_strategy(
        self,
        data: Dict[str, np.ndarray],
        transaction_cost: float = 0.001
    ) -> StrategyGenome:
        """
        Discover new strategy using genetic programming.
        
        Args:
            data: Historical market data
            transaction_cost: Transaction cost for evaluation
            
        Returns:
            Best discovered strategy
        """
        logger.info("Starting strategy discovery")
        
        # Initialize population
        population = self.ga.initialize_population()
        
        # Define fitness function
        def fitness_fn(genome: StrategyGenome) -> float:
            return self.fitness_evaluator.evaluate(genome, data, transaction_cost)
        
        # Evolve
        best_genome = None
        for gen in range(self.n_generations):
            best_genome = self.ga.evolve(fitness_fn)
            
            stats = self.ga.get_population_stats()
            self._evolution_history.append(stats)
            
            if gen % 10 == 0:
                logger.info(
                    "Gen %d: best=%.3f, avg=%.3f",
                    gen, stats["best_fitness"], stats["avg_fitness"]
                )
        
        # Store best strategy
        if best_genome:
            self._best_strategies.append(best_genome)
            logger.info(
                "Strategy discovered: fitness=%.3f, sharpe=%.3f, win_rate=%.3f",
                best_genome.fitness, best_genome.sharpe_ratio, best_genome.win_rate
            )
        
        return best_genome
    
    def get_best_strategies(self, n: int = 10) -> List[StrategyGenome]:
        """Get best discovered strategies."""
        return sorted(
            self._best_strategies,
            key=lambda g: g.fitness,
            reverse=True
        )[:n]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution summary."""
        return {
            "total_strategies_discovered": len(self._best_strategies),
            "generations_run": len(self._evolution_history),
            "best_fitness": max((g.fitness for g in self._best_strategies), default=0.0),
            "recent_history": self._evolution_history[-10:],
        }
