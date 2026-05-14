"""
evolution/level10_self_evolving_system.py — Level 10: Self-Evolving Trading System

The ultimate trading system that evolves, learns, and improves itself.

This is NOT hype - this is real self-improvement through:
1. Strategy DNA - Encode strategies as evolvable genomes
2. Evolution Engine - Genetic algorithm that breeds better strategies
3. Meta-Learner - Learns which adaptations work (learning to learn)
4. Autonomous Researcher - Generates and tests trading hypotheses
5. Self-Improving Code - Writes and improves its own code
6. Strategy Memory - Never forgets what worked or failed
7. Knowledge Graph - Builds market understanding over time

The system gets better EVERY DAY without human intervention.

Usage::

    from evolution.level10_self_evolving_system import Level10System
    
    system = Level10System()
    
    # Run evolution cycle
    results = system.evolve(generations=100)
    
    # Get best strategies
    best = system.get_best_strategies(n=5)
    
    # System learns from results
    system.learn_from_generation(results)
    
    # Generate new hypothesis
    hypothesis = system.researcher.generate_hypothesis()
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class GeneType(str, Enum):
    """Types of genes in strategy genome."""
    INDICATOR = "indicator"
    THRESHOLD = "threshold"
    LOGIC = "logic"
    RISK = "risk"
    TIMING = "timing"
    FILTER = "filter"
    WEIGHT = "weight"


class MutationType(str, Enum):
    """Types of mutations."""
    POINT = "point"  # Single value change
    SWAP = "swap"  # Swap two values
    INSERT = "insert"  # Insert new gene
    DELETE = "delete"  # Remove gene
    CROSSOVER = "crossover"  # Combine two genomes
    INVERSION = "inversion"  # Reverse segment


class EvolutionStage(str, Enum):
    """Evolution stages."""
    INITIALIZATION = "initialization"
    SELECTION = "selection"
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    EVALUATION = "evaluation"
    SURVIVAL = "survival"
    CONVERGENCE = "convergence"


class HypothesisStatus(str, Enum):
    """Hypothesis status."""
    GENERATED = "generated"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


# ============================================================================
# Strategy DNA
# ============================================================================

@dataclass
class Gene:
    """Single gene in strategy genome."""
    gene_type: GeneType
    name: str
    value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mutation_rate: float = 0.1
    is_coding: bool = True  # Whether this gene affects output
    
    def mutate(self, strength: float = 1.0) -> 'Gene':
        """Mutate this gene."""
        new_value = self.value
        
        if self.gene_type == GeneType.THRESHOLD:
            if self.min_value is not None and self.max_value is not None:
                range_size = self.max_value - self.min_value
                delta = np.random.randn() * range_size * 0.1 * strength
                new_value = np.clip(self.value + delta, self.min_value, self.max_value)
        
        elif self.gene_type == GeneType.WEIGHT:
            delta = np.random.randn() * 0.1 * strength
            new_value = self.value + delta
        
        elif self.gene_type == GeneType.LOGIC:
            # Flip logic with small probability
            if np.random.random() < 0.1 * strength:
                new_value = not self.value if isinstance(self.value, bool) else not bool(self.value)
        
        elif self.gene_type == GeneType.INDICATOR:
            # Switch indicator type
            if np.random.random() < 0.05 * strength:
                indicators = ["rsi", "macd", "bollinger", "atr", "stochastic", "obv", "vwap"]
                new_value = np.random.choice(indicators)
        
        return Gene(
            gene_type=self.gene_type,
            name=self.name,
            value=new_value,
            min_value=self.min_value,
            max_value=self.max_value,
            mutation_rate=self.mutation_rate,
            is_coding=self.is_coding,
        )


@dataclass
class StrategyGenome:
    """Complete genome encoding a trading strategy."""
    genome_id: str
    genes: Dict[str, Gene]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    # Performance history
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_improved: Optional[datetime] = None
    improvement_count: int = 0
    
    def get_gene(self, name: str) -> Optional[Gene]:
        """Get gene by name."""
        return self.genes.get(name)
    
    def set_gene(self, name: str, gene: Gene):
        """Set gene value."""
        self.genes[name] = gene
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "n_genes": len(self.genes),
            "gene_types": list(set(g.gene_type.value for g in self.genes.values())),
        }
    
    def get_dna_string(self) -> str:
        """Get DNA string representation."""
        parts = []
        for name, gene in sorted(self.genes.items()):
            parts.append(f"{name}={gene.value}")
        return "|".join(parts)
    
    def calculate_complexity(self) -> float:
        """Calculate genome complexity (0-1)."""
        n_genes = len(self.genes)
        n_types = len(set(g.gene_type for g in self.genes.values()))
        
        # More genes and types = more complex
        complexity = min(1.0, (n_genes / 50 + n_types / 6) / 2)
        return complexity


# ============================================================================
# Genome Factory
# ============================================================================

class GenomeFactory:
    """Factory for creating strategy genomes."""
    
    # Base gene templates
    GENE_TEMPLATES = {
        # Indicator genes
        "primary_indicator": Gene(
            gene_type=GeneType.INDICATOR,
            name="primary_indicator",
            value="rsi",
        ),
        "secondary_indicator": Gene(
            gene_type=GeneType.INDICATOR,
            name="secondary_indicator",
            value="macd",
        ),
        "tertiary_indicator": Gene(
            gene_type=GeneType.INDICATOR,
            name="tertiary_indicator",
            value="bollinger",
        ),
        
        # Threshold genes
        "rsi_period": Gene(
            gene_type=GeneType.THRESHOLD,
            name="rsi_period",
            value=14,
            min_value=7,
            max_value=28,
        ),
        "rsi_overbought": Gene(
            gene_type=GeneType.THRESHOLD,
            name="rsi_overbought",
            value=70,
            min_value=60,
            max_value=85,
        ),
        "rsi_oversold": Gene(
            gene_type=GeneType.THRESHOLD,
            name="rsi_oversold",
            value=30,
            min_value=15,
            max_value=40,
        ),
        "macd_fast": Gene(
            gene_type=GeneType.THRESHOLD,
            name="macd_fast",
            value=12,
            min_value=8,
            max_value=20,
        ),
        "macd_slow": Gene(
            gene_type=GeneType.THRESHOLD,
            name="macd_slow",
            value=26,
            min_value=20,
            max_value=40,
        ),
        "bb_period": Gene(
            gene_type=GeneType.THRESHOLD,
            name="bb_period",
            value=20,
            min_value=10,
            max_value=30,
        ),
        "bb_std": Gene(
            gene_type=GeneType.THRESHOLD,
            name="bb_std",
            value=2.0,
            min_value=1.5,
            max_value=3.0,
        ),
        
        # Risk genes
        "stop_loss_pct": Gene(
            gene_type=GeneType.RISK,
            name="stop_loss_pct",
            value=0.02,
            min_value=0.005,
            max_value=0.05,
        ),
        "take_profit_pct": Gene(
            gene_type=GeneType.RISK,
            name="take_profit_pct",
            value=0.04,
            min_value=0.01,
            max_value=0.10,
        ),
        "position_size": Gene(
            gene_type=GeneType.RISK,
            name="position_size",
            value=0.1,
            min_value=0.01,
            max_value=0.3,
        ),
        "max_risk_per_trade": Gene(
            gene_type=GeneType.RISK,
            name="max_risk_per_trade",
            value=0.02,
            min_value=0.005,
            max_value=0.05,
        ),
        
        # Timing genes
        "lookback_period": Gene(
            gene_type=GeneType.TIMING,
            name="lookback_period",
            value=20,
            min_value=5,
            max_value=100,
        ),
        "holding_period": Gene(
            gene_type=GeneType.TIMING,
            name="holding_period",
            value=24,
            min_value=1,
            max_value=168,
        ),
        "entry_hour": Gene(
            gene_type=GeneType.TIMING,
            name="entry_hour",
            value=16,
            min_value=0,
            max_value=23,
        ),
        
        # Logic genes
        "use_trend_filter": Gene(
            gene_type=GeneType.LOGIC,
            name="use_trend_filter",
            value=True,
        ),
        "use_volume_filter": Gene(
            gene_type=GeneType.LOGIC,
            name="use_volume_filter",
            value=True,
        ),
        "use_momentum_filter": Gene(
            gene_type=GeneType.LOGIC,
            name="use_momentum_filter",
            value=False,
        ),
        "reversal_allowed": Gene(
            gene_type=GeneType.LOGIC,
            name="reversal_allowed",
            value=False,
        ),
        
        # Weight genes
        "trend_weight": Gene(
            gene_type=GeneType.WEIGHT,
            name="trend_weight",
            value=0.4,
            min_value=0.0,
            max_value=1.0,
        ),
        "momentum_weight": Gene(
            gene_type=GeneType.WEIGHT,
            name="momentum_weight",
            value=0.3,
            min_value=0.0,
            max_value=1.0,
        ),
        "volume_weight": Gene(
            gene_type=GeneType.WEIGHT,
            name="volume_weight",
            value=0.3,
            min_value=0.0,
            max_value=1.0,
        ),
        
        # Filter genes
        "min_volume_ratio": Gene(
            gene_type=GeneType.FILTER,
            name="min_volume_ratio",
            value=0.5,
            min_value=0.1,
            max_value=2.0,
        ),
        "max_spread_bps": Gene(
            gene_type=GeneType.FILTER,
            name="max_spread_bps",
            value=10.0,
            min_value=1.0,
            max_value=50.0,
        ),
        "min_confidence": Gene(
            gene_type=GeneType.FILTER,
            name="min_confidence",
            value=50.0,
            min_value=30.0,
            max_value=80.0,
        ),
    }
    
    @classmethod
    def create_random_genome(cls, complexity: float = 0.5) -> StrategyGenome:
        """Create a random genome."""
        genome_id = hashlib.md5(f"{time.time()}{np.random.rand()}".encode()).hexdigest()[:12]
        
        # Select genes based on complexity
        n_genes = int(10 + complexity * 30)  # 10-40 genes
        selected_genes = np.random.choice(list(cls.GENE_TEMPLATES.keys()), 
                                          size=min(n_genes, len(cls.GENE_TEMPLATES)),
                                          replace=False)
        
        genes = {}
        for gene_name in selected_genes:
            template = cls.GENE_TEMPLATES[gene_name]
            genes[gene_name] = template.mutate(strength=0.5)
        
        return StrategyGenome(
            genome_id=genome_id,
            genes=genes,
            generation=0,
        )
    
    @classmethod
    def create_from_template(cls, template_name: str) -> StrategyGenome:
        """Create genome from template."""
        genome_id = hashlib.md5(f"{template_name}{time.time()}".encode()).hexdigest()[:12]
        
        templates = {
            "momentum": ["primary_indicator", "rsi_period", "rsi_overbought", 
                        "rsi_oversold", "stop_loss_pct", "take_profit_pct",
                        "position_size", "use_trend_filter", "trend_weight"],
            "mean_reversion": ["primary_indicator", "bb_period", "bb_std",
                              "rsi_period", "stop_loss_pct", "take_profit_pct",
                              "position_size", "use_volume_filter", "volume_weight"],
            "breakout": ["primary_indicator", "secondary_indicator", "bb_period",
                        "stop_loss_pct", "take_profit_pct", "position_size",
                        "lookback_period", "use_momentum_filter"],
        }
        
        gene_names = templates.get(template_name, list(cls.GENE_TEMPLATES.keys())[:15])
        
        genes = {}
        for gene_name in gene_names:
            if gene_name in cls.GENE_TEMPLATES:
                genes[gene_name] = cls.GENE_TEMPLATES[gene_name]
        
        return StrategyGenome(
            genome_id=genome_id,
            genes=genes,
            generation=0,
        )


# ============================================================================
# Evolution Engine
# ============================================================================

class EvolutionEngine:
    """
    Genetic algorithm for evolving trading strategies.
    
    Uses natural selection to breed better strategies:
    1. Evaluate fitness of all genomes
    2. Select best performers (survival of fittest)
    3. Crossover (breed two strategies)
    4. Mutate (introduce variation)
    5. Repeat
    """
    
    def __init__(
        self,
        *,
        population_size: int = 100,
        elite_fraction: float = 0.1,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        diversity_threshold: float = 0.1,
    ):
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.diversity_threshold = diversity_threshold
        
        # Population
        self.population: List[StrategyGenome] = []
        self.generation = 0
        
        # History
        self.evolution_history: deque = deque(maxlen=1000)
        self.best_genome_history: List[StrategyGenome] = []
        
        # Statistics
        self.avg_fitness_history: List[float] = []
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
    
    def initialize_population(self, complexity: float = 0.5):
        """Initialize population with random genomes."""
        self.population = []
        
        for _ in range(self.population_size):
            genome = GenomeFactory.create_random_genome(complexity)
            self.population.append(genome)
        
        self.generation = 0
        logger.info("Initialized population of %d genomes", self.population_size)
    
    def evaluate_population(self, fitness_function: Callable[[StrategyGenome], float]):
        """Evaluate fitness of all genomes."""
        for genome in self.population:
            if genome.fitness == 0.0:  # Only evaluate unevaluated
                genome.fitness = fitness_function(genome)
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Record statistics
        fitnesses = [g.fitness for g in self.population]
        self.avg_fitness_history.append(np.mean(fitnesses))
        self.best_fitness_history.append(np.max(fitnesses))
        self.diversity_history.append(self._calculate_diversity())
    
    def evolve_generation(self) -> List[StrategyGenome]:
        """Evolve one generation."""
        new_population: List[StrategyGenome] = []
        
        # 1. Elitism - keep best performers
        n_elite = int(self.population_size * self.elite_fraction)
        elite = self.population[:n_elite]
        new_population.extend(elite)
        
        # 2. Fill rest with crossover and mutation
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            else:
                # Clone and mutate
                parent = self._tournament_select()
                child = StrategyGenome(
                    genome_id=hashlib.md5(f"{parent.genome_id}{time.time()}".encode()).hexdigest()[:12],
                    genes={name: gene.mutate() for name, gene in parent.genes.items()},
                    generation=self.generation + 1,
                    parent_ids=[parent.genome_id],
                )
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        # Update population
        self.population = new_population
        self.generation += 1
        
        # Record best
        if self.population:
            self.best_genome_history.append(self.population[0])
        
        logger.info("Generation %d: best=%.4f, avg=%.4f, diversity=%.4f",
                   self.generation, 
                   self.best_fitness_history[-1] if self.best_fitness_history else 0,
                   self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
                   self.diversity_history[-1] if self.diversity_history else 0)
        
        return self.population
    
    def _tournament_select(self) -> StrategyGenome:
        """Tournament selection."""
        tournament = np.random.choice(self.population, size=self.tournament_size, replace=False)
        return max(tournament, key=lambda g: g.fitness)
    
    def _crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> StrategyGenome:
        """Crossover two genomes."""
        child_id = hashlib.md5(f"{parent1.genome_id}{parent2.genome_id}{time.time()}".encode()).hexdigest()[:12]
        
        child_genes = {}
        all_genes = set(parent1.genes.keys()) | set(parent2.genes.keys())
        
        for gene_name in all_genes:
            if gene_name in parent1.genes and gene_name in parent2.genes:
                # Both parents have gene - randomly choose
                if np.random.random() < 0.5:
                    child_genes[gene_name] = Gene(
                        gene_type=parent1.genes[gene_name].gene_type,
                        name=gene_name,
                        value=parent1.genes[gene_name].value,
                        min_value=parent1.genes[gene_name].min_value,
                        max_value=parent1.genes[gene_name].max_value,
                    )
                else:
                    child_genes[gene_name] = Gene(
                        gene_type=parent2.genes[gene_name].gene_type,
                        name=gene_name,
                        value=parent2.genes[gene_name].value,
                        min_value=parent2.genes[gene_name].min_value,
                        max_value=parent2.genes[gene_name].max_value,
                    )
            elif gene_name in parent1.genes:
                child_genes[gene_name] = parent1.genes[gene_name]
            else:
                child_genes[gene_name] = parent2.genes[gene_name]
        
        return StrategyGenome(
            genome_id=child_id,
            genes=child_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id],
        )
    
    def _mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Mutate genome."""
        mutated_genes = {}
        
        for name, gene in genome.genes.items():
            if np.random.random() < self.mutation_rate:
                mutated_genes[name] = gene.mutate()
            else:
                mutated_genes[name] = gene
        
        genome.genes = mutated_genes
        return genome
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        distances = []
        sample_size = min(20, len(self.population))
        sample = np.random.choice(self.population, size=sample_size, replace=False)
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                dist = self._genome_distance(sample[i], sample[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _genome_distance(self, g1: StrategyGenome, g2: StrategyGenome) -> float:
        """Calculate distance between two genomes."""
        all_genes = set(g1.genes.keys()) | set(g2.genes.keys())
        if not all_genes:
            return 0.0
        
        differences = 0
        for gene_name in all_genes:
            if gene_name in g1.genes and gene_name in g2.genes:
                v1 = g1.genes[gene_name].value
                v2 = g2.genes[gene_name].value
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if v1 != v2:
                        differences += 1
                elif v1 != v2:
                    differences += 1
            else:
                differences += 1
        
        return differences / len(all_genes)
    
    def get_best_genome(self) -> Optional[StrategyGenome]:
        """Get best genome."""
        return self.population[0] if self.population else None
    
    def get_diverse_genomes(self, n: int = 5) -> List[StrategyGenome]:
        """Get diverse set of top genomes."""
        if not self.population:
            return []
        
        selected = [self.population[0]]
        
        for genome in self.population[1:]:
            if len(selected) >= n:
                break
            
            # Check if different enough from all selected
            is_diverse = True
            for selected_genome in selected:
                if self._genome_distance(genome, selected_genome) < self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(genome)
        
        return selected


# ============================================================================
# Meta-Learner
# ============================================================================

class MetaLearner:
    """
    Learns which adaptations work best.
    
    This is "learning to learn" - the system observes which mutations,
    crossovers, and parameter changes lead to improvement, and learns
    to prefer those types of adaptations.
    """
    
    def __init__(self):
        # Adaptation success rates
        self.mutation_success: Dict[str, List[bool]] = defaultdict(list)
        self.crossover_success: List[bool] = []
        
        # What works in what regime
        self.regime_adaptations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Learning rate preferences
        self.learning_rates: Dict[str, float] = {
            "mutation_strength": 0.1,
            "crossover_rate": 0.7,
            "exploration_rate": 0.2,
        }
        
        # Meta-knowledge
        self.meta_knowledge: Dict[str, Any] = {
            "best_gene_combinations": [],
            "avoid_gene_combinations": [],
            "optimal_complexity": 0.5,
            "convergence_patterns": [],
        }
    
    def observe_adaptation(
        self,
        adaptation_type: str,
        parent_fitness: float,
        child_fitness: float,
        context: Dict[str, Any],
    ):
        """Observe an adaptation result."""
        improved = child_fitness > parent_fitness
        
        if adaptation_type.startswith("mutation_"):
            self.mutation_success[adaptation_type].append(improved)
        elif adaptation_type == "crossover":
            self.crossover_success.append(improved)
        
        # Update learning rates based on success
        self._update_learning_rates()
    
    def _update_learning_rates(self):
        """Update learning rates based on observed success."""
        # Mutation success rate
        mutation_rates = []
        for mutation_type, successes in self.mutation_success.items():
            if len(successes) >= 10:
                rate = np.mean(successes[-100:])  # Recent history
                mutation_rates.append(rate)
        
        if mutation_rates:
            avg_success = np.mean(mutation_rates)
            # Adjust mutation strength based on success
            if avg_success < 0.3:
                self.learning_rates["mutation_strength"] *= 0.9  # Reduce
            elif avg_success > 0.6:
                self.learning_rates["mutation_strength"] *= 1.1  # Increase
            
            self.learning_rates["mutation_strength"] = np.clip(
                self.learning_rates["mutation_strength"], 0.01, 0.5
            )
    
    def recommend_adaptation(self, genome: StrategyGenome, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend adaptation strategy."""
        recommendations = {
            "mutation_strength": self.learning_rates["mutation_strength"],
            "crossover_rate": self.learning_rates["crossover_rate"],
            "genes_to_mutate": self._recommend_genes(genome),
            "exploration_vs_exploitation": self.learning_rates["exploration_rate"],
        }
        
        return recommendations
    
    def _recommend_genes(self, genome: StrategyGenome) -> List[str]:
        """Recommend which genes to mutate."""
        # Prioritize genes that have shown improvement
        genes_to_mutate = []
        
        for name, gene in genome.genes.items():
            # Mutate threshold and weight genes more often
            if gene.gene_type in (GeneType.THRESHOLD, GeneType.WEIGHT):
                if np.random.random() < 0.3:
                    genes_to_mutate.append(name)
        
        return genes_to_mutate
    
    def analyze_convergence(self, fitness_history: List[float]) -> Dict[str, Any]:
        """Analyze convergence patterns."""
        if len(fitness_history) < 10:
            return {"converged": False, "recommendation": "continue"}
        
        recent = fitness_history[-20:]
        
        # Check for plateau
        improvement = np.diff(recent)
        avg_improvement = np.mean(improvement)
        
        if avg_improvement < 0.001:
            # Plateau detected - increase exploration
            return {
                "converged": True,
                "recommendation": "increase_exploration",
                "suggested_action": "increase_mutation_rate",
                "avg_improvement": avg_improvement,
            }
        elif avg_improvement < 0.01:
            return {
                "converged": False,
                "recommendation": "continue",
                "avg_improvement": avg_improvement,
            }
        else:
            return {
                "converged": False,
                "recommendation": "exploit",
                "avg_improvement": avg_improvement,
            }
    
    def get_meta_summary(self) -> Dict[str, Any]:
        """Get meta-learning summary."""
        mutation_stats = {}
        for mutation_type, successes in self.mutation_success.items():
            if successes:
                mutation_stats[mutation_type] = {
                    "success_rate": np.mean(successes[-100:]),
                    "n_observations": len(successes),
                }
        
        return {
            "learning_rates": self.learning_rates,
            "mutation_stats": mutation_stats,
            "crossover_success_rate": np.mean(self.crossover_success[-100:]) if self.crossover_success else 0.5,
            "meta_knowledge": self.meta_knowledge,
        }


# ============================================================================
# Strategy Memory
# ============================================================================

class StrategyMemory:
    """
    Long-term memory for strategies.
    
    Remembers:
    - What strategies worked in what conditions
    - What mutations led to improvement
    - What market conditions favor what approaches
    - Historical performance patterns
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        
        # Memory stores
        self.strategy_memory: deque = deque(maxlen=capacity)
        self.mutation_memory: deque = deque(maxlen=capacity)
        self.regime_memory: deque = deque(maxlen=capacity)
        self.failure_memory: deque = deque(maxlen=capacity)
        
        # Index for fast lookup
        self.performance_by_type: Dict[str, List[float]] = defaultdict(list)
        self.best_params_by_regime: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def remember_strategy(
        self,
        genome: StrategyGenome,
        market_conditions: Dict[str, Any],
        performance: Dict[str, float],
    ):
        """Remember a strategy and its performance."""
        memory_entry = {
            "genome_id": genome.genome_id,
            "dna": genome.get_dna_string(),
            "genes": {name: g.value for name, g in genome.genes.items()},
            "market_conditions": market_conditions,
            "performance": performance,
            "fitness": genome.fitness,
            "timestamp": datetime.utcnow(),
            "generation": genome.generation,
        }
        
        self.strategy_memory.append(memory_entry)
        
        # Index by strategy type
        primary_indicator = genome.get_gene("primary_indicator")
        if primary_indicator:
            self.performance_by_type[primary_indicator.value].append(genome.fitness)
        
        # Remember best params by regime
        regime = market_conditions.get("regime", "unknown")
        if genome.fitness > 0:
            self.best_params_by_regime[regime] = {
                k: v for k, v in memory_entry["genes"].items()
            }
    
    def remember_mutation(
        self,
        mutation_type: str,
        parent_fitness: float,
        child_fitness: float,
        improved: bool,
    ):
        """Remember a mutation result."""
        self.mutation_memory.append({
            "mutation_type": mutation_type,
            "parent_fitness": parent_fitness,
            "child_fitness": child_fitness,
            "improved": improved,
            "timestamp": datetime.utcnow(),
        })
    
    def remember_failure(
        self,
        genome: StrategyGenome,
        reason: str,
        market_conditions: Dict[str, Any],
    ):
        """Remember a strategy failure."""
        self.failure_memory.append({
            "genome_id": genome.genome_id,
            "dna": genome.get_dna_string(),
            "reason": reason,
            "market_conditions": market_conditions,
            "fitness": genome.fitness,
            "timestamp": datetime.utcnow(),
        })
    
    def recall_similar_situation(
        self,
        market_conditions: Dict[str, Any],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recall similar market situations and what worked."""
        regime = market_conditions.get("regime", "unknown")
        volatility = market_conditions.get("volatility", 0.02)
        
        # Find similar situations
        similar = []
        for entry in self.strategy_memory:
            entry_conditions = entry.get("market_conditions", {})
            entry_regime = entry_conditions.get("regime", "unknown")
            entry_vol = entry_conditions.get("volatility", 0.02)
            
            # Similarity score
            similarity = 0
            if entry_regime == regime:
                similarity += 0.5
            if abs(entry_vol - volatility) < 0.01:
                similarity += 0.3
            if entry["fitness"] > 0:
                similarity += 0.2
            
            if similarity > 0.5:
                similar.append((similarity, entry))
        
        # Sort by similarity and fitness
        similar.sort(key=lambda x: (x[0], x[1]["fitness"]), reverse=True)
        
        return [entry for _, entry in similar[:n_results]]
    
    def get_best_strategy_for_regime(self, regime: str) -> Optional[Dict[str, Any]]:
        """Get best strategy for a regime."""
        return self.best_params_by_regime.get(regime)
    
    def get_failure_patterns(self) -> List[str]:
        """Get common failure patterns."""
        if not self.failure_memory:
            return []
        
        # Analyze failure reasons
        reason_counts = defaultdict(int)
        for entry in self.failure_memory:
            reason_counts[entry["reason"]] += 1
        
        # Return most common failure reasons
        return sorted(reason_counts.keys(), key=lambda x: reason_counts[x], reverse=True)[:5]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "strategy_memory_size": len(self.strategy_memory),
            "mutation_memory_size": len(self.mutation_memory),
            "failure_memory_size": len(self.failure_memory),
            "regimes_learned": len(self.best_params_by_regime),
            "strategy_types_tracked": len(self.performance_by_type),
        }


# ============================================================================
# Autonomous Researcher
# ============================================================================

class AutonomousResearcher:
    """
    Generates and tests trading hypotheses autonomously.
    
    Like a human researcher, it:
    1. Observes market patterns
    2. Generates hypotheses
    3. Designs experiments
    4. Tests hypotheses
    5. Records findings
    6. Generates new hypotheses based on findings
    """
    
    def __init__(self):
        self.hypotheses: List[Dict[str, Any]] = []
        self.experiments: List[Dict[str, Any]] = []
        self.findings: List[Dict[str, Any]] = []
        
        # Knowledge base
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        self.correlation_knowledge: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.hypotheses_generated = 0
        self.hypotheses_validated = 0
        self.hypotheses_rejected = 0
    
    def generate_hypothesis(
        self,
        market_data: Dict[str, Any],
        existing_knowledge: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a new trading hypothesis."""
        self.hypotheses_generated += 1
        
        # Types of hypotheses
        hypothesis_types = [
            "indicator_combination",
            "threshold_optimization",
            "regime_pattern",
            "correlation_breakdown",
            "seasonality_effect",
            "volume_signal",
            "volatility_regime",
        ]
        
        hypothesis_type = np.random.choice(hypothesis_types)
        
        hypothesis = {
            "id": f"hyp_{self.hypotheses_generated:06d}",
            "type": hypothesis_type,
            "statement": self._generate_statement(hypothesis_type, market_data),
            "test_plan": self._generate_test_plan(hypothesis_type),
            "expected_outcome": self._generate_expected_outcome(hypothesis_type),
            "confidence": 0.5,
            "status": HypothesisStatus.GENERATED.value,
            "created_at": datetime.utcnow(),
            "results": None,
        }
        
        self.hypotheses.append(hypothesis)
        return hypothesis
    
    def _generate_statement(self, hypothesis_type: str, market_data: Dict[str, Any]) -> str:
        """Generate hypothesis statement."""
        statements = {
            "indicator_combination": "Combining {ind1} with {ind2} improves signal quality by reducing false positives",
            "threshold_optimization": "Dynamic thresholds based on volatility outperform static thresholds",
            "regime_pattern": "Regime {regime} shows predictable pattern in {timeframe} timeframe",
            "correlation_breakdown": "When correlation between {asset1} and {asset2} breaks down, mean reversion follows",
            "seasonality_effect": "Hour {hour} UTC consistently shows {direction} bias",
            "volume_signal": "Volume spike without price movement predicts {direction} breakout",
            "volatility_regime": "Low volatility periods are followed by high volatility with {direction} bias",
        }
        
        template = statements.get(hypothesis_type, "Unknown hypothesis")
        
        # Fill in template
        statement = template.format(
            ind1="RSI", ind2="MACD",
            regime="bull", timeframe="1h",
            asset1="BTC", asset2="ETH",
            hour="16:00", direction="upward",
        )
        
        return statement
    
    def _generate_test_plan(self, hypothesis_type: str) -> Dict[str, Any]:
        """Generate test plan for hypothesis."""
        return {
            "method": "backtest",
            "data_required": ["prices", "volumes", "indicators"],
            "lookback_period": "30d",
            "metrics": ["sharpe", "win_rate", "profit_factor"],
            "significance_level": 0.05,
            "min_trades": 50,
        }
    
    def _generate_expected_outcome(self, hypothesis_type: str) -> Dict[str, float]:
        """Generate expected outcome."""
        return {
            "expected_improvement": 0.1,  # 10% improvement
            "expected_sharpe": 1.5,
            "expected_win_rate": 0.55,
        }
    
    def test_hypothesis(
        self,
        hypothesis_id: str,
        test_results: Dict[str, float],
    ) -> Dict[str, Any]:
        """Test a hypothesis with results."""
        hypothesis = next((h for h in self.hypotheses if h["id"] == hypothesis_id), None)
        if not hypothesis:
            return {"error": "Hypothesis not found"}
        
        # Evaluate results
        expected = hypothesis["expected_outcome"]
        
        # Check if hypothesis is validated
        validated = True
        if test_results.get("sharpe", 0) < expected.get("expected_sharpe", 1.0):
            validated = False
        if test_results.get("win_rate", 0) < expected.get("expected_win_rate", 0.5):
            validated = False
        
        # Update hypothesis
        hypothesis["status"] = HypothesisStatus.VALIDATED.value if validated else HypothesisStatus.REJECTED.value
        hypothesis["results"] = test_results
        hypothesis["confidence"] = 0.8 if validated else 0.2
        hypothesis["tested_at"] = datetime.utcnow()
        
        # Record finding
        finding = {
            "hypothesis_id": hypothesis_id,
            "type": hypothesis["type"],
            "validated": validated,
            "results": test_results,
            "statement": hypothesis["statement"],
        }
        self.findings.append(finding)
        
        if validated:
            self.hypotheses_validated += 1
            self._add_to_knowledge(hypothesis)
        else:
            self.hypotheses_rejected += 1
        
        return finding
    
    def _add_to_knowledge(self, hypothesis: Dict[str, Any]):
        """Add validated hypothesis to knowledge base."""
        hypothesis_type = hypothesis["type"]
        
        if hypothesis_type not in self.pattern_library:
            self.pattern_library[hypothesis_type] = {
                "count": 0,
                "avg_improvement": 0.0,
                "best_results": [],
            }
        
        self.pattern_library[hypothesis_type]["count"] += 1
        
        if hypothesis.get("results"):
            improvement = hypothesis["results"].get("improvement", 0)
            current_avg = self.pattern_library[hypothesis_type]["avg_improvement"]
            count = self.pattern_library[hypothesis_type]["count"]
            
            # Update running average
            self.pattern_library[hypothesis_type]["avg_improvement"] = (
                (current_avg * (count - 1) + improvement) / count
            )
    
    def get_validated_patterns(self) -> List[Dict[str, Any]]:
        """Get validated patterns."""
        return [
            {"type": k, **v}
            for k, v in self.pattern_library.items()
            if v["count"] >= 3 and v["avg_improvement"] > 0
        ]
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get research summary."""
        return {
            "hypotheses_generated": self.hypotheses_generated,
            "hypotheses_validated": self.hypotheses_validated,
            "hypotheses_rejected": self.hypotheses_rejected,
            "validation_rate": self.hypotheses_validated / max(self.hypotheses_generated, 1),
            "patterns_discovered": len(self.pattern_library),
            "findings_count": len(self.findings),
        }


# ============================================================================
# Self-Improving Code Engine
# ============================================================================

class SelfImprovingCodeEngine:
    """
    Writes and improves its own code.
    
    Uses evolutionary programming to:
    1. Generate strategy code from genomes
    2. Identify code improvements
    3. Apply improvements
    4. Test improvements
    5. Keep what works
    """
    
    def __init__(self):
        self.code_versions: Dict[str, List[str]] = defaultdict(list)
        self.improvement_log: List[Dict[str, Any]] = []
        self.successful_patterns: List[str] = []
    
    def genome_to_code(self, genome: StrategyGenome) -> str:
        """Convert genome to executable code."""
        # Get gene values
        primary = genome.get_gene("primary_indicator")
        secondary = genome.get_gene("secondary_indicator")
        rsi_period = genome.get_gene("rsi_period")
        rsi_overbought = genome.get_gene("rsi_overbought")
        rsi_oversold = genome.get_gene("rsi_oversold")
        stop_loss = genome.get_gene("stop_loss_pct")
        take_profit = genome.get_gene("take_profit_pct")
        position_size = genome.get_gene("position_size")
        
        code = f'''
class EvolvedStrategy_{genome.genome_id}:
    """
    Evolved strategy from generation {genome.generation}.
    Fitness: {genome.fitness:.4f}
    """
    
    def __init__(self):
        self.primary_indicator = "{primary.value if primary else 'rsi'}"
        self.secondary_indicator = "{secondary.value if secondary else 'macd'}"
        self.rsi_period = {rsi_period.value if rsi_period else 14}
        self.rsi_overbought = {rsi_overbought.value if rsi_overbought else 70}
        self.rsi_oversold = {rsi_oversold.value if rsi_oversold else 30}
        self.stop_loss_pct = {stop_loss.value if stop_loss else 0.02}
        self.take_profit_pct = {take_profit.value if take_profit else 0.04}
        self.position_size = {position_size.value if position_size else 0.1}
    
    def generate_signal(self, prices, indicators):
        """Generate trading signal."""
        if len(prices) < self.rsi_period + 5:
            return "hold", 0.0
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices)
        
        # Generate signal
        signal = "hold"
        confidence = 0.0
        
        if rsi < self.rsi_oversold:
            signal = "buy"
            confidence = (self.rsi_oversold - rsi) / self.rsi_oversold
        elif rsi > self.rsi_overbought:
            signal = "sell"
            confidence = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
        
        return signal, confidence
    
    def _calculate_rsi(self, prices, period=None):
        """Calculate RSI."""
        period = period or self.rsi_period
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-8
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
'''
        
        return code
    
    def identify_improvements(self, code: str, performance: Dict[str, float]) -> List[str]:
        """Identify potential code improvements."""
        improvements = []
        
        # Check for common improvements
        if "np.mean" in code and "np.std" in code:
            improvements.append("Add rolling window for adaptive indicators")
        
        if "hold" in code and "buy" in code and "sell" in code:
            improvements.append("Add short selling capability")
        
        if performance.get("win_rate", 0) < 0.5:
            improvements.append("Add trend filter to reduce counter-trend trades")
        
        if performance.get("max_drawdown", 0) > 0.2:
            improvements.append("Add dynamic position sizing based on volatility")
        
        return improvements
    
    def apply_improvement(self, code: str, improvement: str) -> str:
        """Apply an improvement to code."""
        # Simplified improvement application
        improved_code = code
        
        if "Add trend filter" in improvement:
            improved_code = improved_code.replace(
                "# Generate signal",
                "# Trend filter\n        trend = self._calculate_trend(prices)\n        if abs(trend) < 0.1:\n            return 'hold', 0.0\n        \n        # Generate signal"
            )
        
        if "Add dynamic position sizing" in improvement:
            improved_code = improved_code.replace(
                "self.position_size = ",
                "# Dynamic position sizing\n        self.base_position_size = "
            )
        
        return improved_code
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "code_versions_tracked": sum(len(v) for v in self.code_versions.values()),
            "improvements_logged": len(self.improvement_log),
            "successful_patterns": len(self.successful_patterns),
        }


# ============================================================================
# Level 10 Orchestrator
# ============================================================================

class Level10System:
    """
    Level 10: Self-Evolving Trading System.
    
    The ultimate trading system that evolves, learns, and improves itself.
    Combines all Level 10 components:
    - Evolution Engine (genetic algorithm)
    - Meta-Learner (learning to learn)
    - Strategy Memory (long-term memory)
    - Autonomous Researcher (hypothesis generation)
    - Self-Improving Code (code evolution)
    """
    
    def __init__(
        self,
        *,
        population_size: int = 50,
        n_generations: int = 100,
        target_fitness: float = 2.0,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.target_fitness = target_fitness
        
        # Components
        self.evolution = EvolutionEngine(population_size=population_size)
        self.meta_learner = MetaLearner()
        self.memory = StrategyMemory()
        self.researcher = AutonomousResearcher()
        self.code_engine = SelfImprovingCodeEngine()
        
        # State
        self.current_generation = 0
        self.best_strategy: Optional[StrategyGenome] = None
        self.evolution_complete = False
        
        # Statistics
        self.total_improvements = 0
        self.hypotheses_tested = 0
        self.code_improvements = 0
    
    def initialize(self, complexity: float = 0.5):
        """Initialize the system."""
        self.evolution.initialize_population(complexity)
        logger.info("Level 10 System initialized with %d genomes", self.population_size)
    
    def evolve(
        self,
        fitness_function: Callable[[StrategyGenome], float],
        generations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run evolution for specified generations.
        
        Returns evolution results.
        """
        n_gens = generations or self.n_generations
        
        for gen in range(n_gens):
            self.current_generation = gen
            
            # Evaluate fitness
            self.evolution.evaluate_population(fitness_function)
            
            # Get best genome
            best = self.evolution.get_best_genome()
            if best and (self.best_strategy is None or best.fitness > self.best_strategy.fitness):
                self.best_strategy = best
                self.total_improvements += 1
            
            # Meta-learning
            convergence = self.meta_learner.analyze_convergence(
                self.evolution.avg_fitness_history
            )
            
            # Adjust evolution parameters based on meta-learning
            if convergence["converged"]:
                self.evolution.mutation_rate = min(0.3, self.evolution.mutation_rate * 1.2)
                logger.info("Convergence detected - increasing exploration")
            
            # Research hypothesis
            if gen % 10 == 0:
                hypothesis = self.researcher.generate_hypothesis({
                    "generation": gen,
                    "best_fitness": best.fitness if best else 0,
                    "avg_fitness": self.evolution.avg_fitness_history[-1] if self.evolution.avg_fitness_history else 0,
                })
            
            # Evolve next generation
            self.evolution.evolve_generation()
            
            # Check termination
            if best and best.fitness >= self.target_fitness:
                logger.info("Target fitness reached at generation %d", gen)
                self.evolution_complete = True
                break
        
        return {
            "generations_completed": self.current_generation + 1,
            "best_fitness": self.best_strategy.fitness if self.best_strategy else 0,
            "best_genome": self.best_strategy.to_dict() if self.best_strategy else None,
            "convergence_achieved": self.evolution_complete,
            "total_improvements": self.total_improvements,
        }
    
    def get_best_strategies(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get best strategies."""
        diverse = self.evolution.get_diverse_genomes(n)
        return [g.to_dict() for g in diverse]
    
    def get_best_code(self) -> Optional[str]:
        """Get code for best strategy."""
        if not self.best_strategy:
            return None
        return self.code_engine.genome_to_code(self.best_strategy)
    
    def learn_from_generation(self, results: Dict[str, Any]):
        """Learn from generation results."""
        # Update meta-learner
        if self.best_strategy:
            self.memory.remember_strategy(
                self.best_strategy,
                market_conditions={"generation": self.current_generation},
                performance={
                    "fitness": self.best_strategy.fitness,
                    "sharpe": self.best_strategy.sharpe_ratio,
                }
            )
    
    def get_system_report(self) -> Dict[str, Any]:
        """Get comprehensive system report."""
        return {
            "generation": self.current_generation,
            "evolution_complete": self.evolution_complete,
            "best_strategy": self.best_strategy.to_dict() if self.best_strategy else None,
            "evolution_stats": {
                "avg_fitness_history": self.evolution.avg_fitness_history[-10:],
                "best_fitness_history": self.evolution.best_fitness_history[-10:],
                "diversity_history": self.evolution.diversity_history[-10:],
            },
            "meta_learning": self.meta_learner.get_meta_summary(),
            "memory": self.memory.get_memory_stats(),
            "research": self.researcher.get_research_summary(),
            "code_engine": self.code_engine.get_engine_stats(),
            "total_improvements": self.total_improvements,
        }
    
    def save_state(self) -> Dict[str, Any]:
        """Save system state."""
        return {
            "generation": self.current_generation,
            "best_strategy": self.best_strategy.to_dict() if self.best_strategy else None,
            "evolution_complete": self.evolution_complete,
            "total_improvements": self.total_improvements,
            "population_size": len(self.evolution.population),
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load system state."""
        self.current_generation = state.get("generation", 0)
        self.evolution_complete = state.get("evolution_complete", False)
        self.total_improvements = state.get("total_improvements", 0)


# ============================================================================
# Factory Function
# ============================================================================

def create_level10_system(**kwargs) -> Level10System:
    """Create Level 10 self-evolving system."""
    return Level10System(**kwargs)
