"""
Genetic Algorithm for Strategy Parameter Optimization.

Implements evolutionary optimization for trading strategy parameters using
genetic algorithms with tournament selection, crossover, and mutation operators.
Supports multi-objective optimization (NSGA-II) and walk-forward validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Represents a single solution in the genetic algorithm population."""

    genes: np.ndarray
    fitness: float = -np.inf
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> Individual:
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            generation=self.generation,
            metadata=self.metadata.copy(),
        )


@dataclass
class GeneticConfig:
    """Configuration for the genetic algorithm."""

    population_size: int = 100
    n_generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_ratio: float = 0.1
    tournament_size: int = 3
    crossover_type: str = "single_point"
    seed: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    best_individual: Individual
    best_fitness: float
    n_generations_run: int
    fitness_history: List[float]
    diversity_history: List[float]
    final_population: List[Individual]
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneticOperators:
    """Genetic operators for selection, crossover, and mutation."""

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def tournament_selection(
        self, population: List[Individual], k: int = 3
    ) -> Individual:
        """Select an individual using tournament selection.

        Args:
            population: List of individuals to select from.
            k: Tournament size (number of candidates).

        Returns:
            Winner of the tournament (highest fitness).
        """
        indices = self.rng.choice(len(population), size=min(k, len(population)), replace=False)
        candidates = [population[i] for i in indices]
        return max(candidates, key=lambda ind: ind.fitness)

    def single_point_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform single-point crossover between two parents.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Tuple of two child individuals.
        """
        n_genes = len(parent1.genes)
        if n_genes < 2:
            return parent1.copy(), parent2.copy()

        point = self.rng.integers(1, n_genes)
        child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def uniform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform uniform crossover between two parents.

        Each gene is independently chosen from either parent with 50% probability.

        Args:
            parent1: First parent.
            parent2: Second parent.

        Returns:
            Tuple of two child individuals.
        """
        mask = self.rng.random(len(parent1.genes)) < 0.5
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def gaussian_mutation(
        self, individual: Individual, sigma: float = 0.1
    ) -> Individual:
        """Apply Gaussian mutation to an individual's genes.

        Args:
            individual: Individual to mutate.
            sigma: Standard deviation of the Gaussian noise.

        Returns:
            New mutated individual.
        """
        noise = self.rng.normal(0, sigma, size=len(individual.genes))
        mutated_genes = individual.genes + noise
        child = individual.copy()
        child.genes = mutated_genes
        return child

    def mutate(
        self,
        individual: Individual,
        bounds: List[Tuple[float, float]],
        mutation_rate: float = 0.1,
        mutation_sigma: float = 0.1,
    ) -> Individual:
        """Mutate an individual's genes within bounds.

        Each gene has mutation_rate probability of being mutated.
        Mutated genes are perturbed by Gaussian noise and clipped to bounds.

        Args:
            individual: Individual to mutate.
            bounds: List of (min, max) tuples for each gene.
            mutation_rate: Probability of mutating each gene.
            mutation_sigma: Standard deviation for Gaussian mutation.

        Returns:
            New mutated individual with genes clipped to bounds.
        """
        child = individual.copy()
        n_genes = len(child.genes)

        for i in range(n_genes):
            if self.rng.random() < mutation_rate:
                lo, hi = bounds[i]
                gene_range = hi - lo
                sigma = mutation_sigma * gene_range
                child.genes[i] += self.rng.normal(0, sigma)
                child.genes[i] = np.clip(child.genes[i], lo, hi)

        return child


class GeneticOptimizer:
    """Genetic algorithm optimizer for strategy parameter optimization.

    Args:
        fitness_function: Callable that takes a numpy array of parameters
            and returns a fitness score (higher is better).
        parameter_bounds: List of (min, max) tuples for each parameter.
    """

    def __init__(
        self,
        fitness_function: Callable[[np.ndarray], float],
        parameter_bounds: List[Tuple[float, float]],
    ):
        self.fitness_function = fitness_function
        self.parameter_bounds = parameter_bounds
        self.n_params = len(parameter_bounds)
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

    def _initialize_population(self, size: int, rng: np.random.Generator) -> List[Individual]:
        """Create initial random population."""
        population = []
        for _ in range(size):
            genes = np.array([
                rng.uniform(low, high) for low, high in self.parameter_bounds
            ])
            population.append(Individual(genes=genes))
        return population

    def _evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """Evaluate fitness for all individuals in population."""
        for individual in population:
            if individual.fitness == -np.inf:
                try:
                    individual.fitness = self.fitness_function(individual.genes)
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    individual.fitness = -np.inf
        return population

    def compute_diversity(self, population: List[Individual]) -> float:
        """Compute average pairwise genetic distance in population.

        Uses normalized Euclidean distance between gene vectors.

        Args:
            population: List of individuals.

        Returns:
            Average diversity score in [0, 1].
        """
        if len(population) < 2:
            return 0.0

        genes_matrix = np.array([ind.genes for ind in population])
        n = len(population)

        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = genes_matrix[i] - genes_matrix[j]
                ranges = np.array([hi - lo for lo, hi in self.parameter_bounds])
                ranges[ranges == 0] = 1.0
                normalized_diff = diff / ranges
                dist = np.sqrt(np.sum(normalized_diff ** 2)) / np.sqrt(self.n_params)
                distances.append(dist)

        return float(np.mean(distances))

    def run_generation(
        self,
        population: List[Individual],
        config: GeneticConfig,
        operators: GeneticOperators,
    ) -> List[Individual]:
        """Run one generation of the genetic algorithm.

        Args:
            population: Current population.
            config: Genetic algorithm configuration.
            operators: Genetic operators instance.

        Returns:
            New population for next generation.
        """
        population = self._evaluate_population(population)

        elite_count = max(1, int(config.elitism_ratio * config.population_size))
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elites = [ind.copy() for ind in sorted_pop[:elite_count]]

        new_population: List[Individual] = []
        new_population.extend(elites)

        mutation_sigma = np.mean([hi - lo for lo, hi in self.parameter_bounds]) * 0.1

        while len(new_population) < config.population_size:
            parent1 = operators.tournament_selection(population, config.tournament_size)
            parent2 = operators.tournament_selection(population, config.tournament_size)

            if operators.rng.random() < config.crossover_rate:
                if config.crossover_type == "uniform":
                    child1, child2 = operators.uniform_crossover(parent1, parent2)
                else:
                    child1, child2 = operators.single_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = operators.mutate(
                child1, self.parameter_bounds, config.mutation_rate, mutation_sigma
            )
            child2 = operators.mutate(
                child2, self.parameter_bounds, config.mutation_rate, mutation_sigma
            )

            if len(new_population) < config.population_size:
                new_population.append(child1)
            if len(new_population) < config.population_size:
                new_population.append(child2)

        gen = population[0].generation + 1 if population else 1
        for ind in new_population:
            ind.generation = gen

        return new_population

    def optimize(self, config: Optional[GeneticConfig] = None) -> Individual:
        """Run the full genetic algorithm optimization.

        Args:
            config: Genetic algorithm configuration. Uses defaults if None.

        Returns:
            Best individual found.
        """
        config = config or GeneticConfig()
        rng = np.random.default_rng(config.seed)
        operators = GeneticOperators(rng)

        population = self._initialize_population(config.population_size, rng)
        self.fitness_history = []
        self.diversity_history = []

        best_ever: Optional[Individual] = None

        for gen in range(config.n_generations):
            population = self.run_generation(population, config, operators)

            best_in_gen = max(population, key=lambda ind: ind.fitness)
            self.fitness_history.append(best_in_gen.fitness)
            self.diversity_history.append(self.compute_diversity(population))

            if best_ever is None or best_in_gen.fitness > best_ever.fitness:
                best_ever = best_in_gen.copy()

            if gen % 10 == 0 or gen == config.n_generations - 1:
                logger.info(
                    f"Generation {gen}: best_fitness={best_in_gen.fitness:.4f}, "
                    f"diversity={self.diversity_history[-1]:.4f}"
                )

        if best_ever is None:
            raise RuntimeError("Optimization failed: no valid individuals found")

        best_ever.metadata["optimization_complete"] = True
        best_ever.metadata["total_generations"] = config.n_generations
        return best_ever

    def get_result(self, best: Individual, n_generations: int) -> OptimizationResult:
        """Build optimization result from final state.

        Args:
            best: Best individual found.
            n_generations: Number of generations run.

        Returns:
            OptimizationResult with full history.
        """
        return OptimizationResult(
            best_individual=best,
            best_fitness=best.fitness,
            n_generations_run=n_generations,
            fitness_history=self.fitness_history.copy(),
            diversity_history=self.diversity_history.copy(),
            final_population=[],
        )


class MultiObjectiveOptimizer:
    """Multi-objective genetic algorithm optimizer using NSGA-II.

    Optimizes for multiple objectives simultaneously and maintains a
    Pareto front of non-dominated solutions.

    Args:
        fitness_functions: List of callable fitness functions.
        objectives: List of objective names (e.g., "sharpe", "drawdown", "win_rate").
        parameter_bounds: List of (min, max) tuples for each parameter.
    """

    def __init__(
        self,
        fitness_functions: List[Callable[[np.ndarray], float]],
        objectives: List[str],
        parameter_bounds: List[Tuple[float, float]],
    ):
        if len(fitness_functions) != len(objectives):
            raise ValueError("Number of fitness functions must match number of objectives")

        self.fitness_functions = fitness_functions
        self.objectives = objectives
        self.parameter_bounds = parameter_bounds
        self.n_params = len(parameter_bounds)
        self.n_objectives = len(objectives)
        self.operators = GeneticOperators()

    def _evaluate_objectives(self, individual: Individual) -> List[float]:
        """Evaluate all objectives for an individual."""
        scores = []
        for fn in self.fitness_functions:
            try:
                scores.append(fn(individual.genes))
            except Exception as e:
                logger.warning(f"Objective evaluation failed: {e}")
                scores.append(-np.inf)
        return scores

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (all objectives higher is better)."""
        at_least_one_better = False
        for v1, v2 in zip(obj1, obj2):
            if v1 < v2:
                return False
            if v1 > v2:
                at_least_one_better = True
        return at_least_one_better

    def _fast_non_dominated_sort(
        self, population: List[Individual]
    ) -> List[List[Individual]]:
        """Perform fast non-dominated sorting (NSGA-II)."""
        fronts: List[List[Individual]] = []
        dominated_count = {id(ind): 0 for ind in population}
        dominates_set = {id(ind): [] for ind in population}
        obj_cache = {id(ind): self._evaluate_objectives(ind) for ind in population}

        for p in population:
            for q in population:
                if id(p) == id(q):
                    continue
                if self._dominates(obj_cache[id(p)], obj_cache[id(q)]):
                    dominates_set[id(p)].append(q)
                elif self._dominates(obj_cache[id(q)], obj_cache[id(p)]):
                    dominated_count[id(p)] += 1

        front = [ind for ind in population if dominated_count[id(ind)] == 0]
        fronts.append(front)

        i = 0
        while fronts[i]:
            next_front = []
            for ind in fronts[i]:
                for dominated in dominates_set[id(ind)]:
                    dominated_count[id(dominated)] -= 1
                    if dominated_count[id(dominated)] == 0:
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def _crowding_distance(self, front: List[Individual]) -> List[float]:
        """Calculate crowding distance for individuals in a front."""
        if not front:
            return []

        n = len(front)
        distances = [0.0] * n
        obj_values = [self._evaluate_objectives(ind) for ind in front]

        for m in range(self.n_objectives):
            sorted_indices = sorted(range(n), key=lambda i: obj_values[i][m])
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            obj_range = obj_values[sorted_indices[-1]][m] - obj_values[sorted_indices[0]][m]
            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    obj_values[sorted_indices[i + 1]][m]
                    - obj_values[sorted_indices[i - 1]][m]
                ) / obj_range

        return distances

    def pareto_front(self, population: List[Individual]) -> List[Individual]:
        """Extract the Pareto-optimal front from a population.

        Args:
            population: Population to extract from.

        Returns:
            List of non-dominated individuals.
        """
        fronts = self._fast_non_dominated_sort(population)
        return fronts[0] if fronts else []

    def nsga2_selection(self, population: List[Individual]) -> List[Individual]:
        """Select next generation using NSGA-II selection.

        Combines non-dominated sorting with crowding distance to maintain
        diversity along the Pareto front.

        Args:
            population: Combined parent and offspring population.

        Returns:
            Selected population for next generation.
        """
        fronts = self._fast_non_dominated_sort(population)
        new_population: List[Individual] = []

        for front in fronts:
            if len(new_population) + len(front) <= len(population) // 2:
                new_population.extend(front)
            else:
                distances = self._crowding_distance(front)
                sorted_front = sorted(
                    zip(front, distances), key=lambda x: x[1], reverse=True
                )
                remaining = len(population) // 2 - len(new_population)
                new_population.extend([ind for ind, _ in sorted_front[:remaining]])
                break

        return new_population

    def optimize(
        self, config: Optional[GeneticConfig] = None
    ) -> Tuple[List[Individual], OptimizationResult]:
        """Run multi-objective NSGA-II optimization.

        Args:
            config: Genetic algorithm configuration.

        Returns:
            Tuple of (pareto_front, optimization_result).
        """
        config = config or GeneticConfig()
        rng = np.random.default_rng(config.seed)
        self.operators = GeneticOperators(rng)

        population: List[Individual] = []
        for _ in range(config.population_size):
            genes = np.array([
                rng.uniform(low, high) for low, high in self.parameter_bounds
            ])
            population.append(Individual(genes=genes))

        fitness_history = []
        diversity_history = []

        for gen in range(config.n_generations):
            offspring = []
            mutation_sigma = np.mean([hi - lo for lo, hi in self.parameter_bounds]) * 0.1

            while len(offspring) < config.population_size:
                p1 = self.operators.tournament_selection(population, config.tournament_size)
                p2 = self.operators.tournament_selection(population, config.tournament_size)

                if rng.random() < config.crossover_rate:
                    c1, c2 = self.operators.single_point_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = self.operators.mutate(c1, self.parameter_bounds, config.mutation_rate, mutation_sigma)
                c2 = self.operators.mutate(c2, self.parameter_bounds, config.mutation_rate, mutation_sigma)
                offspring.extend([c1, c2])

            combined = population + offspring[:config.population_size]
            population = self.nsga2_selection(combined)

            gen_num = gen + 1
            for ind in population:
                ind.generation = gen_num

            pf = self.pareto_front(population)
            if pf:
                best_fit = max(ind.fitness for ind in pf)
                fitness_history.append(best_fit)
            diversity_history.append(self._compute_population_diversity(population, rng))

            if gen % 10 == 0 or gen == config.n_generations - 1:
                logger.info(f"NSGA-II Generation {gen}: Pareto front size = {len(pf)}")

        final_pareto = self.pareto_front(population)
        result = OptimizationResult(
            best_individual=final_pareto[0] if final_pareto else population[0],
            best_fitness=final_pareto[0].fitness if final_pareto else -np.inf,
            n_generations_run=config.n_generations,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            final_population=population,
            metadata={"pareto_front_size": len(final_pareto)},
        )

        return final_pareto, result

    def _compute_population_diversity(
        self, population: List[Individual], rng: np.random.Generator
    ) -> float:
        """Compute population diversity metric."""
        if len(population) < 2:
            return 0.0
        genes_matrix = np.array([ind.genes for ind in population])
        ranges = np.array([hi - lo for lo, hi in self.parameter_bounds])
        ranges[ranges == 0] = 1.0
        normalized = (genes_matrix - np.array([lo for lo, _ in self.parameter_bounds])) / ranges
        return float(np.std(normalized).mean())


class WalkForwardOptimizer:
    """Walk-forward optimization for strategy parameter validation.

    Implements expanding window optimization and purged cross-validation
    to prevent overfitting and look-ahead bias.

    Args:
        optimizer_factory: Callable that creates a GeneticOptimizer for a given
            training data slice.
    """

    def __init__(
        self,
        optimizer_factory: Callable[[Any], GeneticOptimizer],
    ):
        self.optimizer_factory = optimizer_factory

    def purged_cross_validation(
        self,
        returns: np.ndarray,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged cross-validation splits.

        Applies an embargo period between train and test sets to prevent
        information leakage from overlapping labels.

        Args:
            returns: Array of returns or data indices.
            n_splits: Number of CV splits.
            embargo_pct: Fraction of data to embargo between train/test.

        Returns:
            List of (train_indices, test_indices) tuples.
        """
        n = len(returns)
        split_size = n // (n_splits + 1)
        embargo_size = max(1, int(n * embargo_pct))
        splits = []

        for i in range(n_splits):
            test_start = i * split_size + split_size
            test_end = test_start + split_size
            test_end = min(test_end, n)

            train_end = test_start - embargo_size
            train_start = 0

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        logger.info(
            f"Purged CV: {len(splits)} splits, embargo={embargo_size} samples"
        )
        return splits

    def optimize_walk_forward(
        self,
        strategy: Any,
        data: Any,
        n_splits: int = 5,
        config: Optional[GeneticConfig] = None,
        window_type: str = "expanding",
    ) -> OptimizationResult:
        """Run walk-forward optimization.

        Splits data into train/test windows, optimizes parameters on each
        train window, and evaluates on the corresponding test window.

        Args:
            strategy: Strategy instance to optimize.
            data: Trading data (OHLCV, returns, etc.).
            n_splits: Number of walk-forward splits.
            config: Genetic algorithm configuration.
            window_type: "expanding" or "sliding" window.

        Returns:
            OptimizationResult with aggregated metrics.
        """
        config = config or GeneticConfig()
        n_samples = self._get_data_length(data)
        split_size = n_samples // (n_splits + 1)

        all_fitness: List[float] = []
        all_best_genes: List[np.ndarray] = []
        fitness_history: List[float] = []
        diversity_history: List[float] = []

        for i in range(n_splits):
            if window_type == "expanding":
                train_end = (i + 1) * split_size
                train_start = 0
            else:
                train_end = (i + 1) * split_size
                train_start = max(0, train_end - split_size * 2)

            test_start = train_end
            test_end = min(test_start + split_size, n_samples)

            train_data = self._slice_data(data, train_start, train_end)
            test_data = self._slice_data(data, test_start, test_end)

            logger.info(
                f"Walk-forward split {i+1}/{n_splits}: "
                f"train=[{train_start}:{train_end}], test=[{test_start}:{test_end}]"
            )

            optimizer = self.optimizer_factory(train_data)
            split_config = GeneticConfig(
                population_size=config.population_size,
                n_generations=max(5, config.n_generations // n_splits),
                crossover_rate=config.crossover_rate,
                mutation_rate=config.mutation_rate,
                elitism_ratio=config.elitism_ratio,
                tournament_size=config.tournament_size,
                seed=config.seed,
            )

            best = optimizer.optimize(split_config)
            all_best_genes.append(best.genes)

            test_fitness = self._evaluate_on_test(strategy, best.genes, test_data)
            all_fitness.append(test_fitness)

            fitness_history.extend(optimizer.fitness_history)
            diversity_history.extend(optimizer.diversity_history)

        mean_fitness = float(np.mean(all_fitness))
        std_fitness = float(np.std(all_fitness))

        best_overall_idx = int(np.argmax(all_fitness))
        best_genes = all_best_genes[best_overall_idx]

        result = OptimizationResult(
            best_individual=Individual(genes=best_genes, fitness=mean_fitness),
            best_fitness=mean_fitness,
            n_generations_run=config.n_generations,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            final_population=[],
            metadata={
                "n_splits": n_splits,
                "window_type": window_type,
                "split_fitness": all_fitness,
                "split_std": std_fitness,
                "best_split_idx": best_overall_idx,
            },
        )

        logger.info(
            f"Walk-forward complete: mean_fitness={mean_fitness:.4f}, "
            f"std={std_fitness:.4f}, best_split={best_overall_idx + 1}"
        )
        return result

    def expanding_window_optimization(
        self,
        strategy: Any,
        data: Any,
        initial_window: int = 100,
        step_size: int = 50,
        config: Optional[GeneticConfig] = None,
    ) -> OptimizationResult:
        """Run expanding window optimization.

        Starts with an initial training window and expands it by step_size
        at each iteration, re-optimizing parameters on the growing dataset.

        Args:
            strategy: Strategy instance to optimize.
            data: Trading data.
            initial_window: Size of the initial training window.
            step_size: Number of samples to add at each step.
            config: Genetic algorithm configuration.

        Returns:
            OptimizationResult with aggregated metrics.
        """
        config = config or GeneticConfig()
        n_samples = self._get_data_length(data)

        all_fitness: List[float] = []
        all_best_genes: List[np.ndarray] = []
        fitness_history: List[float] = []
        diversity_history: List[float] = []

        window_end = initial_window
        step = 0

        while window_end + step_size <= n_samples:
            train_start = 0
            train_end = window_end
            test_start = window_end
            test_end = min(window_end + step_size, n_samples)

            train_data = self._slice_data(data, train_start, train_end)
            test_data = self._slice_data(data, test_start, test_end)

            logger.info(
                f"Expanding window step {step}: "
                f"train_size={train_end - train_start}, test_size={test_end - test_start}"
            )

            optimizer = self.optimizer_factory(train_data)
            step_config = GeneticConfig(
                population_size=config.population_size,
                n_generations=max(5, config.n_generations // 10),
                crossover_rate=config.crossover_rate,
                mutation_rate=config.mutation_rate,
                elitism_ratio=config.elitism_ratio,
                tournament_size=config.tournament_size,
                seed=config.seed,
            )

            best = optimizer.optimize(step_config)
            all_best_genes.append(best.genes)

            test_fitness = self._evaluate_on_test(strategy, best.genes, test_data)
            all_fitness.append(test_fitness)

            fitness_history.extend(optimizer.fitness_history)
            diversity_history.extend(optimizer.diversity_history)

            window_end += step_size
            step += 1

        if not all_fitness:
            raise RuntimeError("No valid windows for expanding optimization")

        mean_fitness = float(np.mean(all_fitness))
        best_overall_idx = int(np.argmax(all_fitness))

        result = OptimizationResult(
            best_individual=Individual(
                genes=all_best_genes[best_overall_idx], fitness=mean_fitness
            ),
            best_fitness=mean_fitness,
            n_generations_run=config.n_generations,
            fitness_history=fitness_history,
            diversity_history=diversity_history,
            final_population=[],
            metadata={
                "n_steps": step,
                "initial_window": initial_window,
                "step_size": step_size,
                "step_fitness": all_fitness,
            },
        )

        logger.info(
            f"Expanding window complete: {step} steps, "
            f"mean_fitness={mean_fitness:.4f}"
        )
        return result

    @staticmethod
    def _get_data_length(data: Any) -> int:
        """Get length of data array or DataFrame."""
        if hasattr(data, "__len__"):
            return len(data)
        if hasattr(data, "shape"):
            return data.shape[0]
        raise ValueError("Cannot determine data length")

    @staticmethod
    def _slice_data(data: Any, start: int, end: int) -> Any:
        """Slice data by index range."""
        if hasattr(data, "iloc"):
            return data.iloc[start:end]
        return data[start:end]

    @staticmethod
    def _evaluate_on_test(strategy: Any, genes: np.ndarray, test_data: Any) -> float:
        """Evaluate strategy with given genes on test data.

        Default implementation returns mean return as fitness.
        Override or provide a custom strategy with evaluate() method.

        Args:
            strategy: Strategy instance.
            genes: Optimized parameter values.
            test_data: Test data slice.

        Returns:
            Fitness score on test data.
        """
        if hasattr(strategy, "evaluate"):
            return float(strategy.evaluate(genes, test_data))

        if hasattr(test_data, "__len__") and len(test_data) > 0:
            if hasattr(test_data, "iloc"):
                returns = test_data.iloc[:, -1] if hasattr(test_data, "iloc") else test_data
            else:
                returns = test_data
            if hasattr(returns, "values"):
                returns = returns.values
            if hasattr(returns, "__len__") and len(returns) > 0:
                mean_ret = float(np.mean(returns))
                std_ret = float(np.std(returns))
                if std_ret > 0:
                    return mean_ret / std_ret
                return mean_ret

        return 0.0
