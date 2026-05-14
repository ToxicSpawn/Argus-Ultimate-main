"""
Neural Architecture Evolution — evolves MLP architectures using genetic
algorithms (neuroevolution).

Each ``NetworkGenome`` encodes a variable-depth MLP: layer sizes, activation
functions, and dropout rates.  The population evolves via tournament
selection, layer-level crossover, and structural mutations (add/remove layer,
resize, swap activation).

Evaluation uses PyTorch when available; otherwise falls back to a pure-Python
forward pass with random weights (sufficient for fitness-ranking within a
generation).

Usage::

    ne = Neuroevolution()
    pop = ne.create_population(pop_size=20, input_dim=10, output_dim=1)
    for genome in pop:
        genome.fitness = ne.evaluate(genome, train_data, val_data)
    pop = ne.evolve(pop, top_k=5)
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Attempt PyTorch import — graceful fallback
_TORCH_AVAILABLE = False
try:
    import torch  # type: ignore[import-untyped]
    import torch.nn as nn  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
    logger.info("Neuroevolution: PyTorch available, using GPU-accelerated evaluation")
except ImportError:
    logger.info("Neuroevolution: PyTorch not available, using pure-Python evaluation")
    logger.debug("Neuroevolution: PyTorch available, using GPU-accelerated evaluation")
except ImportError:
    logger.debug("Neuroevolution: PyTorch not available, using pure-Python evaluation")

_ACTIVATIONS = ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "gelu"]


@dataclass
class NetworkGenome:
    """Genetic encoding of a neural network architecture.

    Attributes
    ----------
    layers : list[int]
        Hidden layer sizes (does not include input/output).
    activations : list[str]
        Activation function for each hidden layer.
    dropout_rates : list[float]
        Dropout rate for each hidden layer.
    fitness : float
        Fitness score assigned by the evaluator.
    generation : int
        Generation in which this genome was created.
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output dimension.
    """

    layers: List[int] = field(default_factory=list)
    activations: List[str] = field(default_factory=list)
    dropout_rates: List[float] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0
    input_dim: int = 10
    output_dim: int = 1

    def __repr__(self) -> str:
        arch = "->".join(str(l) for l in self.layers)
        return (
            f"NetworkGenome(arch=[{arch}], acts={self.activations}, "
            f"fitness={self.fitness:.6f}, gen={self.generation})"
        )


class Neuroevolution:
    """Evolve neural network architectures using genetic algorithms.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    min_layers : int
        Minimum number of hidden layers.
    max_layers : int
        Maximum number of hidden layers.
    min_neurons : int
        Minimum neurons per hidden layer.
    max_neurons : int
        Maximum neurons per hidden layer.
    mutation_rate : float
        Probability of mutating each gene.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        min_layers: int = 1,
        max_layers: int = 6,
        min_neurons: int = 8,
        max_neurons: int = 256,
        mutation_rate: float = 0.3,
    ) -> None:
        self._rng = random.Random(seed)
        self._min_layers = min_layers
        self._max_layers = max_layers
        self._min_neurons = min_neurons
        self._max_neurons = max_neurons
        self._mutation_rate = mutation_rate
        self._best_genome: Optional[NetworkGenome] = None
        logger.info("Neuroevolution initialised (torch=%s)", _TORCH_AVAILABLE)
        logger.debug("Neuroevolution initialised (torch=%s)", _TORCH_AVAILABLE)

    # ------------------------------------------------------------------
    # Population creation
    # ------------------------------------------------------------------

    def create_population(
        self,
        pop_size: int = 20,
        input_dim: int = 10,
        output_dim: int = 1,
    ) -> List[NetworkGenome]:
        """Create an initial random population of network genomes.

        Parameters
        ----------
        pop_size : int
            Number of genomes in the population.
        input_dim : int
            Number of input features.
        output_dim : int
            Number of outputs.

        Returns
        -------
        list[NetworkGenome]
            Initial population.
        """
        population: List[NetworkGenome] = []
        for _ in range(pop_size):
            n_layers = self._rng.randint(self._min_layers, self._max_layers)
            layers = [
                self._rng.randint(self._min_neurons, self._max_neurons)
                for _ in range(n_layers)
            ]
            activations = [
                self._rng.choice(_ACTIVATIONS) for _ in range(n_layers)
            ]
            dropout_rates = [
                round(self._rng.uniform(0.0, 0.5), 2) for _ in range(n_layers)
            ]
            genome = NetworkGenome(
                layers=layers,
                activations=activations,
                dropout_rates=dropout_rates,
                fitness=0.0,
                generation=0,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            population.append(genome)

        logger.info("Created population of %d genomes (in=%d, out=%d)", pop_size, input_dim, output_dim)
        logger.debug("Created population of %d genomes (in=%d, out=%d)", pop_size, input_dim, output_dim)
        return population

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        genome: NetworkGenome,
        train_data: Any,
        val_data: Any,
    ) -> float:
        """Evaluate a genome's fitness on train/val data.

        If PyTorch is available, builds and trains a real MLP for a few epochs.
        Otherwise, uses a pure-Python forward pass with random weights as a
        proxy for architecture quality (smaller loss = higher fitness).

        Parameters
        ----------
        genome : NetworkGenome
            The genome to evaluate.
        train_data : tuple[array, array] or similar
            Training data as ``(X, y)`` where X and y are array-like.
        val_data : tuple[array, array] or similar
            Validation data as ``(X, y)``.

        Returns
        -------
        float
            Fitness score (higher is better).  Negative MSE is used so that
            lower loss maps to higher fitness.
        """
        if _TORCH_AVAILABLE:
            return self._evaluate_torch(genome, train_data, val_data)
        return self._evaluate_python(genome, train_data, val_data)

    def _evaluate_torch(
        self, genome: NetworkGenome, train_data: Any, val_data: Any
    ) -> float:
        """Evaluate using PyTorch — build MLP, train briefly, measure val loss."""
        try:
            model = self._build_torch_model(genome)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            X_train = torch.tensor(train_data[0], dtype=torch.float32).to(device)
            y_train = torch.tensor(train_data[1], dtype=torch.float32).to(device)
            if y_train.dim() == 1:
                y_train = y_train.unsqueeze(1)

            X_val = torch.tensor(val_data[0], dtype=torch.float32).to(device)
            y_val = torch.tensor(val_data[1], dtype=torch.float32).to(device)
            if y_val.dim() == 1:
                y_val = y_val.unsqueeze(1)

            optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            # Quick training: 10 epochs
            model.train()
            for _ in range(10):
                optimiser.zero_grad()
                pred = model(X_train)
                loss = criterion(pred, y_train)
                loss.backward()
                optimiser.step()

            # Validation loss as fitness (negated — higher is better)
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            fitness = -val_loss
            genome.fitness = fitness
            return fitness

        except Exception as exc:
            logger.warning("Torch evaluation failed: %s — falling back", exc)
            return self._evaluate_python(genome, train_data, val_data)

    def _build_torch_model(self, genome: NetworkGenome) -> nn.Module:
        """Build a PyTorch Sequential model from a genome."""
        layers: list = []
        prev_dim = genome.input_dim

        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
        }

        for i, (size, act_name, drop) in enumerate(
            zip(genome.layers, genome.activations, genome.dropout_rates)
        ):
            layers.append(nn.Linear(prev_dim, size))
            act_cls = activation_map.get(act_name, nn.ReLU)
            layers.append(act_cls())
            if drop > 0:
                layers.append(nn.Dropout(p=drop))
            prev_dim = size

        layers.append(nn.Linear(prev_dim, genome.output_dim))
        return nn.Sequential(*layers)

    def _evaluate_python(
        self, genome: NetworkGenome, train_data: Any, val_data: Any
    ) -> float:
        """Pure-Python evaluation: random-weight forward pass on val data.

        Fitness proxy: negative mean squared error of a single forward pass
        with Xavier-initialised random weights.
        """
        try:
            X_val, y_val = val_data
            # Build weight matrices
            dims = [genome.input_dim] + genome.layers + [genome.output_dim]
            weights: list = []
            biases: list = []

            for i in range(len(dims) - 1):
                fan_in = dims[i]
                fan_out = dims[i + 1]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                w = [
                    [self._rng.gauss(0, std) for _ in range(fan_out)]
                    for _ in range(fan_in)
                ]
                b = [0.0] * fan_out
                weights.append(w)
                biases.append(b)

            # Forward pass for each validation sample
            total_se = 0.0
            n_samples = len(X_val)
            for idx in range(n_samples):
                x = list(X_val[idx]) if hasattr(X_val[idx], "__iter__") else [X_val[idx]]
                for layer_idx, (w, b) in enumerate(zip(weights, biases)):
                    new_x = []
                    for j in range(len(b)):
                        val = sum(x[k] * w[k][j] for k in range(len(x))) + b[j]
                        new_x.append(val)
                    # Apply activation (except last layer)
                    if layer_idx < len(weights) - 1:
                        act = genome.activations[layer_idx] if layer_idx < len(genome.activations) else "relu"
                        new_x = [self._apply_activation(v, act) for v in new_x]
                    x = new_x

                pred = x[0] if len(x) == 1 else sum(x) / len(x)
                target = y_val[idx] if not hasattr(y_val[idx], "__iter__") else y_val[idx][0]
                total_se += (pred - target) ** 2

            mse = total_se / max(n_samples, 1)
            fitness = -mse
            genome.fitness = fitness
            return fitness

        except Exception as exc:
            logger.warning("Python evaluation failed: %s", exc)
            genome.fitness = float("-inf")
            return genome.fitness

    @staticmethod
    def _apply_activation(x: float, act: str) -> float:
        """Apply an activation function to a scalar value."""
        if act == "relu":
            return max(0.0, x)
        elif act == "tanh":
            return math.tanh(x)
        elif act == "sigmoid":
            return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
        elif act == "leaky_relu":
            return x if x > 0 else 0.01 * x
        elif act == "elu":
            return x if x > 0 else (math.exp(max(-500, x)) - 1.0)
        elif act == "gelu":
            return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
        return x

    # ------------------------------------------------------------------
    # Evolution operators
    # ------------------------------------------------------------------

    def evolve(
        self,
        population: List[NetworkGenome],
        top_k: int = 5,
    ) -> List[NetworkGenome]:
        """Evolve the population for one generation.

        1. Select top-k elites by fitness.
        2. Fill remaining slots via tournament selection + crossover + mutation.

        Parameters
        ----------
        population : list[NetworkGenome]
            Current population (must have fitness scores assigned).
        top_k : int
            Number of elites to carry forward unchanged.

        Returns
        -------
        list[NetworkGenome]
            New population of the same size.
        """
        pop_size = len(population)
        if pop_size == 0:
            return []

        # Sort by fitness descending
        ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
        generation = ranked[0].generation + 1

        # Track best overall
        if self._best_genome is None or ranked[0].fitness > self._best_genome.fitness:
            self._best_genome = copy.deepcopy(ranked[0])

        # Elites
        new_pop: List[NetworkGenome] = []
        for g in ranked[:top_k]:
            elite = copy.deepcopy(g)
            elite.generation = generation
            new_pop.append(elite)

        # Fill rest via crossover + mutation
        while len(new_pop) < pop_size:
            parent_a = self._tournament_select(ranked)
            parent_b = self._tournament_select(ranked)
            child = self._crossover(parent_a, parent_b)
            child = self._mutate(child)
            child.generation = generation
            child.fitness = 0.0
            new_pop.append(child)

        logger.info(
            "Evolved generation %d: best_fitness=%.6f, pop_size=%d",
            generation, ranked[0].fitness, len(new_pop),
        )
        return new_pop

    def _tournament_select(
        self, ranked: List[NetworkGenome], k: int = 3
    ) -> NetworkGenome:
        """Tournament selection: pick k random individuals, return the fittest."""
        candidates = self._rng.sample(ranked, min(k, len(ranked)))
        return max(candidates, key=lambda g: g.fitness)

    def _crossover(
        self, parent_a: NetworkGenome, parent_b: NetworkGenome
    ) -> NetworkGenome:
        """Uniform crossover at the layer level.

        For each layer index, randomly pick from parent_a or parent_b.
        If parents have different depths, the child's depth is randomly
        chosen between the two.
        """
        min_len = min(len(parent_a.layers), len(parent_b.layers))
        max_len = max(len(parent_a.layers), len(parent_b.layers))
        child_len = self._rng.randint(min_len, max_len)

        layers = []
        activations = []
        dropout_rates = []

        for i in range(child_len):
            # Pick from whichever parent has this layer index
            candidates = []
            if i < len(parent_a.layers):
                candidates.append(("a", i))
            if i < len(parent_b.layers):
                candidates.append(("b", i))

            choice, idx = self._rng.choice(candidates)
            parent = parent_a if choice == "a" else parent_b
            layers.append(parent.layers[idx])
            activations.append(parent.activations[idx])
            dropout_rates.append(parent.dropout_rates[idx])

        return NetworkGenome(
            layers=layers,
            activations=activations,
            dropout_rates=dropout_rates,
            input_dim=parent_a.input_dim,
            output_dim=parent_a.output_dim,
        )

    def _mutate(self, genome: NetworkGenome) -> NetworkGenome:
        """Apply mutations to a genome.

        Possible mutations:
        - Resize a layer (gaussian perturbation of neuron count)
        - Swap activation function
        - Adjust dropout rate
        - Add a new layer
        - Remove a layer
        """
        g = copy.deepcopy(genome)

        for i in range(len(g.layers)):
            if self._rng.random() < self._mutation_rate:
                # Resize layer
                delta = self._rng.randint(-32, 32)
                g.layers[i] = max(self._min_neurons, min(self._max_neurons, g.layers[i] + delta))

            if self._rng.random() < self._mutation_rate:
                # Swap activation
                g.activations[i] = self._rng.choice(_ACTIVATIONS)

            if self._rng.random() < self._mutation_rate:
                # Adjust dropout
                g.dropout_rates[i] = round(
                    max(0.0, min(0.5, g.dropout_rates[i] + self._rng.gauss(0, 0.1))), 2
                )

        # Structural mutations
        if self._rng.random() < self._mutation_rate * 0.5 and len(g.layers) < self._max_layers:
            # Add layer
            pos = self._rng.randint(0, len(g.layers))
            size = self._rng.randint(self._min_neurons, self._max_neurons)
            g.layers.insert(pos, size)
            g.activations.insert(pos, self._rng.choice(_ACTIVATIONS))
            g.dropout_rates.insert(pos, round(self._rng.uniform(0.0, 0.3), 2))

        if self._rng.random() < self._mutation_rate * 0.3 and len(g.layers) > self._min_layers:
            # Remove layer
            idx = self._rng.randint(0, len(g.layers) - 1)
            g.layers.pop(idx)
            g.activations.pop(idx)
            g.dropout_rates.pop(idx)

        return g

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_best_genome(self) -> Optional[NetworkGenome]:
        """Return the best genome seen across all generations.

        Returns
        -------
        NetworkGenome | None
            The fittest genome, or ``None`` if no evaluation has occurred.
        """
        return copy.deepcopy(self._best_genome) if self._best_genome else None
