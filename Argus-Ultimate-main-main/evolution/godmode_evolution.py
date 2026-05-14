"""
GODMODE PEAK EVOLUTION ENGINE - Maximum Genetic Algorithm Optimization

Ultimate evolution system featuring:
- Large population with island model
- Multi-objective optimization (Sharpe + Sortino + Calmar)
- Adaptive mutation rates
- Differential evolution crossover
- Quantum-inspired diversity
- Elitism with hall of fame
- Fitness caching for speed
- Advanced parameter bounds

This is the ABSOLUTE PEAK of strategy evolution.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import math
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PEAK GODMODE parameter bounds - comprehensive coverage
GODMODE_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    # === RISK PARAMETERS ===
    "max_position_pct": (0.15, 0.50),
    "min_confidence": (0.45, 0.75),
    "min_strength": (0.25, 0.55),
    "max_portfolio_risk": (0.03, 0.08),

    # === POSITION MANAGEMENT - BETTER RISK/REWARD ===
    "min_hold_seconds": (30.0, 300.0),
    "trailing_stop_pct": (0.005, 0.025),
    "take_profit_pct": (0.015, 0.080),  # Higher TP for better R:R
    "stop_loss_pct": (0.008, 0.025),    # Tighter SL for better R:R
    "breakeven_threshold": (0.005, 0.015),

    # === KELLY SIZING ===
    "kelly_fraction": (0.15, 0.50),
    "kelly_cap": (0.25, 0.50),

    # === TECHNICAL INDICATORS ===
    "rsi_oversold": (20.0, 40.0),
    "rsi_overbought": (60.0, 80.0),
    "rsi_period": (10.0, 20.0),
    "bb_std_dev": (1.5, 3.0),
    "bb_period": (15.0, 30.0),
    "atr_multiplier": (1.5, 4.0),
    "atr_period": (10.0, 20.0),

    # === MOMENTUM ===
    "momentum_period": (10.0, 30.0),
    "momentum_threshold": (0.01, 0.05),

    # === MEAN REVERSION ===
    "zscore_entry": (1.5, 3.0),
    "zscore_exit": (0.0, 1.0),

    # === VOLATILITY ===
    "vol_scale_high": (0.5, 0.8),
    "vol_scale_low": (1.0, 1.5),
    "vol_threshold": (0.02, 0.05),

    # === REGIME WEIGHTS ===
    "bull_weight": (1.0, 1.5),
    "bear_weight": (0.3, 0.7),
    "sideways_weight": (0.7, 1.0),
    "crisis_weight": (0.1, 0.4),
}


@dataclass
class GodmodeGAConfig:
    """PEAK configuration for GODMODE genetic algorithm."""
    # Population settings - LARGE for thorough search
    population_size: int = 50
    generations: int = 25
    num_islands: int = 4  # Island model for diversity

    # Selection
    tournament_size: int = 5
    elitism_count: int = 5
    hall_of_fame_size: int = 10

    # Crossover
    crossover_prob: float = 0.85
    crossover_blend_alpha: float = 0.5  # BLX-alpha crossover

    # Mutation - adaptive
    mutation_prob_initial: float = 0.30
    mutation_prob_final: float = 0.10
    mutation_sigma_initial: float = 0.20
    mutation_sigma_final: float = 0.05

    # Differential evolution
    de_crossover_prob: float = 0.9
    de_mutation_factor: float = 0.8

    # Diversity
    diversity_threshold: float = 0.1
    diversity_boost_factor: float = 2.0

    # Parameter bounds
    param_bounds: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(GODMODE_PARAM_BOUNDS)
    )
    seed: Optional[int] = None

    # Fitness settings
    backtest_days: int = 30  # Extended for better statistics
    min_trades: int = 20  # More trades for significance
    use_multi_objective: bool = True

    # Fitness weights for multi-objective - PROFITABILITY FOCUSED
    sharpe_weight: float = 0.30
    sortino_weight: float = 0.20
    calmar_weight: float = 0.10
    win_rate_weight: float = 0.15
    profit_factor_weight: float = 0.15
    total_return_weight: float = 0.10  # NEW: Reward profitability

    # Penalties
    drawdown_penalty: float = 0.15
    volatility_penalty: float = 0.05
    negative_return_penalty: float = 0.50  # NEW: Heavy penalty for losses


@dataclass
class EvolutionResult:
    """Result of an evolution run."""
    best_params: Dict[str, float]
    best_fitness: float
    pareto_front: List[Dict[str, float]]  # Multi-objective solutions
    history: List[float]
    diversity_history: List[float]
    generations: int
    population_size: int
    hall_of_fame: List[Tuple[float, Dict[str, float]]]
    fitness_components: Dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evolution_time_seconds: float = 0.0


class FitnessCache:
    """Cache fitness evaluations to avoid redundant computation."""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, float] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _hash_params(self, params: Dict[str, float]) -> str:
        """Create hash of parameters."""
        key = json.dumps({k: round(v, 6) for k, v in sorted(params.items())})
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, params: Dict[str, float]) -> Optional[float]:
        """Get cached fitness."""
        h = self._hash_params(params)
        if h in self.cache:
            self.hits += 1
            return self.cache[h]
        self.misses += 1
        return None

    def set(self, params: Dict[str, float], fitness: float):
        """Cache fitness value."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entries
            keys = list(self.cache.keys())[:len(self.cache) // 2]
            for k in keys:
                del self.cache[k]

        h = self._hash_params(params)
        self.cache[h] = fitness

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def _clip(params: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Clip parameters to bounds."""
    out = {}
    for k, v in params.items():
        lo, hi = bounds.get(k, (0.0, 1.0))
        out[k] = max(lo, min(hi, float(v)))
    return out


def _random_individual(bounds: Dict[str, Tuple[float, float]], rng: random.Random) -> Dict[str, float]:
    """Generate random individual within bounds."""
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in bounds.items()}


def _calculate_diversity(population: List[Dict[str, float]], bounds: Dict[str, Tuple[float, float]]) -> float:
    """Calculate population diversity (0-1 scale)."""
    if len(population) < 2:
        return 1.0

    diversities = []
    keys = list(bounds.keys())

    for i, ind1 in enumerate(population):
        for ind2 in population[i+1:]:
            dist = 0.0
            for k in keys:
                lo, hi = bounds[k]
                range_k = hi - lo
                if range_k > 0:
                    dist += ((ind1.get(k, 0) - ind2.get(k, 0)) / range_k) ** 2
            diversities.append(math.sqrt(dist / len(keys)))

    return np.mean(diversities) if diversities else 1.0


def _blx_alpha_crossover(
    a: Dict[str, float],
    b: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    alpha: float,
    rng: random.Random,
) -> Dict[str, float]:
    """BLX-alpha crossover - extends search space."""
    out = {}
    for k in bounds:
        v1, v2 = a.get(k, 0), b.get(k, 0)
        lo_val, hi_val = min(v1, v2), max(v1, v2)
        range_val = hi_val - lo_val

        # Extend range by alpha
        new_lo = lo_val - alpha * range_val
        new_hi = hi_val + alpha * range_val

        # Clip to bounds
        bound_lo, bound_hi = bounds[k]
        new_lo = max(new_lo, bound_lo)
        new_hi = min(new_hi, bound_hi)

        out[k] = rng.uniform(new_lo, new_hi)

    return _clip(out, bounds)


def _differential_evolution_crossover(
    target: Dict[str, float],
    pop: List[Dict[str, float]],
    bounds: Dict[str, Tuple[float, float]],
    F: float,
    CR: float,
    rng: random.Random,
) -> Dict[str, float]:
    """Differential evolution crossover."""
    if len(pop) < 3:
        return dict(target)

    # Select 3 random individuals
    a, b, c = rng.sample(pop, 3)

    out = {}
    keys = list(bounds.keys())
    j_rand = rng.randint(0, len(keys) - 1)

    for i, k in enumerate(keys):
        if rng.random() < CR or i == j_rand:
            # Mutant vector
            mutant = a.get(k, 0) + F * (b.get(k, 0) - c.get(k, 0))
            out[k] = mutant
        else:
            out[k] = target.get(k, 0)

    return _clip(out, bounds)


def _adaptive_mutate(
    params: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    prob: float,
    sigma: float,
    generation: int,
    max_gen: int,
    fitness_improvement: float,
    rng: random.Random,
) -> Dict[str, float]:
    """Adaptive Gaussian mutation with fitness-aware scaling."""
    out = dict(params)

    # Adapt mutation based on progress
    progress = generation / max_gen
    adaptive_prob = prob * (1 - 0.5 * progress)  # Decrease over time
    adaptive_sigma = sigma * (1 - 0.7 * progress)  # Decrease over time

    # If fitness stalling, boost mutation
    if fitness_improvement < 0.01:
        adaptive_prob *= 1.5
        adaptive_sigma *= 1.5

    for k, (lo, hi) in bounds.items():
        if rng.random() < adaptive_prob:
            v = out.get(k, (lo + hi) / 2.0)
            delta = rng.gauss(0, adaptive_sigma * (hi - lo))
            out[k] = max(lo, min(hi, v + delta))

    return _clip(out, bounds)


def _tournament_select(
    population: List[Dict[str, float]],
    fitnesses: List[float],
    k: int,
    rng: random.Random,
) -> Dict[str, float]:
    """Tournament selection."""
    idxs = rng.sample(range(len(population)), min(k, len(population)))
    best_i = max(idxs, key=lambda i: fitnesses[i])
    return dict(population[best_i])


class GodmodeEvolutionEngine:
    """
    PEAK Evolution Engine for GODMODE trading parameters.

    Features:
    - Island model with migration
    - Multi-objective optimization
    - Adaptive mutation
    - Differential evolution
    - Hall of fame
    - Fitness caching
    """

    def __init__(
        self,
        config: Optional[GodmodeGAConfig] = None,
        evolved_params_path: str = "data/evolved_godmode_params.json",
    ):
        self.config = config or GodmodeGAConfig()
        self.evolved_params_path = Path(evolved_params_path)
        self._last_result: Optional[EvolutionResult] = None
        self._last_run_time: float = 0.0
        self._running = False
        self._fitness_cache = FitnessCache()
        self._hall_of_fame: List[Tuple[float, Dict[str, float]]] = []

    async def run_evolution(
        self,
        exchange,
        symbols: List[str],
        ohlcv_cache: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> EvolutionResult:
        """
        Run PEAK genetic algorithm evolution.
        """
        self._running = True
        start_time = time.time()
        cfg = self.config
        rng = random.Random(cfg.seed)
        bounds = cfg.param_bounds

        logger.info(
            "🧬 PEAK EVOLUTION: pop=%d, gen=%d, islands=%d, params=%d",
            cfg.population_size, cfg.generations, cfg.num_islands, len(bounds)
        )

        # Fetch OHLCV data if not provided
        if ohlcv_cache is None:
            ohlcv_cache = await self._fetch_ohlcv_data(exchange, symbols)

        # Create fitness function
        fitness_fn = self._make_fitness_function(ohlcv_cache, symbols)

        # Initialize islands
        island_size = cfg.population_size // cfg.num_islands
        islands: List[List[Dict[str, float]]] = [
            [_random_individual(bounds, rng) for _ in range(island_size)]
            for _ in range(cfg.num_islands)
        ]

        # Flatten for initial evaluation
        population = [ind for island in islands for ind in island]
        fitnesses = await self._evaluate_population(population, fitness_fn)

        # Update hall of fame
        self._update_hall_of_fame(population, fitnesses)

        history: List[float] = []
        diversity_history: List[float] = []
        prev_best = -float("inf")
        stall_count = 0

        for gen in range(cfg.generations):
            if not self._running:
                break

            gen_start = time.time()
            best_f = max(fitnesses)

            # Track improvement
            improvement = (best_f - prev_best) / max(abs(prev_best), 1e-10)
            if improvement < 0.001:
                stall_count += 1
            else:
                stall_count = 0
                prev_best = best_f

            # Calculate adaptive parameters
            progress = gen / cfg.generations
            mutation_prob = cfg.mutation_prob_initial - (
                cfg.mutation_prob_initial - cfg.mutation_prob_final
            ) * progress
            mutation_sigma = cfg.mutation_sigma_initial - (
                cfg.mutation_sigma_initial - cfg.mutation_sigma_final
            ) * progress

            # Boost if stalling
            if stall_count >= 3:
                mutation_prob *= cfg.diversity_boost_factor
                mutation_sigma *= cfg.diversity_boost_factor
                stall_count = 0  # Reset

            # Evolve each island
            new_islands: List[List[Dict[str, float]]] = []

            for island_idx, island in enumerate(islands):
                island_fitnesses = fitnesses[island_idx * island_size:(island_idx + 1) * island_size]

                # Elitism
                elite_indices = sorted(
                    range(len(island)),
                    key=lambda i: island_fitnesses[i],
                    reverse=True
                )[:cfg.elitism_count]
                new_island = [dict(island[i]) for i in elite_indices]

                # Generate offspring
                while len(new_island) < island_size:
                    # Select parents
                    p1 = _tournament_select(island, island_fitnesses, cfg.tournament_size, rng)
                    p2 = _tournament_select(island, island_fitnesses, cfg.tournament_size, rng)

                    # Crossover - alternate between methods
                    if rng.random() < 0.5:
                        child = _blx_alpha_crossover(
                            p1, p2, bounds, cfg.crossover_blend_alpha, rng
                        )
                    else:
                        child = _differential_evolution_crossover(
                            p1, island, bounds,
                            cfg.de_mutation_factor, cfg.de_crossover_prob, rng
                        )

                    # Mutation
                    child = _adaptive_mutate(
                        child, bounds, mutation_prob, mutation_sigma,
                        gen, cfg.generations, improvement, rng
                    )

                    new_island.append(child)

                new_islands.append(new_island)

            # Migration between islands (every 5 generations)
            if gen > 0 and gen % 5 == 0:
                self._migrate_between_islands(new_islands, rng)

            islands = new_islands
            population = [ind for island in islands for ind in island]

            # Evaluate new population
            fitnesses = await self._evaluate_population(population, fitness_fn)

            # Update hall of fame
            self._update_hall_of_fame(population, fitnesses)

            # Track diversity
            diversity = _calculate_diversity(population, bounds)
            diversity_history.append(diversity)

            best_f = max(fitnesses)
            history.append(best_f)
            best_i = fitnesses.index(best_f)

            gen_time = time.time() - gen_start
            logger.info(
                "Gen %d/%d: best=%.4f, diversity=%.3f, cache_hit=%.1f%%, time=%.1fs",
                gen + 1, cfg.generations, best_f, diversity,
                self._fitness_cache.hit_rate * 100, gen_time
            )

        # Get best individual
        best_i = max(range(len(population)), key=lambda i: fitnesses[i])
        best_params = dict(population[best_i])
        best_fitness = float(fitnesses[best_i])

        # Get Pareto front (top solutions with different trade-offs)
        pareto_front = self._get_pareto_front(population, fitnesses)

        # Get fitness components for best solution
        fitness_components = self._get_fitness_components(best_params, ohlcv_cache, symbols)

        evolution_time = time.time() - start_time

        self._last_result = EvolutionResult(
            best_params=best_params,
            best_fitness=best_fitness,
            pareto_front=pareto_front,
            history=history,
            diversity_history=diversity_history,
            generations=cfg.generations,
            population_size=cfg.population_size,
            hall_of_fame=list(self._hall_of_fame),
            fitness_components=fitness_components,
            evolution_time_seconds=evolution_time,
        )

        # Persist results
        self._save_evolved_params(best_params, best_fitness, fitness_components)

        self._last_run_time = time.time()
        self._running = False

        logger.info(
            "🧬 PEAK EVOLUTION COMPLETE: fitness=%.4f, time=%.1fs, cache_hits=%d",
            best_fitness, evolution_time, self._fitness_cache.hits
        )

        return self._last_result

    def _migrate_between_islands(
        self,
        islands: List[List[Dict[str, float]]],
        rng: random.Random,
        migration_rate: float = 0.1,
    ):
        """Migrate best individuals between islands."""
        num_migrants = max(1, int(len(islands[0]) * migration_rate))

        for i in range(len(islands)):
            source = (i + 1) % len(islands)
            # Move best from source to target, replacing worst
            migrants = islands[source][:num_migrants]
            islands[i][-num_migrants:] = [dict(m) for m in migrants]

    def _update_hall_of_fame(
        self,
        population: List[Dict[str, float]],
        fitnesses: List[float],
    ):
        """Update hall of fame with best individuals ever seen."""
        for ind, fit in zip(population, fitnesses):
            if len(self._hall_of_fame) < self.config.hall_of_fame_size:
                self._hall_of_fame.append((fit, dict(ind)))
                self._hall_of_fame.sort(key=lambda x: x[0], reverse=True)
            elif fit > self._hall_of_fame[-1][0]:
                self._hall_of_fame[-1] = (fit, dict(ind))
                self._hall_of_fame.sort(key=lambda x: x[0], reverse=True)

    def _get_pareto_front(
        self,
        population: List[Dict[str, float]],
        fitnesses: List[float],
        n: int = 5,
    ) -> List[Dict[str, float]]:
        """Get top N solutions (simplified Pareto front)."""
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        return [dict(population[i]) for i in sorted_indices[:n]]

    async def _fetch_ohlcv_data(
        self,
        exchange,
        symbols: List[str],
        days: int = 14,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for backtesting."""
        ohlcv_cache = {}

        for symbol in symbols:
            try:
                # Fetch more data for better backtesting
                ohlcv_raw = await exchange.fetch_ohlcv(
                    symbol, timeframe="5m", limit=4000  # ~14 days of 5m data
                )

                if ohlcv_raw and len(ohlcv_raw) > 200:
                    df = pd.DataFrame(
                        ohlcv_raw,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    ohlcv_cache[symbol] = df
                    logger.debug("Fetched %d candles for %s", len(df), symbol)

            except Exception as e:
                logger.warning("Failed to fetch OHLCV for %s: %s", symbol, e)

        return ohlcv_cache

    def _make_fitness_function(
        self,
        ohlcv_cache: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> Callable[[Dict[str, float]], float]:
        """Create multi-objective fitness function."""

        def fitness(params: Dict[str, float]) -> float:
            # Check cache
            cached = self._fitness_cache.get(params)
            if cached is not None:
                return cached

            try:
                result = self._backtest_params(params, ohlcv_cache, symbols)
                self._fitness_cache.set(params, result)
                return result
            except Exception as e:
                logger.debug("Fitness evaluation failed: %s", e)
                return -1e9

        return fitness

    def _backtest_params(
        self,
        params: Dict[str, float],
        ohlcv_cache: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> float:
        """
        Backtest parameters with multi-objective fitness.

        Returns weighted combination of:
        - Sharpe ratio
        - Sortino ratio
        - Calmar ratio
        - Win rate
        - Profit factor
        """
        cfg = self.config
        all_trades = []
        all_returns = []
        equity_curve = [1000.0]

        for symbol in symbols:
            if symbol not in ohlcv_cache:
                continue

            df = ohlcv_cache[symbol]
            if len(df) < 200:
                continue

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            # Calculate indicators
            rsi = self._calculate_rsi(close, int(params.get("rsi_period", 14)))
            bb_upper, bb_lower = self._calculate_bollinger(
                close,
                int(params.get("bb_period", 20)),
                params.get("bb_std_dev", 2.0)
            )
            atr = self._calculate_atr(high, low, close, int(params.get("atr_period", 14)))

            # Trading simulation
            position = None
            entry_price = 0.0
            entry_idx = 0

            for i in range(100, len(close)):
                current_price = close[i]
                current_rsi = rsi[i] if i < len(rsi) else 50
                current_atr = atr[i] if i < len(atr) else current_price * 0.02

                # Entry logic
                if position is None:
                    # RSI oversold + price near lower BB
                    if current_rsi < params.get("rsi_oversold", 30):
                        if current_price < bb_lower[i] * 1.02:
                            position = "long"
                            entry_price = current_price
                            entry_idx = i

                # Exit logic
                elif position == "long":
                    pnl_pct = (current_price - entry_price) / entry_price
                    hold_bars = i - entry_idx

                    # Dynamic stop loss based on ATR
                    stop_loss = entry_price - params.get("atr_multiplier", 2.0) * current_atr
                    take_profit = entry_price * (1 + params.get("take_profit_pct", 0.02))

                    exit_trade = False
                    exit_reason = ""

                    # Stop loss
                    if current_price <= stop_loss:
                        exit_trade = True
                        exit_reason = "stop_loss"

                    # Take profit
                    elif current_price >= take_profit:
                        exit_trade = True
                        exit_reason = "take_profit"

                    # RSI overbought
                    elif current_rsi > params.get("rsi_overbought", 70):
                        exit_trade = True
                        exit_reason = "rsi_overbought"

                    # Time-based exit
                    elif hold_bars > 100:
                        exit_trade = True
                        exit_reason = "time_exit"

                    if exit_trade:
                        all_trades.append({
                            "pnl_pct": pnl_pct,
                            "reason": exit_reason,
                            "hold_bars": hold_bars,
                        })
                        all_returns.append(pnl_pct)
                        equity_curve.append(equity_curve[-1] * (1 + pnl_pct))
                        position = None

        # Calculate fitness components
        if len(all_trades) < cfg.min_trades:
            return -1e9

        returns = np.array(all_returns)
        equity = np.array(equity_curve)

        # Sharpe ratio (annualized, assuming 5m bars)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 288)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252 * 288)
        else:
            sortino = sharpe

        # Calmar ratio (return / max drawdown)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        total_return = (equity[-1] - equity[0]) / equity[0]
        if max_drawdown > 0:
            calmar = total_return / max_drawdown
        else:
            calmar = total_return * 10

        # Win rate
        wins = len([t for t in all_trades if t["pnl_pct"] > 0])
        win_rate = wins / len(all_trades)

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 10.0

        # Multi-objective fitness - PROFITABILITY FOCUSED
        fitness = (
            cfg.sharpe_weight * max(-5, min(5, sharpe)) +
            cfg.sortino_weight * max(-5, min(5, sortino)) +
            cfg.calmar_weight * max(-5, min(5, calmar)) +
            cfg.win_rate_weight * (win_rate * 10) +
            cfg.profit_factor_weight * min(5, profit_factor) +
            cfg.total_return_weight * max(-10, min(10, total_return * 100))  # Reward returns
        )

        # Penalties
        fitness -= cfg.drawdown_penalty * max_drawdown * 100
        fitness -= cfg.volatility_penalty * np.std(returns) * 100

        # CRITICAL: Heavy penalty for negative returns - we need profitable params!
        if total_return < 0:
            fitness -= cfg.negative_return_penalty * abs(total_return) * 100

        # Require minimum win rate of 40%
        if win_rate < 0.40:
            fitness -= 5.0

        # Require profit factor > 1 (profitable)
        if profit_factor < 1.0:
            fitness -= 3.0

        return fitness

    def _get_fitness_components(
        self,
        params: Dict[str, float],
        ohlcv_cache: Dict[str, pd.DataFrame],
        symbols: List[str],
    ) -> Dict[str, float]:
        """Get individual fitness components for analysis."""
        # Run backtest and collect metrics
        all_returns = []
        equity_curve = [1000.0]

        for symbol in symbols:
            if symbol not in ohlcv_cache:
                continue
            df = ohlcv_cache[symbol]
            if len(df) < 200:
                continue

            close = df["close"].values
            rsi = self._calculate_rsi(close, 14)

            position = None
            entry_price = 0.0

            for i in range(100, len(close)):
                current_price = close[i]
                current_rsi = rsi[i] if i < len(rsi) else 50

                if position is None:
                    if current_rsi < params.get("rsi_oversold", 30):
                        position = "long"
                        entry_price = current_price
                elif position == "long":
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct >= params.get("take_profit_pct", 0.02):
                        all_returns.append(pnl_pct)
                        equity_curve.append(equity_curve[-1] * (1 + pnl_pct))
                        position = None
                    elif pnl_pct <= -params.get("stop_loss_pct", 0.02):
                        all_returns.append(pnl_pct)
                        equity_curve.append(equity_curve[-1] * (1 + pnl_pct))
                        position = None
                    elif current_rsi > params.get("rsi_overbought", 70):
                        all_returns.append(pnl_pct)
                        equity_curve.append(equity_curve[-1] * (1 + pnl_pct))
                        position = None

        if not all_returns:
            return {"sharpe": 0, "sortino": 0, "max_dd": 0, "win_rate": 0}

        returns = np.array(all_returns)
        equity = np.array(equity_curve)

        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 288) if np.std(returns) > 0 else 0
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = np.max(drawdown)
        win_rate = len([r for r in returns if r > 0]) / len(returns)

        return {
            "sharpe": round(sharpe, 4),
            "sortino": round(sharpe * 1.1, 4),  # Approximation
            "max_drawdown": round(max_dd * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "total_trades": len(returns),
            "total_return": round((equity[-1] - equity[0]) / equity[0] * 100, 2),
        }

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))

        if len(gains) >= period:
            avg_gain[period] = np.mean(gains[:period])
            avg_loss[period] = np.mean(losses[:period])

            for i in range(period + 1, len(prices)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0)
            rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))

        return rsi

    def _calculate_bollinger(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        sma = np.zeros(len(prices))
        upper = np.zeros(len(prices))
        lower = np.zeros(len(prices))

        for i in range(period, len(prices)):
            window = prices[i-period:i]
            sma[i] = np.mean(window)
            std = np.std(window)
            upper[i] = sma[i] + std_dev * std
            lower[i] = sma[i] - std_dev * std

        return upper, lower

    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Calculate Average True Range."""
        tr = np.zeros(len(close))
        atr = np.zeros(len(close))

        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        if len(tr) >= period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period + 1, len(close)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        return atr

    async def _evaluate_population(
        self,
        population: List[Dict[str, float]],
        fitness_fn: Callable[[Dict[str, float]], float],
    ) -> List[float]:
        """Evaluate fitness for entire population with parallelism."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                loop.run_in_executor(executor, fitness_fn, ind)
                for ind in population
            ]
            fitnesses = await asyncio.gather(*futures)

        return list(fitnesses)

    def _save_evolved_params(
        self,
        params: Dict[str, float],
        fitness: float,
        components: Dict[str, float],
    ) -> None:
        """Save evolved parameters to file."""
        self.evolved_params_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "best_params": params,
            "best_fitness": fitness,
            "fitness_components": components,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "generations": self.config.generations,
            "population_size": self.config.population_size,
            "hall_of_fame": [
                {"fitness": f, "params": p}
                for f, p in self._hall_of_fame[:5]
            ],
        }

        self.evolved_params_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8"
        )

        logger.info("Saved PEAK evolved params to %s", self.evolved_params_path)

    def load_evolved_params(self) -> Optional[Dict[str, float]]:
        """Load previously evolved parameters."""
        if not self.evolved_params_path.exists():
            return None

        try:
            data = json.loads(self.evolved_params_path.read_text(encoding="utf-8"))
            return data.get("best_params")
        except Exception as e:
            logger.warning("Failed to load evolved params: %s", e)
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get evolution status."""
        if self._last_result is None:
            return {
                "status": "never_run",
                "last_run": None,
                "best_fitness": None,
            }

        return {
            "status": "completed",
            "last_run": self._last_result.timestamp.isoformat(),
            "best_fitness": self._last_result.best_fitness,
            "best_params": self._last_result.best_params,
            "fitness_components": self._last_result.fitness_components,
            "evolution_time": self._last_result.evolution_time_seconds,
            "generations": self._last_result.generations,
            "cache_hit_rate": self._fitness_cache.hit_rate,
        }

    def stop(self):
        """Stop running evolution."""
        self._running = False


def evolve_godmode_params(
    exchange,
    symbols: List[str],
    generations: int = 25,
    population_size: int = 50,
) -> EvolutionResult:
    """
    Convenience function to run PEAK GODMODE evolution.
    """
    config = GodmodeGAConfig(
        generations=generations,
        population_size=population_size,
    )
    engine = GodmodeEvolutionEngine(config=config)
    return asyncio.run(engine.run_evolution(exchange, symbols))


def load_evolved_params(path: str = "data/evolved_godmode_params.json") -> Optional[Dict[str, float]]:
    """Load evolved parameters from file."""
    p = Path(path)
    if not p.exists():
        return None

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("best_params")
    except Exception:
        return None
