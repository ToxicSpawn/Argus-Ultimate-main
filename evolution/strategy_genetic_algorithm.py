"""
Genetic algorithm for evolving strategy and brain parameters.

Evolves a fixed set of tunables (se_* RSI/BB, optional min_signal_confidence)
using tournament selection, uniform crossover, and gaussian mutation.
Fitness is backtest Sharpe (or return_pct) via the unified paper loop.
Uses evolution.param_space as single source of truth for bounds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Single source of truth: evolution.param_space.EVOLVABLE_PARAM_BOUNDS; kept in sync here for GA default.
DEFAULT_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "se_buy_rsi": (20.0, 45.0),
    "se_sell_rsi": (55.0, 80.0),
    "se_buy_bb": (0.10, 0.45),
    "se_sell_bb": (0.55, 0.90),
    "se_trend_rsi_buy": (45.0, 65.0),
    "se_trend_rsi_sell": (35.0, 55.0),
    "min_signal_confidence": (0.55, 0.85),
}


@dataclass
class GAConfig:
    population_size: int = 20
    generations: int = 10
    tournament_size: int = 3
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    mutation_sigma: float = 0.15
    elitism_count: int = 1
    param_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PARAM_BOUNDS))
    seed: Optional[int] = None
    # Early stopping
    early_stop_generations: int = 0  # 0 = disabled
    early_stop_threshold: float = 0.001  # stop when improvement < this
    # Optional fitness cache (max size; 0 = disabled)
    fitness_cache_size: int = 0
    # Optional parallelism (max_workers; 0 = sequential)
    parallel_fitness_workers: int = 0
    # Seed one individual from previous best (faster refinement)
    seed_from_params: Optional[Dict[str, float]] = None


def _clip(params: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    out = {}
    for k, v in params.items():
        lo, hi = bounds.get(k, (0.0, 1.0))
        out[k] = max(lo, min(hi, float(v)))
    return out


class _LRUFitnessCache:
    """LRU cache for fitness evaluations."""

    def __init__(self, max_size: int):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def _key(self, params: Dict[str, float]) -> str:
        return hashlib.md5(json.dumps({k: round(v, 6) for k, v in sorted(params.items())}).encode()).hexdigest()

    def get(self, params: Dict[str, float]) -> Optional[float]:
        k = self._key(params)
        if k in self._cache:
            self.hits += 1
            self._cache.move_to_end(k)
            return self._cache[k]
        self.misses += 1
        return None

    def set(self, params: Dict[str, float], value: float) -> None:
        k = self._key(params)
        if k in self._cache:
            self._cache.move_to_end(k)
        self._cache[k] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


def _random_individual(bounds: Dict[str, Tuple[float, float]], rng: random.Random) -> Dict[str, float]:
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in bounds.items()}


def _crossover(
    a: Dict[str, float],
    b: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    prob: float,
    rng: random.Random,
) -> Dict[str, float]:
    out = {}
    for k in bounds:
        if rng.random() < prob:
            out[k] = (float(a.get(k, 0)) + float(b.get(k, 0))) / 2.0
        else:
            out[k] = float(rng.choice([a.get(k), b.get(k)]))
    return _clip(out, bounds)


def _mutate(
    params: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    prob: float,
    sigma: float,
    rng: random.Random,
) -> Dict[str, float]:
    out = dict(params)
    for k, (lo, hi) in bounds.items():
        if rng.random() < prob:
            v = out.get(k, (lo + hi) / 2.0)
            delta = rng.gauss(0, sigma * (hi - lo))
            out[k] = max(lo, min(hi, v + delta))
    return _clip(out, bounds)


def _tournament_select(
    population: List[Dict[str, float]],
    fitnesses: List[float],
    k: int,
    rng: random.Random,
) -> Dict[str, float]:
    idxs = rng.sample(range(len(population)), min(k, len(population)))
    best_i = max(idxs, key=lambda i: fitnesses[i])
    return dict(population[best_i])


def _eval_fitness_batch(
    population: List[Dict[str, float]],
    fitness_fn: Callable[[Dict[str, float]], float],
    cache: Optional[_LRUFitnessCache],
    workers: int,
) -> List[float]:
    if workers <= 0 or len(population) <= 1:
        out = []
        for p in population:
            if cache:
                v = cache.get(p)
                if v is not None:
                    out.append(v)
                    continue
                v = fitness_fn(p)
                cache.set(p, v)
                out.append(v)
            else:
                out.append(fitness_fn(p))
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        def eval_one(i: int) -> Tuple[int, float]:
            p = population[i]
            if cache:
                v = cache.get(p)
                if v is not None:
                    return (i, v)
                return (i, fitness_fn(p))
        futures = [ex.submit(eval_one, i) for i in range(len(population))]
        results = [f.result() for f in as_completed(futures)]
    out = [0.0] * len(population)
    for i, v in results:
        out[i] = v
        if cache and i < len(population):
            cache.set(population[i], v)
    return out


def run_genetic_algorithm(
    fitness_fn: Callable[[Dict[str, float]], float],
    config: Optional[GAConfig] = None,
) -> Tuple[Dict[str, float], float, List[float]]:
    """
    Run the GA and return (best_params, best_fitness, history_of_best_fitness).
    fitness_fn(individual) -> float (higher is better).
    Supports early stopping, optional LRU fitness cache, parallel evaluation, and seed from previous best.
    """
    cfg = config or GAConfig()
    rng = random.Random(cfg.seed)

    bounds = cfg.param_bounds
    cache: Optional[_LRUFitnessCache] = _LRUFitnessCache(cfg.fitness_cache_size) if cfg.fitness_cache_size > 0 else None
    workers = max(0, getattr(cfg, "parallel_fitness_workers", 0))

    population: List[Dict[str, float]] = []
    for i in range(cfg.population_size):
        if i == 0 and cfg.seed_from_params:
            population.append(_clip(dict(cfg.seed_from_params), bounds))
        else:
            population.append(_random_individual(bounds, rng))

    fitnesses = _eval_fitness_batch(population, fitness_fn, cache, workers)
    history: List[float] = []

    stall_count = 0
    prev_best = -float("inf")
    early_stop_count = 0
    base_mutation_prob = cfg.mutation_prob
    early_stop_gens = getattr(cfg, "early_stop_generations", 0) or 0
    early_stop_thr = getattr(cfg, "early_stop_threshold", 0.001) or 0.001

    for gen in range(cfg.generations):
        best_f = max(fitnesses)
        improvement = best_f - prev_best if prev_best > -float("inf") else 1.0
        if improvement < early_stop_thr and early_stop_gens > 0:
            early_stop_count += 1
            if early_stop_count >= early_stop_gens:
                logger.debug("GA early stop at gen %s (improvement < %.4f for %s gens)", gen + 1, early_stop_thr, early_stop_gens)
                break
        else:
            early_stop_count = 0

        if best_f <= prev_best:
            stall_count += 1
        else:
            stall_count = 0
            prev_best = best_f

        mutation_prob_eff = min(0.5, base_mutation_prob * (1.5 if stall_count >= 2 else 1.0))

        order = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        new_pop: List[Dict[str, float]] = [dict(population[order[i]]) for i in range(cfg.elitism_count)]

        while len(new_pop) < cfg.population_size:
            p1 = _tournament_select(population, fitnesses, cfg.tournament_size, rng)
            p2 = _tournament_select(population, fitnesses, cfg.tournament_size, rng)
            child = _crossover(p1, p2, bounds, cfg.crossover_prob, rng)
            child = _mutate(child, bounds, mutation_prob_eff, cfg.mutation_sigma, rng)
            new_pop.append(child)

        population = new_pop
        fitnesses = _eval_fitness_batch(population, fitness_fn, cache, workers)
        best_f = max(fitnesses)
        history.append(best_f)
        best_i = fitnesses.index(best_f)
        logger.debug("GA gen %s best_fitness=%.4f", gen + 1, best_f)
        if (gen + 1) % 5 == 0:
            logger.debug("GA best params sample: %s", {k: round(population[best_i][k], 3) for k in list(bounds)[:3]})
    if cache:
        logger.debug("Fitness cache hit_rate=%.2f", cache.hits / max(1, cache.hits + cache.misses))

    best_i = max(range(len(population)), key=lambda i: fitnesses[i])
    return dict(population[best_i]), float(fitnesses[best_i]), history


def make_paper_loop_fitness(
    config_path: str = "unified_config.yaml",
    days: int = 7,
    timeframe: str = "5m",
    use_sharpe: bool = True,
    ohlcv_by_symbol_override: Optional[Dict[str, Any]] = None,
    warmup: Optional[int] = None,
    penalize_drawdown: bool = True,
    drawdown_penalty_weight: float = 0.05,
    min_trades: int = 1,
    use_quantum_bot: bool = False,
    negative_return_penalty_weight: float = 0.0,
    use_composite: bool = False,
    composite_sharpe_weight: float = 0.6,
    composite_sortino_weight: float = 0.2,
    composite_calmar_weight: float = 0.0,
    walk_forward_train_ratio: Optional[float] = None,
    overfit_penalty_weight: float = 0.0,
    volatility_penalty_weight: float = 0.0,
    multi_timeframes: Optional[List[str]] = None,
    multi_timeframe_weights: Optional[List[float]] = None,
    strategy_whitelist_override: Optional[List[str]] = None,
) -> Callable[[Dict[str, float]], float]:
    """
    Build a fitness function that runs the unified paper loop with cfg_overrides=individual.
    use_composite: weighted sharpe + sortino + calmar - drawdown - neg_return - volatility.
    walk_forward_train_ratio: train/test split; overfit_penalty_weight penalizes train>>test.
    multi_timeframes: run on each TF and combine (weighted avg).
    strategy_whitelist_override: merge into cfg_overrides so only these strategies run in backtest.
    """
    _ohlcv = ohlcv_by_symbol_override
    _warmup = warmup
    _penalize = penalize_drawdown
    _dd_weight = float(drawdown_penalty_weight)
    _min_trades = int(min_trades)
    _use_quantum_bot = bool(use_quantum_bot)
    _neg_weight = float(negative_return_penalty_weight)
    _composite = bool(use_composite)
    _wf_ratio = walk_forward_train_ratio
    _overfit_weight = float(overfit_penalty_weight)
    _vol_weight = float(volatility_penalty_weight)
    _calmar_weight = float(composite_calmar_weight)
    _multi_tf = list(multi_timeframes) if multi_timeframes else None
    _tf_weights = list(multi_timeframe_weights) if (multi_timeframes and multi_timeframe_weights and len(multi_timeframe_weights) == len(multi_timeframes)) else ([1.0 / len(_multi_tf)] * len(_multi_tf) if _multi_tf else [])
    _strategy_whitelist = list(strategy_whitelist_override) if strategy_whitelist_override else None

    def _run_and_score(
        individual: Dict[str, float],
        days_run: int,
        ohlcv_override: Optional[Dict[str, Any]] = None,
        tf_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        from scripts.paper_loop_30d_unified import run_paper_loop
        overrides = dict(individual)
        if _strategy_whitelist:
            overrides["strategy_whitelist"] = _strategy_whitelist
        kwargs: Dict[str, Any] = {
            "config_path": config_path,
            "days": days_run,
            "timeframe": tf_override or timeframe,
            "cfg_overrides": overrides,
        }
        if ohlcv_override is not None:
            kwargs["ohlcv_by_symbol_override"] = ohlcv_override
        if _warmup is not None:
            kwargs["warmup"] = _warmup
        kwargs["use_quantum_bot"] = _use_quantum_bot
        return run_paper_loop(**kwargs)

    def _score_one(results: Dict[str, Any]) -> float:
        trades = int(results.get("trades", 0) or 0)
        if trades < _min_trades:
            return -1e9
        raw: float
        if use_sharpe:
            sharpe = results.get("sharpe")
            if sharpe is not None and isinstance(sharpe, (int, float)) and not (isinstance(sharpe, float) and sharpe != sharpe):
                raw = float(sharpe)
            else:
                ret = results.get("return_pct")
                raw = float(ret) if ret is not None else -1e9
        else:
            ret = results.get("return_pct")
            raw = float(ret) if ret is not None else -1e9
        if _penalize and raw > -1e8:
            max_dd = float(results.get("max_drawdown_pct", 0) or 0)
            raw = raw - _dd_weight * max_dd
        if _neg_weight > 0 and raw > -1e8:
            ret_pct = float(results.get("return_pct", 0) or 0)
            if ret_pct < 0:
                raw = raw - _neg_weight * abs(ret_pct)
        if _vol_weight > 0 and raw > -1e8:
            vol = results.get("returns_volatility") or results.get("volatility")
            if vol is not None and isinstance(vol, (int, float)):
                raw = raw - _vol_weight * float(vol)
        return raw

    def _composite_score(results: Dict[str, Any]) -> float:
        trades = int(results.get("trades", 0) or 0)
        if trades < _min_trades:
            return -1e9
        sharpe = results.get("sharpe")
        sharpe_v = float(sharpe) if sharpe is not None and isinstance(sharpe, (int, float)) and not (isinstance(sharpe, float) and sharpe != sharpe) else 0.0
        sortino = results.get("sortino")
        sortino_v = float(sortino) if sortino is not None and isinstance(sortino, (int, float)) and not (isinstance(sortino, float) and sortino != sortino) else sharpe_v
        max_dd_pct = float(results.get("max_drawdown_pct", 0) or 0)
        max_dd = max_dd_pct / 100.0 if max_dd_pct else 0.001
        ret_pct = float(results.get("return_pct", 0) or 0) / 100.0
        calmar_v = ret_pct / max_dd if max_dd > 0 else 0.0
        raw = composite_sharpe_weight * sharpe_v + composite_sortino_weight * sortino_v + _calmar_weight * calmar_v - _dd_weight * max_dd
        if ret_pct < 0 and _neg_weight > 0:
            raw -= _neg_weight * abs(ret_pct)
        if _vol_weight > 0:
            vol = results.get("returns_volatility") or results.get("volatility")
            if vol is not None and isinstance(vol, (int, float)):
                raw -= _vol_weight * float(vol)
        return raw

    def _single_run_fitness(individual: Dict[str, float], days_run: int, ohlcv: Optional[Dict[str, Any]], tf: Optional[str]) -> float:
        res = _run_and_score(individual, days_run, ohlcv, tf_override=tf)
        if _composite:
            return _composite_score(res)
        return _score_one(res)

    def fitness(individual: Dict[str, float]) -> float:
        try:
            if _multi_tf:
                scores: List[float] = []
                for i, tf in enumerate(_multi_tf):
                    w = _tf_weights[i] if i < len(_tf_weights) else 1.0 / len(_multi_tf)
                    s = _single_run_fitness(individual, days, _ohlcv, tf)
                    if s <= -1e9:
                        return -1e9
                    scores.append(s * w)
                return sum(scores)
            if _wf_ratio is not None and 0 < _wf_ratio < 1 and days > 3:
                train_days = max(2, int(days * _wf_ratio))
                test_days = max(1, days - train_days)
                res_train = _run_and_score(individual, train_days, _ohlcv)
                res_test = _run_and_score(individual, test_days, _ohlcv)
                if _composite:
                    s1 = _composite_score(res_train)
                    s2 = _composite_score(res_test)
                else:
                    s1 = _score_one(res_train)
                    s2 = _score_one(res_test)
                if s1 <= -1e9 or s2 <= -1e9:
                    return -1e9
                combined = _wf_ratio * s1 + (1.0 - _wf_ratio) * s2
                if _overfit_weight > 0 and s1 > s2:
                    combined -= _overfit_weight * (s1 - s2)
                return combined
            results = _run_and_score(individual, days, _ohlcv)
            if _composite:
                return _composite_score(results)
            return _score_one(results)
        except Exception as e:
            logger.warning("Fitness run failed: %s", e)
            return -1e9

    return fitness
