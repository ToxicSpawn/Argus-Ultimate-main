"""
Unified evolution entrypoint: evolve_once + apply_evolved_params.

Used by adaptive/self_improver.py and evolution/continuous_evolution_engine.py
to run GA-based evolution (paper-loop fitness) and apply best params to config.
Supports dry_run, backup, versioning, composite fitness, walk-forward, seed from file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from evolution.apply_evolved_strategies import (
    apply_to_config,
    load_last_best_params,
    write_evolved_params,
)
from evolution.param_space import get_bounds
from evolution.strategy_genetic_algorithm import (
    GAConfig,
    make_paper_loop_fitness,
    run_genetic_algorithm,
)

logger = logging.getLogger(__name__)


def evolve_once(
    generations: int = 5,
    population_size: int = 12,
    fitness_days: int = 7,
    config_path: str = "unified_config.yaml",
    timeframe: str = "1h",
    use_sharpe: bool = True,
    persist: bool = True,
    evolved_params_path: str = "data/evolved_params.json",
    ohlcv_by_symbol_override: Optional[Dict[str, Any]] = None,
    warmup: Optional[int] = None,
    dry_run: bool = False,
    source: str = "interval",
    min_trades: int = 1,
    drawdown_penalty_weight: float = 0.05,
    negative_return_penalty_weight: float = 0.0,
    use_composite_fitness: bool = False,
    walk_forward_train_ratio: Optional[float] = None,
    evolution_seed: Optional[int] = None,
    early_stop_generations: int = 0,
    early_stop_threshold: float = 0.001,
    fitness_cache_size: int = 0,
    parallel_fitness_workers: int = 0,
    mutation_prob: Optional[float] = None,
    mutation_sigma: Optional[float] = None,
    crossover_prob: Optional[float] = None,
    backup_before_apply: bool = True,
    version_history_size: int = 0,
    overfit_penalty_weight: float = 0.0,
    volatility_penalty_weight: float = 0.0,
    composite_calmar_weight: float = 0.0,
    multi_timeframes: Optional[List[str]] = None,
    multi_timeframe_weights: Optional[List[float]] = None,
    strategy_whitelist_override: Optional[List[str]] = None,
    on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run one GA evolution using paper-loop fitness. Returns result dict with
    best_params, best_fitness, history. When dry_run=True, does not persist or apply.
    on_complete: optional callback(result) after persist (e.g. for metrics).
    """
    logger.info(
        "Evolution run started source=%s fitness_days=%s generations=%s population_size=%s dry_run=%s",
        source, fitness_days, generations, population_size, dry_run,
    )
    fitness_fn = make_paper_loop_fitness(
        config_path=config_path,
        days=int(fitness_days),
        timeframe=timeframe,
        use_sharpe=use_sharpe,
        ohlcv_by_symbol_override=ohlcv_by_symbol_override,
        warmup=warmup,
        penalize_drawdown=True,
        drawdown_penalty_weight=drawdown_penalty_weight,
        min_trades=min_trades,
        use_quantum_bot=bool(kwargs.get("use_quantum_bot", False)),
        negative_return_penalty_weight=negative_return_penalty_weight,
        use_composite=use_composite_fitness,
        composite_sharpe_weight=kwargs.get("composite_sharpe_weight", 0.6),
        composite_sortino_weight=kwargs.get("composite_sortino_weight", 0.2),
        composite_calmar_weight=composite_calmar_weight,
        walk_forward_train_ratio=walk_forward_train_ratio,
        overfit_penalty_weight=overfit_penalty_weight,
        volatility_penalty_weight=volatility_penalty_weight,
        multi_timeframes=multi_timeframes,
        multi_timeframe_weights=multi_timeframe_weights,
        strategy_whitelist_override=strategy_whitelist_override,
    )
    bounds = get_bounds()
    seed_from = None if dry_run else load_last_best_params(evolved_params_path)
    ga_config = GAConfig(
        population_size=population_size,
        generations=generations,
        param_bounds=bounds,
        seed=evolution_seed,
        early_stop_generations=early_stop_generations or 0,
        early_stop_threshold=early_stop_threshold,
        fitness_cache_size=fitness_cache_size or 0,
        parallel_fitness_workers=parallel_fitness_workers or 0,
        seed_from_params=seed_from,
        mutation_prob=mutation_prob if mutation_prob is not None else 0.2,
        mutation_sigma=mutation_sigma if mutation_sigma is not None else 0.15,
        crossover_prob=crossover_prob if crossover_prob is not None else 0.7,
    )
    best_params, best_fitness, history = run_genetic_algorithm(fitness_fn, ga_config)
    result: Dict[str, Any] = {
        "best_params": best_params,
        "best_fitness": best_fitness,
        "history": history,
        "source": source,
    }
    if not dry_run and persist and best_params:
        path = Path(evolved_params_path)
        try:
            write_evolved_params(
                best_params,
                path=path,
                meta={
                    "best_fitness": best_fitness,
                    "generations": generations,
                    "population_size": population_size,
                    "fitness_days": fitness_days,
                    "source": source,
                },
                backup_before=backup_before_apply,
                version_history_size=version_history_size,
            )
        except Exception as e:
            logger.warning("Failed to persist evolved params to %s: %s", path, e)
    logger.info(
        "Evolution run finished source=%s best_fitness=%.4f params_count=%s",
        source, best_fitness, len(best_params),
    )
    if on_complete and callable(on_complete):
        try:
            on_complete(result)
        except Exception as e:
            logger.debug("Evolution on_complete callback failed: %s", e)
    return result


def apply_evolved_params(config: Any, params: Dict[str, Any], filter_to_config: bool = True) -> int:
    """
    Apply evolved params to config. Returns number of attributes set.
    When filter_to_config=True, only applies keys that exist on config (avoids no-op).
    """
    if filter_to_config and config is not None:
        try:
            attrs = set(dir(config))
            from evolution.param_space import filter_to_config_keys as _filter
            params = _filter(params, attrs)
        except Exception as _e:
            logger.debug("evolution_unified error: %s", _e)
    return apply_to_config(config, params)
