from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShadowTuneResult:
    baseline_return_pct: float
    candidate_return_pct: float
    baseline_trades: int
    candidate_trades: int
    baseline_maxdd_pct: float
    candidate_maxdd_pct: float
    applied: bool
    reason: str
    quantum_enabled: Optional[bool] = None
    quantum_method: Optional[str] = None
    quantum_strength: Optional[float] = None


class SelfImprover:
    """
    Self-improvement loop:
    - Always-on online learning already happens via StrategyEngine tuner + allocator feedback.
    - This loop periodically runs a *shadow tuning* pass (paper/backtest modes by default)
      and applies new optimized params only if they beat the baseline with guards.
    """

    def __init__(self, *, system: Any) -> None:
        self.system = system
        self.cfg = getattr(system, "config", None)

    def _state_path(self) -> Path:
        p = str(getattr(self.cfg, "self_improvement_state_path", "data/self_improvement_state.json") or "data/self_improvement_state.json")
        return Path(p)

    def _load_state(self) -> Dict[str, Any]:
        p = self._state_path()
        try:
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                return d if isinstance(d, dict) else {}
        except Exception as e:
            logger.debug("Failed to load self-improvement state: %s", e)
            return {}
        return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        p = self._state_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to save self-improvement state: %s", e)
            return

    def record_trade_closed(self) -> None:
        """
        Call when a trade closes so evolution can run instantaneously (after N trades).
        Increments trades_since_evolution; SelfImprover tick will run evolution when count >= evolution_after_n_trades.
        """
        if not getattr(self.cfg, "evolution_trigger_on_trade", False):
            return
        try:
            st = self._load_state()
            n = int(st.get("trades_since_evolution", 0) or 0) + 1
            st["trades_since_evolution"] = n
            self._save_state(st)
        except Exception as e:
            logger.debug("record_trade_closed failed: %s", e)

    def _enabled_in_mode(self) -> bool:
        try:
            mode = str(getattr(self.cfg, "run_mode", "paper") or "paper").lower()
            modes = list(getattr(self.cfg, "self_improvement_modes", ["paper", "backtest"]) or ["paper", "backtest"])
            return bool(getattr(self.cfg, "self_improvement_enabled", True)) and mode in [str(m).lower() for m in modes]
        except Exception as e:
            logger.debug("_enabled_in_mode check failed: %s", e)
            return False

    async def run_forever(self) -> None:
        # Keep this loop cheap; it should never destabilize trading.
        while True:
            try:
                if not self._enabled_in_mode():
                    await asyncio.sleep(30.0)
                    continue

                tick_s = float(getattr(self.cfg, "self_improvement_tick_seconds", 1) or 1)
                tick_s = float(max(1.0, min(60.0, tick_s)))

                # Cheap per-tick improvements (every second):
                # - hot-reload optimized params file (StrategyEngine also does this per call, but be proactive)
                try:
                    brain = getattr(self.system, "ai_brain", None)
                    se = getattr(brain, "strategy_engine", None) if brain is not None else None
                    if se is not None and hasattr(se, "_load_opt_params"):
                        se._load_opt_params()  # type: ignore[attr-defined]
                except Exception as e:
                    logger.debug("Hot-reload opt params failed: %s", e)

                # Reload evolved params from file when file is updated (continuous adaptation)
                try:
                    if getattr(self.cfg, "evolution_continuous_enabled", True):
                        path = Path(str(getattr(self.cfg, "evolution_params_path", "data/evolved_params.json") or "data/evolved_params.json"))
                        if path.exists():
                            mtime = path.stat().st_mtime
                            st = self._load_state()
                            last_mtime = float(st.get("last_evolved_file_mtime", 0) or 0)
                            if mtime > last_mtime:
                                from evolution.apply_evolved_strategies import apply_from_file
                                n = apply_from_file(self.cfg, path=path, key="best_params")
                                if n > 0:
                                    st["last_evolved_file_mtime"] = mtime
                                    self._save_state(st)
                                    logger.debug("Evolution: reloaded %s params from %s", n, path)
                except Exception as e:
                    logger.debug("Evolution reload: %s", e)

                # Continuous evolution: run GA on schedule and/or instantaneously after N trades
                try:
                    continuous = getattr(self.cfg, "evolution_continuous_enabled", True)
                    run_with_market = getattr(self.cfg, "evolution_run_with_market", True)
                    trigger_on_trade = getattr(self.cfg, "evolution_trigger_on_trade", False)
                    after_n_trades = int(getattr(self.cfg, "evolution_after_n_trades", 5) or 5)
                    debounce_min = float(getattr(self.cfg, "evolution_debounce_minutes", 15.0) or 15.0)
                    debounce_sec = max(0.0, debounce_min * 60.0)
                    st = self._load_state()
                    now = time.time()
                    last_ev = float(st.get("last_evolution_ts", 0) or 0)
                    trades_since = int(st.get("trades_since_evolution", 0) or 0)
                    fitness_days = float(getattr(self.cfg, "evolution_realtime_fitness_days", 1.0) or 1.0)

                    # Time-based: run when interval has elapsed
                    interval_elapsed = False
                    realtime_min = 0.0
                    if continuous:
                        realtime_min = float(getattr(self.cfg, "evolution_realtime_interval_minutes", 60.0) or 0.0)
                        if run_with_market and realtime_min > 0:
                            interval_s = max(60.0, realtime_min * 60.0)
                            fitness_days = float(getattr(self.cfg, "evolution_realtime_fitness_days", 1.0) or 1.0)
                        else:
                            interval_h = float(getattr(self.cfg, "evolution_interval_hours", 24.0) or 24.0)
                            interval_s = max(3600.0, interval_h * 3600.0)
                            fitness_days = float(getattr(self.cfg, "evolution_fitness_days", 7) or 7)
                        interval_elapsed = (now - last_ev) >= interval_s

                    # Instantaneous: run as soon as N trades have closed (no wait for interval)
                    trade_trigger_met = continuous and trigger_on_trade and after_n_trades > 0 and trades_since >= after_n_trades
                    should_run = (interval_elapsed or trade_trigger_met) and (now - last_ev) >= debounce_sec

                    if should_run:
                        st["last_evolution_ts"] = now
                        st["trades_since_evolution"] = 0
                        self._save_state(st)
                        if trade_trigger_met and not interval_elapsed:
                            fitness_days = float(getattr(self.cfg, "evolution_realtime_fitness_days", 1.0) or 1.0)
                            source = "trigger_on_trade"
                        elif interval_elapsed:
                            if run_with_market and realtime_min > 0:
                                fitness_days = float(getattr(self.cfg, "evolution_realtime_fitness_days", 1.0) or 1.0)
                                source = "realtime"
                            else:
                                fitness_days = float(getattr(self.cfg, "evolution_fitness_days", 7) or 7)
                                source = "interval"
                        else:
                            source = "interval"
                        await self._evolution_once(fitness_days=int(max(1, fitness_days)), source=source)
                except Exception as e:
                    logger.debug("Evolution tick: %s", e)

                # Heavy shadow tune on its own interval
                shadow_min = int(getattr(self.cfg, "self_improvement_shadow_interval_minutes", 240) or 240)
                # Back-compat: if only interval_minutes exists, use it
                if shadow_min <= 0:
                    shadow_min = int(getattr(self.cfg, "self_improvement_interval_minutes", 240) or 240)
                shadow_s = float(max(60.0, float(shadow_min) * 60.0))

                st = self._load_state()
                now = time.time()
                last_shadow = float(st.get("last_shadow_ts", 0.0) or 0.0)
                if bool(getattr(self.cfg, "self_improvement_shadow_tune_enabled", True)) and (now - last_shadow >= shadow_s):
                    st["last_shadow_ts"] = float(now)
                    self._save_state(st)
                    await self._shadow_tune_once()
            except Exception as e:
                # Never crash; just back off.
                logger.warning("SelfImprover run_forever loop error: %s", e)
                await asyncio.sleep(5.0)

            await asyncio.sleep(tick_s)

    async def _evolution_once(self, fitness_days: Optional[int] = None, source: str = "interval") -> None:
        """
        Run one evolution pass (GA + paper-loop fitness) in a thread, then apply
        best params to config when evolution_auto_apply is True and safe (paper/backtest or allow_apply_live).
        When evolution_use_live_feed is True and market_data_service is available,
        fetches OHLCV from the live feed and passes to evolution (real-time with the market).
        """
        try:
            from evolution.evolution_unified import evolve_once, apply_evolved_params
        except Exception as e:
            logger.debug("Evolution unavailable: %s", e)
            return
        fd = fitness_days if fitness_days is not None else int(getattr(self.cfg, "evolution_fitness_days", 7) or 7)
        is_realtime = fd <= 2
        generations = int(getattr(self.cfg, "evolution_realtime_generations", 3) or 3) if is_realtime else int(getattr(self.cfg, "evolution_generations", 5) or 5)
        population_size = int(getattr(self.cfg, "evolution_realtime_population_size", 8) or 8) if is_realtime else int(getattr(self.cfg, "evolution_population_size", 12) or 12)
        path = str(getattr(self.cfg, "evolution_params_path", "data/evolved_params.json") or "data/evolved_params.json")
        auto_apply = bool(getattr(self.cfg, "evolution_auto_apply", True))
        dry_run = bool(getattr(self.cfg, "evolution_dry_run", False))
        run_mode = str(getattr(self.cfg, "run_mode", "paper") or "paper").lower()
        allow_apply_live = bool(getattr(self.cfg, "evolution_allow_apply_live", False))
        use_live_feed = bool(getattr(self.cfg, "evolution_use_live_feed", True))
        min_bars = int(getattr(self.cfg, "evolution_min_bars_for_live_feed", 20) or 20)
        ohlcv_override = None
        warmup = 5 if fd <= 2 else None
        if use_live_feed and fd <= 3:
            mds = getattr(self.system, "market_data_service", None)
            if mds is not None:
                try:
                    symbols = list(getattr(self.cfg, "trading_pairs", []) or ["BTC/USD", "ETH/USD"])
                    if not symbols:
                        symbols = ["BTC/USD", "ETH/USD"]
                    limit = max(min_bars + 10, int(fd * 24) + 30)
                    ohlcv_override = {}
                    for sym in symbols:
                        try:
                            df = await mds.fetch_ohlcv_df(str(sym), timeframe="1h", limit=limit)
                            if df is not None and not df.empty and "close" in df.columns and len(df) >= min_bars:
                                ohlcv_override[str(sym)] = df
                        except Exception as e:
                            logger.debug("Evolution live feed fetch for %s: %s", sym, e)
                            continue
                    if len(ohlcv_override) < 1:
                        ohlcv_override = None
                    elif ohlcv_override:
                        logger.debug("Evolution: using live feed (%s symbols, ~%s bars) for fitness", len(ohlcv_override), limit)
                except Exception as e:
                    logger.debug("Evolution live feed fetch: %s", e)
        try:
            result = await asyncio.to_thread(
                evolve_once,
                generations=generations,
                population_size=population_size,
                fitness_days=fd,
                config_path="unified_config.yaml",
                timeframe="1h",
                use_sharpe=True,
                persist=not dry_run,
                evolved_params_path=path,
                ohlcv_by_symbol_override=ohlcv_override,
                warmup=warmup,
                dry_run=dry_run,
                source=source,
                min_trades=int(getattr(self.cfg, "evolution_min_trades", 1) or 1),
                drawdown_penalty_weight=0.05,
                negative_return_penalty_weight=float(getattr(self.cfg, "evolution_negative_return_penalty_weight", 0) or 0),
                use_composite_fitness=bool(getattr(self.cfg, "evolution_use_composite_fitness", False)),
                walk_forward_train_ratio=getattr(self.cfg, "evolution_walk_forward_train_ratio", None),
                evolution_seed=getattr(self.cfg, "evolution_seed", None),
                early_stop_generations=int(getattr(self.cfg, "evolution_early_stop_generations", 0) or 0),
                early_stop_threshold=float(getattr(self.cfg, "evolution_early_stop_threshold", 0.001) or 0.001),
                fitness_cache_size=int(getattr(self.cfg, "evolution_fitness_cache_size", 0) or 0),
                parallel_fitness_workers=int(getattr(self.cfg, "evolution_parallel_fitness_workers", 0) or 0),
                mutation_prob=getattr(self.cfg, "evolution_ga_mutation_prob", None),
                mutation_sigma=getattr(self.cfg, "evolution_ga_mutation_sigma", None),
                crossover_prob=getattr(self.cfg, "evolution_ga_crossover_prob", None),
                backup_before_apply=bool(getattr(self.cfg, "evolution_backup_before_apply", True)),
                version_history_size=int(getattr(self.cfg, "evolution_version_history_size", 0) or 0),
                overfit_penalty_weight=float(getattr(self.cfg, "evolution_overfit_penalty_weight", 0) or 0),
                volatility_penalty_weight=float(getattr(self.cfg, "evolution_volatility_penalty_weight", 0) or 0),
                composite_calmar_weight=float(getattr(self.cfg, "evolution_composite_calmar_weight", 0) or 0),
                multi_timeframes=getattr(self.cfg, "evolution_multi_timeframes", None),
                multi_timeframe_weights=getattr(self.cfg, "evolution_multi_timeframe_weights", None),
                strategy_whitelist_override=getattr(self.cfg, "evolution_strategy_whitelist_override", None),
                on_complete=self._evolution_on_complete,
            )
            should_apply = (
                not dry_run
                and auto_apply
                and result
                and isinstance(result.get("best_params"), dict)
                and (run_mode != "live" or allow_apply_live)
            )
            if run_mode == "live" and auto_apply and not allow_apply_live and result and result.get("best_params"):
                logger.debug("Evolution: skipped auto_apply in live mode (allow_apply_live=false) evolution_run=1 best_fitness=%.4f", float(result.get("best_fitness", 0)))
            if should_apply:
                best_params = result["best_params"]
                n = apply_evolved_params(self.cfg, best_params, filter_to_config=True)
                if n > 0:
                    logger.info(
                        "Evolution: applied %s new params (fitness=%.4f) – strategies and bot use them next cycle evolution_run=1 best_fitness=%.4f source=%s",
                        n, float(result.get("best_fitness", 0)), float(result.get("best_fitness", 0)), source,
                    )
                    # Ensure StrategyEngine picks up new params immediately (config is source; refresh _opt_params)
                    try:
                        brain = getattr(self.system, "ai_brain", None)
                        se = getattr(brain, "strategy_engine", None) if brain is not None else None
                        if se is not None and hasattr(se, "_opt_params"):
                            se._opt_params.update(best_params)
                            if hasattr(se, "_load_opt_params"):
                                se._load_opt_params()
                    except Exception as e:
                        logger.debug("Evolution: strategy engine param refresh failed: %s", e)
                    # Optional: decay strategy allocator stats so bandit doesn't over-exploit old behavior
                    decay = getattr(self.cfg, "evolution_allocator_decay_after_apply", None)
                    if decay is not None:
                        try:
                            from evolution.apply_evolved_strategies import decay_strategy_allocator_stats
                            path = str(getattr(self.cfg, "strategy_allocator_persist_path", "data/strategy_allocator_stats.json") or "data/strategy_allocator_stats.json")
                            decay_strategy_allocator_stats(path, float(decay))
                        except Exception as ex:
                            logger.debug("Allocator decay after evolution: %s", ex)
        except Exception as e:
            logger.warning("Evolution run failed: %s", e)

    def _evolution_on_complete(self, result: Dict[str, Any]) -> None:
        """Optional callback after evolution run (e.g. for metrics)."""
        try:
            metrics = getattr(self.system, "metrics", None)
            if metrics is not None and hasattr(metrics, "gauge") and callable(getattr(metrics, "gauge", None)):
                metrics.gauge("evolution.best_fitness", float(result.get("best_fitness", 0)))
            if metrics is not None and hasattr(metrics, "increment") and callable(getattr(metrics, "increment", None)):
                metrics.increment("evolution.runs_total")
        except Exception as e:
            logger.debug("Evolution on_complete metrics error: %s", e)

    async def _shadow_tune_once(self) -> Optional[ShadowTuneResult]:
        """
        Quick shadow-tune for unified_engine using existing scripts (in-process calls),
        then (optionally) apply if it improves baseline.
        """
        try:
            # Only safe for paper/backtest, and requires network if --fetch is used.
            from scripts.paper_loop_30d_unified import run_paper_loop
            from scripts.optimize_strategies import _sample_params, _score, _score_multi_objective  # type: ignore
            from strategies.strategy_param_space import bounds_for
            from utils.universe_builder import select_top_liquid_usd_pairs
            from scripts.paper_loop_30d_unified import _fetch_ohlcv_ccxt
            from adaptive.promotion_gates import PromotionGate
            import numpy as _np
            import pandas as _pd
        except Exception as e:
            logger.debug("Shadow tune imports unavailable: %s", e)
            return None

        async def _paper(**kwargs: Any) -> Dict[str, Any]:
            # run_paper_loop is synchronous but internally uses async; run it in a worker thread
            return await asyncio.to_thread(run_paper_loop, **kwargs)

        tf = str(getattr(self.cfg, "self_improvement_shadow_tune_timeframe", "1h") or "1h")
        top = int(getattr(self.cfg, "self_improvement_shadow_tune_top", 10) or 10)
        days_total = int(getattr(self.cfg, "self_improvement_shadow_tune_days_total", 30) or 30)
        train_days = int(getattr(self.cfg, "self_improvement_shadow_tune_train_days", 10) or 10)
        test_days = int(getattr(self.cfg, "self_improvement_shadow_tune_test_days", 5) or 5)
        evals = int(getattr(self.cfg, "self_improvement_shadow_tune_evals", 6) or 6)
        warmup = int(getattr(self.cfg, "self_improvement_shadow_tune_warmup", 50) or 50)
        validation_tfs = list(getattr(self.cfg, "self_improvement_validation_timeframes", ["1h", "15m"]) or ["1h", "15m"])
        validation_tfs = [str(x) for x in validation_tfs if str(x).strip()]
        if str(tf) not in validation_tfs:
            validation_tfs = [str(tf), *validation_tfs]

        sel = select_top_liquid_usd_pairs(exchange_id=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"), top_n=top)
        symbols = sel.symbols

        # Prefetch OHLCV (best-effort). If overlap is too small, fall back to BTC/USD.
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - int(days_total) * 86400 * 1000
        ohlcv_full: Dict[str, Any] = {}
        for sym in symbols:
            try:
                ohlcv_full[str(sym)] = await asyncio.to_thread(
                    _fetch_ohlcv_ccxt,
                    exchange_id=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                    symbol=str(sym),
                    timeframe=tf,
                    since_ms=since_ms,
                )
            except Exception as e:
                logger.debug("Shadow tune OHLCV fetch for %s: %s", sym, e)
                continue
        if not ohlcv_full:
            return None
        # Ensure at least one reliable symbol
        if "BTC/USD" not in ohlcv_full:
            try:
                ohlcv_full["BTC/USD"] = await asyncio.to_thread(
                    _fetch_ohlcv_ccxt,
                    exchange_id=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                    symbol="BTC/USD",
                    timeframe=tf,
                    since_ms=since_ms,
                )
            except Exception as e:
                logger.debug("Shadow tune BTC/USD fallback fetch: %s", e)

        # Build a small overlap set
        syms = list(ohlcv_full.keys())[: max(1, min(3, len(ohlcv_full)))]
        ohlcv_full = {k: ohlcv_full[k] for k in syms}

        # Evaluate baseline (current config + optimized params load)
        base_overrides: Dict[str, Any] = {
            "strategies_use_all": True,
            "strategies_enabled": ["__all__"],
            "strategy_library_use_all": True,
            "strategy_library_strategies_enabled": ["__all__"],
            "strategies_max_extra_signals": 50,
            "max_concurrent_signals": 5,
            "optimized_params_load": True,
        }
        # Optional: evaluate both quantum ON/OFF and pick the better baseline.
        try_q = bool(getattr(self.cfg, "self_improvement_try_quantum_on_off", True))
        baseline = None
        baseline_q: Optional[bool] = None
        baseline_qm: Optional[str] = None
        baseline_qs: Optional[float] = None
        quantum_variants = [
            {"quantum_features_enabled": False},
            {"quantum_features_enabled": True, "quantum_method": "probability", "quantum_strength": 0.75},
            {"quantum_features_enabled": True, "quantum_method": "probability", "quantum_strength": 1.0},
            {"quantum_features_enabled": True, "quantum_method": "amplitude", "quantum_strength": 0.75},
            {"quantum_features_enabled": True, "quantum_method": "consciousness_weighted", "quantum_strength": 1.0},
        ]
        if try_q:
            best = None
            best_v: Optional[Dict[str, Any]] = None
            for v in quantum_variants:
                ov = dict(base_overrides)
                ov.update(v)
                res = await _paper(
                    days=int(days_total),
                    timeframe=str(tf),
                    exchange=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                    symbols_csv=",".join(syms),
                    fetch=False,
                    warmup=int(warmup),
                    cfg_overrides=ov,
                    ohlcv_by_symbol_override=ohlcv_full,
                )
                if best is None or float(res.get("return_pct", 0.0) or 0.0) > float(best.get("return_pct", 0.0) or 0.0):
                    best = res
                    best_v = dict(v)
            baseline = best
            baseline_q = bool((best_v or {}).get("quantum_features_enabled", False))
            baseline_qm = (best_v or {}).get("quantum_method")
            baseline_qs = (best_v or {}).get("quantum_strength")
        else:
            baseline = await _paper(
                days=int(days_total),
                timeframe=str(tf),
                exchange=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                symbols_csv=",".join(syms),
                fetch=False,
                warmup=int(warmup),
                cfg_overrides=base_overrides,
                ohlcv_by_symbol_override=ohlcv_full,
            )
            baseline_q = bool(getattr(self.cfg, "quantum_features_enabled", True))
            baseline_qm = str(getattr(self.cfg, "quantum_method", "quantum_approximate") or "quantum_approximate")
            baseline_qs = float(getattr(self.cfg, "quantum_strength", 1.0) or 1.0)

        # Compute baseline across validation timeframes (using chosen baseline quantum toggle)
        baseline_by_tf: Dict[str, Any] = {}
        for tf2 in validation_tfs:
            try:
                ov = dict(base_overrides)
                if baseline_q is not None:
                    ov["quantum_features_enabled"] = bool(baseline_q)
                if baseline_qm is not None:
                    ov["quantum_method"] = str(baseline_qm)
                if baseline_qs is not None:
                    ov["quantum_strength"] = float(baseline_qs)
                baseline_by_tf[str(tf2)] = await _paper(
                    days=int(days_total),
                    timeframe=str(tf2),
                    exchange=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                    symbols_csv=",".join(syms),
                    fetch=False,
                    warmup=int(warmup),
                    cfg_overrides=ov,
                    ohlcv_by_symbol_override=ohlcv_full,
                )
            except Exception as e:
                logger.debug("Shadow tune baseline eval for tf=%s: %s", tf2, e)
                continue

        ranges = bounds_for("unified_engine")
        rng = _np.random.default_rng(7)
        best_params: Dict[str, Any] = {}
        best_train = float("-inf")
        use_multi_objective = bool(getattr(self.cfg, "self_improvement_multi_objective", False))
        use_bayesian = bool(getattr(self.cfg, "self_improvement_bayesian_opt", False))

        # Use last (train_days) of history as "train"; test_days as "test" at end.
        end_ts = None
        for df in ohlcv_full.values():
            end_ts = df.index.max() if end_ts is None else min(end_ts, df.index.max())
        if end_ts is None:
            return None
        train_start = end_ts - _pd.Timedelta(days=int(train_days + test_days))
        train_end = end_ts - _pd.Timedelta(days=int(test_days))
        test_end = end_ts

        def _objective(params: Dict[str, Any]) -> float:
            cand_overrides = dict(base_overrides)
            cand_overrides.update(params)
            cand_overrides["optimized_params_load"] = False
            if try_q and baseline_q is not None:
                cand_overrides["quantum_features_enabled"] = bool(baseline_q)
            res_train = run_paper_loop(
                days=int(days_total),
                timeframe=str(tf),
                exchange=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                symbols_csv=",".join(syms),
                fetch=False,
                warmup=int(warmup),
                cfg_overrides=cand_overrides,
                ohlcv_by_symbol_override=ohlcv_full,
                window_start_utc=str(train_start),
                window_end_utc=str(train_end),
            )
            if use_multi_objective:
                return _score_multi_objective(res_train)
            return _score(res_train)

        if use_bayesian:
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def optuna_objective(trial: Any) -> float:
                    params = {}
                    for k, (lo, hi) in ranges.items():
                        params[str(k)] = trial.suggest_float(str(k), float(lo), float(hi))
                    return _objective(params)

                study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(n_startup_trials=2))
                study.optimize(optuna_objective, n_trials=max(1, evals), show_progress_bar=False)
                if study.best_params:
                    best_params = study.best_params
                    best_train = float(study.best_value)
            except Exception as e:
                logger.debug("Bayesian optimization failed, falling back to random: %s", e)
                use_bayesian = False

        if not use_bayesian:
            for _ in range(max(1, evals)):
                params = _sample_params(ranges, rng)
                sc = _objective(params)
                if sc > best_train:
                    best_train = sc
                    best_params = params

        cand_overrides = dict(base_overrides)
        cand_overrides.update(best_params)
        cand_overrides["optimized_params_load"] = False
        # Candidate OOS evaluation; optionally test quantum ON/OFF and keep best.
        candidate = None
        candidate_q: Optional[bool] = baseline_q
        candidate_qm: Optional[str] = baseline_qm
        candidate_qs: Optional[float] = baseline_qs
        if try_q:
            best = None
            best_v: Optional[Dict[str, Any]] = None
            for v in quantum_variants:
                ov = dict(cand_overrides)
                ov.update(v)
                res = await _paper(
                    days=int(days_total),
                    timeframe=str(tf),
                    exchange=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                    symbols_csv=",".join(syms),
                    fetch=False,
                    warmup=int(warmup),
                    cfg_overrides=ov,
                    ohlcv_by_symbol_override=ohlcv_full,
                    window_start_utc=str(train_end),
                    window_end_utc=str(test_end),
                )
                if best is None or float(res.get("return_pct", 0.0) or 0.0) > float(best.get("return_pct", 0.0) or 0.0):
                    best = res
                    best_v = dict(v)
            candidate = best
            candidate_q = bool((best_v or {}).get("quantum_features_enabled", False))
            candidate_qm = (best_v or {}).get("quantum_method")
            candidate_qs = (best_v or {}).get("quantum_strength")
        else:
            candidate = await _paper(
                days=int(days_total),
                timeframe=str(tf),
                exchange=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                symbols_csv=",".join(syms),
                fetch=False,
                warmup=int(warmup),
                cfg_overrides=cand_overrides,
                ohlcv_by_symbol_override=ohlcv_full,
                window_start_utc=str(train_end),
                window_end_utc=str(test_end),
            )

        # Candidate across validation timeframes (using chosen candidate quantum toggle)
        candidate_by_tf: Dict[str, Any] = {}
        for tf2 in validation_tfs:
            try:
                ov = dict(cand_overrides)
                if candidate_q is not None:
                    ov["quantum_features_enabled"] = bool(candidate_q)
                if candidate_qm is not None:
                    ov["quantum_method"] = str(candidate_qm)
                if candidate_qs is not None:
                    ov["quantum_strength"] = float(candidate_qs)
                candidate_by_tf[str(tf2)] = await _paper(
                    days=int(days_total),
                    timeframe=str(tf2),
                    exchange=str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                    symbols_csv=",".join(syms),
                    fetch=False,
                    warmup=int(warmup),
                    cfg_overrides=ov,
                    ohlcv_by_symbol_override=ohlcv_full,
                    window_start_utc=str(train_end),
                    window_end_utc=str(test_end),
                )
            except Exception as e:
                logger.debug("Shadow tune candidate eval for tf=%s: %s", tf2, e)
                continue

        b_ret = float(baseline.get("return_pct", 0.0) or 0.0)
        c_ret = float(candidate.get("return_pct", 0.0) or 0.0)
        b_dd = float(baseline.get("max_drawdown_pct", 0.0) or 0.0)
        c_dd = float(candidate.get("max_drawdown_pct", 0.0) or 0.0)
        b_tr = int(baseline.get("trades", 0) or 0)
        c_tr = int(candidate.get("trades", 0) or 0)

        max_dd = float(getattr(self.cfg, "self_improvement_max_drawdown_pct", 2.0) or 2.0)
        min_trades = int(getattr(self.cfg, "self_improvement_min_trades", 3) or 3)
        apply_only = bool(getattr(self.cfg, "self_improvement_apply_on_improvement_only", True))

        gate = PromotionGate(
            min_delta_score=float(getattr(self.cfg, "self_improvement_promotion_min_delta_score", 0.10) or 0.10),
            max_drawdown_pct=float(max_dd),
            min_trades=int(min_trades),
            require_all_timeframes=bool(getattr(self.cfg, "self_improvement_promotion_require_all_timeframes", True)),
        )
        decision = gate.evaluate(baseline_by_tf=baseline_by_tf, candidate_by_tf=candidate_by_tf, timeframes=validation_tfs)

        applied = False
        reason = "no_change"
        if decision.ok and (not apply_only or decision.ok):
                # Apply: write to optimized params file and enable auto-load
                out_path = Path(str(getattr(self.cfg, "optimized_params_path", "data/optimized_params.json") or "data/optimized_params.json"))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "version": 1,
                    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "exchange": str(getattr(self.cfg, "primary_exchange", "kraken") or "kraken"),
                    "timeframe": tf,
                    "timeframes": {tf: {"unified_engine": {"best_params": best_params}}},
                    "params": {"unified_engine": {"best_params": best_params}},
                }
                out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                setattr(self.cfg, "optimized_params_load", True)
                setattr(self.cfg, "optimized_params_timeframe", tf)
                setattr(self.cfg, "optimized_params_by_timeframe", payload["timeframes"])
                # Apply quantum choice if enabled
                if bool(getattr(self.cfg, "self_improvement_apply_quantum_choice", True)) and candidate_q is not None:
                    setattr(self.cfg, "quantum_features_enabled", bool(candidate_q))
                    setattr(self.cfg, "use_quantum_monte_carlo_risk", bool(candidate_q))
                    setattr(self.cfg, "use_quantum_walk", bool(candidate_q))
                    if candidate_qm is not None:
                        setattr(self.cfg, "quantum_method", str(candidate_qm))
                    if candidate_qs is not None:
                        setattr(self.cfg, "quantum_strength", float(candidate_qs))
                applied = True
                reason = "applied_new_params"
        else:
            reason = f"promotion_failed:{decision.reason}"

        return ShadowTuneResult(
            baseline_return_pct=b_ret,
            candidate_return_pct=c_ret,
            baseline_trades=b_tr,
            candidate_trades=c_tr,
            baseline_maxdd_pct=b_dd,
            candidate_maxdd_pct=c_dd,
            applied=applied,
            reason=reason,
            quantum_enabled=candidate_q,
            quantum_method=str(candidate_qm) if candidate_qm is not None else None,
            quantum_strength=float(candidate_qs) if candidate_qs is not None else None,
        )

