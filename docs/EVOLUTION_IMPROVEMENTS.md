# Every Way Evolution Can Be Improved

This document lists **every** concrete way the Argus evolution subsystem can be improved: code health, fitness, search, integration, operations, and safety.

**Implemented (all items below):** Unified param space (`evolution/param_space.py`), persistence backup + version history + metadata, GA early stopping / fitness cache / parallel / seed-from-file / configurable mutation, composite fitness + min_trades + walk-forward + negative-return penalty, `evolution_unified` dry_run/source/backup and all options, UnifiedConfig + YAML (dry_run, debounce, allow_apply_live, GA, min_trades, etc.), self_improver debounce + safety (no apply in live unless flag) + structured logging, and tests in `tests/evolution/test_apply_and_evolution_unified.py`.

**Phase 2 (further):** Multi-timeframe fitness (e.g. `evolution_multi_timeframes: ["1h", "15m"]` with optional weights), overfit penalty (train>>test in walk-forward), volatility penalty and Calmar in composite fitness, strategy whitelist for evolution backtest (`evolution_strategy_whitelist_override`), allocator decay after apply (`evolution_allocator_decay_after_apply`: 0=reset, 0.5=halve), rollback helper `rollback_to_previous()` in `apply_evolved_strategies`, `decay_strategy_allocator_stats()`, and `on_complete` callback for metrics (`_evolution_on_complete` in self_improver). Config and YAML extended for all of the above.

---

## 1. Critical: Restore or Implement `evolution_unified` ✅

- **Issue:** `evolution/evolution_unified.py` is **stubbed** (raises `RuntimeError` on import). `adaptive/self_improver.py` and `evolution/continuous_evolution_engine.py` import `evolve_once` and `apply_evolved_params` from it, so **continuous evolution and trade-triggered evolution fail at runtime**.
- **Improvement:** Implement `evolution_unified.py` with:
  - `evolve_once(generations, population_size, fitness_days, config_path, timeframe, use_sharpe, persist, evolved_params_path, ohlcv_by_symbol_override, warmup)` → runs `strategy_genetic_algorithm.run_genetic_algorithm` with `make_paper_loop_fitness(...)` and returns `{ "best_params", "best_fitness", ... }`; optionally persist via `apply_evolved_strategies.write_evolved_params`.
  - `apply_evolved_params(config, params)` → thin wrapper around `evolution.apply_evolved_strategies.apply_to_config(config, params)` and return the count.
- **Alternative:** Point self_improver and continuous_evolution_engine at a different module that provides the same API (e.g. a new `evolution/unified_runner.py`).

---

## 2. Unify Param Space: GA vs Ultimate vs Config

- **Issue:** Three different param spaces exist:
  - **strategy_genetic_algorithm**: `DEFAULT_PARAM_BOUNDS` (e.g. `se_buy_rsi`, `se_sell_rsi`, `se_buy_bb`, `min_signal_confidence`) — used by paper-loop fitness.
  - **godmode_evolution / godmode_evolution_v2**: `GODMODE_PARAM_BOUNDS` / `ULTIMATE_PARAM_BOUNDS` (e.g. `max_position_pct`, `min_hold_bars`, `kelly_fraction`, regime weights, timeframes) — used by run_godmode and Ultimate engine.
  - **UnifiedConfig**: only a subset of attributes are actually used by the unified strategy engine and execution; evolved params are applied with `apply_to_config` by key name.
- **Improvement:**
  - Define a **single source of truth** for “evolvable” params: e.g. a mapping from param key → (min, max) and which config attributes they map to. Use it in both the simple GA and the Ultimate engine so evolution always optimizes the same knobs the unified system respects.
  - Ensure every key in evolved params exists on `UnifiedConfig` (or a documented apply layer) so auto_apply never silently no-op.

---

## 3. Fitness Function

- **Sharpe vs return vs multi-objective:** Today `make_paper_loop_fitness` can use Sharpe or `return_pct`; Ultimate engine uses NSGA-II with multiple objectives (sharpe, sortino, calmar, win_rate, profit_factor, total_return). The **unified self_improver path** (when evolution_unified exists) uses a single objective. Unify so the same multi-objective or composite fitness is available to the lightweight GA (e.g. weighted sum or Pareto selection with a single “primary” metric for apply).
- **Min-trades gate:** Fitness already returns a large penalty when `trades < min_trades`. Consider making `min_trades` configurable in YAML (evolution section) and expose it to the self_improver so short windows don’t reward overtrading or noise.
- **Drawdown / risk in fitness:** Strategy GA supports `penalize_drawdown` and `drawdown_penalty_weight`; Ultimate has `drawdown_penalty`, `volatility_penalty`, `negative_return_penalty`, `overfit_penalty`. Ensure the unified evolution path applies at least drawdown and negative-return penalties so evolution doesn’t favor fragile or overly aggressive params.
- **Walk-forward and overfitting:** Ultimate has true rolling K-fold walk-forward; the simple GA does one backtest window. Add an optional walk-forward or train/test split to the unified evolution path to reduce overfitting (e.g. fitness = weighted train + test, or only apply if test Sharpe > threshold).
- **Live feed alignment:** When `evolution_use_live_feed` is True, self_improver fetches OHLCV from `market_data_service` and passes it to evolution. Ensure timeframe and symbol set match what the paper loop and production use so evolved params are validated in the same regime (e.g. same `trading_pairs` and primary timeframe).

---

## 4. Search Algorithm and Diversity

- **Population size and generations:** Config uses small values for real-time (e.g. 8 pop, 3–5 gen) to keep latency low. Document the tradeoff (speed vs quality) and consider adaptive settings: e.g. larger pop/gens when running on interval (e.g. 24h) and smaller when trigger_on_trade or realtime_interval_minutes.
- **Stall / early stopping:** Strategy GA has a simple diversity boost on stall (increase mutation); Ultimate has proper early_stop_generations / early_stop_threshold. Add early stopping to the unified GA path to avoid wasting CPU after convergence.
- **Crossover and mutation:** Ultimate uses adaptive mutation (initial vs final prob/sigma), DE crossover, CMA-ES fraction. The simple GA uses fixed tournament, uniform crossover, gaussian mutation. Expose at least mutation_prob and mutation_sigma in config (or from a small “evolution_ga” subsection) so power users can tune without code change.
- **Seeding:** Seed the GA from last best params (from `evolved_params.json`) so each run starts near the previous solution and refinement is faster. Optional: seed from hall-of-fame or Pareto front when using Ultimate.

---

## 5. Integration with Unified System

- **Apply surface:** Evolved params are applied with `apply_to_config(config, params)` — only attributes that exist on the config object are set. Many Ultimate/GODMODE params (e.g. `min_hold_bars`, `stop_loss_atr_mult`, regime weights) may not exist on `UnifiedConfig`. Either extend `UnifiedConfig` (and the YAML loader) for all evolvable keys or maintain a small “evolved → config” mapping that writes to nested structures or strategy-engine-specific opts.
- **StrategyEngine refresh:** Self_improver already updates `ai_brain.strategy_engine._opt_params` and calls `_load_opt_params()` after apply. Ensure every evolved param that the StrategyEngine uses is in `_opt_params` (or that StrategyEngine reads from config on each cycle). Same for execution engine (slippage, position caps, etc.).
- **Allocator and evolution:** Strategy allocator favors strategies by PnL; evolution changes strategy params. Ensure allocator state (or its persist file) is not invalidated when params change — e.g. no stale strategy IDs — and consider resetting or decaying allocator stats after a major evolution apply so the bandit doesn’t over-exploit old behavior.
- **Capital and risk:** Evolution runs paper loop with config overrides; ensure capital and risk limits in the overridden config match the intended environment (e.g. paper capital) so fitness is comparable across runs.

---

## 6. Persistence and Versioning

- **Single file:** Evolved params are stored in one JSON (e.g. `data/evolved_params.json`) with `best_params` and `timestamp_utc`. Add optional versioning: e.g. keep last N best params or timestamps for rollback (e.g. “revert to previous evolution”).
- **Metadata:** Already timestamp; add optional `fitness`, `generations`, `population_size`, `fitness_days`, `source` (e.g. "realtime" vs "interval" vs "trigger_on_trade") so operators can correlate with logs and A/B tests.
- **Backup before apply:** When `evolution_auto_apply` is True, write a backup of current config (or current evolved_params.json) before overwriting so a bad evolution can be reverted quickly.

---

## 7. Triggering and Scheduling

- **Interval vs trade-trigger:** Config has `realtime_interval_minutes`, `interval_hours`, and `trigger_on_trade` + `after_n_trades`. Ensure only one “realtime” schedule is active (e.g. if realtime_interval_minutes > 0, use it; else interval_hours) and document that trigger_on_trade runs in addition to the time-based schedule.
- **Debouncing:** When trigger_on_trade is True and after_n_trades is low (e.g. 3), evolution can run very frequently. Add a minimum interval between evolution runs (e.g. at most once per 15 minutes) to avoid CPU thrashing and overlapping runs.
- **Trades_since_evolution reset:** Self_improver increments `trades_since_evolution` and should reset it after a run; verify the state is reset so the next N trades trigger again as intended.

---

## 8. Live vs Paper and Safety

- **Evolution never trades:** Evolution should only run paper/backtest fitness. Ensure no evolution path ever sends live orders; self_improver already gates by run_mode. Double-check that `run_paper_loop` and any backtest used for fitness never use live exchange APIs when called from evolution.
- **Apply only in safe modes:** Config already drives “load_evolved” and “auto_apply”; consider explicitly disabling auto_apply when run_mode is live (or require an explicit “allow_evolution_apply_live” flag) so a misconfiguration doesn’t apply untested params to live.
- **Live feed for fitness only:** Using live OHLCV for fitness is fine; ensure the same code path never uses live order/balance data for evolution.

---

## 9. Observability and Debugging

- **Logging:** Log evolution start/end, best_fitness, number of params applied, and source (interval vs trade-trigger). Use structured fields (e.g. `evolution_run=1`, `best_fitness=0.42`) so logs are grep’able.
- **Metrics:** If a metrics system exists, expose counters/histograms for evolution runs, fitness value, apply count, and failure count.
- **Dry run:** Add a config flag `evolution_dry_run` that runs the GA and logs the best params but does not write to disk or apply, for validation without changing state.

---

## 10. Performance and Cost

- **Fitness cache:** Ultimate engine has `LRUFitnessCache`; the simple GA does not. Add an optional small LRU cache for fitness (key = hash of params) when using the unified path to avoid duplicate backtests for identical or near-identical params.
- **Parallel fitness:** Ultimate uses ProcessPoolExecutor for parallelism; the simple GA evaluates the population sequentially. Parallelize fitness evaluation in the unified GA (e.g. multiprocessing or asyncio with run_paper_loop in threads) to reduce wall-clock time.
- **Reduce backtest length when possible:** For realtime evolution, `realtime_fitness_days` (e.g. 1) already shortens the window. Ensure the paper loop respects this and doesn’t fetch more data than needed so runs stay fast.

---

## 11. Optional Advanced Features

- **Regime-specific params:** Ultimate has regime detection and regime weights in the param bounds. The unified config could support loading different evolved_params per regime (e.g. `evolved_params_high_vol.json`) and switching by current regime — requires regime to be available at apply time.
- **Multi-timeframe evolution:** Ultimate uses multiple timeframes for fitness; the unified path uses a single timeframe from config. Expose multi-timeframe fitness (or at least 1h + 15m) in the unified path so evolved params are robust across timeframes.
- **Quantum / shadow tuning:** Self_improver already has shadow tuning with optional quantum on/off. Ensure evolution and shadow tuning don’t fight (e.g. same param set and apply order, or document that evolution overwrites shadow-tuned params if both run).
- **Strategy subset:** Evolve only a subset of strategies (e.g. by whitelist) and keep others fixed to reduce search space and focus improvement.

---

## 12. Testing and Reproducibility

- **Determinism:** Use a configurable seed (e.g. `evolution_seed`) for the GA so runs are reproducible; already supported in GAConfig, ensure it’s passed from config.
- **Unit tests:** Add tests for: apply_from_file (existing), apply_to_config with various keys, fitness function returning penalty when trades < min_trades, and that evolution_unified (once implemented) returns the expected structure.
- **Integration test:** One end-to-end test that runs a single evolution (1 generation, 2 individuals), checks that evolved_params.json is written and that apply_from_file sets at least one attribute on a dummy config.

---

## Summary Table

| Area              | Improvement |
|-------------------|------------|
| **Critical**      | Implement or restore `evolution_unified` (evolve_once + apply_evolved_params). |
| **Param space**   | Unify GA, Ultimate, and UnifiedConfig evolvable params; ensure apply surface covers all. |
| **Fitness**       | Multi-objective or composite fitness, min_trades/drawdown/overfit penalties, optional walk-forward. |
| **Search**        | Early stopping, configurable mutation/crossover, seed from last best, optional parallelism. |
| **Integration**   | Config and StrategyEngine/allocator aligned with evolved params; apply and refresh documented. |
| **Persistence**   | Versioning/rollback, metadata in JSON, backup before apply. |
| **Scheduling**    | Clear interval vs trade-trigger, debounce, reset trades_since_evolution. |
| **Safety**        | Evolution only paper/backtest; optional disable auto_apply in live. |
| **Observability** | Logging, metrics, dry_run. |
| **Performance**   | Fitness cache, parallel fitness evaluation. |
| **Advanced**      | Regime-specific params, multi-timeframe, strategy subset. |
| **Testing**       | Seed, unit tests for apply/fitness, one integration test. |

Implementing **section 1** is required for continuous and trade-triggered evolution to work at all. The rest can be prioritized by impact and effort.
