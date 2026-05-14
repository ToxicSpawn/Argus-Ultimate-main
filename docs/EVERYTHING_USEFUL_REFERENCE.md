# Everything Useful – Codebase Reference

A single reference for **what exists and what can be used** across the repo. Grouped by: **already wired**, **ready to wire**, **high-value optional**, **supporting**, and **empty/stub**.

---

## 1. Already wired into the unified system (main path)

These are imported and used by `unified_trading_system.py`, `unified_execution_engine.py`, `unified_ai_brain.py`, or the backtester.

| Area | What's used | Where |
|------|-------------|--------|
| **Execution** | `state_store`, `risk_compliance_audit`, `implementation_shortfall`, `multi_venue_execution`, `twap_slicer`, `vwap_pov_core`, `latency_attribution` | `execution/` |
| **Risk** | `unified_risk_manager`, `real_time_risk_api` (volatility-adjusted limits), `unified_risk_facade` (position sizing: Kelly, volatility, drawdown, regime, correlation, VaR, risk contribution) | `risk/` |
| **Adaptive** | `regime` (MarketRegime, RegimeDetector), `online_tuner`, `strategy_allocator`, `adaptive_risk_controller`, `universe_selector`, `self_improver` | `adaptive/` |
| **Services** | `market_data_service`, `continuous_best_trade_scanner` (AI + strategy engine + HFT + **strategy library** when enabled) | `services/` |
| **Strategies** | `strategies/unified/strategy_engine` (canonical engine); **strategy library** (tier + algorithmic) as optional 4th scanner source via `strategy_library_impl.get_library_strategies_for_names` | `strategies/unified/`, `strategies/strategy_library_impl.py` |
| **Evolution** | `apply_evolved_strategies.apply_from_file` (load evolved params into config) | `evolution/` |
| **Monitoring** | `trade_ledger`, `audit_trail` (via risk_compliance_audit) | `monitoring/` |
| **Utils** | `circuit_breaker`, `self_healing`, `universe_builder` (select_top_liquid_usd_pairs) | `utils/` |
| **HFT** | `advanced_realtime_hft_infrastructure` (get_hft_infrastructure) | `hft/` |
| **Backtest** | `unified_event_backtester` (StrategyEngine + CapitalOptimizer + HistoricalMarketDataService) | `backtest/` |
| **Core** | `config_manager` (load_unified_trading_config), `indicators` (technical indicators) | `core/` |

---

## 2. Ready to use (implemented, can be wired or scripted)

- **Strategy library & tiers**  
  `strategies/strategy_library_impl.py`, `strategies/tier_strategies_impl.py`  
  Momentum, mean reversion, trend following, pairs, market making, candlestick, high-freq grid, regime switching, stat arb, quantum elites, tier ensembles (absolute, akashic, omega, etc.). **Now wired:** when `strategy_library.enabled` and run_mode in `strategy_library.modes` (paper/backtest/live), the continuous scanner runs `_gather_strategy_library()` and merges signals. Add library strategy names to `strategy_whitelist` to allow their signals.

- **Execution (extra)**  
  - `execution/pipeline.py` – ExecutionPipeline, PipelineConfig (used by run_peak, run_ultimate, run_paper, run_godmode).  
  - `execution/smart_order_execution.py`, `smart_order_router.py`, `market_impact.py`, `order_types_advanced.py` – advanced order and routing logic.

- **Risk (rich set)**  
  - **Position sizing:** `risk/position_sizing/` – Kelly, volatility_adjusted, drawdown_adjusted, market_regime, correlation_adjusted, ensemble_sizing, ml_optimized, dynamic_sizing, cvar_based.  
  - **Stop loss:** `risk/stop_loss/` – trailing, ATR, volatility, time-based, support/resistance, adaptive, profit target, and many others.  
  - **Portfolio risk:** `risk/portfolio_risk/` – VaR, Monte Carlo, correlation_matrix, risk_contribution.  
  - **Unified risk facade** already wires several of these; you can swap or add sizers via config.

- **Alpha / order flow**  
  - `alpha/orderbook/orderbook_pipeline.py` – VPIN, order flow imbalance, book pressure.  
  - `alpha/order_book_signals.py`, `alpha/arbitrage/arbitrage_engine.py`, `alpha/regime_detector/regime_detector.py`, `alpha/sentiment/sentiment_engine.py`.  
  Useful for signal enhancement or a separate alpha pipeline; not in the unified loop by default.

- **Evolution**  
  - `evolution/apply_evolved_strategies.py` – load/save evolved params (already used).  
  - `evolution/godmode_evolution_v2.py`, `godmode_evolution.py`, `strategy_genetic_algorithm.py`, `continuous_evolution_engine.py` – parameter evolution; used by `run_ultimate_evolution.py`, `run_evolution.py`, `run_godmode.py`.

- **Data**  
  - `data/ccxt_data_provider.py`, `data/paper_data_hooks.py`, `data/data_lake.py`, `data/tick_store.py`, `data/cross_asset_dataset.py`, `data/research_platform.py` – feeds, lake, tick store, research.  
  Market data service can be extended to use these.

- **Core**  
  - `core/indicators.py` – TechnicalIndicators (EMA, SMA, RSI, MACD, ATR, Bollinger, etc.).  
  - `core/position_tracker.py`, `core/rate_limiter.py`, `core/data_sanitizer.py`, `core/audit_chain.py`, `core/health.py`, `core/config_unified.py` – tracking, rate limit, sanitization, audit, health, config.

- **Utils**  
  - `utils/circuit_breaker.py`, `utils/self_healing.py`, `utils/universe_builder.py`, `utils/retry.py`, `utils/validators.py`, `utils/tracing.py`, `utils/logger.py`, `utils/helpers.py`, `utils/math_utils.py`, `utils/datetime_utils.py`.  
  Circuit breaker and self-healing are already used; others for scripts and extensions.

- **ML**  
  - `ml/online_learning.py`, `ml/features.py`, `ml/ensemble.py`, `ml/features/feature_library.py`, `feature_engineer.py` – features and models for optional signal or sizing models.

---

## 3. High-value optional (quantum, monitoring, scripts)

- **Quantum** (`quantum/`)  
  Large set: QAOA, VQE, annealing, portfolio optimization, risk, backtesting, production simulator, cost optimizer, vendors (IBM, Google, D-Wave, etc.).  
  Unified system uses `quantum_unified_stubs`; full integration is optional (see `quantum/README.md`).

- **Monitoring**  
  - `monitoring/metrics.py`, `monitoring/alerting.py` – metrics and alerts (used in tests and can be wired to dashboard/notifications).

- **Scripts**  
  - `scripts/run_30d_backtest.py` – 30-day backtest on OHLCV.  
  - `scripts/health_check.py` – validate config and optional paper smoke.  
  - Other scripts under `scripts/` for reporting, stress tests, export, etc.

- **Docs**  
  - `docs/ALL_WAYS_BOT_TRADES.md` – how the bot trades.  
  - `docs/EVERYTHING_ELSE_IN_THE_BOT.md` – architecture and config.  
  - `docs/IMPROVEMENT_EVERYTHING.md`, `docs/ARGUS_IMPROVEMENTS_MASTER.md` – improvement checklists.

---

## 4. Alternative run paths (invokable from main.py)

You can run them via **main.py** or the script directly:

| main.py command | Script | Description |
|-----------------|--------|-------------|
| `main.py peak` | run_peak.py | Peak performance: best pairs, Peak Alpha strategy, aggressive sizing. |
| `main.py ultimate` | run_ultimate.py | ExecutionPipeline + PositionSizer + adapted strategy library. |
| `main.py evolution` | run_ultimate_evolution.py | Parameter evolution (godmode_evolution_v2). |
| `main.py godmode` | run_godmode.py | Godmode evolution + pipeline + sizing. |

- **run_ultimate.py** – Uses `execution.pipeline.ExecutionPipeline`, `risk.position_sizing.PositionSizer`, evolution (godmode).  
- **run_peak.py** – Same execution pipeline + risk sizing.  
- **run_paper.py** – Paper loop with pipeline and position sizer.  
- **run_godmode.py** – Godmode evolution + pipeline + sizing.  
- **run_ultimate_evolution.py** – Evolution engine (godmode_evolution_v2).  
- **run_evolution.py** – Godmode evolution (godmode_evolution).  

See **docs/ALTERNATIVE_RUN_PATHS.md** for usage and differences vs the unified path. Use these if you want the legacy pipeline + rich position sizing (Kelly, volatility, etc.) instead of the unified capital optimizer.

---

## 5. Supporting / infrastructure (no direct trading logic)

- **Config** – `unified_config.yaml`, `core/config_manager.py` (load_unified_trading_config).  
- **Exchanges** – CCXT-based exchange setup (see unified_trading_system exchange init).  
- **CLI** – `cli/commands.py` (can call execution pipeline).  
- **Tests** – `tests/`, `tests_unified/` – import and test risk, execution, monitoring, config.

---

## 6. Empty or stub / name-only

These directories either had no Python files or are not imported by the unified path; treat as placeholder or future use.

- **void_breaker** – Has `__init__.py` stub (placeholder for future void-breaker logic).  
- **web3_defi** – Has `__init__.py` stub (placeholder for DeFi / web3 integration).  
- **arbitrage** (top-level) – No `.py` files (arbitrage lives under `alpha/arbitrage/` and strategy library).  
- **indicators** (top-level) – No files (indicators are in `core/indicators.py`).  
- **portfolio** (top-level) – No `.py` files.  
- **config** (top-level) – No `.py` files (config is YAML + core/config_manager).  
- **analytics** – No `.py` files.  
- **api** – No `.py` files.  
- **sentiment_analysis** – No `.py` files (sentiment under `alpha/sentiment/`).  

Other dirs you listed (e.g. `advanced_improvements`, `advanced_quantum`, `agents`, `agi`, `ai`, `blockchain`, `crystal_impl`, etc.) may contain code; the **unified production path** only pulls from the modules listed in section 1. To use anything else, wire it explicitly (e.g. new signal source, new sizer, or new script).

---

## 7. Quick “what to use when”

| Goal | Use |
|------|-----|
| Run the main bot (paper/live) | `main.py paper` / `main.py live` + `unified_config.yaml`. |
| Backtest 30 days | `scripts/run_30d_backtest.py`. |
| Evolve strategy params | `evolution/` + `run_ultimate_evolution.py` or load via `apply_evolved_strategies`. |
| Richer position sizing | `risk/position_sizing/` via `risk/unified_risk_facade` or run_peak/run_ultimate pipeline. |
| Order flow / VPIN alpha | `alpha/orderbook/orderbook_pipeline.py` (wire into a custom signal or scanner). |
| Extra strategies | Strategy library is now a 4th scanner source when `strategy_library.enabled` and run_mode in `strategy_library.modes`; add names to `strategy_whitelist`. |
| Alternative run paths | `main.py peak` / `ultimate` / `evolution` / `godmode` or run_peak.py etc.; see docs/ALTERNATIVE_RUN_PATHS.md. |
| Technical indicators | `core/indicators.py` (add_indicators, RSI, MACD, etc.). |
| Circuit breaker / self-heal | `utils/circuit_breaker.py`, `utils/self_healing.py` (already used). |
| Quantum optimization | `quantum/` (optional; stubs used by unified; full stack in quantum/). |

---

**Summary:** The **unified system** (main.py → unified_trading_system → execution_engine, strategy_engine, scanner, risk, adaptive, evolution apply) is the single production path. Everything in **section 1** is already useful there. **Sections 2–4** are useful for extending signals, sizing, execution, data, evolution, or for alternative run paths. **Section 6** dirs are currently empty or stub for the unified path.
