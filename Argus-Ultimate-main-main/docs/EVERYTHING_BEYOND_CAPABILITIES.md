# Everything Possible That Will Take the Bot Beyond Its Capabilities

Single reference for **every** improvement, enhancement, and idea documented across Argus: what’s done, what’s config-only, what to build, optional modules, infrastructure, and process. Use this as the master list.

**Implemented in "do it all" pass:** Emergency shutdown; backtest oos_train_ratio + rate_limit_reject_pct; universe builder real CCXT volume; 23-lang per-task timeouts + get_ready/get_metrics/get_capabilities; export_performance_series script; DCA levels; funding rates stub; volatility regime scale; LSTM/GRU regime stub. **10/10 pass:** Edge validation gate (live_require_paper_edge); backtest fill_probability; optional imports (risk, weight_provider, gpu_inference); readiness_score.py; live_vs_backtest_consistency.py; tca_summary.py; quantum_simulated_disclosure; iceberg_enabled/dark_pool_enabled placeholders; RATING_AND_ROADMAP.md. **Keep going:** pre_trade_risk_block(); use_evolution_strategy_reward config.

---

## Legend

| Mark | Meaning |
|------|--------|
| ✅ | Implemented in codebase |
| ⚙️ | Config-only (turn on or tune in YAML) |
| 📋 | To add (documented, not yet built) |
| 🔌 | Optional module (add/restore when available) |
| 🖥️ | Infrastructure / hardware / ops |
| 📌 | Process / discipline |

---

## 1. Edge and signal quality (trade only when edge is real)

| Item | Status | Where / action |
|------|--------|----------------|
| Edge-cost gate (min edge, buffer mult) | ✅ | `edge_cost_gate.*`; applies when modes include run_mode. |
| Live-stricter edge gate | ✅ | `edge_cost_gate.live_min_edge_pct`, `live_buffer_mult`; used in live mode. |
| Min signal confidence | ✅ | `ai_brain.min_signal_confidence`. |
| Live min confidence (0.78–0.82) | ✅ | `ai_brain.live_min_signal_confidence`; strategy engine uses in live. |
| Strategy whitelist (only positive PnL) | ✅ | `strategies.strategy_whitelist`; scanner filters. |
| Disabled strategies (paper/live) | ✅ | `paper_trading_disabled_strategies`, `live_disabled_strategies`. |
| Implementation shortfall computed & logged | ✅ | `execution/implementation_shortfall.py`; audit + execution. |
| IS by strategy/symbol + gate | ✅ | `execution/is_tracker.py`; `use_is_gate`, `max_avg_is_bps`; reject when avg IS > threshold. |
| Regime filter (trend/MR by regime) | ✅ | `strategies.regime_filter_enabled`. |
| Regime boost (momentum/vol from closes) | ✅ | `ml/regime_boost.py`; `strategies.use_regime_lstm_boost`. |
| Kill losers / weekly review script | ✅ | `scripts/kill_losers_review.py`; suggests removals from whitelist. |
| Raise live confidence (config) | ⚙️ | Set `min_signal_confidence` 0.72+ for live (paper can stay lower). |
| Tighten edge gate for live (config) | ⚙️ | Set `live_min_edge_pct: 1.0`, `live_buffer_mult: 2.2`. |
| Use IS to down-weight / disable strategies | 📌 | Review IS by strategy; remove from whitelist or reduce size when bad. |

---

## 2. Learning and adaptation (learn from what just happened)

| Item | Status | Where / action |
|------|--------|----------------|
| Strategy allocator (PnL-based ranking) | ✅ | `strategy_allocator.enabled`; rank_signals in loop. |
| Online tuner (confidence mult from PnL) | ✅ | `adaptive/online_tuner.py`; strategy engine. |
| Evolved params load at startup | ✅ | `evolution.load_evolved`, `data/evolved_params.json`. |
| Evolution continuous / interval / trigger on trade | ✅ | Config: `continuous_enabled`, `interval_hours`, `trigger_on_trade`, `after_n_trades`. |
| Evolution auto-apply | ✅ | `evolution.auto_apply`; apply best params after run. |
| Kill losers script (suggest whitelist removals) | ✅ | `scripts/kill_losers_review.py`. |
| Generate evolved params (run evolution to populate file) | ⚙️ | Run paper or evolution 7–14 days; ensure `evolved_params.json` written; restart with load_evolved. |
| Bias allocator to winners | ⚙️ | Lower `strategy_allocator.exploration_c` (e.g. 0.85). |
| Evolution unified (evolve_once, apply_evolved_params) | ✅ | evolution_unified.evolve_once + apply_evolved_params; self_improver calls them (interval + trigger_on_trade). |
| Single param space (GA vs Ultimate vs config) | ✅ | evolution/param_space.EVOLVABLE_PARAM_BOUNDS; GA and unified use get_bounds(). |
| Multi-objective / composite fitness (Sharpe, drawdown, etc.) | ✅ | evolution_use_composite_fitness, composite_calmar_weight, drawdown_penalty, overfit_penalty in evolve_once. |
| Walk-forward in evolution | ✅ | evolution_walk_forward_train_ratio; make_paper_loop_fitness uses it. |
| Rollback / versioning of evolved params | ✅ | apply_evolved_strategies.rollback_to_previous(), get_version_history; scripts/rollback_evolved_params.py. |
| Allocator decay after evolution apply | ✅ | evolution_allocator_decay_after_apply; self_improver calls decay_strategy_allocator_stats after apply. |
| Weekly PnL review; drop one loser or raise confidence | 📌 | Process: run kill_losers_review; trim whitelist; tune confidence. |

---

## 3. Alpha and strategies (one better source beats many weak)

| Item | Status | Where / action |
|------|--------|----------------|
| Unified strategy engine (RSI/BB/MACD + regime) | ✅ | `strategies/unified/strategy_engine.py`. |
| Multi-timeframe signals (primary + entry TF) | ✅ | `signal_multi_timeframe_enabled`, `signal_primary_timeframe`, `signal_entry_timeframe`. |
| External alpha (URL → signals) | ✅ | `external_alpha_enabled`, `external_alpha_url`; scanner merges. |
| Strategy plugin registry | ✅ | `strategy_plugin_modules`; `get_strategies(config)` → analyze(market_data). |
| Strategy library (tier + algorithmic) | ✅ | `strategy_library.enabled`, `strategy_library_strategies_enabled`. |
| Regime boost (numpy momentum/vol) | ✅ | `ml/regime_boost.py`; `use_regime_lstm_boost`. |
| Unified AI brain (PinnacleAIBrain) | 🔌 | `unified_ai_brain`; multi-agent signals when scanner has no cache. |
| LSTM/GRU for next-bar or regime | ✅ | ml/lstm_regime.py stub (re-exports regime_boost); lstm_regime_forward placeholder for real model. |
| Evolution-strategy reward loop | ✅ | ml/evolution_strategy_reward.py (record_reward, suggest_param_jitter); use_evolution_strategy_reward in execution. |
| Feature pipeline: autoencoder | 📋 | Compress features before classifier/regressor in `ml/`. |
| Retrain XGBoost / ensemble periodically | ⚙️ | Run `ml/train_xgboost.py` on recent data (e.g. monthly). |
| Order-flow imbalance or volatility regime in engine | ✅ | use_volatility_regime_scale + volatility_regime_high_threshold in strategy engine (scale confidence in high vol). |

---

## 4. Execution (don’t give back the edge)

| Item | Status | Where / action |
|------|--------|----------------|
| Max slippage enforced (reject over limit) | ✅ | Execution engine returns rejected when slippage > max_slippage_pct. |
| Order fill timeout / cancel | ✅ | `order_fill_timeout_seconds`; wait then cancel if unfilled. |
| VWAP/TWAP for large orders | ✅ | `vwap_large_order_threshold_aud`, `use_twap_for_large_orders`; twap_slicer. |
| Multi-venue (Kraken + Coinbase) | ✅ | `multi_venue_enabled`, `multi_venue_min_notional_aud`. |
| Spread gate (skip when spread > bps) | ✅ | `max_spread_bps`; _get_spread_bps in execution. |
| Implementation shortfall by strategy (for gate) | ✅ | IS tracker; append with strategy; use_is_gate. |
| Pre-trade exposure/position gate | ✅ | Main loop: (exposure + order) ≤ max; (position + order) ≤ max per symbol. |
| Correlation-aware sizing | ✅ | `use_correlation_aware_sizing`; correlation_matrix fed from risk engine. |
| Volatility-adjusted limits | ✅ | `use_volatility_adjusted_limits`; realized_vol_pct from equity. |
| Edge-cost gate in live | ✅ | `edge_cost_gate.modes` includes "live"; live params when set. |
| TWAP in live for large orders (config) | ⚙️ | Set `use_twap_for_large_orders: true`, threshold (e.g. 80 AUD). |
| Tighter slippage for live | ⚙️ | `execution_engine.max_slippage_pct` (e.g. 0.5%). |
| Iceberg / dark pool execution | ⚙️ | `execution_engine.iceberg_enabled`, `dark_pool_enabled` (placeholders; enable when venue supports). See RATING_AND_ROADMAP.md. |
| DCA levels (multi-level entries) | ✅ | execution_engine.dca_levels_pct (e.g. [0.33, 0.33, 0.34]); _dca_expand_signals in execution. |

---

## 5. Risk and safety

| Item | Status | Where / action |
|------|--------|----------------|
| Circuit breaker (daily loss, consecutive losses) | ✅ | `UnifiedRiskManager`; check_circuit_breaker in loop. |
| VaR breach → trip circuit breaker | ✅ | After quant_fund risk; trip_circuit_breaker(); var_breach_alert_enabled. |
| Correlation matrix fed to config | ✅ | From quant_fund risk each cycle; correlation-aware sizing. |
| Stress test script | ✅ | `scripts/stress_test_risk.py`. |
| trip_circuit_breaker(reason) public API | ✅ | `risk/unified_risk_manager.py`. |
| Emergency stop conditions (drawdown, etc.) | ✅ | _check_emergency_stop; state = EMERGENCY_STOP. |
| Circuit breaker in alert triggers | ✅ | `monitoring.alerts.triggers` includes circuit_breaker. |
| All risk config (daily loss, drawdown, stops, auto-reduce) | ⚙️ | See IMPROVEMENT_EVERYTHING §2; tune in unified_config.yaml. |
| Emergency shutdown (latency, flash crash, network, arb) | ✅ | `risk.emergency_shutdown` in config; _check_emergency_stop in unified_trading_system. |
| Multi-factor risk engine (factor exposure) | 🔌 | `quant_fund_upgrades.multi_factor_risk_engine`. |
| Quantum Monte Carlo VaR/CVaR + circuit breaker | 🔌 | `quantum_unified_stubs`; use_quantum_monte_carlo_risk, use_quantum_var_circuit_breaker. |
| Quantum transparency (simulated disclosure) | ✅ | `quantum_features.quantum_simulated_disclosure: true`; docs state quantum is simulated unless real hardware connected. |
| FPGA-style risk block (single trade_valid → trade_approved) | ✅ | execution/risk_compliance_audit.pre_trade_risk_block(); one call returns (approved, reason). |

---

## 6. Data, universe, and observability

| Item | Status | Where / action |
|------|--------|----------------|
| Dynamic universe (refresh from universe_builder) | ✅ | `dynamic_universe_enabled`, `dynamic_universe_interval_cycles`, `dynamic_universe_top_n`. |
| Correlation ID (cycle → execution → audit) | ✅ | cycle_correlation_id; execute_signals(..., correlation_id); append_order/fill. |
| Latency budget (cycle_total_s, exec_s) | ✅ | Logged each cycle in main loop. |
| Tick store / data lake | ✅ | Config: persist_tick_store, use_lake_read, persist_to_lake. |
| Alerts (Telegram/email, triggers) | ✅ | monitoring.alerts; triggers: drawdown, daily_loss, consecutive_losses, error_rate, circuit_breaker. |
| Prometheus/Grafana | ✅ | monitoring.prometheus, monitoring.grafana. |
| Turn alerts on (TELEGRAM_BOT_TOKEN, etc.) | ⚙️ | Set in .env; required for live. |
| Universe builder with real volume rank | ✅ | `utils.universe_builder`: fetch_markets + fetch_tickers, sort by quote volume. |
| Funding rates feed (crypto) | ✅ | ml/funding_rates.py; use_funding_rate_filter, funding_rate_skip_long_threshold in strategy engine. |
| Options flow / volatility surface | ✅ | `ml/options_flow.py` stub; options_flow_score(data) when data provided; regime or sentiment. |
| 23-language: position_sizing, signal_filter, drawdown, slippage | ✅ | Orchestrator; use_*_gate in config. |
| Per-task timeouts for 23-language | ✅ | `multi_language.task_timeouts`; orchestrator uses per-task timeout in execute_task. |
| /ready, /metrics, /capabilities per service | ✅ | Orchestrator: get_ready(), get_metrics(), get_capabilities(); HTTP services can expose same. |
| Export performance series script | ✅ | scripts/export_performance_series.py (paper_results, ledger, allocator → CSV/JSON). |
| TCA summary (slippage by strategy/symbol) | ✅ | scripts/tca_summary.py → data/tca_summary.json. |

---

## 7. Ops, scripts, and process

| Item | Status | Where / action |
|------|--------|----------------|
| Pre-live check | ✅ | `scripts/pre_live_check.py` (includes edge gate when live_require_paper_edge). |
| Readiness score (0–100) | ✅ | `scripts/readiness_score.py` (--include-paper for paper evidence). |
| Live vs backtest consistency | ✅ | `scripts/live_vs_backtest_consistency.py` (slippage vs backtest assumptions). |
| Health check (validate + paper-smoke) | ✅ | `scripts/health_check.py --paper-smoke`. |
| Walk-forward backtest | ✅ | `scripts/walk_forward_gate.py --csv ... --train-days 21 --test-days 7`. |
| Stress test risk | ✅ | `scripts/stress_test_risk.py`. |
| Kill losers review | ✅ | `scripts/kill_losers_review.py`. |
| Backup config and logs | ✅ | `scripts/backup_config_and_logs.sh`. |
| Deploy production | ✅ | `scripts/deploy_production_linux.sh`. |
| Health check on schedule | ⚙️ | Cron: health_check.py weekly or after config change. |
| Backup daily | ⚙️ | Cron: backup_config_and_logs.sh. |
| Paper 2–4 weeks before live | 📌 | Process. |
| Go live small; scale after sustained results | 📌 | Process. |
| One config change at a time | 📌 | Process. |
| When you get an alert, do something | 📌 | Process. |
| Run validate after pulls/config changes | 📌 | `python main.py validate`. |
| Complete LIVE_CHECKLIST before live | 📌 | See docs/LIVE_CHECKLIST.md. |

---

## 8. Backtest and validation

| Item | Status | Where / action |
|------|--------|----------------|
| Unified event backtester | ✅ | `backtest/unified_event_backtester.py`; same flow as live. |
| Slippage/latency in backtest | ✅ | slippage_bps, latency_ms in backtester. |
| Walk-forward script | ✅ | scripts/walk_forward_gate.py. |
| Backtest realism (slippage bps, spread, market impact) | ⚙️ | backtest.* in config; tune so backtest doesn’t overstate edge. |
| OOS train/test ratio | ✅ | backtest.oos_train_ratio in config; backtester reads it; run_backtest_oos uses train_ratio. |
| Rate limit reject simulation | ✅ | backtest.rate_limit_reject_pct; backtester randomly skips signals at that %. |
| Fill probability (realism) | ✅ | backtest.fill_probability (e.g. 0.98 = 2% of fills rejected); backtester skips fill with (1-p) prob. |

---

## 9. Evolution (deeper)

| Item | Status | Where / action |
|------|--------|----------------|
| Load evolved, continuous, trigger_on_trade, auto_apply | ✅ | Config; evolution_unified may be stubbed. |
| Param space, persistence, GA options | ✅ | evolution/param_space.py; apply_evolved_strategies; tests. |
| evolution_unified (evolve_once, apply_evolved_params) | ✅ | evolution_unified.py; self_improver._evolution_once calls evolve_once + apply_evolved_params. |
| Single evolvable param space (GA + Ultimate + config) | ✅ | param_space.EVOLVABLE_PARAM_BOUNDS; filter_to_config_keys for apply. |
| Multi-timeframe fitness in evolution | ✅ | evolution_multi_timeframes, evolution_multi_timeframe_weights in evolve_once. |
| Overfit penalty (train >> test in walk-forward) | ✅ | evolution_overfit_penalty_weight in make_paper_loop_fitness. |
| Strategy whitelist for evolution backtest | ✅ | evolution_strategy_whitelist_override in config and evolve_once. |
| Rollback to previous evolved params | ✅ | rollback_to_previous(); scripts/rollback_evolved_params.py. |
| Backup before auto_apply | ✅ | evolution_backup_before_apply; write_evolved_params(backup_before=True). |
| Allow evolution apply in live (explicit flag) | ✅ | evolution_allow_apply_live; self_improver checks run_mode and allow_apply_live. |

---

## 10. Infrastructure and 10G

| Item | Status | Where / action |
|------|--------|----------------|
| 10G verification | 🖥️ | Run 10gbe_performance_test.py when R730 + Solarflare + switches up. |
| Switch tuning (jumbo, QoS, cut-through) | 🖥️ | SWITCH_IMPROVEMENTS_CHECKLIST.md. |
| NTP on R730 and switches | 🖥️ | Same clock for logs and latency. |
| R730: 64GB RAM, Ubuntu, Solarflare, OpenOnload | 🖥️ | DEPLOYMENT_R730_AND_DESKTOP.md, R730_SETUP.md. |
| network_10gbe config (latency, ring sizes, arb) | ⚙️ | unified_config.yaml; enable when hardware ready. |
| HFT enabled (after 10G verified) | ⚙️ | hft.enabled; keep disabled until PnL positive in paper. |
| Real 23-language HTTP services (Rust/C++/Go) | 📋 | Docker mesh; multi_language.endpoints; MULTI_LANGUAGE_23_README.md. |

---

## 11. Optional modules (unlock more when present)

| Module | Adds |
|--------|------|
| `unified_ai_brain` (PinnacleAIBrain) | Multi-agent AI signals; fallback when scanner has no cache. |
| `unified_language_orchestrator` | 23-language task mesh; position sizing, signal filter, risk gates. |
| `hft_engine.hft_scalping_engine` | HFT order-book/tick signals (enable when PnL positive in paper). |
| `hft.advanced_realtime_hft_infrastructure` | Advanced HFT event loop; OB/trades → signal ring. |
| `utils.universe_builder` | Adaptive symbol selection (stub; replace with real volume fetch). |
| `quantum_unified_stubs` | Quantum Monte Carlo VaR/CVaR; optional circuit breaker. |
| `quant_fund_upgrades.multi_factor_risk_engine` | Factor exposure, portfolio risk (already used when present). |

---

## 12. Beyond (institutional / multi-asset / external)

| Item | Category | Action |
|------|----------|--------|
| Spot–perpetual arbitrage | Execution | Add Kraken futures; Hummingbot-style arb detection and hedging. |
| Cross-exchange market making | Execution | Two-venue order books and skew. |
| Avellaneda-style market making | Execution | Inventory skew, reservation price (ref: Hummingbot). |
| Funding rates (crypto) | Data | Optional feed; regime or filter (e.g. skip longs when funding high). |
| Options flow / vol surface | Data | If options data; regime or sentiment. |
| Cross-asset signals (equities → crypto) | Data | Lead-lag or correlation as macro input. |
| Hummingbot: TWAP/DCA, connector patterns | Ref | EXTERNAL_SOURCES_INTEGRATION.md; already have TWAP/VWAP. |
| Stock-Prediction-Models: LSTM, autoencoder, evolution-strategy | Ref | Port concepts to ml/; evolution-strategy reward loop. |
| HFT FPGA: risk block, order types, pipeline | Ref | Pre-trade risk aligned; document order types; FPGA spec later. |
| Solana / Jito / DEX (if adding chains) | Ref | Auto-solana pattern; Jito, RPC, Telegram. |
| CCXT upgrade | Ref | pip install ccxt --upgrade; use repo for exchange reference. |

---

## 13. Config quick reference (unified_config.yaml)

| Section | Key knobs |
|---------|-----------|
| capital | starting_capital_aud, min/max_position_size_aud, max_total_exposure_pct, max_concurrent_positions |
| risk | max_daily_loss_pct, max_drawdown_pct, stop_loss_pct, take_profit_pct, max_consecutive_losses, circuit_breaker_dd_pct, var_breach_pct, use_volatility_adjusted_limits, auto_reduce_* |
| edge_cost_gate | enabled, modes (include "live"), min_edge_pct, buffer_mult, live_min_edge_pct, live_buffer_mult |
| ai_brain | min_signal_confidence, live_min_signal_confidence, max_concurrent_signals |
| execution_engine | max_slippage_pct, order_fill_timeout_seconds, max_spread_bps, use_is_gate, max_avg_is_bps, vwap_large_order_threshold_aud, use_twap_for_large_orders |
| strategies | strategy_whitelist, use_regime_lstm_boost, regime_filter_enabled |
| continuous_scan | signal_multi_timeframe_enabled, external_alpha_*, strategy_plugin_modules, dynamic_universe_* |
| evolution | load_evolved, auto_apply, continuous_enabled, trigger_on_trade, after_n_trades |
| strategy_allocator | enabled, exploration_c, ema_alpha |
| monitoring.alerts | enabled, telegram, triggers (drawdown, daily_loss, consecutive_losses, error_rate, circuit_breaker) |

---

## 14. Priority order (what to do first)

**Full walkthrough with checklists and commands:** [PRIORITY_ORDER.md](PRIORITY_ORDER.md). Validate Priority 1: `python scripts/validate_priority_order.py --config unified_config.yaml`. Cron examples: `scripts/cron_example.txt`. Presets: `config/priority1_live_presets.yaml`.

1. **Config + process:** Alerts on (Telegram/email); live confidence 0.78+; edge gate for live; run pre_live_check and health_check; paper 2–4 weeks; kill_losers_review weekly.
2. **Already in code:** Use IS gate, pre-trade gate, latency budget, regime boost, evolution load, allocator, walk-forward and stress_test scripts.
3. **Next code:** evolution_unified implementation; LSTM/evolution-strategy in ml/; emergency shutdown conditions; 23-lang per-task timeouts.
4. **Optional modules:** AI brain, HFT engine, real 23-lang services, universe builder with volume.
5. **Infra:** 10G test, switch checklist, NTP, backups cron, deploy.
6. **Beyond:** Perps/arb, market making, funding/options data, cross-asset (when you want institutional/multi-asset).

---

## References (source docs)

| Doc | Content |
|-----|---------|
| [FURTHER_AND_BEYOND.md](FURTHER_AND_BEYOND.md) | Absolute levers, next tier, beyond. |
| [CAPABILITY_EXPANSION.md](CAPABILITY_EXPANSION.md) | What “more capable” means; implemented expansion; cron. |
| [IMPROVEMENT_EVERYTHING.md](IMPROVEMENT_EVERYTHING.md) | Exhaustive checklist (capital, risk, execution, strategies, evolution, HFT, 10G, monitoring, etc.). |
| [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) | Prioritized actions. |
| [ADVANCE_BOT_FURTHER.md](ADVANCE_BOT_FURTHER.md) | Tiers 1–6. |
| [BEYOND_ABSOLUTE.md](BEYOND_ABSOLUTE.md) | The five absolute levers. |
| [EVOLUTION_IMPROVEMENTS.md](EVOLUTION_IMPROVEMENTS.md) | Evolution subsystem improvements. |
| [IMPLEMENTABLE_NOW.md](IMPLEMENTABLE_NOW.md) | Config-only, wiring, new features, optional modules. |
| [EXTERNAL_SOURCES_INTEGRATION.md](EXTERNAL_SOURCES_INTEGRATION.md) | Hummingbot, Stock-Prediction-Models, HFT FPGA, Auto-solana, CCXT. |
| [ARGUS_IMPROVEMENTS_MASTER.md](ARGUS_IMPROVEMENTS_MASTER.md) | Architecture, optional modules, config, quick wins. |
| [LIVE_CHECKLIST.md](LIVE_CHECKLIST.md) | Before going live. |

This document is the **single place** for “everything possible that will take the bot beyond its capabilities.” Use the tables above to see what’s done (✅), what to turn on (⚙️), what to build (📋), what to plug in (🔌), what’s infra (🖥️), and what’s process (📌).
