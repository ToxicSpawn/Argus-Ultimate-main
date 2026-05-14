# Everything Else in the Bot (Beyond the 23 Languages)

Overview of the **rest of Argus**: entry points, trading loop, strategies, execution, risk, capital, monitoring, config, data, and scripts. The 23-language mesh is one part; this doc is the rest.

---

## 1. Entry points (main.py)

| Command | What it does |
|---------|----------------|
| **paper** | Paper trading: full loop, no real money; uses `unified_config.yaml`, run_mode paper. |
| **live** | Live trading: same loop, real orders (Kraken/CCXT); requires credentials and pre-live check. |
| **backtest** | Backtest on historical data (symbol, days, capital); optional CSV export. |
| **validate** | Validate config and dependencies. |
| **setup** | One-time setup (e.g. config, dirs). |
| **quantum-status** | Show quantum-related config and status. |
| **rl-demo** | RL demo (if enabled). |

**Production entry:** `python main.py paper` or `python main.py live` (after checklist).

---

## 2. Architecture (one cycle)

The **UnifiedSystemArchitecture** (unified_trading_system.py) runs a single loop. Each cycle:

| Step | Component | What happens |
|------|-----------|--------------|
| 1. **Signals** | ContinuousBestTradeScanner (parallel) | AI brain + strategy engine (unified_engine) + optional HFT. Ranks by score, applies strategy_whitelist, diversity, top N. |
| 2. **Disabled filter** | unified_trading_system | Drops signals whose strategy is in paper_trading_disabled_strategies (paper) or live_disabled_strategies (live). |
| 3. **ARGUS processing** | _process_argus_strategies | Order book + risk via language orchestrator (if enabled); ARGUS filters; confidence nudge. |
| 4. **Capital optimize** | CapitalOptimizer1K | Position sizing for ~$1K AUD; portfolio guardrails (no sell when flat, no buy without cash). |
| 5. **23-language cycle** | language_orchestrator | All 23 languages run cycle_plan; median (or conservative) boost applied to signal confidence. |
| 6. **Optional 23-language risk gate** | language_orchestrator | If use_risk_all or use_conservative_risk: run risk across all 23; skip execution if gate fails. |
| 7. **Pre-trade guardrails** | execution + risk | Emergency stop check; pre-trade exposure/position gate (max exposure, max position). |
| 8. **Execute** | KrakenDCAExecutionEngine | Place/cancel orders (CCXT); edge-cost gate; implementation shortfall logged; optional order fill timeout and TWAP for large orders. |
| 9. **Record & monitor** | trade ledger, monitoring | Record fills; update portfolio, PnL, drawdown; call monitoring.update_metrics. |

So: **scanner → filter disabled → ARGUS (OB/risk) → capital optimizer → 23-language boost (and optional risk gate) → guardrails → execute → record**.

**Integration (everything works together):** At startup, when `evolution.load_evolved` is true, evolved params are loaded from `evolution.params_path` into config (supports keys `best_params`, `evolved_params`, `params`). The **strategy allocator** is initialized from config (enabled, timeframe, persist_path, modes); each cycle it **rank_signals** (PnL-based) after the disabled filter, and on each closed trade **record_trade** + **save** so ranking improves over time. Optional **multi_language** gates (all read from config): **use_regime_estimate** runs 23-language regime_estimate on OHLCV closes, adds regime to cycle context, and injects the consensus regime into the strategy engine so allocator/adaptive risk see it; **use_drawdown_check** runs 23-language drawdown_check before execution and skips execution if the gate fails; **use_slippage_estimate** runs 23-language slippage_estimate and skips execution if median slippage (bps) exceeds **max_slippage_bps**. So evolution, allocator, regime injection, drawdown, and slippage gates are wired end-to-end.

---

## 3. Main components (what they do)

| Component | Where | Role |
|-----------|--------|------|
| **ContinuousBestTradeScanner** | services/continuous_best_trade_scanner.py | Fetches opportunities from AI brain, strategy engine, HFT. Applies strategy_whitelist, diversity, top N. |
| **Strategy engine** | strategies/unified/strategy_engine.py | RSI/BB/MACD + regime + online tuner; emits signals tagged unified_engine. |
| **KrakenDCAExecutionEngine** | unified_execution_engine.py | CCXT orders; edge-cost gate; pre-trade exposure/position gate; VWAP/TWAP for large orders; order fill timeout/cancel; implementation shortfall. |
| **CapitalOptimizer1K** | unified_capital_optimizer.py | Sizing for ~$1K; uses strategy allocator and config limits. |
| **Unified risk** | risk/unified_risk_manager.py, risk/unified_risk_facade.py | Daily loss, drawdown, consecutive losses, circuit breaker; adaptive risk controller optional. |
| **Strategy allocator** | Allocator in config | Tracks PnL per strategy (persist_path: data/strategy_allocator_stats.json); favors better performers (ema_alpha, exploration_c). |
| **Market data** | services/market_data_service.py | OHLCV, order book, trades (CCXT or other provider). |
| **Exchanges** | exchanges/ (CCXT) | Kraken primary, Coinbase Advanced secondary; credentials from config/env. |
| **Monitoring** | core/health.py (ArgusHealthMonitor) | update_metrics (portfolio, PnL, drawdown, win rate, error rate); alerts (Telegram/email) when triggers fire. |
| **Trade ledger** | monitoring/trade_ledger.py | Records trades and optional language call logs (unified_trades.db). |

---

## 4. Config (unified_config.yaml) – main areas

| Area | What you control |
|------|-------------------|
| **runtime** | run_mode (paper/live), starting_capital_aud, primary_exchange. |
| **strategies** | strategy_whitelist (e.g. akashic_tier, unified_engine, quantum_momentum_elite), regime_filter_enabled, paper_trading_disabled_strategies, live_disabled_strategies. |
| **paper_trading** | min_signal_confidence, partial_tp_at_pct, trailing_stop_enabled. |
| **risk** | max_daily_loss_pct, max_drawdown_pct, stop_loss_pct, take_profit_pct, circuit_breaker_dd_pct, max_consecutive_losses, auto_reduce_after_n_losses. |
| **edge_cost_gate** | min_edge_pct, buffer_mult, modes (include "live" for production). |
| **execution_engine** | max_slippage_pct, order_fill_timeout_seconds, use_twap_for_large_orders, vwap_large_order_threshold_aud. |
| **continuous_scan** | interval_seconds, top_n, diversity_max_per_symbol, strategy_whitelist. |
| **evolution** | load_evolved (load data/evolved_params.json), params_path, continuous_enabled, trigger_on_trade, after_n_trades. |
| **strategy_allocator** | enabled, modes, ema_alpha, exploration_c, persist_path. |
| **ai_brain** | min_signal_confidence, num_ai_agents (when unified_ai_brain present). |
| **multi_language** | enabled, use_cycle_aggregate_boost, use_conservative_cycle_boost, use_risk_all, use_conservative_risk, use_regime_estimate, use_slippage_estimate, use_drawdown_check, endpoints (23 services). |
| **monitoring** | alerts (telegram, email), triggers (drawdown, daily_loss, consecutive_losses, error_rate, circuit_breaker). |

---

## 5. Data files (where things are stored)

| Path | Purpose |
|------|---------|
| **data/paper_results.json** | Paper trading results (PnL, trades, per-strategy stats). |
| **data/evolved_params.json** | Evolved strategy params; loaded when evolution.load_evolved is true. |
| **data/strategy_allocator_stats.json** | Allocator PnL/weights per strategy. |
| **data/unified_trades.db** | Trade ledger DB. |
| **data/lake/** | Data lake path (config). |
| **data/ccxt_data_provider** | CCXT cache/config (if used). |

---

## 6. Scripts (operations and checks)

| Script | Use |
|--------|-----|
| **scripts/health_check.py** | `--paper-smoke` or full; run before/after config changes and on a schedule. |
| **scripts/pre_live_check.py** | Validate config (credentials, edge gate modes, etc.) before going live. |
| **scripts/backup_config_and_logs.sh** | Backup config and logs (cron daily). |
| **scripts/deploy_production_linux.sh** | Deploy to production host. |
| **scripts/earnings_report.py** | Earnings reporting. |
| **scripts/paper_loop_30d_unified.py** | Extended paper run. |
| **scripts/show_earnings.py** | Show earnings from results. |
| **scripts/stress_test_risk.py** | Stress test risk logic. |
| **scripts/walk_forward_gate.py** | Walk-forward backtest gate. |
| **scripts/export_performance_series.py** | Export performance series. |
| **scripts/cron_example.txt** | Example cron (health check, backup). |

---

## 7. Optional / missing modules (improve when added)

These are **referenced** by the unified system but may be **missing** or off-path. Enabling or adding them extends the bot.

| Module | What it adds |
|--------|----------------|
| **unified_ai_brain** (PinnacleAIBrain) | Multi-agent AI signals when scanner has no cached opportunities. |
| **quant_fund_upgrades.multi_factor_risk_engine** | Multi-factor portfolio risk. |
| **hft_engine.hft_scalping_engine** | HFT order-book/tick signals (often disabled in paper until PnL positive). |
| **hft.advanced_realtime_hft_infrastructure** | Advanced HFT event loop (OB/trades → signals). |
| **utils.universe_builder** | Adaptive symbol universe. |
| **quantum_unified_stubs** | Quantum Monte Carlo VaR/CVaR, optional circuit breaker. |

The bot runs without them using strategy_engine + whitelist + execution + risk.

---

## 8. Execution and risk (detail)

| Feature | Where | Behavior |
|---------|--------|----------|
| **Edge-cost gate** | unified_execution_engine | Rejects signals when edge &lt; min_edge_pct (with buffer); modes include paper/live. |
| **Pre-trade exposure/position gate** | execution/risk_compliance_audit.py, unified_execution_engine | (Exposure + new order) ≤ max exposure; new order ≤ max position; reject otherwise. |
| **Order fill timeout** | unified_execution_engine | If order still open after order_fill_timeout_seconds, cancel. |
| **TWAP for large orders** | execution/twap_slicer.py, unified_execution_engine | When use_twap_for_large_orders and order size &gt; threshold, slice over time. |
| **Implementation shortfall** | execution/implementation_shortfall.py | Computed and logged for analysis. |
| **Circuit breaker** | risk, unified_trading_system | Stops trading when drawdown or other limits hit; can trigger alerts. |

---

## 9. Docs (where to read more)

| Doc | Content |
|-----|---------|
| **ARGUS_IMPROVEMENTS_MASTER.md** | Architecture, optional modules, config knobs, scripts, code paths, quick wins. |
| **ADVANCE_BOT_FURTHER.md** | Tiers 1–6: alerts, evolution, execution, ML, allocator, 10G, optional. |
| **BEYOND_ABSOLUTE.md** | What actually pushes the bot beyond (edge, learning, alpha, execution, you in the loop). |
| **CODING_LANGUAGES_REFERENCE.md** | All 23 languages and how each is used. |
| **LANGUAGE_STRENGTHS.md** | Strength-based routing and aggregation. |
| **MULTI_LANGUAGE_23_README.md** | 23-language system, config, protocol. |
| **TWENTY_THREE_LANGUAGES_EVERYTHING_ADDABLE.md** | Everything addable to the 23 languages. |
| **EXTERNAL_SOURCES_INTEGRATION.md** | What to take from Hummingbot, Stock-Prediction-Models, HFT FPGA, etc. |
| **LIVE_CHECKLIST.md** | Before going live. |
| **SWITCH_IMPROVEMENTS_CHECKLIST.md** | 10G switch tuning. |

---

## 10. Summary

- **Entry:** main.py → paper / live / backtest / validate / setup / quantum-status.
- **Loop:** Scanner → disabled filter → ARGUS (OB/risk, 23-lang) → capital optimizer → 23-language boost (and optional risk gate) → guardrails → execute → record.
- **Core:** Scanner, strategy engine, execution engine, risk, capital optimizer, monitoring, trade ledger.
- **Config:** unified_config.yaml (runtime, strategies, risk, execution, scan, evolution, allocator, multi_language, monitoring).
- **Data:** paper_results.json, evolved_params.json, strategy_allocator_stats.json, unified_trades.db.
- **Scripts:** health_check, pre_live_check, backup, deploy, earnings, paper loops, stress test, walk-forward, cron example.
- **Optional:** AI brain, HFT engine, multi-factor risk, quantum stubs, universe builder – add to unlock more.

The 23 languages are one slice (cycle plan, order book, risk, volatility, regime, slippage, etc.); **everything else** above is the rest of the bot.
