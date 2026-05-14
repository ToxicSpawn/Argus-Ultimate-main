# Argus Capability Expansion – Making the Bot Do More

This document defines how to make Argus **more than capable**: broader strategies, better execution, stronger risk, more assets, and clearer observability. Use it as a prioritized roadmap.

---

## 1. What “More Capable” Means

| Area | Today | More capable |
|------|------|--------------|
| **Strategies** | Unified engine (RSI/BB/MACD + regime), whitelist, allocator | More timeframes, more strategy types, external alpha, cross-asset signals |
| **Data** | OHLCV, ticker, optional order book | Order book depth, funding rates, options flow, alternative data |
| **Execution** | Single/multi-venue, VWAP/TWAP, slippage guardrails | Spread/liquidity gates, fill timeout, smarter slicing, dark pools |
| **Risk** | VaR/CVaR, circuit breaker, volatility-adjusted limits, correlation-aware sizing | Auto-fed correlation, stress tests, factor exposure, margin-aware limits |
| **Universe** | Fixed trading_pairs / best_symbols | Dynamic universe (liquidity/volatility filters), multi-asset (FX, equities) |
| **Observability** | Logs, optional Prometheus/Grafana, audit | Alerts (Telegram/email), correlation IDs, performance attribution |

---

## 2. Implemented in This Expansion (All Done)

- **Correlation matrix feed:** When the quant fund risk engine runs each cycle, its `correlation_matrix` is written to `config.correlation_matrix` so **correlation-aware position sizing** works without manual YAML.
- **Optional spread gate:** `execution_engine.max_spread_bps` (e.g. 50). If set > 0 and order book is available, orders are skipped when (ask-bid)/mid > threshold.
- **Multi-timeframe signals:** `signal_multi_timeframe_enabled`, `signal_primary_timeframe` (e.g. 1h), `signal_entry_timeframe` (e.g. 15m). Scanner runs strategy engine on both; keeps only entry-TF signals where primary-TF agrees (same symbol + direction).
- **External alpha:** `external_alpha_enabled`, `external_alpha_url`, `external_alpha_timeout_seconds`. Scanner GETs URL; expects JSON array (or `signals`/`signals_list`) of `{symbol, action, confidence, price, ...}`; merges into candidates as `external_alpha`.
- **Strategy plugin registry:** `strategy_plugin_modules`: list of module paths. Each module can expose `get_strategies(config)` (or `register_strategies`) returning `{name: strategy}`; strategies must have `analyze(market_data) -> dict`. Scanner merges as `plugin_<name>`.
- **Dynamic universe:** `dynamic_universe_enabled`, `dynamic_universe_interval_cycles`, `dynamic_universe_top_n`. Every N cycles, `utils.universe_builder.select_top_liquid_usd_pairs` is called and `trading_pairs` is refreshed (merged with current).
- **VaR breach:** After quant_fund risk, if `|VaR_95| * 100 > var_breach_pct`, circuit breaker is tripped via `UnifiedRiskManager.trip_circuit_breaker()`; optional `var_breach_alert_enabled` emits `rec_event(stage="var_breach_alert")`.
- **Correlation ID:** Each cycle gets `cycle_<id>_<timestamp>`; passed to `execute_signals(..., correlation_id=...)` and into audit (order/fill) for tracing.
- **Stress test script:** `scripts/stress_test_risk.py` – runs risk manager with synthetic losses and checks circuit breaker / `trip_circuit_breaker()`.
- **Walk-forward script:** `scripts/walk_forward_gate.py` – train/test split backtest on CSV; use for periodic validation.
- **Pre-live check:** `scripts/pre_live_check.py` – validates config, .env, alerts, risk limits before going live.
- **Health check:** `scripts/health_check.py --paper-smoke` – validate + short paper run; use weekly or after config changes.

---

## 3. Config-Only (No Code) – Do First

From [IMPLEMENTABLE_NOW.md](IMPLEMENTABLE_NOW.md) and [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md):

| What | Config | Action |
|------|--------|--------|
| Live confidence | `ai_brain.min_signal_confidence: 0.72` | Fewer, higher-conviction trades in live. |
| Volatility-adjusted limits | `risk.use_volatility_adjusted_limits: true` | Already computed each cycle; keeps limits tighter when vol is high. |
| Correlation-aware sizing | `execution_engine.use_correlation_aware_sizing: true` | Now works with auto-fed correlation when quant_fund risk is enabled. |
| Order fill timeout | `execution_engine.order_fill_timeout_seconds: 30` | Cancel unfilled orders after N seconds. |
| Spread gate | `execution_engine.max_spread_bps: 50` | Skip orders when spread > 50 bps (if order book available). |
| Alerts | `monitoring.alerts.telegram` / `email`, `triggers` | Drawdown, daily loss, circuit breaker, error rate. |
| Strategy allocator | `strategy_allocator.enabled: true` | PnL-based strategy ranking. |
| Evolution | `evolution.load_evolved: true`, `evolution.auto_apply` | Use evolved params; evolve on schedule or after N trades. |

---

## 4. High-Impact Additions (Next)

### 4.1 Strategies and signals

- **Multi-timeframe signals:** Run strategy engine on 1h and 15m (or configurable); combine or filter (e.g. only take 1h trend + 15m entry).
- **External alpha:** Webhook or API to inject external signals; score and merge with internal signals.
- **Strategy plugin registry:** Load strategies from a directory or config list so new strategies can be added without editing core engine.

### 4.2 Data and universe

- **Dynamic universe:** Use `utils.universe_builder` (or equivalent) to select symbols by liquidity, volatility, and spread; refresh daily or weekly.
- **Order book and spread:** Fetch L2 in the execution path when `max_spread_bps` is set; optional liquidity score for position caps.
- **Funding / options:** Optional funding rate and options flow feeds for crypto; use as regime or filter.

### 4.3 Execution and risk

- **VaR breach alert:** When VaR/CVaR from monitoring or quantum stub exceeds `risk.var_breach_pct`, trigger circuit breaker or alert.
- **Stress test script:** Run `scripts/stress_test_risk.py` (or add it) on a schedule; document in runbook.
- **Walk-forward backtest:** Integrate or document `scripts/walk_forward_gate.py` for periodic validation.

### 4.4 Observability and ops

- **Correlation ID:** Thread `correlation_id` through orchestrator and execution for tracing.
- **Pre-live check:** Run `scripts/pre_live_check.py` before switching to live; add to runbook and checklist.
- **Health check cron:** Run `scripts/health_check.py --paper-smoke` weekly or after config changes.

---

## 5. Optional Modules (Unlock More)

From [ARGUS_IMPROVEMENTS_MASTER.md](ARGUS_IMPROVEMENTS_MASTER.md). Adding or restoring these makes Argus capable of more:

| Module | Adds |
|--------|------|
| `unified_ai_brain` | Multi-agent AI signals; fallback when scanner has no cached opportunities. |
| `unified_language_orchestrator` | 23-language task mesh: position sizing aggregation, signal filter, risk gates. |
| `hft_engine.hft_scalping_engine` | HFT order-book/tick signals (enable when module exists and paper PnL is positive). |
| `utils.universe_builder` | Adaptive symbol selection. |
| `quantum_unified_stubs` | Quantum Monte Carlo VaR/CVaR and optional circuit breaker. |

---

## 6. Summary

- **Implemented:** Multi-timeframe signals, external alpha, strategy plugins, dynamic universe, VaR breach + alert, correlation ID, spread gate, correlation matrix feed, stress test and walk-forward scripts. Config in `unified_config.yaml` under `continuous_scan` and `risk`.
- **Optional modules:** AI brain, language orchestrator, HFT engine, universe builder, quantum stubs – add when available.

## 7. Cron and Ops

Run before going live and on a schedule:

- Pre-live: `python scripts/pre_live_check.py --config unified_config.yaml`
- Health check: `python scripts/health_check.py --config unified_config.yaml --paper-smoke`
- Stress test: `python scripts/stress_test_risk.py --config unified_config.yaml`
- Walk-forward: `python scripts/walk_forward_gate.py --csv data/ohlcv.csv --train-days 21 --test-days 7`

Example crontab (weekly Sunday 2am):  
`0 2 * * 0 cd /path/to/Argus-Ultimate-main && python scripts/health_check.py --config unified_config.yaml --paper-smoke`

Making Argus “more capable” is incremental: enable existing knobs first, then add the high-impact features above, then plug in optional modules as needed.
