# Everything That Can Be Implemented Now

A single reference of **all** improvements and features that are either already in the codebase (config/wiring only), partially implemented, or explicitly documented as addable. Use this to choose what to implement next.

---

## 1. Config-only (no code changes)

Turn on or tune in **unified_config.yaml**; behavior already exists in code.

| What | Config path | Action |
|------|-------------|--------|
| **Evolved params at startup** | `evolution.load_evolved: true`, `evolution.params_path` | Load best params from JSON so strategies use evolved values. |
| **23-language regime** | `multi_language.use_regime_estimate: true` | Run regime_estimate each cycle; inject into strategy engine and cycle_ctx. |
| **23-language drawdown gate** | `multi_language.use_drawdown_check: true` | Skip execution when 23-language drawdown check fails. |
| **23-language slippage gate** | `multi_language.use_slippage_estimate: true`, `max_slippage_bps: 100` | Skip execution when median slippage (bps) > threshold. |
| **23-language risk gate** | `multi_language.use_risk_all: true` or `use_conservative_risk: true` | Skip execution when risk check fails (all 23 or correctness-only). |
| **Conservative cycle boost** | `multi_language.use_conservative_cycle_boost: true` | Use correctness-language median for confidence boost. |
| **Live min confidence** | `ai_brain.min_signal_confidence: 0.72` (or 0.75) | Fewer, higher-conviction trades in live. |
| **Edge-cost gate for live** | `edge_cost_gate.modes: ["paper", "backtest", "live"]` | Apply edge-cost gate in production. |
| **Strategy allocator** | `strategy_allocator.enabled: true`, `timeframe: "1m"`, `modes` | PnL-based ranking; already wired in loop and _record_trade. |
| **Volatility-adjusted limits** | `risk.use_volatility_adjusted_limits: true` | Scale position/daily loss by `realized_vol_pct` (fed from equity in loop). |
| **Correlation-aware sizing** | `execution_engine.use_correlation_aware_sizing: true` + `correlation_matrix` | Scale weights by correlation; need to feed `correlation_matrix` from monitoring or compute. |
| **Alerts** | `monitoring.alerts.telegram` / `email`, `triggers` | Drawdown, daily loss, consecutive losses, error rate; add circuit_breaker to triggers if desired. |
| **Circuit breaker alert** | Add to `monitoring.alerts.triggers` | Alert when circuit breaker trips (already in risk). |
| **Backtest realism** | `backtest.slippage_bps`, `latency_ms`, `spread_bps`, etc. | Tune so backtest doesn’t overstate edge. |
| **Paper overrides** | `paper_trading.min_signal_confidence`, `partial_tp_at_pct`, `trailing_stop_enabled` | Stricter paper or let winners run. |
| **Regime filter** | `strategies.regime_filter_enabled: true` | Trend strategies in trend regime, MR in range. |
| **Live disabled strategies** | `runtime.live_disabled_strategies` | Disable known losers in live. |
| **Execution** | `execution_engine.max_slippage_pct`, `order_fill_timeout_seconds`, `use_twap_for_large_orders` | Tighter slippage, timeout, TWAP for large orders. |
| **Continuous scan** | `continuous_scan.interval_seconds`, `top_n`, `diversity_max_per_symbol` | Faster reaction, diversity. |
| **Evolution** | `evolution.trigger_on_trade`, `after_n_trades`, `continuous_enabled`, `auto_apply` | Evolve every N trades or on schedule; auto-apply best. |

---

## 2. Wiring / small code (config exists or trivial to add)

Behavior or config is present; needs one-off wiring or a small code path.

| What | Where / how | Action |
|------|-------------|--------|
| **Feed correlation_matrix into config** | Loop or monitoring → `config.correlation_matrix` | Compute or fetch correlation (e.g. from returns); set on config each cycle so `use_correlation_aware_sizing` and execution engine use it. |
| **Feed realized_vol_pct into config** | Already in loop (equity history → vol) | Ensure `config.realized_vol_pct` is set each cycle when `use_volatility_adjusted_limits` is true (loop already updates; verify execution/risk read it). |
| **Pre-live check script** | `scripts/pre_live_check.py` | Run before going live; ensure it validates credentials, edge_cost_gate modes, alerts. |
| **Health check** | `scripts/health_check.py --paper-smoke` | Run weekly or pre/post config change; add to cron. |
| **Backup cron** | `scripts/backup_config_and_logs.sh` | Schedule daily. |
| **Config version migration** | `config_version` in YAML + `UnifiedConfig.config_version` | When you introduce breaking config changes, bump `config_version` and add migration logic in config loader. |
| **Logging level / rotation** | `logging.level`, `logging.file`, `max_file_size_mb` | If present in config, wire to logging config at startup. |

---

## 3. New features (documented, addable in code)

From **docs/TWENTY_THREE_LANGUAGES_EVERYTHING_ADDABLE.md**, **IMPROVEMENT_EVERYTHING.md**, and **ARGUS_IMPROVEMENTS_MASTER.md**.

### 3.1 23-language mesh (orchestrator + protocol)

| Feature | Description | Effort |
|---------|-------------|--------|
| **position_sizing aggregation** | Call `execute_position_sizing_all`; use `size_pct_median` or `size_pct_conservative` to cap or scale signal size. | Small: orchestrator has method; wire in loop before execution. |
| **signal_filter aggregation** | Call `execute_signal_filter_all`; accept only if majority (or unanimity) accept. | Small: orchestrator has method; add gate before execution. |
| **correlation_estimate** | New task type: series_a/series_b → correlation; stats languages preferred. | Medium: add in-process handler + optional HTTP; call from loop when multi-symbol. |
| **liquidity_score** | New task type: order book → liquidity_score, depth_bps. | Medium: in-process + optional HTTP. |
| **market_impact** | New task type: side, quantity, adv, volatility → impact_bps. | Medium: in-process + optional HTTP. |
| **confidence_calibration** | New task type: historical confidences + PnL → calibrated_confidence. | Medium: stats languages; need history store. |
| **heartbeat** | New task type: cycle_id, timestamp → ok, latency_ms; observability. | Small: in-process only. |
| **Per-task timeouts** | Already in config `multi_language.task_timeouts`; ensure orchestrator uses them per task type. | Small: pass timeouts into orchestrator. |
| **/ready, /metrics, /capabilities, /batch, /warm** | Per multilang service: add endpoints for readiness, metrics, batch execute, warm-up. | Medium: multilang service already has some; extend all 23. |

### 3.2 Execution and risk

| Feature | Description | Effort |
|---------|-------------|--------|
| **Order fill timeout** | Cancel order if not filled after `order_fill_timeout_seconds`; config exists. | Small: check execution engine uses it. |
| **TWAP for large orders** | Slice order when size > threshold; `use_twap_for_large_orders`, `vwap_large_order_threshold_aud` in config. | Small: verify twap_slicer is called. |
| **VaR breach trigger** | Feed VaR from monitoring; trip circuit breaker or alert when breach > `risk.var_breach_pct`. | Medium: need VaR computation in loop or monitoring. |
| **Emergency shutdown conditions** | Implement each condition in `emergency_shutdown.conditions` (latency spike, execution delay, flash crash, network failure, arb failure) if not already. | Medium: audit and add missing. |

### 3.3 Monitoring and observability

| Feature | Description | Effort |
|---------|-------------|--------|
| **Telegram/email alerts** | Configure `monitoring.alerts.telegram` / `email` and triggers; ensure alerting code sends on drawdown, daily loss, consecutive losses, error rate, circuit breaker. | Small: config + verify send path. |
| **Prometheus metrics** | Expose portfolio, PnL, drawdown, win rate, error rate, version; Grafana dashboards. | Small–medium: already have client; add labels and dashboards. |
| **Trade ledger language calls** | Already have `record_language_call`; extend with task_type, correlation_id, latency in logs. | Small. |
| **Correlation ID** | Pass correlation_id through orchestrator and multilang requests for tracing. | Small: protocol supports it; thread through. |

### 3.4 Data and backtest

| Feature | Description | Effort |
|---------|-------------|--------|
| **Walk-forward backtest** | Train on rolling window, test on next period; script exists (`scripts/walk_forward_gate.py`); integrate or document. | Medium. |
| **Export performance series** | `scripts/export_performance_series.py` for analysis; ensure it runs on paper_results or ledger. | Small. |
| **Stress test risk** | `scripts/stress_test_risk.py`; run periodically. | Small. |
| **Backtest: market impact, rate limit reject** | Add market_impact_bps_per_10k_usd, rate_limit_reject_pct to backtester. | Small. |

### 3.5 ML and evolution

| Feature | Description | Effort |
|---------|-------------|--------|
| **Retrain XGBoost periodically** | `ml/train_xgboost.py` on recent OHLCV; cron or manual. | Small. |
| **Feature review** | Add/remove features in `ml/feature_library_300.py` from importance. | Small. |
| **Evolution auto-apply** | Already in config; ensure evolution pipeline writes `evolved_params.json` and that load at startup is on. | Small. |
| **Paper run versioned** | `scripts/paper_run_versioned_90d.py` for versioned long paper runs. | Small. |

---

## 4. Optional modules (add or restore)

Referenced by the unified system but **missing** or off-path. Adding them unlocks more.

| Module | Path | Status |
|--------|------|--------|
| **Pinnacle AI brain** | `unified_ai_brain.py` → `PinnacleAIBrain` | Restored: wraps StrategyEngine; generate_trading_signals, get_adaptation_status, on_trade_closed. |
| **Multi-factor risk engine** | `quant_fund_upgrades.multi_factor_risk_engine` | Restored: price history, correlation_matrix, VaR/CVaR stub. |
| **HFT scalping engine** | `hft_engine.hft_scalping_engine` | Restored: stub (analyze_* return None; scan_for_opportunities returns []). |
| **Advanced HFT infrastructure** | `hft.advanced_realtime_hft_infrastructure` | Restored: stub (run_event_loop, push_*, drain_signals no-op/empty). |
| **Universe builder** | `utils.universe_builder` | Restored: default liquid USD list; replace with CCXT volume for production. |
| **Quantum unified stubs** | `quantum_unified_stubs.py` + `quantum.quantum_unified_stubs` | Restored: re-export; VaR/CVaR classical fallback. |

23-language **position_sizing** and **signal_filter** gates are wired; set `use_position_sizing_gate` / `use_signal_filter_gate` under `multi_language` to enable.

---

## 5. Infrastructure and ops (no app code)

| Item | Action |
|------|--------|
| **10G path** | Run `10gbe_performance_test.py`; fix MTU/latency; then consider HFT and micro-trading. |
| **Switch tuning** | Follow [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md): jumbo, QoS 7, cut-through, edge ports. |
| **NTP** | On R730 and switches. |
| **Backups** | Cron daily: `scripts/backup_config_and_logs.sh`. |
| **Deploy** | `scripts/deploy_production_linux.sh`; systemd with correct WorkingDirectory and config. |
| **CI** | Run `python main.py validate` (and optionally `scripts/health_check.py --paper-smoke`) in GitHub Actions. |

---

## 6. Process and discipline

| Action |
|--------|
| Paper 2–4 weeks before live; review win rate, drawdown, monthly return. |
| Go live with small capital; scale only after sustained results. |
| One config change at a time. |
| Run `python main.py validate` after pulls and config changes. |
| Complete [LIVE_CHECKLIST.md](LIVE_CHECKLIST.md) before switching to live. |

---

## 7. Quick priority order

1. **Config-only (fast):** Enable evolution load, 23-language gates (regime/drawdown/slippage) if desired, alerts, edge_cost_gate for live, strategy allocator, live min_confidence.
2. **Wiring (1–2 days):** Correlation matrix feed, pre_live_check and health_check in cron, backup cron.
3. **Small code (per item):** position_sizing/signal_filter gates, per-task timeouts, heartbeat, order fill timeout/TWAP verification, alert send path.
4. **Larger (per roadmap):** New 23-lang task types, VaR breach, emergency shutdown conditions, walk-forward, optional modules (AI brain, HFT, quant risk, universe builder, quantum stubs).
5. **Infra/ops:** 10G test, switch checklist, NTP, backups, deploy, CI.

---

## References

| Doc | Content |
|-----|---------|
| **ARGUS_IMPROVEMENTS_MASTER.md** | Architecture, optional modules, config knobs, quick wins. |
| **IMPROVEMENT_EVERYTHING.md** | Exhaustive checklist (capital, risk, execution, strategies, evolution, HFT, 10G, monitoring, etc.). |
| **IMPROVEMENT_ROADMAP.md** | Prioritized actions. |
| **TWENTY_THREE_LANGUAGES_EVERYTHING_ADDABLE.md** | All addable task types, fields, API, observability for 23 languages. |
| **EVERYTHING_ELSE_IN_THE_BOT.md** | Entry points, loop, components, config, data, scripts. |
| **LIVE_CHECKLIST.md** | Before going live. |
| **UPGRADE.md** | v3.0 upgrade and integration recap. |
