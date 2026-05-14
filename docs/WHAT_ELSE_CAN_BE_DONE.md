# What Else Can Be Done for the Bot

Everything that can still be added, wired, or improved beyond what’s already in place. Use this after [WHAT_CAN_BE_USED_TO_MAKE_PROFIT.md](WHAT_CAN_BE_USED_TO_MAKE_PROFIT.md) and [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md).

**Implemented (do-it-all pass):** TCA by venue (`tca_summary.py` → `by_venue`); optional PyTorch LSTM (`LSTM_REGIME_MODEL_PATH` → `ml/lstm_regime.py`); options flow stub (`ml/options_flow.py`); CI health_check (`.github/workflows/ci.yml`); cron XGBoost example; [BEYOND_PLACEHOLDERS.md](BEYOND_PLACEHOLDERS.md); .env.example Telegram + LSTM note.

**Everything else added to the bot (this pass):** Quantum VaR breach → circuit breaker and alert when `|VaR_95|*100 > var_breach_pct` (in addition to quant_fund VaR breach). Fallback `correlation_matrix` from OHLCV returns when `use_correlation_aware_sizing` is true and quant_fund does not set it (refreshed every 5 cycles). `correlation_id` in `cycle_ctx` for 23-language tracing. Logging level and file from `unified_config.yaml` → `logging.level`, `logging.file`, `max_file_size_mb`, `backup_count` applied at config load (config_manager). Fixed config load order: `quantum_simulator_use_production` set after config is created.

**"Do it all" pass:** Pre-trade risk block is **wired**: execution engine calls `pre_trade_risk_block()` before every order and skips with rejected fill when not approved; main loop sets `_pre_trade_positions`, `_pre_trade_prices`, `_pre_trade_equity_aud`, etc. Evolution-strategy reward **on by default**: `evolution.use_evolution_strategy_reward: true`. Run-before-live script runs validate_priority_order, pre_live_check, and readiness_score --include-paper. 10/10 config defaults: live_require_paper_edge, fill_probability 0.98, quantum_simulated_disclosure, live_min_trades_paper, live_min_win_rate_pct.

---

## 1. Quick wins (config or 1–2 day wiring)

| What | Where / action |
|------|----------------|
| **Call pre_trade_risk_block before execution** | ✅ **Done.** Execution engine calls `pre_trade_risk_block()` before each order; main loop sets positions/prices/equity on config. Rejected orders return simulated fill with `error=pre_trade_risk_block:reason`. |
| **Turn on evolution-strategy reward** | ✅ **Done.** Default `evolution.use_evolution_strategy_reward: true`; execution records PnL per strategy/symbol for `ml.evolution_strategy_reward`. |
| **23-language gates** | `multi_language.use_regime_estimate: true`, `use_drawdown_check: true`, `use_slippage_estimate: true`, `use_position_sizing_gate: true`, `use_signal_filter_gate: true` (orchestrator already supports; enable in config). |
| **Feed correlation_matrix** | Compute or fetch correlation from returns; set `config.correlation_matrix` each cycle so `use_correlation_aware_sizing` can scale weights (execution engine reads it when true). |
| **External alpha URL** | When you have a signal API: set `continuous_scan.external_alpha_enabled: true` and `external_alpha_url`. |
| **Alerts** | Set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` in .env; ensure `monitoring.alerts.triggers` includes `circuit_breaker`. |
| **Dynamic universe** | `continuous_scan.dynamic_universe_enabled: true`, `dynamic_universe_interval_cycles`, `dynamic_universe_top_n` so scanner refreshes symbols from universe builder. |

---

## 2. Code / features to add (documented)

| What | Description | Effort |
|------|-------------|--------|
| **Feature pipeline: autoencoder** | Compress features in `ml/` before classifier/regressor. | Medium |
| **LSTM/GRU next-bar or regime** | Real model in `ml/` (e.g. `lstm_regime_forward`); currently stub. | Medium |
| **Options flow / volatility surface** | If you add options data; use for regime or sentiment. | Medium |
| **Real 23-language HTTP services** | Docker mesh; `multi_language.endpoints`; Rust/C++/Go services. | Large |
| **New 23-lang task types** | `correlation_estimate`, `liquidity_score`, `market_impact`, `confidence_calibration`, `heartbeat`; see IMPLEMENTABLE_NOW.md §3.1. | Small–medium each |
| **VaR breach → circuit breaker** | Compute VaR in loop or monitoring; call `trip_circuit_breaker` when \|VaR\| > `risk.var_breach_pct`. | Medium |
| **Partial TP / trailing stop in unified loop** | Main loop currently uses signal TP/SL; add position monitor that applies `paper_trading.partial_tp_at_pct` and `trailing_stop_pct` (high-water) on each cycle. | Medium |
| **Walk-forward in backtest** | Train on rolling window, test on next; script `walk_forward_gate.py` exists; integrate or document. | Medium |

---

## 3. Optional modules (all added and enabled)

All optional modules are **wired and enabled** in config. If a module fails to load (e.g. missing deps), the bot falls back so it still runs.

| Module | Config / behaviour |
|--------|---------------------|
| **Pinnacle AI brain** | `ai_brain.enabled`; uses `unified_ai_brain.PinnacleAIBrain`. On load failure, **FallbackAIBrain** (strategy engine only) is used so scanner and loop still get signals. |
| **Multi-factor risk engine** | `quant_fund_upgrades.enabled: true`, `modes: ["paper", "backtest", "live"]`, `risk_engine.enabled: true`. |
| **Quantum Monte Carlo risk** | `quantum_bot.use_quantum_monte_carlo_risk: true`; VaR/CVaR hook runs each cycle (classical fallback in stubs). |
| **HFT scalping engine** | `hft.enabled: true`; `hft_engine.hft_scalping_engine` loaded; analyze_order_book / analyze_trade_flow (stubs return None until you add logic). |
| **Advanced HFT infrastructure** | `hft.use_advanced_hft_infrastructure: true`; OB/trades pushed to event loop, signals drained (stub no-op until you add logic). |
| **Universe builder (real volume)** | `continuous_scan.dynamic_universe_enabled: true`, `dynamic_universe_interval_cycles: 100`; `utils.universe_builder.select_top_liquid_usd_pairs` (CCXT) refreshes `trading_pairs` every N cycles. |

See `unified_config.yaml` section **OPTIONAL MODULES** and `docs/WHAT_ELSE_CAN_BE_DONE.md`.

---

## 4. Execution and venue (when you need them)

| What | Notes |
|------|--------|
| **Iceberg / dark pool** | Config placeholders exist (`iceberg_enabled`, `dark_pool_enabled`); enable when venue supports. |
| **Real TCA by venue** | Extend `scripts/tca_summary.py` or execution to report by venue; tune routing. |
| **Multi-venue routing** | Already have `multi_venue_enabled`; add smarter routing (e.g. by spread/liquidity). |

---

## 5. Process and discipline (no code)

| Action |
|--------|
| Paper 2–4 weeks before live; review win rate, drawdown, monthly return. |
| Go live with small capital; scale only after sustained results. |
| One config change at a time. |
| Run `python main.py validate` after pulls and config changes. |
| Complete [LIVE_CHECKLIST.md](LIVE_CHECKLIST.md) before switching to live. |
| When you get an alert, do something (pause, reduce size, or change config). |
| Run `scripts/weekly_profitability_check.py` weekly (cron in `scripts/cron_example.txt`). |

---

## 6. Infrastructure and ops

| Item | Action |
|------|--------|
| **10G path** | Run 10G performance test; fix MTU/latency if using HFT/micro-trading. |
| **Switch tuning** | Jumbo, QoS, cut-through (see SWITCH_IMPROVEMENTS_CHECKLIST.md if present). |
| **NTP** | On host and switches. |
| **Backups** | Cron daily: `scripts/backup_config_and_logs.sh`. |
| **Deploy** | `scripts/deploy_production_linux.sh`; systemd with correct WorkingDirectory. |
| **CI** | Run `python main.py validate` and optionally `scripts/health_check.py --paper-smoke` in GitHub Actions. |

---

## 7. Beyond (institutional / multi-asset)

When you want to go further:

- **Perps / funding** – Funding rate filter is on; add more perp-specific logic if needed.
- **Market making** – Strategy library has placeholder; add logic when desired.
- **Cross-asset** – Correlation matrix and multi-venue set the base; add more assets and risk aggregation.
- **External integrations** – See [EXTERNAL_SOURCES_INTEGRATION.md](EXTERNAL_SOURCES_INTEGRATION.md) (Hummingbot, Stock-Prediction-Models, HFT FPGA, etc.).

---

## Priority order (what to do next)

1. **Immediate:** ✅ Pre_trade_risk_block wired; `use_evolution_strategy_reward: true` by default.
2. **This week:** Enable 23-language gates you care about (many on by default); set alerts (Telegram/email); add weekly_profitability_check to cron (see cron_example.txt).
3. **Next:** Feed correlation_matrix if you have multi-symbol; enable external_alpha when you have a URL.
4. **When needed:** Optional modules (AI brain, HFT, quantum stubs, universe builder with real volume).
5. **Ongoing:** Process (alerts, weekly review, validate after changes).

See also: [IMPLEMENTABLE_NOW.md](IMPLEMENTABLE_NOW.md), [PRIORITY_ORDER.md](PRIORITY_ORDER.md), [RATING_AND_ROADMAP.md](RATING_AND_ROADMAP.md).
