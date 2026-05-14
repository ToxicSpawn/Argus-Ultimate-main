# Argus – What Can Be Used to Improve

Audit of the **entire Argus codebase**: entry points, optional modules, config, scripts, and docs. Use this to decide what to enable, tune, or add.

---

## 1. Architecture (how it runs)

| Component | Role |
|-----------|------|
| **main.py** | Single production entrypoint: `paper`, `live`, `backtest`, `validate`, `setup`, `quantum-status`. |
| **UnifiedSystemArchitecture** (unified_trading_system.py) | Loads unified_config.yaml, initializes exchange, strategies, execution, capital optimizer, monitoring, then runs the trading loop. |
| **Trading loop** | Each cycle: get signals (continuous scanner → else AI brain) → optional HFT → filter disabled strategies → process ARGUS strategies → capitalize → execute. |
| **ContinuousBestTradeScanner** | In parallel: AI brain, strategy engine (unified_engine), HFT. Ranks by score, applies strategy_whitelist, diversity, top N. |
| **Strategy engine** (strategies/unified/strategy_engine.py) | RSI/BB/MACD + regime + online tuner; emits signals tagged `unified_engine`. |

Paper/live both use this path; config and `run_mode` control behavior.

---

## 2. Optional modules (referenced but missing – enable to improve)

These are **imported** by the unified system but **not present** in the repo (or not on Python path). Adding or restoring them would unlock more features.

| Module | Where used | What it would add |
|--------|------------|--------------------|
| **unified_ai_brain** | unified_trading_system.py (`PinnacleAIBrain`) | Multi-agent AI signals; fallback when scanner has no cached opportunities. Without it, scanner still works (strategy_engine + whitelist). |
| **unified_language_orchestrator** | unified_trading_system.py (multi-lang cycle plan) | Optional 23+ language task routing; not required for PnL. |
| **quant_fund_upgrades.multi_factor_risk_engine** | unified_trading_system.py | Multi-factor risk (e.g. factor exposure, portfolio risk). Without it, unified_risk_manager still runs. |
| **hft_engine.hft_scalping_engine** | unified_trading_system.py, scanner | HFT order-book/tick signals. Currently disabled in paper via paper_trading_disabled_strategies; enable when module exists and PnL is positive. |
| **hft.advanced_realtime_hft_infrastructure** | unified_trading_system.py | Advanced HFT event loop (OB/trades → signal ring). Set `hft.use_advanced_hft_infrastructure: true` when available. |
| **utils.universe_builder** | unified_trading_system.py | Optional adaptive universe (symbol selection). |
| **quantum_unified_stubs** (quantum/) | unified_trading_system.py | Quantum Monte Carlo VaR/CVaR; optional circuit breaker. |

**Action:** Either add these modules under the expected names/paths, or leave as-is and rely on strategy_engine + whitelist + execution/risk (current pinnacle setup).

---

## 3. Config already present – tune for improvement

All in **unified_config.yaml**. See also IMPROVEMENT_EVERYTHING.md and IMPROVEMENT_ROADMAP.md.

| Area | Key knobs | Use to improve |
|------|-----------|-----------------|
| **Strategies** | `strategies.strategy_whitelist`, `regime_filter_enabled`, `regime_filter_trend_strategies` / `_mr_strategies` | Whitelist = positive-edge only (already pinnacle: akashic_tier, unified_engine, quantum_momentum_elite). Regime filter reduces wrong-regime trades. |
| **Paper** | `paper_trading.min_signal_confidence`, `paper_trading_disabled_strategies`, `partial_tp_at_pct`, `trailing_stop_enabled` | Raise confidence for fewer/better trades; keep losers disabled; partial TP + trail let winners run. |
| **Risk** | `risk.max_daily_loss_pct`, `max_drawdown_pct`, `stop_loss_pct`, `take_profit_pct`, `circuit_breaker_dd_pct`, `auto_reduce_after_n_losses` | Already tight; add alerting when circuit breaker or daily limit hits. |
| **Edge-cost gate** | `edge_cost_gate.min_edge_pct`, `buffer_mult`, `modes` (include `"live"` for live) | Only trade when edge > cost; enforced in execution engine. |
| **Execution** | `execution.slippage_pct`, `signal_cooldown_bars`; execution_engine `max_slippage_pct`, retry, VWAP threshold | Tighter slippage and cooldown reduce bad fills and overtrading. |
| **Continuous scan** | `continuous_scan.interval_seconds`, `top_n`, `use_liquidity_boost`, `diversity_max_per_symbol` / `_strategy`, `adaptive_interval_enabled` | Shorter interval = faster reaction; diversity avoids single-symbol/strategy overload. |
| **Evolution** | `evolution.load_evolved`, `continuous_enabled`, `realtime_interval_minutes`, `auto_apply`, `trigger_on_trade`, `after_n_trades` | Load evolved params from data/evolved_params.json; evolve on schedule or every N trades. |
| **Strategy allocator** | `strategy_allocator.enabled`, `modes`, `ema_alpha`, `exploration_c` | Favors strategies with better PnL; already used in paper loop. |
| **AI brain** | `ai_brain.min_signal_confidence`, `num_ai_agents`, `agents.*`, `quantum_optimization` | When unified_ai_brain is present: higher confidence = fewer, better signals. |
| **Quantum bot** | `quantum_bot.use_quantum_monte_carlo_risk`, `use_quantum_var_circuit_breaker`, `use_quantum_walk`, `use_multi_timeframe_walk` | Optional VaR/CVaR and confidence boost when stubs/deps available. |
| **HFT** | `hft.enabled`, `max_hft_signals_per_cycle`, `disabled_in_regimes`, `use_advanced_hft_infrastructure` | Enable when HFT module exists and paper PnL is positive; disable in bad regimes. |
| **Monitoring** | `monitoring.alerts.telegram` / `email`, `triggers` | Alerts on drawdown, daily loss, consecutive losses, errors – critical for live. |

---

## 4. Scripts and docs to use

| Item | Path | Use to improve |
|------|------|----------------|
| **Health check** | scripts/health_check.py | Run `--paper-smoke` (or full) before/after config changes; catch regressions. |
| **Pre-live check** | scripts/pre_live_check.py | Validates config (credentials, edge_cost_gate modes, etc.) before going live. |
| **Backup** | scripts/backup_config_and_logs.sh | Cron daily; restore point for config and logs. |
| **Deploy** | scripts/deploy_production_linux.sh | Deploy to production host. |
| **Improvement roadmap** | docs/IMPROVEMENT_ROADMAP.md | Prioritized actions: config, risk, execution, ML, infra, process. |
| **Improvement everything** | docs/IMPROVEMENT_EVERYTHING.md | Exhaustive checklist (capital, risk, execution, strategies, evolution, HFT, 10G, monitoring, etc.). |
| **Live checklist** | docs/LIVE_CHECKLIST.md | Before setting runtime.mode to live. |
| **Switch checklist** | docs/SWITCH_IMPROVEMENTS_CHECKLIST.md | 10G switches (X460-G2): jumbo, QoS, cut-through. |
| **Deployment** | docs/DEPLOYMENT_R730_AND_DESKTOP.md, R730_SETUP.md | Server and topology setup. |

---

## 5. Code paths already in use (no extra setup)

| Feature | Where | Status |
|---------|--------|--------|
| Strategy whitelist | continuous_best_trade_scanner.py (filter by strategy_whitelist) | Active when whitelist non-empty. |
| Disabled strategies (paper/live) | unified_trading_system.py (filter ai_signals by paper_trading_disabled_strategies / live_disabled_strategies) | Active. |
| Regime detection | adaptive/regime.py (RegimeDetector); strategy_engine uses it | Active. |
| Online tuner | adaptive/online_tuner.py; strategy_engine uses it | Active. |
| Strategy allocator | Allocator tracks PnL per strategy; paper loop records by source_strategy | Active; persist_path in config. |
| Edge-cost gate | unified_execution_engine.py (rejects low-edge signals when enabled) | Active; ensure modes include "live" for live. |
| Implementation shortfall | execution/implementation_shortfall.py; execution engine records it | Active for analysis. |
| Quantum Monte Carlo risk hook | unified_trading_system.py (_quantum_monte_carlo_risk_hook) | Active when quantum_unified_stubs available and use_quantum_monte_carlo_risk true. |
| Evolved params | evolution loads data/evolved_params.json when evolution.load_evolved true | Enable in config and run evolution to populate file. |

---

## 6. Quick wins (do first)

1. **Alerts** – Enable Telegram (or email) in `monitoring.alerts` and set triggers (drawdown, daily loss, consecutive losses, error rate). Add circuit breaker to triggers if desired.
2. **Live confidence** – When going live, ensure `ai_brain.min_signal_confidence` is 0.72+ (paper overrides can stay higher, e.g. 0.82).
3. **Edge-cost gate for live** – In `edge_cost_gate.modes` include `"live"` so the gate applies in production.
4. **Pre-live check** – Run `py scripts/pre_live_check.py` (or `python scripts/pre_live_check.py`) before switching to live.
5. **Health check** – Run `py scripts/health_check.py --paper-smoke` periodically.
6. **Evolution** – Set `evolution.load_evolved: true` and run evolution (e.g. run_evolution or paper loop with self-improvement) to generate data/evolved_params.json, then restart so strategies use evolved params.
7. **10G** – When R730 + switches are up, run 10gbe_performance_test.py and fix MTU/latency; then consider enabling HFT and micro-trading if desired.

---

## 7. Summary

- **Already used for improvement:** Strategy whitelist (pinnacle), disabled-strategies filter, regime filter, edge-cost gate, execution/risk/config from IMPROVEMENT_*.md, health/pre_live/backup scripts.
- **Optional modules:** unified_ai_brain, HFT engine, quant_fund risk engine, language orchestrator, quantum stubs – add or restore to unlock those features.
- **Config:** One place to tune is unified_config.yaml; IMPROVEMENT_EVERYTHING.md and IMPROVEMENT_ROADMAP.md list every knob.
- **Process:** Paper 2–4 weeks, then live small; one change at a time; alerts and backups on.

For a short ordered list, use **docs/IMPROVEMENT_ROADMAP.md**. For the full checklist, use **docs/IMPROVEMENT_EVERYTHING.md**. This doc ties them to the actual codebase and optional modules.
