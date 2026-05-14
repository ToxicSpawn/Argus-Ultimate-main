# Priority Order (What to Do First)

This is the actionable version of **Section 14** in [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md). Do these in order.

---

## 1. Config + process

**Goal:** Alerts on; live confidence 0.78+; edge gate for live; run pre_live_check and health_check; paper 2–4 weeks; kill_losers_review weekly.

### 1.1 Alerts (Telegram/email)

- [x] In `unified_config.yaml`: `monitoring.alerts.enabled: true` **(default)**.
- [ ] Set in `.env` (not committed):
  - **Telegram:** `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
  - **Email (optional):** SMTP vars, `EMAIL_FROM`, `EMAIL_TO`.
- [x] In config: `monitoring.alerts.telegram.enabled: true` (or email) **(default)**.
- [x] Triggers include: `drawdown`, `daily_loss`, `consecutive_losses`, `error_rate`, `circuit_breaker` **(default)**.
- [ ] Send one test alert to confirm delivery.

### 1.2 Live confidence and edge gate

- [x] `ai_brain.live_min_signal_confidence: 0.78` (or 0.78–0.82) **(default 0.80)**.
- [x] `edge_cost_gate.modes` includes `"live"` **(default)**.
- [x] `edge_cost_gate.live_min_edge_pct: 1.0` and `live_buffer_mult: 2.2` **(default)**.

### 1.3 Pre-live and health checks

- [ ] Before going live:  
  `python scripts/pre_live_check.py --config unified_config.yaml`  
  Exit 0.
- [ ] After config changes or weekly:  
  `python scripts/health_check.py --config unified_config.yaml --paper-smoke`  
  Exit 0.

### 1.4 Paper and review discipline

- [ ] Run **paper for 2–4 weeks** before switching to live.
- [ ] **Weekly:** run  
  `python scripts/kill_losers_review.py --config unified_config.yaml`  
  and remove suggested strategies from whitelist if you agree.

### 1.5 Validate priority-1 settings

- [ ] Run:  
  `python scripts/validate_priority_order.py --config unified_config.yaml`  
  Fix any reported gaps before live.

---

## 2. Already in code (use it)

**Goal:** Ensure these are enabled and used: IS gate, pre-trade gate, latency budget, regime boost, evolution load, allocator, walk-forward and stress_test scripts.

### 2.1 Config to turn on

- [x] **IS gate:** `execution_engine.use_is_gate: true`, `max_avg_is_bps: 20` **(default)**.
- [x] **Pre-trade exposure/position gate:** `capital.max_total_exposure_pct`, `max_position_size_aud`, `max_position_pct` set in config **(default)**.
- [ ] **Regime boost:** `strategies.use_regime_lstm_boost: true` (optional but recommended).
- [x] **Evolution load:** `evolution.load_evolved: true`, `evolution.params_path: "data/evolved_params.json"` **(default)**.
- [x] **Allocator:** `strategy_allocator.enabled: true` **(default)**.

### 2.2 Scripts to run

- [ ] **Walk-forward:**  
  `python scripts/walk_forward_gate.py --csv data/ohlcv.csv [--train-days 21] [--test-days 7]`
- [ ] **Stress test risk:**  
  `python scripts/stress_test_risk.py`

Latency budget is logged each cycle automatically when the main loop runs; no extra config.

---

## 3. Next code (already implemented)

**Goal:** evolution_unified; LSTM/evolution-strategy in ml/; emergency shutdown; 23-lang per-task timeouts.

These are already in the codebase. Default config uses them where relevant:

- [ ] **Evolution:** `evolution.continuous_enabled`, `trigger_on_trade`, `after_n_trades` if you want trade-triggered evolution; `evolution_allow_apply_live: false` unless you explicitly want apply in live.
- [x] **Emergency shutdown:** `risk.emergency_shutdown.enabled: true` with `latency_spike_ms`, `flash_crash_pct`, `network_fail`, `arb_spread_bps` **(default)**.
- [x] **23-lang timeouts:** `multi_language.task_timeouts` in config for all task types **(default)**.
- [x] **23-lang risk gate:** `multi_language.use_conservative_risk: true` **(default)** so correctness languages gate execution.
- [ ] **LSTM/evolution-strategy:** `strategies.use_regime_lstm_boost: true`; optional `evolution.use_evolution_strategy_reward: true`.

---

## 4. Optional modules

**Goal:** AI brain, HFT engine, real 23-lang services, universe builder with volume.

- [ ] **Universe builder with volume:** Implemented in `utils.universe_builder` (CCXT volume fetch). Enable `continuous_scan.dynamic_universe_enabled` and set `dynamic_universe_interval_cycles`, `dynamic_universe_top_n`.
- [ ] **AI brain (PinnacleAIBrain):** Use when `unified_ai_brain` is available; scanner falls back to it when cache is stale.
- [ ] **HFT engine:** Enable only when paper PnL is positive; set `hft.enabled: true` and ensure 10G/infra is ready.
- [ ] **23-lang HTTP services:** Deploy per [MULTI_LANGUAGE_23_README.md](MULTI_LANGUAGE_23_README.md); set `multi_language.endpoints` in config.

---

## 5. Infra

**Goal:** 10G test, switch checklist, NTP, backups cron, deploy.

- [ ] **Backups:** Cron daily, e.g.  
  `0 2 * * * /path/to/scripts/backup_config_and_logs.sh /path/to/repo /path/to/backups`
- [ ] **Health check:** Cron weekly, e.g.  
  `0 3 * * 0 cd /path/to/repo && python scripts/health_check.py --config unified_config.yaml --paper-smoke >> logs/health_check.log 2>&1`
- [ ] **Kill losers review:** Cron weekly, e.g.  
  `0 4 * * 0 cd /path/to/repo && python scripts/kill_losers_review.py >> logs/kill_losers.log 2>&1`
- [ ] **10G:** Run `10gbe_performance_test.py` or `test_10gbe_adaptive.py` when R730 + Solarflare + switches are up.
- [ ] **Switches:** Follow [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md) (jumbo, QoS, cut-through).
- [ ] **NTP:** Same clock on R730 and switches for logs and latency.
- [ ] **Deploy:** Use `scripts/deploy_production_linux.sh` or your playbook; run `main.py validate` after deploy.

---

## 6. Beyond (institutional / multi-asset)

**Goal:** Perps/arb, market making, funding/options data, cross-asset when you want to go further.

- [ ] **Funding rates:** Implemented (optional): `strategies.use_funding_rate_filter: true`, `funding_rate_skip_long_threshold`, `funding_rates_url` if you have a feed.
- [ ] **Perps/arb, market making, options, cross-asset:** See [EXTERNAL_SOURCES_INTEGRATION.md](EXTERNAL_SOURCES_INTEGRATION.md) and [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md) §12; implement when moving to institutional/multi-asset.

---

## Quick reference

| Priority | Focus |
|----------|--------|
| 1 | Alerts, live confidence 0.78+, edge gate, pre_live_check, health_check, paper 2–4 weeks, kill_losers weekly |
| 2 | IS gate, pre-trade gate, regime boost, evolution load, allocator, walk_forward, stress_test_risk |
| 3 | Emergency shutdown, 23-lang timeouts, evolution_unified (already in code) |
| 4 | Dynamic universe, AI brain, HFT, 23-lang HTTP services (optional) |
| 5 | Backups cron, health cron, kill_losers cron, 10G, switches, NTP, deploy |
| 6 | Funding filter (done); perps/arb, market making, options (when needed) |

See also: [LIVE_CHECKLIST.md](LIVE_CHECKLIST.md), [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md) §14.
