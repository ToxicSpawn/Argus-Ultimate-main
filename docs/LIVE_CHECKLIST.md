# Live Trading Checklist

Complete every item before switching `runtime.mode` to `live` or running `main.py live`.

**See also:** [PRIORITY_ORDER.md](PRIORITY_ORDER.md) for the full priority order (Section 14); run `python scripts/validate_priority_order.py --config unified_config.yaml` to validate Priority 1 settings.

---

## 1. Credentials and environment

- [ ] `.env` exists and is not committed (in `.gitignore`).
- [ ] `KRAKEN_API_KEY` and `KRAKEN_SECRET_KEY` set (for live).
- [ ] If using secondary venue: `COINBASE_ADVANCED_API_KEY` and `COINBASE_ADVANCED_API_SECRET` set.
- [ ] Exchange API keys are **restricted**: trading + read only; **no withdraw** permission.
- [ ] `unified_config.yaml` (or production override) has correct exchange fee tiers for your account.

---

## 2. Alerts (required for live)

- [ ] `monitoring.alerts.enabled` is `true`.
- [ ] At least one channel is on and configured:
  - [ ] **Telegram:** `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`; `monitoring.alerts.telegram.enabled: true`.
  - [ ] **Email:** SMTP and `EMAIL_FROM` / `EMAIL_TO` set; `monitoring.alerts.email.enabled: true`.
- [ ] Alert triggers match risk limits (e.g. drawdown 12%, daily loss 2%, consecutive losses 2, error rate 5%).
- [ ] Test alert once (e.g. trigger a test message) to confirm delivery.

---

## 3. Config (production-safe)

- [ ] `ai_brain.live_min_signal_confidence` is **0.78 or higher** for live (Priority 1; e.g. 0.78–0.82).
- [ ] `ai_brain.min_signal_confidence` is 0.70+ for paper; use live_min_signal_confidence for live.
- [ ] `execution_engine.max_slippage_pct` is tight (e.g. 0.005).
- [ ] `edge_cost_gate.modes` includes `"live"` and live edge params are set (e.g. `live_min_edge_pct: 1.0`, `live_buffer_mult: 2.2`).
- [ ] `strategies.strategy_whitelist` only includes strategies with positive paper/backtest PnL.
- [ ] `runtime.live_disabled_strategies` lists all strategies you do not want in live (e.g. farmer, high_freq_grid, apeiron_tier, ultimate).
- [ ] `risk.max_daily_loss_pct`, `max_drawdown_pct`, `circuit_breaker_dd_pct`, `auto_reduce_*` are set and acceptable.
- [ ] `capital.starting_capital_aud` equals the capital you are actually deploying.

---

## 4. Paper and validation

- [ ] `python main.py validate` passes.
- [ ] Paper trading has been run for **at least 2 weeks** (or your chosen minimum).
- [ ] Paper results reviewed: win rate, max drawdown, monthly return are acceptable.
- [ ] No recent config change without a follow-up paper or backtest check.

---

## 5. Infrastructure (if using R730 + 10G)

- [ ] R730: OS installed; Solarflare drivers and OpenOnload installed; NTP configured.
- [ ] 10G: Cables connected (Solarflare → switch 1 & 2); MTU 9000; `10gbe_performance_test.py` or `test_10gbe_adaptive.py` run and OK.
- [ ] Switches: MLAG, jumbo, QoS per [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md).
- [ ] Backups: `scripts/backup_config_and_logs.sh` scheduled (e.g. daily cron).

---

## 6. Run pre-live and priority-order validation

Run the combined script (runs validate, validate_priority_order, pre_live_check and prints reminders):

```bash
bash scripts/run_before_live.sh unified_config.yaml
```

Or run individually:

```bash
python main.py validate
python scripts/validate_priority_order.py --config unified_config.yaml
python scripts/pre_live_check.py --config unified_config.yaml
```

- [ ] All pass (Priority 1 checks passed).

---

## 7. Go live

- [ ] Start with **small capital** (e.g. $1k AUD as in config).
- [ ] Use production config: `python main.py live --config unified_config.production.yaml` (or your merged config).
- [ ] Monitor first hours: logs, alerts, first fills.
- [ ] Scale capital only after **sustained** positive results (e.g. 2–3 months).

---

## Reference

- [PRIORITY_ORDER.md](PRIORITY_ORDER.md) – **Section 14 full walkthrough** (config + process, already in code, infra, beyond)
- [IMPROVEMENT_EVERYTHING.md](IMPROVEMENT_EVERYTHING.md) – full improvement list
- [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) – prioritized actions
- [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md) – master list §14
- [R730_SETUP.md](R730_SETUP.md) – R730 deployment
- [UNIFIED_RUNBOOK.md](../UNIFIED_RUNBOOK.md) – canonical commands
