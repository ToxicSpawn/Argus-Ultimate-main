# Improvement Roadmap – Bot, Infrastructure, and Operations

Actions that can improve results and reliability across the bot, hardware, and process. Do these in order that fits your setup; not all are required.

---

## 1. Config and strategy (software)

| Action | Where | Why |
|--------|--------|-----|
| **Raise min_signal_confidence for live** | `unified_config.yaml` → `ai_brain.min_signal_confidence` | Paper uses 0.55 so you get more trades; for live use **0.70–0.75** (or higher) so only higher-conviction signals trade. Reduces noise and bad entries. |
| **Keep strategy_whitelist tight** | `strategies.strategy_whitelist` | Already set to strategies with positive paper PnL. Review paper logs and remove any that consistently lose; add only after positive backtest/paper. |
| **Tighten slippage for live** | `execution_engine.max_slippage_pct` | Config has 0.5% in strict gate; ensure execution path uses it. Lower = fewer filled-at-worse-price trades. |
| **Regime filter** | `strategies.regime_filter_enabled` | Keep **true**. Only trend strategies in trend regime, mean-reversion in range. Reduces wrong-regime trades. |
| **Position size** | `capital.max_position_size_aud`, `max_position_pct` | Start conservative ($100 / 10% with $1k). Scale up only after sustained positive equity; never scale up after a drawdown. |
| **Partial take-profit** | `strategies.partial_tp_at_pct` | Already 3% in config. Lets you lock profit and let remainder run (6% full TP). |

---

## 2. Risk (software)

| Action | Where | Why |
|--------|--------|-----|
| **Volatility-adjusted limits** | `risk.use_volatility_adjusted_limits` | Keep **true**. Tightens size and daily loss when realized vol is high. |
| **Circuit breaker** | `risk.circuit_breaker_dd_pct` (8%) | Already set. Add **alerting** (e.g. Discord/Telegram or Grafana) when circuit breaker trips so you can intervene. |
| **Auto-reduce after losses** | `risk.auto_reduce_after_n_losses`, `auto_reduce_factor` | Already 3 losses → 0.6× size. Optionally tighten to 2 losses or 0.5× if you want faster de-risk. |
| **Max consecutive losses** | `risk.max_consecutive_losses: 2` | Already tight. Keeps you from doubling down after a bad streak. |

---

## 3. Execution and 10G (software + hardware)

| Action | Where | Why |
|--------|--------|-----|
| **Verify 10G path** | R730 + Solarflare + switches | Run `10gbe_performance_test.py` and/or `test_10gbe_adaptive.py` after cabling. Confirm MTU 9000 and &lt;1ms latency. |
| **Switch tuning** | Both X460-G2 | Follow [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md): jumbo, QoS priority 7, cut-through, edge ports. Fewer drops = better execution. |
| **Execution timeouts** | `execution_engine` / config | Ensure orders that don’t fill within N seconds are cancelled or revised. Prevents stale orders. |
| **Fee and slippage in sizing** | Execution / risk | Size positions after fees and expected slippage so net edge stays positive. |

---

## 4. ML and signals (software)

| Action | Where | Why |
|--------|--------|-----|
| **Retrain models periodically** | `ml/train_xgboost.py`, ensemble | Retrain XGBoost (and any other models) on recent data (e.g. monthly). Stale models hurt edge. |
| **Feature and model review** | `ml/feature_library_300.py`, `ml/model_ensemble.py` | Add or tune features that correlate with next-bar direction; drop weak ones. Improves signal quality. |
| **Online learning** | `ml/online_learning.py`, adaptive | If enabled, ensure weights or priors update from recent PnL so the system adapts to regime change. |

---

## 5. Infrastructure and ops

| Action | Where | Why |
|--------|--------|-----|
| **R730: 64GB RAM** | Hardware | 32GB is OK; 64GB gives headroom for many strategies and monitoring. |
| **NTP on R730 and switches** | OS + switch CLI | Same clock everywhere. Needed for logs, latency attribution, and PTP later if you add TM-CLK. |
| **Monitoring and alerts** | Grafana, Prometheus, Discord/Telegram | Alert on: circuit breaker, max daily loss hit, error rate &gt; threshold, bot down. Lets you react fast. |
| **Backups** | `scripts/backup_config_and_logs.sh` | Cron daily. Restore point for config and logs after a bad change or for analysis. |
| **Health check** | `scripts/health_check.py --paper-smoke` | Run weekly or before/after config changes. Catches regressions. |

---

## 6. Process (how you run the bot)

| Action | Why |
|--------|-----|
| **Paper 2–4 weeks before live** | See real win rate, drawdown, and monthly return in simulation. If paper is flat or negative, fix before live. |
| **Start live small** | Use $1k (or your chosen minimum). Prove the system with real money at low risk before scaling. |
| **Scale capital only after sustained results** | e.g. 2–3 months positive, drawdown under control. Then increase in steps (e.g. 1k → 2k → 5k). |
| **One change at a time** | When tuning (strategy list, confidence, risk), change one thing and observe. Easier to attribute cause. |
| **Walk-forward or periodic backtest** | Before going live or after big config change, run backtest on recent data. Sanity-check that edge still holds. |

---

## 7. Quick wins (do first)

1. **Live confidence:** Set `ai_brain.min_signal_confidence` to **0.72** (or 0.75) when you go live.
2. **Alerts:** Add one channel (Discord or Telegram) and trigger on circuit breaker and daily loss limit.
3. **10G check:** After R730 + switches are up, run the 10G test script and fix any MTU/latency issues.
4. **Backups:** Schedule `backup_config_and_logs.sh` daily on the R730.
5. **Paper first:** Run paper for at least 2 weeks and review logs (win rate, max drawdown, monthly return) before switching to live.

---

## 8. Reference

- **Config:** `unified_config.yaml`
- **Deployment:** [DEPLOYMENT_R730_AND_DESKTOP.md](DEPLOYMENT_R730_AND_DESKTOP.md), [R730_SETUP.md](R730_SETUP.md)
- **Switches:** [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md)
- **Runbook:** [UNIFIED_RUNBOOK.md](../UNIFIED_RUNBOOK.md)
- **Full checklist:** [IMPROVEMENT_EVERYTHING.md](IMPROVEMENT_EVERYTHING.md) – exhaustive list of every possible improvement
