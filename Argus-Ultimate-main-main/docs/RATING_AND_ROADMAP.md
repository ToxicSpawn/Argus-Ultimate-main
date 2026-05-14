# Argus-Ultimate Rating and 10/10 Roadmap

## Target: 10/10

This doc summarizes what “10/10” means for Argus and what was implemented to get there.

---

## Rating criteria (what was addressed)

| Gap | What we did |
|-----|-------------|
| **Backtest vs live gap** | `backtest.fill_probability` (e.g. 0.98 = 2% of fills rejected for realism). `scripts/live_vs_backtest_consistency.py` compares realized slippage to backtest assumptions. |
| **Quantum = simulated** | `quantum_features.quantum_simulated_disclosure: true` and this doc state clearly: quantum is simulated/quantum-inspired unless real hardware is connected. |
| **No proof of edge** | `runtime.live_require_paper_edge` + `live_min_trades_paper` / `live_min_win_rate_pct`. Pre-live check enforces paper evidence before allowing live. |
| **Optional deps** | `risk.real_time_risk_api`, `portfolio.weight_provider`, `utils.gpu_inference` wrapped in try/except with fallbacks so the bot runs without them. |
| **Institutional polish** | `scripts/readiness_score.py` (0–100). `scripts/tca_summary.py` (avg slippage/IS by strategy). `execution_engine.iceberg_enabled` / `dark_pool_enabled` placeholders. |

---

## 10/10 checklist (operational)

1. **Config:** Alerts on, live_min_signal_confidence ≥ 0.78, edge gate for live, risk limits set.
2. **Process:** Pre-live check and `validate_priority_order` pass; paper 2–4 weeks; kill_losers weekly.
3. **Edge gate:** Set `runtime.live_require_paper_edge: true` and run paper until min_trades and min_win_rate are met; then pre_live_check allows live.
4. **Backtest realism:** Set `backtest.fill_probability: 0.98` (or 0.99) so backtest doesn’t overstate fills.
5. **Transparency:** Keep `quantum_simulated_disclosure: true`; document that quantum is simulated unless a real backend is configured.
6. **Monitoring:** Run `readiness_score.py --include-paper`; run `tca_summary.py` after live trades; run `live_vs_backtest_consistency.py` periodically.

---

## Scripts added for 10/10

| Script | Purpose |
|--------|---------|
| `scripts/validate_priority_order.py` | Priority 1 config (alerts, confidence, edge gate). |
| `scripts/readiness_score.py` | 0–100 readiness score (config + optional paper). |
| `scripts/pre_live_check.py` | Full pre-live checklist; includes edge gate when `live_require_paper_edge` is true. |
| `scripts/live_vs_backtest_consistency.py` | Compare live slippage to backtest assumptions. |
| `scripts/tca_summary.py` | TCA: avg slippage bps by symbol/strategy. |
| `scripts/rollback_evolved_params.py` | Rollback evolution to previous version. |
| `scripts/export_performance_series.py` | Export performance time series. |
| `scripts/kill_losers_review.py` | Suggest strategies to remove from whitelist. |
| `scripts/run_before_live.sh` | Run before live: validate, validate_priority_order, pre_live_check + reminders. |
| `scripts/walk_forward_gate.py` | Walk-forward backtest (train then test); writes `data/walk_forward_result.json`. |
| `scripts/tca_summary.py` | TCA: avg slippage bps by symbol, strategy, **and venue** (`by_venue` from ledger). |

---

## Implemented in code (do-it-all)

- **Pre-trade risk block:** Execution engine calls `execution.risk_compliance_audit.pre_trade_risk_block()` before every order; main loop sets `_pre_trade_positions`, `_pre_trade_prices`, `_pre_trade_equity_aud` on config. Rejected → simulated fill with `error=pre_trade_risk_block:reason`.
- **Evolution-strategy reward:** `evolution.use_evolution_strategy_reward: true` by default; execution records PnL per strategy/symbol for param jitter.
- **Run before live:** `scripts/run_before_live.sh` runs validate, validate_priority_order, pre_live_check, and readiness_score --include-paper.

---

## Tilting toward profitability (config + code)

These settings and behaviors are tuned to *improve the odds* of profitability (fewer low-edge trades, more capital on best ideas). They do not guarantee profit.

| Lever | What we did |
|-------|-------------|
| **Entry quality** | `ai_brain.min_signal_confidence` 0.76 (paper), `live_min_signal_confidence` 0.80; only high-conviction signals trade. |
| **Edge gate** | `edge_cost_gate.min_edge_pct` 0.85%, `buffer_mult` 2.0; trade only when expected edge clearly exceeds costs. |
| **Strategy allocator** | `exploration_c` 0.75 (favor proven winners), `max_total_signals` 4; concentrate on fewer, better signals. |
| **Evolution** | `use_composite_fitness: true`, `negative_return_penalty_weight: 0.3`, `composite_calmar_weight: 0.2`, `min_trades: 3`; GA optimizes for risk-adjusted return and penalizes losing param sets. |
| **Regime-aligned boost** | Strategy engine gives regime-aligned signals +8% confidence so they rank higher when competing with others. |
| **Cooldown** | `execution.signal_cooldown_bars: 4` (and paper_trading) to reduce overtrading on the same symbol. |

Run paper 2–4 weeks, then `scripts/kill_losers_review.py` and trim `strategy_whitelist` to strategies with positive PnL EMA. Keep `evolution.load_evolved: true` and `auto_apply: true` so the bot keeps adapting params to recent market.

---

## What 10/10 does not guarantee

- **Profitability:** 10/10 = safety, transparency, and process. Profit depends on edge, execution, and market; the above tilts the system toward higher-quality trades and evolution toward profitable params.
- **Real quantum:** Simulated quantum remains classical unless you connect a real quantum backend.
- **Institutional execution:** Iceberg/dark pool are placeholders; full TCA and venue-specific order types depend on your venue.

---

## Infrastructure

- **Cron:** See `scripts/cron_example.txt` (backup daily, health + weekly profitability, optional validate / run_before_live).
- **Deploy:** `scripts/deploy_production_linux.sh` (systemd, WorkingDirectory). Run from repo root.
- **23-language HTTP:** Optional Docker services per language; see [MULTILANG_HTTP_SERVICES.md](MULTILANG_HTTP_SERVICES.md) and `scripts/docker-compose-multilang.example.yml`.

---

## References

- [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md) – master list
- [BEYOND_PLACEHOLDERS.md](BEYOND_PLACEHOLDERS.md) – spot–perp arb, market making, cross-asset placeholders
- [PRIORITY_ORDER.md](PRIORITY_ORDER.md) – what to do first
- [LIVE_CHECKLIST.md](LIVE_CHECKLIST.md) – before going live
