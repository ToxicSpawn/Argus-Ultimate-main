# Everything That Can Be Done to Improve the System

Exhaustive list of improvements across **config**, **risk**, **execution**, **strategies**, **AI/ML**, **data**, **evolution**, **HFT**, **10GbE**, **hardware**, **monitoring**, **ops**, **security**, and **process**. Use this as a master checklist; implement in an order that fits your priorities.

---

## 1. Capital and position sizing

| Item | Config path | Action |
|------|-------------|--------|
| Starting capital | `capital.starting_capital_aud` | Set to what you will actually deploy; don’t overstate. |
| Min position | `capital.min_position_size_aud` | High enough to avoid fee drag (e.g. 15–20 AUD); not so high that you skip small edges. |
| Max position | `capital.max_position_size_aud`, `max_position_pct` | Start conservative (e.g. 10% of capital); increase only after sustained positive results. |
| Max total exposure | `capital.max_total_exposure_pct` | Limit total capital at risk (e.g. 40%); reduce in volatile regimes. |
| Max concurrent positions | `capital.max_concurrent_positions` | Cap open positions (e.g. 4); avoids over-diversification and keeps focus. |
| Micro-positions | `capital.enable_micro_positions`, `flash_position_max_aud` | Enable only if 10G path is verified and you want sub-$10 trades. |

---

## 2. Risk management

| Item | Config path | Action |
|------|-------------|--------|
| Max daily loss | `risk.max_daily_loss_pct` | Keep tight (e.g. 2%); add alert when hit. |
| Max drawdown | `risk.max_drawdown_pct` | e.g. 12%; align with circuit breaker. |
| Stop loss | `risk.stop_loss_pct` | 1.5% avoids noise; tighten in high vol if needed. |
| Take profit | `risk.take_profit_pct` | 6% gives ~4:1 R:R; adjust if backtest suggests different. |
| Max consecutive losses | `risk.max_consecutive_losses` | 2 is strict; keeps you from doubling down. |
| Max error rate | `risk.max_error_rate` | 5% shutdown; investigate when triggered. |
| Latency-based stops | `risk.enable_latency_based_stops` | Keep on with 10G. |
| Volatility-adjusted limits | `risk.use_volatility_adjusted_limits` | Keep on; ensure `realized_vol_pct` is fed from equity. |
| Circuit breaker | `risk.circuit_breaker_dd_pct` (8%) | Add alert (Telegram/Discord/Grafana) when it trips. |
| VaR breach | `risk.var_breach_pct` | Optional: feed VaR from monitoring and trip on breach. |
| Auto-reduce | `risk.auto_reduce_after_n_losses`, `auto_reduce_factor` | e.g. 3 losses → 0.6× size; optionally 2 → 0.5×. |

---

## 3. Exchanges and fees

| Item | Config path | Action |
|------|-------------|--------|
| Kraken fees | `exchanges.kraken.maker_fee`, `taker_fee` | Match your actual tier (e.g. 0.16% / 0.26%); wrong fees distort edge. |
| Coinbase fees | `exchanges.coinbase_advanced.maker_fee`, `taker_fee` | Match your tier (e.g. 0.6% / 0.8%). |
| Min order size | `exchanges.*.min_order_size` | Respect exchange minimums to avoid rejects. |
| Primary/secondary | `exchanges.primary`, `secondary` | Use Kraken + Coinbase Advanced as configured; ensure both credentials in `.env` if multi-venue. |

---

## 4. Trading pairs and symbols

| Item | Config path | Action |
|------|-------------|--------|
| Best symbols | `best_symbols` | Focus on liquid pairs (BTC/USD, ETH/USD, etc.); add/remove based on spread and volume. |
| Trading pairs | `trading_pairs` | Keep aligned with `best_symbols`; drop illiquid pairs to reduce slippage. |
| Per-pair limits | (code/allocator) | Optionally cap exposure or frequency per symbol. |

---

## 5. Execution engine

| Item | Config path | Action |
|------|-------------|--------|
| Order type | `execution_engine.order_type` | "market" for speed; consider "limit" or "twap"/"vwap" for larger size. |
| Retry | `execution_engine.retry_attempts`, `retry_delay_seconds` | Balance persistence vs stale orders; add timeout/cancel on no fill. |
| Max slippage | `execution_engine.max_slippage_pct` | Tighten for live (e.g. 0.5%); ensure execution path enforces it. |
| VWAP threshold | `execution_engine.vwap_large_order_threshold_aud` | Use VWAP slicing above this (e.g. $80); reduce market impact. |
| Portfolio weight | `execution_engine.portfolio_weight_method` | "hrp" / "bl" / "mpt" for risk-parity style sizing when you have correlation/vol. |
| Multi-venue | `execution_engine.multi_venue_enabled`, `multi_venue_min_notional_aud` | Split large orders across Kraken + Coinbase above threshold. |
| Correlation-aware sizing | `execution_engine.use_correlation_aware_sizing`, `correlation_matrix` | Enable when you feed correlation (e.g. BTC/ETH); reduces correlated risk. |
| Trade ledger | `execution_engine.trade_ledger` | Keep on; set retention (e.g. 365 days) for audit and analysis. |

---

## 6. Commission and slippage (global)

| Item | Config path | Action |
|------|-------------|--------|
| Slippage assumption | `execution.slippage_pct` | Match reality (e.g. 0.15%); used in edge/cost math. |
| Commission reserve | `execution.commission_reserve_pct` | Reserve % for fees (e.g. 5%) so net edge is after costs. |
| Signal cooldown | `execution.signal_cooldown_bars` | Min bars between trades (e.g. 4); reduces overtrading. |

---

## 7. Continuous scan

| Item | Config path | Action |
|------|-------------|--------|
| Interval | `continuous_scan.interval_seconds`, `min_interval_seconds`, `max_interval_seconds` | Shorter when you want faster reaction; adaptive_interval_enabled balances load. |
| Top N | `continuous_scan.top_n` | Number of best opportunities cached (e.g. 5). |
| Liquidity boost | `continuous_scan.use_liquidity_boost`, `liquidity_spread_pct_cap` | Prefer tighter spreads; tune cap. |
| Diversity | `continuous_scan.diversity_max_per_symbol`, `diversity_max_per_strategy` | Avoid all signals from one pair or one strategy. |
| Parallel sources | `continuous_scan.parallel_sources` | Keep true for AI + strategy engine + HFT in parallel. |

---

## 8. Edge-cost gate

| Item | Config path | Action |
|------|-------------|--------|
| Min edge | `edge_cost_gate.min_edge_pct` | Only trade when expected edge &gt; this (e.g. 0.75%). |
| Buffer mult | `edge_cost_gate.buffer_mult` | Edge must exceed cost by this multiple (e.g. 1.9×). |
| Fee/slippage mult | `edge_cost_gate.fee_mult`, `slippage_mult` | Conservative multiples so net edge is positive after costs. |
| Modes | `edge_cost_gate.modes` | Keep ["paper", "backtest"]; add "live" when going live. |

---

## 9. AI brain

| Item | Config path | Action |
|------|-------------|--------|
| Min signal confidence | `ai_brain.min_signal_confidence` | **Paper:** 0.55 for more trades. **Live:** raise to 0.70–0.75+ to reduce bad entries. |
| Max concurrent signals | `ai_brain.max_concurrent_signals` | e.g. 2; concentrate on best opportunities. |
| Num agents | `ai_brain.num_ai_agents` | More = more diversity; balance with CPU. |
| Agent types | `ai_brain.agents.*` | Enable market_analyst, risk_manager, execution_specialist, sentiment, arbitrage, regime, momentum, mean_reversion as needed. |
| Consciousness weights | `ai_brain.consciousness.*` | Tune sentiment/fear_greed/pattern/intuition weights from backtest. |
| Quantum optimization | `ai_brain.quantum_optimization` | Keep on if you use QAOA; method "quantum_approximate", risk_parity. |
| Arbitrage/flash/latency | `ai_brain.enable_arbitrage_boost`, `enable_flash_signals`, `latency_optimization` | On when 10G is verified. |

---

## 10. Strategies

| Item | Config path | Action |
|------|-------------|--------|
| Strategy whitelist | `strategies.strategy_whitelist` | **Critical.** Only strategies with positive paper/backtest PnL; review logs and trim losers. |
| Enabled list | `strategies.enabled` | Align with whitelist; add only after positive evidence. |
| Regime filter | `strategies.regime_filter_enabled`, `regime_filter_trend_strategies`, `regime_filter_mr_strategies` | Keep on; trend strategies in trend regime, MR in range. |
| Max extra signals | `strategies.max_extra_signals` | e.g. 2. |
| Strategy library | `strategy_library.enabled`, `enabled_strategies`, `modes` | Enable for paper/backtest; restrict for live via live_disabled_strategies. |
| Live disabled | `runtime.live_disabled_strategies` | Disable farmer, high_freq_grid, apeiron_tier, ultimate (and any other proven losers) in live. |

---

## 11. Paper trading overrides

| Item | Config path | Action |
|------|-------------|--------|
| Min confidence (paper) | `paper_trading.min_signal_confidence` | 0.82 is strict; 0.45 allows more strategies for testing. |
| Partial TP | `paper_trading.partial_tp_at_pct`, `partial_tp_close_pct` | 3% partial, close 50%; locks profit. |
| Trailing stop | `paper_trading.trailing_stop_enabled`, `trailing_stop_pct` | 1% trail below high-water mark. |
| Disabled strategies | `paper_trading.paper_trading_disabled_strategies` | Disable known losers in paper too. |
| Quality threshold | `paper_trading.paper_trading_quality_threshold` | Filter low-quality strategies. |
| Continuous scan | `paper_trading.continuous_scan_enabled`, `use_cached_best` | Keep on. |
| Adaptive universe | `paper_trading.adaptive_universe_enabled`, `adaptive_universe_top_n` | Top N symbols by opportunity. |

---

## 12. Runtime and live checklist

| Item | Config path | Action |
|------|-------------|--------|
| Mode | `runtime.mode` | Set "live" only after: credentials in `.env`, alerts on, optional vol/correlation feeds. |
| Institutional mode | `runtime.institutional_mode` | Keep true for compliance/audit. |
| Live disabled strategies | `runtime.live_disabled_strategies` | Same losers as paper; no HFT in live until proven if desired. |

---

## 13. Backtest realism

| Item | Config path | Action |
|------|-------------|--------|
| Slippage bps | `backtest.slippage_bps` | Realistic (e.g. 8 bps) so backtest doesn’t overstate edge. |
| Latency ms | `backtest.latency_ms` | Simulate delay between signal and fill. |
| OOS ratio | `backtest.oos_train_ratio` | e.g. 0.8 train / 0.2 OOS; report OOS metrics. |
| Spread bps | `backtest.spread_bps` | Bid-ask (e.g. 5 bps). |
| Max slippage reject | `backtest.max_slippage_pct` | Reject trade if implied slippage exceeds (e.g. 0.5%). |
| Market impact | `backtest.market_impact_bps_per_10k_usd` | Size-based drag. |
| Fee maker ratio | `backtest.fee_maker_ratio` | Blended maker/taker (e.g. 20% maker). |
| Rate limit reject | `backtest.rate_limit_reject_pct` | Simulate occasional rejects. |

---

## 14. Evolution

| Item | Config path | Action |
|------|-------------|--------|
| Load evolved | `evolution.load_evolved` | Set true to load `data/evolved_params.json` at startup. |
| Continuous | `evolution.continuous_enabled`, `interval_hours` | Run GA on a schedule (e.g. 24h). |
| Realtime | `evolution.realtime_interval_minutes`, `realtime_fitness_days`, `use_live_feed` | Shorter interval + recent data for faster adaptation. |
| Generations / population | `evolution.generations`, `realtime_generations`, `population_size` | More = better search but slower. |
| Auto-apply | `evolution.auto_apply` | true = apply best params after each run. |
| Trigger on trade | `evolution.trigger_on_trade`, `after_n_trades` | Evolve every N closed trades. |
| Strategy allocator | `strategy_allocator.enabled`, `modes`, `min_trades_before_bias`, `exploration_c`, `ema_alpha` | Favors strategies with better PnL; tune exploration vs exploit. |

---

## 15. HFT

| Item | Config path | Action |
|------|-------------|--------|
| Enabled | `hft.enabled` | On when 10G is verified and you want order-book signals. |
| Max signals per cycle | `hft.max_hft_signals_per_cycle` | Cap (e.g. 2) to avoid flooding. |
| Min imbalance ratio | `hft.min_imbalance_ratio` | OBI threshold (e.g. 2.5×). |
| Min spread bps | `hft.min_spread_bps` | Don’t trade when spread too wide. |
| Arb min profit bps | `hft.arb_min_profit_bps` | Min profit after fees for cross-exchange arb. |
| Tick window | `hft.tick_window` | Ticks for momentum. |
| Disabled in regimes | `hft.disabled_in_regimes` | e.g. ["high_vol"] to disable in volatile regimes. |
| Advanced HFT infra | `hft.use_advanced_hft_infrastructure` | Set true for advanced event loop and signal ring if available. |

---

## 16. Quantum bot (optional)

| Item | Config path | Action |
|------|-------------|--------|
| Quantum walk | `quantum_bot.use_quantum_walk`, `use_multi_timeframe_walk` | Improves signal confidence when deps available. |
| Quantum Monte Carlo risk | `quantum_bot.use_quantum_monte_carlo_risk` | VaR/CVaR; optional. |
| Volatility/correlation | `quantum_bot.volatility_adjusted_limits`, `correlation_aware_sizing` | Set true when you feed vol and correlation. |

---

## 17. Emergency shutdown

| Item | Config path | Action |
|------|-------------|--------|
| Enabled | `emergency_shutdown.enabled`, `auto_shutdown` | Keep on. |
| Conditions | `emergency_shutdown.conditions` | Already includes drawdown, daily loss, consecutive losses, error rate, latency spike, execution delay, flash crash, network failure, arb failure. Ensure code implements each and add alerting. |

---

## 18. 10GbE network

| Item | Config path | Action |
|------|-------------|--------|
| Enabled | `network_10gbe.enabled` | On when Solarflare is installed and drivers loaded. |
| Driver / Onload | `network_10gbe.driver_mode`, `onload_profile` | "onload", "latency". |
| Target latency/jitter | `network_10gbe.target_latency_us`, `max_jitter_us` | &lt;1 μs. |
| Ring sizes | `network_10gbe.rx_ring_size`, `tx_ring_size` | 4096. |
| SFN8522 options | `network_10gbe.sfn8522_optimizations` | pio_mode, ctpio_mode, rx_coalescing false, busy_poll_us. |
| Arbitrage | `network_10gbe.arbitrage.*` | min_profit_threshold_pct, max_execution_time_ms, enable_flash_arbitrage, cross_exchange_enabled. |
| Switch config | `network_10gbe.switch_infrastructure` | Matches 2× X460-G2 MLAG; ensure switches are actually configured (see switch checklist). |
| Trading frequency | `network_10gbe.trading_frequency` | min_trade_interval_ms, max_trades_per_hour, enable_micro_trading. |

---

## 19. Monitoring and alerts

| Item | Config path | Action |
|------|-------------|--------|
| Prometheus/Grafana | `monitoring.prometheus`, `monitoring.grafana` | Enable; set ports and credentials. |
| Telegram | `monitoring.alerts.telegram` | Enable and set bot_token, chat_id for drawdown/daily loss/consecutive losses/errors. |
| Email | `monitoring.alerts.email` | Optional; smtp_*, from_email, to_email. |
| Triggers | `monitoring.alerts.triggers` | drawdown 15%, daily_loss 5%, consecutive_losses 5, error_rate 0.10. Add more if needed (e.g. circuit breaker). |

---

## 20. Data and persistence

| Item | Config path | Action |
|------|-------------|--------|
| Tick store | `data.persist_tick_store`, `persist_to_tick_store` | On for live/paper if you use tick-level analysis. |
| Data lake | `data.use_lake_read`, `persist_to_lake`, `lake_path` | Use for OHLCV cache and backtest data. |

---

## 21. Multi-language (optional)

| Item | Config path | Action |
|------|-------------|--------|
| Endpoints | `multi_language.endpoints` | Point to Rust/Cpp/Cuda/Go/… services if you run them; otherwise disable or leave unused. |

---

## 22. Logging

| Item | Config path | Action |
|------|-------------|--------|
| Level | `logging.level` | INFO for production; DEBUG for troubleshooting. |
| File / rotation | `logging.file`, `max_file_size_mb`, `backup_count` | Avoid unbounded growth; rotate and archive. |

---

## 23. Performance targets (reference)

| Item | Config path | Action |
|------|-------------|--------|
| Win rate / Sharpe / drawdown / monthly return | `targets.*` | 65%, 2.0, 15%, 12%; use for monitoring dashboards and alerts. |
| Network/execution/arb targets | `targets.network_latency_us`, `execution_time_ms`, `arbitrage_win_rate` | Track in Grafana. |

---

## 24. Hardware and infrastructure

| Area | Action |
|------|--------|
| **R730** | 64GB RAM; Ubuntu 22.04; Solarflare + OpenOnload; NTP; static IPs per 10gbe_network_config; run setup-linux.sh then deploy_production_linux.sh. |
| **Desktop** | Use for dev, backtest, ML (RTX 5080); clone same repo; optional 10G later. |
| **Switches** | 2× SFP+ for ISC; jumbo, QoS 7, cut-through, edge ports; NTP; optional second PSU, VIM-2x, TM-CLK, spare DACs. See [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md). |
| **Cabling** | Short cables (1–3 m); label ISC and server uplinks; spare SFP+ in rack. |
| **Power** | Dual PDUs; UPS for R730 + switches; metered PDUs optional. |
| **Cooling** | Blanking panels; airflow; temp sensor optional. |
| **Management** | 1G management switch for iDRAC/switch management/SSH; out-of-band. |

---

## 25. Deployment and operations

| Area | Action |
|------|--------|
| **Deploy script** | Run `scripts/deploy_production_linux.sh` on R730 after setup-linux.sh. |
| **Systemd** | WorkingDirectory = repo dir; ExecStart with `--config unified_config.yaml` (or production override). |
| **Backups** | Cron daily: `scripts/backup_config_and_logs.sh`. |
| **Health check** | `scripts/health_check.py --config unified_config.yaml`; add `--paper-smoke` weekly or pre/post change. |
| **Config override** | Use `config/unified_config.production.example.yaml` (run_template for R730) and merge into production file. |

---

## 26. Security and credentials

| Area | Action |
|------|--------|
| **.env** | Never commit; only on host (e.g. /opt/argus/repo/.env). Backup script redacts. |
| **API keys** | Kraken + Coinbase in .env; restrict exchange API to trading + read; no withdraw. |
| **SSH** | Key-based; disable root login; use management VLAN for access. |
| **TACACS/RADIUS** | Optional on switches for central auth. |

---

## 27. Testing and validation

| Area | Action |
|------|--------|
| **Validate** | `python main.py validate` after every pull or config change. |
| **Paper smoke** | `scripts/health_check.py --paper-smoke` (and optionally `--paper-days N`). |
| **Backtest** | Run backtest on recent data before live or after strategy/config changes. |
| **Walk-forward** | Optional: train on rolling window, test on next period; repeat. |

---

## 28. ML and models

| Area | Action |
|------|--------|
| **XGBoost** | Retrain periodically (`ml/train_xgboost.py`) on recent OHLCV; save to data/models. |
| **Ensemble** | `ml/model_ensemble.py` loads trained model; ensure feature names match feature_library_300. |
| **Features** | Add/remove features in `ml/feature_library_300.py` based on importance and correlation. |
| **Online learning** | If enabled, ensure adaptive weights update from recent PnL. |

---

## 29. Process and discipline

| Action |
|--------|
| Paper 2–4 weeks before live; review win rate, drawdown, monthly return. |
| Go live with small capital ($1k or your minimum). |
| Scale capital only after 2–3 months of acceptable results. |
| One config change at a time; observe before next change. |
| Alert on circuit breaker, daily loss limit, error rate, bot down. |
| Keep strategy_whitelist and live_disabled_strategies aligned with paper/backtest results. |
| Document any manual overrides and reasons. |

---

## 30. Documentation and repo

| Area | Action |
|------|--------|
| **Runbook** | [UNIFIED_RUNBOOK.md](../UNIFIED_RUNBOOK.md) – canonical commands. |
| **Deployment** | [DEPLOYMENT_R730_AND_DESKTOP.md](DEPLOYMENT_R730_AND_DESKTOP.md), [R730_SETUP.md](R730_SETUP.md). |
| **Switches** | [SWITCH_IMPROVEMENTS_CHECKLIST.md](SWITCH_IMPROVEMENTS_CHECKLIST.md). |
| **Roadmap** | [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) – prioritized improvements. |
| **README** | Clone URL, badges, links to all above. |
| **CI** | Run validate (and optionally paper-smoke) in GitHub Actions on push/PR. |

---

## Summary: highest-impact order

1. **Before live:** Paper 2–4 weeks; raise `min_signal_confidence` for live (0.72–0.75); enable Telegram/email alerts; verify 10G; backups cron.
2. **Config:** Tight strategy_whitelist; regime filter on; edge-cost gate; realistic fees/slippage; volatility-adjusted limits.
3. **Infrastructure:** R730 64GB, NTP, switch checklist (ISC, jumbo, QoS); monitoring + alerts.
4. **Ongoing:** Retrain ML; review strategy PnL; one change at a time; scale capital only after sustained results.

This document is the master list; [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) gives a shorter prioritized set.
