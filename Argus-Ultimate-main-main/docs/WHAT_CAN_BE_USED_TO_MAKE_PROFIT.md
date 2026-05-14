# What Can Be Used to Make Profit

A single reference for **every mechanism in the bot** that can be used to improve or protect profitability. No guarantee of profit; these are the levers you can tune and the features that affect PnL.

---

## Best ways this bot can earn profit (prioritized)

These are the **highest-impact** ways to tilt the bot toward profit. Do these first.

| # | Best way | What to do |
|---|----------|------------|
| **1** | **Trade only high-conviction setups** | Set `ai_brain.min_signal_confidence` 0.76+ (paper), `live_min_signal_confidence` 0.80+ (live). Fewer trades, but each has a real edge. |
| **2** | **Require edge to exceed costs** | Keep `edge_cost_gate` on with `min_edge_pct` 0.85%+, `buffer_mult` 2.0+. In live use `live_min_edge_pct` 1.0, `live_buffer_mult` 2.2. Avoids fee-churn and marginal trades. |
| **3** | **Only use strategies that make money** | Run paper 2–4 weeks, then `scripts/kill_losers_review.py`. Set `strategies.strategy_whitelist` to strategies with **positive PnL EMA** in paper_results or allocator. Remove losers from whitelist. |
| **4** | **Let the allocator favor winners** | Keep `strategy_allocator.enabled: true`, `exploration_c` 0.75–0.9 (favor proven winners). Allocator ranks by PnL EMA so better strategies get more weight. |
| **5** | **Evolve params to the current market** | Use `evolution_load_evolved: true`, `evolution_use_composite_fitness: true`, `negative_return_penalty_weight: 0.3`. Run evolution on a schedule or `trigger_on_trade` + `after_n_trades`. Best params go to `evolved_params.json` and get applied. |
| **6** | **Better risk:reward per trade** | Set `take_profit_pct` 5–6%+, `stop_loss_pct` 1.5–2% (e.g. 3:1 or 4:1 R:R). Enable `paper_trading.partial_tp_at_pct` and `trailing_stop_enabled` to lock in profit and let winners run. |
| **7** | **Concentrate capital on best ideas** | Limit `max_concurrent_signals` to 2, `strategy_allocator.max_total_signals` to 4. Size by confidence so the best signals get more capital (optimizer already does this). |
| **8** | **Trade in the right regime** | Turn on `strategies.regime_filter_enabled` so trend strategies only fire in trend and mean-reversion in range. Regime-aligned signals already get a confidence boost. |
| **9** | **Reduce execution drag** | Use realistic `slippage_pct` and `backtest.slippage_bps`; set `execution_engine.max_slippage_pct` to reject bad fills. For larger orders use VWAP/TWAP (`vwap_large_order_threshold_aud`) or DCA levels. |
| **10** | **Protect capital so you can keep trading** | Set circuit breaker, `max_daily_loss_pct`, `max_drawdown_pct`. One bad day shouldn’t wipe out gains. |
| **11** | **Add external alpha (optional)** | If you have another signal source (e.g. API), set `external_alpha_enabled: true` and `external_alpha_url`. Scanner merges it and ranks by score. |
| **12** | **Weekly discipline** | Run `kill_losers_review`, trim whitelist, check paper_results and allocator stats. Small, consistent tweaks beat rare big changes. |

**One-line summary:** Trade fewer, higher-conviction setups with real edge; only allow strategies that are profitable in paper; let the allocator and evolution favor winners; use good R:R and execution; protect capital.

**Implementation status (all 12 + full-doc levers applied in config + scripts):**  
1–2: `ai_brain` + `edge_cost_gate` (and paper_trading edge aligned). 3: `strategy_whitelist` + `scripts/kill_losers_review.py` (run weekly; trim whitelist). 4: `strategy_allocator` (exploration_c 0.75, max_total_signals 4). 5: `evolution` (load_evolved, use_composite_fitness, negative_return_penalty_weight, allocator_decay_after_apply: 0.5). 6: `risk` / `paper_trading` (take_profit_pct 6%, stop_loss_pct 1.5%, partial_tp, trailing_stop). 7: max_concurrent_signals 2, allocator max 4, scanner top_n 4. 8: `regime_filter_enabled`, `use_volatility_regime_scale: true`, scanner `signal_multi_timeframe_enabled: true`. 9: `max_slippage_pct`, `max_spread_bps: 50`, VWAP threshold, `use_twap_for_large_orders: true`, `dca_levels_pct: [0.33, 0.33, 0.34]`, `use_is_gate: true`, `max_avg_is_bps: 25`. 10: circuit_breaker, max_daily_loss_pct, max_drawdown_pct, `emergency_shutdown.enabled: true`. 11: `external_alpha_*` (set URL when you have one). 12: `scripts/weekly_profitability_check.py`; cron in `scripts/cron_example.txt`.  
**Also on:** `use_funding_rate_filter: true`, `backtest.fill_probability: 0.98`.

---

## 1. Signal quality (only trade when edge is real)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Min signal confidence** | `ai_brain.min_signal_confidence`, `live_min_signal_confidence` | Only signals with confidence ≥ this are traded. Higher = fewer, higher-conviction trades. |
| **Strategy whitelist** | `strategies.strategy_whitelist` | When non-empty, only these strategies may emit signals. Use after paper/allocator shows which are profitable. |
| **Disabled strategies** | `paper_trading_disabled_strategies`, `live_disabled_strategies` | Block specific strategies (e.g. losers). |
| **Edge-cost gate** | `edge_cost_gate.*` | Only allow a trade when expected move to take-profit exceeds round-trip costs by a buffer. `min_edge_pct`, `buffer_mult`, `live_min_edge_pct`, `live_buffer_mult`. |
| **Regime filter** | `strategies.regime_filter_enabled`, `regime_filter_trend_strategies`, `regime_filter_mr_strategies` | Only allow trend strategies in trend regime, mean-reversion in range. Reduces wrong-regime trades. |
| **Regime-aligned boost** | Strategy engine | Regime-aligned signals get +8% confidence so they rank higher. |
| **Implementation shortfall gate** | `execution_engine.use_is_gate`, `max_avg_is_bps` | Reject execution when strategy/symbol avg IS (cost) > threshold. |

**Config:** `unified_config.yaml` → `ai_brain`, `edge_cost_gate`, `strategies`.

---

## 2. Which signals get traded (selection and ranking)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Continuous scanner** | `services/continuous_best_trade_scanner.py` | Aggregates strategy engine, HFT, external alpha, strategy library; scores by confidence×strength + liquidity + edge; diversity (max per symbol/strategy); top N. |
| **Strategy allocator** | `adaptive/strategy_allocator.py` | PnL-based ranking (UCB-style): favors strategies with higher PnL EMA, some exploration. `rank_signals()` limits `max_total_signals`, `max_per_strategy`. |
| **Scanner score** | `_score_signal()` in scanner | Base = confidence×strength×liquidity_mult; strategy bonus (e.g. arbitrage +15%); expected_profit_pct (e.g. HFT); recency. Higher score = chosen first. |
| **Multi-timeframe confirmation** | Scanner `_gather_multi_tf_signals()` | Only keep entry-TF signals that agree with primary-TF direction. Fewer but higher-quality. |
| **Max concurrent signals** | `ai_brain.max_concurrent_signals`, allocator `max_total_signals` | Cap how many ideas are traded; concentrate capital. |

**Config:** `continuous_scan`, `strategy_allocator`, `ai_brain.max_concurrent_signals`.

---

## 3. Position sizing and risk per trade

| Lever | Where | What it does |
|-------|--------|---------------|
| **Min/max position size (AUD)** | `min_position_size_aud`, `max_position_size_aud` | Each trade size is bounded. Larger max = more profit per winner (and more loss per loser). |
| **Max position %** | `max_position_pct` | Cap position as % of equity. |
| **Capital optimizer** | `unified_capital_optimizer.py` | Sizes by confidence×strength, respects min/max, optional volatility scaling (higher vol = smaller size). |
| **Volatility/ATR scaling** | Optimizer + signal `atr_pct`/`volatility` | Reduces size when vol is high (regime/ATR-aware). |
| **Correlation-aware sizing** | `use_correlation_aware_sizing`, risk correlation_matrix | HRP/BL/MPT-style weights; can reduce size in correlated directions. |
| **Max concurrent positions** | `max_concurrent_positions` | Spread capital across up to N positions. |
| **Signal cooldown** | `execution.signal_cooldown_bars`, paper_trading | Min bars between trades on same symbol. Reduces overtrading. |

**Config:** `risk`, `execution`, `paper_trading`, capital in config.

---

## 4. Entry/exit (R:R and trade management)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Take profit %** | `take_profit_pct` (risk / paper_trading) | Target exit. Higher = let winners run more (and may reduce win rate). |
| **Stop loss %** | `stop_loss_pct` | Hard exit on loss. Tighter = cut losers faster; wider = more room but bigger losses. |
| **Partial take profit** | `paper_trading.partial_tp_at_pct`, `partial_tp_close_pct` | Take partial at X% profit (e.g. close 50% at 3%). Locks in some profit. |
| **Trailing stop** | `paper_trading.trailing_stop_enabled`, `trailing_stop_pct` | Trail stop below high-water (long) or above low-water (short). Lets winners run while protecting profit. |
| **Strategy engine TP/SL** | `strategies/unified/strategy_engine.py` | Each signal carries stop_loss and take_profit; execution/position logic can use them. |

**Config:** `risk`, `paper_trading`.

---

## 5. Execution (don’t give back the edge)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Slippage assumption** | `slippage_pct`, `execution_engine.max_slippage_pct` | Paper/backtest use slippage; live rejects if realized > max. Tighter = fewer bad fills. |
| **Spread gate** | `execution_engine.max_spread_bps` | Skip order when bid-ask spread > threshold. |
| **VWAP/TWAP for large orders** | `vwap_large_order_threshold_aud`, `use_twap_for_large_orders` | Slice large orders over time to reduce impact. |
| **DCA levels** | `execution_engine.dca_levels_pct` | Split entry into multiple levels (e.g. [0.33, 0.33, 0.34]). Can improve average entry. |
| **Order fill timeout** | `order_fill_timeout_seconds` | Cancel if not filled in time; avoid stale prices. |
| **Implementation shortfall tracking** | `execution/implementation_shortfall.py`, `is_tracker` | Measure and gate by execution cost. |

**Config:** `execution_engine`, `execution`.

---

## 6. Alpha sources (where signals come from)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Unified strategy engine** | `strategies/unified/strategy_engine.py` | RSI, Bollinger, MACD, regime, order-book bias; mean-reversion + trend + volume momentum + breakout + MACD trend; regime-aligned. Evolved params from `evolved_params.json`. |
| **Strategy library** | `strategies/strategy_library_impl`, scanner `_gather_strategy_library()` | Tier + algorithmic strategies; enable via `strategy_library_strategies_enabled`. |
| **External alpha** | Scanner `_gather_external_alpha()` | HTTP URL returns signals; merged into scanner. Set `external_alpha_enabled`, `external_alpha_url`. |
| **Strategy plugins** | Scanner `_gather_strategy_plugins()` | `strategy_plugin_modules` → get_strategies(config) → analyze(market_data). Add custom strategies. |
| **HFT (order book / trade flow)** | HFT engine, scanner | Order-book imbalance, trade-flow momentum. `hft_enabled`, `max_hft_signals_per_cycle`. |
| **Multi-timeframe** | Scanner | Primary TF + entry TF; only signals that agree. |
| **Regime boost** | `ml/regime_boost.py`, `strategies.use_regime_lstm_boost` | Scale confidence by regime momentum/vol. |
| **Funding rate filter** | `use_funding_rate_filter`, `funding_rate_skip_long_threshold` | Skip long when funding rate ≥ threshold (e.g. avoid expensive longs). |

**Config:** `strategies`, `strategy_library`, `continuous_scan`, `hft`.

---

## 7. Learning and adaptation (improve over time)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Strategy allocator** | `adaptive/strategy_allocator.py` | Records PnL per strategy/regime; rank_signals favors higher PnL EMA. |
| **Online tuner** | `adaptive/online_tuner.py` | Strategy engine calls it; adjusts RSI/BB selectivity from realized PnL per regime/mode. |
| **Evolution (GA)** | `evolution/evolution_unified.py`, `strategy_genetic_algorithm.py` | Paper-loop fitness (Sharpe/composite); best params written to `evolved_params.json`; auto_apply pushes to config. |
| **Composite fitness** | `evolution_use_composite_fitness`, `negative_return_penalty_weight`, `composite_calmar_weight` | Evolution optimizes Sharpe + Sortino + Calmar − drawdown − negative return. Favors profitable, stable params. |
| **Evolved params at startup** | `evolution_load_evolved`, `data/evolved_params.json` | Strategy engine and others load tuned params (RSI, BB, confidence, etc.). |
| **Kill losers** | `scripts/kill_losers_review.py` | Suggests strategies to remove from whitelist (negative PnL EMA). Trim whitelist weekly. |
| **Allocator decay after evolution** | `evolution_allocator_decay_after_apply` | After applying new params, decay allocator stats so bandit doesn’t over-exploit old behavior. |

**Config:** `evolution`, `strategy_allocator`.

---

## 8. Risk and safety (protect capital)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Circuit breaker** | `UnifiedRiskManager.check_circuit_breaker()`, `trip_circuit_breaker()` | Stops trading on daily loss, consecutive losses, drawdown, VaR breach. |
| **Max daily loss %** | `risk.max_daily_loss_pct` | Stop when daily loss exceeds this. |
| **Max drawdown %** | `risk.max_drawdown_pct` | Emergency stop when drawdown exceeds. |
| **Portfolio guardrails** | Main loop `_portfolio_allows_signal()` | Block SELL when flat; block BUY when insufficient cash. |
| **Pre-trade risk block** | `execution/risk_compliance_audit.pre_trade_risk_block()` | Single call: (approved, reason). Centralized risk check. |
| **Emergency stop** | `_check_emergency_stop()` | Latency, flash crash, network, arb conditions. |

**Config:** `risk`, `paper_trading` (circuit_breaker_dd_pct, etc.).

---

## 9. Backtest and paper realism (tune so live matches)

| Lever | Where | What it does |
|-------|--------|---------------|
| **Backtest slippage** | `backtest.slippage_bps` | Per-trade slippage in backtest. Higher = more conservative PnL. |
| **Fill probability** | `backtest.fill_probability` | Probability each fill happens (e.g. 0.98 = 2% rejected). Reduces overstatement. |
| **Paper simulates live** | `paper_trading.simulate_live` | Paper uses same config as live + simulated slippage. |
| **Live vs backtest check** | `scripts/live_vs_backtest_consistency.py` | Compare realized slippage to backtest assumptions. |

**Config:** `backtest`, `paper_trading`.

---

## 10. Process and discipline (no code change)

| Action | Purpose |
|--------|---------|
| Run paper 2–4 weeks before live | Validate edge and behavior. |
| Run `scripts/kill_losers_review.py` weekly | Remove strategies with negative PnL from whitelist. |
| Run `scripts/readiness_score.py --include-paper` | Checklist before live. |
| Run `scripts/pre_live_check.py` | Full pre-live check; edge gate when `live_require_paper_edge`. |
| Run evolution regularly (or trigger on trade) | Keep params adapted to recent market. |
| Review `data/paper_results.json`, `data/strategy_allocator_stats.json` | See which strategies/symbols are profitable. |
| Run `scripts/tca_summary.py` after live trades | Avg slippage by strategy/symbol; tune execution. |

---

## Quick map: where to tune for profit

- **Fewer, better trades:** `min_signal_confidence`, `live_min_signal_confidence`, `edge_cost_gate.*`, `strategy_whitelist`, `regime_filter_enabled`, `signal_cooldown_bars`.
- **Larger edge per trade:** `take_profit_pct`, `stop_loss_pct` (R:R), `partial_tp_*`, `trailing_stop_*`.
- **Better selection:** `strategy_allocator.exploration_c`, `max_total_signals`, scanner `top_n`, multi-timeframe confirmation.
- **Better sizing:** `min_position_size_aud`, `max_position_size_aud`, `max_position_pct`, volatility scaling, correlation-aware sizing.
- **Better execution:** `slippage_pct`, `max_slippage_pct`, `max_spread_bps`, VWAP/TWAP, DCA levels.
- **Adaptation:** `evolution_load_evolved`, `evolution_use_composite_fitness`, `negative_return_penalty_weight`, `strategy_allocator`, kill_losers + whitelist.
- **More alpha:** `external_alpha_*`, `strategy_library_*`, `strategy_plugin_modules`, HFT, regime boost, funding filter.

See also: [RATING_AND_ROADMAP.md](RATING_AND_ROADMAP.md) (profitability tuning), [EVERYTHING_BEYOND_CAPABILITIES.md](EVERYTHING_BEYOND_CAPABILITIES.md) (full feature list).
