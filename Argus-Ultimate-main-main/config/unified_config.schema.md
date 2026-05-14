# unified_config.yaml — Schema Reference

> **Canonical source:** `unified_config.yaml` (repo root)  
> **Loaded by:** `core/config_manager.py`, `argus_live/`  
> **Validated by:** `config/validated_config.py` against `config/schema.py`

---

## Top-Level Keys

### `config_version`
| Attribute | Value |
|-----------|-------|
| Type | `int` |
| Default | `1` |
| Description | Schema version. Increment when making breaking schema changes. Used by the migration system to detect stale config files. |

---

### `run_template`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Default | *(hardware spec)* |
| Description | Target hardware specification. Informational only — does not affect runtime behaviour. Documents the CPU, GPU, RAM, NIC, and network topology the bot is tuned for. |

Sub-keys: `cpu`, `motherboard`, `ram`, `gpu`, `gpu_specs`, `switches`, `switch_topology`, `nic`.

---

### `capital`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Default | See sub-keys |
| Description | Capital sizing, position limits, and currency settings. |

| Sub-key | Type | Default | Description |
|---------|------|---------|-------------|
| `starting_capital_aud` | `float` | `1000.0` | Starting capital in AUD. |
| `currency` | `string` | `"AUD"` | Base currency for accounting. |
| `min_position_size_aud` | `float` | `15.0` | Minimum order size in AUD (avoids fee drag on tiny trades). |
| `max_position_size_aud` | `float` | `180.0` | Maximum single position in AUD. |
| `max_position_pct` | `float` | `0.35` | Maximum fraction of capital in one position (0–1). |
| `max_total_exposure_pct` | `float` | `0.95` | Maximum fraction of capital deployed at once (0–1). |
| `max_concurrent_positions` | `int` | `5` | Maximum number of open positions simultaneously. |
| `enable_micro_positions` | `bool` | `false` | Enable 10GbE micro-position mode. |
| `flash_position_max_aud` | `float` | `0.0` | Maximum flash position size (0 = disabled). |

---

### `fx`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Foreign exchange rates for cross-currency sizing. |

| Sub-key | Type | Default | Description |
|---------|------|---------|-------------|
| `aud_to_usd` | `float` | `0.65` | USD per 1 AUD. Used when trading USD-quoted pairs with AUD capital. |

---

### `risk`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Core risk-management controls. Applied to both live and paper trading. |

| Sub-key | Type | Default | Description |
|---------|------|---------|-------------|
| `max_daily_loss_pct` | `float` | `0.10` | Maximum daily loss as fraction of capital before shutdown (validator minimum). |
| `max_drawdown_pct` | `float` | `0.10` | Maximum peak-to-trough drawdown allowed. |
| `stop_loss_pct` | `float` | `0.012` | Per-trade stop-loss percentage (1.2%). |
| `take_profit_pct` | `float` | `0.035` | Per-trade take-profit percentage (3.5%). |
| `max_consecutive_losses` | `int` | `20` | Consecutive losses before position-size tightening. |
| `max_error_rate` | `float` | `0.25` | Fraction of erroneous cycles that triggers shutdown. |
| `use_volatility_adjusted_limits` | `bool` | `true` | Scale risk limits by realised volatility. |
| `realized_vol_pct` | `float` | `0.0` | Realised volatility (computed each cycle; 0 = not yet computed). |
| `portfolio_var_limit_pct` | `float` | `0.10` | Block new risk when 95% VaR ≥ this fraction of capital. |
| `portfolio_cvar_limit_pct` | `float` | `0.08` | Block new risk when 99% CVaR ≥ this fraction of capital. |
| `portfolio_var_confidence` | `float` | `0.95` | Confidence level for VaR calculation. |
| `portfolio_var_lookback_trades` | `int` | `50` | Number of trades used for rolling VaR calculation. |
| `cluster_drawdown_brake_pct` | `float` | `0.0` | Block new risk when cluster drawdown exceeds this (0 = disabled). |
| `target_cluster_cap_pct` | `float` | `0.40` | Cluster risk budget as fraction of equity. |
| `risk_cluster_map` | `object` | `{}` | Optional mapping of symbol → cluster name. |
| `portfolio_vol_target_pct` | `float` | `2.0` | Realised volatility target for exposure scaling (%). |
| `portfolio_liquidity_spread_ref_bps` | `float` | `20.0` | Spread reference (basis points) for liquidity-adjusted sizing. |
| `portfolio_exposure_min_scale` | `float` | `0.30` | Minimum exposure scale so position cap never collapses to zero. |
| `circuit_breaker_dd_pct` | `float` | `8.0` | Trip circuit breaker when drawdown ≥ this percentage. |
| `var_breach_pct` | `float` | `5.0` | VaR breach threshold (%) for circuit breaker. |
| `var_breach_alert_enabled` | `bool` | `true` | Emit alert event on VaR breach. |
| `auto_reduce_after_n_losses` | `int` | `3` | Consecutive losses before automatic size reduction. |
| `auto_reduce_factor` | `float` | `0.6` | Position-size multiplier during auto-reduce (60%). |
| `emergency_shutdown` | `object` | — | Hard-stop conditions (see sub-keys). |
| `emergency_shutdown.enabled` | `bool` | `true` | Enable emergency shutdown logic. |
| `emergency_shutdown.latency_spike_ms` | `int` | `600000` | Trip if a cycle takes longer than this (ms). |
| `emergency_shutdown.flash_crash_pct` | `float` | `15.0` | Trip if any symbol moves > this % in one cycle. |
| `emergency_shutdown.network_fail` | `bool` | `true` | Trip when exchange is unreachable. |
| `emergency_shutdown.arb_spread_bps` | `int` | `500` | Trip if spread > this many basis points. |

---

### `runtime_safety`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Soak-stability safety controls. |

| Sub-key | Type | Default | Description |
|---------|------|---------|-------------|
| `latency_grace_cycles` | `int` | `2` | Number of slow cycles tolerated before latency alert. |
| `live_safe_disable_pinnacle_ai_brain` | `bool` | `false` | Disable AI brain in live-safe mode for stability. |

---

### `market_data`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Market-data polling and caching configuration. |

| Sub-key | Type | Default | Description |
|---------|------|---------|-------------|
| `ohlcv_cache_seconds` | `int` | `30` | OHLCV cache TTL in seconds. |
| `ohlcv_poll_interval_seconds` | `int` | `30` | How often to poll for new OHLCV data. |
| `ohlcv_retry_attempts` | `int` | `2` | Number of retries on OHLCV fetch failure. |
| `enable_historical_preload` | `bool` | `true` | Pre-load historical candles on startup. |
| `historical_lookback_hours` | `int` | `4380` | Hours of historical data to pre-load (≈ 6 months). |
| `historical_timeframe` | `string` | `"1h"` | Timeframe for historical pre-load. |
| `historical_cache_dir` | `string` | `"data/historical"` | Directory for cached historical data. |

---

### `exchanges`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Exchange connectivity settings. |

| Sub-key | Type | Default | Description |
|---------|------|---------|-------------|
| `primary` | `string` | `"kraken"` | Primary exchange identifier. |
| `secondary` | `string` | `"coinbase_advanced"` | Secondary/fallback exchange identifier. |
| `use_ccxt` | `bool` | `true` | Use CCXT for primary exchange market data and execution. |
| `supported` | `list[string]` | `["kraken", "coinbase_advanced"]` | List of supported exchange identifiers. |
| `kraken.maker_fee` | `float` | `0.0016` | Kraken maker fee (0.16%). |
| `kraken.taker_fee` | `float` | `0.0026` | Kraken taker fee (0.26%). |
| `kraken.min_order_size` | `float` | `0.0001` | Minimum order size on Kraken. |
| `coinbase_advanced.maker_fee` | `float` | `0.006` | Coinbase Advanced maker fee (0.60%). |
| `coinbase_advanced.taker_fee` | `float` | `0.008` | Coinbase Advanced taker fee (0.80%). |
| `coinbase_advanced.min_order_size` | `float` | `1.0` | Minimum order size in USD on Coinbase. |

---

### `best_symbols`
| Attribute | Value |
|-----------|-------|
| Type | `list[string]` |
| Default | `["BTC/USD", "ETH/USD", "SOL/USD", ...]` |
| Description | Curated high-liquidity symbols for scanner and live trading. Subset of `trading_pairs`. |

---

### `trading_pairs`
| Attribute | Value |
|-----------|-------|
| Type | `list[string]` |
| Default | 48-pair universe (Tier 1–3 by liquidity) |
| Description | Full tradeable universe. Shared by live trading and paper trading. Tier 1 = major coins; Tier 2 = large cap; Tier 3 = mid cap. |

---

### `execution`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Order execution parameters including slippage model and timing. |

---

### `execution_routing`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Smart order routing: venue selection logic and routing heuristics. |

---

### `continuous_scan`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Continuous market scanner: scan interval, top-N pair selection, regime filter. |

---

### `liquidity_scanner`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Spread and volume thresholds for filtering illiquid instruments. |

---

### `edge_cost_gate`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Minimum expected edge (after fees and slippage) required before entering a trade. |

---

### `portfolio_target_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Portfolio-level exposure targets and rebalance thresholds. |

---

### `liquidity_risk_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Liquidity-adjusted position sizing configuration. |

---

### `execution_alpha_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Smart execution timing parameters to capture alpha at order placement. |

---

### `strategy_evaluation_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Real-time strategy scoring and promotion/demotion criteria. |

---

### `self_optimizing_meta_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Meta-learning configuration for automated parameter self-tuning. |

---

### `champion_challenger`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | A/B strategy testing framework. Controls allocation split and evaluation window. |

---

### `market_microstructure_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Order-book depth and microstructure analysis settings. |

---

### `recon_recovery_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Reconciliation and position recovery on restart or exchange disconnect. |

---

### `system_health_metrics`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Health score thresholds, alerting configuration, and metric export. |

---

### `ai_brain`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Pinnacle AI Brain: model selection, confidence thresholds, feature list, inference settings. |

---

### `strategies`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Per-strategy enable flags and default parameter values (EMA crossover, RSI, MACD, Bollinger, etc.). |

---

### `strategy_library`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Extended strategy definitions and composite signal logic. |

---

### `strategy_router`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Regime-to-strategy routing matrix. Maps detected market regime to active strategy set. |

---

### `quantum_features`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Quantum-enhanced signal generation: feature flags, qubit budgets, backend selection. |

---

### `quantum_simulator`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Quantum simulator backend settings: maximum qubits, shot count, backend type (statevector / MPS). |

---

### `quant_fund_upgrades`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Institutional quant features: factor models, alpha decay monitoring. |

---

### `transcendent`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Experimental transcendent/godmode AI features. Not enabled in production by default. |

---

### `argus_strategies`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Argus-specific strategy compositions and ensemble definitions. |

---

### `execution_engine`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Execution engine: concurrency limits, order batching, fill tracking parameters. |

---

### `data`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Data pipeline: feed providers, cache TTLs, normalisation rules. |

---

### `multi_language`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Polyglot engine configuration for Rust/Go/C++ sub-processes. |

---

### `monitoring`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Prometheus/Grafana export settings, alerting webhooks, log levels. |

---

### `paper_trading`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Paper trading mode settings: simulated fills, virtual balance, session reporting. |

---

### `runtime`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Boot sequence, hot-reload configuration, graceful shutdown timeouts. |

---

### `reconciliation`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | End-of-day reconciliation rules and tolerance thresholds for position and balance checks. |

---

### `backtest`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Backtesting engine parameters: mode, slippage model, data range, commission model. |

---

### `evolution`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Evolutionary optimisation settings: GA/CMA-ES configuration, population size, generations, fitness weights. |

---

### `strategy_allocator`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Capital allocation across strategies: Kelly sizing, equal-weight, and risk-parity modes. |

---

### `hft`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | High-frequency trading micro-optimisation flags (latency-sensitive paths). |

---

### `quantum_bot`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Quantum-enhanced bot integration: circuit parameters and hybrid classical/quantum settings. |

---

### `emergency_shutdown`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Top-level alias for hard-stop conditions. See `risk.emergency_shutdown` for the canonical location. |

---

### `logging`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Logging format, file rotation, and per-module verbosity levels. |

---

### `targets`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Performance targets: target Sharpe ratio, win rate, monthly return, and Calmar ratio. |

---

### `perp_exchanges`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Perpetual futures exchange configuration (leverage limits, funding-rate thresholds). |

---

### `funding_rate_harvester`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Funding-rate arbitrage strategy settings (min rate, hedge ratio). |

---

### `alternative_data`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Alternative data feed configuration: news, on-chain metrics, sentiment API keys and weights. |

---

### `ml_models`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | ML model registry: model type, checkpoint path, feature set, and retrain schedule. |

---

### `advanced_risk`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Advanced risk overlays: CVaR computation, stress-test scenarios, tail hedging. |

---

### `market_making`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Market-making spread and inventory management settings. |

---

### `stat_arb_cointegration`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Statistical arbitrage: cointegration pair selection and mean-reversion parameters. |

---

### `storage`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Database and storage backend configuration (SQLite, PostgreSQL, Redis). |

---

### `websocket_orders`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | WebSocket-based order placement configuration (reconnect logic, heartbeat). |

---

### `market_impact`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Market-impact model parameters (Almgren-Chriss: eta, lambda, gamma). |

---

### `algo_execution`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Algorithmic execution settings: TWAP, VWAP, POV slice parameters. |

---

### `backtesting`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Extended backtesting settings (alias/extension of `backtest`). |

---

### `live_gate`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Live-trading gate: readiness checks, minimum capital, required confirmations before first trade. |

---

### `position_registry`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Position registry: persistence backend and reconciliation interval. |

---

### `mtf_confluence`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Multi-timeframe confluence signal parameters (required agreement across timeframes). |

---

### `portfolio_optimizer`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Portfolio optimiser: mean-variance, Black-Litterman, and risk-parity settings. |

---

### `rolling_performance_feeder`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Rolling performance statistics fed to the meta-learner for online adaptation. |

---

### `hot_reload`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Config hot-reload watcher: debounce interval, watched paths. |

---

### `telegram`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Telegram alert bot: bot token, chat ID, alert level thresholds. |

---

### `news_sentiment`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | News sentiment NLP model configuration and feed endpoint settings. |

---

### `defi_lp`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | DeFi liquidity-provider strategy: pool selection, impermanent-loss thresholds. |

---

### `tft_training`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Temporal Fusion Transformer training configuration (epochs, learning rate, sequence length). |

---

### `fill_tracker`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Fill-tracking and slippage measurement configuration. |

---

### `maker_enforcement`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Enforcement of maker-only order submission to minimise taker fees. |

---

### `intraday_var`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Intraday VaR calculation parameters (window, confidence, update frequency). |

---

### `stress_tester`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Stress-test scenario definitions (historical crash replay, custom shock scenarios). |

---

### `counterparty_monitor`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Exchange and counterparty health monitoring (API latency thresholds, error-rate limits). |

---

### `funding_cost_limiter`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Maximum allowable funding cost per perpetual position before forced close. |

---

### `tail_hedge`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Tail-risk hedging configuration (put spreads, inverse ETF allocation). |

---

### `liquidation_cascade`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Liquidation cascade detection parameters and avoidance logic. |

---

### `macro_event_calendar`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Macro economic event calendar integration (FOMC, CPI, NFP) for pre-event risk reduction. |

---

### `cross_exchange_arb`
| Attribute | Value |
|-----------|-------|
| Type | `object` |
| Description | Cross-exchange arbitrage detection: minimum spread threshold and execution parameters. |

---

## Required for Live Mode

The following keys must be present and non-default to operate in live trading mode:

| Key | Requirement |
|-----|-------------|
| `capital.starting_capital_aud` | Must be > 0 |
| `capital.currency` | Must be set (e.g., `"AUD"`) |
| `exchanges.primary` | Must be a supported exchange identifier |
| `exchanges.<primary>.maker_fee` | Must be set to actual exchange fee |
| `exchanges.<primary>.taker_fee` | Must be set to actual exchange fee |
| `risk.max_daily_loss_pct` | Must be ≥ 0.10 (validator minimum) |
| `risk.max_drawdown_pct` | Must be ≥ 0.10 (validator minimum) |
| `risk.stop_loss_pct` | Must be > 0 |
| `risk.take_profit_pct` | Must be > 0 |
| `risk.emergency_shutdown.enabled` | Must be `true` |
| `risk.portfolio_var_limit_pct` | Must be ≥ 0.10 (validator minimum) |
| `live_gate.enabled` | Must be `true` to unlock live order placement |
| `trading_pairs` | Must contain at least one valid pair |
| `monitoring.enabled` | Should be `true` for operational visibility |

Additionally, the following environment variables must be set:

| Variable | Description |
|----------|-------------|
| `KRAKEN_API_KEY` | Kraken API key (or equivalent for primary exchange) |
| `KRAKEN_API_SECRET` | Kraken API secret |
| `ARGUS_LIVE_MODE` | Set to `"1"` or `"true"` to enable live order placement |

---

## Required for Paper Mode

Paper trading has relaxed requirements. The following keys must be set:

| Key | Requirement |
|-----|-------------|
| `capital.starting_capital_aud` | Must be > 0 (virtual balance) |
| `paper_trading.enabled` | Must be `true` |
| `paper_trading.virtual_balance_aud` | Initial virtual balance (defaults to `capital.starting_capital_aud`) |
| `trading_pairs` | Must contain at least one valid pair |
| `exchanges.primary` | Must be set (used for market data even in paper mode) |
| `risk.max_daily_loss_pct` | Must be set (enforced in paper mode too) |
| `risk.stop_loss_pct` | Must be > 0 |
| `market_data.ohlcv_poll_interval_seconds` | Must be > 0 |

No API keys with trading permissions are required for paper mode — read-only
market data access is sufficient.
