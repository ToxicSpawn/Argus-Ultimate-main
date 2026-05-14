# Configuration Migration Guide

## Overview

Argus Ultimate ships with two root-level configuration files.
This document explains what each file contains, which is canonical, and how
to migrate from the legacy file to the canonical one.

---

## Files at a Glance

| File | Status | Purpose |
|------|--------|---------|
| `unified_config.yaml` | **Canonical** | Single source of truth for all runtime, risk, exchange, and strategy settings |
| `config.yaml` | **Deprecated** | Legacy "god-mode" / experimental config; preserved for reference only |

---

## `unified_config.yaml` — Canonical Configuration

`unified_config.yaml` is the **authoritative** runtime configuration file.
It is loaded by `core/config_manager.py` and `argus_live/` at startup.

### Top-level sections

| Section | Description |
|---------|-------------|
| `config_version` | Schema version integer. Increment on breaking changes. |
| `run_template` | Target hardware specification (CPU, GPU, RAM, NIC, network topology). |
| `capital` | Starting capital, position size limits, currency, concurrent positions. |
| `fx` | FX rates (e.g., AUD→USD) used for cross-currency sizing. |
| `risk` | Risk controls: drawdown, stop-loss, circuit breakers, VaR limits, auto-reduce. |
| `runtime_safety` | Soak-stability controls (latency grace cycles, live-safe flags). |
| `market_data` | OHLCV polling intervals, retry config, historical pre-load settings. |
| `exchanges` | Primary/secondary exchange selection, fee tiers, CCXT flag. |
| `best_symbols` | Curated high-liquidity symbols for scanner/live use. |
| `trading_pairs` | Full universe of tradeable pairs (Tier 1–3 by liquidity). |
| `execution` | Order routing, slippage, maker enforcement, latency targets. |
| `execution_routing` | Venue selection logic, smart order routing parameters. |
| `continuous_scan` | Scanner interval, top-N selection, regime filter. |
| `liquidity_scanner` | Spread and volume thresholds for liquidity filtering. |
| `edge_cost_gate` | Minimum expected edge required after fees/slippage before entry. |
| `portfolio_target_engine` | Portfolio-level exposure targets and rebalance thresholds. |
| `liquidity_risk_engine` | Liquidity-adjusted position sizing configuration. |
| `execution_alpha_engine` | Smart execution timing and alpha capture settings. |
| `strategy_evaluation_engine` | Real-time strategy scoring and promotion criteria. |
| `self_optimizing_meta_engine` | Meta-learning configuration for parameter self-tuning. |
| `champion_challenger` | A/B strategy testing framework configuration. |
| `market_microstructure_engine` | Order-book and microstructure analysis settings. |
| `recon_recovery_engine` | Reconciliation and position recovery parameters. |
| `system_health_metrics` | Health score thresholds and alerting configuration. |
| `ai_brain` | Pinnacle AI Brain model selection, confidence thresholds, feature list. |
| `strategies` | Per-strategy enable flags, parameter defaults (EMA, RSI, MACD, etc.). |
| `strategy_library` | Extended strategy definitions and composite signal logic. |
| `strategy_router` | Regime-to-strategy routing matrix. |
| `quantum_features` | Quantum-enhanced signal generation flags and qubit budgets. |
| `quantum_simulator` | Quantum simulator backend settings (max qubits, shots, backend type). |
| `quant_fund_upgrades` | Institutional quant features (factor models, alpha decay). |
| `transcendent` | Experimental transcendent/godmode AI features. |
| `argus_strategies` | Argus-specific strategy compositions. |
| `execution_engine` | Execution engine concurrency, order batching, fill tracking. |
| `data` | Data pipeline settings (feed providers, cache TTLs, normalisation). |
| `multi_language` | Polyglot engine configuration (Rust/Go/C++ sub-processes). |
| `monitoring` | Prometheus/Grafana export, alerting webhooks, log levels. |
| `paper_trading` | Paper trading mode: simulated fills, virtual balance, reporting. |
| `runtime` | Boot sequence, hot-reload, graceful shutdown timeouts. |
| `reconciliation` | End-of-day reconciliation rules and tolerance thresholds. |
| `backtest` | Backtesting engine parameters (mode, slippage, data range). |
| `evolution` | Evolutionary optimisation (GA/CMA-ES) settings. |
| `strategy_allocator` | Capital allocation across strategies (Kelly, equal-weight, etc.). |
| `hft` | High-frequency trading micro-optimisation flags. |
| `quantum_bot` | Quantum-enhanced bot integration parameters. |
| `emergency_shutdown` | Hard-stop conditions (flash crash %, latency spike, network fail). |
| `logging` | Log format, rotation, verbosity per module. |
| `targets` | Performance targets (Sharpe, win rate, monthly return). |
| `perp_exchanges` | Perpetual futures exchange configuration. |
| `funding_rate_harvester` | Funding-rate arbitrage strategy settings. |
| `alternative_data` | News, on-chain, sentiment feed configuration. |
| `ml_models` | ML model registry (type, path, feature set, retrain schedule). |
| `advanced_risk` | Advanced risk overlays (CVaR, stress test, tail hedge). |
| `market_making` | Market-making spread and inventory management settings. |
| `stat_arb_cointegration` | Statistical arbitrage cointegration pair parameters. |
| `storage` | Database/storage backend (SQLite, PostgreSQL, Redis) settings. |
| `websocket_orders` | WebSocket order placement configuration. |
| `market_impact` | Market-impact model (Almgren-Chriss) parameters. |
| `algo_execution` | TWAP/VWAP/POV algorithmic execution settings. |
| `backtesting` | (Alias) Extended backtesting settings. |
| `live_gate` | Live-trading gate: readiness checks, required confirmations. |
| `position_registry` | Position registry persistence and reconciliation interval. |
| `mtf_confluence` | Multi-timeframe confluence signal parameters. |
| `portfolio_optimizer` | Mean-variance / Black-Litterman portfolio optimiser settings. |
| `rolling_performance_feeder` | Rolling performance statistics feed to meta-learner. |
| `hot_reload` | Config hot-reload watcher settings. |
| `telegram` | Telegram alert bot configuration. |
| `news_sentiment` | News sentiment NLP model and feed settings. |
| `defi_lp` | DeFi liquidity-provider strategy settings. |
| `tft_training` | Temporal Fusion Transformer training configuration. |
| `fill_tracker` | Fill-tracking and slippage measurement. |
| `maker_enforcement` | Enforcement of maker-only order submission. |
| `intraday_var` | Intraday VaR calculation parameters. |
| `stress_tester` | Stress-test scenario definitions. |
| `counterparty_monitor` | Exchange/counterparty health monitoring. |
| `funding_cost_limiter` | Maximum allowable funding cost per position. |
| `tail_hedge` | Tail-risk hedging (put spreads, inverse ETF) settings. |
| `liquidation_cascade` | Liquidation cascade detection and avoidance. |
| `macro_event_calendar` | Macro event (FOMC, CPI) calendar integration. |
| `cross_exchange_arb` | Cross-exchange arbitrage detection parameters. |

---

## `config.yaml` — Legacy / Deprecated

`config.yaml` was the original monolithic configuration used by early versions
of Argus Ultimate and the `argus_omega` package. It contains:

- `adaptive_system` — market regime detection and position multipliers
- `advanced_backtesting` — backtest modes (event-driven, Monte Carlo, walk-forward)
- `advanced_derivatives` — exotic options and derivatives configuration
- `advanced_features` / `advanced_monitoring` — experimental feature flags
- `ai_consciousness` / `god_consciousness` — legacy "godmode" AI settings
- `alternative_data` — early news/sentiment feed config
- `arbitrage_enhancement` — cross-venue arbitrage parameters
- `consciousness_level_ai` — experimental self-aware AI config
- `continuous_improvement` — online learning settings
- `defi_integration` / `defi_yield_farming` — DeFi strategy configs
- `elite_infrastructure` / `infrastructure_upgrades` — infrastructure flags
- `exchanges` / `exchange_integrations` — exchange connectivity (superseded)
- `evolution` / `functional_execution` — evolutionary and execution params
- `global_markets` — multi-region market access
- `hft_engine` / `high_frequency_trading` — HFT configs (superseded)
- `legendary_strategies` / `high_volatility_strategies` — strategy definitions
- `market_making` — market-making config (superseded)
- `ml_enhancement` / `ml_quantum_hybrid` / `ml_supremacy` — ML configs
- `monitoring` — legacy monitoring (superseded by `unified_config.yaml`)
- `multi_asset` / `multi_exchange` — multi-venue configs
- `options_trading` — options strategy config
- `performance` / `performance_optimization` / `performance_scaling` — perf flags
- `quantum_acceleration` / `quantum_supremacy` / `quantum_system` — quantum configs
- `revolutionary` / `revolutionary_system` — experimental flags
- And many more experimental sections...

> **Do not add new settings to `config.yaml`.** All new configuration must go
> into `unified_config.yaml` under the appropriate section.

---

## Migration Steps

### 1. Identify overrides in `config.yaml`

Scan `config.yaml` for any values that differ from the defaults. These are
your custom overrides that need to be migrated.

### 2. Map sections to `unified_config.yaml`

Use the table below for common mappings:

| `config.yaml` section | `unified_config.yaml` section |
|---|---|
| `exchanges` | `exchanges` |
| `monitoring` | `monitoring` |
| `evolution` | `evolution` |
| `market_making` | `market_making` |
| `alternative_data` | `alternative_data` |
| `hft_engine` | `hft` |
| `performance` | `targets` |
| `quantum_acceleration` | `quantum_features` |

### 3. Update loading code

Replace any code that loads `config.yaml` directly:

```python
# Old
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# New
from core.config_manager import ConfigManager
cfg = ConfigManager.load()  # reads unified_config.yaml
```

### 4. Validate

Run `python -m config.validated_config` to validate your `unified_config.yaml`
against the schema defined in `config/schema.py`.

---

## Canonical Config Load Order

At runtime, configuration is resolved in the following priority order
(highest priority first):

1. Environment variables (prefixed `ARGUS_`)
2. `config/runtime/*.yaml` overlays
3. `config/constitution/*.yaml` constraints
4. `unified_config.yaml` (base)

`config.yaml` is **not** loaded by the runtime and exists solely for
historical reference and manual migration.
