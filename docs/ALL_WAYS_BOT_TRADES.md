# All Ways This Bot Can Trade

A single reference for every mode, source, strategy, execution path, and exit the bot uses.

---

## 1. Run modes (how you start the bot)

| Mode | Command | What it does |
|------|---------|--------------|
| **Paper** | `python main.py paper` | Full loop, no real money; exchange in dry-run; optional `--paper-days 30` and `--cycle-seconds 0`. |
| **Live** | `python main.py live` | Same loop, real orders; requires Kraken (and optional Coinbase) credentials. |
| **Backtest** | `python main.py backtest --symbol BTC/USD --days 30 --unified-backtest --csv data/ohlcv.csv` | Historical run on OHLCV CSV; no real orders. |

Paper can be set to **simulate live** (`paper_trading.simulate_live: true`): same config as live, same disabled strategies, simulated slippage on fills.

---

## 2. Signal sources (where trades come from)

Each cycle, the **Continuous Best-Trade Scanner** gathers candidates in parallel from:

| Source | Description | Config / note |
|--------|-------------|----------------|
| **AI brain** | Pinnacle AI (strategy engine + agents) | `ai_brain.*`; produces `unified_engine` and other strategy tags. |
| **Strategy engine** | RSI/BB/MACD + regime + tuner; emits multiple strategies per symbol. | `strategies.unified.strategy_engine`; same OHLCV/regime as AI path. |
| **HFT engine** | Scalping / order-book pressure (e.g. OBI). | `hft_engine`; can be disabled in paper via `paper_trading_disabled_strategies`. |

Only signals whose **strategy** is in **strategy_whitelist** are kept. Then **disabled strategies** (paper vs live list) are filtered out. Optional **regime filter** restricts trend strategies to trend regime and mean-reversion to range.

---

## 3. Strategies that can emit signals (whitelist)

These are the strategy tags that are allowed to produce trades (from config):

| Strategy | Type | When it fires |
|----------|------|----------------|
| **akashic_tier** | Tier / library | From strategy pack. |
| **unified_engine** | Core | RSI/BB/MACD + regime (mean-reversion or trend). |
| **quantum_momentum_elite** | Quantum / library | Momentum-style. |
| **regime_aligned** | Unified | Same direction as regime (BUY in RANGE/TREND_UP, SELL in RANGE/TREND_DOWN). |
| **volume_momentum** | Unified | EMA cross + volume above average. |
| **mean_reversion** | Unified | RSI oversold/overbought in RANGE only. |
| **breakout_bb** | Unified | Price at Bollinger extreme + volume confirmation. |
| **macd_trend** | Unified | TREND_UP + MACD rising / TREND_DOWN + MACD falling. |

Other strategies (e.g. from strategy library or HFT) only contribute if their name is in **strategy_whitelist**. **Regime filter** (if enabled) further restricts trend vs mean-reversion strategies by current regime.

---

## 4. Execution paths (how orders are sent)

| Path | When it’s used | Config |
|------|----------------|--------|
| **Single venue (primary)** | Default: one order on primary exchange (e.g. Kraken). | `primary_exchange` |
| **Multi-venue split** | When **multi_venue_enabled** and order notional ≥ **multi_venue_min_notional_aud**; split across primary + secondary (e.g. Kraken + Coinbase Advanced). | `execution_engine.multi_venue_*` |
| **Single market/limit order** | Normal path: one `create_order` (market or limit). | `execution_engine.order_type`: `market` or `limit` |
| **VWAP (sliced)** | When order size ≥ **vwap_large_order_threshold_aud**; order split into slices over time (VWAP schedule). | `vwap_large_order_threshold_aud`, `vwap_duration_sec`, `vwap_num_slices` |
| **TWAP (sliced)** | Same size threshold, but **use_twap_for_large_orders: true**; equal slices over time. | `use_twap_for_large_orders` |

So the bot can trade via: **single exchange (market/limit)**, **two exchanges (split)**, and **single exchange with VWAP or TWAP** for large size.

---

## 5. Order types

| Type | Meaning |
|------|--------|
| **Market** | Fill at best available price (default; `order_type: market`). |
| **Limit** | Place at a specific price (`order_type: limit`). |

Slippage and fees are applied in sizing and edge-cost gate; in paper with **simulate_live**, fill price is adjusted by **slippage_pct** to mimic live drag.

---

## 6. Exchanges / venues

| Venue | Role | Config |
|-------|------|--------|
| **Kraken** | Primary (CCXT); paper uses dry-run wrapper. | `exchanges.primary`, `KRAKEN_API_KEY` / `KRAKEN_SECRET` |
| **Coinbase Advanced** | Secondary for multi-venue and optional routing. | `exchanges.secondary`, Coinbase credentials |

Both are optional; live requires at least Kraken credentials for real orders.

---

## 7. Exits and risk (per-trade and global)

| Mechanism | Description | Config |
|-----------|-------------|--------|
| **Stop loss** | Exit long when price drops by **stop_loss_pct**; exit short when price rises. | `risk.stop_loss_pct`, paper overrides |
| **Take profit** | Exit at **take_profit_pct** (e.g. 6% for longs). | `risk.take_profit_pct` |
| **Partial take profit** | Close a fraction of position at **partial_tp_at_pct** (e.g. 3%); **partial_tp_close_pct** (e.g. 50%). | `paper_trading.partial_tp_*` |
| **Trailing stop** | Trail stop at **trailing_stop_pct** below high-water mark. | `paper_trading.trailing_stop_enabled`, `trailing_stop_pct` |
| **Circuit breaker** | Halt or reduce trading when drawdown or other breach. | `risk.circuit_breaker_*`, `auto_reduce_*` |
| **Daily loss / drawdown** | Hard limits: **max_daily_loss_pct**, **max_drawdown_pct**. | `risk.*` |

So the bot can exit via: **fixed stop**, **fixed target**, **partial TP**, **trailing stop**, and **circuit breaker / daily/drawdown limits**.

---

## 8. Pre-trade gates (what can block a trade)

- **Edge-cost gate**: Expected edge must exceed fees + slippage by a multiple (**min_edge_pct**, **buffer_mult**); can apply in paper/backtest/live.
- **Strategy whitelist**: Only listed strategies can emit; then **disabled** list (paper or live) is applied.
- **Regime filter**: Trend strategies only in trend regime; mean-reversion only in range (if enabled).
- **Max concurrent signals / positions**: **max_concurrent_signals** (e.g. 2), **max_concurrent_positions** (e.g. 4, cap 6).
- **Slippage guardrail**: Execution engine can reject if realized slippage > **max_slippage_pct**.
- **23-language risk / drawdown / slippage**: Optional multi-language risk gate, drawdown check, slippage estimate; can skip execution if they fail.
- **Emergency stop / exposure**: Pre-trade checks (e.g. max exposure, position limits) before sending order.

---

## 9. Summary table

| Dimension | Options |
|-----------|--------|
| **Run** | Paper, live, backtest (and paper “simulate live”) |
| **Signal sources** | AI brain (strategy engine), strategy engine directly, HFT engine |
| **Strategies (whitelist)** | akashic_tier, unified_engine, quantum_momentum_elite, regime_aligned, volume_momentum, mean_reversion, breakout_bb, macd_trend (+ any you add) |
| **Execution** | Single venue (market/limit), multi-venue split, VWAP/TWAP for large orders |
| **Venues** | Kraken (primary), Coinbase Advanced (secondary) |
| **Exits** | Stop loss, take profit, partial TP, trailing stop, circuit breaker, daily/drawdown limits |

All of these together define “all the ways this bot can trade.”
