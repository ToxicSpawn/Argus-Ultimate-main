# Argus Ultimate — Grafana Dashboards

> **Push 83** | Auto-provisioned | Refresh: 2–5s

## Dashboards

| File | UID | Title | Refresh |
|---|---|---|---|
| `dashboards/strategy_performance.json` | `argus-strategy-perf-83` | Argus — Strategy Performance | 5s |
| `dashboards/order_book_live.json` | `argus-orderbook-83` | Argus — Order Book Live | 2s |
| `argus_dashboard.json` | `argus-main` | Argus — Main (push 79) | 10s |

## How it works

Grafana auto-provisions dashboards on startup via:
```
grafana/provisioning/dashboards/dashboards.yaml   ← dashboard loader
grafana/provisioning/datasources/prometheus.yaml  ← Prometheus datasource
```

The `docker-compose.yml` mounts these under `/etc/grafana/provisioning/`.

## Strategy Performance Dashboard

**Panels (6 rows, 20+ panels):**

### Row 1 — Strategy Overview (stat panels)
- Active Strategies, Total PnL, Max Drawdown, Sharpe Ratio, Win Rate, Profit Factor

### Row 2 — Equity & PnL
- Equity curve per strategy (timeseries, smooth line + fill)
- PnL per trade realised (bar chart)

### Row 3 — Risk & Drawdown
- Drawdown over time (filled red area, inverted)
- Sharpe / Sortino / Calmar rolling 24h (multi-line)

### Row 4 — Signal Activity
- Signals generated per minute (bar chart per direction)
- Signal confidence rolling average (0–1 bounded)

### Row 5 — Order Execution
- Orders placed per minute
- Order latency p50/p95/p99 (histogram quantiles)
- Slippage rate

### Row 6 — Live Feed (Push 82)
- Live price from Binance mid (real-time)
- Spread in bps (colour-coded: green/yellow/red)
- Order book imbalance gauge
- WS messages/sec + reconnects
- KlineBuffer fill progress

### Strategy Scorecard Table
- All strategies side-by-side: PnL, Sharpe, Win Rate, Drawdown, Profit Factor, Trades
- Colour-coded cells, sorted by Sharpe descending

## Order Book Live Dashboard

- Bid / Ask / Mid price (2s refresh, colour-coded lines)
- Spread bps gauge
- Book imbalance gauge
- Imbalance history
- Trade volume per minute

## Template Variables

| Variable | Source | Purpose |
|---|---|---|
| `$datasource` | Prometheus | Datasource selector |
| `$strategy` | `label_values(argus_strategy_active, strategy)` | Filter by strategy |
| `$symbol` | `label_values(argus_orderbook_mid_price, symbol)` | Filter by trading pair |

## Prometheus Metrics (Push 83)

All metrics defined in `monitoring/metrics.py` — `METRICS` singleton.

```python
# Strategy
argus_strategy_active{strategy}
argus_strategy_pnl_total{strategy}
argus_strategy_equity{strategy}
argus_strategy_sharpe_ratio{strategy}
argus_strategy_sortino_ratio{strategy}
argus_strategy_calmar_ratio{strategy}
argus_strategy_max_drawdown{strategy}
argus_strategy_drawdown_current{strategy}
argus_strategy_win_rate{strategy}
argus_strategy_profit_factor{strategy}
argus_strategy_trades_total{strategy, side}

# Signals
argus_signals_total{strategy, direction}
argus_signal_confidence{strategy}

# Orders
argus_orders_total{strategy, side, type}
argus_order_latency_seconds{strategy}  (histogram)
argus_order_slippage{strategy}

# Order Book (Push 82)
argus_orderbook_best_bid{symbol}
argus_orderbook_best_ask{symbol}
argus_orderbook_mid_price{symbol}
argus_orderbook_spread_bps{symbol}
argus_orderbook_imbalance{symbol}

# WebSocket
argus_ws_messages_total{stream}
argus_ws_reconnects_total{stream}
argus_ws_errors_total{stream}

# Kline
argus_kline_buffer_size{symbol, interval}
argus_trade_volume_total{symbol}

# Risk
argus_risk_portfolio_heat
argus_risk_daily_loss
argus_risk_kill_switch_active
argus_risk_orders_blocked_total{reason}
```

## Quick Start

```bash
# Full stack (includes Grafana + Prometheus)
docker-compose up -d

# Open Grafana
open http://localhost:3000
# Default: admin / argus-admin

# Strategy Performance dashboard
open http://localhost:3000/d/argus-strategy-perf-83

# Order Book Live dashboard
open http://localhost:3000/d/argus-orderbook-83
```
