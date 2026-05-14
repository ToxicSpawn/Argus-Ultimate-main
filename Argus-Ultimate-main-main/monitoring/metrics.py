"""monitoring/metrics.py

Argus Ultimate — Prometheus metrics registry (Push 83 expanded).

All metrics used across Argus are defined here as module-level singletons.
Import and use directly:

    from monitoring.metrics import METRICS
    METRICS.strategy_pnl.labels(strategy="momentum").set(1234.56)
    METRICS.order_latency.labels(strategy="momentum").observe(0.003)

Metric naming follows:
    argus_{subsystem}_{name}_{unit}

Subsystems:
    strategy    — per-strategy performance
    orderbook   — live order book state (push 82)
    orders      — order execution
    signals     — signal generation
    risk        — risk manager
    ws          — WebSocket feed
    kline       — kline buffer
    system      — system health
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Info,
        CollectorRegistry, REGISTRY,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    log.warning("prometheus_client not installed — metrics are no-ops")


LATENCY_BUCKETS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


class _NoOpMetric:
    """Silent no-op for when prometheus_client is not installed."""
    def labels(self, **_): return self
    def inc(self, *_, **__): pass
    def dec(self, *_, **__): pass
    def set(self, *_, **__): pass
    def observe(self, *_, **__): pass
    def info(self, *_, **__): pass


def _gauge(name: str, doc: str, labels=()) -> object:
    if not _HAS_PROMETHEUS:
        return _NoOpMetric()
    try:
        return Gauge(name, doc, labels)
    except Exception:
        return REGISTRY._names_to_collectors.get(name, _NoOpMetric())


def _counter(name: str, doc: str, labels=()) -> object:
    if not _HAS_PROMETHEUS:
        return _NoOpMetric()
    try:
        return Counter(name, doc, labels)
    except Exception:
        return REGISTRY._names_to_collectors.get(name, _NoOpMetric())


def _histogram(name: str, doc: str, labels=(), buckets=LATENCY_BUCKETS) -> object:
    if not _HAS_PROMETHEUS:
        return _NoOpMetric()
    try:
        return Histogram(name, doc, labels, buckets=buckets)
    except Exception:
        return REGISTRY._names_to_collectors.get(name, _NoOpMetric())


@dataclass
class ArgusMetrics:
    # ─────────────────────────────────────────────────────────────
    # Strategy metrics
    # ─────────────────────────────────────────────────────────────
    strategy_active: object = field(default_factory=lambda: _gauge(
        "argus_strategy_active", "1 if strategy is running, 0 if stopped", ["strategy"]
    ))
    strategy_pnl: object = field(default_factory=lambda: _gauge(
        "argus_strategy_pnl_total", "Total realised + unrealised PnL (USD)", ["strategy"]
    ))
    strategy_pnl_realised: object = field(default_factory=lambda: _counter(
        "argus_strategy_pnl_realised_total", "Cumulative realised PnL (USD)", ["strategy"]
    ))
    strategy_equity: object = field(default_factory=lambda: _gauge(
        "argus_strategy_equity", "Current equity curve value (USD)", ["strategy"]
    ))
    strategy_sharpe: object = field(default_factory=lambda: _gauge(
        "argus_strategy_sharpe_ratio", "Rolling Sharpe ratio (24h)", ["strategy"]
    ))
    strategy_sortino: object = field(default_factory=lambda: _gauge(
        "argus_strategy_sortino_ratio", "Rolling Sortino ratio (24h)", ["strategy"]
    ))
    strategy_calmar: object = field(default_factory=lambda: _gauge(
        "argus_strategy_calmar_ratio", "Rolling Calmar ratio (24h)", ["strategy"]
    ))
    strategy_max_drawdown: object = field(default_factory=lambda: _gauge(
        "argus_strategy_max_drawdown", "Maximum drawdown (fraction, negative)", ["strategy"]
    ))
    strategy_drawdown_current: object = field(default_factory=lambda: _gauge(
        "argus_strategy_drawdown_current", "Current drawdown from peak (fraction)", ["strategy"]
    ))
    strategy_win_rate: object = field(default_factory=lambda: _gauge(
        "argus_strategy_win_rate", "Win rate (fraction 0-1)", ["strategy"]
    ))
    strategy_profit_factor: object = field(default_factory=lambda: _gauge(
        "argus_strategy_profit_factor", "Profit factor (gross profit / gross loss)", ["strategy"]
    ))
    strategy_trades: object = field(default_factory=lambda: _counter(
        "argus_strategy_trades_total", "Total trades executed", ["strategy", "side"]
    ))

    # ─────────────────────────────────────────────────────────────
    # Signal metrics
    # ─────────────────────────────────────────────────────────────
    signals_total: object = field(default_factory=lambda: _counter(
        "argus_signals_total", "Total signals generated", ["strategy", "direction"]
    ))
    signal_confidence: object = field(default_factory=lambda: _gauge(
        "argus_signal_confidence", "Last signal confidence (0-1)", ["strategy"]
    ))
    signal_strength: object = field(default_factory=lambda: _gauge(
        "argus_signal_strength", "Last signal strength (0-1)", ["strategy"]
    ))

    # ─────────────────────────────────────────────────────────────
    # Order execution metrics
    # ─────────────────────────────────────────────────────────────
    orders_total: object = field(default_factory=lambda: _counter(
        "argus_orders_total", "Total orders placed", ["strategy", "side", "type"]
    ))
    order_latency: object = field(default_factory=lambda: _histogram(
        "argus_order_latency_seconds", "Order round-trip latency (seconds)", ["strategy"]
    ))
    order_slippage: object = field(default_factory=lambda: _gauge(
        "argus_order_slippage", "Order slippage (fraction)", ["strategy"]
    ))
    order_fill_rate: object = field(default_factory=lambda: _gauge(
        "argus_order_fill_rate", "Fill rate (filled / placed)", ["strategy"]
    ))

    # ─────────────────────────────────────────────────────────────
    # Order book metrics (Push 82 — BinanceOrderBook)
    # ─────────────────────────────────────────────────────────────
    ob_best_bid: object = field(default_factory=lambda: _gauge(
        "argus_orderbook_best_bid", "Best bid price", ["symbol"]
    ))
    ob_best_ask: object = field(default_factory=lambda: _gauge(
        "argus_orderbook_best_ask", "Best ask price", ["symbol"]
    ))
    ob_mid_price: object = field(default_factory=lambda: _gauge(
        "argus_orderbook_mid_price", "Mid price ((bid+ask)/2)", ["symbol"]
    ))
    ob_spread_bps: object = field(default_factory=lambda: _gauge(
        "argus_orderbook_spread_bps", "Bid-ask spread in basis points", ["symbol"]
    ))
    ob_imbalance: object = field(default_factory=lambda: _gauge(
        "argus_orderbook_imbalance", "Order book imbalance [-1, 1]", ["symbol"]
    ))

    # ─────────────────────────────────────────────────────────────
    # WebSocket feed metrics
    # ─────────────────────────────────────────────────────────────
    ws_messages: object = field(default_factory=lambda: _counter(
        "argus_ws_messages_total", "Total WebSocket messages received", ["stream"]
    ))
    ws_reconnects: object = field(default_factory=lambda: _counter(
        "argus_ws_reconnects_total", "Total WebSocket reconnections", ["stream"]
    ))
    ws_errors: object = field(default_factory=lambda: _counter(
        "argus_ws_errors_total", "Total WebSocket errors", ["stream"]
    ))
    ws_latency: object = field(default_factory=lambda: _histogram(
        "argus_ws_message_latency_seconds", "WS message processing latency", ["stream"]
    ))

    # ─────────────────────────────────────────────────────────────
    # Kline buffer metrics
    # ─────────────────────────────────────────────────────────────
    kline_buffer_size: object = field(default_factory=lambda: _gauge(
        "argus_kline_buffer_size", "Number of candles in KlineBuffer", ["symbol", "interval"]
    ))
    trade_volume: object = field(default_factory=lambda: _counter(
        "argus_trade_volume_total", "Cumulative trade volume (base asset)", ["symbol"]
    ))

    # ─────────────────────────────────────────────────────────────
    # Risk metrics
    # ─────────────────────────────────────────────────────────────
    risk_portfolio_heat: object = field(default_factory=lambda: _gauge(
        "argus_risk_portfolio_heat", "Portfolio heat (total notional exposure / equity)"
    ))
    risk_daily_loss: object = field(default_factory=lambda: _gauge(
        "argus_risk_daily_loss", "Current daily loss (USD)"
    ))
    risk_kill_switch: object = field(default_factory=lambda: _gauge(
        "argus_risk_kill_switch_active", "1 if kill switch is active"
    ))
    risk_orders_blocked: object = field(default_factory=lambda: _counter(
        "argus_risk_orders_blocked_total", "Orders blocked by risk manager", ["reason"]
    ))

    # ─────────────────────────────────────────────────────────────
    # System health
    # ─────────────────────────────────────────────────────────────
    system_uptime: object = field(default_factory=lambda: _gauge(
        "argus_system_uptime_seconds", "System uptime in seconds"
    ))
    system_tick_rate: object = field(default_factory=lambda: _gauge(
        "argus_system_tick_rate", "Ticks processed per second"
    ))
    system_errors: object = field(default_factory=lambda: _counter(
        "argus_system_errors_total", "Total system errors", ["component"]
    ))


# Module-level singleton
METRICS = ArgusMetrics()


def update_orderbook_metrics(symbol: str, book) -> None:
    """Convenience: update all order book metrics from a BinanceOrderBook instance."""
    if book is None or not book.is_ready:
        return
    sym = symbol.upper()
    if book.best_bid is not None:
        METRICS.ob_best_bid.labels(symbol=sym).set(book.best_bid)
    if book.best_ask is not None:
        METRICS.ob_best_ask.labels(symbol=sym).set(book.best_ask)
    if book.mid_price is not None:
        METRICS.ob_mid_price.labels(symbol=sym).set(book.mid_price)
    if book.spread_bps is not None:
        METRICS.ob_spread_bps.labels(symbol=sym).set(book.spread_bps)
    imb = book.imbalance()
    if imb is not None:
        METRICS.ob_imbalance.labels(symbol=sym).set(imb)


def update_strategy_metrics(strategy: str, stats: dict) -> None:
    """Convenience: update strategy metrics from a stats dict."""
    METRICS.strategy_active.labels(strategy=strategy).set(1 if stats.get("active") else 0)
    if (pnl := stats.get("pnl_total")) is not None:
        METRICS.strategy_pnl.labels(strategy=strategy).set(pnl)
    if (equity := stats.get("equity")) is not None:
        METRICS.strategy_equity.labels(strategy=strategy).set(equity)
    if (sharpe := stats.get("sharpe")) is not None:
        METRICS.strategy_sharpe.labels(strategy=strategy).set(sharpe)
    if (sortino := stats.get("sortino")) is not None:
        METRICS.strategy_sortino.labels(strategy=strategy).set(sortino)
    if (calmar := stats.get("calmar")) is not None:
        METRICS.strategy_calmar.labels(strategy=strategy).set(calmar)
    if (dd := stats.get("max_drawdown")) is not None:
        METRICS.strategy_max_drawdown.labels(strategy=strategy).set(dd)
    if (dd_cur := stats.get("drawdown_current")) is not None:
        METRICS.strategy_drawdown_current.labels(strategy=strategy).set(dd_cur)
    if (wr := stats.get("win_rate")) is not None:
        METRICS.strategy_win_rate.labels(strategy=strategy).set(wr)
    if (pf := stats.get("profit_factor")) is not None:
        METRICS.strategy_profit_factor.labels(strategy=strategy).set(pf)
