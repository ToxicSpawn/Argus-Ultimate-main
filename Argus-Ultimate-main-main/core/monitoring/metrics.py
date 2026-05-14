"""Push 68 — Prometheus metrics registry for Argus.

All metrics exported to /metrics endpoint via prometheus_client.
Falls back to no-op stubs if prometheus_client not installed.

Metrics:
  argus_risk_halted          Gauge   1=halted 0=active
  argus_p99_fill_latency_ms  Histogram fill latency percentile
  argus_daily_pnl_usd        Gauge   session PnL in USD
  argus_position_count       Gauge   open positions
  argus_signal_count_total   Counter signals emitted
  argus_funding_collected_usd Gauge  funding arb collected
  argus_cvar_95              Gauge   CVaR at 95% confidence
  argus_cvar_99              Gauge   CVaR at 99% confidence
  argus_kelly_fraction       Gauge   current Kelly fraction
  argus_order_refresh_total  Counter order refresh cycles
  argus_drawdown_pct         Gauge   current peak-to-trough %
  argus_equity_usd           Gauge   current total equity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class _NoOpGauge:
    def set(self, v): pass
    def inc(self, v=1): pass
    def dec(self, v=1): pass
    def labels(self, **kw): return self

class _NoOpCounter:
    def inc(self, v=1): pass
    def labels(self, **kw): return self

class _NoOpHistogram:
    def observe(self, v): pass
    def labels(self, **kw): return self
    def time(self): return _NoOpTimer()

class _NoOpTimer:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def _try_prometheus():
    try:
        from prometheus_client import Gauge, Counter, Histogram, start_http_server
        return Gauge, Counter, Histogram, start_http_server
    except ImportError:
        return None, None, None, None


class ArgusMetrics:
    """Central Prometheus metrics registry for Argus.

    Args:
        port: HTTP port to expose /metrics on (default 8000)
        namespace: Prometheus metric namespace prefix
    """

    def __init__(self, port: int = 8000, namespace: str = "argus"):
        self.port = port
        self.namespace = namespace
        self._server_started = False
        Gauge, Counter, Histogram, self._start_http = _try_prometheus()
        self._prom_available = Gauge is not None

        def G(name, doc, labels=None):
            if self._prom_available:
                kwargs = {"namespace": namespace}
                if labels:
                    return Gauge(name, doc, labels, **kwargs)
                return Gauge(name, doc, **kwargs)
            return _NoOpGauge()

        def C(name, doc, labels=None):
            if self._prom_available:
                kwargs = {"namespace": namespace}
                if labels:
                    return Counter(name, doc, labels, **kwargs)
                return Counter(name, doc, **kwargs)
            return _NoOpCounter()

        def H(name, doc, buckets=None):
            if self._prom_available:
                kwargs = {"namespace": namespace}
                if buckets:
                    kwargs["buckets"] = buckets
                return Histogram(name, doc, **kwargs)
            return _NoOpHistogram()

        # Core trading metrics
        self.risk_halted       = G("risk_halted", "1 if risk system halted trading")
        self.daily_pnl         = G("daily_pnl_usd", "Session PnL in USD")
        self.equity            = G("equity_usd", "Current total equity USD")
        self.position_count    = G("position_count", "Number of open positions")
        self.drawdown_pct      = G("drawdown_pct", "Current peak-to-trough drawdown %")

        # Signal / execution metrics
        self.signal_count      = C("signal_count_total", "Total RL signals emitted",
                                   labels=["algorithm", "side"])
        self.order_refresh     = C("order_refresh_total", "Total order refresh cycles")
        self.fill_latency      = H("fill_latency_ms",
                                   "Fill latency distribution in milliseconds",
                                   buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000])

        # Risk metrics
        self.cvar_95           = G("cvar_95", "CVaR at 95% confidence")
        self.cvar_99           = G("cvar_99", "CVaR at 99% confidence")
        self.kelly_fraction    = G("kelly_fraction", "Current Kelly position fraction")
        self.funding_collected = G("funding_collected_usd", "Total funding arb collected USD")

        # Per-strategy labels
        self.strategy_pnl      = G("strategy_pnl_usd", "PnL per strategy",
                                   labels=["strategy"])
        self.strategy_halted   = G("strategy_halted", "1 if strategy halted",
                                   labels=["strategy"])

    def start_server(self) -> bool:
        """Start the Prometheus HTTP metrics server."""
        if self._prom_available and not self._server_started:
            try:
                self._start_http(self.port)
                self._server_started = True
                return True
            except Exception:
                pass
        return False

    def record_fill(self, latency_ms: float, algorithm: str = "PPO",
                    side: str = "buy") -> None:
        self.fill_latency.observe(latency_ms)
        self.signal_count.labels(algorithm=algorithm, side=side).inc()

    def update_risk_snapshot(self, halted: bool, drawdown_pct: float,
                              cvar_95: float, cvar_99: float,
                              daily_pnl: float, equity: float,
                              position_count: int) -> None:
        self.risk_halted.set(1 if halted else 0)
        self.drawdown_pct.set(drawdown_pct)
        self.cvar_95.set(cvar_95)
        self.cvar_99.set(cvar_99)
        self.daily_pnl.set(daily_pnl)
        self.equity.set(equity)
        self.position_count.set(position_count)

    def update_kelly(self, fraction: float) -> None:
        self.kelly_fraction.set(fraction)

    def update_funding(self, total_usd: float) -> None:
        self.funding_collected.set(total_usd)

    def record_refresh(self) -> None:
        self.order_refresh.inc()

    def update_strategy(self, name: str, pnl: float, halted: bool) -> None:
        self.strategy_pnl.labels(strategy=name).set(pnl)
        self.strategy_halted.labels(strategy=name).set(1 if halted else 0)
