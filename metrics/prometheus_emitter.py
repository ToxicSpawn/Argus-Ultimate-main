"""PrometheusEmitter — Push 46.

Emits Prometheus /metrics for Argus Ultimate.

Metrics
-------
  argus_bars_total              Counter    bars processed
  argus_bar_duration_seconds    Histogram  bar processing latency
  argus_trades_total            Counter    trades executed
  argus_trade_notional_usd      Gauge      last trade notional
  argus_equity_usd              Gauge      current equity
  argus_position                Gauge      current position fraction
  argus_session_pnl             Gauge      session P&L
  argus_drawdown_pct            Gauge      current drawdown %
  argus_regime_scalar           Gauge      regime scalar (1.3/1.0/0.6)
  argus_errors_total            Counter    errors by kind label
  argus_mtf_aggregate_bias      Gauge      MTF aggregate bias (Push 39)
  argus_mtf_confidence          Gauge      MTF confidence (Push 39)
  argus_mtf_direction           Gauge      MTF direction encoded: long=1, flat=0, short=-1
  argus_regime_label            Gauge      HMM regime: bull=1, sideways=0, bear=-1 (Push 46)
  argus_regime_bull_prob        Gauge      HMM bull state probability (Push 46)
  argus_regime_sideways_prob    Gauge      HMM sideways state probability (Push 46)
  argus_regime_bear_prob        Gauge      HMM bear state probability (Push 46)
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8001

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, start_http_server, REGISTRY,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class PrometheusEmitter:
    """Thin wrapper around prometheus_client metrics.

    Gracefully degrades (no-ops) if prometheus_client is not installed.

    Parameters
    ----------
    port        : HTTP port for /metrics endpoint (default 8001,
                  overrides via ARGUS_METRICS_PORT env var)
    exchange    : Exchange label value
    symbol      : Symbol label value
    matrix_mode : Matrix mode label value
    version     : Bot version string (e.g. 'push46')
    """

    def __init__(
        self,
        port: Optional[int] = None,
        exchange: str = "kraken",
        symbol: str = "XBT/USD",
        matrix_mode: str = "WEIGHTED",
        version: str = "unknown",
    ) -> None:
        self._port = port or int(os.environ.get("ARGUS_METRICS_PORT", _DEFAULT_PORT))
        self._labels = {
            "exchange":    exchange,
            "symbol":      symbol,
            "matrix_mode": matrix_mode,
            "version":     version,
        }
        self._started = False
        self._bar_start: float = 0.0

        if not _HAS_PROMETHEUS:
            logger.warning("prometheus_client not installed — metrics disabled")
            return

        lv = list(self._labels.keys())

        self._bars_total = Counter(
            "argus_bars_total", "Total bars processed", lv)
        self._bar_duration = Histogram(
            "argus_bar_duration_seconds", "Bar processing latency", lv,
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
        self._trades_total = Counter(
            "argus_trades_total", "Total trades executed", lv)
        self._trade_notional = Gauge(
            "argus_trade_notional_usd", "Last trade notional USD", lv)
        self._equity = Gauge(
            "argus_equity_usd", "Current equity USD", lv)
        self._position = Gauge(
            "argus_position", "Current position fraction", lv)
        self._session_pnl = Gauge(
            "argus_session_pnl", "Session P&L", lv)
        self._drawdown_pct = Gauge(
            "argus_drawdown_pct", "Current drawdown %", lv)
        self._regime_scalar = Gauge(
            "argus_regime_scalar", "Regime scalar (1.3=bull/1.0=sideways/0.6=bear)", lv)
        self._errors_total = Counter(
            "argus_errors_total", "Errors by kind", lv + ["kind"])

        # Push 39: MTF metrics
        self._mtf_bias = Gauge(
            "argus_mtf_aggregate_bias", "MTF aggregate bias [-1, 1]", lv)
        self._mtf_confidence = Gauge(
            "argus_mtf_confidence", "MTF confidence [0, 1]", lv)
        self._mtf_direction = Gauge(
            "argus_mtf_direction",
            "MTF direction encoded: long=1, flat=0, short=-1", lv)

        # Push 46: HMM regime metrics
        self._regime_label = Gauge(
            "argus_regime_label",
            "HMM regime: bull=1, sideways=0, bear=-1", lv)
        self._regime_bull_prob = Gauge(
            "argus_regime_bull_prob", "HMM bull state probability [0,1]", lv)
        self._regime_sideways_prob = Gauge(
            "argus_regime_sideways_prob", "HMM sideways state probability [0,1]", lv)
        self._regime_bear_prob = Gauge(
            "argus_regime_bear_prob", "HMM bear state probability [0,1]", lv)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_http_server(self) -> None:
        if not _HAS_PROMETHEUS or self._started:
            return
        try:
            start_http_server(self._port)
            self._started = True
            logger.info("Prometheus metrics server started on port %d", self._port)
        except Exception as exc:
            logger.warning("Could not start Prometheus server: %s", exc)

    # ------------------------------------------------------------------
    # Instrumentation helpers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def bar_timer(self):
        if not _HAS_PROMETHEUS:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            lv = list(self._labels.values())
            self._bars_total.labels(*lv).inc()
            self._bar_duration.labels(*lv).observe(elapsed)

    def inc_trade(self, notional: float = 0.0) -> None:
        if not _HAS_PROMETHEUS:
            return
        lv = list(self._labels.values())
        self._trades_total.labels(*lv).inc()
        self._trade_notional.labels(*lv).set(notional)

    def inc_error(self, kind: str = "unknown") -> None:
        if not _HAS_PROMETHEUS:
            return
        lv = list(self._labels.values())
        self._errors_total.labels(*lv, kind).inc()

    def set_regime_scalar(self, scalar: float) -> None:
        if not _HAS_PROMETHEUS:
            return
        self._regime_scalar.labels(*list(self._labels.values())).set(scalar)

    def emit_state(
        self,
        equity: float,
        position: float,
        session_pnl: float,
        drawdown_pct: float,
        regime_scalar: float,
    ) -> None:
        if not _HAS_PROMETHEUS:
            return
        lv = list(self._labels.values())
        self._equity.labels(*lv).set(equity)
        self._position.labels(*lv).set(position)
        self._session_pnl.labels(*lv).set(session_pnl)
        self._drawdown_pct.labels(*lv).set(drawdown_pct)
        self._regime_scalar.labels(*lv).set(regime_scalar)

    def emit_mtf(
        self,
        bias: float,
        direction: str,
        confidence: float,
    ) -> None:
        """Emit MTF metrics (Push 39).

        Parameters
        ----------
        bias        : aggregate_bias in [-1, 1]
        direction   : 'long', 'flat', or 'short'
        confidence  : confidence in [0, 1]
        """
        if not _HAS_PROMETHEUS:
            return
        lv = list(self._labels.values())
        dir_encoded = 1.0 if direction == "long" else (-1.0 if direction == "short" else 0.0)
        self._mtf_bias.labels(*lv).set(bias)
        self._mtf_confidence.labels(*lv).set(confidence)
        self._mtf_direction.labels(*lv).set(dir_encoded)

    def emit_regime(
        self,
        label: str,
        probs: "np.ndarray",
        scalar: float,
    ) -> None:
        """Emit HMM regime metrics (Push 46).

        Parameters
        ----------
        label  : 'bull', 'sideways', or 'bear'
        probs  : np.ndarray shape (3,) — [bull_prob, sideways_prob, bear_prob]
        scalar : regime scalar (1.3 / 1.0 / 0.6)
        """
        if not _HAS_PROMETHEUS:
            return
        lv = list(self._labels.values())
        label_encoded = 1.0 if label == "bull" else (-1.0 if label == "bear" else 0.0)
        self._regime_label.labels(*lv).set(label_encoded)
        self._regime_scalar.labels(*lv).set(scalar)
        if probs is not None and len(probs) == 3:
            self._regime_bull_prob.labels(*lv).set(float(probs[0]))
            self._regime_sideways_prob.labels(*lv).set(float(probs[1]))
            self._regime_bear_prob.labels(*lv).set(float(probs[2]))
