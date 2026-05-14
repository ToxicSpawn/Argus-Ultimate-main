"""
Tests for the metrics module.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUndefinedVariable=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportPossiblyUnboundVariable=false, reportUninitializedInstanceVariable=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportMissingTypeArgument=false, reportOptionalSubscript=false

import pytest
import time

from prometheus_client import Counter, Gauge, Histogram, Summary


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_increment(self):
        """Test basic counter increment."""
        counter = Counter("test_counter", "A test counter")

        counter.inc()
        assert counter.get_value() == 1.0

        counter.inc(5)
        assert counter.get_value() == 6.0

    def test_counter_negative_increment_raises(self):
        """Test that negative increment raises error."""
        counter = Counter("test_counter", "A test counter")

        with pytest.raises(ValueError):
            counter.inc(-1)

    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter(
            "test_counter_labels",
            "A test counter with labels",
            labels=["method", "status"],
        )

        counter.inc(labels={"method": "GET", "status": "200"})
        counter.inc(labels={"method": "POST", "status": "201"})
        counter.inc(2, labels={"method": "GET", "status": "200"})

        assert counter.get_value({"method": "GET", "status": "200"}) == 3.0
        assert counter.get_value({"method": "POST", "status": "201"}) == 1.0

    def test_counter_prometheus_export(self):
        """Test Prometheus format export."""
        counter = Counter("http_requests_total", "Total HTTP requests")
        counter.inc(10)

        output = counter.to_prometheus()
        assert "# HELP http_requests_total Total HTTP requests" in output
        assert "# TYPE http_requests_total counter" in output
        assert "http_requests_total 10" in output


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_set(self):
        """Test gauge set operation."""
        gauge = Gauge("temperature", "Current temperature")

        gauge.set(25.5)
        assert gauge.get_value() == 25.5

        gauge.set(30.0)
        assert gauge.get_value() == 30.0

    def test_gauge_inc_dec(self):
        """Test gauge increment and decrement."""
        gauge = Gauge("connections", "Active connections")

        gauge.set(10)
        gauge.inc(5)
        assert gauge.get_value() == 15.0

        gauge.dec(3)
        assert gauge.get_value() == 12.0

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge(
            "cpu_usage",
            "CPU usage by core",
            labels=["core"],
        )

        gauge.set(50.0, labels={"core": "0"})
        gauge.set(75.0, labels={"core": "1"})

        assert gauge.get_value({"core": "0"}) == 50.0
        assert gauge.get_value({"core": "1"}) == 75.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_observe(self):
        """Test histogram observations."""
        histogram = Histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )

        histogram.observe(0.05)  # < 0.1
        histogram.observe(0.3)   # < 0.5
        histogram.observe(0.7)   # < 1.0
        histogram.observe(2.0)   # < inf

    def test_histogram_prometheus_export(self):
        """Test histogram Prometheus export."""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=(1.0, 5.0, float("inf")),
        )

        histogram.observe(0.5)
        histogram.observe(2.0)
        histogram.observe(10.0)

        output = histogram.to_prometheus()
        assert "# TYPE test_histogram histogram" in output
        assert "_bucket" in output
        assert "_sum" in output
        assert "_count" in output

    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        histogram = Histogram(
            "api_latency",
            "API latency",
            labels=["endpoint"],
            buckets=(0.1, 0.5, 1.0, float("inf")),
        )

        histogram.observe(0.05, labels={"endpoint": "/users"})
        histogram.observe(0.3, labels={"endpoint": "/posts"})


class TestSummary:
    """Tests for Summary metric."""

    def test_summary_observe(self):
        """Test summary observations."""
        summary = Summary(
            "request_size_bytes",
            "Request size in bytes",
            quantiles=(0.5, 0.9, 0.99),
        )

        for i in range(100):
            summary.observe(i * 10)

    def test_summary_prometheus_export(self):
        """Test summary Prometheus export."""
        summary = Summary(
            "test_summary",
            "Test summary",
            quantiles=(0.5, 0.99),
        )

        for i in range(100):
            summary.observe(float(i))

        output = summary.to_prometheus()
        assert "# TYPE test_summary summary" in output
        assert "quantile=" in output


class TestTradingMetrics:
    """Tests for TradingMetrics collection."""

    def test_trading_metrics_creation(self):
        """Test creating trading metrics."""
        metrics = TradingMetrics(prefix="test")

        assert metrics.total_pnl is not None
        assert metrics.unrealized_pnl is not None
        assert metrics.total_trades is not None
        assert metrics.current_drawdown is not None

    def test_record_trade(self):
        """Test recording a trade."""
        metrics = TradingMetrics()

        metrics.record_trade(
            symbol="BTC/AUD",
            side="buy",
            strategy="momentum",
            pnl=100.0,
        )

        # Check counters were incremented
        assert metrics.total_trades.get_value(
            {"symbol": "BTC/AUD", "side": "buy", "strategy": "momentum"}
        ) == 1.0

        # Winning trade counter
        assert metrics.winning_trades.get_value(
            {"symbol": "BTC/AUD", "strategy": "momentum"}
        ) == 1.0

    def test_record_losing_trade(self):
        """Test recording a losing trade."""
        metrics = TradingMetrics()

        metrics.record_trade(
            symbol="ETH/AUD",
            side="sell",
            strategy="mean_reversion",
            pnl=-50.0,
        )

        assert metrics.losing_trades.get_value(
            {"symbol": "ETH/AUD", "strategy": "mean_reversion"}
        ) == 1.0

    def test_update_risk_metrics(self):
        """Test updating risk metrics."""
        metrics = TradingMetrics()

        metrics.update_risk_metrics(
            drawdown=0.05,
            max_drawdown=0.08,
            var_95=150.0,
            exposure=5000.0,
            position_count=3,
            circuit_breaker=False,
        )

        assert metrics.current_drawdown.get_value() == 5.0  # Converted to %
        assert metrics.max_drawdown.get_value() == 8.0
        assert metrics.var_95.get_value() == 150.0
        assert metrics.total_exposure.get_value() == 5000.0
        assert metrics.position_count.get_value() == 3
        assert metrics.circuit_breaker_active.get_value() == 0

    def test_update_capital_metrics(self):
        """Test updating capital metrics."""
        metrics = TradingMetrics()

        metrics.update_capital_metrics(
            capital=10000.0,
            daily_pnl=150.0,
            daily_return_pct=0.015,
        )

        assert metrics.current_capital.get_value() == 10000.0
        assert metrics.daily_pnl.get_value() == 150.0
        assert metrics.daily_return_pct.get_value() == 1.5  # Converted to %

    def test_record_exchange_call(self):
        """Test recording exchange API call."""
        metrics = TradingMetrics()

        metrics.record_exchange_call(
            exchange="kraken",
            operation="fetch_ticker",
            latency_seconds=0.15,
        )

        # With error
        metrics.record_exchange_call(
            exchange="kraken",
            operation="create_order",
            latency_seconds=0.5,
            error="timeout",
        )

        assert metrics.exchange_errors.get_value(
            {"exchange": "kraken", "error_type": "timeout"}
        ) == 1.0

    def test_record_signal(self):
        """Test recording strategy signal."""
        metrics = TradingMetrics()

        metrics.record_signal(
            strategy="momentum",
            action="BUY",
            confidence=0.85,
        )

        assert metrics.signal_generated.get_value(
            {"strategy": "momentum", "action": "BUY"}
        ) == 1.0

    def test_update_market_data(self):
        """Test updating market data metrics."""
        metrics = TradingMetrics()

        metrics.update_market_data(
            symbol="BTC/AUD",
            price=50000.0,
            spread_bps=5.0,
            volume_24h=1000000.0,
        )

        assert metrics.price.get_value({"symbol": "BTC/AUD"}) == 50000.0
        assert metrics.spread_bps.get_value({"symbol": "BTC/AUD"}) == 5.0

    def test_get_uptime(self):
        """Test uptime calculation."""
        metrics = TradingMetrics()
        time.sleep(0.1)

        uptime = metrics.get_uptime()
        assert uptime >= 0.1

    def test_export_prometheus(self):
        """Test Prometheus export."""
        metrics = TradingMetrics()
        metrics.update_pnl_metrics(total_pnl=500.0, unrealized_pnl=100.0)

        output = metrics.export_prometheus()
        assert "argus_total_pnl_aud" in output
        assert "argus_unrealized_pnl_aud" in output

    def test_get_summary(self):
        """Test getting metrics summary."""
        metrics = TradingMetrics()
        metrics.update_pnl_metrics(total_pnl=500.0, unrealized_pnl=100.0)
        metrics.update_risk_metrics(
            drawdown=0.05,
            max_drawdown=0.08,
            var_95=150.0,
            exposure=5000.0,
            position_count=2,
            circuit_breaker=False,
        )

        summary = metrics.get_summary()

        assert summary["total_pnl"] == 500.0
        assert summary["unrealized_pnl"] == 100.0
        assert summary["position_count"] == 2
        assert summary["circuit_breaker_active"] is False


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        histogram = Histogram(
            "operation_duration",
            "Operation duration",
            buckets=(0.01, 0.1, 1.0, float("inf")),
        )

        with Timer(histogram):
            time.sleep(0.05)

    def test_timer_with_labels(self):
        """Test timer with labels."""
        histogram = Histogram(
            "operation_duration",
            "Operation duration",
            labels=["operation"],
            buckets=(0.01, 0.1, 1.0, float("inf")),
        )

        with Timer(histogram, labels={"operation": "fetch"}):
            time.sleep(0.01)


class TestGlobalFunctions:
    """Tests for global metric functions."""

    def test_get_metrics_singleton(self):
        """Test global metrics singleton."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2

    def test_export_prometheus_function(self):
        """Test global export function."""
        output = export_prometheus()
        assert "argus_" in output
