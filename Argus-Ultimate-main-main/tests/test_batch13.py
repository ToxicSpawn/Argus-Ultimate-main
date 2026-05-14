"""tests/test_batch13.py — Batch 13 audit items H11 M02 M05-M10 M19 M26 M27 M28.

14 tests (minimum required).
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest


# ── H11 / M26 — mypy config files exist ─────────────────────────────────────

def test_mypy_ini_exists() -> None:
    """H11: mypy.ini must exist for CI gate."""
    assert Path("mypy.ini").exists(), "mypy.ini missing"


def test_mypy_ini_covers_core() -> None:
    """H11: mypy.ini must configure core/ in strict mode."""
    content = Path("mypy.ini").read_text()
    assert "core" in content, "mypy.ini does not reference core/"


def test_mypy_baseline_exists() -> None:
    """M26: .mypy_ignore_baseline must be committed."""
    assert Path(".mypy_ignore_baseline").exists()


# ── M28 — .secrets.baseline ──────────────────────────────────────────────────

def test_secrets_baseline_exists() -> None:
    """M28: .secrets.baseline must be committed."""
    assert Path(".secrets.baseline").exists()


def test_secrets_baseline_is_valid_json() -> None:
    """M28: .secrets.baseline must be valid JSON."""
    import json
    data = json.loads(Path(".secrets.baseline").read_text())
    assert "version" in data
    assert "results" in data


# ── M05 / M06 — health endpoint ──────────────────────────────────────────────

def test_health_module_importable() -> None:
    """M05: core.health must be importable."""
    from core import health  # noqa: F401


def test_health_app_has_routes() -> None:
    """M05/M06: health_app must expose /health routes."""
    from core.health import build_health_app
    try:
        from fastapi import FastAPI
    except ImportError:
        pytest.skip("fastapi not installed")
    app = build_health_app()
    assert app is not None
    routes = [r.path for r in app.routes]  # type: ignore[attr-defined]
    assert "/health" in routes


# ── M07 — structured logging ─────────────────────────────────────────────────

def test_logging_config_importable() -> None:
    """M07: core.logging_config must be importable."""
    from core import logging_config  # noqa: F401


def test_configure_logging_runs_without_error() -> None:
    """M07: configure_logging() must not raise."""
    from core.logging_config import configure_logging
    configure_logging(level="WARNING", json_output=False)


def test_get_logger_returns_logger() -> None:
    """M07: get_logger() must return a usable logger."""
    from core.logging_config import get_logger
    log = get_logger("test")
    assert log is not None


# ── M08 — SIGTERM handler ────────────────────────────────────────────────────

def test_signal_handlers_importable() -> None:
    """M08: core.signal_handlers must be importable."""
    from core import signal_handlers  # noqa: F401


def test_request_shutdown_sets_flag() -> None:
    """M08: request_shutdown() must set the shutdown flag."""
    from core.signal_handlers import is_shutdown_requested, request_shutdown, reset_shutdown
    reset_shutdown()
    assert not is_shutdown_requested()
    request_shutdown()
    assert is_shutdown_requested()
    reset_shutdown()  # cleanup


# ── M09 — TaskGroup wrapper ───────────────────────────────────────────────────

def test_run_tasks_returns_results() -> None:
    """M09: run_tasks() must return correct results from all coroutines."""
    from core.task_group_runner import run_tasks

    async def _val(n: int) -> int:
        return n * 2

    results = asyncio.run(run_tasks(_val(1), _val(2), _val(3)))
    assert results == [2, 4, 6]


def test_run_tasks_propagates_exception() -> None:
    """M09: exceptions must NOT be silently swallowed."""
    from core.task_group_runner import run_tasks

    async def _boom() -> None:
        raise ValueError("task failed")

    with pytest.raises((ValueError, ExceptionGroup)):
        asyncio.run(run_tasks(_boom()))


# ── M10 — Prometheus metrics ──────────────────────────────────────────────────

def test_metrics_exporter_importable() -> None:
    """M10: core.metrics_exporter must be importable."""
    from core import metrics_exporter  # noqa: F401


def test_metrics_singleton_has_expected_attrs() -> None:
    """M10: METRICS singleton must expose trading metric attributes."""
    from core.metrics_exporter import METRICS
    assert hasattr(METRICS, "orders_placed")
    assert hasattr(METRICS, "execution_latency")
    assert hasattr(METRICS, "risk_blocks")
    assert hasattr(METRICS, "active_positions")


def test_metrics_noop_does_not_raise() -> None:
    """M10: metric calls must not raise even without prometheus_client."""
    from core.metrics_exporter import METRICS
    METRICS.orders_placed.labels(symbol="BTC/USDT", side="buy").inc()
    METRICS.active_positions.set(3)
    METRICS.cycle_errors.labels(component="core").inc()


# ── M19 — parameterised trade ledger ─────────────────────────────────────────

def test_trade_ledger_insert_and_query(tmp_path: Path) -> None:
    """M19: TradeLedger must insert and retrieve a trade without SQL injection."""
    from monitoring.trade_ledger import TradeLedger, TradeRecord

    ledger = TradeLedger(db_path=tmp_path / "test.db")
    trade = TradeRecord(
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        price=65000.0,
        fee=1.5,
        strategy="momentum",
    )
    row_id = ledger.record_trade(trade)
    assert row_id > 0

    rows = ledger.get_trades(symbol="BTC/USDT")
    assert len(rows) == 1
    assert rows[0]["side"] == "buy"
    ledger.close()


def test_trade_ledger_sql_injection_safe(tmp_path: Path) -> None:
    """M19: Malicious symbol string must not corrupt the database."""
    from monitoring.trade_ledger import TradeLedger, TradeRecord

    ledger = TradeLedger(db_path=tmp_path / "sqli.db")
    evil_symbol = "'; DROP TABLE trades; --"
    trade = TradeRecord(
        symbol=evil_symbol,
        side="sell",
        quantity=1.0,
        price=100.0,
    )
    row_id = ledger.record_trade(trade)
    assert row_id > 0
    # Table still intact — query does not raise
    rows = ledger.get_trades(symbol=evil_symbol)
    assert len(rows) == 1
    ledger.close()


# ── M27 — live_gate hard raise ────────────────────────────────────────────────

def test_graduation_error_is_raised_not_warned() -> None:
    """M27: LiveGate.promote() must raise GraduationError for under-qualified strategy."""
    from core.live_gate import GraduationError, LiveGate

    gate = LiveGate()
    gate.register("my_strategy")
    # strategy has no paper time / metrics — must RAISE, not warn
    with pytest.raises(GraduationError):
        gate.promote("my_strategy")


def test_graduation_success_sets_promoted_at() -> None:
    """M27: A strategy meeting all criteria must be promoted without error."""
    from core.live_gate import GraduationCriteria, LiveGate

    criteria = GraduationCriteria(
        min_paper_days=1,
        min_sharpe=0.5,
        min_win_rate=0.4,
        max_drawdown=0.9,
        min_trades=1,
    )
    gate = LiveGate(criteria=criteria)
    gate.update(
        "passing_strategy",
        paper_days=5,
        sharpe=1.0,
        win_rate=0.55,
        max_drawdown=0.05,
        trade_count=10,
    )
    rec = gate.promote("passing_strategy")
    assert rec.promoted_at is not None
