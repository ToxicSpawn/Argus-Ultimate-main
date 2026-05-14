"""Batch 11 tests — health endpoint, JSON logging, signal handlers, task group, metrics, trade_ledger_safe."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# TestHealthPayload
# ---------------------------------------------------------------------------

class TestHealthPayload:
    @pytest.fixture
    def health_mod(self):
        return _load("core/health.py", "_health_b11")

    def test_healthy_status_empty_state(self, health_mod):
        with patch.dict(sys.modules, {"core.shared_state": MagicMock(
            SharedState=MagicMock(instance=MagicMock(return_value={"trading_loop_started": True}))
        )}):
            payload, code = health_mod._build_health_payload()
        assert payload["status"] == "healthy"
        assert code == 200

    def test_starting_status_when_loop_not_started(self, health_mod):
        with patch.dict(sys.modules, {"core.shared_state": MagicMock(
            SharedState=MagicMock(instance=MagicMock(return_value={"trading_loop_started": False}))
        )}):
            payload, code = health_mod._build_health_payload()
        assert payload["status"] == "starting"
        assert code == 503

    def test_halted_status_on_halt_flag(self, health_mod):
        with patch.dict(sys.modules, {"core.shared_state": MagicMock(
            SharedState=MagicMock(instance=MagicMock(return_value={
                "trading_loop_started": True, "halt_active": True
            }))
        )}):
            payload, code = health_mod._build_health_payload()
        assert payload["status"] == "halted"
        assert code == 503

    def test_degraded_on_high_error_count(self, health_mod):
        with patch.dict(sys.modules, {"core.shared_state": MagicMock(
            SharedState=MagicMock(instance=MagicMock(return_value={
                "trading_loop_started": True, "error_count": 51
            }))
        )}):
            payload, code = health_mod._build_health_payload()
        assert payload["status"] == "degraded"
        assert code == 503

    def test_version_field_present(self, health_mod):
        with patch.dict(sys.modules, {"core.shared_state": MagicMock(
            SharedState=MagicMock(instance=MagicMock(return_value={"trading_loop_started": True}))
        )}):
            payload, _ = health_mod._build_health_payload()
        assert "version" in payload


# ---------------------------------------------------------------------------
# TestJsonLogging
# ---------------------------------------------------------------------------

class TestJsonLogging:
    @pytest.fixture
    def log_mod(self):
        return _load("core/logging_config.py", "_logcfg_b11")

    def test_configure_returns_bool(self, log_mod):
        import io
        stream = io.StringIO()
        result = log_mod.configure_json_logging(level="WARNING", stream=stream, force=True)
        assert isinstance(result, bool)

    def test_is_json_logging_active_after_configure(self, log_mod):
        import io
        stream = io.StringIO()
        log_mod.configure_json_logging(level="WARNING", stream=stream, force=True)
        assert log_mod.is_json_logging_active() is True

    def test_ecs_formatter_produces_json(self, log_mod):
        import io, json
        formatter = log_mod._ECSJsonFormatter(service="test-argus")
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0, msg="hello world",
            args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello world"
        assert data["service.name"] == "test-argus"
        assert "@timestamp" in data
        assert "log.level" in data


# ---------------------------------------------------------------------------
# TestSignalHandlers
# FIX: load the module ONCE and share it across all tests so reset_for_testing()
# clears the same _state object that shutdown_requested() reads.
# ---------------------------------------------------------------------------

# Module loaded at class-definition time — single import, shared state
_SIG_MOD = _load("core/signal_handlers.py", "_sig_b11_shared")


class TestSignalHandlers:
    @pytest.fixture(autouse=True)
    def reset_state(self):
        _SIG_MOD.reset_for_testing()
        yield
        _SIG_MOD.reset_for_testing()

    def test_shutdown_not_requested_initially(self):
        assert _SIG_MOD.shutdown_requested() is False

    def test_request_shutdown_sets_flag(self):
        _SIG_MOD.request_shutdown(reason="test")
        assert _SIG_MOD.shutdown_requested() is True

    def test_reset_clears_flag(self):
        _SIG_MOD.request_shutdown(reason="test")
        _SIG_MOD.reset_for_testing()
        assert _SIG_MOD.shutdown_requested() is False


# ---------------------------------------------------------------------------
# TestTaskGroupRunner
# ---------------------------------------------------------------------------

class TestTaskGroupRunner:
    def test_run_single_task_completes(self):
        mod = _load("core/task_group_runner.py", "_tgr_b11a")
        results = []

        async def _task():
            results.append(1)

        async def _run():
            runner = mod.TaskGroupRunner()
            runner.add(_task, name="t1")
            await runner.run_until_complete()

        asyncio.run(_run())
        assert results == [1]

    def test_error_isolation_other_task_still_runs(self):
        mod = _load("core/task_group_runner.py", "_tgr_b11b")
        good_ran = []

        async def _bad():
            raise ValueError("intentional")

        async def _good():
            good_ran.append(1)

        async def _run():
            runner = mod.TaskGroupRunner()
            runner.add(_bad, name="bad", restart_on_failure=False)
            runner.add(_good, name="good", restart_on_failure=False)
            await runner.run_until_complete()

        asyncio.run(_run())
        assert good_ran == [1]

    def test_error_count_incremented(self):
        mod = _load("core/task_group_runner.py", "_tgr_b11c")

        async def _bad():
            raise RuntimeError("boom")

        async def _run():
            runner = mod.TaskGroupRunner()
            runner.add(_bad, name="bad", restart_on_failure=False)
            await runner.run_until_complete()
            return runner.error_counts()

        counts = asyncio.run(_run())
        assert counts.get("bad", 0) == 1


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------

class TestMetrics:
    @pytest.fixture
    def metrics_mod(self):
        return _load("core/metrics.py", "_metrics_b11")

    def test_null_metric_inc_no_error(self, metrics_mod):
        metrics_mod._NULL.inc()
        metrics_mod._NULL.labels(side="buy").inc()

    def test_get_prometheus_text_returns_string(self, metrics_mod):
        text = metrics_mod.get_prometheus_text()
        assert isinstance(text, str)

    def test_record_cycle_no_error(self, metrics_mod):
        metrics_mod.record_cycle(42.5, error=False)
        metrics_mod.record_cycle(99.0, error=True)


# ---------------------------------------------------------------------------
# TestSafeTradeLedger
# ---------------------------------------------------------------------------

class TestSafeTradeLedger:
    @pytest.fixture
    def ledger(self, tmp_path):
        mod = _load("monitoring/trade_ledger_safe.py", "_ledger_b11")
        l = mod.SafeTradeLedger(db_path=str(tmp_path / "test.db"))
        yield l
        l.close()

    def test_record_and_fetch_trade(self, ledger):
        row_id = ledger.record_trade(
            run_id="run001", symbol="BTC/AUD",
            side="buy", qty=0.01, price=95000.0
        )
        assert row_id > 0
        trades = ledger.get_trades("run001")
        assert len(trades) == 1

    def test_invalid_side_raises(self, ledger):
        with pytest.raises(ValueError, match="side"):
            ledger.record_trade("r1", "ETH/AUD", "hold", 1.0, 3000.0)

    def test_sql_injection_in_run_id_raises(self, ledger):
        with pytest.raises(ValueError):
            ledger.record_trade("'; DROP TABLE safe_trades;--", "BTC/AUD", "buy", 0.01, 1.0)

    def test_record_event_persists(self, ledger):
        row_id = ledger.record_event("run002", "HALT", detail="drawdown exceeded")
        assert row_id > 0
