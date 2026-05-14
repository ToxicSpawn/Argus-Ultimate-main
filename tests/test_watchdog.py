"""
Tests for the ARGUS self-healing watchdog system.

Covers:
  - AutoRestarter: start/stop/restart, backoff, max restart limit
  - LogAnomalyDetector: all pattern matches, stats, hung detection
  - ConfigAutoTuner: all 5 rules, dry-run, apply, DB persistence
  - HealthMonitorDaemon: disk, memory, log staleness, HTTP checks
  - ArgusWatchdog: health check, kill switch, consecutive failures

30+ tests total.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# AutoRestarter tests
# ---------------------------------------------------------------------------
from ops.auto_restart import AutoRestarter, _BACKOFF_DELAYS, _MAX_RESTARTS


class TestAutoRestarter:

    def test_not_running_initially(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        assert r.is_running() is False
        assert r.get_uptime() == 0.0
        assert r.get_pid() is None

    def test_start_sets_process(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc
            assert r.start_argus(mode="paper") is True
            assert r.is_running() is True
            assert r.get_pid() == 12345
            assert r.get_uptime() > 0.0

    def test_start_when_already_running(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.pid = 111
            mock_popen.return_value = mock_proc
            r.start_argus(mode="paper")
            result = r.start_argus(mode="paper")
            assert result is True
            assert mock_popen.call_count == 1  # Not called twice

    def test_stop_when_not_running(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        assert r.stop_argus() is True

    def test_stop_graceful(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        mock_proc = mock.MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 222
        mock_proc.wait.return_value = 0
        r._process = mock_proc
        assert r.stop_argus(timeout=1.0) is True
        assert r._process is None

    def test_backoff_delays(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        assert r._get_backoff_delay() == 0  # First restart, no delay
        r._consecutive_restarts = 1
        assert r._get_backoff_delay() == 10
        r._consecutive_restarts = 2
        assert r._get_backoff_delay() == 30
        r._consecutive_restarts = 3
        assert r._get_backoff_delay() == 60
        r._consecutive_restarts = 4
        assert r._get_backoff_delay() == 120
        r._consecutive_restarts = 5
        assert r._get_backoff_delay() == 300
        r._consecutive_restarts = 100
        assert r._get_backoff_delay() == 300  # Caps at last value

    def test_max_restart_limit(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        # Fill restart history to exceed limit
        now = time.time()
        from ops.auto_restart import RestartRecord
        for i in range(_MAX_RESTARTS):
            r._restart_history.append(RestartRecord(now - 60 + i, "test", True))
        assert r._can_restart() is False

    def test_restart_respects_limit(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        now = time.time()
        from ops.auto_restart import RestartRecord
        for i in range(_MAX_RESTARTS):
            r._restart_history.append(RestartRecord(now - 60 + i, "test", True))
        result = r.restart_argus(reason="test")
        assert result is False

    def test_old_restarts_not_counted(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        from ops.auto_restart import RestartRecord
        old_time = time.time() - 7200  # 2 hours ago
        for i in range(_MAX_RESTARTS):
            r._restart_history.append(RestartRecord(old_time + i, "test", True))
        assert r._can_restart() is True

    def test_clean_paper_state_removes_kill_switch(self):
        tmpdir = Path(tempfile.mkdtemp())
        ks = tmpdir / "KILL_SWITCH"
        ks.write_text("test")
        r = AutoRestarter(project_root=tmpdir)
        r._clean_paper_state()
        assert not ks.exists()

    def test_reset_restart_counter(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()))
        r._consecutive_restarts = 5
        r.reset_restart_counter()
        assert r._consecutive_restarts == 0

    def test_start_failure_returns_false(self):
        r = AutoRestarter(project_root=Path(tempfile.mkdtemp()), py_executable="nonexistent_py_12345")
        result = r.start_argus(mode="paper")
        assert result is False


# ---------------------------------------------------------------------------
# LogAnomalyDetector tests
# ---------------------------------------------------------------------------
from ops.log_anomaly_detector import LogAnomalyDetector


class TestLogAnomalyDetector:

    def test_empty_line(self):
        d = LogAnomalyDetector()
        assert d.analyze_line("") is None
        assert d.analyze_line("   ") is None

    def test_normal_line(self):
        d = LogAnomalyDetector()
        assert d.analyze_line("2026-03-24 INFO: All systems normal") is None

    def test_emergency_stop(self):
        cb = mock.MagicMock()
        d = LogAnomalyDetector(on_restart=cb, on_clear_kill_switch=mock.MagicMock())
        result = d.analyze_line("2026-03-24 CRITICAL: Emergency stop triggered")
        assert result == "clear_and_restart"
        cb.assert_called_once_with("kill_switch")

    def test_kill_switch_detected(self):
        cb = mock.MagicMock()
        d = LogAnomalyDetector(on_restart=cb, on_clear_kill_switch=mock.MagicMock())
        result = d.analyze_line("WARNING: KILL_SWITCH detected — halting")
        assert result == "clear_and_restart"

    def test_consecutive_losses(self):
        d = LogAnomalyDetector()
        result = d.analyze_line("WARNING: Maximum consecutive losses reached (5)")
        assert result == "warning_only"

    def test_timeout_single(self):
        d = LogAnomalyDetector()
        result = d.analyze_line("ERROR: TimeoutError connecting to exchange")
        assert result == "logged"

    def test_timeout_flood_triggers_restart(self):
        cb_restart = mock.MagicMock()
        cb_timeout = mock.MagicMock()
        d = LogAnomalyDetector(on_restart=cb_restart, on_increase_timeout=cb_timeout)
        # Simulate 5 timeouts in quick succession
        for i in range(5):
            result = d.analyze_line(f"ERROR: TimeoutError attempt {i}")
        assert result == "increase_timeout_and_restart"
        cb_restart.assert_called()
        cb_timeout.assert_called()

    def test_rate_limit(self):
        cb = mock.MagicMock()
        d = LogAnomalyDetector(on_increase_delay=cb)
        result = d.analyze_line("WARNING: rate limit exceeded, backing off")
        assert result == "increase_delay"
        cb.assert_called_once()

    def test_throttle(self):
        cb = mock.MagicMock()
        d = LogAnomalyDetector(on_increase_delay=cb)
        result = d.analyze_line("WARNING: request throttled by exchange")
        assert result == "increase_delay"

    def test_position_drift(self):
        cb = mock.MagicMock()
        d = LogAnomalyDetector(on_clear_positions=cb)
        result = d.analyze_line("ERROR: POSITION DRIFT detected for BTC/USD")
        assert result == "clear_positions"
        cb.assert_called_once()

    def test_coroutine_never_awaited(self):
        d = LogAnomalyDetector()
        result = d.analyze_line("RuntimeWarning: coroutine 'foo' was never awaited")
        assert result == "known_issue"

    def test_cycle_complete_resets_timer(self):
        d = LogAnomalyDetector()
        d._last_cycle_time = time.time() - 1000
        d.analyze_line("INFO: Cycle 42 complete in 5.2s")
        assert time.time() - d._last_cycle_time < 2

    def test_hung_process_detection(self):
        cb = mock.MagicMock()
        d = LogAnomalyDetector(on_restart=cb)
        d._last_cycle_time = time.time() - 700  # 11+ minutes ago
        assert d.check_hung_process() is True
        cb.assert_called_once_with("hung_process")

    def test_not_hung_if_recent_cycle(self):
        d = LogAnomalyDetector()
        d._last_cycle_time = time.time()
        assert d.check_hung_process() is False

    def test_anomaly_stats(self):
        d = LogAnomalyDetector()
        d.analyze_line("ERROR: TimeoutError")
        d.analyze_line("ERROR: TimeoutError")
        d.analyze_line("WARNING: rate limit hit")
        stats = d.get_anomaly_stats()
        assert stats["timeout_error"] == 2
        assert stats["rate_limit"] == 1

    def test_anomaly_history(self):
        d = LogAnomalyDetector()
        d.analyze_line("ERROR: POSITION DRIFT found")
        history = d.get_anomaly_history()
        assert len(history) == 1
        assert history[0].pattern_name == "position_drift"


# ---------------------------------------------------------------------------
# ConfigAutoTuner tests
# ---------------------------------------------------------------------------
from ops.config_auto_tuner import ConfigAutoTuner, TuningChange

import yaml


class TestConfigAutoTuner:

    def _make_config(self, tmpdir: Path, data: dict) -> Path:
        cfg = tmpdir / "config.yaml"
        with open(cfg, "w") as f:
            yaml.dump(data, f)
        return cfg

    def _make_log(self, tmpdir: Path, lines: list) -> Path:
        log = tmpdir / "test.log"
        log.write_text("\n".join(lines), encoding="utf-8")
        return log

    def test_no_log_file(self):
        t = ConfigAutoTuner(db_path=Path(tempfile.mkdtemp()) / "t.db")
        changes = t.analyze_and_tune("/nonexistent/log.txt", "/nonexistent/cfg.yaml")
        assert changes == []

    def test_small_delta_rule(self):
        tmpdir = Path(tempfile.mkdtemp())
        lines = (
            ["INFO signal generated"] * 10
            + ["WARNING trade suppressed: small_delta"] * 20
        )
        log = self._make_log(tmpdir, lines)
        cfg = self._make_config(tmpdir, {"edge_cost_gate": {"min_delta_pct": 0.5}})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=True)
        changes = t.analyze_and_tune(str(log), str(cfg))
        assert len(changes) >= 1
        delta_change = [c for c in changes if "min_delta_pct" in c.key]
        assert len(delta_change) == 1
        assert delta_change[0].new_value == 0.25

    def test_liquidity_thin_rule(self):
        tmpdir = Path(tempfile.mkdtemp())
        lines = (
            ["INFO signal check"] * 10
            + ["WARNING trade suppressed: liquidity_thin"] * 20
        )
        log = self._make_log(tmpdir, lines)
        cfg = self._make_config(tmpdir, {"execution_engine": {"assume_normal_without_l2": False}})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=True)
        changes = t.analyze_and_tune(str(log), str(cfg))
        liq_change = [c for c in changes if "assume_normal_without_l2" in c.key]
        assert len(liq_change) == 1
        assert liq_change[0].new_value is True

    def test_timeout_rate_rule(self):
        tmpdir = Path(tempfile.mkdtemp())
        lines = (
            ["INFO Cycle 1 complete in 10s"] * 10
            + ["ERROR TimeoutError connecting"] * 5
        )
        log = self._make_log(tmpdir, lines)
        cfg = self._make_config(tmpdir, {"execution_engine": {"latency_spike_ms": 500}})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=True)
        changes = t.analyze_and_tune(str(log), str(cfg))
        timeout_change = [c for c in changes if "latency_spike_ms" in c.key]
        assert len(timeout_change) == 1
        assert timeout_change[0].new_value == 750

    def test_cooldown_blocking_rule(self):
        tmpdir = Path(tempfile.mkdtemp())
        lines = (
            ["INFO signal generated"] * 10
            + ["WARNING STRATEGY_COOLDOWN blocked signal"] * 20
        )
        log = self._make_log(tmpdir, lines)
        cfg = self._make_config(tmpdir, {"strategies": {"cooldown_minutes": 60}})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=True)
        changes = t.analyze_and_tune(str(log), str(cfg))
        cd_change = [c for c in changes if "cooldown_minutes" in c.key]
        assert len(cd_change) == 1
        assert cd_change[0].new_value == 30

    def test_cycle_time_rule(self):
        tmpdir = Path(tempfile.mkdtemp())
        lines = ["INFO Cycle complete in 150s"] * 10
        log = self._make_log(tmpdir, lines)
        cfg = self._make_config(tmpdir, {"trading_pairs": ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=True)
        changes = t.analyze_and_tune(str(log), str(cfg))
        pair_change = [c for c in changes if "trading_pairs" in c.key]
        assert len(pair_change) == 1
        assert len(pair_change[0].new_value) == 2

    def test_apply_changes_writes_yaml(self):
        tmpdir = Path(tempfile.mkdtemp())
        cfg = self._make_config(tmpdir, {"execution_engine": {"latency_spike_ms": 500}})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=False)
        t.apply_changes({"execution_engine.latency_spike_ms": 750}, str(cfg))
        with open(cfg) as f:
            data = yaml.safe_load(f)
        assert data["execution_engine"]["latency_spike_ms"] == 750

    def test_dry_run_does_not_write(self):
        tmpdir = Path(tempfile.mkdtemp())
        lines = (
            ["INFO signal generated"] * 10
            + ["WARNING trade suppressed: small_delta"] * 20
        )
        log = self._make_log(tmpdir, lines)
        cfg = self._make_config(tmpdir, {"edge_cost_gate": {"min_delta_pct": 0.5}})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=True)
        changes = t.analyze_and_tune(str(log), str(cfg))
        assert len(changes) >= 1
        assert all(not c.applied for c in changes)
        # Config should be unchanged
        with open(cfg) as f:
            data = yaml.safe_load(f)
        assert data["edge_cost_gate"]["min_delta_pct"] == 0.5

    def test_tuning_history_persistence(self):
        tmpdir = Path(tempfile.mkdtemp())
        lines = (
            ["INFO signal generated"] * 10
            + ["WARNING trade suppressed: small_delta"] * 20
        )
        log = self._make_log(tmpdir, lines)
        cfg = self._make_config(tmpdir, {"edge_cost_gate": {"min_delta_pct": 0.5}})
        t = ConfigAutoTuner(db_path=tmpdir / "t.db", dry_run=True)
        t.analyze_and_tune(str(log), str(cfg))
        history = t.get_tuning_history()
        assert len(history) >= 1
        assert history[0]["key"] == "edge_cost_gate.min_delta_pct"


# ---------------------------------------------------------------------------
# HealthMonitorDaemon tests
# ---------------------------------------------------------------------------
from ops.health_monitor_daemon import HealthMonitorDaemon, HealthCheck


class TestHealthMonitorDaemon:

    def test_disk_space_check_passes(self):
        d = HealthMonitorDaemon(
            project_root=Path(tempfile.mkdtemp()),
            min_disk_gb=0.001,  # Very low threshold
        )
        result = d._check_disk_space()
        assert result.healthy is True
        assert "Disk free" in result.message

    def test_disk_space_check_fails(self):
        d = HealthMonitorDaemon(
            project_root=Path(tempfile.mkdtemp()),
            min_disk_gb=999999,  # Impossibly high threshold
        )
        result = d._check_disk_space()
        assert result.healthy is False
        assert "critically low" in result.message

    def test_memory_check(self):
        d = HealthMonitorDaemon(
            project_root=Path(tempfile.mkdtemp()),
            max_memory_pct=99.9,  # Very high threshold so it passes
        )
        result = d._check_memory()
        assert result.healthy is True

    def test_log_growing_no_file(self):
        d = HealthMonitorDaemon(
            project_root=Path(tempfile.mkdtemp()),
            log_path=Path(tempfile.mkdtemp()) / "nonexistent.log",
        )
        result = d._check_log_growing()
        assert result.healthy is True  # Skips gracefully

    def test_log_growing_active(self):
        tmpdir = Path(tempfile.mkdtemp())
        log = tmpdir / "test.log"
        log.write_text("line 1\n")
        d = HealthMonitorDaemon(
            project_root=tmpdir,
            log_path=log,
        )
        d._last_log_size = 0  # Smaller than current
        result = d._check_log_growing()
        assert result.healthy is True
        assert "Log active" in result.message

    def test_log_stale_triggers_restart(self):
        tmpdir = Path(tempfile.mkdtemp())
        log = tmpdir / "test.log"
        log.write_text("line 1\n")
        d = HealthMonitorDaemon(
            project_root=tmpdir,
            log_path=log,
            log_stale_timeout=1,  # 1 second for testing
        )
        d._last_log_size = log.stat().st_size  # Same size
        d._last_log_check = time.time() - 10  # 10 seconds ago
        with mock.patch.object(d, "_trigger_restart") as mock_restart:
            result = d._check_log_growing()
        assert result.healthy is False
        assert "stale" in result.message
        mock_restart.assert_called_once()

    def test_http_health_success(self):
        d = HealthMonitorDaemon(project_root=Path(tempfile.mkdtemp()))
        response_data = json.dumps({"status": "ok"}).encode()
        with mock.patch("urllib.request.urlopen") as mock_url:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_url.return_value = mock_resp
            result = d._check_http_health()
        assert result.healthy is True

    def test_http_health_failure(self):
        d = HealthMonitorDaemon(project_root=Path(tempfile.mkdtemp()), unresponsive_timeout=1)
        d._first_unresponsive = time.time() - 10  # Already been unresponsive
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            with mock.patch.object(d, "_trigger_restart"):
                result = d._check_http_health()
        assert result.healthy is False


# ---------------------------------------------------------------------------
# ArgusWatchdog tests
# ---------------------------------------------------------------------------
from ops.watchdog import ArgusWatchdog


class TestArgusWatchdog:

    def test_health_check_success(self):
        w = ArgusWatchdog(project_root=Path(tempfile.mkdtemp()))
        response_data = json.dumps({"status": "ok"}).encode()
        with mock.patch("urllib.request.urlopen") as mock_url:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_url.return_value = mock_resp
            assert w.check_health() is True
        assert w._last_health_status == "ok"

    def test_health_check_failure(self):
        w = ArgusWatchdog(project_root=Path(tempfile.mkdtemp()))
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert w.check_health() is False

    def test_kill_switch_paper_mode_clears(self):
        tmpdir = Path(tempfile.mkdtemp())
        ks = tmpdir / "KILL_SWITCH"
        ks.write_text("test:emergency_stop")
        w = ArgusWatchdog(mode="paper", project_root=tmpdir)
        # Mock restarter
        mock_restarter = mock.MagicMock()
        w._restarter = mock_restarter
        result = w.check_kill_switch()
        assert result is True
        assert not ks.exists()
        mock_restarter.restart_argus.assert_called_once()

    def test_kill_switch_live_mode_does_not_clear(self):
        tmpdir = Path(tempfile.mkdtemp())
        ks = tmpdir / "KILL_SWITCH"
        ks.write_text("test:emergency")
        w = ArgusWatchdog(mode="live", project_root=tmpdir)
        result = w.check_kill_switch()
        assert result is True
        assert ks.exists()  # NOT cleared in live mode

    def test_kill_switch_not_present(self):
        w = ArgusWatchdog(project_root=Path(tempfile.mkdtemp()))
        assert w.check_kill_switch() is False

    def test_consecutive_failure_counting(self):
        w = ArgusWatchdog(max_consecutive_failures=3, project_root=Path(tempfile.mkdtemp()))
        mock_restarter = mock.MagicMock()
        mock_restarter.is_running.return_value = True
        w._restarter = mock_restarter

        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            # Fail 1
            w._check_cycle()
            assert w._consecutive_failures == 1
            mock_restarter.restart_argus.assert_not_called()

            # Fail 2
            w._check_cycle()
            assert w._consecutive_failures == 2
            mock_restarter.restart_argus.assert_not_called()

            # Fail 3 — triggers restart
            w._check_cycle()
            assert w._consecutive_failures == 0  # Reset after restart
            mock_restarter.restart_argus.assert_called_once()

    def test_health_recovery_resets_counter(self):
        w = ArgusWatchdog(project_root=Path(tempfile.mkdtemp()))
        mock_restarter = mock.MagicMock()
        mock_restarter.is_running.return_value = True
        w._restarter = mock_restarter
        w._consecutive_failures = 2

        response_data = json.dumps({"status": "ok"}).encode()
        with mock.patch("urllib.request.urlopen") as mock_url:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_url.return_value = mock_resp
            w._check_cycle()

        assert w._consecutive_failures == 0
        mock_restarter.reset_restart_counter.assert_called_once()

    def test_auto_start_if_not_running(self):
        w = ArgusWatchdog(mode="paper", project_root=Path(tempfile.mkdtemp()))
        mock_restarter = mock.MagicMock()
        mock_restarter.is_running.return_value = False
        w._restarter = mock_restarter

        w._check_cycle()
        mock_restarter.start_argus.assert_called_once_with(mode="paper")

    def test_get_status(self):
        w = ArgusWatchdog(mode="paper", project_root=Path(tempfile.mkdtemp()))
        mock_restarter = mock.MagicMock()
        mock_restarter.is_running.return_value = False
        mock_restarter.get_uptime.return_value = 0.0
        w._restarter = mock_restarter
        status = w.get_status()
        assert status["mode"] == "paper"
        assert status["running"] is False
        assert status["consecutive_failures"] == 0

    def test_degraded_health_is_acceptable(self):
        w = ArgusWatchdog(project_root=Path(tempfile.mkdtemp()))
        response_data = json.dumps({"status": "degraded"}).encode()
        with mock.patch("urllib.request.urlopen") as mock_url:
            mock_resp = mock.MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = mock.MagicMock(return_value=False)
            mock_url.return_value = mock_resp
            assert w.check_health() is True
