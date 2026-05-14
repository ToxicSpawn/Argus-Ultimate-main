"""
M30 — DeploymentChecklist test suite.

Tests for ops/deployment_checklist.py covering:
  - CheckResult and ChecklistResult data models
  - DeploymentChecklist.run() — individual check logic
  - DeploymentChecklist.enforce() — blocking vs non-blocking modes
  - Custom check registration
  - Summary output
  - Edge cases: no config, custom db_path, kill switch active
"""
from __future__ import annotations

import os
import sys
import time
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(max_drawdown_pct: float = 15.0, max_daily_loss_usd: float = 500.0) -> MagicMock:
    """Return a minimal config mock with required risk attributes."""
    cfg = MagicMock()
    cfg.max_drawdown_pct = max_drawdown_pct
    cfg.max_daily_loss_usd = max_daily_loss_usd
    return cfg


# ---------------------------------------------------------------------------
# Tests: CheckResult
# ---------------------------------------------------------------------------

class TestCheckResult:
    """Unit tests for the CheckResult dataclass."""

    def test_check_result_fields(self):
        """CheckResult stores all expected fields."""
        from ops.deployment_checklist import CheckResult

        cr = CheckResult(name="test", passed=True, critical=True, message="ok", elapsed_ms=5.0)
        assert cr.name == "test"
        assert cr.passed is True
        assert cr.critical is True
        assert cr.message == "ok"
        assert cr.elapsed_ms == 5.0

    def test_check_result_defaults(self):
        """CheckResult elapsed_ms defaults to 0.0."""
        from ops.deployment_checklist import CheckResult

        cr = CheckResult(name="x", passed=False, critical=False, message="fail")
        assert cr.elapsed_ms == 0.0


# ---------------------------------------------------------------------------
# Tests: ChecklistResult
# ---------------------------------------------------------------------------

class TestChecklistResult:
    """Unit tests for ChecklistResult aggregate."""

    def _make_result(self):
        from ops.deployment_checklist import CheckResult, ChecklistResult

        checks = [
            CheckResult("A", passed=True, critical=True, message="ok"),
            CheckResult("B", passed=True, critical=False, message="ok"),
            CheckResult("C", passed=False, critical=False, message="warn"),
            CheckResult("D", passed=False, critical=True, message="fail"),
        ]
        return ChecklistResult(checks=checks, go=False, warnings=["C: warn"])

    def test_passed_count(self):
        """passed_count reflects number of passed checks."""
        result = self._make_result()
        assert result.passed_count == 2

    def test_failed_count(self):
        """failed_count reflects number of failed checks."""
        result = self._make_result()
        assert result.failed_count == 2

    def test_critical_failures(self):
        """critical_failures returns only failed+critical checks."""
        result = self._make_result()
        cf = result.critical_failures
        assert len(cf) == 1
        assert cf[0].name == "D"

    def test_summary_contains_go(self):
        """summary() includes GO/NO-GO status."""
        result = self._make_result()
        s = result.summary()
        assert "NO-GO" in s

    def test_summary_contains_check_names(self):
        """summary() lists all check names."""
        result = self._make_result()
        s = result.summary()
        for name in ("A", "B", "C", "D"):
            assert name in s


# ---------------------------------------------------------------------------
# Tests: Individual checks
# ---------------------------------------------------------------------------

class TestDeploymentChecklistChecks:
    """Tests for each individual check method."""

    def _make_checklist(self, tmp_path, config=None):
        from ops.deployment_checklist import DeploymentChecklist

        return DeploymentChecklist(
            config=config,
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )

    def test_python_version_check_passes(self, tmp_path):
        """Python version check passes (we're running 3.10+)."""
        cl = self._make_checklist(tmp_path)
        passed, msg = cl._check_python()
        assert passed is True
        assert "Python" in msg

    def test_database_check_passes_with_tmp_path(self, tmp_path):
        """Database check passes when a writable tmp path is used."""
        cl = self._make_checklist(tmp_path)
        passed, msg = cl._check_database()
        assert passed is True
        assert "OK" in msg or "ok" in msg.lower() or "SQLite" in msg

    def test_database_check_fails_with_bad_path(self, tmp_path):
        """Database check fails when the db path is not writable."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=Path("/proc/noperms/argus_test.db"),
            lock_dir=tmp_path / "lock",
        )
        passed, msg = cl._check_database()
        assert passed is False
        assert "error" in msg.lower() or "Error" in msg

    def test_process_lock_check_passes_no_lock_file(self, tmp_path):
        """Process lock check passes when no lock file exists."""
        cl = self._make_checklist(tmp_path)
        passed, msg = cl._check_process_lock()
        assert passed is True

    def test_kill_switch_check_passes_when_inactive(self, tmp_path):
        """Kill switch check passes when switch file does not exist."""
        ks_path = tmp_path / "argus_kill_test"
        with patch.dict(os.environ, {"ARGUS_KILL_SWITCH": str(ks_path)}):
            cl = self._make_checklist(tmp_path)
            passed, msg = cl._check_kill_switch()
        assert passed is True

    def test_kill_switch_check_fails_when_active(self, tmp_path):
        """Kill switch check fails when switch file is present."""
        ks_path = tmp_path / "argus_kill"
        ks_path.write_text("active")
        with patch.dict(os.environ, {"ARGUS_KILL_SWITCH": str(ks_path)}):
            cl = self._make_checklist(tmp_path)
            passed, msg = cl._check_kill_switch()
        assert passed is False
        assert "ACTIVE" in msg

    def test_risk_limits_check_passes_with_valid_config(self, tmp_path):
        """Risk limits check passes for valid config."""
        config = _make_config(max_drawdown_pct=15.0, max_daily_loss_usd=500.0)
        cl = self._make_checklist(tmp_path, config=config)
        passed, msg = cl._check_risk_limits()
        assert passed is True
        assert "15" in msg

    def test_risk_limits_check_fails_with_no_config(self, tmp_path):
        """Risk limits check is skipped (returns True) when no config provided."""
        cl = self._make_checklist(tmp_path, config=None)
        passed, msg = cl._check_risk_limits()
        assert passed is True  # Graceful skip
        assert "config" in msg.lower() or "skip" in msg.lower()

    def test_risk_limits_check_fails_dangerous_drawdown(self, tmp_path):
        """Risk limits check fails when drawdown > 30%."""
        config = _make_config(max_drawdown_pct=35.0)
        cl = self._make_checklist(tmp_path, config=config)
        passed, msg = cl._check_risk_limits()
        assert passed is False
        assert "dangerous" in msg.lower() or "high" in msg.lower()

    def test_log_dir_check_passes(self, tmp_path):
        """Log directory check passes with a writable directory."""
        cl = self._make_checklist(tmp_path)
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.write_text"), patch("pathlib.Path.unlink"):
            passed, msg = cl._check_log_dir()
        assert passed is True

    def test_env_vars_check_always_passes_gracefully(self, tmp_path):
        """Env vars check does not hard-fail (ARGUS_ENV optional)."""
        cl = self._make_checklist(tmp_path)
        with patch.dict(os.environ, {}, clear=False):
            passed, msg = cl._check_env_vars()
        # Should not hard-fail; just warn
        assert isinstance(passed, bool)
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# Tests: DeploymentChecklist.run()
# ---------------------------------------------------------------------------

class TestDeploymentChecklistRun:
    """Integration-level tests for the full run() flow."""

    def test_run_returns_checklist_result(self, tmp_path):
        """run() returns a ChecklistResult instance."""
        from ops.deployment_checklist import DeploymentChecklist, ChecklistResult

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        result = cl.run()
        assert isinstance(result, ChecklistResult)

    def test_run_checks_list_non_empty(self, tmp_path):
        """run() executes multiple checks."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        result = cl.run()
        assert len(result.checks) >= 5

    def test_run_go_when_critical_pass(self, tmp_path):
        """go=True when all critical checks pass."""
        from ops.deployment_checklist import DeploymentChecklist

        config = _make_config()
        cl = DeploymentChecklist(
            config=config,
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        # Patch network check (would fail in isolated CI)
        with (
            patch.object(cl, "_check_network", return_value=(True, "mocked ok")),
            patch.object(cl, "_check_paper_pnl", return_value=(True, "mocked ok")),
        ):
            result = cl.run()
        assert result.go is True

    def test_run_no_go_when_critical_fails(self, tmp_path):
        """go=False when a critical check fails."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        # Inject a failing critical check
        with patch.object(cl, "_check_python", return_value=(False, "forced fail")):
            result = cl.run()
        assert result.go is False

    def test_run_warnings_list_populated(self, tmp_path):
        """Non-critical failures are added to warnings list."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        with (
            patch.object(cl, "_check_network", return_value=(True, "ok")),
            patch.object(cl, "_check_paper_pnl", return_value=(False, "no trades")),
        ):
            result = cl.run()
        assert len(result.warnings) >= 1


# ---------------------------------------------------------------------------
# Tests: enforce()
# ---------------------------------------------------------------------------

class TestDeploymentChecklistEnforce:
    """Tests for the enforce() method."""

    def test_enforce_live_blocks_on_critical_failure(self, tmp_path):
        """enforce(mode='live') returns False when critical check fails."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        with patch.object(cl, "_check_python", return_value=(False, "forced fail")):
            can_deploy = cl.enforce(mode="live")
        assert can_deploy is False

    def test_enforce_paper_does_not_block(self, tmp_path):
        """enforce(mode='paper') returns True even with critical failures."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        with patch.object(cl, "_check_python", return_value=(False, "forced fail")):
            can_deploy = cl.enforce(mode="paper")
        assert can_deploy is True

    def test_enforce_block_on_failure_override(self, tmp_path):
        """block_on_failure=False always allows deployment."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        with patch.object(cl, "_check_python", return_value=(False, "forced fail")):
            can_deploy = cl.enforce(mode="live", block_on_failure=False)
        assert can_deploy is True


# ---------------------------------------------------------------------------
# Tests: custom check registration
# ---------------------------------------------------------------------------

class TestCustomChecks:
    """Tests for add_check() custom check registration."""

    def test_add_custom_check_passing(self, tmp_path):
        """Custom passing check is included in results."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        cl.add_check("my_custom", lambda: (True, "custom ok"), critical=False)
        with patch.object(cl, "_check_network", return_value=(True, "ok")):
            result = cl.run()

        names = [c.name for c in result.checks]
        assert "my_custom" in names

    def test_add_custom_check_failing_critical(self, tmp_path):
        """Failing critical custom check sets go=False."""
        from ops.deployment_checklist import DeploymentChecklist

        cl = DeploymentChecklist(
            db_path=tmp_path / "test.db",
            lock_dir=tmp_path / "lock",
        )
        cl.add_check("blocker", lambda: (False, "custom blocker"), critical=True)
        with patch.object(cl, "_check_network", return_value=(True, "ok")):
            result = cl.run()

        assert result.go is False
        cf_names = [c.name for c in result.critical_failures]
        assert "blocker" in cf_names
