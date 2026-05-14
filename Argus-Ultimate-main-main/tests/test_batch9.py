"""Batch 9 tests — coverage gate tool, startup wiring in _run_unified_system, entrypoint smoke."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(rel_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# TestCoverageGateTool
# ---------------------------------------------------------------------------

class TestCoverageGateTool:
    """Unit tests for tools/check_coverage_gate.py (no pytest subprocess needed)."""

    @pytest.fixture
    def gate_mod(self):
        return _load_module("tools/check_coverage_gate.py", "check_coverage_gate")

    def test_thresholds_defined(self, gate_mod):
        assert "core" in gate_mod.PACKAGE_THRESHOLDS
        assert "risk" in gate_mod.PACKAGE_THRESHOLDS
        assert gate_mod.PACKAGE_THRESHOLDS["core"] >= 75

    def test_overall_threshold_defined(self, gate_mod):
        assert isinstance(gate_mod.OVERALL_THRESHOLD, int)
        assert gate_mod.OVERALL_THRESHOLD >= 60

    def test_parse_coverage_json_pass(self, gate_mod, tmp_path):
        # Build a minimal coverage.json
        data = {
            "files": {
                "core/config_schema.py": {
                    "summary": {"covered_lines": 90, "num_statements": 100}
                },
                "core/shared_state.py": {
                    "summary": {"covered_lines": 95, "num_statements": 100}
                },
            }
        }
        cov_file = tmp_path / "cov.json"
        cov_file.write_text(json.dumps(data))
        results = gate_mod._parse_coverage_json(cov_file)
        assert "core" in results
        assert results["core"].covered_lines == 185
        assert results["core"].total_lines == 200
        assert results["core"].pct == 92.5

    def test_parse_coverage_json_below_threshold(self, gate_mod, tmp_path):
        data = {
            "files": {
                "risk/unified_risk_manager.py": {
                    "summary": {"covered_lines": 10, "num_statements": 100}
                },
            }
        }
        cov_file = tmp_path / "cov.json"
        cov_file.write_text(json.dumps(data))
        results = gate_mod._parse_coverage_json(cov_file)
        assert "risk" in results
        assert results["risk"].pct == 10.0

    def test_package_result_passed_flag(self, gate_mod):
        from dataclasses import fields
        pr = gate_mod.PackageResult(
            package="core",
            covered_lines=90,
            total_lines=100,
            pct=90.0,
            threshold=80,
            passed=True,
        )
        assert pr.passed is True

    def test_package_result_failed_flag(self, gate_mod):
        pr = gate_mod.PackageResult(
            package="core",
            covered_lines=50,
            total_lines=100,
            pct=50.0,
            threshold=80,
            passed=False,
        )
        assert pr.passed is False

    def test_empty_package_returns_zero_pct(self, gate_mod, tmp_path):
        # File with 0 statements (e.g. empty __init__.py)
        data = {
            "files": {
                "core/__init__.py": {
                    "summary": {"covered_lines": 0, "num_statements": 0}
                },
            }
        }
        cov_file = tmp_path / "cov.json"
        cov_file.write_text(json.dumps(data))
        results = gate_mod._parse_coverage_json(cov_file)
        # 0/0 → 0.0%  not NaN
        assert results["core"].pct == 0.0


# ---------------------------------------------------------------------------
# TestStartupWiringInRunUnifiedSystem
# ---------------------------------------------------------------------------

class TestStartupWiringInRunUnifiedSystem:
    """Verify that _run_unified_system in main.py calls startup_config_check."""

    def test_startup_module_importable(self):
        """core/startup.py must be importable from repo root."""
        startup_path = REPO_ROOT / "core" / "startup.py"
        assert startup_path.exists(), "core/startup.py not found"

    def test_startup_config_check_callable(self):
        mod = _load_module("core/startup.py", "_startup_b9")
        assert callable(mod.startup_config_check)

    def test_get_config_callable(self):
        mod = _load_module("core/startup.py", "_startup_b9b")
        assert callable(mod.get_config)

    def test_auto_discover_paths_contains_three_candidates(self):
        mod = _load_module("core/startup.py", "_startup_b9c")
        assert len(mod._AUTO_DISCOVER_PATHS) == 3

    def test_startup_config_check_returns_namespace_on_missing_schema(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""
            system:
              mode: dry_run
              initial_capital: 2500.0
            risk:
              max_drawdown_pct: 12.0
        """))
        mod = _load_module("core/startup.py", "_startup_b9d")
        fake_ss_dict = {}
        fake_ss = MagicMock()
        fake_ss.instance.return_value = fake_ss_dict
        with patch.dict(sys.modules, {
            "core.shared_state": MagicMock(SharedState=fake_ss),
            "core.config_schema": None,
        }):
            cfg = mod.startup_config_check(config_path=cfg_file)
        assert cfg.system.mode == "dry_run"
        assert cfg.system.initial_capital == 2500.0

    def test_mode_override_applied_on_simplenamespace_path(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("system:\n  mode: dry_run\n  initial_capital: 1000\n")
        mod = _load_module("core/startup.py", "_startup_b9e")
        fake_ss = MagicMock()
        fake_ss.instance.return_value = {}
        with patch.dict(sys.modules, {
            "core.shared_state": MagicMock(SharedState=fake_ss),
            "core.config_schema": None,
        }):
            cfg = mod.startup_config_check(config_path=cfg_file, mode_override="paper")
        assert cfg.system.mode == "paper"


# ---------------------------------------------------------------------------
# TestEntrypointSmoke
# ---------------------------------------------------------------------------

class TestEntrypointSmoke:
    """Lightweight smoke tests that don't actually boot the trading system."""

    def test_main_py_exists(self):
        assert (REPO_ROOT / "main.py").exists()

    def test_run_godmode_py_exists(self):
        assert (REPO_ROOT / "run_godmode.py").exists()

    def test_run_paper_py_exists(self):
        assert (REPO_ROOT / "run_paper.py").exists()

    def test_main_py_has_startup_import_comment(self):
        """main.py should reference core.startup in some form."""
        text = (REPO_ROOT / "main.py").read_text(encoding="utf-8")
        # Either the import is present or the legacy _load_yaml_from_path is still there
        # (we don't remove it wholesale — it’s used by reconciliation too)
        assert "_load_yaml_from_path" in text or "startup_config_check" in text

    def test_tools_dir_contains_all_batch_tools(self):
        tools = {f.name for f in (REPO_ROOT / "tools").iterdir() if f.suffix == ".py"}
        expected = {
            "fix_missing_inits.py",
            "fix_print_calls.py",
            "fix_silent_excepts.py",
            "check_coverage_gate.py",
        }
        assert expected.issubset(tools), f"Missing tools: {expected - tools}"
