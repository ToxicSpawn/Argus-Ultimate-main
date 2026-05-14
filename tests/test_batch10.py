"""Batch 10 tests — tombstone, version, ruff config, dockerignore, renovate, pytest timeout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# TestMainLegacyTombstone
# ---------------------------------------------------------------------------

class TestMainLegacyTombstone:
    def test_main_legacy_exists(self):
        assert (REPO_ROOT / "main_legacy.py").exists()

    def test_main_legacy_raises_import_error(self):
        """Importing main_legacy must raise ImportError immediately."""
        spec = importlib.util.spec_from_file_location(
            "_main_legacy_tombstone", REPO_ROOT / "main_legacy.py"
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        with pytest.raises(ImportError, match="main_legacy.py has been removed"):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

    def test_main_legacy_no_trading_logic(self):
        """Tombstone must be < 20 lines."""
        lines = (REPO_ROOT / "main_legacy.py").read_text(encoding="utf-8").splitlines()
        assert len(lines) < 20, f"Tombstone too long ({len(lines)} lines) — looks like real code"


# ---------------------------------------------------------------------------
# TestVersionModule
# ---------------------------------------------------------------------------

class TestVersionModule:
    def test_version_py_exists(self):
        assert (REPO_ROOT / "version.py").exists()

    def test_version_string_format(self):
        spec = importlib.util.spec_from_file_location("_version_b10", REPO_ROOT / "version.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        assert hasattr(mod, "__version__")
        parts = mod.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_version_info_tuple(self):
        spec = importlib.util.spec_from_file_location("_version_b10b", REPO_ROOT / "version.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        assert isinstance(mod.__version_info__, tuple)
        assert len(mod.__version_info__) == 3

    def test_version_matches_pyproject(self):
        spec = importlib.util.spec_from_file_location("_version_b10c", REPO_ROOT / "version.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert f'version = "{mod.__version__}"' in pyproject


# ---------------------------------------------------------------------------
# TestPyprojectConfig
# ---------------------------------------------------------------------------

class TestPyprojectConfig:
    @pytest.fixture
    def pyproject_text(self):
        return (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    def test_t201_not_globally_ignored(self, pyproject_text):
        """T201 must NOT appear in the global [tool.ruff.lint] ignore list."""
        import re
        # Find the global ignore line
        match = re.search(r'ignore\s*=\s*\[([^\]]+)\]', pyproject_text)
        assert match, "Could not find ruff ignore list"
        ignore_list = match.group(1)
        assert "T201" not in ignore_list, "T201 still in global ignore — print() not linted"

    def test_pytest_timeout_set(self, pyproject_text):
        assert "--timeout=" in pyproject_text

    def test_coverage_gate_still_75(self, pyproject_text):
        assert "--cov-fail-under=75" in pyproject_text


# ---------------------------------------------------------------------------
# TestDockIgnore
# ---------------------------------------------------------------------------

class TestDockerIgnore:
    @pytest.fixture
    def di_text(self):
        return (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    def test_fix_txt_excluded(self, di_text):
        assert "fix_*.txt" in di_text

    def test_scripts_excluded(self, di_text):
        assert "scripts/" in di_text

    def test_tools_excluded(self, di_text):
        assert "tools/" in di_text

    def test_coverage_artifacts_excluded(self, di_text):
        assert ".coverage" in di_text


# ---------------------------------------------------------------------------
# TestRenovate
# ---------------------------------------------------------------------------

class TestRenovate:
    def test_renovate_json_exists(self):
        assert (REPO_ROOT / ".github" / "renovate.json").exists()

    def test_renovate_json_valid(self):
        data = json.loads(
            (REPO_ROOT / ".github" / "renovate.json").read_text(encoding="utf-8")
        )
        assert "extends" in data
        assert "packageRules" in data

    def test_renovate_has_ccxt_rule(self):
        data = json.loads(
            (REPO_ROOT / ".github" / "renovate.json").read_text(encoding="utf-8")
        )
        pkg_names = [
            pkg
            for rule in data.get("packageRules", [])
            for pkg in rule.get("matchPackageNames", [])
        ]
        assert "ccxt" in pkg_names, "ccxt must have a dedicated renovate rule (manual review)"
