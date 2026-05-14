"""Batch 8 tests — fix_missing_inits, requirements hygiene, startup wiring."""

from __future__ import annotations

import importlib
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_fix_missing_inits():
    """Import tools/fix_missing_inits.py regardless of sys.path state."""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "fix_missing_inits", repo_root / "tools" / "fix_missing_inits.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# TestMissingInitFixer
# ---------------------------------------------------------------------------

class TestMissingInitFixer:
    """Tests for tools/fix_missing_inits.py."""

    def test_detects_package_missing_init(self, tmp_path):
        pkg = tmp_path / "mypackage"
        pkg.mkdir()
        (pkg / "module.py").write_text("x = 1\n")

        mod = _load_fix_missing_inits()
        missing = mod.find_missing(tmp_path)
        assert pkg in missing

    def test_skips_package_with_existing_init(self, tmp_path):
        pkg = tmp_path / "mypackage"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "module.py").write_text("x = 1\n")

        mod = _load_fix_missing_inits()
        missing = mod.find_missing(tmp_path)
        assert pkg not in missing

    def test_skips_skip_dirs(self, tmp_path):
        for skip in ["__pycache__", ".git", "node_modules", "artifacts"]:
            d = tmp_path / skip
            d.mkdir()
            (d / "something.py").write_text("")

        mod = _load_fix_missing_inits()
        missing = mod.find_missing(tmp_path)
        assert missing == []

    def test_skips_dir_with_no_py_files(self, tmp_path):
        d = tmp_path / "empty_dir"
        d.mkdir()
        (d / "README.md").write_text("hello")

        mod = _load_fix_missing_inits()
        missing = mod.find_missing(tmp_path)
        assert d not in missing

    def test_writes_init_with_docstring(self, tmp_path):
        pkg = tmp_path / "newpkg"
        pkg.mkdir()
        (pkg / "foo.py").write_text("")

        mod = _load_fix_missing_inits()
        init_path = pkg / "__init__.py"
        assert not init_path.exists()

        missing = mod.find_missing(tmp_path)
        for p in missing:
            content = mod._make_docstring(p, tmp_path)
            (p / "__init__.py").write_text(content)

        assert init_path.exists()
        text = init_path.read_text()
        assert "newpkg" in text
        assert '"""' in text

    def test_generated_init_is_importable(self, tmp_path):
        pkg = tmp_path / "importable_pkg"
        pkg.mkdir()
        (pkg / "bar.py").write_text("VALUE = 42\n")

        mod = _load_fix_missing_inits()
        content = mod._make_docstring(pkg, tmp_path)
        (pkg / "__init__.py").write_text(content)

        sys.path.insert(0, str(tmp_path))
        try:
            import importlib as il
            m = il.import_module("importable_pkg")
            assert m.__doc__ is not None
        finally:
            sys.path.pop(0)
            sys.modules.pop("importable_pkg", None)


# ---------------------------------------------------------------------------
# TestRequirementsHygiene
# ---------------------------------------------------------------------------

class TestRequirementsHygiene:
    """Ensure requirements.txt no longer contains removed dead deps."""

    @pytest.fixture
    def req_text(self):
        req_file = Path(__file__).resolve().parent.parent / "requirements.txt"
        if not req_file.exists():
            pytest.skip("requirements.txt not found")
        return req_file.read_text()

    def test_yfinance_removed(self, req_text):
        uncommented = [line for line in req_text.splitlines() if not line.startswith("#")]
        assert not any("yfinance" in line for line in uncommented), (
            "yfinance must be removed from requirements.txt (zero prod usage)"
        )

    def test_asyncpg_removed(self, req_text):
        uncommented = [line for line in req_text.splitlines() if not line.startswith("#")]
        assert not any("asyncpg" in line for line in uncommented), (
            "asyncpg must be removed from requirements.txt (replaced by aiosqlite)"
        )

    def test_pytz_is_pinned(self, req_text):
        uncommented = [line for line in req_text.splitlines() if not line.startswith("#")]
        pytz_lines = [l for l in uncommented if "pytz" in l]
        assert pytz_lines, "pytz must still be present in requirements.txt"
        # Must have a version specifier, not bare 'pytz'
        assert any(">=" in l or "==" in l or "~=" in l for l in pytz_lines), (
            "pytz must be pinned (e.g. pytz>=2024.1), not left bare"
        )


# ---------------------------------------------------------------------------
# TestStartupConfigWiring
# ---------------------------------------------------------------------------

class TestStartupConfigWiring:
    """Tests for core/startup.py startup_config_check and get_config."""

    def _make_cfg_yaml(self, tmp_path, mode="dry_run", capital=5000.0, max_dd=10.0):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent(f"""
                system:
                  mode: {mode}
                  initial_capital: {capital}
                risk:
                  max_drawdown_pct: {max_dd}
                exchanges:
                  kraken:
                    enabled: true
            """)
        )
        return cfg_file

    def test_config_stored_in_shared_state(self, tmp_path):
        cfg_file = self._make_cfg_yaml(tmp_path)
        fake_shared = {}
        fake_ss = MagicMock()
        fake_ss.instance.return_value = fake_shared

        with patch.dict(sys.modules, {
            "core.shared_state": MagicMock(SharedState=fake_ss),
            "core.config_schema": None,  # trigger SimpleNamespace fallback
        }):
            repo_root = Path(__file__).resolve().parent.parent
            spec = importlib.util.spec_from_file_location(
                "startup_test", repo_root / "core" / "startup.py"
            )
            startup_mod = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(startup_mod)  # type: ignore

            cfg = startup_mod.startup_config_check(config_path=cfg_file)

        assert "argus_config" in fake_shared
        assert cfg is fake_shared["argus_config"]

    def test_capital_override_applied(self, tmp_path):
        cfg_file = self._make_cfg_yaml(tmp_path, capital=1000.0)

        with patch.dict(sys.modules, {
            "core.shared_state": MagicMock(SharedState=MagicMock(
                instance=MagicMock(return_value={})
            )),
            "core.config_schema": None,
        }):
            repo_root = Path(__file__).resolve().parent.parent
            spec = importlib.util.spec_from_file_location(
                "startup_test2", repo_root / "core" / "startup.py"
            )
            startup_mod = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(startup_mod)  # type: ignore

            cfg = startup_mod.startup_config_check(
                config_path=cfg_file,
                capital_override=99999.0,
            )

        assert cfg.system.initial_capital == 99999.0

    def test_missing_config_raises_file_not_found(self, tmp_path):
        repo_root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "startup_test3", repo_root / "core" / "startup.py"
        )
        startup_mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(startup_mod)  # type: ignore

        with pytest.raises(FileNotFoundError):
            startup_mod.startup_config_check(
                config_path=str(tmp_path / "nonexistent.yaml")
            )

    def test_get_config_returns_none_before_startup(self):
        with patch.dict(sys.modules, {
            "core.shared_state": MagicMock(
                SharedState=MagicMock(instance=MagicMock(return_value={}))
            ),
        }):
            repo_root = Path(__file__).resolve().parent.parent
            spec = importlib.util.spec_from_file_location(
                "startup_test4", repo_root / "core" / "startup.py"
            )
            startup_mod = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(startup_mod)  # type: ignore

            result = startup_mod.get_config()
            # SharedState is empty — should return None
            assert result is None
