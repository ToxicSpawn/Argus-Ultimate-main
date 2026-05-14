"""Push 74 — Tests: CI workflow YAML, release workflow, Makefile targets,
pyproject.toml config. 16 tests.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# CI workflow (4)
# ---------------------------------------------------------------------------

class TestCIWorkflow:
    def _load(self):
        try:
            import yaml
            return yaml.safe_load(
                (ROOT / ".github" / "workflows" / "ci.yml").read_text()
            )
        except ImportError:
            return None

    def test_file_exists(self):
        assert (ROOT / ".github" / "workflows" / "ci.yml").exists()

    def test_triggers_on_main(self):
        cfg = self._load()
        if cfg:
            branches = cfg["on"]["push"]["branches"]
            assert "main" in branches

    def test_lint_job_defined(self):
        cfg = self._load()
        if cfg:
            assert "lint" in cfg["jobs"]

    def test_test_job_defined(self):
        cfg = self._load()
        if cfg:
            assert "test" in cfg["jobs"]


# ---------------------------------------------------------------------------
# Release workflow (4)
# ---------------------------------------------------------------------------

class TestReleaseWorkflow:
    def _load(self):
        try:
            import yaml
            return yaml.safe_load(
                (ROOT / ".github" / "workflows" / "release.yml").read_text()
            )
        except ImportError:
            return None

    def test_file_exists(self):
        assert (ROOT / ".github" / "workflows" / "release.yml").exists()

    def test_triggers_on_vtag(self):
        cfg = self._load()
        if cfg:
            tags = cfg["on"]["push"]["tags"]
            assert any("v*" in t for t in tags)

    def test_docker_publish_job(self):
        cfg = self._load()
        if cfg:
            assert "docker-publish" in cfg["jobs"]

    def test_github_release_job(self):
        cfg = self._load()
        if cfg:
            assert "github-release" in cfg["jobs"]


# ---------------------------------------------------------------------------
# Security scan workflow (2)
# ---------------------------------------------------------------------------

class TestSecurityWorkflow:
    def test_file_exists(self):
        assert (ROOT / ".github" / "workflows" / "security_scan.yml").exists()

    def test_has_trivy_job(self):
        try:
            import yaml
            cfg = yaml.safe_load(
                (ROOT / ".github" / "workflows" / "security_scan.yml").read_text()
            )
            assert "trivy" in cfg["jobs"]
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Makefile (3)
# ---------------------------------------------------------------------------

class TestMakefile:
    def _mk(self):
        return (ROOT / "Makefile").read_text()

    def test_file_exists(self):
        assert (ROOT / "Makefile").exists()

    def test_test_target_exists(self):
        assert "test:" in self._mk()

    def test_docker_up_target_exists(self):
        assert "docker-up:" in self._mk()


# ---------------------------------------------------------------------------
# pyproject.toml (3)
# ---------------------------------------------------------------------------

class TestPyproject:
    def _load(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return None
        return tomllib.loads((ROOT / "pyproject.toml").read_text())

    def test_file_exists(self):
        assert (ROOT / "pyproject.toml").exists()

    def test_ruff_line_length(self):
        cfg = self._load()
        if cfg:
            assert cfg["tool"]["ruff"]["line-length"] == 100

    def test_pytest_asyncio_mode(self):
        cfg = self._load()
        if cfg:
            assert cfg["tool"]["pytest.ini_options"]["asyncio_mode"] == "auto"
