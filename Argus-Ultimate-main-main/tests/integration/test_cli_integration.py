"""Push 65 — CLI integration tests: 8 tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from typer.testing import CliRunner
    _TYPER_AVAILABLE = True
except ImportError:
    _TYPER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TYPER_AVAILABLE, reason="Typer not installed")


@pytest.fixture(scope="module")
def runner():
    from typer.testing import CliRunner
    return CliRunner()


@pytest.fixture(scope="module")
def app():
    from cli.main import app
    return app


class TestCliVersion:
    def test_version_exit_zero(self, runner, app):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0

    def test_version_shows_version_number(self, runner, app):
        result = runner.invoke(app, ["version"])
        assert "8.1.0" in result.output or "Argus" in result.output

    def test_version_shows_python(self, runner, app):
        result = runner.invoke(app, ["version"])
        assert str(sys.version_info.major) in result.output


class TestCliDoctor:
    def test_doctor_runs(self, runner, app):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code in (0, 1)

    def test_doctor_output_nonempty(self, runner, app):
        result = runner.invoke(app, ["doctor"])
        assert len(result.output) > 10

    def test_doctor_shows_core_modules(self, runner, app):
        result = runner.invoke(app, ["doctor"])
        assert "core" in result.output.lower() or "health" in result.output.lower()


class TestCliHelp:
    def test_help_exit_zero(self, runner, app):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_commands(self, runner, app):
        result = runner.invoke(app, ["--help"])
        assert "start" in result.output
        assert "backtest" in result.output
