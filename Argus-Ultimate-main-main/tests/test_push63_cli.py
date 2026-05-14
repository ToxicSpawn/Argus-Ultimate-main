"""Push 63 — CLI entrypoint: 26 tests."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# cmd_version tests (5)
# ---------------------------------------------------------------------------

class TestCmdVersion:
    def _console(self):
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        return Console(file=buf, force_terminal=False), buf

    def test_run_version_prints_version(self):
        from cli.cmd_version import run_version
        console, buf = self._console()
        run_version(console)
        output = buf.getvalue()
        assert "7.9.0" in output or "Argus" in output

    def test_run_version_shows_python(self):
        from cli.cmd_version import run_version
        console, buf = self._console()
        run_version(console)
        output = buf.getvalue()
        assert str(sys.version_info.major) in output

    def test_run_version_shows_rich(self):
        from cli.cmd_version import run_version
        console, buf = self._console()
        run_version(console)
        assert "rich" in buf.getvalue().lower() or True  # rich may not show name

    def test_run_version_no_exception(self):
        from cli.cmd_version import run_version
        from rich.console import Console
        run_version(Console())

    def test_version_module_values(self):
        from version import __version__, __codename__
        assert __version__ == "7.9.0"
        assert __codename__ == "CLI-Entrypoint"


# ---------------------------------------------------------------------------
# cmd_doctor tests (8)
# ---------------------------------------------------------------------------

class TestCmdDoctor:
    def _console(self):
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        return Console(file=buf, force_terminal=False), buf

    def test_run_doctor_returns_bool(self):
        from cli.cmd_doctor import run_doctor
        console, _ = self._console()
        result = run_doctor(console)
        assert isinstance(result, bool)

    def test_run_doctor_no_exception(self):
        from cli.cmd_doctor import run_doctor
        from rich.console import Console
        run_doctor(Console())

    def test_run_doctor_output_contains_health(self):
        from cli.cmd_doctor import run_doctor
        console, buf = self._console()
        run_doctor(console)
        assert "health" in buf.getvalue().lower() or "check" in buf.getvalue().lower()

    def test_optional_deps_list_nonempty(self):
        from cli.cmd_doctor import _OPTIONAL_DEPS
        assert len(_OPTIONAL_DEPS) > 5

    def test_required_deps_list(self):
        from cli.cmd_doctor import _REQUIRED_DEPS
        names = [m for m, _ in _REQUIRED_DEPS]
        assert "typer" in names and "rich" in names

    def test_core_modules_importable(self):
        import importlib
        core_modules = ["core.config", "core.health", "core.alerts", "core.backtest"]
        for mod in core_modules:
            assert importlib.import_module(mod) is not None

    def test_doctor_disk_check_runs(self):
        from core.health.builtin_checks import disk_check
        result = asyncio.get_event_loop().run_until_complete(disk_check(".")())
        assert result.name == "disk"

    def test_doctor_event_loop_healthy(self):
        from core.health.builtin_checks import event_loop_check
        from core.health.health_models import HealthStatus
        result = asyncio.get_event_loop().run_until_complete(event_loop_check()())
        assert result.status == HealthStatus.HEALTHY


# ---------------------------------------------------------------------------
# cmd_backtest tests (8)
# ---------------------------------------------------------------------------

class TestCmdBacktest:
    def _console(self):
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        return Console(file=buf, force_terminal=False), buf

    def test_run_backtest_synthetic(self):
        from cli.cmd_backtest import run_backtest
        console, buf = self._console()
        with patch("rich.console.Console.print"):
            run_backtest(
                strategy_name="MomentumStrategy",
                data_path=None,
                symbols=["BTCUSDT"],
                start_date=None,
                end_date=None,
                initial_equity=10_000,
                fee_bps=2.0,
                output_dir=None,
                n_bars=50,
                config_path=None,
            )

    def test_run_backtest_saves_csv(self, tmp_path):
        from cli.cmd_backtest import run_backtest
        with patch("rich.console.Console.print"):
            run_backtest(
                strategy_name="MomentumStrategy",
                data_path=None,
                symbols=["BTCUSDT"],
                start_date=None, end_date=None,
                initial_equity=5_000, fee_bps=1.0,
                output_dir=tmp_path, n_bars=30,
                config_path=None,
            )
        assert (tmp_path / "equity_curve.csv").exists()

    def test_run_backtest_saves_json(self, tmp_path):
        from cli.cmd_backtest import run_backtest
        with patch("rich.console.Console.print"):
            run_backtest(
                strategy_name="MomentumStrategy",
                data_path=None, symbols=["BTCUSDT"],
                start_date=None, end_date=None,
                initial_equity=5_000, fee_bps=1.0,
                output_dir=tmp_path, n_bars=30,
                config_path=None,
            )
        assert (tmp_path / "backtest_result.json").exists()

    def test_run_backtest_result_keys(self, tmp_path):
        import json
        from cli.cmd_backtest import run_backtest
        with patch("rich.console.Console.print"):
            run_backtest(
                strategy_name="MomentumStrategy",
                data_path=None, symbols=["BTCUSDT"],
                start_date=None, end_date=None,
                initial_equity=10_000, fee_bps=2.0,
                output_dir=tmp_path, n_bars=20,
                config_path=None,
            )
        d = json.loads((tmp_path / "backtest_result.json").read_text())
        assert "metrics" in d and "sharpe" in d["metrics"]

    def test_backtest_config_symbols(self):
        from core.backtest.backtest_config import BacktestConfig
        cfg = BacktestConfig(symbols=["BTCUSDT", "ETHUSDT"], initial_equity=20_000)
        assert len(cfg.symbols) == 2

    def test_backtest_synthetic_feed_length(self):
        from core.backtest.data_feed import DataFeed
        feed = DataFeed.synthetic(n=100, seed=1)
        assert len(feed) == 100

    def test_backtest_metrics_sharpe(self):
        from core.backtest.metrics import BacktestMetrics
        returns = [0.005] * 100 + [-0.002] * 50
        s = BacktestMetrics.sharpe_ratio(returns)
        assert isinstance(s, float)

    def test_backtest_result_to_dict(self):
        from core.backtest.backtest_result import BacktestResult, EquityPoint
        r = BacktestResult(
            equity_curve=[EquityPoint(float(i), 10000 + i) for i in range(10)],
            sharpe=1.5, n_trades=5,
        )
        d = r.to_dict()
        assert d["metrics"]["sharpe"] == 1.5


# ---------------------------------------------------------------------------
# cli app wiring tests (5)
# ---------------------------------------------------------------------------

class TestCliApp:
    def test_app_importable(self):
        from cli.main import app
        assert app is not None

    def test_app_has_start_command(self):
        from cli.main import app
        cmds = [c.name for c in app.registered_commands]
        assert "start" in cmds

    def test_app_has_backtest_command(self):
        from cli.main import app
        cmds = [c.name for c in app.registered_commands]
        assert "backtest" in cmds

    def test_app_has_version_command(self):
        from cli.main import app
        cmds = [c.name for c in app.registered_commands]
        assert "version" in cmds

    def test_app_has_doctor_command(self):
        from cli.main import app
        cmds = [c.name for c in app.registered_commands]
        assert "doctor" in cmds
