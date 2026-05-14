"""Argus CLI root — Push 63.

Usage::

    argus --help
    argus start [--config argus.yaml]
    argus backtest --strategy MomentumStrategy --data data/btc.csv
    argus version
    argus doctor
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
except ImportError as exc:
    print(f"[argus] Missing dependency: {exc}. Run: pip install typer rich", file=sys.stderr)
    sys.exit(1)

app = typer.Typer(
    name="argus",
    help="Argus Ultimate — algorithmic trading bot CLI",
    add_completion=True,
    no_args_is_help=True,
)
console = Console()

# ------------------------------------------------------------------
# Global options (injected via callback)
# ------------------------------------------------------------------

_config_path: Optional[Path] = None


@app.callback()
def _global(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to argus.yaml config file",
        envvar="ARGUS_CONFIG_PATH",
    ),
) -> None:
    global _config_path
    _config_path = config


# ------------------------------------------------------------------
# Sub-commands (imported lazily to keep startup fast)
# ------------------------------------------------------------------

@app.command()
def start(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port"),
    workers: int = typer.Option(1, "--workers", "-w", help="Uvicorn workers"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and exit"),
) -> None:
    """Start the Argus trading engine."""
    from cli.cmd_start import run_start
    run_start(
        config_path=config or _config_path,
        host=host,
        port=port,
        workers=workers,
        dry_run=dry_run,
    )


@app.command()
def backtest(
    strategy: str = typer.Option("MomentumStrategy", "--strategy", "-s", help="Strategy class name"),
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="OHLCV CSV/Parquet file"),
    symbols: str = typer.Option("BTCUSDT", "--symbols", help="Comma-separated symbols"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date YYYY-MM-DD"),
    end: Optional[str] = typer.Option(None, "--end", help="End date YYYY-MM-DD"),
    equity: float = typer.Option(10_000.0, "--equity", "-e", help="Initial equity USD"),
    fee: float = typer.Option(2.0, "--fee", help="Fee in basis points"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    n_bars: int = typer.Option(500, "--bars", help="Synthetic bars if no data file"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file"),
) -> None:
    """Run a historical backtest."""
    from cli.cmd_backtest import run_backtest
    run_backtest(
        strategy_name=strategy,
        data_path=data,
        symbols=[s.strip() for s in symbols.split(",")],
        start_date=start,
        end_date=end,
        initial_equity=equity,
        fee_bps=fee,
        output_dir=output,
        n_bars=n_bars,
        config_path=config or _config_path,
    )


@app.command()
def version() -> None:
    """Show Argus version and dependency info."""
    from cli.cmd_version import run_version
    run_version(console)


@app.command()
def doctor() -> None:
    """Run health checks and import probes."""
    from cli.cmd_doctor import run_doctor
    ok = run_doctor(console)
    if not ok:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
