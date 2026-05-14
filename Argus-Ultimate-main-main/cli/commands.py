"""
Argus Trading System - CLI Commands
===================================

Click-based command-line interface for the trading system.

Commands:
- run: Start the trading bot
- status: Show current status
- backtest: Run strategy backtest
- config: Manage configuration
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging."""
    handlers = [
        RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
    ]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        handlers=handlers,
    )


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: Optional[str]) -> None:
    """Argus Trading System - Adaptive Crypto Trading Bot"""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["config_path"] = config or "config.yaml"

    if debug:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["live", "paper", "backtest"]),
    default="paper",
    help="Trading mode",
)
@click.option(
    "--exchange",
    type=click.Choice(["kraken", "coinbase", "both"]),
    default="kraken",
    help="Exchange to use",
)
@click.option("--symbols", "-s", multiple=True, help="Symbols to trade (e.g., BTC/AUD)")
@click.option("--dashboard", is_flag=True, help="Show live dashboard")
@click.pass_context
def run(
    ctx: click.Context,
    mode: str,
    exchange: str,
    symbols: tuple,
    dashboard: bool,
) -> None:
    """Start the trading bot."""
    console.print(f"[bold cyan]Starting Argus Trading System[/bold cyan]")
    console.print(f"  Mode: [yellow]{mode}[/yellow]")
    console.print(f"  Exchange: [yellow]{exchange}[/yellow]")
    console.print(f"  Symbols: [yellow]{', '.join(symbols) or 'default'}[/yellow]")

    if mode == "live":
        if not click.confirm(
            "\n⚠️  You are about to trade with REAL money. Continue?",
            default=False,
        ):
            console.print("[red]Aborted.[/red]")
            return

    async def _run():
        try:
            # Import here to avoid circular imports
            from exchanges.centralized.kraken import KrakenClient
            from execution.pipeline import ExecutionPipeline
            from cli.dashboard import Dashboard

            # Initialize exchange
            if exchange in ("kraken", "both"):
                kraken = KrakenClient(
                    api_key=os.environ.get("KRAKEN_API_KEY"),
                    secret=os.environ.get("KRAKEN_API_SECRET"),
                    dry_run=(mode != "live"),
                )
                await kraken.connect()

            # Create execution pipeline
            pipeline = ExecutionPipeline(
                exchange=kraken,
            )

            # Dashboard or simple logging
            if dashboard:
                dash = Dashboard()
                dash.log("Trading system started")
                await dash.start()
            else:
                console.print("[green]Trading bot running. Press Ctrl+C to stop.[/green]")

                # Simple run loop
                while True:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if ctx.obj.get("debug"):
                raise
        finally:
            # Cleanup
            if "kraken" in dir():
                await kraken.close()

    asyncio.run(_run())


@cli.command()
@click.option("--exchange", type=click.Choice(["kraken", "coinbase"]), default="kraken")
@click.pass_context
def status(ctx: click.Context, exchange: str) -> None:
    """Show current trading status."""

    async def _status():
        try:
            from exchanges.centralized.kraken import KrakenClient

            console.print(f"[cyan]Connecting to {exchange}...[/cyan]")

            if exchange == "kraken":
                client = KrakenClient(dry_run=True)
                await client.connect()

                # Fetch balance
                balance = await client.fetch_balance()

                console.print("\n[bold]Account Balance:[/bold]")
                for currency, amounts in balance.items():
                    if isinstance(amounts, dict) and amounts.get("total", 0) > 0:
                        console.print(f"  {currency}: {amounts['total']:.8f}")

                await client.close()

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if ctx.obj.get("debug"):
                raise

    asyncio.run(_status())


@cli.command()
@click.option("--strategy", "-s", help="Strategy name (informational; unified engine uses config)")
@click.option("--symbol", required=True, help="Symbol to test (e.g., BTC/USD)")
@click.option("--csv", "csv_path", type=click.Path(exists=True), help="OHLCV CSV (timestamp,open,high,low,close,volume). Required for unified backtest.")
@click.option("--days", type=int, default=30, help="Days of data (used when CSV has no date range)")
@click.option("--start", help="Start date (YYYY-MM-DD) – informational")
@click.option("--end", help="End date (YYYY-MM-DD) – informational")
@click.option("--capital", type=float, default=1000.0, help="Starting capital in AUD")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None, help="Config file (default: unified_config.yaml)")
@click.pass_context
def backtest(
    ctx: click.Context,
    strategy: Optional[str],
    symbol: str,
    csv_path: Optional[str],
    days: int,
    start: Optional[str],
    end: Optional[str],
    capital: float,
    config_path: Optional[str],
) -> None:
    """Run unified backtest. Requires --csv with OHLCV (timestamp,open,high,low,close,volume)."""
    console.print(f"[bold]Backtesting on {symbol}[/bold]")
    console.print(f"  Capital: ${capital:,.2f}")
    if strategy:
        console.print(f"  Strategy: {strategy}")
    cfg = config_path or ctx.obj.get("config_path") or "unified_config.yaml"
    if not Path(cfg).exists():
        cfg = "unified_config.yaml"
    csv_file = csv_path or (Path("data") / "ohlcv.csv" if (Path("data") / "ohlcv.csv").exists() else None)
    if not csv_file:
        console.print("\n[yellow]No CSV provided. For unified backtest use:[/yellow]")
        console.print("  [cyan]python main.py backtest --unified-backtest --csv <path> --symbol BTC/USD --days 30 --capital 1000[/cyan]")
        console.print("  Or: [cyan]argus backtest --symbol BTC/USD --csv data/ohlcv.csv[/cyan]")
        return
    try:
        from main import run_backtesting
        run_backtesting(symbol=symbol, days=days, capital=capital, unified=True, csv=str(csv_file), config_file=cfg if Path(cfg).exists() else None)
        console.print("[green]Backtest completed.[/green]")
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")
        if ctx.obj.get("debug"):
            raise


@cli.command()
@click.argument("action", type=click.Choice(["show", "validate", "generate"]))
@click.pass_context
def config(ctx: click.Context, action: str) -> None:
    """Manage configuration."""
    config_path = ctx.obj.get("config_path", "config.yaml")

    if action == "show":
        if Path(config_path).exists():
            with open(config_path) as f:
                console.print(f.read())
        else:
            console.print(f"[red]Config file not found: {config_path}[/red]")

    elif action == "validate":
        if Path(config_path).exists():
            try:
                import yaml
                with open(config_path) as f:
                    yaml.safe_load(f)
                console.print("[green]✓ Configuration is valid YAML[/green]")
            except Exception as e:
                console.print(f"[red]✗ Invalid configuration: {e}[/red]")
        else:
            console.print(f"[red]Config file not found: {config_path}[/red]")

    elif action == "generate":
        template = """# Argus Trading System Configuration
# ===================================

system:
  mode: paper  # paper, live
  log_level: INFO
  timezone: Australia/Sydney

exchanges:
  kraken:
    enabled: true
    api_key: ${KRAKEN_API_KEY}
    api_secret: ${KRAKEN_API_SECRET}

  coinbase:
    enabled: false
    api_key: ${COINBASE_API_KEY}
    api_secret: ${COINBASE_API_SECRET}

trading:
  symbols:
    - BTC/AUD
    - ETH/AUD

  capital_aud: 1000.0
  max_position_pct: 0.20
  max_drawdown_pct: 0.10

strategies:
  momentum:
    enabled: true
    weight: 0.3

  mean_reversion:
    enabled: true
    weight: 0.3

  breakout:
    enabled: true
    weight: 0.2

  scalping:
    enabled: false
    weight: 0.2

alerts:
  discord_webhook: ${DISCORD_WEBHOOK_URL}
  telegram_bot_token: ${TELEGRAM_BOT_TOKEN}
  telegram_chat_id: ${TELEGRAM_CHAT_ID}
"""
        output_path = "config.example.yaml"
        with open(output_path, "w") as f:
            f.write(template)
        console.print(f"[green]Generated example config: {output_path}[/green]")


@cli.command()
def version() -> None:
    """Show version information."""
    console.print("[bold cyan]Argus Trading System[/bold cyan]")
    console.print("  Version: 2.0.0")
    console.print("  Python: " + sys.version.split()[0])


@cli.command()
@click.option("--exchange", type=click.Choice(["kraken", "coinbase"]), default="kraken")
def test_connection(exchange: str) -> None:
    """Test exchange connection."""

    async def _test():
        console.print(f"[cyan]Testing connection to {exchange}...[/cyan]")

        try:
            if exchange == "kraken":
                from exchanges.centralized.kraken import KrakenClient
                client = KrakenClient(dry_run=True)
            else:
                console.print("[yellow]Coinbase test not implemented[/yellow]")
                return

            # Test connection
            connected = await client.connect()
            if connected:
                console.print("[green]✓ Connected successfully[/green]")

                # Test ticker
                ticker = await client.fetch_ticker("BTC/AUD")
                console.print(f"  BTC/AUD: ${ticker.get('last', 'N/A')}")

                # Test ping
                if await client.ping():
                    console.print("[green]✓ Ping successful[/green]")
            else:
                console.print("[red]✗ Connection failed[/red]")

            await client.close()

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")

    asyncio.run(_test())


if __name__ == "__main__":
    cli()
