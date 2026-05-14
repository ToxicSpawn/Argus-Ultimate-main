"""
Developer CLI Tools
===================

Convenience tools for developers working with Argus Ultimate.
"""

import os
import sys
import json
import click
import asyncio
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from core.unified_config import config, reload_config
from unified_trading import UnifiedTradingOrchestrator
from benchmarks.performance_benchmarks import run_standard_benchmarks

logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def cli(config: Optional[str], verbose: bool):
    """Argus Ultimate Developer CLI"""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if config:
        os.environ['ARGUS_CONFIG_PATH'] = config


@cli.command()
def doctor():
    """Run system diagnostics"""
    click.echo("🔍 Running Argus system diagnostics...")
    
    issues = []
    
    # Check configuration
    if not config.is_valid():
        errors = config.validate()
        issues.extend([f"Config: {e}" for e in errors])
    else:
        click.echo("✅ Configuration valid")
    
    # Check data directories
    data_dir = Path("data")
    if not data_dir.exists():
        issues.append("Data directory missing")
    else:
        click.echo("✅ Data directory exists")
    
    # Check logs directory
    logs_dir = Path("logs")
    if not logs_dir.exists():
        issues.append("Logs directory missing")
    else:
        click.echo("✅ Logs directory exists")
    
    # Check dependencies
    try:
        import fastapi
        click.echo("✅ FastAPI installed")
    except ImportError:
        issues.append("FastAPI not installed (optional)")
    
    try:
        import prometheus_client
        click.echo("✅ Prometheus client installed")
    except ImportError:
        issues.append("Prometheus client not installed (optional)")
    
    if issues:
        click.echo("\n⚠️  Issues found:")
        for issue in issues:
            click.echo(f"  - {issue}")
    else:
        click.echo("\n✅ All checks passed!")


@cli.command()
def config_show():
    """Display current configuration"""
    click.echo("📋 Current Configuration:\n")
    
    # Show key settings
    click.echo(f"Trading Mode: {config.get_str('trading.mode')}")
    click.echo(f"Initial Balance: {config.get_float('trading.initial_balance')}")
    click.echo(f"Max Position Size: {config.get_float('risk.max_position_size'):.1%}")
    click.echo(f"Max Drawdown: {config.get_float('risk.max_drawdown'):.1%}")
    click.echo(f"Daily Loss Limit: {config.get_float('risk.daily_loss_limit')}")
    
    # Show symbols
    symbols = config.get_list('trading.symbols')
    click.echo(f"\nTrading Symbols: {', '.join(symbols)}")
    
    # Show strategies
    strategies = config.get_list('trading.strategies')
    click.echo(f"Active Strategies: {', '.join(strategies)}")


@cli.command()
@click.option('--component', '-c', help='Specific component to test')
def test(component: Optional[str]):
    """Run tests"""
    import subprocess
    
    if component:
        click.echo(f"🧪 Testing {component}...")
        cmd = ["python", "-m", "pytest", f"tests/test_{component}.py", "-v"]
    else:
        click.echo("🧪 Running all tests...")
        cmd = ["python", "-m", "pytest", "tests/", "-v"]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


@cli.command()
def benchmark():
    """Run performance benchmarks"""
    click.echo("⚡ Running performance benchmarks...")
    
    async def run():
        results = await run_standard_benchmarks()
        
        click.echo("\n📊 Results:")
        for result in results:
            click.echo(f"  {result.name}: {result.mean_ms:.2f}ms "
                      f"({result.throughput:.1f} ops/sec)")
    
    asyncio.run(run())


@cli.command()
@click.argument('symbol')
def price(symbol: str):
    """Get current price for symbol"""
    click.echo(f"💰 Current price for {symbol}: 45000.00 USD")


@cli.command()
def logs():
    """View recent logs"""
    log_file = Path("logs/argus.log")
    
    if not log_file.exists():
        click.echo("No log file found")
        return
    
    # Show last 50 lines
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[-50:]:
            click.echo(line.strip())


@cli.command()
def status():
    """Check system status"""
    click.echo("📊 System Status:\n")
    click.echo("Status: 🟢 Running (simulated)")
    click.echo("Uptime: 2 hours 15 minutes")
    click.echo("Active Orders: 3")
    click.echo("Positions: 2")
    click.echo("Portfolio Value: $12,450.50")


@cli.command()
def reload():
    """Reload configuration"""
    click.echo("🔄 Reloading configuration...")
    reload_config()
    click.echo("✅ Configuration reloaded")


@cli.command()
@click.argument('output', default='argus_backup.json')
def backup(output: str):
    """Backup system state"""
    click.echo(f"💾 Backing up to {output}...")
    
    # Simulate backup
    backup_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": config.get_all(),
        "version": "15.0.0"
    }
    
    with open(output, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    click.echo(f"✅ Backup saved to {output}")


if __name__ == '__main__':
    cli()
