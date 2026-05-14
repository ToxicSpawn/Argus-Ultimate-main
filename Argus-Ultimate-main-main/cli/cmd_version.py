"""argus version command implementation — Push 63."""
from __future__ import annotations

import platform
import sys


def run_version(console) -> None:
    from rich.table import Table
    from rich.panel import Panel

    from version import __version__, __codename__

    console.print(Panel(
        f"[bold cyan]Argus Ultimate[/bold cyan] v[green]{__version__}[/green] "
        f"([yellow]{__codename__}[/yellow])",
        border_style="cyan",
    ))

    table = Table(title="Environment", border_style="dim")
    table.add_column("Component", style="bold")
    table.add_column("Version", justify="right")

    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    table.add_row("Platform", platform.platform())

    deps = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("aiohttp", "aiohttp"),
        ("pydantic", "pydantic"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("prometheus_client", "prometheus_client"),
        ("optuna", "optuna"),
        ("yaml", "PyYAML"),
        ("psutil", "psutil"),
    ]

    for module, display in deps:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "installed")
        except ImportError:
            ver = "[red]not installed[/red]"
        table.add_row(display, ver)

    console.print(table)
