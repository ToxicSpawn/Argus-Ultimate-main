"""argus doctor command implementation — Push 63.

Runs:
  1. Health checks (disk, memory, event_loop)
  2. Import probes for all optional dependencies

Prints colour-coded Rich table.
Returns True if all checks pass, False if any UNHEALTHY.
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from typing import List, Tuple


_OPTIONAL_DEPS = [
    ("fastapi", "FastAPI web framework"),
    ("uvicorn", "ASGI server"),
    ("aiohttp", "Async HTTP client"),
    ("pydantic", "Data validation"),
    ("pandas", "DataFrame / data feed"),
    ("numpy", "Numerical computing"),
    ("typer", "CLI framework"),
    ("rich", "Terminal UI"),
    ("yaml", "YAML config loader (PyYAML)"),
    ("prometheus_client", "Prometheus metrics"),
    ("optuna", "Hyperparameter tuning"),
    ("psutil", "System resource checks"),
    ("websockets", "WebSocket server"),
    ("matplotlib", "Equity curve plots"),
    ("torch", "PyTorch (AI models)"),
]

_REQUIRED_DEPS = [
    ("typer", "Typer CLI"),
    ("rich", "Rich console"),
]


def run_doctor(console) -> bool:
    from rich.table import Table
    from rich.panel import Panel

    console.print(Panel(
        "[bold]Running Argus system diagnostics...[/bold]",
        border_style="yellow",
    ))

    all_ok = True

    # ------------------------------------------------------------------
    # 1. Health checks
    # ------------------------------------------------------------------
    from core.health.health_registry import HealthRegistry
    from core.health.builtin_checks import disk_check, memory_check, event_loop_check
    from core.health.health_models import HealthStatus

    reg = HealthRegistry()
    reg.register_check("disk", disk_check("/", min_free_mb=100))
    reg.register_check("memory", memory_check(max_pct=95.0))
    reg.register_check("event_loop", event_loop_check())

    system = asyncio.get_event_loop().run_until_complete(reg.run_checks())

    health_table = Table(title="System Health", border_style="cyan")
    health_table.add_column("Check", style="bold")
    health_table.add_column("Status", justify="center")
    health_table.add_column("Message")
    health_table.add_column("Latency", justify="right")

    for name, comp in system.components.items():
        if comp.status == HealthStatus.HEALTHY:
            status_str = "[green]✓ healthy[/green]"
        elif comp.status == HealthStatus.DEGRADED:
            status_str = "[yellow]⚠ degraded[/yellow]"
        else:
            status_str = "[red]✗ unhealthy[/red]"
            all_ok = False
        health_table.add_row(
            name,
            status_str,
            comp.message,
            f"{comp.latency_ms:.1f}ms",
        )
    console.print(health_table)

    # ------------------------------------------------------------------
    # 2. Import probes
    # ------------------------------------------------------------------
    dep_table = Table(title="Dependency Probes", border_style="cyan")
    dep_table.add_column("Package", style="bold")
    dep_table.add_column("Status", justify="center")
    dep_table.add_column("Version", justify="right")
    dep_table.add_column("Notes")

    for module, notes in _OPTIONAL_DEPS:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "installed")
            required = module in dict(_REQUIRED_DEPS)
            dep_table.add_row(
                module,
                "[green]✓ ok[/green]",
                ver,
                notes,
            )
        except ImportError:
            required = module in dict(_REQUIRED_DEPS)
            if required:
                all_ok = False
            status = "[red]✗ missing[/red]" if required else "[yellow]○ optional[/yellow]"
            dep_table.add_row(module, status, "—", notes)

    console.print(dep_table)

    # ------------------------------------------------------------------
    # 3. Core module probes
    # ------------------------------------------------------------------
    core_modules = [
        "core.config",
        "core.health",
        "core.alerts",
        "core.backtest",
        "core.broadcast",
        "core.execution",
        "core.pnl",
        "core.risk",
        "core.strategy",
    ]
    core_table = Table(title="Core Modules", border_style="cyan")
    core_table.add_column("Module", style="bold")
    core_table.add_column("Status", justify="center")

    for mod_name in core_modules:
        try:
            importlib.import_module(mod_name)
            core_table.add_row(mod_name, "[green]✓ ok[/green]")
        except Exception as exc:  # noqa: BLE001
            core_table.add_row(mod_name, f"[red]✗ {exc}[/red]")
            all_ok = False

    console.print(core_table)

    # Final verdict
    if all_ok:
        console.print("\n[bold green]✓ All checks passed — Argus is healthy.[/bold green]")
    else:
        console.print("\n[bold red]✗ Some checks failed — review the tables above.[/bold red]")

    return all_ok
