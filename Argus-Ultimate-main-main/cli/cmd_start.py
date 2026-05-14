"""argus start command implementation — Push 63."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_start(
    config_path: Optional[Path],
    host: str,
    port: int,
    workers: int,
    dry_run: bool,
) -> None:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    from core.config.config_loader import ConfigLoader
    loader = ConfigLoader()
    cfg = loader.load(config_path)
    cfg.logging.apply()

    # Override from CLI flags
    cfg.server.host = host
    cfg.server.port = port
    cfg.server.workers = workers

    from version import __version__, __codename__
    console.print(Panel(
        f"[bold cyan]Argus Ultimate v{__version__} ({__codename__})[/bold cyan]\n"
        f"env=[yellow]{cfg.env}[/yellow]  "
        f"exchange=[green]{cfg.exchange.name}[/green]  "
        f"host=[white]{host}:{port}[/white]",
        title="Starting Argus",
        border_style="cyan",
    ))

    if dry_run:
        console.print("[green]✓ Dry run: config valid[/green]")
        return

    # ------------------------------------------------------------------
    # Build FastAPI app
    # ------------------------------------------------------------------
    try:
        from fastapi import FastAPI
        import uvicorn
    except ImportError as exc:
        console.print(f"[red]Missing dependency: {exc}[/red]")
        sys.exit(1)

    from core.health.health_registry import HealthRegistry
    from core.health.health_router import health_router
    from core.health.builtin_checks import disk_check, memory_check, event_loop_check
    from core.alerts.alert_manager import AlertManager
    from core.broadcast.ws_hub import WsHub
    from core.broadcast.ws_router import ws_router
    from core.pnl.pnl_tracker import PnLTracker
    from core.risk.risk_manager import RiskManager
    from core.execution.execution_engine import ExecutionEngine
    from core.strategy.strategy_registry import StrategyRegistry
    from core.strategy.strategy_runner import StrategyRunner
    from core.config.config_watcher import ConfigWatcher

    start_time = time.time()
    registry = HealthRegistry(
        version=__version__,
        env=cfg.env,
        start_time=start_time,
    )
    registry.register_check("disk", disk_check("/"))
    registry.register_check("memory", memory_check())
    registry.register_check("event_loop", event_loop_check())

    alert_mgr = AlertManager()
    ws_hub = WsHub()
    pnl = PnLTracker(initial_equity=cfg.risk.max_position_usd)
    risk_cfg = cfg.risk.to_risk_config()
    rm = RiskManager(risk_cfg, pnl_tracker=pnl, alert_manager=alert_mgr)
    exec_engine = ExecutionEngine(pnl_tracker=pnl, paper_trading=(cfg.env != "production"))
    strat_reg = StrategyRegistry()
    runner = StrategyRunner(strat_reg)

    app = FastAPI(title="Argus Ultimate", version=__version__)
    _hr = health_router(registry)
    _wr = ws_router(ws_hub)
    if _hr:
        app.include_router(_hr)
    if _wr:
        app.include_router(_wr)

    # Config hot-reload watcher
    watcher = ConfigWatcher(loader, interval=10.0)
    watcher.add_callback(lambda new_cfg: logger.info("Config reloaded: env=%s", new_cfg.env))

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(watcher.watch())
        await alert_mgr.start_worker()
        logger.info("Argus started on %s:%d", host, port)

    @app.on_event("shutdown")
    async def _shutdown():
        watcher.stop()
        await alert_mgr.stop_worker()
        logger.info("Argus shutdown complete")

    console.print(f"[cyan]Listening on http://{host}:{port}[/cyan]")
    uvicorn.run(app, host=host, port=port, workers=workers, log_level="warning")
