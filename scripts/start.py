"""Push 80 — Argus Ultimate start script.

Loads config from:
  1. config/default_config.yml  (base)
  2. config/config.yml          (override, optional)
  3. Environment variables      (ARGUS_ prefix, highest priority)

Starts:
  - ArgusSystem (all trading subsystems)
  - uvicorn serving FastAPI on api_host:api_port
  - Both run in the same asyncio event loop

Usage:
  python scripts/start.py
  ARGUS_PAPER_MODE=false ARGUS_API_PORT=9000 python scripts/start.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_config() -> dict:
    """Load config from YAML files + env overrides."""
    config = {}

    # Load YAML if available
    try:
        import yaml
        for fname in ("config/default_config.yml", "config/config.yml"):
            path = ROOT / fname
            if path.exists():
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                    config.update(data)
    except ImportError:
        pass

    # Environment overrides (ARGUS_ prefix, snake_case)
    env_map = {
        "ARGUS_PAPER_MODE":         ("paper_mode",       lambda v: v.lower() == "true"),
        "ARGUS_INITIAL_EQUITY":     ("initial_equity",   float),
        "ARGUS_INITIAL_BALANCE":    ("initial_balance",  float),
        "ARGUS_API_HOST":           ("api_host",         str),
        "ARGUS_API_PORT":           ("api_port",         int),
        "ARGUS_LOG_LEVEL":          ("log_level",        str),
        "ARGUS_BINANCE_API_KEY":    ("binance_api_key",  str),
        "ARGUS_BINANCE_API_SECRET": ("binance_api_secret", str),
        "ARGUS_BINANCE_TESTNET":    ("binance_testnet",  lambda v: v.lower() == "true"),
        "ARGUS_SIGNAL_COOLDOWN":    ("signal_cooldown",  float),
        "ARGUS_MAX_OPEN_ORDERS":    ("max_open_orders",  int),
        "ARGUS_MAX_POSITION_USD":   ("max_position_usd", float),
    }
    for env_key, (cfg_key, cast) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                config[cfg_key] = cast(val)
            except (ValueError, TypeError):
                pass

    return config


async def _main() -> None:
    config = _load_config()
    log_level = config.get("log_level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("argus.start")

    from core.system import ArgusSystem
    system = ArgusSystem.from_config(config)
    system._build()

    log.info("Argus Ultimate v8.16.0 / Integration starting...")
    log.info(f"Paper mode: {system.config.paper_mode}")
    log.info(f"Strategies: {[s['name'] for s in system.config.strategies]}")

    await system.start()
    log.info("All subsystems started.")

    # Start API server
    host = system.config.api_host
    port = system.config.api_port
    try:
        import uvicorn
        app = system.get_app()
        server_config = uvicorn.Config(
            app, host=host, port=port,
            log_level=log_level.lower(),
            loop="none",
        )
        server = uvicorn.Server(server_config)
        log.info(f"API server starting on http://{host}:{port}")
        await server.serve()
    except ImportError:
        log.warning("uvicorn not installed — API server disabled. pip install uvicorn")
        # Keep bot running without API
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass
    finally:
        log.info("Shutting down...")
        await system.stop()
        log.info("Argus stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
