"""scripts/validate_config.py

Validates Argus config files before deployment.
Checks for required fields, type correctness, and dangerous defaults.

Usage:
    python scripts/validate_config.py [--config PATH]

Exits 0 if valid, 1 if invalid.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("PyYAML not installed — run: pip install pyyaml")
    sys.exit(1)


REQUIRED_TOP_LEVEL = [
    "mode",
    "exchange",
    "risk",
    "strategies",
]

REQUIRED_RISK_FIELDS = [
    "max_daily_loss_pct",
    "max_position_pct",
    "kill_switch_drawdown_pct",
]

DANGEROUS_DEFAULTS = {
    "risk.max_daily_loss_pct": (5.0, "Consider tightening daily loss limit for live trading"),
    "risk.max_position_pct": (10.0, "Consider reducing max position size"),
}


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_get(obj: dict, key_path: str) -> Any:
    parts = key_path.split(".")
    for part in parts:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def validate(config: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    # Support both newer config schema and the legacy/default_config.yml schema.
    effective_mode = config.get("mode")
    if effective_mode is None and "paper_mode" in config:
        effective_mode = "paper" if bool(config.get("paper_mode")) else "live"

    effective_exchange = config.get("exchange")
    if effective_exchange is None and ("binance_api_key" in config or "binance_api_secret" in config):
        effective_exchange = {"name": "binance", "api_key": config.get("binance_api_key"), "api_secret": config.get("binance_api_secret")}

    risk = config.get("risk", {})
    risk_aliases = {
        "max_daily_loss_pct": ["daily_loss_limit_pct", "max_daily_loss_pct"],
        "max_position_pct": ["max_position_pct", "max_portfolio_heat"],
        "kill_switch_drawdown_pct": ["kill_switch_drawdown_pct", "max_symbol_drawdown_pct", "max_drawdown_pct"],
    }

    # Required top-level keys
    for key in REQUIRED_TOP_LEVEL:
        if key == "mode" and effective_mode is not None:
            continue
        if key == "exchange" and effective_exchange is not None:
            continue
        if key not in config:
            errors.append(f"Missing required top-level key: '{key}'")

    # Required risk fields
    if isinstance(risk, dict):
        for field in REQUIRED_RISK_FIELDS:
            aliases = risk_aliases.get(field, [field])
            if not any(alias in risk for alias in aliases):
                errors.append(f"Missing required risk field: 'risk.{field}'")

    # Mode validation
    mode = effective_mode or ""
    if mode not in ("paper", "live", "backtest", "test"):
        errors.append(f"Invalid mode: '{mode}' - must be paper/live/backtest/test")

    # Exchange validation
    exchange = effective_exchange or {}
    if isinstance(exchange, dict):
        name = exchange.get("name", "")
        if name not in ("binance", "kraken", "coinbase", "paper", ""):
            warnings.append(f"Unknown exchange: '{name}'")

    # Dangerous defaults
    for key_path, (threshold, msg) in DANGEROUS_DEFAULTS.items():
        val = deep_get(config, key_path)
        if val is not None and float(val) >= threshold:
            warnings.append(f"WARNING {key_path}={val}: {msg}")

    # Live mode safety checks
    if mode == "live":
        if not os.environ.get("ARGUS_API_KEY") and not deep_get(config, "exchange.api_key"):
            errors.append("Live mode requires ARGUS_API_KEY env var or exchange.api_key in config")
        if not os.environ.get("ARGUS_API_SECRET") and not deep_get(config, "exchange.api_secret"):
            errors.append("Live mode requires ARGUS_API_SECRET env var or exchange.api_secret in config")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Argus config")
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to config file (default: config/config.yml)",
    )
    args = parser.parse_args()
    path = Path(args.config)

    if not path.exists():
        # Try default
        path = Path("config/default_config.yml")
        if not path.exists():
            print(f"Config file not found: {args.config}")
            return 1

    print(f"Validating: {path}")
    try:
        config = load_config(path)
    except Exception as e:
        print(f"Failed to parse config: {e}")
        return 1

    errors, warnings = validate(config, path)

    for w in warnings:
        print(f"  WARNING: {w}")
    for e in errors:
        print(f"  ERROR: {e}")

    if not errors and not warnings:
        print("  OK: Config is valid.")

    if errors:
        print(f"\n{len(errors)} error(s) found - config is invalid.")
        return 1

    print(f"\nConfig valid ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
