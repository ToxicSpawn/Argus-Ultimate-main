"""Package: core.startup

Single authoritative boot sequence for Argus.

Usage
-----
In main.py (or any entrypoint)::

    from core.startup import startup_config_check, get_config

    def main():
        cfg = startup_config_check(
            config_path=args.config,          # optional — auto-discovers if None
            capital_override=args.capital,    # optional float
            mode_override=args.mode,          # optional str  e.g. "dry_run" / "live"
        )

Anywhere in a subsystem (after boot)::

    from core.startup import get_config
    cfg = get_config()           # returns ArgusConfig stored in SharedState
    max_dd = cfg.risk.max_drawdown

Boot sequence
-------------
1. Resolve config path (arg → config.yaml → unified_config.yaml → config/config.yaml)
2. Load & validate via Pydantic ArgusConfig — raises ValidationError immediately on bad values
3. Apply CLI overrides with re-validation
4. Emit WARNING if mode == "live"
5. Log one-liner summary
6. Store ArgusConfig in SharedState so every subsystem can call get_config()

Batch 9 changes
---------------
- Handle mode_override when cfg is SimpleNamespace fallback (Pydantic schema not installed)
- Expose `_AUTO_DISCOVER_PATHS` for tests that need to inspect the search list
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config path discovery
# ---------------------------------------------------------------------------

_AUTO_DISCOVER_PATHS: list[str] = [
    "config.yaml",
    "unified_config.yaml",
    "config/config.yaml",
]


def _resolve_config_path(config_path: Optional[str | Path]) -> Path:
    """Return the first existing config file; raise FileNotFoundError if none found."""
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p

    for candidate in _AUTO_DISCOVER_PATHS:
        p = Path(candidate)
        if p.exists():
            logger.debug("Auto-discovered config: %s", p)
            return p

    raise FileNotFoundError(
        f"No config file found. Searched: {_AUTO_DISCOVER_PATHS}. "
        "Pass --config <path> or create config.yaml in the repo root."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def startup_config_check(
    config_path: Optional[str | Path] = None,
    capital_override: Optional[float] = None,
    mode_override: Optional[str] = None,
) -> object:  # returns ArgusConfig but typed as object to avoid hard import cycle
    """Load, validate, and register the Argus config.  Call once at startup."""
    import yaml  # late import — not needed at module load time
    from pydantic import ValidationError

    # 1. Resolve path
    resolved = _resolve_config_path(config_path)
    logger.info("Loading config from %s", resolved)

    # 2. Parse YAML
    with resolved.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    # 3. Apply CLI overrides before validation
    if capital_override is not None:
        raw.setdefault("system", {})
        raw["system"]["initial_capital"] = capital_override

    if mode_override is not None:
        raw.setdefault("system", {})
        raw["system"]["mode"] = mode_override

    # 4. Validate via Pydantic schema (raises ValidationError on bad values)
    cfg: object
    try:
        from core.config_schema import ArgusConfig
        cfg = ArgusConfig.model_validate(raw)
    except ImportError:
        # ArgusConfig not yet available — fall back to a lightweight namespace
        from types import SimpleNamespace
        system_raw = raw.get("system", {})
        risk_raw = raw.get("risk", {})
        exchanges_raw = raw.get("exchanges", {})
        cfg = SimpleNamespace(
            system=SimpleNamespace(
                mode=system_raw.get("mode", "dry_run"),
                initial_capital=float(system_raw.get("initial_capital", 1000.0)),
            ),
            risk=SimpleNamespace(
                max_drawdown=float(risk_raw.get("max_drawdown_pct", 15.0)),
            ),
            exchanges=SimpleNamespace(
                enabled_names=list(exchanges_raw.keys()) if isinstance(exchanges_raw, dict) else [],
            ),
            _raw=raw,
        )
        logger.warning(
            "core.config_schema not found — using SimpleNamespace fallback. "
            "Run batch 6 to install ArgusConfig."
        )
    except ValidationError as exc:  # noqa: F841
        logger.critical("Config validation failed:\n%s", exc)
        raise

    # 5. Safety gate for live mode
    mode = getattr(getattr(cfg, "system", None), "mode", "dry_run")
    if mode == "live":
        logger.warning(
            "LIVE MODE ENABLED — real capital at risk. "
            "Ensure API keys, risk limits, and exchange connections are verified."
        )

    # 6. Summary log
    capital = getattr(getattr(cfg, "system", None), "initial_capital", "?")
    max_dd = getattr(getattr(cfg, "risk", None), "max_drawdown", "?")
    exchanges_list = getattr(getattr(cfg, "exchanges", None), "enabled_names", [])
    logger.info(
        "Config validated: mode=%s capital=%s max_dd=%s%% exchanges=%s",
        mode, capital, max_dd, exchanges_list,
    )

    # 7. Store in SharedState
    try:
        from core.shared_state import SharedState
        SharedState.instance()["argus_config"] = cfg
        logger.debug("ArgusConfig stored in SharedState[argus_config]")
    except Exception:  # noqa: BLE001
        logger.warning("SharedState unavailable — config not cached globally.")

    return cfg


def get_config() -> Optional[object]:
    """Retrieve the validated ArgusConfig from SharedState.

    Returns None if startup_config_check() has not been called yet.
    """
    try:
        from core.shared_state import SharedState
        return SharedState.instance().get("argus_config")
    except Exception:  # noqa: BLE001
        logger.debug("get_config(): SharedState unavailable.")
        return None
