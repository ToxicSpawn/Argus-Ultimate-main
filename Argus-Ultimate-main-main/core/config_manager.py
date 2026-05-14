#!/usr/bin/env python3
"""
Unified configuration helpers.

Goal: make `unified_config.yaml` the runtime source-of-truth for the main unified
trading path, while remaining backward compatible with legacy configs.

Migration status
----------------
* ConfigManager.load_split_config() still returns LegacyResolvedConfig so all
  existing call-sites (full_wiring.py, node_orchestrator.py, etc.) are unchanged.
* NEW: LegacyResolvedConfig.to_argus_config() returns the fully-validated
  Pydantic v2 ArgusConfig.  New modules should consume ArgusConfig directly.
* NEW: load_argus_config(path) is a one-liner for code that bypasses the
  legacy split-file loading and wants ArgusConfig straight from YAML.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml  # type: ignore[import-untyped]

from argus_live.control_plane.profile_resolver import load_profile_bundle
from argus_live.control_plane.runtime_manifest import build_manifest, write_manifest
from core.argus_config import ArgusConfig

logger = logging.getLogger(__name__)

_PROFILE_ENV_VAR = "ARGUS_CONFIG_PROFILE"


# ---------------------------------------------------------------------------
# Legacy dataclass (kept for backward compat)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LegacyResolvedConfig:
    profile: str
    constitution: dict[str, Any]
    runtime: dict[str, Any]
    exchange: dict[str, Any]
    strategy: dict[str, Any]
    manifest_hash: str

    # ------------------------------------------------------------------
    # Migration bridge
    # ------------------------------------------------------------------
    def to_argus_config(self) -> ArgusConfig:
        """
        Convert this legacy bundle to a fully-validated ArgusConfig.

        Call-site migration path:

            # Old (still works):
            resolved = mgr.load_split_config(...)
            some_val = resolved.runtime["runtime"]["tick_interval_ms"]

            # New (preferred for any code written going forward):
            cfg = resolved.to_argus_config()
            some_val = cfg.runtime.tick_interval_ms  # typed, IDE-complete
        """
        return ArgusConfig.from_legacy(self)


# ---------------------------------------------------------------------------
# ConfigManager (legacy facade — interface unchanged)
# ---------------------------------------------------------------------------

class ConfigManager:
    """Legacy compatibility facade for split-config loading."""

    def __init__(self) -> None:
        self._resolved: LegacyResolvedConfig | None = None

    def load_split_config(
        self,
        *,
        constitution_path: str,
        runtime_path: str,
        exchange_path: str,
        strategy_path: str,
        git_commit: str = "unknown",
    ) -> LegacyResolvedConfig:
        bundle = load_profile_bundle(
            constitution_path=constitution_path,
            runtime_path=runtime_path,
            exchange_path=exchange_path,
            strategy_path=strategy_path,
        )
        profile = bundle["constitution"]["constitution"]["profile"]
        runtime_cfg = bundle["runtime"]["runtime"]
        node_role = runtime_cfg["node_role"]
        manifest = build_manifest(
            profile=profile,
            constitution_cfg=bundle["constitution"],
            runtime_cfg=bundle["runtime"],
            exchange_cfg=bundle["exchange"],
            strategy_cfg=bundle["strategy"],
            git_commit=git_commit,
            node_role=node_role,
        )
        manifest_emit_path = runtime_cfg.get("manifest_emit_path")
        if manifest_emit_path:
            write_manifest(manifest, manifest_emit_path)

        resolved = LegacyResolvedConfig(
            profile=profile,
            constitution=bundle["constitution"],
            runtime=bundle["runtime"],
            exchange=bundle["exchange"],
            strategy=bundle["strategy"],
            manifest_hash=manifest.manifest_hash,
        )
        self._resolved = resolved

        # Eagerly validate via ArgusConfig so any config errors blow up at
        # load time instead of deep in the hot path.
        try:
            resolved.to_argus_config()
            logger.info("ArgusConfig validation passed (profile=%s)", profile)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ArgusConfig validation FAILED for profile=%s: %s", profile, exc
            )
            raise

        return resolved

    def get_resolved(self) -> LegacyResolvedConfig:
        if self._resolved is None:
            raise RuntimeError("No config has been resolved yet")
        return self._resolved

    def get_argus_config(self) -> ArgusConfig:
        """Convenience: get the validated ArgusConfig from the last load."""
        return self.get_resolved().to_argus_config()

    def load_unified_config(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        # DEPRECATED: use load_argus_config() or load_unified_trading_config() instead.
        # The split config (constitution/runtime/exchange/strategy) is a future goal.
        import warnings
        warnings.warn(
            "load_unified_config() is deprecated. Use load_argus_config() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from core.config_manager import load_argus_config
        return load_argus_config(*_args, **_kwargs)


# ---------------------------------------------------------------------------
# Path helpers (unchanged public API)
# ---------------------------------------------------------------------------

def resolve_unified_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    if path is None:
        return (Path(__file__).resolve().parent.parent / "unified_config.yaml").resolve()
    return Path(path).expanduser().resolve()


def _deep_merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = _deep_merge_dicts(current, value)
        else:
            out[key] = value
    return out


def resolve_unified_profile_path(
    profile: Optional[Union[str, Path]] = None,
    *,
    config_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    profile_raw = str(profile or os.getenv(_PROFILE_ENV_VAR, "")).strip()
    if not profile_raw:
        return None
    cfg_path = resolve_unified_config_path(config_path)
    profile_file = cfg_path.parent / "profiles" / f"{profile_raw}.yaml"
    return profile_file if profile_file.exists() else None


# ---------------------------------------------------------------------------
# One-liner direct loader (new code preferred entry point)
# ---------------------------------------------------------------------------

def load_argus_config(yaml_path: Optional[Union[str, Path]] = None) -> ArgusConfig:
    """
    Load ArgusConfig from unified_config.yaml.

    This is the canonical config loading path. The unified_config.yaml file
    is the single source of truth for all configuration.

    Future: split config (constitution/runtime/exchange/strategy) is planned
    but not yet implemented. See load_split_config() for the draft interface.

    Usage in new modules::

        from core.config_manager import load_argus_config
        cfg = load_argus_config()          # uses default unified_config.yaml
        cfg = load_argus_config("path/to/override.yaml")

        # Then access typed fields:
        if cfg.network.dpdk.enabled:
            init_dpdk(cfg.network.dpdk.eal_args)
    """
    path = resolve_unified_config_path(yaml_path)
    logger.info("Loading ArgusConfig from %s", path)
    return ArgusConfig.from_yaml(str(path))


def load_unified_trading_config(yaml_path: Optional[Union[str, Path]] = None, profile: Optional[str] = None) -> ArgusConfig:
    """Load unified trading config. Alias for load_argus_config for backward compatibility."""
    return load_argus_config(yaml_path)


def load_unified_yaml(yaml_path: Optional[Union[str, Path]] = None, *, profile: Optional[str] = None) -> Dict[str, Any]:
    """Load unified config as a plain dict. Used by evolution and legacy code.
    
    Args:
        yaml_path: Path to YAML config file. Defaults to unified_config.yaml.
        profile: Optional profile name (currently unused, kept for API compat).
    
    Returns:
        Dict representation of the YAML config.
    """
    path = resolve_unified_config_path(yaml_path)
    logger.debug("Loading unified YAML from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def validate_unified_config_dict(config: Dict[str, Any]) -> bool:
    """Validate a unified config dict. Returns True if valid, raises on error."""
    # Basic validation - check required top-level keys exist
    required_keys = ["exchange", "risk"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    return True
