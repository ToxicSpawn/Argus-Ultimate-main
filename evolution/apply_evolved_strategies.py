"""
Apply evolved (or optimized) strategy parameters to config and persist.

Used by the evolution pipeline to write best params to disk and by the
unified system to load them at startup or in the paper loop.
Supports backup before write, version history, and rich metadata.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EVOLVED_PARAMS_PATH = "data/evolved_params.json"
DEFAULT_VERSION_HISTORY_SIZE = 5
DEFAULT_BACKUP_DIR = "data/evolved_backups"


def apply_to_config(config: Any, params: Dict[str, Any]) -> int:
    """
    Set config attributes from params. Keys must be valid config attribute names.
    Returns number of attributes set.
    """
    count = 0
    for k, v in params.items():
        try:
            if hasattr(config, k):
                setattr(config, k, v)
                count += 1
        except Exception as e:
            logger.debug("Could not set config.%s: %s", k, e)
    return count


def apply_from_file(
    config: Any,
    path: str | Path = DEFAULT_EVOLVED_PARAMS_PATH,
    key: str = "best_params",
) -> int:
    """
    Load JSON from path; if key is given use params = data[key], else data is params.
    Apply to config. Returns number of attributes set.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("Evolved params file not found: %s", p)
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return 0
        # Prefer requested key; fallback to common keys so multiple writers work
        for candidate in (key, "best_params", "evolved_params", "params"):
            if candidate and data.get(candidate) is not None and isinstance(data[candidate], dict):
                params = data[candidate]
                break
        else:
            params = data
        if not isinstance(params, dict):
            return 0
        return apply_to_config(config, params)
    except Exception as e:
        logger.warning("Failed to load evolved params from %s: %s", p, e)
        return 0


def backup_if_exists(path: Path, backup_dir: Optional[Path] = None) -> None:
    """Copy current file to backup_dir/evolved_params_<iso>.json if it exists."""
    if not path.exists():
        return
    backup_root = backup_dir or Path(DEFAULT_BACKUP_DIR)
    backup_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = backup_root / f"evolved_params_{ts}.json"
    try:
        shutil.copy2(path, dest)
        logger.debug("Backed up evolved params to %s", dest)
    except Exception as e:
        logger.warning("Backup evolved params failed: %s", e)


def write_evolved_params(
    params: Dict[str, Any],
    path: str | Path = DEFAULT_EVOLVED_PARAMS_PATH,
    meta: Optional[Dict[str, Any]] = None,
    backup_before: bool = True,
    version_history_size: int = 0,
    version_history_dir: Optional[str] = None,
) -> None:
    """
    Write params to JSON. If meta is provided, file is { "best_params": params, **meta }.
    Always adds timestamp_utc for versioning/rollback.
    backup_before: copy current file to data/evolved_backups/ before overwriting.
    version_history_size: keep last N versions in version_history_dir (e.g. data/evolved_history/).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if backup_before:
        backup_if_exists(p)

    payload = dict(meta or {})
    payload["best_params"] = params
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote evolved params to %s", p)

    if version_history_size > 0:
        hist_dir = Path(version_history_dir or str(p.parent / "evolved_history"))
        hist_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        hist_file = hist_dir / f"evolved_{ts}.json"
        hist_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        existing = sorted(hist_dir.glob("evolved_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        for old in existing[version_history_size:]:
            try:
                old.unlink()
            except Exception as _e:
                logger.debug("apply_evolved_strategies error: %s", _e)


def load_last_best_params(path: str | Path = DEFAULT_EVOLVED_PARAMS_PATH) -> Optional[Dict[str, Any]]:
    """Load best_params from file for seeding GA. Returns None if file missing or invalid."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        for key in ("best_params", "evolved_params", "params"):
            if data.get(key) is not None and isinstance(data[key], dict):
                return data[key]
    except Exception:
        return None
    return None


def get_version_history(
    version_history_dir: str | Path,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return list of past payloads (newest first) for rollback."""
    d = Path(version_history_dir)
    if not d.exists():
        return []
    files = sorted(d.glob("evolved_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def decay_strategy_allocator_stats(
    persist_path: str | Path,
    decay_factor: float,
) -> bool:
    """
    Load strategy allocator persist file, multiply bucket stats by decay_factor (0=clear),
    and save. Returns True if file was updated. Use after evolution apply so allocator
    doesn't over-exploit old strategy behavior.
    """
    p = Path(persist_path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return False
        buckets = data.get("buckets")
        if not isinstance(buckets, dict):
            return False
        if decay_factor <= 0:
            data["buckets"] = {}
        else:
            new_buckets = {}
            for k, v in buckets.items():
                if not isinstance(v, dict):
                    continue
                new_buckets[k] = {
                    "trades": max(0, int((v.get("trades") or 0) * decay_factor)),
                    "wins": max(0, int((v.get("wins") or 0) * decay_factor)),
                    "pnl_ema": float((v.get("pnl_ema") or 0) * decay_factor),
                    "pnl2_ema": float((v.get("pnl2_ema") or 0) * decay_factor),
                }
            data["buckets"] = new_buckets
        p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Decayed strategy allocator stats at %s by factor %.2f", p, decay_factor)
        return True
    except Exception as e:
        logger.warning("Decay allocator stats failed: %s", e)
        return False


def rollback_to_previous(
    path: str | Path = DEFAULT_EVOLVED_PARAMS_PATH,
    version_history_dir: Optional[str] = None,
    index: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Load the index-th previous version from history (0 = most recent in history,
    i.e. the one before current file). Write it to path and return the payload.
    Returns None if no history or index out of range.
    """
    p = Path(path)
    hist_dir = Path(version_history_dir or str(p.parent / "evolved_history"))
    history = get_version_history(hist_dir, limit=index + 5)
    if index >= len(history):
        return None
    payload = history[index]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Rolled back evolved params to version index=%s from %s", index, hist_dir)
    return payload
