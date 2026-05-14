from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def _normalize_config(config: Any) -> Any:
    if isinstance(config, dict):
        return {str(k): _normalize_config(v) for k, v in sorted(config.items(), key=lambda item: str(item[0]))}
    if isinstance(config, (list, tuple)):
        return [_normalize_config(v) for v in config]
    return config


def canonical_config_text(config: dict[str, Any]) -> str:
    normalized = _normalize_config(config)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ConfigLock:
    config_hash: str
    config: dict[str, Any]


def lock_config(config: dict[str, Any]) -> ConfigLock:
    text = canonical_config_text(config)
    digest = 'sha256:' + sha256(text.encode('utf-8')).hexdigest()
    return ConfigLock(config_hash=digest, config=json.loads(text))


def build_config_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    previous = previous or {}
    changed: dict[str, dict[str, Any]] = {}
    prev_keys = set(previous)
    curr_keys = set(current)
    for key in sorted(prev_keys | curr_keys):
        old = previous.get(key)
        new = current.get(key)
        if old != new:
            changed[str(key)] = {'old': old, 'new': new}
    return {'changed': changed, 'previous_key_count': len(previous), 'current_key_count': len(current)}


def write_config_artifacts(*, artifact_dir: str | Path, run_id: str, config_lock: ConfigLock, previous_config: dict[str, Any] | None = None) -> tuple[Path, Path]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_path = artifact_dir / f'{run_id}.json'
    diff_path = artifact_dir / f'{run_id}.diff.json'
    config_path.write_text(json.dumps({'run_id': run_id, 'config_hash': config_lock.config_hash, 'config': config_lock.config}, indent=2, sort_keys=True), encoding='utf-8')
    diff_path.write_text(json.dumps({'run_id': run_id, 'config_hash': config_lock.config_hash, 'diff': build_config_diff(previous_config, config_lock.config)}, indent=2, sort_keys=True), encoding='utf-8')
    return config_path, diff_path
