from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RuntimeManifest:
    manifest_version: int
    profile: str
    constitution_hash: str
    runtime_hash: str
    exchange_hash: str
    strategy_hash: str
    git_commit: str
    generated_at_utc: str
    node_role: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeManifestWithHash:
    manifest: RuntimeManifest
    manifest_hash: str


def build_manifest(
    *,
    profile: str,
    constitution_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    exchange_cfg: dict[str, Any],
    strategy_cfg: dict[str, Any],
    git_commit: str,
    node_role: str,
) -> RuntimeManifestWithHash:
    manifest = RuntimeManifest(
        manifest_version=1,
        profile=profile,
        constitution_hash=stable_hash(constitution_cfg),
        runtime_hash=stable_hash(runtime_cfg),
        exchange_hash=stable_hash(exchange_cfg),
        strategy_hash=stable_hash(strategy_cfg),
        git_commit=git_commit,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        node_role=node_role,
    )
    return RuntimeManifestWithHash(manifest=manifest, manifest_hash=stable_hash(manifest.to_dict()))


def write_manifest(manifest: RuntimeManifestWithHash, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.manifest.to_dict() | {"manifest_hash": manifest.manifest_hash}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
