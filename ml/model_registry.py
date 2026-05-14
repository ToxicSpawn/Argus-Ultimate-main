"""Model Versioning and Registry.

Track, version, and manage trained ML models.
"""

from __future__ import annotations

import logging
import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    version: str
    model_type: str
    created_at: float
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    file_path: Optional[str] = None
    parent_version: Optional[str] = None
    status: str = "training"


@dataclass
class ModelRegistry:
    name: str
    versions: List[ModelVersion] = field(default_factory=list)
    current_version: Optional[str] = None


class ModelVersionManager:
    """Manages model versions and registry."""

    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        self.registries: Dict[str, ModelRegistry] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        registry_file = self.base_path / "registry.json"

        if registry_file.exists():
            with open(registry_file) as f:
                data = json.load(f)
                for name, reg_data in data.items():
                    registry = ModelRegistry(name=name)
                    for v in reg_data.get("versions", []):
                        registry.versions.append(ModelVersion(**v))
                    registry.current_version = reg_data.get("current_version")
                    self.registries[name] = registry

    def _save_registry(self) -> None:
        data = {}
        for name, registry in self.registries.items():
            data[name] = {
                "versions": [vars(v) for v in registry.versions],
                "current_version": registry.current_version,
            }

        with open(self.base_path / "registry.json", "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_type: str,
        hyperparameters: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        file_path: Optional[str] = None,
    ) -> str:
        if model_name not in self.registries:
            self.registries[model_name] = ModelRegistry(name=model_name)

        registry = self.registries[model_name]

        timestamp = int(time.time())
        version = f"v{timestamp}"

        new_version = ModelVersion(
            version=version,
            model_type=model_type,
            created_at=time.time(),
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            file_path=file_path,
            parent_version=registry.current_version,
            status="trained",
        )

        registry.versions.append(new_version)
        registry.current_version = version

        self._save_registry()

        logger.info(f"Registered model {model_name}: {version}")
        return version

    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        if model_name not in self.registries:
            return None

        for v in self.registries[model_name].versions:
            if v.version == version:
                return v

        return None

    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        if model_name not in self.registries:
            return None

        registry = self.registries[model_name]
        if not registry.versions:
            return None

        return registry.versions[-1]

    def get_all_versions(self, model_name: str) -> List[ModelVersion]:
        if model_name not in self.registries:
            return []

        return self.registries[model_name].versions

    def get_best_version(
        self,
        model_name: str,
        metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> Optional[ModelVersion]:
        versions = self.get_all_versions(model_name)

        if not versions:
            return None

        sorted_versions = sorted(
            versions,
            key=lambda v: v.metrics.get(metric, 0),
            reverse=higher_is_better,
        )

        return sorted_versions[0]

    def get_leaderboard(
        self,
        model_name: str,
        metric: str = "accuracy",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        versions = self.get_all_versions(model_name)

        if not versions:
            return []

        sorted_versions = sorted(
            versions,
            key=lambda v: v.metrics.get(metric, 0),
            reverse=True,
        )[:top_k]

        return [
            {
                "version": v.version,
                "model_type": v.model_type,
                "metrics": v.metrics,
                "created_at": v.created_at,
                "status": v.status,
            }
            for v in sorted_versions
        ]

    def compare_versions(
        self,
        model_name: str,
        versions: List[str],
    ) -> Dict[str, Any]:
        comparison = {}

        for version in versions:
            v = self.get_version(model_name, version)
            if v:
                comparison[version] = {
                    "metrics": v.metrics,
                    "created_at": v.created_at,
                    "parent": v.parent_version,
                }

        return comparison

    def rollback(self, model_name: str, version: str) -> bool:
        if model_name not in self.registries:
            return False

        v = self.get_version(model_name, version)
        if not v:
            return False

        self.registries[model_name].current_version = version
        self._save_registry()

        return True

    def export_model_card(
        self,
        model_name: str,
        version: str,
        output_path: str,
    ) -> None:
        v = self.get_version(model_name, version)
        if not v:
            return

        model_card = {
            "model_name": model_name,
            "version": v.version,
            "model_type": v.model_type,
            "hyperparameters": v.hyperparameters,
            "metrics": v.metrics,
            "created_at": v.created_at,
            "parent_version": v.parent_version,
            "status": v.status,
        }

        with open(output_path, "w") as f:
            json.dump(model_card, f, indent=2)

        logger.info(f"Model card exported to {output_path}")


def create_version_manager(base_path: str = "models") -> ModelVersionManager:
    return ModelVersionManager(base_path)