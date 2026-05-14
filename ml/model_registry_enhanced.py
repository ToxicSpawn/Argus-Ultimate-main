"""
Enhanced Model Registry with Metadata.

Extends basic model management with:
- Training metadata (hyperparameters, dataset version, training time)
- Metric tracking (Sharpe, accuracy, loss per epoch)
- Model comparison and best model selection
- Lineage tracking (which data → which model)
- Automatic promotion based on performance

Usage:
    from ml.model_registry_enhanced import EnhancedModelRegistry
    
    registry = EnhancedModelRegistry()
    
    # Register with full metadata
    registry.register(
        name="regime_classifier",
        model=model,
        metrics={"sharpe": 1.2, "accuracy": 0.85, "val_loss": 0.3},
        hyperparams={"n_estimators": 100, "max_depth": 10},
        dataset_version="v20260426",
        tags=["production", "xgboost"],
    )
    
    # Get best model by metric
    best = registry.get_best("regime_classifier", metric="sharpe")
    
    # Compare all versions
    comparison = registry.compare_versions("regime_classifier")
"""

import json
import logging
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Enhanced schema with metadata
_ENHANCED_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_registry_enhanced (
    model_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1,
    artifact_path   TEXT,
    
    -- Training metadata
    hyperparams_json    TEXT,
    dataset_version     TEXT,
    dataset_hash        TEXT,
    train_samples       INTEGER,
    val_samples         INTEGER,
    n_features          INTEGER,
    
    -- Metrics
    train_metrics_json  TEXT,
    val_metrics_json    TEXT,
    test_metrics_json   TEXT,
    
    -- Training info
    training_time_seconds   REAL,
    early_stopped           INTEGER DEFAULT 0,
    best_epoch              INTEGER,
    total_epochs            INTEGER,
    
    -- Status
    status          TEXT DEFAULT 'staging',
    tags_json       TEXT,
    notes           TEXT,
    parent_version  INTEGER,
    
    -- Timestamps
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    
    active          INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_mre_name ON model_registry_enhanced(name);
CREATE INDEX IF NOT EXISTS idx_mre_status ON model_registry_enhanced(status);
CREATE INDEX IF NOT EXISTS idx_mre_name_version ON model_registry_enhanced(name, version);
"""


@dataclass
class ModelMetadata:
    """Complete metadata for a registered model."""
    model_id: str
    name: str
    version: int
    
    # Training metadata
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    dataset_version: str = ""
    dataset_hash: str = ""
    train_samples: int = 0
    val_samples: int = 0
    n_features: int = 0
    
    # Metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training info
    training_time_seconds: float = 0.0
    early_stopped: bool = False
    best_epoch: int = 0
    total_epochs: int = 0
    
    # Status
    status: str = "staging"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    parent_version: Optional[int] = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Artifact
    artifact_path: str = ""


class EnhancedModelRegistry:
    """
    Enhanced Model Registry with full metadata support.
    
    Features:
    - Register models with training metadata
    - Track metrics across training epochs
    - Compare model versions
    - Automatic best model selection
    - Model promotion (staging → production)
    - Dataset versioning and lineage
    
    Args:
        db_path: Path to SQLite database
        artifacts_dir: Directory for model artifacts
    
    Example:
        >>> registry = EnhancedModelRegistry()
        >>> registry.register(
        ...     name="regime_classifier",
        ...     model=xgb_model,
        ...     metrics={"sharpe": 1.2, "accuracy": 0.85},
        ...     hyperparams={"n_estimators": 100},
        ...     dataset_version="2026-04-26_btc_1h",
        ... )
        >>> best = registry.get_best("regime_classifier", metric="sharpe")
    """
    
    def __init__(
        self,
        db_path: str = "data/model_registry_enhanced.db",
        artifacts_dir: str = "data/models",
    ):
        self.db_path = db_path
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}
        
        self._init_schema()
        logger.info(f"EnhancedModelRegistry initialized: db={db_path}")
    
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.row_factory = sqlite3.Row
        return con
    
    def _init_schema(self) -> None:
        try:
            con = self._connect()
            con.executescript(_ENHANCED_SCHEMA)
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("EnhancedModelRegistry schema init: %s", e)
    
    def register(
        self,
        name: str,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        dataset_version: str = "",
        dataset_hash: str = "",
        train_samples: int = 0,
        val_samples: int = 0,
        n_features: int = 0,
        training_time_seconds: float = 0.0,
        early_stopped: bool = False,
        best_epoch: int = 0,
        total_epochs: int = 0,
        tags: Optional[List[str]] = None,
        notes: str = "",
        parent_version: Optional[int] = None,
        status: str = "staging",
    ) -> ModelMetadata:
        """
        Register a model with full metadata.
        """
        import hashlib
        
        version = self._get_next_version(name)
        model_id = hashlib.sha1(f"{name}_{version}_{time.time()}".encode()).hexdigest()[:16]
        
        artifact_path = ""
        try:
            artifact_path = str(self.artifacts_dir / f"{name}_v{version}_{model_id}.pkl")
            with open(artifact_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning("Could not save artifact for %s: %s", name, e)
        
        now = time.time()
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            hyperparams=hyperparams or {},
            dataset_version=dataset_version,
            dataset_hash=dataset_hash,
            train_samples=train_samples,
            val_samples=val_samples,
            n_features=n_features,
            train_metrics={},
            val_metrics=metrics or {},
            test_metrics={},
            training_time_seconds=training_time_seconds,
            early_stopped=early_stopped,
            best_epoch=best_epoch,
            total_epochs=total_epochs,
            status=status,
            tags=tags or [],
            notes=notes,
            parent_version=parent_version,
            created_at=now,
            updated_at=now,
            artifact_path=artifact_path,
        )
        
        self._save_metadata(metadata)
        
        with self._lock:
            self._cache[name] = model
        
        logger.info(f"Registered {name} v{version} (id={model_id}, metrics={metrics})")
        return metadata
    
    def _save_metadata(self, metadata: ModelMetadata) -> None:
        """Save metadata to database."""
        try:
            con = self._connect()
            con.execute("""
                INSERT OR REPLACE INTO model_registry_enhanced
                (model_id, name, version, artifact_path,
                 hyperparams_json, dataset_version, dataset_hash,
                 train_samples, val_samples, n_features,
                 train_metrics_json, val_metrics_json, test_metrics_json,
                 training_time_seconds, early_stopped, best_epoch, total_epochs,
                 status, tags_json, notes, parent_version,
                 created_at, updated_at, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                metadata.model_id, metadata.name, metadata.version, metadata.artifact_path,
                json.dumps(metadata.hyperparams), metadata.dataset_version, metadata.dataset_hash,
                metadata.train_samples, metadata.val_samples, metadata.n_features,
                json.dumps(metadata.train_metrics), json.dumps(metadata.val_metrics), json.dumps(metadata.test_metrics),
                metadata.training_time_seconds, int(metadata.early_stopped), metadata.best_epoch, metadata.total_epochs,
                metadata.status, json.dumps(metadata.tags), metadata.notes, metadata.parent_version,
                metadata.created_at, metadata.updated_at,
            ))
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("Failed to save metadata: %s", e)
    
    def get_best(
        self,
        name: str,
        metric: str = "val_sharpe",
        status: Optional[str] = None,
    ) -> Optional[ModelMetadata]:
        """Get best model version by metric."""
        try:
            con = self._connect()
            
            status_filter = "AND status=?" if status else ""
            params: list = [name]
            if status:
                params.append(status)
            
            rows = con.execute(f"""
                SELECT * FROM model_registry_enhanced 
                WHERE name=? {status_filter}
                ORDER BY version DESC
            """, params).fetchall()
            con.close()
            
            if not rows:
                return None
            
            best_row = None
            best_value = float('-inf')
            
            for row in rows:
                metrics = json.loads(row["val_metrics_json"] or "{}")
                value = metrics.get(metric, float('-inf'))
                if value > best_value:
                    best_value = value
                    best_row = row
            
            if best_row:
                return self._row_to_metadata(best_row)
            
            return None
            
        except Exception as e:
            logger.warning("Failed to get best model: %s", e)
            return None
    
    def compare_versions(self, name: str) -> List[Dict[str, Any]]:
        """Compare all versions of a model."""
        try:
            con = self._connect()
            rows = con.execute("""
                SELECT * FROM model_registry_enhanced 
                WHERE name=? AND active=1
                ORDER BY version DESC
            """, (name,)).fetchall()
            con.close()
            
            return [
                {
                    "version": r["version"],
                    "status": r["status"],
                    "created": datetime.fromtimestamp(r["created_at"]).isoformat(),
                    "training_time": r["training_time_seconds"],
                    "dataset_version": r["dataset_version"],
                    "val_metrics": json.loads(r["val_metrics_json"] or "{}"),
                    "hyperparams": json.loads(r["hyperparams_json"] or "{}"),
                }
                for r in rows
            ]
            
        except Exception as e:
            logger.warning("Failed to compare versions: %s", e)
            return []
    
    def promote(self, name: str, version: int) -> bool:
        """Promote model version to production."""
        try:
            con = self._connect()
            con.execute(
                "UPDATE model_registry_enhanced SET status='archived' WHERE name=? AND status='production'",
                (name,)
            )
            con.execute(
                "UPDATE model_registry_enhanced SET status='production', updated_at=? WHERE name=? AND version=?",
                (time.time(), name, version)
            )
            con.commit()
            con.close()
            logger.info(f"Promoted {name} v{version} to production")
            return True
        except Exception as e:
            logger.warning("Failed to promote model: %s", e)
            return False
    
    def get_production(self, name: str) -> Optional[Any]:
        """Get production model (loads if not cached)."""
        with self._lock:
            if name in self._cache:
                return self._cache[name]
        
        try:
            con = self._connect()
            row = con.execute("""
                SELECT artifact_path FROM model_registry_enhanced 
                WHERE name=? AND status='production' 
                ORDER BY version DESC LIMIT 1
            """, (name,)).fetchone()
            con.close()
            
            if row and row["artifact_path"] and Path(row["artifact_path"]).exists():
                with open(row["artifact_path"], "rb") as f:
                    model = pickle.load(f)
                with self._lock:
                    self._cache[name] = model
                return model
            
        except Exception as e:
            logger.warning("Failed to load production model %s: %s", name, e)
        
        return None
    
    def _get_next_version(self, name: str) -> int:
        """Get next version number for model."""
        try:
            con = self._connect()
            row = con.execute(
                "SELECT MAX(version) as max_v FROM model_registry_enhanced WHERE name=?",
                (name,)
            ).fetchone()
            con.close()
            return (row["max_v"] or 0) + 1
        except Exception:
            return 1
    
    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            model_id=row["model_id"],
            name=row["name"],
            version=row["version"],
            hyperparams=json.loads(row["hyperparams_json"] or "{}"),
            dataset_version=row["dataset_version"] or "",
            dataset_hash=row["dataset_hash"] or "",
            train_samples=row["train_samples"] or 0,
            val_samples=row["val_samples"] or 0,
            n_features=row["n_features"] or 0,
            train_metrics=json.loads(row["train_metrics_json"] or "{}"),
            val_metrics=json.loads(row["val_metrics_json"] or "{}"),
            test_metrics=json.loads(row["test_metrics_json"] or "{}"),
            training_time_seconds=row["training_time_seconds"] or 0.0,
            early_stopped=bool(row["early_stopped"]),
            best_epoch=row["best_epoch"] or 0,
            total_epochs=row["total_epochs"] or 0,
            status=row["status"] or "staging",
            tags=json.loads(row["tags_json"] or "[]"),
            notes=row["notes"] or "",
            parent_version=row["parent_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            artifact_path=row["artifact_path"] or "",
        )
    
    def list_models(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all models, optionally filtered by status."""
        try:
            con = self._connect()
            
            if status:
                rows = con.execute(
                    "SELECT name, version, status, val_metrics_json, created_at FROM model_registry_enhanced WHERE status=? ORDER BY name, version DESC",
                    (status,)
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT name, version, status, val_metrics_json, created_at FROM model_registry_enhanced WHERE active=1 ORDER BY name, version DESC"
                ).fetchall()
            
            con.close()
            
            return [
                {
                    "name": r["name"],
                    "version": r["version"],
                    "status": r["status"],
                    "metrics": json.loads(r["val_metrics_json"] or "{}"),
                    "created": datetime.fromtimestamp(r["created_at"]).isoformat(),
                }
                for r in rows
            ]
            
        except Exception as e:
            logger.warning("Failed to list models: %s", e)
            return []
