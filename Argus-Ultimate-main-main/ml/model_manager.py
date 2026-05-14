#!/usr/bin/env python3
"""
ML Model Manager — lifecycle management for all ML/AI models used in Argus.

Responsibilities:
- Register, load, save, and hot-swap trained models
- SQLite-backed model registry with versioning
- Triggered retraining hooks (best-effort, non-blocking)
- Thread-safe model retrieval for the trading loop
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_registry (
    model_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    artifact_path TEXT,
    metrics_json  TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_mr_name ON model_registry(name);
"""


@dataclass
class ModelRecord:
    model_id: str
    name: str
    version: int = 1
    artifact_path: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active: bool = True


class ModelManager:
    """Thread-safe ML model lifecycle manager."""

    def __init__(self, config: Any = None, db_path: str = "data/model_registry.db", artifacts_dir: str = "data/models"):
        self.config = config
        self.db_path = str(getattr(config, "model_registry_db_path", None) or db_path)
        self.artifacts_dir = Path(str(getattr(config, "model_artifacts_dir", None) or artifacts_dir))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.db_path) or ".").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, Any] = {}  # name -> live model object
        self._init_schema()

    # ------------------------------------------------------------------ schema

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        try:
            con = self._connect()
            con.executescript(_SCHEMA)
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("ModelManager schema init: %s", e)

    # ------------------------------------------------------------------ CRUD

    def register(self, name: str, model: Any, *, metrics: Optional[Dict[str, Any]] = None, version: int = 1) -> ModelRecord:
        """Register (or update) a model in the registry and cache it."""
        import hashlib, uuid
        model_id = hashlib.sha1(f"{name}_{version}_{time.time()}".encode()).hexdigest()[:16]
        artifact_path = ""
        try:
            artifact_path = str(self.artifacts_dir / f"{name}_v{version}_{model_id}.pkl")
            with open(artifact_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning("ModelManager: could not persist artifact for %s: %s", name, e)
            artifact_path = ""
        now = time.time()
        rec = ModelRecord(
            model_id=model_id,
            name=name,
            version=version,
            artifact_path=artifact_path,
            metrics=metrics or {},
            created_at=now,
            updated_at=now,
        )
        try:
            con = self._connect()
            con.execute(
                "INSERT OR REPLACE INTO model_registry "
                "(model_id,name,version,artifact_path,metrics_json,created_at,updated_at,active) "
                "VALUES (?,?,?,?,?,?,?,1)",
                (rec.model_id, rec.name, rec.version, rec.artifact_path,
                 json.dumps(rec.metrics, default=str), rec.created_at, rec.updated_at),
            )
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("ModelManager register DB: %s", e)
        with self._lock:
            self._cache[name] = model
        logger.info("ModelManager: registered %s v%d (id=%s)", name, version, model_id)
        return rec

    def get(self, name: str) -> Optional[Any]:
        """Return live cached model or load from artifact."""
        with self._lock:
            if name in self._cache:
                return self._cache[name]
        return self._load_from_db(name)

    def _load_from_db(self, name: str) -> Optional[Any]:
        try:
            con = self._connect()
            row = con.execute(
                "SELECT artifact_path FROM model_registry WHERE name=? AND active=1 ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
            con.close()
            if row and row["artifact_path"] and Path(row["artifact_path"]).exists():
                with open(row["artifact_path"], "rb") as f:
                    model = pickle.load(f)
                with self._lock:
                    self._cache[name] = model
                logger.info("ModelManager: loaded %s from artifact", name)
                return model
        except Exception as e:
            logger.warning("ModelManager load %s: %s", name, e)
        return None

    def load_all(self) -> int:
        """Eagerly load all active models into cache. Returns count loaded."""
        loaded = 0
        try:
            con = self._connect()
            rows = con.execute(
                "SELECT DISTINCT name FROM model_registry WHERE active=1"
            ).fetchall()
            con.close()
            for row in rows:
                if self._load_from_db(row["name"]) is not None:
                    loaded += 1
        except Exception as e:
            logger.warning("ModelManager load_all: %s", e)
        logger.info("ModelManager: loaded %d models from registry", loaded)
        return loaded

    def deactivate(self, name: str) -> None:
        """Mark all versions of a model inactive."""
        try:
            con = self._connect()
            con.execute("UPDATE model_registry SET active=0 WHERE name=?", (name,))
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("ModelManager deactivate %s: %s", name, e)
        with self._lock:
            self._cache.pop(name, None)

    def list_models(self) -> List[Dict[str, Any]]:
        """Return all active model records."""
        try:
            con = self._connect()
            rows = con.execute(
                "SELECT model_id,name,version,artifact_path,metrics_json,created_at,updated_at "
                "FROM model_registry WHERE active=1 ORDER BY name,version DESC"
            ).fetchall()
            con.close()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["metrics"] = json.loads(d.pop("metrics_json", "{}") or "{}")
                except Exception:
                    d["metrics"] = {}
                out.append(d)
            return out
        except Exception as e:
            logger.warning("ModelManager list_models: %s", e)
            return []

    # ----------------------------------------------------------------- retrain hook

    def trigger_retrain(self, name: str, train_fn: Any, *args: Any, **kwargs: Any) -> None:
        """Fire-and-forget retrain in a daemon thread."""
        def _run() -> None:
            try:
                model = train_fn(*args, **kwargs)
                if model is not None:
                    self.register(name, model)
                    logger.info("ModelManager: retrain of %s complete", name)
            except Exception as e:
                logger.warning("ModelManager retrain %s: %s", name, e)
        t = threading.Thread(target=_run, daemon=True, name=f"retrain_{name}")
        t.start()
