"""
Episodic memory — append-only event log with vector embeddings.

Stores individual events (trades, decisions, alerts) as tuples of
(timestamp, event_type, content, metadata, embedding). The embeddings
are produced by Commit 1's RAG embedding model so they can be retrieved
via cosine similarity later.

Writes to ``data/semantic_memory.db`` (NEW database — does NOT touch
``cross_session_memory.sqlite``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# EpisodicEntry
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class EpisodicEntry:
    """A single episodic memory."""

    id: int
    timestamp: float
    event_type: str  # e.g. "trade", "signal", "drawdown", "regime_change"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "content": self.content,
            "metadata": self.metadata,
        }


# ═════════════════════════════════════════════════════════════════════════════
# EpisodicMemory
# ═════════════════════════════════════════════════════════════════════════════


class EpisodicMemory:
    """
    Append-only episodic event log with vector embeddings.

    Parameters
    ----------
    db_path : str, default ``data/semantic_memory.db``
        Path to the SQLite file. Distinct from ``cross_session_memory.sqlite``.
    embedding_dim : int, default 128
    embedding_model : optional
        Instance of ``core/rag/embedding_model.EmbeddingModel``. If ``None``,
        we defer to the default factory.
    """

    def __init__(
        self,
        db_path: str = "data/semantic_memory.db",
        embedding_dim: int = 128,
        embedding_model: Optional[Any] = None,
    ) -> None:
        self.db_path = db_path
        self.embedding_dim = int(embedding_dim)
        self._lock = threading.Lock()

        if embedding_model is None:
            try:
                from core.rag import EmbeddingModel
                embedding_model = EmbeddingModel(dim=self.embedding_dim)
            except Exception as exc:
                logger.warning("EpisodicMemory: embedding model unavailable: %s", exc)
                embedding_model = None
        self._embedder = embedding_model

        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_ts ON episodic_entries (timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodic_type ON episodic_entries (event_type)"
            )
            conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def add(
        self,
        event_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Append a new episodic entry.

        Returns the auto-generated id.
        """
        metadata = dict(metadata or {})
        timestamp = float(metadata.get("timestamp", time.time()))

        # Compute embedding
        embedding: Optional[bytes] = None
        if self._embedder is not None:
            try:
                vec = self._embedder.encode(content)
                embedding = np.asarray(vec, dtype=np.float32).tobytes()
            except Exception as exc:
                logger.debug("EpisodicMemory: embedding failed: %s", exc)

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "INSERT INTO episodic_entries (timestamp, event_type, content, metadata, embedding) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (timestamp, event_type, content, json.dumps(metadata), embedding),
                )
                entry_id = int(cur.lastrowid or 0)
                conn.commit()
        return entry_id

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_recent(
        self,
        n: int = 100,
        event_type: Optional[str] = None,
    ) -> List[EpisodicEntry]:
        """Return the ``n`` most recent entries, optionally filtered by type."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if event_type:
                    rows = conn.execute(
                        "SELECT id, timestamp, event_type, content, metadata, embedding "
                        "FROM episodic_entries WHERE event_type = ? "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (event_type, n),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, timestamp, event_type, content, metadata, embedding "
                        "FROM episodic_entries ORDER BY timestamp DESC LIMIT ?",
                        (n,),
                    ).fetchall()

        entries: List[EpisodicEntry] = []
        for row in rows:
            entries.append(self._row_to_entry(row))
        return entries

    def get_range(
        self,
        start_ts: float,
        end_ts: float,
    ) -> List[EpisodicEntry]:
        """Return all entries in the time window."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, timestamp, event_type, content, metadata, embedding "
                    "FROM episodic_entries WHERE timestamp BETWEEN ? AND ? "
                    "ORDER BY timestamp",
                    (start_ts, end_ts),
                ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM episodic_entries").fetchone()
                return int(row[0]) if row else 0

    def _row_to_entry(self, row: tuple) -> EpisodicEntry:
        entry_id, ts, etype, content, meta_json, emb_blob = row
        try:
            metadata = json.loads(meta_json) if meta_json else {}
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        embedding: Optional[np.ndarray] = None
        if emb_blob is not None:
            try:
                embedding = np.frombuffer(emb_blob, dtype=np.float32)
            except Exception:
                embedding = None
        return EpisodicEntry(
            id=int(entry_id),
            timestamp=float(ts),
            event_type=str(etype),
            content=str(content),
            metadata=metadata,
            embedding=embedding,
        )

    # ── Search by similarity ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        event_type: Optional[str] = None,
    ) -> List[EpisodicEntry]:
        """Retrieve top-k entries by embedding cosine similarity."""
        if self._embedder is None:
            return self.get_recent(n=k, event_type=event_type)

        try:
            q_vec = np.asarray(self._embedder.encode(query), dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm < 1e-9:
                return []
            q_vec = q_vec / q_norm
        except Exception:
            return self.get_recent(n=k, event_type=event_type)

        # Load all entries with embeddings — small-scale only
        candidates = self.get_recent(n=1000, event_type=event_type)
        scored: List[tuple] = []
        for entry in candidates:
            if entry.embedding is None:
                continue
            norm = float(np.linalg.norm(entry.embedding))
            if norm < 1e-9:
                continue
            sim = float(np.dot(q_vec, entry.embedding / norm))
            scored.append((sim, entry))
        scored.sort(key=lambda t: -t[0])
        return [e for _, e in scored[:k]]
