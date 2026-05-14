"""
Semantic memory — compressed, high-level knowledge graph.

Stores distilled "facts" extracted from episodic memory via LLM or
rule-based summarization. Each fact has a subject, predicate, object,
and an activation strength that decays over time but is reinforced on
retrieval.

Written to the SAME database as episodic memory
(``data/semantic_memory.db``) but in a separate table. Still DISJOINT
from ``cross_session_memory.sqlite``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SemanticFact
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class SemanticFact:
    """A compressed knowledge graph fact."""

    id: int
    subject: str   # e.g. "BTC/USD"
    predicate: str  # e.g. "exhibits_regime_preference"
    object: str    # e.g. "momentum in HIGH_VOL regime"
    confidence: float  # [0, 1]
    activation: float  # decays over time
    created_at: float
    last_accessed: float
    source_episode_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decayed_activation(self, tau_seconds: float = 86400 * 7) -> float:
        """Activation with exponential time decay. Default tau = 7 days."""
        age = max(0.0, time.time() - self.last_accessed)
        return float(self.activation * math.exp(-age / tau_seconds))


# ═════════════════════════════════════════════════════════════════════════════
# SemanticMemory
# ═════════════════════════════════════════════════════════════════════════════


class SemanticMemory:
    """
    Compressed knowledge graph of semantic facts.

    Parameters
    ----------
    db_path : str, default ``data/semantic_memory.db``
    """

    def __init__(
        self,
        db_path: str = "data/semantic_memory.db",
    ) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    activation REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    source_episode_ids TEXT,
                    metadata TEXT,
                    UNIQUE(subject, predicate, object)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_facts (subject)"
            )
            conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 0.5,
        source_episode_ids: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert or reinforce a fact. On conflict (same subject/predicate/object),
        bumps activation and averages confidence.
        """
        now = time.time()
        source_episode_ids = list(source_episode_ids or [])
        metadata = dict(metadata or {})

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT id, confidence, activation FROM semantic_facts "
                    "WHERE subject=? AND predicate=? AND object=?",
                    (subject, predicate, obj),
                ).fetchone()

                if existing:
                    fact_id = int(existing[0])
                    old_conf = float(existing[1])
                    old_act = float(existing[2])
                    new_conf = 0.5 * (old_conf + confidence)
                    new_act = min(5.0, old_act + 0.5)
                    conn.execute(
                        "UPDATE semantic_facts SET confidence=?, activation=?, last_accessed=?, "
                        "source_episode_ids=? WHERE id=?",
                        (new_conf, new_act, now, json.dumps(source_episode_ids), fact_id),
                    )
                    conn.commit()
                    return fact_id

                cur = conn.execute(
                    "INSERT INTO semantic_facts (subject, predicate, object, confidence, "
                    "activation, created_at, last_accessed, source_episode_ids, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        subject, predicate, obj, float(confidence), 1.0,
                        now, now, json.dumps(source_episode_ids), json.dumps(metadata),
                    ),
                )
                conn.commit()
                return int(cur.lastrowid or 0)

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        min_confidence: float = 0.0,
        min_activation: float = 0.0,
        limit: int = 100,
    ) -> List[SemanticFact]:
        """Query semantic facts with optional filters."""
        query = "SELECT id, subject, predicate, object, confidence, activation, " \
                "created_at, last_accessed, source_episode_ids, metadata " \
                "FROM semantic_facts WHERE confidence >= ? AND activation >= ?"
        params: List[Any] = [min_confidence, min_activation]
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)
        query += " ORDER BY activation DESC, last_accessed DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(query, params).fetchall()

        out: List[SemanticFact] = []
        for r in rows:
            (
                fact_id, subj, pred, obj, conf, act, created, accessed,
                src_json, meta_json,
            ) = r
            try:
                sources = json.loads(src_json) if src_json else []
            except (TypeError, json.JSONDecodeError):
                sources = []
            try:
                metadata = json.loads(meta_json) if meta_json else {}
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            out.append(SemanticFact(
                id=int(fact_id),
                subject=str(subj),
                predicate=str(pred),
                object=str(obj),
                confidence=float(conf),
                activation=float(act),
                created_at=float(created),
                last_accessed=float(accessed),
                source_episode_ids=sources,
                metadata=metadata,
            ))
        return out

    def count(self) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM semantic_facts").fetchone()
                return int(row[0]) if row else 0

    def decay_all(self, factor: float = 0.95) -> int:
        """Apply exponential decay to all activation values."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE semantic_facts SET activation = activation * ?",
                    (float(factor),),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT COUNT(*) FROM semantic_facts WHERE activation < 0.05"
                ).fetchone()
                stale = int(row[0]) if row else 0
        return stale

    def prune_stale(self, min_activation: float = 0.05) -> int:
        """Remove facts with activation below threshold."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM semantic_facts WHERE activation < ?",
                    (float(min_activation),),
                )
                conn.commit()
                return int(cur.rowcount or 0)
