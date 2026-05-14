"""
Event RAG — semantic retrieval over macro/news events.

Indexes recent macro events (FOMC, CPI, Fed speeches, news headlines) and
provides semantic search by topic. Thin wrapper around the existing
``data/macro/fred_calendar.py`` feed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .embedding_model import embed_text
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class EventRAG:
    """Semantic search over macro/news events."""

    def __init__(self, embed_dim: int = 128, max_age_days: int = 30) -> None:
        self.store = VectorStore(dim=embed_dim, metric="cosine")
        self.max_age_days = int(max_age_days)
        self._n_indexed = 0

    def index_event(
        self,
        event_id: str,
        title: str,
        description: str = "",
        impact: str = "medium",
        timestamp: Optional[float] = None,
    ) -> None:
        """Index a single event."""
        text = f"{title}. {description}" if description else title
        vec = embed_text(text)
        metadata = {
            "title": title,
            "description": description,
            "impact": impact,
            "timestamp": timestamp or time.time(),
        }
        self.store.add(event_id, vec, metadata)
        self._n_indexed += 1

    def retrieve_similar(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Return the top-k events most similar to the query string."""
        if len(self.store) == 0:
            return []
        vec = embed_text(query)
        results = self.store.search(vec, k=k)
        now = time.time()
        formatted = []
        for id, score, meta in results:
            age_days = (now - float(meta.get("timestamp", now))) / 86400.0
            if age_days > self.max_age_days:
                continue
            formatted.append({
                "id": id,
                "similarity": float(score),
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "impact": meta.get("impact", "medium"),
                "age_days": age_days,
            })
        return formatted

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_indexed": self._n_indexed,
            "store_size": len(self.store),
        }
