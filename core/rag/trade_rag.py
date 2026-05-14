"""
Trade RAG — semantic retrieval over past trades and decisions.

Given the current trading context (symbol, regime, volatility, recent price
action, etc.), return top-k historically similar setups with their outcomes.
Used by ``_apply_intelligence_gates`` to nudge sizing based on how similar
past setups performed.

The corpus comes from:
- ``monitoring/decision_journal.py`` — per-decision trace records
- ``core/cross_session_memory.py`` — cross-session insights
- ``monitoring/drawdown_autopsy.py`` — post-mortem drawdown reports

This module is READ-ONLY with respect to those corpora (per Plan-agent
risk R1) — it indexes them into a separate vector store.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embedding_model import embed_context, embed_text
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# TradeRAG
# ═════════════════════════════════════════════════════════════════════════════


class TradeRAG:
    """
    Retrieval engine over past trades.

    Parameters
    ----------
    embed_dim : int
        Embedding dimensionality.
    """

    def __init__(self, embed_dim: int = 128) -> None:
        self.store = VectorStore(dim=embed_dim, metric="cosine")
        self._n_indexed = 0
        self._last_reindex_ts: float = 0.0

    def index_decision(
        self,
        decision_id: str,
        context: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single decision record to the RAG store.

        Parameters
        ----------
        decision_id : str
            Unique identifier (e.g. decision_journal entry id).
        context : Dict
            Pre-trade context: symbol, regime, volatility, signal strength,
            advisory keys, etc.
        outcome : Dict, optional
            Post-trade outcome: pnl, holding_period, exit_reason. If None,
            the entry is indexed without outcome metadata.
        """
        vec = embed_context(context)
        metadata = {
            "context": dict(context),
            "outcome": dict(outcome) if outcome else None,
            "timestamp": time.time(),
        }
        self.store.add(decision_id, vec, metadata)
        self._n_indexed += 1

    def index_from_decision_journal(
        self, journal_path: str, max_entries: int = 5000
    ) -> int:
        """
        Bulk-index recent entries from the decision journal JSONL file.

        Parameters
        ----------
        journal_path : str
            Path to the decision_journal.jsonl file.
        max_entries : int
            Maximum number of recent entries to index.

        Returns
        -------
        int
            Number of entries indexed.
        """
        import json
        from pathlib import Path

        p = Path(journal_path)
        if not p.exists():
            return 0

        entries = []
        try:
            with open(p, "r") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.debug("Failed to read journal %s: %s", journal_path, exc)
            return 0

        recent = entries[-max_entries:]
        n = 0
        for i, entry in enumerate(recent):
            try:
                decision_id = str(entry.get("id", f"journal_{i}_{int(time.time())}"))
                context = {
                    "symbol": entry.get("symbol", ""),
                    "side": entry.get("side", ""),
                    "regime": entry.get("regime", ""),
                    "volatility": entry.get("volatility", 0.0),
                    "signal_strength": entry.get("confidence", 0.0),
                    "strategy": entry.get("strategy", ""),
                    "entry_price": entry.get("entry_price", 0.0),
                }
                outcome = entry.get("outcome") or entry.get("result")
                self.index_decision(decision_id, context, outcome)
                n += 1
            except Exception as exc:
                logger.debug("Failed to index journal entry %d: %s", i, exc)
                continue
        self._last_reindex_ts = time.time()
        return n

    def retrieve_similar(
        self,
        current_context: Dict[str, Any],
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k historically similar decisions.

        Parameters
        ----------
        current_context : Dict
            Current context (same schema as indexed contexts).
        k : int
            Number of neighbors to return.

        Returns
        -------
        List[Dict]
            Each entry has: id, similarity, context, outcome, pnl,
            won (bool).
        """
        if len(self.store) == 0:
            return []

        query_vec = embed_context(current_context)
        results = self.store.search(query_vec, k=k)

        formatted = []
        for id, score, meta in results:
            outcome = meta.get("outcome") if isinstance(meta, dict) else None
            pnl = 0.0
            won = False
            if outcome and isinstance(outcome, dict):
                pnl = float(outcome.get("pnl", 0.0) or 0.0)
                won = bool(outcome.get("won", pnl > 0))
            formatted.append({
                "id": id,
                "similarity": float(score),
                "context": meta.get("context", {}) if isinstance(meta, dict) else {},
                "outcome": outcome,
                "pnl": pnl,
                "won": won,
            })
        return formatted

    def win_rate_of_similar(
        self,
        current_context: Dict[str, Any],
        *,
        k: int = 10,
        min_similarity: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Compute the historical win rate of similar setups.

        Returns
        -------
        Dict
            ``{"win_rate", "n_similar", "avg_pnl", "confidence"}`` where
            confidence = n_similar / k.
        """
        neighbors = self.retrieve_similar(current_context, k=k)
        if not neighbors:
            return {
                "win_rate": 0.5,
                "n_similar": 0,
                "avg_pnl": 0.0,
                "confidence": 0.0,
            }
        # Filter by minimum similarity
        usable = [
            n for n in neighbors
            if n["similarity"] >= min_similarity and n["outcome"] is not None
        ]
        if not usable:
            return {
                "win_rate": 0.5,
                "n_similar": 0,
                "avg_pnl": 0.0,
                "confidence": 0.0,
            }
        win_rate = float(sum(1 for n in usable if n["won"]) / len(usable))
        avg_pnl = float(np.mean([n["pnl"] for n in usable]))
        confidence = float(len(usable) / k)
        return {
            "win_rate": win_rate,
            "n_similar": len(usable),
            "avg_pnl": avg_pnl,
            "confidence": confidence,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_indexed": self._n_indexed,
            "store_size": len(self.store),
            "backend": self.store.backend,
            "last_reindex_ts": self._last_reindex_ts,
            "dim": self.store.dim,
        }
