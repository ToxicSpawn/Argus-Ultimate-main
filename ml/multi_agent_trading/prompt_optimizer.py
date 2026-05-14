from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_variants (
    variant_id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    name TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS prompt_performance (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id TEXT NOT NULL,
    market_regime TEXT NOT NULL,
    reward REAL NOT NULL,
    accuracy REAL NOT NULL,
    latency_ms REAL NOT NULL,
    feedback_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


@dataclass(slots=True)
class PromptVariant:
    variant_id: str
    agent_name: str
    name: str
    prompt_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    active: bool = True


@dataclass(slots=True)
class PromptPerformance:
    variant_id: str
    market_regime: str
    reward: float
    accuracy: float
    latency_ms: float
    feedback: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class PromptOptimizer:
    """Tracks prompt variants, A/B tests them, and promotes winners automatically."""

    def __init__(self, db_path: str = "data/multi_agent_trading/prompt_optimizer.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self.persistence_enabled = True
        self._variants_in_memory: Dict[str, PromptVariant] = {}
        self._performance_in_memory: List[PromptPerformance] = []
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
        except (OSError, sqlite3.Error) as exc:
            self.persistence_enabled = False
            logger.warning("PromptOptimizer: sqlite disabled, using in-memory store: %s", exc)

    def register_variant(
        self,
        agent_name: str,
        name: str,
        prompt_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVariant:
        fingerprint = hashlib.sha256(f"{agent_name}|{name}|{prompt_text}".encode("utf-8")).hexdigest()[:16]
        variant = PromptVariant(
            variant_id=fingerprint,
            agent_name=str(agent_name),
            name=str(name),
            prompt_text=str(prompt_text),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._variants_in_memory[variant.variant_id] = variant
            if not self.persistence_enabled:
                return variant
            con = self._connect()
            try:
                con.execute(
                    """
                    INSERT OR REPLACE INTO prompt_variants
                    (variant_id, agent_name, name, prompt_text, metadata_json, created_at, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        variant.variant_id,
                        variant.agent_name,
                        variant.name,
                        variant.prompt_text,
                        json.dumps(variant.metadata, sort_keys=True),
                        variant.created_at,
                        int(variant.active),
                    ),
                )
                con.commit()
            finally:
                con.close()
        return variant

    def record_performance(self, performance: PromptPerformance) -> None:
        with self._lock:
            self._performance_in_memory.append(performance)
            if not self.persistence_enabled:
                return
            con = self._connect()
            try:
                con.execute(
                    """
                    INSERT INTO prompt_performance
                    (variant_id, market_regime, reward, accuracy, latency_ms, feedback_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        performance.variant_id,
                        performance.market_regime,
                        performance.reward,
                        performance.accuracy,
                        performance.latency_ms,
                        json.dumps(performance.feedback, sort_keys=True),
                        performance.created_at,
                    ),
                )
                con.commit()
            finally:
                con.close()

    def choose_variant(self, agent_name: str, market_regime: str = "unknown") -> Optional[PromptVariant]:
        variants = self._get_variants(agent_name)
        if not variants:
            return None
        scored = []
        for variant in variants:
            summary = self.summarize_variant_performance(variant.variant_id, market_regime=market_regime)
            score = (0.60 * summary["avg_reward"]) + (0.30 * summary["avg_accuracy"]) - (0.10 * min(summary["avg_latency_ms"] / 1000.0, 1.0))
            exploration_bonus = 0.10 if summary["sample_size"] < 5 else 0.0
            scored.append((score + exploration_bonus, variant))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def ab_test_variants(self, agent_name: str, variant_a: str, variant_b: str, market_regime: str = "unknown") -> Dict[str, Any]:
        summary_a = self.summarize_variant_performance(variant_a, market_regime=market_regime)
        summary_b = self.summarize_variant_performance(variant_b, market_regime=market_regime)
        score_a = (0.65 * summary_a["avg_reward"]) + (0.35 * summary_a["avg_accuracy"])
        score_b = (0.65 * summary_b["avg_reward"]) + (0.35 * summary_b["avg_accuracy"])
        if score_a > score_b:
            winner = variant_a
        elif score_b > score_a:
            winner = variant_b
        else:
            winner = "tie"
        result = {
            "agent_name": agent_name,
            "market_regime": market_regime,
            "variant_a": summary_a,
            "variant_b": summary_b,
            "winner": winner,
            "score_delta": score_a - score_b,
        }
        logger.info("PromptOptimizer: A/B test for %s selected %s", agent_name, winner)
        return result

    def auto_optimize(self, agent_name: str, market_regime: str = "unknown") -> Dict[str, Any]:
        variants = self._get_variants(agent_name)
        if not variants:
            return {"agent_name": agent_name, "winner": None, "reason": "no variants registered"}
        ranked = []
        for variant in variants:
            summary = self.summarize_variant_performance(variant.variant_id, market_regime=market_regime)
            optimize_score = (0.55 * summary["avg_reward"]) + (0.35 * summary["avg_accuracy"]) - (0.10 * min(summary["avg_latency_ms"] / 1000.0, 1.0))
            ranked.append((optimize_score, variant, summary))
        ranked.sort(key=lambda item: item[0], reverse=True)
        winner_score, winner_variant, winner_summary = ranked[0]
        recommendation = {
            "agent_name": agent_name,
            "market_regime": market_regime,
            "winner": winner_variant.variant_id,
            "winner_name": winner_variant.name,
            "winner_score": winner_score,
            "summary": winner_summary,
            "challengers": [
                {"variant_id": variant.variant_id, "score": score, "summary": summary}
                for score, variant, summary in ranked[1:4]
            ],
        }
        return recommendation

    def summarize_variant_performance(self, variant_id: str, market_regime: Optional[str] = None) -> Dict[str, Any]:
        if not self.persistence_enabled:
            rows = [
                perf for perf in self._performance_in_memory
                if perf.variant_id == variant_id and (market_regime is None or perf.market_regime == market_regime)
            ]
            if not rows:
                return {
                    "variant_id": variant_id,
                    "sample_size": 0,
                    "avg_reward": 0.0,
                    "avg_accuracy": 0.0,
                    "avg_latency_ms": 0.0,
                }
            sample_size = len(rows)
            return {
                "variant_id": variant_id,
                "sample_size": sample_size,
                "avg_reward": sum(float(row.reward) for row in rows) / sample_size,
                "avg_accuracy": sum(float(row.accuracy) for row in rows) / sample_size,
                "avg_latency_ms": sum(float(row.latency_ms) for row in rows) / sample_size,
            }
        query = "SELECT reward, accuracy, latency_ms FROM prompt_performance WHERE variant_id = ?"
        params: List[Any] = [variant_id]
        if market_regime is not None:
            query += " AND market_regime = ?"
            params.append(market_regime)
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(query, tuple(params)).fetchall()
            finally:
                con.close()
        if not rows:
            return {
                "variant_id": variant_id,
                "sample_size": 0,
                "avg_reward": 0.0,
                "avg_accuracy": 0.0,
                "avg_latency_ms": 0.0,
            }
        sample_size = len(rows)
        avg_reward = sum(float(row[0]) for row in rows) / sample_size
        avg_accuracy = sum(float(row[1]) for row in rows) / sample_size
        avg_latency_ms = sum(float(row[2]) for row in rows) / sample_size
        return {
            "variant_id": variant_id,
            "sample_size": sample_size,
            "avg_reward": avg_reward,
            "avg_accuracy": avg_accuracy,
            "avg_latency_ms": avg_latency_ms,
        }

    def _get_variants(self, agent_name: str) -> List[PromptVariant]:
        if not self.persistence_enabled:
            return [variant for variant in self._variants_in_memory.values() if variant.agent_name == agent_name and variant.active]
        with self._lock:
            con = self._connect()
            try:
                rows = con.execute(
                    "SELECT variant_id, agent_name, name, prompt_text, metadata_json, created_at, active FROM prompt_variants WHERE agent_name = ? AND active = 1",
                    (agent_name,),
                ).fetchall()
            finally:
                con.close()
        variants: List[PromptVariant] = []
        for row in rows:
            variants.append(
                PromptVariant(
                    variant_id=str(row[0]),
                    agent_name=str(row[1]),
                    name=str(row[2]),
                    prompt_text=str(row[3]),
                    metadata=json.loads(str(row[4]) or "{}"),
                    created_at=float(row[5]),
                    active=bool(row[6]),
                )
            )
        return variants

    def _init_schema(self) -> None:
        con = self._connect()
        try:
            con.executescript(_SCHEMA)
            con.commit()
        finally:
            con.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def ensure_default_variants(self, agent_name: str) -> List[PromptVariant]:
        existing = self._get_variants(agent_name)
        if existing:
            return existing
        baseline = self.register_variant(
            agent_name,
            "baseline",
            f"Baseline trading prompt for {agent_name}.",
            metadata={"score_bias": 0.0, "confidence_bias": 0.0, "emphasis": "balanced"},
        )
        conservative = self.register_variant(
            agent_name,
            "conservative",
            f"Conservative trading prompt for {agent_name} prioritizing risk-aware reasoning.",
            metadata={"score_bias": -0.03, "confidence_bias": -0.02, "emphasis": "risk"},
        )
        return [baseline, conservative]
