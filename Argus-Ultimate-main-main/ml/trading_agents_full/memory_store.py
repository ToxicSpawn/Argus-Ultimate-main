from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class DecisionRecord:
    decision_id: str
    symbol: str
    action: str
    confidence: float
    regime: str
    net_score: float
    reasoning: List[str] = field(default_factory=list)
    context_signature: Dict[str, float] = field(default_factory=dict)
    outcome_return: Optional[float] = None
    outcome_success: Optional[bool] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class SimilarSituation:
    decision_id: str
    symbol: str
    action: str
    similarity: float
    realized_return: float
    success: bool
    regime: str
    created_at: float


@dataclass(slots=True)
class PatternInsight:
    symbol: str
    regime: str
    sample_size: int
    success_rate: float
    average_return: float
    preferred_action: str
    summary: str


class DecisionMemory:
    def __init__(self, db_path: str = "data/trading_agents_full/decision_memory.sqlite") -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def store_decision(
        self,
        *,
        symbol: str,
        action: str,
        confidence: float,
        regime: str,
        net_score: float,
        reasoning: Sequence[str],
        context_signature: Mapping[str, float],
    ) -> DecisionRecord:
        record = DecisionRecord(
            decision_id=uuid.uuid4().hex,
            symbol=str(symbol),
            action=str(action),
            confidence=float(confidence),
            regime=str(regime),
            net_score=float(net_score),
            reasoning=[str(item) for item in reasoning],
            context_signature={str(k): _safe_float(v) for k, v in context_signature.items()},
        )
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO decisions (
                        decision_id, symbol, action, confidence, regime, net_score,
                        reasoning_json, context_signature_json, outcome_return,
                        outcome_success, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.decision_id,
                        record.symbol,
                        record.action,
                        record.confidence,
                        record.regime,
                        record.net_score,
                        json.dumps(record.reasoning),
                        json.dumps(record.context_signature, sort_keys=True),
                        None,
                        None,
                        record.created_at,
                        record.updated_at,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("DecisionMemory store_decision failed for %s: %s", record.symbol, exc)
        return record

    def update_outcome(self, decision_id: str, realized_return: float) -> None:
        success = 1 if realized_return > 0 else 0
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE decisions SET outcome_return = ?, outcome_success = ?, updated_at = ? WHERE decision_id = ?",
                    (float(realized_return), success, time.time(), decision_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("DecisionMemory update_outcome failed for %s: %s", decision_id, exc)

    def query_similar_situations(
        self,
        symbol: str,
        regime: str,
        context_signature: Mapping[str, float],
        limit: int = 5,
    ) -> List[SimilarSituation]:
        rows: List[sqlite3.Row]
        try:
            with self._lock, self._connect() as conn:
                rows = list(
                    conn.execute(
                        """
                        SELECT decision_id, symbol, action, regime, context_signature_json,
                               COALESCE(outcome_return, 0.0) AS outcome_return,
                               COALESCE(outcome_success, 0) AS outcome_success,
                               created_at
                        FROM decisions
                        WHERE symbol = ? AND regime = ?
                        ORDER BY created_at DESC
                        LIMIT 50
                        """,
                        (symbol, regime),
                    )
                )
        except sqlite3.Error as exc:
            logger.warning("DecisionMemory query_similar_situations failed for %s/%s: %s", symbol, regime, exc)
            return []
        results: List[SimilarSituation] = []
        for row in rows:
            other_signature = json.loads(row["context_signature_json"] or "{}")
            similarity = self._signature_similarity(context_signature, other_signature)
            results.append(
                SimilarSituation(
                    decision_id=str(row["decision_id"]),
                    symbol=str(row["symbol"]),
                    action=str(row["action"]),
                    similarity=similarity,
                    realized_return=float(row["outcome_return"]),
                    success=bool(row["outcome_success"]),
                    regime=str(row["regime"]),
                    created_at=float(row["created_at"]),
                )
            )
        results.sort(key=lambda item: item.similarity, reverse=True)
        return results[: max(1, int(limit))]

    def learn_pattern_summary(self, symbol: str, regime: str) -> PatternInsight:
        try:
            with self._lock, self._connect() as conn:
                rows = list(
                    conn.execute(
                        """
                        SELECT action, COALESCE(outcome_return, 0.0) AS outcome_return,
                               COALESCE(outcome_success, 0) AS outcome_success
                        FROM decisions
                        WHERE symbol = ? AND regime = ? AND outcome_success IS NOT NULL
                        """,
                        (symbol, regime),
                    )
                )
        except sqlite3.Error as exc:
            logger.warning("DecisionMemory learn_pattern_summary failed for %s/%s: %s", symbol, regime, exc)
            return PatternInsight(symbol=symbol, regime=regime, sample_size=0, success_rate=0.0, average_return=0.0, preferred_action="hold", summary="Historical memory unavailable due to storage error.")
        if not rows:
            return PatternInsight(symbol=symbol, regime=regime, sample_size=0, success_rate=0.0, average_return=0.0, preferred_action="hold", summary="No historical outcomes available.")
        returns = [float(row[1]) for row in rows]
        success_rate = sum(int(row[2]) for row in rows) / len(rows)
        average_return = sum(returns) / len(returns)
        action_scores: Dict[str, List[float]] = {}
        for action, realized_return, _success in rows:
            action_scores.setdefault(str(action), []).append(float(realized_return))
        preferred_action = max(action_scores.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]
        return PatternInsight(
            symbol=symbol,
            regime=regime,
            sample_size=len(rows),
            success_rate=success_rate,
            average_return=average_return,
            preferred_action=preferred_action,
            summary=f"Historical pattern for {symbol}/{regime} favors {preferred_action} with {success_rate:.2%} win rate.",
        )

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute("SELECT * FROM decisions WHERE decision_id = ?", (decision_id,)).fetchone()
        except sqlite3.Error as exc:
            logger.warning("DecisionMemory get_decision failed for %s: %s", decision_id, exc)
            return None
        if row is None:
            return None
        return DecisionRecord(
            decision_id=str(row["decision_id"]),
            symbol=str(row["symbol"]),
            action=str(row["action"]),
            confidence=float(row["confidence"]),
            regime=str(row["regime"]),
            net_score=float(row["net_score"]),
            reasoning=list(json.loads(row["reasoning_json"] or "[]")),
            context_signature={str(k): _safe_float(v) for k, v in json.loads(row["context_signature_json"] or "{}").items()},
            outcome_return=_safe_float(row["outcome_return"]) if row["outcome_return"] is not None else None,
            outcome_success=bool(row["outcome_success"]) if row["outcome_success"] is not None else None,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def _initialize(self) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS decisions (
                        decision_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        regime TEXT NOT NULL,
                        net_score REAL NOT NULL,
                        reasoning_json TEXT NOT NULL,
                        context_signature_json TEXT NOT NULL,
                        outcome_return REAL,
                        outcome_success INTEGER,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_symbol_regime ON decisions(symbol, regime)")
                conn.commit()
        except sqlite3.Error as exc:
            logger.warning("DecisionMemory initialization failed for %s: %s", self.db_path, exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _signature_similarity(self, left: Mapping[str, float], right: Mapping[str, float]) -> float:
        keys = sorted(set(left) | set(right))
        if not keys:
            return 0.0
        delta = 0.0
        for key in keys:
            delta += abs(_safe_float(left.get(key)) - _safe_float(right.get(key)))
        normalized = delta / max(len(keys), 1)
        return max(0.0, 1.0 - min(normalized, 1.0))
