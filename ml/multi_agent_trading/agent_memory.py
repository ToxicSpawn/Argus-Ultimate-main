from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MemoryNode:
    node_id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class MemoryEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class AgentDecisionRecord:
    decision_id: str
    symbol: str
    action: str
    confidence: float
    net_score: float
    reasoning_chain: List[str] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class AgentMemory:
    """Neo4j-style graph memory backed by local JSON for portability and resilience."""

    def __init__(self, storage_path: str = "data/multi_agent_trading/agent_memory.json") -> None:
        self.storage_path = Path(storage_path)
        self.persistence_enabled = True
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        self.decisions: Dict[str, AgentDecisionRecord] = {}
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.persistence_enabled = False
            logger.warning("AgentMemory: persistence disabled during initialization: %s", exc)
        self._load()

    def store_decision(
        self,
        symbol: str,
        action: str,
        confidence: float,
        net_score: float,
        reasoning_chain: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        decision_id = uuid.uuid4().hex
        record = AgentDecisionRecord(
            decision_id=decision_id,
            symbol=str(symbol),
            action=str(action),
            confidence=float(confidence),
            net_score=float(net_score),
            reasoning_chain=list(reasoning_chain),
            outcome=dict(metadata or {}),
        )
        self.decisions[decision_id] = record
        self.nodes[decision_id] = MemoryNode(
            node_id=decision_id,
            label="Decision",
            properties={
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "net_score": net_score,
            },
        )
        self._persist()
        return decision_id

    def track_reasoning_chain(self, decision_id: str, reasoning_chain: Sequence[str]) -> List[str]:
        record = self.decisions.get(decision_id)
        if record is None:
            raise KeyError(f"Unknown decision_id: {decision_id}")
        step_node_ids: List[str] = []
        previous_node_id = decision_id
        for index, step in enumerate(reasoning_chain):
            node_id = uuid.uuid4().hex
            self.nodes[node_id] = MemoryNode(
                node_id=node_id,
                label="ReasoningStep",
                properties={"index": index, "text": str(step)},
            )
            edge_id = uuid.uuid4().hex
            self.edges[edge_id] = MemoryEdge(
                edge_id=edge_id,
                source_id=previous_node_id,
                target_id=node_id,
                relation="LEADS_TO",
                properties={"index": index},
            )
            previous_node_id = node_id
            step_node_ids.append(node_id)
        self._persist()
        return step_node_ids

    def store_outcome(self, decision_id: str, outcome: Dict[str, Any]) -> None:
        record = self.decisions.get(decision_id)
        if record is None:
            raise KeyError(f"Unknown decision_id: {decision_id}")
        record.outcome.update(dict(outcome or {}))
        node_id = uuid.uuid4().hex
        self.nodes[node_id] = MemoryNode(node_id=node_id, label="Outcome", properties=dict(outcome or {}))
        edge_id = uuid.uuid4().hex
        self.edges[edge_id] = MemoryEdge(
            edge_id=edge_id,
            source_id=decision_id,
            target_id=node_id,
            relation="RESULTED_IN",
        )
        self._persist()

    def get_related_decisions(self, symbol: str, limit: int = 10) -> List[AgentDecisionRecord]:
        relevant = [record for record in self.decisions.values() if record.symbol == symbol]
        relevant.sort(key=lambda item: item.created_at, reverse=True)
        return relevant[: max(1, int(limit))]

    def learn_from_feedback(self, symbol: str, realized_return: float, horizon: str = "1d") -> Dict[str, Any]:
        relevant = self.get_related_decisions(symbol, limit=20)
        if not relevant:
            return {"symbol": symbol, "horizon": horizon, "sample_size": 0, "avg_quality": 0.0}
        quality_scores: List[float] = []
        for record in relevant:
            if record.action == "hold":
                directional_quality = 0.5
            elif (record.action == "buy" and realized_return > 0.0) or (record.action == "sell" and realized_return < 0.0):
                directional_quality = 1.0
            else:
                directional_quality = 0.0
            quality_scores.append(directional_quality * min(abs(realized_return) + record.confidence, 1.0))
        summary = {
            "symbol": symbol,
            "horizon": horizon,
            "sample_size": len(relevant),
            "avg_quality": sum(quality_scores) / len(quality_scores),
            "realized_return": float(realized_return),
        }
        logger.info("AgentMemory: feedback for %s yielded avg_quality=%.2f", symbol, summary["avg_quality"])
        return summary

    def learn_from_decision(self, decision_id: str, realized_return: float, horizon: str = "1d") -> Dict[str, Any]:
        record = self.decisions.get(decision_id)
        if record is None:
            raise KeyError(f"Unknown decision_id: {decision_id}")
        if record.action == "hold":
            directional_quality = 0.5
        elif (record.action == "buy" and realized_return > 0.0) or (record.action == "sell" and realized_return < 0.0):
            directional_quality = 1.0
        else:
            directional_quality = 0.0
        quality = directional_quality * min(abs(realized_return) + record.confidence, 1.0)
        summary = {
            "decision_id": decision_id,
            "symbol": record.symbol,
            "horizon": horizon,
            "realized_return": float(realized_return),
            "quality": quality,
        }
        record.outcome.update(summary)
        self._persist()
        return summary

    def _persist(self) -> None:
        if not self.persistence_enabled:
            return
        payload = {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()],
            "decisions": [asdict(record) for record in self.decisions.values()],
        }
        try:
            with self.storage_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2)
        except OSError as exc:
            logger.warning("AgentMemory: failed to persist graph memory: %s", exc)

    def _load(self) -> None:
        if not self.persistence_enabled:
            return
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.nodes = {item["node_id"]: MemoryNode(**item) for item in payload.get("nodes", [])}
            self.edges = {item["edge_id"]: MemoryEdge(**item) for item in payload.get("edges", [])}
            self.decisions = {item["decision_id"]: AgentDecisionRecord(**item) for item in payload.get("decisions", [])}
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("AgentMemory: failed to load %s: %s", self.storage_path, exc)
