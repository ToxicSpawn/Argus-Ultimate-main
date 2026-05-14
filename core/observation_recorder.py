"""
Observation Recorder — experience replay buffer for continuous learning.

Records every (input, decision, outcome) triple so the adaptation engine
can compute gradients on any parameter or rule. This is the foundation of
"learn from every observation" — without an observation log, you can't
attribute outcomes back to inputs.

Stored observations:
  - Market state at decision time (prices, regime, vol)
  - Decision made (signal, sizing, gate decisions)
  - Outcome (filled? what price? P&L? duration?)
  - Counterfactual (what alternatives existed)

The buffer is bounded (default 100,000 observations) and supports:
  - Random sampling for batch training
  - Recency-weighted sampling for online learning
  - Filtered queries (by strategy, regime, symbol, outcome)
  - Time-window queries
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """One complete (input, decision, outcome) triple."""
    obs_id: str
    timestamp: float

    # INPUT — market state at decision time
    symbol: str
    regime: str
    price: float
    volatility: float
    spread_bps: float
    market_features: Dict[str, float] = field(default_factory=dict)

    # DECISION — what ARGUS chose to do
    strategy: str = ""
    action: str = ""              # BUY / SELL / HOLD
    confidence: float = 0.0
    raw_size_pct: float = 0.0
    final_size_pct: float = 0.0
    gate_decisions: Dict[str, str] = field(default_factory=dict)
    gate_multipliers: Dict[str, float] = field(default_factory=dict)

    # OUTCOME — what actually happened
    executed: bool = False
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    slippage_bps: float = 0.0
    pnl_aud: float = 0.0
    holding_seconds: float = 0.0
    exit_reason: str = ""

    # COUNTERFACTUAL — alternatives that existed
    alternative_actions: List[Dict[str, Any]] = field(default_factory=list)

    # METADATA
    cycle_number: int = 0
    health_score: float = 100.0
    portfolio_value_aud: float = 0.0


class ObservationRecorder:
    """
    Bounded experience buffer + query interface.

    Usage::

        recorder = ObservationRecorder(max_size=100_000)

        # On each decision:
        obs = recorder.record_decision(
            symbol="BTC/USD", regime="TRENDING_UP",
            price=60000.0, strategy="momentum",
            action="BUY", confidence=0.7,
            final_size_pct=0.15,
        )

        # On fill:
        recorder.complete_observation(obs.obs_id, fill_price=60005.0, ...)

        # Query:
        wins = recorder.query(strategy="momentum", min_pnl=0)
        recent = recorder.recent(hours=24)
        sample = recorder.sample(100)
    """

    def __init__(
        self,
        max_size: int = 100_000,
        persist_path: Optional[str] = None,
        persist_interval_obs: int = 1000,
    ) -> None:
        self._max_size = max_size
        self._buffer: deque[Observation] = deque(maxlen=max_size)
        self._index_by_id: Dict[str, Observation] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self._persist_interval = persist_interval_obs
        self._records_since_persist = 0
        self._counter = 0
        self._stats = {
            "recorded": 0,
            "completed": 0,
            "evicted": 0,
        }
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("ObservationRecorder: initialized (max_size=%d)", max_size)

    def record_decision(
        self,
        symbol: str,
        regime: str,
        price: float,
        strategy: str,
        action: str,
        confidence: float,
        final_size_pct: float,
        cycle_number: int = 0,
        **kwargs: Any,
    ) -> Observation:
        """Record a decision before fill is known. Returns observation handle."""
        self._counter += 1
        obs = Observation(
            obs_id=f"obs_{int(time.time()*1000)}_{self._counter}",
            timestamp=time.time(),
            symbol=symbol,
            regime=regime,
            price=price,
            volatility=float(kwargs.get("volatility", 0.0)),
            spread_bps=float(kwargs.get("spread_bps", 0.0)),
            market_features=kwargs.get("market_features", {}),
            strategy=strategy,
            action=action,
            confidence=confidence,
            raw_size_pct=float(kwargs.get("raw_size_pct", final_size_pct)),
            final_size_pct=final_size_pct,
            gate_decisions=kwargs.get("gate_decisions", {}),
            gate_multipliers=kwargs.get("gate_multipliers", {}),
            cycle_number=cycle_number,
            health_score=float(kwargs.get("health_score", 100.0)),
            portfolio_value_aud=float(kwargs.get("portfolio_value_aud", 0.0)),
            alternative_actions=kwargs.get("alternative_actions", []),
        )

        # Evict oldest if buffer full
        if len(self._buffer) == self._max_size and self._buffer:
            evicted = self._buffer[0]
            self._index_by_id.pop(evicted.obs_id, None)
            self._stats["evicted"] += 1

        self._buffer.append(obs)
        self._index_by_id[obs.obs_id] = obs
        self._stats["recorded"] += 1
        self._records_since_persist += 1

        if self._persist_path and self._records_since_persist >= self._persist_interval:
            self._persist()
            self._records_since_persist = 0

        return obs

    def complete_observation(
        self,
        obs_id: str,
        executed: bool = True,
        fill_price: float = 0.0,
        fill_quantity: float = 0.0,
        slippage_bps: float = 0.0,
        pnl_aud: float = 0.0,
        holding_seconds: float = 0.0,
        exit_reason: str = "",
    ) -> bool:
        """Update an observation with actual outcome data."""
        obs = self._index_by_id.get(obs_id)
        if obs is None:
            return False
        obs.executed = executed
        obs.fill_price = fill_price
        obs.fill_quantity = fill_quantity
        obs.slippage_bps = slippage_bps
        obs.pnl_aud = pnl_aud
        obs.holding_seconds = holding_seconds
        obs.exit_reason = exit_reason
        self._stats["completed"] += 1
        return True

    def query(
        self,
        *,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        action: Optional[str] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        min_confidence: Optional[float] = None,
        executed_only: bool = False,
        completed_only: bool = False,
    ) -> List[Observation]:
        """Filter observations by criteria."""
        results: List[Observation] = []
        for obs in self._buffer:
            if symbol and obs.symbol != symbol:
                continue
            if strategy and obs.strategy != strategy:
                continue
            if regime and obs.regime != regime:
                continue
            if action and obs.action != action:
                continue
            if min_pnl is not None and obs.pnl_aud < min_pnl:
                continue
            if max_pnl is not None and obs.pnl_aud > max_pnl:
                continue
            if min_confidence is not None and obs.confidence < min_confidence:
                continue
            if executed_only and not obs.executed:
                continue
            if completed_only and obs.fill_price == 0.0:
                continue
            results.append(obs)
        return results

    def recent(self, hours: float = 24) -> List[Observation]:
        """All observations within the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return [obs for obs in self._buffer if obs.timestamp >= cutoff]

    def sample(self, n: int, seed: Optional[int] = None) -> List[Observation]:
        """Random sample of N observations (for batch training)."""
        if not self._buffer:
            return []
        rng = random.Random(seed)
        n = min(n, len(self._buffer))
        return rng.sample(list(self._buffer), n)

    def recency_weighted_sample(self, n: int, decay: float = 0.99) -> List[Observation]:
        """
        Sample weighted by recency. Most recent observations have highest weight.
        """
        if not self._buffer:
            return []
        buf = list(self._buffer)
        weights = [decay ** (len(buf) - i - 1) for i in range(len(buf))]
        n = min(n, len(buf))
        return random.choices(buf, weights=weights, k=n)

    def get(self, obs_id: str) -> Optional[Observation]:
        return self._index_by_id.get(obs_id)

    def aggregate_pnl_by(
        self,
        group_by: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group observations by attribute and aggregate P&L.
        Useful for: 'strategy', 'regime', 'symbol', 'action'.
        """
        groups: Dict[str, Dict[str, Any]] = {}
        for obs in self._buffer:
            if not obs.executed:
                continue
            key = getattr(obs, group_by, "unknown")
            if key not in groups:
                groups[key] = {
                    "count": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "win_rate": 0.0,
                }
            g = groups[key]
            g["count"] += 1
            g["total_pnl"] += obs.pnl_aud
            if obs.pnl_aud > 0:
                g["wins"] += 1
            elif obs.pnl_aud < 0:
                g["losses"] += 1

        for g in groups.values():
            if g["count"] > 0:
                g["avg_pnl"] = g["total_pnl"] / g["count"]
                g["win_rate"] = g["wins"] / g["count"]
        return groups

    def _persist(self) -> None:
        """Append observations to JSONL file (best-effort, non-blocking)."""
        if self._persist_path is None:
            return
        try:
            with self._persist_path.open("a", encoding="utf-8") as f:
                for obs in list(self._buffer)[-self._persist_interval:]:
                    f.write(json.dumps(asdict(obs), default=str) + "\n")
        except Exception as exc:
            logger.debug("ObservationRecorder persist error: %s", exc)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "size": len(self._buffer),
            "max_size": self._max_size,
            "recorded": self._stats["recorded"],
            "completed": self._stats["completed"],
            "evicted": self._stats["evicted"],
            "completion_rate": (
                self._stats["completed"] / max(self._stats["recorded"], 1)
            ),
        }
