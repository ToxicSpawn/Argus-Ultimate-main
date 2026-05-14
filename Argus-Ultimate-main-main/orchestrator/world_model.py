"""
orchestrator/world_model.py
============================
Unified World Model — the single source of truth for "what is happening right now."

Maintains a coherent, real-time picture of:
  - Market state (regime, volatility, liquidity, correlations)
  - Self state (positions, P&L, risk exposure)
  - Causal model (why things move, counterfactuals)
  - Agent state (performance, confidence, conflicts)
  - Intent (goals, active strategies, capital allocation)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Regime(Enum):
    """Market regime classification."""
    LOW_VOL       = "low_vol"
    NORMAL        = "normal"
    HIGH_VOL      = "high_vol"
    CRISIS        = "crisis"
    TRANSITION    = "transition"
    UNKNOWN       = "unknown"


class AgentHealth(Enum):
    """Agent health status."""
    HEALTHY       = "healthy"
    DEGRADED      = "degraded"
    FAILING       = "failing"
    ISOLATED      = "isolated"
    REPAIRING     = "repairing"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Current regime state with prediction."""
    current         : Regime = Regime.UNKNOWN
    predicted       : Regime = Regime.UNKNOWN
    confidence      : float  = 0.5
    transition_prob : float  = 0.0  # probability of regime change in next N minutes
    duration_minutes: float  = 0.0  # how long in current regime
    volatility      : float  = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current"         : self.current.value,
            "predicted"       : self.predicted.value,
            "confidence"      : self.confidence,
            "transition_prob" : self.transition_prob,
            "duration_minutes": self.duration_minutes,
            "volatility"      : self.volatility,
        }


@dataclass
class LiquidityState:
    """Per-venue liquidity state."""
    venue           : str
    bid_depth       : float  # total bid depth in USD
    ask_depth       : float  # total ask depth in USD
    spread_bps      : float
    imbalance       : float  # -1 to +1 (sell heavy to buy heavy)
    last_update     : float  = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue"      : self.venue,
            "bid_depth"  : self.bid_depth,
            "ask_depth"  : self.ask_depth,
            "spread_bps" : self.spread_bps,
            "imbalance"  : self.imbalance,
        }


@dataclass
class PositionState:
    """Current position state."""
    symbol          : str
    quantity        : float
    notional        : float
    entry_price     : float
    current_price   : float
    unrealized_pnl  : float
    side            : str  # "long", "short", "flat"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol"         : self.symbol,
            "quantity"       : self.quantity,
            "notional"       : self.notional,
            "entry_price"    : self.entry_price,
            "current_price"  : self.current_price,
            "unrealized_pnl" : self.unrealized_pnl,
            "side"           : self.side,
        }


@dataclass
class RiskState:
    """Current risk state."""
    var_95          : float = 0.0
    var_99          : float = 0.0
    cvar_95         : float = 0.0
    max_drawdown_pct: float = 0.0
    current_dd_pct  : float = 0.0
    leverage        : float = 0.0
    risk_utilisation: float = 0.0  # 0-1, how close to limits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var_95"          : self.var_95,
            "var_99"          : self.var_99,
            "cvar_95"         : self.cvar_95,
            "max_drawdown_pct": self.max_drawdown_pct,
            "current_dd_pct"  : self.current_dd_pct,
            "leverage"        : self.leverage,
            "risk_utilisation": self.risk_utilisation,
        }


@dataclass
class AgentState:
    """State of a single agent in the system."""
    name            : str
    category        : str
    health          : AgentHealth = AgentHealth.HEALTHY
    confidence      : float = 0.5
    decisions_made  : int = 0
    success_rate    : float = 0.0
    last_active     : float = field(default_factory=time.time)
    weight          : float = 1.0  # contribution weight to decisions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name"          : self.name,
            "category"      : self.category,
            "health"        : self.health.value,
            "confidence"    : self.confidence,
            "decisions_made": self.decisions_made,
            "success_rate"  : self.success_rate,
            "weight"        : self.weight,
        }


@dataclass
class Goal:
    """An active goal for the system."""
    name            : str
    priority        : int  # 1-10
    target          : Dict[str, Any]
    progress        : float  # 0-1
    deadline        : Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name"    : self.name,
            "priority": self.priority,
            "target"  : self.target,
            "progress": self.progress,
            "deadline": self.deadline,
        }


# ---------------------------------------------------------------------------
# World Model
# ---------------------------------------------------------------------------

class WorldModel:
    """
    Unified World Model — the single source of truth.

    Thread-safe, updated by perception agents, read by all other agents.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Market state
        self.regime         : RegimeState = RegimeState()
        self.liquidity      : Dict[str, LiquidityState] = {}
        self.correlation    : Optional[np.ndarray] = None
        self.correlation_ts : float = 0.0

        # Self state
        self.positions      : Dict[str, PositionState] = {}
        self.portfolio_value: float = 0.0
        self.cash           : float = 0.0
        self.daily_pnl      : float = 0.0
        self.risk           : RiskState = RiskState()

        # Agent state
        self.agents         : Dict[str, AgentState] = {}

        # Intent
        self.goals          : List[Goal] = []
        self.active_strategies: List[str] = []
        self.allocations    : Dict[str, float] = {}

        # Causal model (simplified)
        self.causal_edges   : List[Tuple[str, str, float]] = []  # (cause, effect, strength)
        self.causal_confidence: float = 0.0

        # Timestamps
        self.last_update    : float = time.time()
        self.created_at     : float = time.time()

        logger.info("WorldModel: initialised")

    # ------------------------------------------------------------------ Updates (called by perception agents)

    def update_regime(self, regime: RegimeState) -> None:
        with self._lock:
            self.regime = regime
            self.last_update = time.time()

    def update_liquidity(self, venue: str, liquidity: LiquidityState) -> None:
        with self._lock:
            self.liquidity[venue] = liquidity
            self.last_update = time.time()

    def update_position(self, symbol: str, position: PositionState) -> None:
        with self._lock:
            self.positions[symbol] = position
            self.last_update = time.time()

    def update_portfolio(self, value: float, cash: float, daily_pnl: float) -> None:
        with self._lock:
            self.portfolio_value = value
            self.cash = cash
            self.daily_pnl = daily_pnl
            self.last_update = time.time()

    def update_risk(self, risk: RiskState) -> None:
        with self._lock:
            self.risk = risk
            self.last_update = time.time()

    def update_correlation(self, matrix: np.ndarray) -> None:
        with self._lock:
            self.correlation = matrix.copy()
            self.correlation_ts = time.time()

    def update_agent(self, agent: AgentState) -> None:
        with self._lock:
            self.agents[agent.name] = agent

    def update_goals(self, goals: List[Goal]) -> None:
        with self._lock:
            self.goals = sorted(goals, key=lambda g: g.priority, reverse=True)

    def update_allocations(self, allocations: Dict[str, float]) -> None:
        with self._lock:
            self.allocations = allocations

    def add_causal_edge(self, cause: str, effect: str, strength: float) -> None:
        with self._lock:
            self.causal_edges.append((cause, effect, strength))
            self.causal_confidence = min(1.0, self.causal_confidence + 0.01)

    # ------------------------------------------------------------------ Queries (read by reasoning/acting agents)

    def get_regime(self) -> RegimeState:
        with self._lock:
            return self.regime

    def get_positions(self) -> Dict[str, PositionState]:
        with self._lock:
            return dict(self.positions)

    def get_total_exposure(self) -> float:
        with self._lock:
            return sum(abs(p.notional) for p in self.positions.values())

    def get_risk(self) -> RiskState:
        with self._lock:
            return self.risk

    def get_agent(self, name: str) -> Optional[AgentState]:
        with self._lock:
            return self.agents.get(name)

    def get_top_agents(self, category: str, n: int = 5) -> List[AgentState]:
        """Get top N agents by success rate in a category."""
        with self._lock:
            agents = [a for a in self.agents.values() if a.category == category]
            return sorted(agents, key=lambda a: a.success_rate * a.weight, reverse=True)[:n]

    def get_goals(self) -> List[Goal]:
        with self._lock:
            return list(self.goals)

    def get_causal_parents(self, effect: str) -> List[Tuple[str, float]]:
        """Get all known causes of an effect."""
        with self._lock:
            return [(cause, strength) for cause, eff, strength in self.causal_edges if eff == effect]

    def get_causal_children(self, cause: str) -> List[Tuple[str, float]]:
        """Get all known effects of a cause."""
        with self._lock:
            return [(effect, strength) for c, effect, strength in self.causal_edges if c == cause]

    # ------------------------------------------------------------------ Snapshot

    def snapshot(self) -> Dict[str, Any]:
        """Full world model snapshot for dashboard/debugging."""
        with self._lock:
            return {
                "timestamp"      : time.time(),
                "uptime_seconds" : time.time() - self.created_at,
                "regime"         : self.regime.to_dict(),
                "positions"      : {s: p.to_dict() for s, p in self.positions.items()},
                "portfolio_value": self.portfolio_value,
                "cash"           : self.cash,
                "daily_pnl"      : self.daily_pnl,
                "risk"           : self.risk.to_dict(),
                "liquidity"      : {v: l.to_dict() for v, l in self.liquidity.items()},
                "agents"         : {n: a.to_dict() for n, a in self.agents.items()},
                "goals"          : [g.to_dict() for g in self.goals],
                "allocations"    : self.allocations,
                "causal_edges"   : len(self.causal_edges),
                "causal_confidence": self.causal_confidence,
            }

    # ------------------------------------------------------------------ Reset

    def reset(self) -> None:
        """Reset world model (for new trading session)."""
        with self._lock:
            self.regime = RegimeState()
            self.liquidity.clear()
            self.correlation = None
            self.positions.clear()
            self.portfolio_value = 0.0
            self.cash = 0.0
            self.daily_pnl = 0.0
            self.risk = RiskState()
            self.agents.clear()
            self.goals.clear()
            self.allocations.clear()
            self.causal_edges.clear()
            self.causal_confidence = 0.0
            self.last_update = time.time()
            logger.info("WorldModel: reset")
