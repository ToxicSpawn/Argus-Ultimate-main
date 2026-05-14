"""
NeuralArB-style multi-agent RL simulation scaffold for Argus-Ultimate v5.0.0.

Implements a lightweight multi-agent environment with specialized agents:
- market maker agent
- arbitrage / spread agent
- inventory risk agent
- execution optimizer agent

Designed as an extensible simulation harness rather than a full training stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class AgentObservation:
    mid_price: float
    spread_bps: float
    imbalance: float
    volatility: float
    inventory: float
    latency_ms: float
    fees_bps: float


@dataclass
class AgentAction:
    side: str
    size_pct: float
    urgency: float
    quote_offset_bps: float


@dataclass
class AgentStepResult:
    reward: float
    pnl: float
    inventory_change: float
    info: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all trading agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.total_reward = 0.0
        self.total_pnl = 0.0
        self.steps = 0

    @abstractmethod
    def act(self, obs: AgentObservation) -> AgentAction:
        """Generate an action given current observation."""
        ...

    def observe_result(self, result: AgentStepResult) -> None:
        self.total_reward += result.reward
        self.total_pnl += result.pnl
        self.steps += 1

    def stats(self) -> Dict[str, float]:
        return {
            "reward": self.total_reward,
            "pnl": self.total_pnl,
            "steps": self.steps,
        }


class MarketMakerAgent(BaseAgent):
    def __init__(self):
        super().__init__("market_maker")

    def act(self, obs: AgentObservation) -> AgentAction:
        quote_offset = max(obs.spread_bps * 0.4, 0.8)
        size_pct = min(0.04 + obs.spread_bps / 300.0, 0.12)
        return AgentAction(side="both", size_pct=size_pct, urgency=0.2, quote_offset_bps=quote_offset)


class ArbitrageAgent(BaseAgent):
    def __init__(self):
        super().__init__("arbitrage")

    def act(self, obs: AgentObservation) -> AgentAction:
        side = "buy" if obs.imbalance > 0 else "sell"
        size_pct = min(abs(obs.imbalance) * 0.08 + 0.02, 0.10)
        urgency = min(obs.latency_ms / 10.0, 1.0)
        return AgentAction(side=side, size_pct=size_pct, urgency=urgency, quote_offset_bps=0.3)


class InventoryRiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("inventory_risk")

    def act(self, obs: AgentObservation) -> AgentAction:
        if abs(obs.inventory) < 0.15:
            return AgentAction(side="hold", size_pct=0.0, urgency=0.0, quote_offset_bps=0.0)
        side = "sell" if obs.inventory > 0 else "buy"
        size_pct = min(abs(obs.inventory) * 0.12, 0.14)
        return AgentAction(side=side, size_pct=size_pct, urgency=0.8, quote_offset_bps=0.1)


class ExecutionOptimizerAgent(BaseAgent):
    def __init__(self):
        super().__init__("execution_optimizer")

    def act(self, obs: AgentObservation) -> AgentAction:
        maker_preferred = obs.spread_bps > obs.fees_bps * 1.5
        side = "buy" if obs.imbalance >= 0 else "sell"
        size_pct = 0.03 if maker_preferred else 0.015
        urgency = 0.25 if maker_preferred else 0.7
        quote_offset = 1.0 if maker_preferred else 0.2
        return AgentAction(side=side, size_pct=size_pct, urgency=urgency, quote_offset_bps=quote_offset)


class MultiAgentCoordinator:
    """
    Coordinates specialized agents and combines their actions.
    NeuralArB-inspired decomposition: each agent optimizes one micro-objective.
    """

    def __init__(self, capital: float = 1000.0):
        self.capital = capital
        self.agents: List[BaseAgent] = [
            MarketMakerAgent(),
            ArbitrageAgent(),
            InventoryRiskAgent(),
            ExecutionOptimizerAgent(),
        ]
        self.inventory = 0.0
        self.cash = capital
        self.step_num = 0

    def observe(self, market: Dict[str, float]) -> AgentObservation:
        return AgentObservation(
            mid_price=market.get("mid_price", 100.0),
            spread_bps=market.get("spread_bps", 2.0),
            imbalance=market.get("imbalance", 0.0),
            volatility=market.get("volatility", 0.01),
            inventory=self.inventory,
            latency_ms=market.get("latency_ms", 2.0),
            fees_bps=market.get("fees_bps", 1.0),
        )

    def combine_actions(self, actions: List[AgentAction]) -> AgentAction:
        """
        Weighted combination of agent actions.
        Inventory risk gets veto-like priority when inventory is high.
        """
        if not actions:
            return AgentAction(side="hold", size_pct=0.0, urgency=0.0, quote_offset_bps=0.0)

        side_scores = {"buy": 0.0, "sell": 0.0, "both": 0.0, "hold": 0.0}
        total_size = 0.0
        total_urgency = 0.0
        total_offset = 0.0

        for idx, action in enumerate(actions):
            weight = 1.0
            agent = self.agents[idx]
            if agent.name == "inventory_risk" and abs(self.inventory) > 0.20:
                weight = 2.5
            elif agent.name == "arbitrage":
                weight = 1.2
            elif agent.name == "execution_optimizer":
                weight = 1.1

            side_scores[action.side] = side_scores.get(action.side, 0.0) + weight
            total_size += action.size_pct * weight
            total_urgency += action.urgency * weight
            total_offset += action.quote_offset_bps * weight

        side = max(side_scores, key=side_scores.get)
        norm = max(sum(side_scores.values()), 1e-9)
        return AgentAction(
            side=side,
            size_pct=min(total_size / norm, 0.15),
            urgency=min(total_urgency / norm, 1.0),
            quote_offset_bps=max(total_offset / norm, 0.0),
        )

    def simulate_execution(self, action: AgentAction, obs: AgentObservation) -> AgentStepResult:
        """
        Lightweight fill / reward model.
        """
        if action.side == "hold":
            return AgentStepResult(reward=0.0, pnl=0.0, inventory_change=0.0, info={"filled": False})

        fill_prob = max(0.1, 1.0 - action.quote_offset_bps / max(obs.spread_bps + 1e-6, 1.0))
        filled = random.random() < fill_prob
        inventory_change = 0.0
        pnl = 0.0

        if filled:
            direction = 1.0 if action.side == "buy" else -1.0 if action.side == "sell" else 0.0
            if action.side == "both":
                pnl = self.capital * action.size_pct * (obs.spread_bps / 10000.0) * 0.5
            else:
                edge = (abs(obs.imbalance) * 0.002) - (obs.fees_bps / 10000.0)
                pnl = self.capital * action.size_pct * edge * (1.0 + (0.5 - obs.volatility))
                inventory_change = direction * action.size_pct

        reward = pnl - abs(self.inventory) * 0.05 - obs.volatility * action.size_pct * self.capital * 0.01
        return AgentStepResult(
            reward=reward,
            pnl=pnl,
            inventory_change=inventory_change,
            info={"filled": filled, "fill_prob": fill_prob},
        )

    def step(self, market: Dict[str, float]) -> Dict[str, Any]:
        obs = self.observe(market)
        actions = [agent.act(obs) for agent in self.agents]
        combined = self.combine_actions(actions)
        result = self.simulate_execution(combined, obs)

        self.inventory += result.inventory_change
        self.cash += result.pnl
        self.step_num += 1

        shared_reward = result.reward
        for agent in self.agents:
            agent.observe_result(result)

        return {
            "step": self.step_num,
            "observation": obs,
            "agent_actions": actions,
            "combined_action": combined,
            "result": result,
            "inventory": self.inventory,
            "cash": self.cash,
            "equity": self.cash,
        }

    def run_episode(self, market_path: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        history = []
        for market in market_path:
            history.append(self.step(market))
        return history

    def stats(self) -> Dict[str, Any]:
        return {
            "capital": self.capital,
            "cash": self.cash,
            "inventory": self.inventory,
            "step_num": self.step_num,
            "agents": {agent.name: agent.stats() for agent in self.agents},
        }
