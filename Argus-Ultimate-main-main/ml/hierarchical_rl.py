"""
Hierarchical RL module for Argus-Ultimate v5.0.0 (HFT-Pinnacle).
EarnHFT-style hierarchical policy with manager-worker separation.

Manager selects macro regime / objective.
Worker selects low-level execution actions conditioned on manager intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging
import math
import random

logger = logging.getLogger(__name__)


@dataclass
class HRLConfig:
    manager_update_interval: int = 16
    worker_discount: float = 0.99
    manager_discount: float = 0.995
    entropy_coef: float = 0.01
    intrinsic_reward_scale: float = 0.2
    max_position_pct: float = 0.15
    risk_penalty_coef: float = 0.5
    inventory_penalty_coef: float = 0.2


@dataclass
class MarketState:
    mid_price: float
    spread_bps: float
    volatility: float
    imbalance: float
    inventory: float
    cash: float
    regime_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class ManagerDecision:
    objective: str
    target_inventory: float
    aggression: float
    hold_horizon: int


@dataclass
class WorkerAction:
    side: str
    size_pct: float
    order_type: str
    price_offset_bps: float


class HierarchicalPolicy:
    """
    Hierarchical policy with:
    - manager choosing strategic objective every N steps
    - worker emitting execution actions each step
    - intrinsic reward for aligning worker behaviour to manager goals
    """

    OBJECTIVES = [
        "market_make",
        "inventory_rebalance",
        "momentum_capture",
        "mean_revert",
        "risk_off",
    ]

    def __init__(self, config: Optional[HRLConfig] = None):
        self.config = config or HRLConfig()
        self.step_count = 0
        self.current_decision: Optional[ManagerDecision] = None
        self.manager_value_estimates = {obj: 0.0 for obj in self.OBJECTIVES}
        self.worker_stats = {
            "actions": 0,
            "aligned_actions": 0,
            "reward_total": 0.0,
        }

    def select_manager_decision(self, state: MarketState) -> ManagerDecision:
        """
        Choose strategic objective based on regime features.
        Lightweight heuristic scaffold that can be replaced with a learned policy.
        """
        vol = state.volatility
        spread = state.spread_bps
        inv = state.inventory
        imb = state.imbalance

        if vol > 0.03:
            objective = "risk_off"
            aggression = 0.2
            horizon = 8
        elif abs(inv) > 0.65:
            objective = "inventory_rebalance"
            aggression = 0.7
            horizon = 6
        elif spread > 4.0 and vol < 0.02:
            objective = "market_make"
            aggression = 0.55
            horizon = 12
        elif imb > 0.2:
            objective = "momentum_capture"
            aggression = 0.75
            horizon = 10
        else:
            objective = "mean_revert"
            aggression = 0.45
            horizon = 10

        target_inventory = 0.0
        if objective == "momentum_capture":
            target_inventory = min(max(imb, -1.0), 1.0) * 0.5
        elif objective == "market_make":
            target_inventory = 0.0
        elif objective == "inventory_rebalance":
            target_inventory = 0.0
        elif objective == "risk_off":
            target_inventory = 0.0
        elif objective == "mean_revert":
            target_inventory = -imb * 0.25

        decision = ManagerDecision(
            objective=objective,
            target_inventory=target_inventory,
            aggression=aggression,
            hold_horizon=horizon,
        )
        self.current_decision = decision
        logger.debug("Manager decision: %s", decision)
        return decision

    def _needs_manager_refresh(self) -> bool:
        if self.current_decision is None:
            return True
        return self.step_count % self.config.manager_update_interval == 0

    def select_worker_action(self, state: MarketState) -> WorkerAction:
        """
        Choose low-level execution action conditioned on current manager decision.
        """
        if self._needs_manager_refresh():
            self.select_manager_decision(state)

        decision = self.current_decision
        assert decision is not None

        inv_gap = decision.target_inventory - state.inventory
        size_pct = min(abs(inv_gap) * 0.2 + decision.aggression * 0.05, self.config.max_position_pct)

        if decision.objective == "market_make":
            side = "both"
            order_type = "maker"
            price_offset_bps = max(state.spread_bps * 0.4, 1.0)
        elif decision.objective == "inventory_rebalance":
            side = "buy" if inv_gap > 0 else "sell"
            order_type = "taker"
            price_offset_bps = 0.2
        elif decision.objective == "momentum_capture":
            side = "buy" if state.imbalance >= 0 else "sell"
            order_type = "taker"
            price_offset_bps = 0.5
        elif decision.objective == "mean_revert":
            side = "buy" if state.imbalance < 0 else "sell"
            order_type = "maker"
            price_offset_bps = 1.2
        else:
            side = "sell" if state.inventory > 0 else "buy"
            order_type = "taker"
            size_pct *= 0.5
            price_offset_bps = 0.1

        action = WorkerAction(
            side=side,
            size_pct=max(0.0, size_pct),
            order_type=order_type,
            price_offset_bps=price_offset_bps,
        )
        self.step_count += 1
        self.worker_stats["actions"] += 1
        logger.debug("Worker action: %s", action)
        return action

    def intrinsic_reward(self, state: MarketState, action: WorkerAction) -> float:
        """
        Reward worker for alignment with manager goals.
        """
        if self.current_decision is None:
            return 0.0

        decision = self.current_decision
        inv_gap_before = abs(decision.target_inventory - state.inventory)

        alignment = 0.0
        if decision.objective == "market_make" and action.order_type == "maker":
            alignment += 1.0
        if decision.objective == "inventory_rebalance":
            if (decision.target_inventory - state.inventory) > 0 and action.side == "buy":
                alignment += 1.0
            elif (decision.target_inventory - state.inventory) <= 0 and action.side == "sell":
                alignment += 1.0
        if decision.objective == "momentum_capture":
            if state.imbalance >= 0 and action.side == "buy":
                alignment += 1.0
            elif state.imbalance < 0 and action.side == "sell":
                alignment += 1.0
        if decision.objective == "mean_revert":
            if state.imbalance < 0 and action.side == "buy":
                alignment += 1.0
            elif state.imbalance >= 0 and action.side == "sell":
                alignment += 1.0
        if decision.objective == "risk_off" and action.size_pct <= 0.05:
            alignment += 1.0

        reward = alignment * self.config.intrinsic_reward_scale
        self.worker_stats["aligned_actions"] += int(alignment > 0)
        self.worker_stats["reward_total"] += reward
        return reward

    def shaped_reward(self, pnl: float, state: MarketState, action: WorkerAction) -> float:
        """
        Combined extrinsic + intrinsic reward with inventory/risk penalties.
        """
        intrinsic = self.intrinsic_reward(state, action)
        risk_penalty = state.volatility * self.config.risk_penalty_coef * action.size_pct
        inventory_penalty = abs(state.inventory) * self.config.inventory_penalty_coef
        return pnl + intrinsic - risk_penalty - inventory_penalty

    def get_stats(self) -> Dict[str, Any]:
        actions = max(1, self.worker_stats["actions"])
        return {
            "step_count": self.step_count,
            "current_decision": self.current_decision,
            "alignment_rate": self.worker_stats["aligned_actions"] / actions,
            "intrinsic_reward_total": self.worker_stats["reward_total"],
            "manager_value_estimates": dict(self.manager_value_estimates),
        }
