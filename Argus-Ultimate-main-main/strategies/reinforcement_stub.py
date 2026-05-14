"""
Reinforcement Learning Execution Agent — stub framework.

Architecture: PPO (Proximal Policy Optimization) agent that learns
optimal order timing and sizing from market microstructure.

State space:
  - Current position, unrealised P&L
  - Rolling volatility, spread, order book imbalance
  - Time of day, time since last trade
  - Slippage budget remaining

Action space:
  - HOLD / BUY_SMALL / BUY_LARGE / SELL_SMALL / SELL_LARGE

This stub provides the interface and environment structure.
Activate by setting use_rl_agent: true in config and training the model
via scripts/train_rl_agent.py (requires stable-baselines3 + gymnasium).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym  # type: ignore
    from stable_baselines3 import PPO  # type: ignore
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False
    logger.debug("gymnasium/stable-baselines3 not available; RL agent in stub mode")

# Action enum
HOLD = 0
BUY_SMALL = 1
BUY_LARGE = 2
SELL_SMALL = 3
SELL_LARGE = 4

ACTION_NAMES = {HOLD: "HOLD", BUY_SMALL: "BUY_SMALL", BUY_LARGE: "BUY_LARGE",
                SELL_SMALL: "SELL_SMALL", SELL_LARGE: "SELL_LARGE"}
ACTION_SIZE_FACTOR = {HOLD: 0.0, BUY_SMALL: 0.25, BUY_LARGE: 1.0,
                      SELL_SMALL: 0.25, SELL_LARGE: 1.0}


@dataclass
class RLState:
    position_usd: float
    unrealised_pnl: float
    volatility_1h: float
    spread_bps: float
    ob_imbalance: float        # [-1, +1] bid vs ask depth
    time_of_day_sin: float     # cyclical encoding
    time_of_day_cos: float
    slippage_budget_remaining: float  # bps
    bars_since_last_trade: int

    def to_array(self) -> np.ndarray:
        return np.array([
            self.position_usd / 10000,
            self.unrealised_pnl / 1000,
            self.volatility_1h,
            self.spread_bps / 100,
            self.ob_imbalance,
            self.time_of_day_sin,
            self.time_of_day_cos,
            self.slippage_budget_remaining / 20,
            min(self.bars_since_last_trade / 100, 1.0),
        ], dtype=np.float32)


@dataclass
class RLDecision:
    action: int
    action_name: str
    size_factor: float   # 0 to 1.0
    confidence: float    # 0-1 from policy entropy
    from_model: bool     # False = rule-based fallback


def _rule_based_action(state: RLState) -> RLDecision:
    """Simple rule-based fallback when no trained model available."""
    # High volatility → hold
    if state.volatility_1h > 0.80:
        return RLDecision(HOLD, "HOLD", 0.0, 0.5, False)
    # OB strongly bullish + position flat → small buy
    if state.ob_imbalance > 0.4 and abs(state.position_usd) < 100:
        return RLDecision(BUY_SMALL, "BUY_SMALL", 0.25, 0.4, False)
    # OB strongly bearish + long position → small sell
    if state.ob_imbalance < -0.4 and state.position_usd > 100:
        return RLDecision(SELL_SMALL, "SELL_SMALL", 0.25, 0.4, False)
    return RLDecision(HOLD, "HOLD", 0.0, 0.6, False)


class RLExecutionAgent:
    """
    Reinforcement learning execution agent.

    When no trained model is available, uses rule-based fallback.
    Train via: scripts/train_rl_agent.py

    Usage::

        agent = RLExecutionAgent(model_path="models/rl_agent.zip")
        state = RLState(...)
        decision = agent.decide(state)
        if decision.action in (BUY_SMALL, BUY_LARGE):
            # Place order scaled by decision.size_factor
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_position_usd: float = 1000.0,
        slippage_budget_bps: float = 20.0,
    ) -> None:
        self._model: Optional[Any] = None
        self._max_pos = max_position_usd
        self._slippage_budget = slippage_budget_bps
        self._episode_slippage = 0.0

        if model_path and _RL_AVAILABLE:
            try:
                self._model = PPO.load(model_path)
                logger.info("RL agent loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Could not load RL model from %s: %s", model_path, exc)

    def decide(self, state: RLState) -> RLDecision:
        """Select action given current market state."""
        # Budget check — force HOLD if slippage exhausted
        if state.slippage_budget_remaining <= 0:
            return RLDecision(HOLD, "HOLD", 0.0, 1.0, False)

        if self._model is not None:
            try:
                obs = state.to_array().reshape(1, -1)
                action_raw, _ = self._model.predict(obs, deterministic=True)
                action = int(action_raw)
                # Estimate confidence from value function (stub: fixed)
                return RLDecision(
                    action=action,
                    action_name=ACTION_NAMES.get(action, "HOLD"),
                    size_factor=ACTION_SIZE_FACTOR.get(action, 0.0),
                    confidence=0.75,
                    from_model=True,
                )
            except Exception as exc:
                logger.warning("RL model inference failed: %s", exc)

        return _rule_based_action(state)

    def record_slippage(self, slippage_bps: float) -> None:
        """Call after each fill to track slippage consumption."""
        self._episode_slippage += abs(slippage_bps)

    def reset_episode(self) -> None:
        """Reset per-session slippage tracking."""
        self._episode_slippage = 0.0

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def rl_available(self) -> bool:
        return _RL_AVAILABLE

    @staticmethod
    def state_dim() -> int:
        return 9

    @staticmethod
    def action_dim() -> int:
        return 5


# Training: python scripts/train_rl_agent.py --train
# Evaluate: python scripts/train_rl_agent.py --evaluate
# Model saved to: models/rl_execution_agent_ppo.zip
