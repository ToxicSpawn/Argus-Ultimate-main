"""Push 66 — ArgusRLEnv: Gymnasium trading environment.

Full Gymnasium-compliant env wiring:
  DataFeed -> FeatureBuilder -> obs -> PPO/TD3/SAC agent ->
  continuous action [-1,+1] -> ExecutionEngine -> PnLTracker
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any

from core.rl.rl_config import RLConfig
from core.rl.rl_feature_builder import FeatureBuilder


class ArgusRLEnv(gym.Env):
    """Argus trading environment for RL agents.

    Observation space: Box(7,) float32
    Action space:      Box(1,) float32  [-1=full short, +1=full long]
    Reward:            Sharpe-shaped with transaction cost + inventory penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, feed: list, config: RLConfig | None = None):
        super().__init__()
        self.feed = feed
        self.cfg = config or RLConfig()
        self._fb = FeatureBuilder()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.cfg.action_low, high=self.cfg.action_high,
            shape=(1,), dtype=np.float32
        )
        self._reset_state()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset_state()
        if self.feed:
            self._fb.update(self.feed[0])
            obs = self._fb.build(
                self.feed[0],
                inventory=self._position,
                pnl_norm=0.0,
            )
        else:
            obs = np.zeros(self.cfg.obs_dim, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert len(self.feed) > 0, "Empty feed"
        bar = self.feed[self._step_idx]
        target_pos = float(np.clip(action[0], -1.0, 1.0))
        delta = target_pos - self._position

        # Transaction cost (bps of notional moved)
        cost = abs(delta) * self._equity * (self.cfg.fee_bps / 10_000.0)

        # Mark-to-market PnL
        price_ret = (bar.close - bar.open) / bar.open if bar.open > 0 else 0.0
        pnl = self._position * price_ret * self._equity
        self._equity += pnl - cost
        self._position = target_pos
        self._step_idx += 1

        # Rolling return for Sharpe reward
        ret = pnl / self.cfg.initial_equity
        self._returns.append(ret)
        if len(self._returns) > self.cfg.sharpe_window:
            self._returns = self._returns[-self.cfg.sharpe_window:]

        reward = self._compute_reward(cost)

        terminated = self._step_idx >= len(self.feed) - 1
        truncated = False
        pnl_norm = (self._equity - self.cfg.initial_equity) / self.cfg.initial_equity

        if not terminated:
            self._fb.update(self.feed[self._step_idx])
            obs = self._fb.build(
                self.feed[self._step_idx],
                inventory=self._position,
                pnl_norm=pnl_norm,
            )
        else:
            obs = np.zeros(self.cfg.obs_dim, dtype=np.float32)

        info = {
            "equity": self._equity,
            "position": self._position,
            "step": self._step_idx,
            "total_return": pnl_norm,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        print(f"[RLEnv] step={self._step_idx} equity={self._equity:.2f} "
              f"pos={self._position:.3f}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._step_idx = 0
        self._equity = self.cfg.initial_equity
        self._position = 0.0
        self._returns: list[float] = []
        self._peak_equity = self.cfg.initial_equity
        self._fb = FeatureBuilder()

    def _compute_reward(self, cost: float) -> float:
        # Sharpe-shaped reward
        if len(self._returns) >= 2:
            mu = float(np.mean(self._returns))
            sigma = float(np.std(self._returns)) + 1e-8
            sharpe_r = mu / sigma
        else:
            sharpe_r = 0.0

        cost_penalty = (cost / self.cfg.initial_equity) * self.cfg.cost_penalty_scale
        inventory_penalty = abs(self._position) * self.cfg.inventory_penalty
        return sharpe_r - cost_penalty - inventory_penalty

    @property
    def current_equity(self) -> float:
        return self._equity

    @property
    def total_return(self) -> float:
        return (self._equity - self.cfg.initial_equity) / self.cfg.initial_equity
