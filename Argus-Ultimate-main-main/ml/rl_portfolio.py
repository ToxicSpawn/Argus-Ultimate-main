"""
Deep RL Portfolio Manager — uses REINFORCE (policy gradient) to learn
optimal strategy weight allocation based on portfolio state.

Pure numpy implementation. Optional torch for GPU-accelerated forward pass.

State vector:
  - Per-strategy: recent return, volatility, Sharpe (n_strategies * 3)
  - Global: regime encoding (3 dims), portfolio drawdown, avg correlation

Action:
  - Strategy weight allocation via softmax (sums to 1.0)

Reward:
  - Risk-adjusted portfolio return (Sharpe-like)

Usage:
    mgr = RLPortfolioManager(n_strategies=15)
    state = mgr.get_state(strategy_stats, regime, drawdown)
    allocation = mgr.allocate(state)
    mgr.record_reward(portfolio_return)
    mgr.update()  # policy gradient update after episode
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Regime encoding (one-hot style)
_REGIME_MAP = {
    "TRENDING_UP": [1.0, 0.0, 0.0],
    "TRENDING_DOWN": [-1.0, 0.0, 0.0],
    "BREAKOUT": [0.5, 1.0, 0.0],
    "MEAN_REVERT": [0.0, -1.0, 0.0],
    "HIGH_VOL": [0.0, 0.0, 1.0],
    "LOW_VOL": [0.0, 0.0, -1.0],
    "CRISIS": [-1.0, 0.0, 1.0],
    "UNKNOWN": [0.0, 0.0, 0.0],
}

# Default strategy names (when not provided)
_DEFAULT_STRATEGIES = [
    "momentum", "mean_reversion", "breakout", "trend_following",
    "pairs_trading", "volatility_arb", "funding_arb", "market_making",
    "liquidation_cascade", "macro_event", "session_effect",
    "whale_tracking", "sentiment", "statistical_arb", "cross_exchange_arb",
]


class RLPortfolioManager:
    """
    REINFORCE policy gradient agent for dynamic strategy weight allocation.

    Policy network: state → hidden (ReLU) → action logits → softmax → weights
    """

    def __init__(
        self,
        n_strategies: int = 15,
        state_dim: int = 20,
        hidden_dim: int = 32,
        max_weight: float = 0.40,
        strategy_names: Optional[List[str]] = None,
        gamma: float = 0.99,
        lr: float = 0.001,
        baseline_decay: float = 0.95,
        update_every: int = 20,
    ) -> None:
        self.n_strategies = n_strategies
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.max_weight = max_weight
        self.gamma = gamma
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.update_every = update_every

        self.strategy_names = list(strategy_names or _DEFAULT_STRATEGIES[:n_strategies])
        # Pad/trim to match n_strategies
        while len(self.strategy_names) < n_strategies:
            self.strategy_names.append(f"strategy_{len(self.strategy_names)}")
        self.strategy_names = self.strategy_names[:n_strategies]

        # Policy network weights (Xavier init)
        self._W1 = self._xavier(state_dim, hidden_dim)
        self._b1 = np.zeros(hidden_dim)
        self._W2 = self._xavier(hidden_dim, n_strategies)
        self._b2 = np.zeros(n_strategies)

        # Episode buffer for REINFORCE
        self._log_probs: List[float] = []
        self._rewards: List[float] = []
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []

        # Baseline (moving average of returns)
        self._baseline = 0.0

        # History
        self._allocation_history: Deque[dict] = deque(maxlen=500)
        self._episode_count = 0
        self._total_updates = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(
        self,
        strategy_stats: Dict[str, dict],
        regime: str,
        drawdown: float,
    ) -> np.ndarray:
        """
        Build state vector from current portfolio information.

        strategy_stats: {name: {return_1d, volatility, sharpe}}
        regime: current market regime string
        drawdown: current portfolio drawdown (0 to 1)

        Returns state vector of size state_dim.
        """
        state = np.zeros(self.state_dim)
        idx = 0

        # Per-strategy features (return, vol, sharpe) — up to n_strategies * 3
        for sname in self.strategy_names:
            stats = strategy_stats.get(sname, {})
            if idx + 2 < self.state_dim:
                state[idx] = np.clip(float(stats.get("return_1d", 0.0)) * 10.0, -1.0, 1.0)
                state[idx + 1] = np.clip(float(stats.get("volatility", 0.0)) * 20.0, 0.0, 1.0)
                state[idx + 2] = np.clip(float(stats.get("sharpe", 0.0)) / 3.0, -1.0, 1.0)
                idx += 3
            if idx >= self.state_dim - 4:
                break

        # Global features in last 4 slots
        regime_enc = _REGIME_MAP.get(regime, _REGIME_MAP["UNKNOWN"])
        end = self.state_dim
        if end >= 4:
            state[end - 4] = regime_enc[0]
            state[end - 3] = regime_enc[1]
            state[end - 2] = regime_enc[2]
            state[end - 1] = np.clip(-drawdown * 5.0, -1.0, 0.0)

        return state

    def allocate(self, state: np.ndarray) -> Dict[str, float]:
        """
        Compute strategy weight allocation from current state.

        Returns {strategy_name: weight} where weights sum to 1.0.
        """
        state = np.asarray(state, dtype=np.float64).flatten()
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))

        # Forward pass
        h = self._relu(state @ self._W1 + self._b1)
        logits = h @ self._W2 + self._b2

        # Softmax
        probs = self._softmax(logits)

        # Clip to max_weight and renormalize
        probs = np.minimum(probs, self.max_weight)
        probs = probs / probs.sum()

        # Add small exploration noise during training
        noise = np.random.dirichlet(np.ones(self.n_strategies) * 50.0)
        probs_noisy = 0.95 * probs + 0.05 * noise
        probs_noisy = probs_noisy / probs_noisy.sum()

        # Compute log probability for policy gradient
        log_prob = float(np.sum(probs_noisy * np.log(np.maximum(probs, 1e-10))))
        self._log_probs.append(log_prob)
        self._states.append(state.copy())
        self._actions.append(probs_noisy.copy())

        # Build result
        allocation = {
            self.strategy_names[i]: round(float(probs_noisy[i]), 6)
            for i in range(self.n_strategies)
        }

        # Record history
        self._allocation_history.append({
            "allocation": allocation.copy(),
            "episode": self._episode_count,
        })

        return allocation

    def record_reward(self, portfolio_return: float) -> None:
        """Record reward for the last allocation decision."""
        # Risk-adjusted reward: penalize large losses more
        reward = float(portfolio_return)
        if reward < 0:
            reward *= 1.5  # asymmetric penalty for losses

        self._rewards.append(reward)

        # Auto-update if enough steps
        if len(self._rewards) >= self.update_every:
            self.update()

    def update(self) -> Dict[str, Any]:
        """
        REINFORCE policy gradient update.

        Computes discounted returns, subtracts baseline, updates weights.
        Returns update summary.
        """
        if len(self._rewards) < 2:
            return {"status": "insufficient_data", "n_steps": len(self._rewards)}

        # Compute discounted returns (reverse accumulation)
        T = len(self._rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        # Normalize returns
        r_mean = returns.mean()
        r_std = returns.std()
        if r_std > 1e-8:
            returns = (returns - r_mean) / r_std

        # Update baseline
        self._baseline = (
            self.baseline_decay * self._baseline +
            (1.0 - self.baseline_decay) * r_mean
        )

        # Policy gradient: accumulate gradients
        total_grad_W1 = np.zeros_like(self._W1)
        total_grad_b1 = np.zeros_like(self._b1)
        total_grad_W2 = np.zeros_like(self._W2)
        total_grad_b2 = np.zeros_like(self._b2)

        for t in range(min(T, len(self._states))):
            state = self._states[t]
            advantage = returns[t] - self._baseline

            # Forward pass (recompute activations)
            h = self._relu(state @ self._W1 + self._b1)
            logits = h @ self._W2 + self._b2
            probs = self._softmax(logits)

            # Gradient of log-softmax w.r.t. logits
            # For REINFORCE: d/d_theta log(pi) * advantage
            action = self._actions[t] if t < len(self._actions) else probs
            d_logits = (action - probs) * advantage  # (n_strategies,)

            # Backprop through output layer
            total_grad_W2 += np.outer(h, d_logits)
            total_grad_b2 += d_logits

            # Backprop through hidden layer
            d_h = d_logits @ self._W2.T
            d_h = d_h * (h > 0).astype(float)  # ReLU gradient
            total_grad_W1 += np.outer(state, d_h)
            total_grad_b1 += d_h

        # Average gradients
        total_grad_W1 /= T
        total_grad_b1 /= T
        total_grad_W2 /= T
        total_grad_b2 /= T

        # Gradient ascent (maximizing expected return)
        self._W1 += self.lr * total_grad_W1
        self._b1 += self.lr * total_grad_b1
        self._W2 += self.lr * total_grad_W2
        self._b2 += self.lr * total_grad_b2

        # Gradient clipping (prevent explosion)
        for param in (self._W1, self._b1, self._W2, self._b2):
            np.clip(param, -5.0, 5.0, out=param)

        avg_reward = float(np.mean(self._rewards))
        self._episode_count += 1
        self._total_updates += 1

        # Clear episode buffer
        n_steps = len(self._rewards)
        self._log_probs.clear()
        self._rewards.clear()
        self._states.clear()
        self._actions.clear()

        logger.debug(
            "RLPortfolio: update #%d, %d steps, avg_reward=%.6f, baseline=%.6f",
            self._total_updates, n_steps, avg_reward, self._baseline,
        )

        return {
            "status": "updated",
            "episode": self._episode_count,
            "n_steps": n_steps,
            "avg_reward": avg_reward,
            "baseline": self._baseline,
        }

    def get_allocation_history(self) -> list:
        """Return recent allocation decisions and rewards."""
        return list(self._allocation_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()  # numerical stability
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    @staticmethod
    def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_in, fan_out))
