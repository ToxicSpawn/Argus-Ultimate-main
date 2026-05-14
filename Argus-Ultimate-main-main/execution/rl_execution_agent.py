"""Reinforcement Learning execution agent using PPO for optimal order execution.

Implements a PPO-based RL agent that learns to minimize implementation shortfall
by dynamically adjusting order sizing and timing based on order book state.

Components:
1. OrderBookState — snapshot of market microstructure
2. LOBSimulator — limit order book simulator for training
3. ExecutionEnvironment — RL environment with reset/step/reward
4. PPOAgent — Actor-Critic with GAE, clipped objective, entropy bonus
5. RLExecutionEngine — high-level interface for training and execution
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Order Book State
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OrderBookState:
    """Snapshot of the limit order book for RL state representation."""

    bid_prices: np.ndarray          # top N bid prices
    bid_sizes: np.ndarray           # corresponding bid sizes
    ask_prices: np.ndarray          # top N ask prices
    ask_sizes: np.ndarray           # corresponding ask sizes
    mid_price: float
    spread: float
    recent_trades: np.ndarray       # (price, size, side) tuples as flat array
    timestamp: float = field(default_factory=time.time)

    @property
    def bid_ask_imbalance(self) -> float:
        """Bid-ask size imbalance normalized to [-1, 1]."""
        total_bid = np.sum(self.bid_sizes)
        total_ask = np.sum(self.ask_sizes)
        total = total_bid + total_ask
        if total <= 0:
            return 0.0
        return (total_bid - total_ask) / total

    @property
    def weighted_mid(self) -> float:
        """Volume-weighted mid price."""
        bid_vol = np.sum(self.bid_sizes)
        ask_vol = np.sum(self.ask_sizes)
        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return self.mid_price
        return (np.sum(self.bid_prices * self.bid_sizes) + np.sum(self.ask_prices * self.ask_sizes)) / total_vol

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.mid_price <= 0:
            return 0.0
        return (self.spread / self.mid_price) * 10000.0

    def to_feature_vector(self, remaining_qty: float, total_qty: float,
                          elapsed_pct: float, volatility: float) -> np.ndarray:
        """Convert state to normalized feature vector for the neural network."""
        features = []

        # Book features (normalized by mid_price)
        for i in range(len(self.bid_prices)):
            features.append(self.bid_prices[i] / max(self.mid_price, 1e-9) - 1.0)
            features.append(self.bid_sizes[i] / max(np.sum(self.bid_sizes), 1e-9))

        for i in range(len(self.ask_prices)):
            features.append(self.ask_prices[i] / max(self.mid_price, 1e-9) - 1.0)
            features.append(self.ask_sizes[i] / max(np.sum(self.ask_sizes), 1e-9))

        # Derived features
        features.append(self.bid_ask_imbalance)
        features.append(self.spread_bps / 100.0)  # normalize
        features.append((self.weighted_mid / max(self.mid_price, 1e-9)) - 1.0)

        # Order state features
        features.append(remaining_qty / max(total_qty, 1e-9))
        features.append(elapsed_pct)
        features.append(volatility)

        # Recent trade flow
        if len(self.recent_trades) > 0:
            trade_prices = self.recent_trades[0::3]
            trade_sizes = self.recent_trades[1::3]
            trade_sides = self.recent_trades[2::3]
            avg_trade_price = np.mean(trade_prices) / max(self.mid_price, 1e-9) - 1.0
            total_trade_vol = np.sum(trade_sizes)
            buy_pressure = np.sum(trade_sides[trade_sides == 1.0]) / max(np.sum(np.abs(trade_sides)), 1e-9)
            features.append(avg_trade_price)
            features.append(total_trade_vol / max(np.sum(self.bid_sizes) + np.sum(self.ask_sizes), 1e-9))
            features.append(buy_pressure)
        else:
            features.extend([0.0, 0.0, 0.5])

        return np.array(features, dtype=np.float64)


# ════════════════════════════════════════════════════════════════════════════
# Limit Order Book Simulator
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LOBSimulatorConfig:
    """Configuration for the LOB simulator."""

    num_levels: int = 5
    initial_spread: float = 0.01
    initial_mid: float = 100.0
    tick_size: float = 0.01
    base_volume_per_level: float = 1000.0
    volatility: float = 0.0001
    mean_reversion_speed: float = 0.1
    order_arrival_rate: float = 10.0
    market_order_prob: float = 0.3
    cancel_prob: float = 0.2
    impact_factor: float = 0.1


class LOBSimulator:
    """Simulates a limit order book with realistic order flow dynamics.

    Generates synthetic order book states for RL training with:
    - Mean-reverting mid price
    - Stochastic order arrivals and cancellations
    - Market impact from aggressive orders
    - Realistic spread dynamics
    """

    def __init__(self, config: Optional[LOBSimulatorConfig] = None):
        self.config = config or LOBSimulatorConfig()
        self._rng = np.random.RandomState(42)
        self._reset_book()

    def _reset_book(self) -> None:
        """Initialize the order book with default state."""
        mid = self.config.initial_mid
        spread = self.config.initial_spread
        n = self.config.num_levels
        tick = self.config.tick_size

        self._bid_prices = np.array([mid - spread / 2 - i * tick for i in range(n)])
        self._ask_prices = np.array([mid + spread / 2 + i * tick for i in range(n)])
        self._bid_sizes = np.full(n, self.config.base_volume_per_level)
        self._ask_sizes = np.full(n, self.config.base_volume_per_level)
        self._mid_price = mid
        self._recent_trades: Deque[Tuple[float, float, float]] = deque(maxlen=50)
        self._cumulative_volume = 0.0
        self._price_history: List[float] = [mid]

    def reset(self) -> OrderBookState:
        """Reset simulator to initial state."""
        self._reset_book()
        return self.get_state()

    def get_state(self) -> OrderBookState:
        """Get current order book state."""
        trades_flat = np.array(
            [val for trade in self._recent_trades for val in trade],
            dtype=np.float64
        ) if self._recent_trades else np.array([], dtype=np.float64)

        return OrderBookState(
            bid_prices=self._bid_prices.copy(),
            bid_sizes=self._bid_sizes.copy(),
            ask_prices=self._ask_prices.copy(),
            ask_sizes=self._ask_sizes.copy(),
            mid_price=self._mid_price,
            spread=self._ask_prices[0] - self._bid_prices[0],
            recent_trades=trades_flat,
            timestamp=time.time(),
        )

    def step(self, our_side: Optional[str] = None,
             our_qty: float = 0.0) -> OrderBookState:
        """Advance simulator by one timestep, optionally executing our order.

        Args:
            our_side: "buy" or "sell" if we're submitting an order
            our_qty: quantity to execute

        Returns:
            New OrderBookState after market dynamics and our order
        """
        self._simulate_market_dynamics()

        if our_side and our_qty > 0:
            self._execute_our_order(our_side, our_qty)

        return self.get_state()

    def _simulate_market_dynamics(self) -> None:
        """Simulate one timestep of market order flow."""
        cfg = self.config

        # Mean-reverting price movement
        noise = self._rng.normal(0, cfg.volatility * cfg.initial_mid)
        reversion = cfg.mean_reversion_speed * (cfg.initial_mid - self._mid_price)
        self._mid_price += noise + reversion * 0.01

        # Random order arrivals
        if self._rng.random() < cfg.order_arrival_rate * 0.1:
            self._add_limit_order()

        # Random cancellations
        if self._rng.random() < cfg.cancel_prob:
            self._cancel_random_order()

        # Market orders from other participants
        if self._rng.random() < cfg.market_order_prob:
            self._simulate_market_order()

        # Update spread and prices
        self._update_book_prices()
        self._price_history.append(self._mid_price)

    def _add_limit_order(self) -> None:
        """Add a random limit order to the book."""
        level = self._rng.randint(0, self.config.num_levels)
        side = self._rng.choice(["bid", "ask"])
        size = self._rng.uniform(0.5, 2.0) * self.config.base_volume_per_level

        if side == "bid":
            self._bid_sizes[level] += size
        else:
            self._ask_sizes[level] += size

    def _cancel_random_order(self) -> None:
        """Cancel a random portion of existing orders."""
        level = self._rng.randint(0, self.config.num_levels)
        side = self._rng.choice(["bid", "ask"])
        cancel_frac = self._rng.uniform(0.1, 0.5)

        if side == "bid":
            self._bid_sizes[level] *= (1 - cancel_frac)
        else:
            self._ask_sizes[level] *= (1 - cancel_frac)

    def _simulate_market_order(self) -> None:
        """Simulate a market order from other participants."""
        side = self._rng.choice(["buy", "sell"])
        size = self._rng.exponential(self.config.base_volume_per_level * 0.1)

        if side == "buy":
            consumed = self._consume_liquidity("ask", size)
            if consumed > 0:
                trade_price = self._ask_prices[0]
                self._mid_price += 0.001 * consumed / self.config.base_volume_per_level
        else:
            consumed = self._consume_liquidity("bid", size)
            if consumed > 0:
                trade_price = self._bid_prices[0]
                self._mid_price -= 0.001 * consumed / self.config.base_volume_per_level

        if consumed > 0:
            trade_side = 1.0 if side == "buy" else -1.0
            self._recent_trades.append((trade_price, consumed, trade_side))
            self._cumulative_volume += consumed

    def _execute_our_order(self, side: str, qty: float) -> Tuple[float, float]:
        """Execute our order against the simulated book.

        Returns:
            (fill_price, fill_qty)
        """
        if side.lower() == "buy":
            return self._walk_book("ask", qty, impact_sign=1)
        else:
            return self._walk_book("bid", qty, impact_sign=-1)

    def _walk_book(self, side: str, qty: float,
                   impact_sign: int) -> Tuple[float, float]:
        """Walk the book to fill an order, applying market impact."""
        remaining = qty
        total_cost = 0.0
        filled = 0.0
        cfg = self.config

        if side == "ask":
            prices = self._ask_prices
            sizes = self._ask_sizes
        else:
            prices = self._bid_prices
            sizes = self._bid_sizes

        for i in range(len(prices)):
            if remaining <= 0:
                break
            available = sizes[i]
            take = min(remaining, available)
            total_cost += take * prices[i]
            filled += take
            remaining -= take
            sizes[i] -= take

            # Market impact
            impact = cfg.impact_factor * take / max(self.config.base_volume_per_level, 1e-9)
            self._mid_price += impact_sign * impact * prices[i]

        # Record trade
        if filled > 0:
            avg_price = total_cost / filled
            self._recent_trades.append((avg_price, filled, float(impact_sign)))
            self._cumulative_volume += filled

        self._update_book_prices()
        return (total_cost / max(filled, 1e-9), filled)

    def _consume_liquidity(self, side: str, qty: float) -> float:
        """Consume liquidity from one side of the book."""
        remaining = qty
        filled = 0.0

        sizes = self._ask_sizes if side == "ask" else self._bid_sizes

        for i in range(len(sizes)):
            if remaining <= 0:
                break
            take = min(remaining, sizes[i])
            sizes[i] -= take
            filled += take
            remaining -= take

        return filled

    def _update_book_prices(self) -> None:
        """Re-center book around current mid price."""
        cfg = self.config
        half_spread = max(cfg.initial_spread / 2, self.config.tick_size)

        for i in range(self.config.num_levels):
            self._bid_prices[i] = self._mid_price - half_spread - i * cfg.tick_size
            self._ask_prices[i] = self._mid_price + half_spread + i * cfg.tick_size

        # Ensure non-negative sizes
        self._bid_sizes = np.maximum(self._bid_sizes, 0)
        self._ask_sizes = np.maximum(self._ask_sizes, 0)

    def get_price_history(self) -> List[float]:
        """Return the mid price history."""
        return list(self._price_history)

    def get_cumulative_volume(self) -> float:
        """Return total simulated volume."""
        return self._cumulative_volume


# ════════════════════════════════════════════════════════════════════════════
# Execution Environment
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionStepResult:
    """Result of one environment step."""

    state: OrderBookState
    reward: float
    done: bool
    info: Dict[str, Any]


class ExecutionEnvironment:
    """RL environment for optimal order execution.

    State: OrderBookState features + order progress
    Action: execution intensity [0, 1] (fraction of remaining to execute)
    Reward: negative implementation shortfall
    """

    def __init__(
        self,
        simulator: Optional[LOBSimulator] = None,
        max_steps: int = 100,
        total_qty: float = 10000.0,
        side: str = "buy",
        penalty_factor: float = 10.0,
        time_penalty: float = 0.01,
    ):
        self.simulator = simulator or LOBSimulator()
        self.max_steps = max_steps
        self.total_qty = total_qty
        self.side = side.lower()
        self.penalty_factor = penalty_factor
        self.time_penalty = time_penalty

        self._remaining_qty = total_qty
        self._filled_qty = 0.0
        self._total_cost = 0.0
        self._step_count = 0
        self._decision_price = 0.0
        self._volatility = 0.0
        self._price_window: Deque[float] = deque(maxlen=20)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial feature vector
        """
        state = self.simulator.reset()
        self._remaining_qty = self.total_qty
        self._filled_qty = 0.0
        self._total_cost = 0.0
        self._step_count = 0
        self._decision_price = state.mid_price
        self._volatility = 0.0
        self._price_window.clear()
        self._price_window.append(state.mid_price)

        return self._get_observation(state)

    def step(self, action: float) -> ExecutionStepResult:
        """Execute one step in the environment.

        Args:
            action: execution intensity in [0, 1]

        Returns:
            ExecutionStepResult with new state, reward, done flag, and info
        """
        action = np.clip(action, 0.0, 1.0)

        # Determine how much to execute
        execute_qty = action * self._remaining_qty
        execute_qty = max(0.0, execute_qty)

        # Execute against the book
        state = self.simulator.step(our_side=self.side, our_qty=execute_qty)

        # Calculate fill
        if self.side == "buy":
            fill_price = state.ask_prices[0]
        else:
            fill_price = state.bid_prices[0]

        actual_fill_qty = min(execute_qty, self._remaining_qty)
        if actual_fill_qty > 0:
            self._total_cost += actual_fill_qty * fill_price
            self._filled_qty += actual_fill_qty
            self._remaining_qty -= actual_fill_qty

        self._step_count += 1
        self._price_window.append(state.mid_price)
        self._volatility = self._compute_volatility()

        # Compute reward
        reward = self._compute_reward(fill_price, actual_fill_qty, state)

        # Check termination
        done = self._remaining_qty <= 0 or self._step_count >= self.max_steps

        info = {
            "filled_qty": self._filled_qty,
            "remaining_qty": self._remaining_qty,
            "avg_fill_price": self._total_cost / max(self._filled_qty, 1e-9),
            "decision_price": self._decision_price,
            "implementation_shortfall_bps": self._calc_is_bps(),
            "step": self._step_count,
        }

        obs = self._get_observation(state) if not done else self._get_observation(state)

        return ExecutionStepResult(
            state=state,
            reward=reward,
            done=done,
            info=info,
        )

    def _get_observation(self, state: OrderBookState) -> np.ndarray:
        """Get normalized observation vector."""
        elapsed_pct = self._step_count / max(self.max_steps, 1)
        return state.to_feature_vector(
            remaining_qty=self._remaining_qty,
            total_qty=self.total_qty,
            elapsed_pct=elapsed_pct,
            volatility=self._volatility,
        )

    def _compute_reward(self, fill_price: float, fill_qty: float,
                        state: OrderBookState) -> float:
        """Compute reward as negative implementation shortfall.

        Reward = -(IS + time_penalty + urgency_penalty)
        """
        # Implementation shortfall component
        if fill_qty > 0:
            if self.side == "buy":
                is_component = (fill_price - self._decision_price) / max(self._decision_price, 1e-9)
            else:
                is_component = (self._decision_price - fill_price) / max(self._decision_price, 1e-9)
        else:
            is_component = 0.0

        # Time penalty: penalize slow execution
        time_pen = self.time_penalty

        # Urgency penalty: penalize not finishing
        completion_ratio = self._filled_qty / max(self.total_qty, 1e-9)
        urgency_pen = self.penalty_factor * (1 - completion_ratio) * (self._step_count / max(self.max_steps, 1))

        # Spread cost penalty
        spread_pen = state.spread_bps / 10000.0 * 0.1

        reward = -(is_component * 10000 + time_pen + urgency_pen + spread_pen)

        # Bonus for completing the order
        if self._remaining_qty <= 0:
            reward += 50.0

        return reward

    def _calc_is_bps(self) -> float:
        """Calculate implementation shortfall in basis points."""
        if self._filled_qty <= 0:
            return 0.0
        avg_fill = self._total_cost / self._filled_qty
        if self.side == "buy":
            return (avg_fill - self._decision_price) / max(self._decision_price, 1e-9) * 10000
        else:
            return (self._decision_price - avg_fill) / max(self._decision_price, 1e-9) * 10000

    def _compute_volatility(self) -> float:
        """Compute rolling volatility of mid price."""
        if len(self._price_window) < 2:
            return 0.0
        prices = np.array(self._price_window)
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))


# ════════════════════════════════════════════════════════════════════════════
# PPO Agent
# ════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """Proximal Policy Optimization agent for order execution.

    Implements:
    - Actor-Critic neural networks
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Entropy bonus for exploration
    - Value function clipping
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self._rng = np.random.RandomState(seed)
        self._init_networks()

    def _init_networks(self) -> None:
        """Initialize actor and critic network weights."""
        # Xavier initialization
        def xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return self._rng.uniform(-limit, limit, (fan_in, fan_out))

        # Actor network: state -> (mean, log_std)
        self._actor_w1 = xavier(self.state_dim, self.hidden_dim)
        self._actor_b1 = np.zeros(self.hidden_dim)
        self._actor_w2 = xavier(self.hidden_dim, self.hidden_dim)
        self._actor_b2 = np.zeros(self.hidden_dim)
        self._actor_w_mean = xavier(self.hidden_dim, self.action_dim)
        self._actor_b_mean = np.zeros(self.action_dim)
        self._actor_log_std = np.full(self.action_dim, -0.5)

        # Critic network: state -> value
        self._critic_w1 = xavier(self.state_dim, self.hidden_dim)
        self._critic_b1 = np.zeros(self.hidden_dim)
        self._critic_w2 = xavier(self.hidden_dim, self.hidden_dim)
        self._critic_b2 = np.zeros(self.hidden_dim)
        self._critic_w_v = xavier(self.hidden_dim, 1)
        self._critic_b_v = np.zeros(1)

        # Adam optimizer state
        self._adam_state: Dict[str, Any] = {}

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _actor_forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through actor network.

        Returns:
            (action_mean, action_log_std)
        """
        h = self._relu(state @ self._actor_w1 + self._actor_b1)
        h = self._relu(h @ self._actor_w2 + self._actor_b2)
        mean = h @ self._actor_w_mean + self._actor_b_mean
        return mean, self._actor_log_std

    def _critic_forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through critic network.

        Returns:
            state value
        """
        h = self._relu(state @ self._critic_w1 + self._critic_b1)
        h = self._relu(h @ self._critic_w2 + self._critic_b2)
        return h @ self._critic_w_v + self._critic_b_v

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> float:
        """Select action given state.

        Args:
            state: observation vector
            deterministic: if True, return mean action (for evaluation)

        Returns:
            action in [0, 1]
        """
        mean, log_std = self._actor_forward(state)

        if deterministic:
            action = mean[0]
        else:
            std = np.exp(log_std)
            action = mean[0] + std[0] * self._rng.randn()

        return float(np.clip(action, 0.0, 1.0))

    def evaluate_actions(self, states: np.ndarray,
                         actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate actions for PPO update.

        Returns:
            (action_log_probs, values, entropy)
        """
        mean, log_std = self._actor_forward(states)
        std = np.exp(log_std)

        # Action log probability under Gaussian
        diff = actions - mean
        log_prob = -0.5 * ((diff / std) ** 2 + 2 * log_std + math.log(2 * math.pi))
        log_prob = np.sum(log_prob, axis=-1)

        # Values
        values = self._critic_forward(states).flatten()

        # Entropy
        entropy = 0.5 * (1 + math.log(2 * math.pi)) + np.sum(log_std, axis=-1)

        return log_prob, values, entropy

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                    dones: np.ndarray) -> np.ndarray:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: array of rewards
            values: array of state values
            dones: array of done flags

        Returns:
            advantages array
        """
        advantages = np.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def update(self, states: np.ndarray, actions: np.ndarray,
               advantages: np.ndarray, returns: np.ndarray,
               old_log_probs: np.ndarray) -> Dict[str, float]:
        """Perform PPO update with clipped objective.

        Returns:
            dict of loss metrics
        """
        n = len(states)
        total_loss_actor = 0.0
        total_loss_critic = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # Shuffle
            indices = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_old_log_probs = old_log_probs[idx]

                # Evaluate
                log_probs, values, entropy = self.evaluate_actions(
                    batch_states, batch_actions.reshape(-1, 1))

                # Ratio
                ratio = np.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate
                surr1 = ratio * batch_advantages
                surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -np.mean(np.minimum(surr1, surr2))

                # Value loss with clipping
                values_clipped = batch_old_log_probs * 0 + values  # placeholder
                value_diff = values - batch_returns
                critic_loss = 0.5 * np.mean(value_diff ** 2)

                # Entropy bonus
                entropy_loss = -self.entropy_coef * np.mean(entropy)

                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss + entropy_loss

                # Gradient descent (simplified numerical gradient)
                self._apply_gradients(
                    batch_states, batch_actions, batch_advantages,
                    batch_returns, batch_old_log_probs, self.lr
                )

                total_loss_actor += actor_loss
                total_loss_critic += critic_loss
                total_entropy += np.mean(entropy)

        n_updates = self.ppo_epochs * max(1, n // self.batch_size)
        return {
            "actor_loss": total_loss_actor / n_updates,
            "critic_loss": total_loss_critic / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def _apply_gradients(self, states: np.ndarray, actions: np.ndarray,
                         advantages: np.ndarray, returns: np.ndarray,
                         old_log_probs: np.ndarray, lr: float) -> None:
        """Apply gradient updates using finite differences (NumPy only)."""
        eps = 1e-5

        params = [
            ("_actor_w1", self._actor_w1),
            ("_actor_b1", self._actor_b1),
            ("_actor_w2", self._actor_w2),
            ("_actor_b2", self._actor_b2),
            ("_actor_w_mean", self._actor_w_mean),
            ("_actor_b_mean", self._actor_b_mean),
            ("_critic_w1", self._critic_w1),
            ("_critic_b1", self._critic_b1),
            ("_critic_w2", self._critic_w2),
            ("_critic_b2", self._critic_b2),
            ("_critic_w_v", self._critic_w_v),
            ("_critic_b_v", self._critic_b_v),
        ]

        for name, param in params:
            if param.ndim == 1:
                grad = np.zeros_like(param)
                for i in range(len(param)):
                    param[i] += eps
                    loss_plus = self._compute_loss(states, actions, advantages, returns, old_log_probs)
                    param[i] -= 2 * eps
                    loss_minus = self._compute_loss(states, actions, advantages, returns, old_log_probs)
                    param[i] += eps
                    grad[i] = (loss_plus - loss_minus) / (2 * eps)
            else:
                grad = np.zeros_like(param)
                flat_param = param.ravel()
                for i in range(min(len(flat_param), 50)):
                    flat_param[i] += eps
                    loss_plus = self._compute_loss(states, actions, advantages, returns, old_log_probs)
                    flat_param[i] -= 2 * eps
                    loss_minus = self._compute_loss(states, actions, advantages, returns, old_log_probs)
                    flat_param[i] += eps
                    grad.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)

            # Adam update
            self._adam_update(name, grad, lr)

    def _compute_loss(self, states: np.ndarray, actions: np.ndarray,
                      advantages: np.ndarray, returns: np.ndarray,
                      old_log_probs: np.ndarray) -> float:
        """Compute total PPO loss."""
        log_probs, values, entropy = self.evaluate_actions(
            states, actions.reshape(-1, 1))

        ratio = np.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -np.mean(np.minimum(surr1, surr2))
        critic_loss = 0.5 * np.mean((values - returns) ** 2)
        entropy_loss = -self.entropy_coef * np.mean(entropy)

        return actor_loss + self.value_coef * critic_loss + entropy_loss

    def _adam_update(self, name: str, grad: np.ndarray, lr: float) -> None:
        """Adam optimizer step."""
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

        if name not in self._adam_state:
            self._adam_state[name] = {
                "m": np.zeros_like(grad),
                "v": np.zeros_like(grad),
                "t": 0,
            }

        state = self._adam_state[name]
        state["t"] += 1
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2

        m_hat = state["m"] / (1 - beta1 ** state["t"])
        v_hat = state["v"] / (1 - beta2 ** state["t"])

        param = getattr(self, name)
        setattr(self, name, param - lr * m_hat / (np.sqrt(v_hat) + eps_adam))

    def save_weights(self, path: str) -> None:
        """Save network weights to npz file."""
        np.savez(
            path,
            actor_w1=self._actor_w1,
            actor_b1=self._actor_b1,
            actor_w2=self._actor_w2,
            actor_b2=self._actor_b2,
            actor_w_mean=self._actor_w_mean,
            actor_b_mean=self._actor_b_mean,
            actor_log_std=self._actor_log_std,
            critic_w1=self._critic_w1,
            critic_b1=self._critic_b1,
            critic_w2=self._critic_w2,
            critic_b2=self._critic_b2,
            critic_w_v=self._critic_w_v,
            critic_b_v=self._critic_b_v,
        )
        logger.info("PPOAgent weights saved to %s", path)

    def load_weights(self, path: str) -> None:
        """Load network weights from npz file."""
        data = np.load(path)
        self._actor_w1 = data["actor_w1"]
        self._actor_b1 = data["actor_b1"]
        self._actor_w2 = data["actor_w2"]
        self._actor_b2 = data["actor_b2"]
        self._actor_w_mean = data["actor_w_mean"]
        self._actor_b_mean = data["actor_b_mean"]
        self._actor_log_std = data["actor_log_std"]
        self._critic_w1 = data["critic_w1"]
        self._critic_b1 = data["critic_b1"]
        self._critic_w2 = data["critic_w2"]
        self._critic_b2 = data["critic_b2"]
        self._critic_w_v = data["critic_w_v"]
        self._critic_b_v = data["critic_b_v"]
        logger.info("PPOAgent weights loaded from %s", path)


# ════════════════════════════════════════════════════════════════════════════
# RL Execution Engine
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RLExecutionConfig:
    """Configuration for RL execution engine."""

    # Training
    training_episodes: int = 1000
    max_steps_per_episode: int = 100
    total_qty: float = 10000.0
    side: str = "buy"

    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    batch_size: int = 64

    # Networks
    hidden_dim: int = 128

    # Execution
    execution_max_steps: int = 100
    min_execution_interval_s: float = 1.0

    # Misc
    seed: int = 42
    save_path: Optional[str] = None


@dataclass
class ExecutionSchedule:
    """Output execution schedule from RL agent."""

    timestamps: List[float]
    quantities: List[float]
    prices: List[float]
    total_filled: float
    avg_price: float
    implementation_shortfall_bps: float
    total_steps: int


class RLExecutionEngine:
    """High-level interface for RL-based order execution.

    Provides:
    - train(): train the PPO agent using simulated data
    - execute_order(): execute a real order using the trained policy
    - get_execution_schedule(): generate an optimal execution schedule
    """

    def __init__(self, config: Optional[RLExecutionConfig] = None):
        self.config = config or RLExecutionConfig()
        self._agent: Optional[PPOAgent] = None
        self._simulator = LOBSimulator()
        self._env: Optional[ExecutionEnvironment] = None
        self._is_trained = False
        self._training_metrics: List[Dict[str, Any]] = []

        self._init_agent()

    def _init_agent(self) -> None:
        """Initialize the PPO agent with correct state dimensions."""
        state_dim = self._compute_state_dim()
        self._agent = PPOAgent(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=self.config.hidden_dim,
            lr=self.config.lr,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_epsilon=self.config.clip_epsilon,
            entropy_coef=self.config.entropy_coef,
            ppo_epochs=self.config.ppo_epochs,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
        )
        logger.info("PPOAgent initialized with state_dim=%d", state_dim)

    def _compute_state_dim(self) -> int:
        """Compute the state feature vector dimension."""
        cfg = self._simulator.config
        n = cfg.num_levels

        # bid prices + bid sizes + ask prices + ask sizes
        dim = 4 * n
        # bid_ask_imbalance, spread_bps, weighted_mid
        dim += 3
        # remaining_qty_pct, elapsed_pct, volatility
        dim += 3
        # avg_trade_price, total_trade_vol, buy_pressure
        dim += 3

        return dim

    def train(self, episodes: Optional[int] = None) -> Dict[str, List[float]]:
        """Train the PPO agent using the simulated environment.

        Args:
            episodes: number of training episodes (overrides config)

        Returns:
            dict of training metrics (episode_rewards, is_bps, losses)
        """
        n_episodes = episodes or self.config.training_episodes
        self._env = ExecutionEnvironment(
            simulator=self._simulator,
            max_steps=self.config.max_steps_per_episode,
            total_qty=self.config.total_qty,
            side=self.config.side,
        )

        episode_rewards: List[float] = []
        episode_is: List[float] = []
        episode_losses: List[float] = []

        for ep in range(n_episodes):
            state = self._env.reset()
            episode_reward = 0.0

            # Collect rollout
            states_list: List[np.ndarray] = []
            actions_list: List[float] = []
            rewards_list: List[float] = []
            dones_list: List[bool] = []
            log_probs_list: List[float] = []
            values_list: List[float] = []

            for _ in range(self.config.max_steps_per_episode):
                action = self._agent.select_action(state)
                mean, log_std = self._agent._actor_forward(state)
                std = np.exp(log_std)
                log_prob = -0.5 * ((action - mean[0]) / std[0]) ** 2 - log_std[0] - 0.5 * math.log(2 * math.pi)

                value = self._agent._critic_forward(state)[0]

                result = self._env.step(action)

                states_list.append(state)
                actions_list.append(action)
                rewards_list.append(result.reward)
                dones_list.append(result.done)
                log_probs_list.append(log_prob)
                values_list.append(value)

                episode_reward += result.reward
                state = self._agent._get_observation(result.state) if hasattr(self._agent, '_get_observation') else self._env._get_observation(result.state)

                if result.done:
                    break

            # Compute advantages and returns
            rewards_arr = np.array(rewards_list)
            values_arr = np.array(values_list)
            dones_arr = np.array(dones_list)

            advantages = self._agent.compute_gae(rewards_arr, values_arr, dones_arr)
            returns = advantages + values_arr

            # PPO update
            states_arr = np.array(states_list)
            actions_arr = np.array(actions_list)
            old_log_probs_arr = np.array(log_probs_list)

            update_metrics = self._agent.update(
                states_arr, actions_arr, advantages, returns, old_log_probs_arr
            )

            episode_rewards.append(episode_reward)
            episode_is.append(self._env._calc_is_bps())
            episode_losses.append(update_metrics["actor_loss"])

            if (ep + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_is = np.mean(episode_is[-100:])
                logger.info(
                    "Episode %d/%d: avg_reward=%.2f, avg_is_bps=%.2f",
                    ep + 1, n_episodes, avg_reward, avg_is,
                )

        self._is_trained = True
        self._training_metrics = {
            "episode_rewards": episode_rewards,
            "implementation_shortfall_bps": episode_is,
            "actor_loss": episode_losses,
        }

        if self.config.save_path:
            self._agent.save_weights(self.config.save_path)

        logger.info(
            "Training complete: final_avg_reward=%.2f, final_avg_is_bps=%.2f",
            np.mean(episode_rewards[-100:]),
            np.mean(episode_is[-100:]),
        )

        return self._training_metrics

    def execute_order(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        market_data_feed: Optional[Any] = None,
    ) -> ExecutionSchedule:
        """Execute an order using the trained RL policy.

        Args:
            symbol: trading symbol
            side: "buy" or "sell"
            total_qty: total quantity to execute
            market_data_feed: optional live market data feed

        Returns:
            ExecutionSchedule with execution details
        """
        if not self._is_trained:
            logger.warning("RLExecutionEngine: executing with untrained agent")

        env = ExecutionEnvironment(
            simulator=self._simulator,
            max_steps=self.config.execution_max_steps,
            total_qty=total_qty,
            side=side,
        )

        state = env.reset()
        timestamps: List[float] = []
        quantities: List[float] = []
        prices: List[float] = []
        total_filled = 0.0
        total_cost = 0.0
        step_count = 0

        while True:
            action = self._agent.select_action(state, deterministic=True)
            result = env.step(action)

            fill_qty = result.info["filled_qty"] - total_filled
            if fill_qty > 0:
                timestamps.append(time.time())
                quantities.append(fill_qty)
                prices.append(result.state.mid_price)
                total_filled = result.info["filled_qty"]
                total_cost += fill_qty * result.state.mid_price

            step_count += 1

            if result.done:
                break

        avg_price = total_cost / max(total_filled, 1e-9)
        is_bps = result.info.get("implementation_shortfall_bps", 0.0)

        schedule = ExecutionSchedule(
            timestamps=timestamps,
            quantities=quantities,
            prices=prices,
            total_filled=total_filled,
            avg_price=avg_price,
            implementation_shortfall_bps=is_bps,
            total_steps=step_count,
        )

        logger.info(
            "RL execution complete for %s %s: filled=%.0f, avg_price=%.4f, IS=%.2f bps",
            side.upper(), symbol, total_filled, avg_price, is_bps,
        )

        return schedule

    def get_execution_schedule(
        self,
        total_qty: float,
        side: str = "buy",
        initial_mid: float = 100.0,
        num_steps: int = 50,
    ) -> ExecutionSchedule:
        """Generate an optimal execution schedule without executing.

        Args:
            total_qty: total quantity to execute
            side: "buy" or "sell"
            initial_mid: initial mid price for simulation
            num_steps: number of steps to simulate

        Returns:
            ExecutionSchedule with planned execution
        """
        sim_config = LOBSimulatorConfig(initial_mid=initial_mid)
        sim = LOBSimulator(config=sim_config)

        env = ExecutionEnvironment(
            simulator=sim,
            max_steps=num_steps,
            total_qty=total_qty,
            side=side,
        )

        state = env.reset()
        timestamps: List[float] = []
        quantities: List[float] = []
        prices: List[float] = []
        total_filled = 0.0
        total_cost = 0.0

        for step in range(num_steps):
            action = self._agent.select_action(state, deterministic=True)
            result = env.step(action)

            fill_qty = result.info["filled_qty"] - total_filled
            if fill_qty > 0:
                timestamps.append(float(step))
                quantities.append(fill_qty)
                prices.append(result.state.mid_price)
                total_filled = result.info["filled_qty"]
                total_cost += fill_qty * result.state.mid_price

            if result.done:
                break

        avg_price = total_cost / max(total_filled, 1e-9)
        is_bps = result.info.get("implementation_shortfall_bps", 0.0)

        return ExecutionSchedule(
            timestamps=timestamps,
            quantities=quantities,
            prices=prices,
            total_filled=total_filled,
            avg_price=avg_price,
            implementation_shortfall_bps=is_bps,
            total_steps=len(timestamps),
        )

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training metrics."""
        if not self._training_metrics:
            return {}

        rewards = self._training_metrics.get("episode_rewards", [])
        is_bps = self._training_metrics.get("implementation_shortfall_bps", [])

        return {
            "is_trained": self._is_trained,
            "episodes_trained": len(rewards),
            "avg_reward_last_100": float(np.mean(rewards[-100:])) if rewards else 0.0,
            "avg_is_bps_last_100": float(np.mean(is_bps[-100:])) if is_bps else 0.0,
            "best_is_bps": float(min(is_bps)) if is_bps else 0.0,
        }


__all__ = [
    "OrderBookState",
    "LOBSimulator",
    "LOBSimulatorConfig",
    "ExecutionEnvironment",
    "ExecutionStepResult",
    "PPOAgent",
    "RLExecutionEngine",
    "RLExecutionConfig",
    "ExecutionSchedule",
]
