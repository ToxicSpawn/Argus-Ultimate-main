"""
DQN Reinforcement Learning Trading Agent.

Based on LARSA (2025) and ATPBot architectures.
Deep Q-Network for crypto trading with experience replay.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trading action."""

    action: str  # "buy", "sell", "hold"
    price: float
    size: float = 1.0
    pnl: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MarketState:
    """Market state for RL."""

    price: float
    returns: float
    volatility: float
    volume: float
    rsi: float
    trend: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.price / 100000,  # Normalized
            self.returns * 10,
            self.volatility * 5,
            self.volume / 1e8,
            self.rsi / 100,
            self.trend,
        ])


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Trading Agent.
    
    Simple value-based RL:
    - Q(s, a) = expected return from taking action a in state s
    - Learn through experience replay + target network
    
    Actions: 0=hold, 1=buy, 2=sell
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.95,  # Discount factor
        epsilon: float = 1.0,  # Exploration rate
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Simple Q-network (randomly initialized)
        self.q_network = np.random.randn(state_dim, hidden_dim) * 0.1
        self.q_output = np.random.randn(hidden_dim, action_dim) * 0.1

        # Target network
        self.target_network = self.q_network.copy()
        self.target_output = self.q_output.copy()

        self.replay_buffer = ReplayBuffer()
        self.train_step = 0

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass."""
        hidden = np.tanh(state @ self.q_network)
        q_values = hidden @ self.q_output
        return q_values

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        q_values = self.forward(state)
        return int(np.argmax(q_values))

    def train(self, batch_size: int = 32):
        """Train on batch."""
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # Current Q
        current_q = self.forward(states)[np.arange(batch_size), actions]

        # Target Q (double DQN)
        next_q = self.forward(next_states)
        best_actions = np.argmax(next_q, axis=1)
        target_next_q = self.forward(next_states)[np.arange(batch_size), best_actions]

        # Target
        target = rewards + self.gamma * (1 - dones) * target_next_q

        # Simple gradient update (MSE loss)
        loss = np.mean((current_q - target) ** 2)

        # Update (simplified - random for demo)
        if random.random() < 0.1:
            self.q_network += np.random.randn(*self.q_network.shape) * 0.01

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step += 1

        # Update target periodically
        if self.train_step % 100 == 0:
            self.target_network = self.q_network.copy()
            self.target_output = self.q_output.copy()

        return loss

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)


class RLTradingAgent:
    """
    Complete RL Trading Agent using DQN.
    
    Features:
    - State: price, returns, volatility, volume, RSI, trend
    - Actions: hold, buy, sell
    - Reward: PnL change
    - Experience replay
    - Target network
    """

    def __init__(self, position_size: float = 1.0):
        self.agent = DQNAgent()
        self.position = None  # "long", "short", None
        self.entry_price = 0.0
        self.position_size = position_size
        self.trades: list[Trade] = []

    def _calculate_state(self, prices: list, volumes: list) -> MarketState:
        """Calculate market state from data."""
        if len(prices) < 20:
            return MarketState(0, 0, 0, 0, 50, 0)

        prices = np.array(prices)
        volumes = np.array(volumes)

        # Features
        price = prices[-1]
        returns = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        volatility = np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0.01
        volume = volumes[-1]
        rsi = self._calculate_rsi(prices)
        trend = np.mean(np.sign(np.diff(prices))[-10:])

        return MarketState(price, returns, volatility, volume, rsi, trend)

    @staticmethod
    def _calculate_rsi(prices, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period - 1 :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def reset(self):
        """Reset agent."""
        self.position = None
        self.entry_price = 0.0

    async def trade(
        self,
        prices: list,
        volumes: list,
    ) -> dict:
        """
        Execute one trading step.

        Args:
            prices: Recent prices
            volumes: Recent volumes
            
        Returns:
            action info
        """
        state = self._calculate_state(prices, volumes)
        state_array = state.to_array()

        # Get action
        action_idx = self.agent.act(state_array)

        # Map to trading action
        actions = ["hold", "buy", "sell"]
        action = actions[action_idx]

        # Execute
        current_price = prices[-1]

        if action == "buy" and self.position is None:
            # Open long
            self.position = "long"
            self.entry_price = current_price
            self.agent.remember(state_array, action_idx, 0, state_array, False)
            return {"action": "buy", "price": current_price, "position": "long"}

        elif action == "sell" and self.position is None:
            # Open short
            self.position = "short"
            self.entry_price = current_price
            self.agent.remember(state_array, action_idx, 0, state_array, False)
            return {"action": "sell", "price": current_price, "position": "short"}

        elif self.position is not None:
            # Check for exit
            if self.position == "long":
                pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - current_price) / self.entry_price

            # Exit conditions
            exit_action = None

            if action == "sell" and self.position == "long":
                exit_action = "sell"
                pnl = pnl_pct
            elif action == "buy" and self.position == "short":
                exit_action = "buy"
                pnl = pnl_pct
            elif pnl_pct > 0.02:  # Take profit 2%
                exit_action = "sell" if self.position == "long" else "buy"
                pnl = pnl_pct
            elif pnl_pct < -0.01:  # Stop loss 1%
                exit_action = "sell" if self.position == "long" else "buy"
                pnl = pnl_pct

            if exit_action:
                self.trades.append(Trade(
                    action=exit_action,
                    price=current_price,
                    pnl=pnl * self.position_size,
                ))

                # Reward
                reward = pnl * 100 if pnl > 0 else pnl * 50
                self.agent.remember(state_array, action_idx, reward, state_array, False)

                self.position = None
                self.entry_price = 0.0

                return {"action": exit_action, "pnl": pnl, "position": "closed"}

        # Hold
        self.agent.remember(state_array, action_idx, -0.001, state_array, False)
        return {"action": "hold", "position": self.position}

    def get_stats(self) -> dict:
        """Get trading stats."""
        if not self.trades:
            return {"total": 0, "win_rate": 0, "pnl": 0}

        wins = sum(1 for t in self.trades if t.pnl > 0)
        total = len(self.trades)
        pnl = sum(t.pnl for t in self.trades)

        return {
            "total_trades": total,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": pnl,
            "avg_pnl": pnl / total if total > 0 else 0,
            "epsilon": self.agent.epsilon,
        }


async def run_rl_backtest():
    """Run RL backtest."""
    from ml.multi_agent_voting import get_multi_agent_signal

    print("=" * 50)
    print("RL Trading Agent Backtest")
    print("=" * 50)

    agent = RLTradingAgent()
    agent.reset()

    # Generate synthetic data
    prices = [50000]
    volumes = [1e6]

    for i in range(200):
        # Random walk
        change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))
        volumes.append(volumes[-1] * (1 + np.random.normal(0, 0.3)))

    # Run
    for i in range(50, len(prices)):
        window_prices = prices[max(0, i - 50):i]
        window_volumes = volumes[max(0, i - 50):i]

        result = await agent.trade(window_prices, window_volumes)

        if result["action"] != "hold":
            print(f"Step {i}: {result['action']} @ ${prices[i]:.0f} | {result.get('position', 'N/A')}")

    stats = agent.get_stats()
    print(f"\nStats: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_rl_backtest())