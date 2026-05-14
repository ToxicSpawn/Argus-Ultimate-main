"""Q-learning strategy selector for buy/hold/sell and risk actions."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


ACTIONS = ("sell_small", "sell_normal", "hold", "buy_small", "buy_normal")


@dataclass
class RLDecision:
    action: str
    risk_multiplier: float
    q_value: float


class QLearningStrategyAgent:
    def __init__(self, learning_rate: float = 0.12, discount: float = 0.92, epsilon: float = 0.12):
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table: dict[tuple[int, int, int], np.ndarray] = {}

    def discretise(self, volatility: float, trend: float, drawdown: float) -> tuple[int, int, int]:
        vol_bin = int(np.clip(volatility / 0.01, 0, 4))
        trend_bin = int(np.clip((trend + 0.03) / 0.015, 0, 4))
        dd_bin = int(np.clip(drawdown / 0.05, 0, 4))
        return vol_bin, trend_bin, dd_bin

    def decide(self, volatility: float, trend: float, drawdown: float) -> RLDecision:
        state = self.discretise(volatility, trend, drawdown)
        values = self.q_table.setdefault(state, np.zeros(len(ACTIONS)))
        if random.random() < self.epsilon:
            idx = random.randrange(len(ACTIONS))
        else:
            idx = int(np.argmax(values))
        risk = {"sell_small": 0.5, "sell_normal": 1.0, "hold": 0.0, "buy_small": 0.5, "buy_normal": 1.0}[ACTIONS[idx]]
        return RLDecision(ACTIONS[idx], risk, float(values[idx]))

    def learn(self, state: tuple[int, int, int], action: str, reward: float, next_state: tuple[int, int, int]) -> None:
        values = self.q_table.setdefault(state, np.zeros(len(ACTIONS)))
        next_values = self.q_table.setdefault(next_state, np.zeros(len(ACTIONS)))
        idx = ACTIONS.index(action)
        target = reward + self.discount * float(np.max(next_values))
        values[idx] += self.learning_rate * (target - values[idx])


def _demo() -> None:
    agent = QLearningStrategyAgent(epsilon=0.0)
    state = agent.discretise(0.02, 0.01, 0.03)
    for _ in range(40):
        decision = agent.decide(0.02, 0.01, 0.03)
        reward = 0.003 if decision.action.startswith("buy") else -0.001
        agent.learn(state, decision.action, reward, state)
    print("RL strategy agent ready")
    print(agent.decide(0.02, 0.01, 0.03))


if __name__ == "__main__":
    _demo()
