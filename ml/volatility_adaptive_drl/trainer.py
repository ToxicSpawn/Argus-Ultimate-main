"""Walk-forward trainer for volatility-adaptive SAC agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .environment import TradingEnvironment
from .sac_agent import SACAgent, SACUpdateMetrics

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainerConfig:
    train_steps_per_split: int = 2_000
    evaluation_episodes: int = 2
    checkpoint_every_split: bool = True
    checkpoint_dir: str = "checkpoints/volatility_adaptive_drl"


@dataclass(slots=True)
class TrainingSummary:
    mean_training_reward: float
    mean_eval_reward: float
    sharpe_like: float
    max_drawdown: float
    regime_scores: dict[str, float] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)


class VolatilityAdaptiveTrainer:
    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def walk_forward_train(
        self,
        env_factory: Callable[[int], TradingEnvironment],
        agent_factory: Callable[[int], SACAgent],
        num_splits: int,
    ) -> TrainingSummary:
        if num_splits <= 0:
            raise ValueError("num_splits must be positive")
        training_rewards: list[float] = []
        evaluation_rewards: list[float] = []
        drawdowns: list[float] = []
        regime_scores: dict[str, list[float]] = {}
        checkpoints: list[str] = []
        for split_index in range(num_splits):
            env = env_factory(split_index)
            agent = agent_factory(split_index)
            split_rewards = self._train_single_split(env, agent)
            training_rewards.extend(split_rewards)
            evaluation = self.regime_stratified_evaluation(agent, env, episodes=self.config.evaluation_episodes)
            evaluation_rewards.extend(evaluation.regime_scores.values())
            drawdowns.extend(evaluation.drawdowns)
            for regime, score in evaluation.regime_scores.items():
                regime_scores.setdefault(regime, []).append(score)
            if self.config.checkpoint_every_split:
                checkpoint_path = self._checkpoint(agent, f"split_{split_index}")
                checkpoints.append(checkpoint_path)
        training_array = np.asarray(training_rewards or [0.0], dtype=np.float32)
        eval_array = np.asarray(evaluation_rewards or [0.0], dtype=np.float32)
        return TrainingSummary(
            mean_training_reward=float(training_array.mean()),
            mean_eval_reward=float(eval_array.mean()),
            sharpe_like=float(eval_array.mean() / max(eval_array.std(), 1e-6)),
            max_drawdown=float(np.min(drawdowns) if drawdowns else 0.0),
            regime_scores={name: float(np.mean(values)) for name, values in regime_scores.items()},
            checkpoints=checkpoints,
        )

    def regime_stratified_evaluation(self, agent: SACAgent, env: TradingEnvironment, episodes: int = 1) -> "_EvaluationResult":
        regime_scores: dict[str, list[float]] = {}
        drawdowns: list[float] = []
        for _ in range(max(1, episodes)):
            state, _ = env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                current_regime = env.regime_detector.current_regime()
                action = agent.select_action(
                    state,
                    volatility=current_regime.realized_volatility,
                    regime=current_regime.label,
                    deterministic=True,
                )
                state, reward, done, truncated, info = env.step(action)
                regime_scores.setdefault(str(info.get("regime", "medium")), []).append(float(reward))
                drawdowns.append(float(info.get("drawdown", 0.0)))
        return _EvaluationResult(
            regime_scores={name: float(np.mean(values)) for name, values in regime_scores.items()},
            drawdowns=drawdowns,
        )

    def _train_single_split(self, env: TradingEnvironment, agent: SACAgent) -> list[float]:
        state, _ = env.reset()
        rewards: list[float] = []
        for _ in range(self.config.train_steps_per_split):
            current_regime = env.regime_detector.current_regime()
            deterministic = agent.total_steps < agent.config.warmup_steps
            action = agent.select_action(
                state,
                volatility=current_regime.realized_volatility,
                regime=current_regime.label,
                deterministic=deterministic,
            )
            next_state, reward, done, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done or truncated, str(info.get("regime", "medium")))
            update_metrics = agent.update()
            if isinstance(update_metrics, SACUpdateMetrics):
                rewards.append(float(reward))
            state = next_state
            if done or truncated:
                state, _ = env.reset()
        return rewards

    def _checkpoint(self, agent: SACAgent, name: str) -> str:
        path = self.checkpoint_dir / f"{name}.pt"
        agent.save(str(path))
        return str(path)


@dataclass(slots=True)
class _EvaluationResult:
    regime_scores: dict[str, float]
    drawdowns: list[float]
