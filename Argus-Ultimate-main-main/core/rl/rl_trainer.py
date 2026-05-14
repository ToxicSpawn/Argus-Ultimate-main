"""Push 66 — Unified RL trainer: PPO / TD3 / SAC via stable-baselines3.

Usage:
    trainer = RLTrainer(config, feed)
    model = trainer.train()            # offline on historical data
    trainer.save(model, "models/rl")
    model2 = trainer.load("models/rl/argus_PPO_final")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from core.rl.rl_config import RLConfig
from core.rl.rl_env import ArgusRLEnv

if TYPE_CHECKING:
    pass


class RLTrainer:
    """Trains PPO, TD3, or SAC agents on the ArgusRLEnv."""

    def __init__(self, config: RLConfig, feed: list):
        self.cfg = config
        self.feed = feed
        self._model = None
        self._vec_env = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        """Train and return the model. Requires stable-baselines3."""
        try:
            from stable_baselines3 import PPO, TD3, SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.callbacks import (
                EvalCallback, CheckpointCallback
            )
        except ImportError as e:
            raise ImportError(
                "stable-baselines3 required: pip install stable-baselines3"
            ) from e

        os.makedirs(self.cfg.model_dir, exist_ok=True)

        def make_env():
            return Monitor(ArgusRLEnv(self.feed, self.cfg))

        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )
        self._vec_env = vec_env

        algo_map = {"PPO": PPO, "TD3": TD3, "SAC": SAC}
        AlgoClass = algo_map[self.cfg.algorithm]
        kwargs = self._build_kwargs()

        model = AlgoClass(
            "MlpPolicy", vec_env,
            tensorboard_log=str(Path(self.cfg.model_dir) / "tb"),
            verbose=1,
            **kwargs,
        )

        callbacks = [
            EvalCallback(
                vec_env,
                eval_freq=self.cfg.eval_freq,
                n_eval_episodes=self.cfg.n_eval_episodes,
                best_model_save_path=self.cfg.model_dir,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=self.cfg.checkpoint_freq,
                save_path=self.cfg.model_dir,
                name_prefix=f"argus_{self.cfg.algorithm}",
            ),
        ]

        model.learn(
            total_timesteps=self.cfg.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        self._model = model
        return model

    def save(self, model, path: str | None = None) -> Path:
        p = Path(path or self.cfg.model_dir)
        p.mkdir(parents=True, exist_ok=True)
        model_path = p / f"argus_{self.cfg.algorithm}_final"
        model.save(str(model_path))
        if self._vec_env is not None:
            try:
                from stable_baselines3.common.vec_env import VecNormalize
                if isinstance(self._vec_env, VecNormalize):
                    self._vec_env.save(str(p / "vec_normalize.pkl"))
            except Exception:
                pass
        return model_path

    def load(self, path: str):
        try:
            from stable_baselines3 import PPO, TD3, SAC
        except ImportError as e:
            raise ImportError("stable-baselines3 required") from e
        algo_map = {"PPO": PPO, "TD3": TD3, "SAC": SAC}
        AlgoClass = algo_map[self.cfg.algorithm]
        return AlgoClass.load(path)

    # ------------------------------------------------------------------
    # Hyperparameter builders
    # ------------------------------------------------------------------

    def _build_kwargs(self) -> dict:
        c = self.cfg
        if c.algorithm == "PPO":
            return dict(
                learning_rate=c.ppo_learning_rate,
                n_steps=c.ppo_n_steps,
                batch_size=c.ppo_batch_size,
                n_epochs=c.ppo_n_epochs,
                gamma=c.ppo_gamma,
                gae_lambda=c.ppo_gae_lambda,
                clip_range=c.ppo_clip_range,
                ent_coef=c.ppo_ent_coef,
                vf_coef=c.ppo_vf_coef,
                max_grad_norm=c.ppo_max_grad_norm,
            )
        elif c.algorithm == "TD3":
            return dict(
                learning_rate=c.td3_learning_rate,
                buffer_size=c.td3_buffer_size,
                batch_size=c.td3_batch_size,
                gamma=c.td3_gamma,
                tau=c.td3_tau,
                policy_delay=c.td3_policy_delay,
                target_policy_noise=c.td3_target_policy_noise,
                target_noise_clip=c.td3_target_noise_clip,
                learning_starts=c.td3_learning_starts,
            )
        else:  # SAC
            return dict(
                learning_rate=c.sac_learning_rate,
                buffer_size=c.sac_buffer_size,
                batch_size=c.sac_batch_size,
                gamma=c.sac_gamma,
                tau=c.sac_tau,
                ent_coef=c.sac_ent_coef,
                learning_starts=c.sac_learning_starts,
                train_freq=c.sac_train_freq,
                gradient_steps=c.sac_gradient_steps,
            )
