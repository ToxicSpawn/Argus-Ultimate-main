"""Push 66 — RL hyperparameter configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RLConfig:
    # Algorithm selection
    algorithm: Literal["PPO", "TD3", "SAC"] = "PPO"

    # Environment
    initial_equity: float = 10_000.0
    obs_dim: int = 7
    action_low: float = -1.0
    action_high: float = 1.0
    fee_bps: float = 2.0           # 2bps round-trip fee
    min_conviction: float = 0.30   # abs(action) threshold to emit signal

    # PPO hyperparams (Optuna-tuned for crypto, 2025)
    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 10
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5

    # TD3 hyperparams
    td3_learning_rate: float = 1e-3
    td3_buffer_size: int = 1_000_000
    td3_batch_size: int = 256
    td3_gamma: float = 0.99
    td3_tau: float = 0.005
    td3_policy_delay: int = 2
    td3_target_policy_noise: float = 0.2
    td3_target_noise_clip: float = 0.5
    td3_learning_starts: int = 10_000

    # SAC hyperparams
    sac_learning_rate: float = 3e-4
    sac_buffer_size: int = 1_000_000
    sac_batch_size: int = 256
    sac_gamma: float = 0.99
    sac_tau: float = 0.005
    sac_ent_coef: str = "auto"     # automatic entropy tuning
    sac_learning_starts: int = 10_000
    sac_train_freq: int = 1
    sac_gradient_steps: int = 1

    # Training
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 50_000
    model_dir: str = "models/rl"

    # Reward shaping
    sharpe_window: int = 20        # rolling window for Sharpe reward
    cost_penalty_scale: float = 100.0
    inventory_penalty: float = 0.001

    def __post_init__(self):
        assert self.algorithm in ("PPO", "TD3", "SAC"), f"Unknown algorithm: {self.algorithm}"
        assert self.initial_equity > 0
        assert 0.0 < self.min_conviction <= 1.0
