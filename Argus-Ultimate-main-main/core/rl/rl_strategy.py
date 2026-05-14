"""Push 66 — RLStrategy: live inference wrapper around trained PPO/TD3/SAC.

Wires the trained model into the Argus strategy pipeline:
  on_bar() -> FeatureBuilder -> VecNormalize -> model.predict() -> emit_signal()
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from core.rl.rl_config import RLConfig
from core.rl.rl_feature_builder import FeatureBuilder


class RLStrategy:
    """Live-inference strategy consuming a trained SB3 model.

    Compatible with Argus StrategyRegistry when subclassed from BaseStrategy.
    Kept dependency-free at class level — SB3 loaded lazily.
    """

    name = "RLStrategy"

    def __init__(
        self,
        model_path: str,
        config: RLConfig | None = None,
        normalizer_path: str | None = None,
    ):
        self.cfg = config or RLConfig()
        self._model_path = model_path
        self._normalizer_path = normalizer_path
        self._model = None
        self._normalizer = None
        self._fb = FeatureBuilder()
        self._position: float = 0.0
        self._equity: float = self.cfg.initial_equity
        self._loaded = False
        self._signals_emitted: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Lazy-load model + normalizer."""
        try:
            from stable_baselines3 import PPO, TD3, SAC
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        except ImportError as e:
            raise ImportError("stable-baselines3 required") from e

        algo_map = {"PPO": PPO, "TD3": TD3, "SAC": SAC}
        AlgoClass = algo_map[self.cfg.algorithm]
        self._model = AlgoClass.load(self._model_path)

        if self._normalizer_path and Path(self._normalizer_path).exists():
            from core.rl.rl_env import ArgusRLEnv
            dummy = DummyVecEnv([lambda: ArgusRLEnv([], self.cfg)])
            self._normalizer = VecNormalize.load(self._normalizer_path, dummy)
            self._normalizer.training = False
            self._normalizer.norm_reward = False

        self._loaded = True

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def predict(self, bar: Any, pnl_norm: float = 0.0) -> dict | None:
        """Build obs from bar, run inference, return signal dict or None."""
        if not self._loaded:
            self.load()

        self._fb.update(bar)
        obs = self._fb.build(bar, inventory=self._position, pnl_norm=pnl_norm)

        if self._normalizer is not None:
            obs_input = obs.reshape(1, -1)
            obs_input = self._normalizer.normalize_obs(obs_input)
        else:
            obs_input = obs.reshape(1, -1)

        action, _states = self._model.predict(obs_input, deterministic=True)
        target_pos = float(np.clip(action.flat[0], -1.0, 1.0))
        conviction = abs(target_pos)

        if conviction < self.cfg.min_conviction:
            return None

        side = "buy" if target_pos > 0 else "sell"
        self._signals_emitted += 1
        return {
            "symbol": getattr(bar, "symbol", "UNKNOWN"),
            "side": side,
            "confidence": float(conviction),
            "target_position": target_pos,
            "algorithm": self.cfg.algorithm,
            "signals_emitted": self._signals_emitted,
        }

    def update_state(self, position: float, equity: float) -> None:
        """Called by execution engine after each fill."""
        self._position = position
        self._equity = equity
        self._fb.regime = getattr(self, "_regime", 0.0)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
