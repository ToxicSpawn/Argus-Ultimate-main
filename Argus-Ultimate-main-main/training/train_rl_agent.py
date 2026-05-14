"""
PPO RL Execution Agent Training Script  (Push 27)
===================================================
Now consumes FeaturePipeline output (~1,000 features) as the observation
space instead of the previous 8-dim hardcoded state vector.

Fallback: if FeaturePipeline is unavailable, reverts to the legacy
8-dimensional observation space so training never hard-crashes.

Usage:
    python training/train_rl_agent.py --log ultimate_trading.log --timesteps 500000
    python training/train_rl_agent.py --synthetic --timesteps 200000
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Action constants (must match reinforcement_stub.py)
# ---------------------------------------------------------------------------
BUY_SMALL  = 0
BUY_LARGE  = 1
HOLD       = 2
SELL_SMALL = 3
SELL_LARGE = 4
N_ACTIONS  = 5

# Legacy fallback obs dim (used when FeaturePipeline is unavailable)
_LEGACY_OBS_DIM = 8

# ---------------------------------------------------------------------------
# FeaturePipeline import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from ml.feature_pipeline import FeaturePipeline
    _PIPELINE_AVAILABLE = True
    logger.info("FeaturePipeline loaded — using full multi-TF feature set")
except ImportError:
    _PIPELINE_AVAILABLE = False
    logger.warning("FeaturePipeline not found — falling back to legacy 8-dim obs")


def _get_obs_dim() -> int:
    """Probe FeaturePipeline for its output dimension, or return legacy dim."""
    if not _PIPELINE_AVAILABLE:
        return _LEGACY_OBS_DIM
    try:
        dummy = _make_dummy_candles(200)
        fp = FeaturePipeline(timeframes=["1m", "5m", "15m", "1h", "4h"])
        result = fp.build(dummy)
        dim = int(result.X.shape[1])
        logger.info("FeaturePipeline obs dim = %d", dim)
        return dim
    except Exception as exc:
        logger.warning("FeaturePipeline probe failed (%s) — using legacy dim", exc)
        return _LEGACY_OBS_DIM


# ---------------------------------------------------------------------------
# Candle helpers
# ---------------------------------------------------------------------------

def _make_dummy_candles(n: int = 300) -> List[Dict[str, Any]]:
    """Generate synthetic OHLCV candles for pipeline probing / synthetic training."""
    rng = np.random.default_rng(42)
    price = 50_000.0
    candles = []
    for i in range(n):
        o = price
        h = price * (1 + rng.uniform(0, 0.005))
        l = price * (1 - rng.uniform(0, 0.005))
        c = rng.uniform(l, h)
        v = rng.uniform(0.5, 20.0)
        candles.append({
            "timestamp": 1_700_000_000 + i * 60,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        })
        price = c
    return candles


def _candles_from_episode(ep: Dict[str, Any], n: int = 300) -> List[Dict[str, Any]]:
    """Re-create a plausible candle history from a trade episode dict."""
    rng = np.random.default_rng(abs(hash(str(ep.get("pnl", 0)))) % (2**31))
    base_price = float(ep.get("entry_price", 50_000.0))
    vol_1h = float(ep.get("volatility_1h", 0.03))
    candles = []
    price = base_price
    for i in range(n):
        move = rng.normal(0, vol_1h / math.sqrt(60))
        o = price
        h = price * (1 + abs(rng.normal(0, vol_1h / 4)))
        l = price * (1 - abs(rng.normal(0, vol_1h / 4)))
        c = price * (1 + move)
        c = max(l, min(h, c))
        v = rng.uniform(0.5, 20.0)
        candles.append({
            "timestamp": 1_700_000_000 + i * 60,
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        })
        price = c
    return candles


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("gymnasium not installed — env stub only")


if GYM_AVAILABLE:
    class ArgusExecutionEnv(gym.Env):
        """
        RL execution environment — Push 27 upgrade.

        Observation space:
            If FeaturePipeline is available: ~1,000 multi-timeframe features
            Fallback: 8-dim legacy state vector

        Action space:
            Discrete(5)  — BUY_SMALL / BUY_LARGE / HOLD / SELL_SMALL / SELL_LARGE
        """

        metadata = {"render_modes": []}

        def __init__(self, episodes: List[Dict[str, Any]], obs_dim: int):
            super().__init__()
            self._episodes  = episodes
            self._obs_dim   = obs_dim
            self._idx       = 0
            self._current_episode: Optional[Dict] = None
            self._pipeline  = None

            if _PIPELINE_AVAILABLE:
                try:
                    self._pipeline = FeaturePipeline(
                        timeframes=["1m", "5m", "15m", "1h", "4h"]
                    )
                except Exception as exc:
                    logger.warning("FeaturePipeline init failed: %s", exc)

            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(self._obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(N_ACTIONS)

        # ------------------------------------------------------------------
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            ep = self._episodes[self._idx % len(self._episodes)]
            self._idx += 1
            self._current_episode = ep
            return self._obs(ep), {}

        def step(self, action: int):
            ep      = self._current_episode
            reward  = self._reward(action, ep)
            obs     = self._obs(ep)
            info    = {"pnl": ep.get("pnl", 0.0), "action": action}
            return obs, reward, True, False, info

        # ------------------------------------------------------------------
        def _obs(self, ep: Dict) -> np.ndarray:
            """Build observation vector from episode — pipeline or legacy."""
            if self._pipeline is not None:
                try:
                    candles = _candles_from_episode(ep)
                    result  = self._pipeline.build(candles)
                    row     = result.X[-1]  # most recent bar
                    # Pad or trim to obs_dim
                    if len(row) >= self._obs_dim:
                        return np.clip(row[:self._obs_dim].astype(np.float32), -10.0, 10.0)
                    padded = np.zeros(self._obs_dim, dtype=np.float32)
                    padded[:len(row)] = row
                    return np.clip(padded, -10.0, 10.0)
                except Exception:
                    pass  # fall through to legacy

            # Legacy 8-dim fallback
            return np.array([
                float(ep.get("position_usd",             0.0)),
                float(ep.get("unrealised_pnl",           0.0)),
                float(ep.get("volatility_1h",            0.3)),
                float(ep.get("spread_bps",               5.0)),
                float(ep.get("ob_imbalance",             0.0)),
                float(ep.get("time_of_day_sin",          0.0)),
                float(ep.get("time_of_day_cos",          1.0)),
                float(ep.get("slippage_budget_remaining",20.0)),
            ], dtype=np.float32)

        def _reward(self, action: int, ep: Dict) -> float:
            pnl          = float(ep.get("pnl",                    0.0))
            cost         = float(ep.get("cost",                  100.0))
            slippage_bps = float(ep.get("slippage_bps",           0.0))
            budget_rem   = float(ep.get("slippage_budget_remaining", 20.0))
            pnl_norm     = pnl / max(1.0, cost)
            slip_pen     = max(0.0, slippage_bps - 3.0) * 0.01

            if action == BUY_LARGE:
                r = pnl_norm - slip_pen
            elif action == BUY_SMALL:
                r = pnl_norm * 0.25 - slip_pen * 0.25
            elif action == HOLD:
                r = 0.0
            else:  # SELL while flat
                r = -0.05

            if budget_rem > 10.0:
                r += 0.01
            return float(r)


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def parse_trade_log(log_path: str) -> List[Dict[str, Any]]:
    episodes: List[Dict] = []
    pnl_re  = re.compile(r"CLOSED\s+(\S+)\s+\(\S+\).*P&L=\$([\-\d\.]+)")
    cost_re = re.compile(r"cost=(\S+)")
    ep_re   = re.compile(r"entry_price=(\S+)")

    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        logger.warning("Log file not found: %s", log_path)
        return []

    for line in lines:
        m = pnl_re.search(line)
        if not m:
            continue
        pnl    = float(m.group(2))
        cost_m = cost_re.search(line)
        ep_m   = ep_re.search(line)
        episodes.append({
            "pnl":                       pnl,
            "cost":                      float(cost_m.group(1)) if cost_m else 100.0,
            "entry_price":               float(ep_m.group(1))  if ep_m  else 50_000.0,
            "position_usd":              0.0,
            "unrealised_pnl":            0.0,
            "volatility_1h":             0.03,
            "spread_bps":                5.0,
            "ob_imbalance":              0.0,
            "time_of_day_sin":           0.0,
            "time_of_day_cos":           1.0,
            "slippage_budget_remaining": 20.0,
            "slippage_bps":              2.0,
        })

    logger.info("Parsed %d trade episodes from %s", len(episodes), log_path)
    return episodes


def generate_synthetic_episodes(n: int = 5000) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(42)
    episodes = []
    for _ in range(n):
        frac = float(rng.uniform(0.0, 1.0))
        episodes.append({
            "pnl":                       float(rng.normal(0.5, 3.0)),
            "cost":                      float(rng.uniform(50.0, 500.0)),
            "entry_price":               float(rng.uniform(20_000.0, 70_000.0)),
            "position_usd":              0.0,
            "unrealised_pnl":            0.0,
            "volatility_1h":             float(rng.uniform(0.005, 0.08)),
            "spread_bps":                float(rng.uniform(2.0, 25.0)),
            "ob_imbalance":              float(rng.uniform(-1.0, 1.0)),
            "time_of_day_sin":           math.sin(2 * math.pi * frac),
            "time_of_day_cos":           math.cos(2 * math.pi * frac),
            "slippage_budget_remaining": float(rng.uniform(0.0, 20.0)),
            "slippage_bps":              float(rng.uniform(0.5, 8.0)),
        })
    return episodes


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(log_path: str, timesteps: int, output_dir: str, synthetic: bool = False) -> None:
    if not GYM_AVAILABLE:
        logger.error("gymnasium not installed. Run: pip install gymnasium stable-baselines3")
        return

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
        return

    # Probe obs dim once (expensive if pipeline, cheap if legacy)
    obs_dim = _get_obs_dim()
    logger.info("Observation dimension: %d", obs_dim)

    # Load episodes
    episodes: List[Dict] = []
    if not synthetic:
        episodes = parse_trade_log(log_path)
    if len(episodes) < 50:
        logger.info("Augmenting with synthetic episodes (have %d real)", len(episodes))
        episodes += generate_synthetic_episodes(max(1000, 5000 - len(episodes)))

    logger.info("Training on %d episodes | timesteps=%d | obs_dim=%d",
                len(episodes), timesteps, obs_dim)

    env      = ArgusExecutionEnv(episodes=episodes, obs_dim=obs_dim)
    eval_env = ArgusExecutionEnv(
        episodes=episodes[:max(1, len(episodes) // 5)], obs_dim=obs_dim
    )

    # Wider network to handle the expanded observation space
    policy_kwargs = dict(net_arch=[512, 512, 256])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "tb_logs"),
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(1, timesteps // 20),
        n_eval_episodes=50,
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb)

    out_path = os.path.join(output_dir, "rl_execution_agent_ppo")
    model.save(out_path)
    logger.info("Model saved → %s.zip", out_path)
    logger.info("obs_dim=%d  net_arch=%s", obs_dim, policy_kwargs["net_arch"])


def main():
    parser = argparse.ArgumentParser(description="Train Argus PPO RL Execution Agent (Push 27)")
    parser.add_argument("--log",        default="ultimate_trading.log")
    parser.add_argument("--timesteps",  type=int, default=500_000)
    parser.add_argument("--output",     default="models")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Skip log parsing, train on synthetic data only")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    train(args.log, args.timesteps, args.output, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
