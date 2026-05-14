"""
RL Execution Agent Training — PPO on synthetic crypto microstructure.

Trains a PPO agent (stable-baselines3) to optimise order timing and sizing.
Uses a synthetic Gymnasium environment that replays historical OHLCV data,
simulating spreads, slippage, and position P&L.

Usage:
    python -m ml.training.train_rl_agent
    python -m ml.training.train_rl_agent --timesteps 2_000_000 --symbol BTC/USD
    python -m ml.training.train_rl_agent --timesteps 500_000 --no-gpu   # CPU fallback

Output:
    models/rl_agent.zip        — trained PPO model
    models/rl_agent_best.zip   — best checkpoint (highest eval reward)
    ml/weights/rl_metadata.json

After training, update unified_config.yaml:
    reinforcement_agent:
      model_path: "models/rl_agent.zip"
      use_rl_agent: true
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_OK = True
except ImportError:
    _GYM_OK = False
    logger.error("gymnasium not installed. Run: pip install gymnasium")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    _SB3_OK = True
except ImportError:
    _SB3_OK = False
    logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3[extra]")

try:
    import torch
    _CUDA = torch.cuda.is_available()
    if _CUDA:
        logger.info("CUDA available: %s (%s)", torch.cuda.get_device_name(0),
                    f"{torch.cuda.get_device_properties(0).total_memory // 2**20}MB")
except ImportError:
    _CUDA = False

try:
    import ccxt
    _CCXT = True
except ImportError:
    _CCXT = False

# ---------------------------------------------------------------------------
# Synthetic OHLCV generator (GBM) — used when live data unavailable
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(n_bars: int = 50_000, start_price: float = 60_000.0,
                      sigma_daily: float = 0.03) -> np.ndarray:
    """Returns (n, 5) array: open, high, low, close, volume."""
    dt = 1 / (24 * 12)          # 5-minute bars → fraction of day
    sigma_bar = sigma_daily * math.sqrt(dt)
    prices = [start_price]
    for _ in range(n_bars - 1):
        prices.append(prices[-1] * math.exp(np.random.normal(0, sigma_bar)))
    prices = np.array(prices)

    spread = prices * 0.0005    # 5 bps spread
    high = prices + spread
    low = prices - spread
    volume = np.random.lognormal(10, 1, n_bars)
    ohlcv = np.column_stack([prices, high, low, prices, volume])
    return ohlcv

# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class CryptoExecutionEnv(gym.Env):
    """
    Simple crypto execution environment.

    Observation (9 dims):
        position_norm, unrealised_pnl_norm, vol_1h, spread_bps_norm,
        ob_imbalance, tod_sin, tod_cos, slippage_budget_norm, bars_since_trade_norm

    Actions: HOLD(0) BUY_SMALL(1) BUY_LARGE(2) SELL_SMALL(3) SELL_LARGE(4)
    """
    metadata = {"render_modes": []}

    HOLD, BUY_SMALL, BUY_LARGE, SELL_SMALL, SELL_LARGE = 0, 1, 2, 3, 4
    SIZE_FACTORS = {0: 0.0, 1: 0.25, 2: 1.0, 3: 0.25, 4: 1.0}
    MAX_POSITION_USD = 1_000.0
    SLIPPAGE_BUDGET_BPS = 20.0
    EPISODE_BARS = 2_016          # 7 days of 5-min bars

    def __init__(self, ohlcv: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self._ohlcv = ohlcv if ohlcv is not None else _synthetic_ohlcv()
        self._n = len(self._ohlcv)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self._idx = 0
        self._position_usd = 0.0
        self._entry_price = 0.0
        self._slippage_used = 0.0
        self._bars_since_trade = 0
        self._episode_reward = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Random start within data (leave room for episode length)
        self._idx = int(np.random.randint(0, max(1, self._n - self.EPISODE_BARS - 100)))
        self._episode_end = min(self._idx + self.EPISODE_BARS, self._n - 1)
        self._position_usd = 0.0
        self._entry_price = 0.0
        self._slippage_used = 0.0
        self._bars_since_trade = 0
        self._episode_reward = 0.0
        return self._obs(), {}

    def _price(self) -> float:
        return float(self._ohlcv[self._idx, 3])  # close

    def _spread_bps(self) -> float:
        return 5.0 + float(np.random.exponential(2.0))  # realistic spread noise

    def _ob_imbalance(self) -> float:
        # Proxy from price change: positive → bullish OB
        if self._idx < 1:
            return 0.0
        ret = (self._ohlcv[self._idx, 3] / self._ohlcv[self._idx - 1, 3]) - 1.0
        return float(np.clip(ret * 200, -1.0, 1.0))

    def _vol_1h(self) -> float:
        start = max(0, self._idx - 12)
        closes = self._ohlcv[start:self._idx + 1, 3]
        if len(closes) < 2:
            return 0.02
        return float(np.std(np.diff(np.log(closes + 1e-10))) * math.sqrt(12 * 24 * 252))

    def _obs(self) -> np.ndarray:
        price = self._price()
        tod_frac = (self._idx % (24 * 12)) / (24 * 12)
        unrealised = 0.0
        if self._position_usd != 0 and self._entry_price > 0:
            unrealised = self._position_usd * (price / self._entry_price - 1.0)
        return np.array([
            self._position_usd / self.MAX_POSITION_USD,
            unrealised / 100.0,
            self._vol_1h(),
            self._spread_bps() / 100.0,
            self._ob_imbalance(),
            math.sin(2 * math.pi * tod_frac),
            math.cos(2 * math.pi * tod_frac),
            (self.SLIPPAGE_BUDGET_BPS - self._slippage_used) / self.SLIPPAGE_BUDGET_BPS,
            min(self._bars_since_trade / 100.0, 1.0),
        ], dtype=np.float32)

    def step(self, action: int):
        price = self._price()
        spread_bps = self._spread_bps()
        sf = self.SIZE_FACTORS.get(int(action), 0.0)

        reward = 0.0
        slippage_cost = 0.0

        if action in (self.BUY_SMALL, self.BUY_LARGE):
            trade_usd = self.MAX_POSITION_USD * sf
            if self._position_usd + trade_usd <= self.MAX_POSITION_USD:
                slippage_bps = spread_bps * 0.5
                fill_price = price * (1 + slippage_bps / 10_000)
                self._entry_price = (
                    (self._position_usd * self._entry_price + trade_usd * fill_price)
                    / (self._position_usd + trade_usd)
                ) if self._position_usd > 0 else fill_price
                self._position_usd += trade_usd
                slippage_cost = slippage_bps
                self._slippage_used += slippage_bps
                self._bars_since_trade = 0
            else:
                reward -= 0.001  # Penalty for trying to over-size

        elif action in (self.SELL_SMALL, self.SELL_LARGE):
            trade_usd = min(self._position_usd, self.MAX_POSITION_USD * sf)
            if trade_usd > 0 and self._position_usd > 0:
                slippage_bps = spread_bps * 0.5
                fill_price = price * (1 - slippage_bps / 10_000)
                pnl = trade_usd * (fill_price / self._entry_price - 1.0)
                reward += pnl / 100.0   # Scale reward to ~(-1, +1) range
                self._position_usd -= trade_usd
                slippage_cost = slippage_bps
                self._slippage_used += slippage_bps
                self._bars_since_trade = 0
                if self._position_usd <= 0:
                    self._entry_price = 0.0
            else:
                reward -= 0.001

        # Mark-to-market reward on existing position (small incentive for good timing)
        if self._position_usd > 0 and self._idx > 0 and self._entry_price > 0:
            bar_ret = (price / float(self._ohlcv[self._idx - 1, 3])) - 1.0
            reward += self._position_usd / self.MAX_POSITION_USD * bar_ret * 0.1

        # Slippage budget penalty
        if self._slippage_used > self.SLIPPAGE_BUDGET_BPS:
            reward -= 0.01

        self._bars_since_trade += 1
        self._idx += 1
        terminated = self._idx >= self._episode_end
        truncated = False

        # Terminal: force close position penalty if still open
        if terminated and self._position_usd > 0:
            close_slippage = spread_bps * 0.5
            pnl = self._position_usd * (price / self._entry_price - 1.0)
            reward += pnl / 100.0 - close_slippage / 1000.0

        self._episode_reward += reward
        return self._obs(), reward, terminated, truncated, {}

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    timesteps: int = 1_000_000,
    symbol: str = "BTC/USD",
    use_gpu: bool = True,
    n_envs: int = 8,
    eval_freq: int = 50_000,
    save_dir: str = "models",
) -> str:
    """Train PPO agent. Returns path to saved model."""
    if not _GYM_OK or not _SB3_OK:
        raise RuntimeError("gymnasium and stable-baselines3 are required. "
                           "pip install gymnasium stable-baselines3[extra]")

    Path(save_dir).mkdir(exist_ok=True)
    Path("ml/weights").mkdir(exist_ok=True)

    # Load historical data if ccxt available; else synthetic
    ohlcv_data: Optional[np.ndarray] = None
    if _CCXT:
        try:
            logger.info("Fetching historical OHLCV for %s from Kraken...", symbol)
            exchange = ccxt.kraken()
            bars = exchange.fetch_ohlcv(symbol, "5m", limit=10_000)
            ohlcv_data = np.array([[b[1], b[2], b[3], b[4], b[5]] for b in bars])
            logger.info("Loaded %d bars from Kraken", len(ohlcv_data))
        except Exception as e:
            logger.warning("Could not fetch live data (%s); using synthetic", e)

    if ohlcv_data is None or len(ohlcv_data) < 500:
        logger.info("Generating synthetic OHLCV data (%d bars)", 100_000)
        ohlcv_data = _synthetic_ohlcv(n_bars=100_000)

    device = "cuda" if (use_gpu and _CUDA) else "cpu"
    logger.info("Training PPO on %s | %d timesteps | %d envs | device=%s",
                symbol, timesteps, n_envs, device)

    def make_env():
        env = CryptoExecutionEnv(ohlcv_data)
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=device,
        tensorboard_log=f"{save_dir}/tb_logs",
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )

    best_path = f"{save_dir}/rl_agent_best"
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_path,
        log_path=f"{save_dir}/eval_logs",
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=True)
    elapsed = time.time() - t0

    out_path = f"{save_dir}/rl_agent.zip"
    model.save(out_path)
    logger.info("Saved model to %s (%.1f min)", out_path, elapsed / 60)

    # Write metadata
    meta = {
        "symbol": symbol,
        "timesteps": timesteps,
        "n_envs": n_envs,
        "device": device,
        "training_minutes": round(elapsed / 60, 1),
        "model_path": out_path,
        "best_model_path": f"{best_path}.zip",
        "state_dim": CryptoExecutionEnv.EPISODE_BARS,
        "actions": ["HOLD", "BUY_SMALL", "BUY_LARGE", "SELL_SMALL", "SELL_LARGE"],
    }
    meta_path = "ml/weights/rl_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)
    logger.info("\n=== Next step: update unified_config.yaml ===")
    logger.info("  reinforcement_agent:")
    logger.info("    use_rl_agent: true")
    logger.info("    model_path: \"%s\"", out_path)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ARGUS RL execution agent")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        symbol=args.symbol,
        use_gpu=not args.no_gpu,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        save_dir=args.save_dir,
    )
