"""
train_rl_agent.py — PPO reinforcement learning training script.

Trains a Proximal Policy Optimisation (PPO) agent on a trading environment
using Stable-Baselines3. The agent learns to output position signals
{-1 (short), 0 (flat), 1 (long)} to maximise risk-adjusted P&L.

Dependencies
------------
    pip install stable-baselines3 gymnasium numpy pandas

Usage
-----
    python scripts/train_rl_agent.py \
        --symbol XBT/USD \
        --bars 5000 \
        --timesteps 500000 \
        --output models/ppo_argus.zip

Features
--------
  - Custom Gymnasium trading environment with Kraken fee model
  - Observation space: configurable window of OHLCV + technical features
  - Reward: step P&L net of fees, with Sharpe-shaping option
  - PPO with tuned hyperparameters for financial time series
  - Periodic evaluation callback with early stopping on reward plateau
  - Model saved as SB3 zip for direct loading in production
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
        _GYM_AVAILABLE = True
    except ImportError:
        _GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        StopTrainingOnNoModelImprovement,
    )
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KRAKEN_MAKER_FEE = 0.0016
KRAKEN_TAKER_FEE = 0.0040


# ---------------------------------------------------------------------------
# Trading Environment
# ---------------------------------------------------------------------------

class TradingEnv(gym.Env):  # type: ignore[misc]
    """
    Single-asset discrete-action trading environment.

    Observation
    -----------
    Window of `obs_window` bars, each bar containing:
      [open, high, low, close, volume] normalised + RSI + EMA-ratio
    Shape: (obs_window, n_features)

    Actions
    -------
      0 = flat / close position
      1 = long
      2 = short

    Reward
    ------
      Net step P&L (after Kraken maker fee) + optional Sharpe shaping
    """

    metadata = {"render_modes": []}

    N_ACTIONS   = 3
    N_FEATURES  = 7   # open, high, low, close, volume, rsi, ema_ratio

    def __init__(
        self,
        prices: np.ndarray,          # shape (T,) close prices
        ohlcv: Optional[np.ndarray] = None,  # shape (T, 5) OHLCV; uses prices if None
        obs_window: int = 50,
        fee_rate: float = KRAKEN_MAKER_FEE,
        initial_balance: float = 10_000.0,
        sharpe_shaping: bool = True,
        sharpe_window: int = 20,
    ) -> None:
        super().__init__()
        self._prices = np.asarray(prices, dtype=np.float64)
        self._ohlcv  = ohlcv
        self._obs_window = obs_window
        self._fee_rate   = fee_rate
        self._initial_balance = initial_balance
        self._sharpe_shaping  = sharpe_shaping
        self._sharpe_window   = sharpe_window

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_window, self.N_FEATURES),
            dtype=np.float32,
        )

        self._t: int = 0
        self._position: int = 0          # -1, 0, 1
        self._balance: float = initial_balance
        self._returns: list = []
        self._build_features()

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self) -> None:
        T = len(self._prices)
        p = self._prices

        # EMA ratio (fast/slow)
        def ema(arr: np.ndarray, period: int) -> np.ndarray:
            out = np.zeros_like(arr)
            k = 2.0 / (period + 1)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = arr[i] * k + out[i-1] * (1 - k)
            return out

        ema_fast  = ema(p, 10)
        ema_slow  = ema(p, 30)
        ema_ratio = np.where(ema_slow != 0, ema_fast / ema_slow - 1.0, 0.0)

        # RSI
        delta = np.diff(p, prepend=p[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        period = 14
        avg_g = np.convolve(gain, np.ones(period)/period, mode="same")
        avg_l = np.convolve(loss, np.ones(period)/period, mode="same")
        rsi   = 100 - (100 / (1 + avg_g / (avg_l + 1e-10)))
        rsi_n = (rsi - 50.0) / 50.0   # normalise to [-1, 1]

        if self._ohlcv is not None:
            raw = self._ohlcv.astype(np.float64)
        else:
            raw = np.column_stack([p, p, p, p, np.ones(T)])

        # Normalise OHLCV by rolling mean close
        close_mean = np.convolve(p, np.ones(20)/20, mode="same") + 1e-10
        norm_ohlcv = raw / close_mean[:, None]

        self._features = np.column_stack([
            norm_ohlcv,           # 5 cols
            rsi_n,                # 1 col
            ema_ratio,            # 1 col
        ]).astype(np.float32)     # shape (T, 7)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t        = self._obs_window
        self._position = 0
        self._balance  = self._initial_balance
        self._returns  = []
        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        new_pos = action - 1   # 0->-1, 1->0, 2->1
        price_now  = self._prices[self._t]
        price_prev = self._prices[self._t - 1]

        # Price return
        price_ret = (price_now - price_prev) / price_prev

        # Fee on position change
        fee = abs(new_pos - self._position) * self._fee_rate

        # Step P&L
        step_pnl = self._position * price_ret - fee
        self._balance *= (1.0 + step_pnl)
        self._returns.append(step_pnl)
        self._position = new_pos
        self._t += 1

        # Reward
        reward = float(step_pnl)
        if self._sharpe_shaping and len(self._returns) >= self._sharpe_window:
            window_rets = np.array(self._returns[-self._sharpe_window:])
            std = float(np.std(window_rets))
            if std > 0:
                reward += 0.1 * float(np.mean(window_rets)) / std

        terminated = self._t >= len(self._prices) - 1
        truncated  = False
        info = {
            "balance": self._balance,
            "position": self._position,
            "step_pnl": step_pnl,
        }
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        start = self._t - self._obs_window
        return self._features[start:self._t].copy()

    def render(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Price loader stub
# ---------------------------------------------------------------------------

def load_prices(symbol: str, bars: int) -> np.ndarray:
    logger.warning(
        "Using synthetic price data for %s (%d bars). "
        "Replace load_prices() with a real data source.",
        symbol, bars,
    )
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0, 0.01, bars)
    return 30000.0 * np.cumprod(1 + returns)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

PPO_HYPERPARAMS: Dict[str, Any] = {
    "learning_rate"    : 3e-4,
    "n_steps"          : 2048,
    "batch_size"       : 64,
    "n_epochs"         : 10,
    "gamma"            : 0.99,
    "gae_lambda"       : 0.95,
    "clip_range"       : 0.2,
    "ent_coef"         : 0.01,
    "vf_coef"          : 0.5,
    "max_grad_norm"    : 0.5,
    "policy_kwargs"    : {"net_arch": [256, 256]},
}


def train(
    prices: np.ndarray,
    total_timesteps: int,
    output_path: str,
    eval_freq: int = 10_000,
    obs_window: int = 50,
    fee_rate: float = KRAKEN_MAKER_FEE,
    verbose: int = 1,
) -> None:
    if not _SB3_AVAILABLE:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3")
        sys.exit(1)
    if not _GYM_AVAILABLE:
        logger.error("gymnasium not installed. Run: pip install gymnasium")
        sys.exit(1)

    # Split 80/20 train/eval
    split = int(len(prices) * 0.8)
    train_prices = prices[:split]
    eval_prices  = prices[split:]

    def make_train_env():
        env = TradingEnv(train_prices, obs_window=obs_window, fee_rate=fee_rate)
        return Monitor(env)

    def make_eval_env():
        env = TradingEnv(eval_prices, obs_window=obs_window, fee_rate=fee_rate)
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])
    eval_env  = DummyVecEnv([make_eval_env])

    # Validate env
    check_env(make_train_env(), warn=True)

    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=20,
        verbose=verbose,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(output_path).parent),
        log_path=str(Path(output_path).parent / "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        callback_after_eval=stop_cb,
        verbose=verbose,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        **PPO_HYPERPARAMS,
    )

    logger.info(
        "Training PPO agent for %d timesteps on %d bars (fee=%.4f%%)...",
        total_timesteps, len(train_prices), fee_rate * 100,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save(output_path)
    logger.info("Model saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO RL agent trainer for Argus Ultimate")
    p.add_argument("--symbol",      default="XBT/USD",                   help="Trading pair")
    p.add_argument("--bars",        type=int,   default=5000,             help="Price history bars")
    p.add_argument("--timesteps",   type=int,   default=500_000,          help="PPO training timesteps")
    p.add_argument("--obs-window",  type=int,   default=50,               help="Observation window")
    p.add_argument("--fee-rate",    type=float, default=KRAKEN_MAKER_FEE, help="Maker fee rate")
    p.add_argument("--eval-freq",   type=int,   default=10_000,           help="Eval callback frequency")
    p.add_argument("--output",      default="models/ppo_argus.zip",       help="Output model path")
    p.add_argument("--verbose",     type=int,   default=1)
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prices = load_prices(args.symbol, args.bars)
    train(
        prices=prices,
        total_timesteps=args.timesteps,
        output_path=args.output,
        eval_freq=args.eval_freq,
        obs_window=args.obs_window,
        fee_rate=args.fee_rate,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
