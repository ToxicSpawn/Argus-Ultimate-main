"""Gym-compatible trading environment with multimodal observations and costs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

class _FallbackEnv:
    metadata: dict[str, object] = {}

    def __init__(self) -> None:
        self.np_random = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[None, dict[str, object]]:
        self.np_random = np.random.default_rng(seed)
        return None, {}


class _FallbackBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: object) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class _FallbackSpaces:
    Box = _FallbackBox


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:  # pragma: no cover - lightweight fallback
        spaces = _FallbackSpaces()

from .regime_detector import RegimeDetector
from .volatility_adapter import VolatilityAdapter


@dataclass(slots=True)
class EnvironmentConfig:
    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    max_leverage: float = 1.0
    max_drawdown: float = 0.25
    reward_scale: float = 1.0
    allow_short: bool = True


class TradingEnvironment(_FallbackEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_series: np.ndarray,
        technical_features: np.ndarray | None = None,
        sentiment_features: np.ndarray | None = None,
        order_flow_features: np.ndarray | None = None,
        regime_detector: RegimeDetector | None = None,
        volatility_adapter: VolatilityAdapter | None = None,
        config: EnvironmentConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.price_series = self._validate_array(price_series, "price_series")
        self.length = self.price_series.shape[0]
        if self.length < 3:
            raise ValueError("price_series must contain at least 3 rows")
        self.technical_features = self._aligned_features(technical_features, name="technical_features")
        self.sentiment_features = self._aligned_features(sentiment_features, name="sentiment_features")
        self.order_flow_features = self._aligned_features(order_flow_features, name="order_flow_features")
        self.regime_detector = regime_detector or RegimeDetector()
        self.volatility_adapter = volatility_adapter or VolatilityAdapter()
        self.num_assets = int(self.price_series.shape[1])
        feature_dim = self.num_assets
        for feature_block in (self.technical_features, self.sentiment_features, self.order_flow_features):
            if feature_block is not None:
                feature_dim += int(feature_block.shape[1])
        feature_dim += self.num_assets + 3
        low = -1.0 if self.config.allow_short else 0.0
        self.action_space = spaces.Box(low=low, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32)
        self._reset_portfolio_state()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_portfolio_state()
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.step_index >= self.length - 2:
            return self._observation(), 0.0, True, False, {"regime": self.regime_detector.current_regime().label}
        target_weights = np.asarray(action, dtype=np.float32).reshape(self.num_assets)
        target_weights = np.clip(target_weights, self.action_space.low, self.action_space.high)
        prices_now = self.price_series[self.step_index]
        prices_next = self.price_series[self.step_index + 1]
        if np.any(prices_now <= 0) or np.any(prices_next <= 0):
            raise ValueError("price_series must contain strictly positive prices")
        regime = self.regime_detector.update(float(np.mean(prices_now)))
        volatility = regime.realized_volatility
        adjusted_weights = self.volatility_adapter.adapt_position(target_weights, volatility, regime.label)
        adjusted_weights = adjusted_weights * min(1.0, self.config.max_leverage / max(float(np.sum(np.abs(adjusted_weights))), 1.0))
        turnover = float(np.sum(np.abs(adjusted_weights - self.positions)))
        transaction_cost = turnover * self.config.transaction_cost_bps / 10_000.0
        slippage_cost = turnover * self.config.slippage_bps / 10_000.0
        asset_returns = (prices_next - prices_now) / np.maximum(prices_now, 1e-8)
        gross_return = float(np.dot(adjusted_weights, asset_returns))
        net_reward = self.config.reward_scale * (gross_return - transaction_cost - slippage_cost)
        shaped_reward = self.volatility_adapter.shape_reward(net_reward, volatility, turnover, regime.label)
        self.positions = adjusted_weights.astype(np.float32)
        self.equity *= 1.0 + net_reward
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.equity - self.peak_equity) / max(self.peak_equity, 1e-8)
        self.step_index += 1
        terminated = self.step_index >= self.length - 2 or drawdown <= -self.config.max_drawdown
        info = {
            "regime": regime.label,
            "trend_regime": regime.trend.value,
            "volatility": volatility,
            "turnover": turnover,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "slippage_cost": slippage_cost,
            "equity": self.equity,
            "drawdown": drawdown,
        }
        return self._observation(), float(shaped_reward), terminated, False, info

    def render(self) -> None:  # pragma: no cover - convenience method
        logger.info("step=%d equity=%.2f positions=%s", self.step_index, self.equity, self.positions.tolist())

    def _observation(self) -> np.ndarray:
        blocks = [self.price_series[self.step_index].astype(np.float32)]
        for feature_block in (self.technical_features, self.sentiment_features, self.order_flow_features):
            if feature_block is not None:
                blocks.append(feature_block[self.step_index].astype(np.float32))
        blocks.append(self.positions.astype(np.float32))
        current_regime = self.regime_detector.current_regime()
        blocks.append(
            np.asarray(
                [
                    current_regime.probability,
                    current_regime.realized_volatility,
                    self.equity / max(self.config.initial_cash, 1e-8) - 1.0,
                ],
                dtype=np.float32,
            )
        )
        return np.concatenate(blocks, dtype=np.float32)

    def _reset_portfolio_state(self) -> None:
        self.step_index = 0
        self.equity = float(self.config.initial_cash)
        self.peak_equity = float(self.config.initial_cash)
        self.positions = np.zeros(self.num_assets, dtype=np.float32)

    def _aligned_features(self, values: np.ndarray | None, name: str) -> np.ndarray | None:
        if values is None:
            return None
        array = self._validate_array(values, name)
        if array.shape[0] != self.length:
            raise ValueError(f"{name} must have the same number of rows as price_series")
        return array

    @staticmethod
    def _validate_array(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 1D or 2D numeric array")
        return array
