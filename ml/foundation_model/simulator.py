"""Closed-loop market simulation for foundation-model-generated order flow."""

# pyright: reportMissingImports=false

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SimulationConfig:
    """Controls synthetic order-flow generation."""

    steps: int = 512
    base_price: float = 100.0
    base_spread_bps: float = 5.0
    average_trade_size: float = 1.0
    regime: str = "neutral"
    volatility: float = 0.01
    random_seed: int = 42
    validate_returns_autocorrelation_lag: int = 1


@dataclass(slots=True)
class SimulationResult:
    """Synthetic order flow and stylized fact diagnostics."""

    events: List[Dict[str, Any]]
    diagnostics: Dict[str, float]
    final_mid_price: float
    regime: str


class ClosedLoopMarketSimulator:
    """Simulates order flow and feeds it back into a simple order-book state."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self._last_trade_sign: float = 1.0

    def _regime_drift(self) -> float:
        return {"bull": 0.25, "bear": -0.25, "volatile": 0.0, "neutral": 0.0}.get(self.config.regime, 0.0)

    def generate_synthetic_order_flow(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        mid_price = self.config.base_price
        buy_depth = 100.0
        sell_depth = 100.0
        volatility = max(self.config.volatility, 1e-6)
        drift = self._regime_drift()

        for step in range(self.config.steps):
            shock = self.rng.standard_t(df=4) * volatility
            clustering = 0.85 + 0.15 * self.rng.random()
            shock *= clustering
            mid_price *= max(0.5, 1.0 + shock + drift * 1e-3)

            spread_bps = max(1.0, self.config.base_spread_bps * (1.0 + abs(shock) * 10.0))
            spread = mid_price * spread_bps / 10000.0
            order_type = self.rng.choice(["limit", "market", "cancel"], p=[0.55, 0.3, 0.15])
            side = self.rng.choice(["buy", "sell"])
            size = max(0.01, self.config.average_trade_size * math.exp(self.rng.normal(0.0, 0.5)))
            signed_size = size if side == "buy" else -size
            self._last_trade_sign = 0.8 * self._last_trade_sign + 0.2 * (1.0 if signed_size >= 0.0 else -1.0)

            if order_type == "market":
                if side == "buy":
                    sell_depth = max(1.0, sell_depth - size)
                else:
                    buy_depth = max(1.0, buy_depth - size)
            elif order_type == "limit":
                if side == "buy":
                    buy_depth += size
                else:
                    sell_depth += size
            else:
                if side == "buy":
                    buy_depth = max(1.0, buy_depth - size * 0.5)
                else:
                    sell_depth = max(1.0, sell_depth - size * 0.5)

            bid = mid_price - spread / 2.0
            ask = mid_price + spread / 2.0
            events.append(
                {
                    "timestamp": float(step),
                    "price": mid_price,
                    "mid_price": mid_price,
                    "bid": bid,
                    "ask": ask,
                    "spread": spread,
                    "spread_bps": spread_bps,
                    "size": size,
                    "signed_size": signed_size,
                    "order_type": order_type,
                    "side": side,
                    "buy_depth": buy_depth,
                    "sell_depth": sell_depth,
                    "order_flow_imbalance": (buy_depth - sell_depth) / max(buy_depth + sell_depth, 1e-8),
                    "regime": self.config.regime,
                }
            )
        return events

    @staticmethod
    def validate_stylized_facts(events: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(events) < 4:
            return {
                "volatility_clustering": 0.0,
                "heavy_tail_kurtosis": 0.0,
                "mean_spread_bps": 0.0,
                "returns_autocorrelation": 0.0,
                "imbalance_mean": 0.0,
            }
        prices = np.asarray([float(event["mid_price"]) for event in events], dtype=np.float64)
        returns = np.diff(np.log(np.maximum(prices, 1e-8)))
        abs_returns = np.abs(returns)
        if len(abs_returns) > 1:
            vol_cluster = float(np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1])
        else:
            vol_cluster = 0.0
        centered = returns - returns.mean()
        variance = float(np.mean(centered ** 2))
        kurtosis = 0.0
        if variance > 0.0:
            kurtosis = float(np.mean(centered ** 4) / (variance ** 2))
        mean_spread = float(np.mean([float(event.get("spread_bps", 0.0)) for event in events]))
        return_autocorr = 0.0
        if len(returns) > 2:
            return_autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        imbalance_mean = float(np.mean([float(event.get("order_flow_imbalance", 0.0)) for event in events]))
        return {
            "volatility_clustering": vol_cluster if math.isfinite(vol_cluster) else 0.0,
            "heavy_tail_kurtosis": kurtosis,
            "mean_spread_bps": mean_spread,
            "returns_autocorrelation": return_autocorr if math.isfinite(return_autocorr) else 0.0,
            "imbalance_mean": imbalance_mean,
        }

    def run(self) -> SimulationResult:
        events = self.generate_synthetic_order_flow()
        diagnostics = self.validate_stylized_facts(events)
        final_mid_price = float(events[-1]["mid_price"]) if events else self.config.base_price
        logger.info(
            "Simulation complete regime=%s final_mid=%.4f vol_cluster=%.4f kurtosis=%.4f",
            self.config.regime,
            final_mid_price,
            diagnostics.get("volatility_clustering", 0.0),
            diagnostics.get("heavy_tail_kurtosis", 0.0),
        )
        return SimulationResult(events=events, diagnostics=diagnostics, final_mid_price=final_mid_price, regime=self.config.regime)
