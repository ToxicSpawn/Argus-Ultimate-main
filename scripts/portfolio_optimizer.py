"""Portfolio optimisation: risk parity plus Black-Litterman-style blending."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AllocationResult:
    weights: dict[str, float]
    expected_return: float
    volatility: float
    concentration: float


class PortfolioOptimizer:
    def __init__(self, symbols: list[str], max_weight: float = 0.45):
        self.symbols = symbols
        self.max_weight = max_weight

    def risk_parity(self, covariance: np.ndarray) -> np.ndarray:
        vol = np.sqrt(np.maximum(np.diag(covariance), 1e-9))
        inv_vol = 1 / vol
        return self._cap_and_normalise(inv_vol / inv_vol.sum())

    def black_litterman(self, market_weights: np.ndarray, covariance: np.ndarray, views: np.ndarray, confidence: float = 0.5) -> np.ndarray:
        implied = covariance @ market_weights
        blended = (1 - confidence) * implied + confidence * views
        raw = np.maximum(blended, 0)
        if raw.sum() == 0:
            raw = self.risk_parity(covariance)
        return self._cap_and_normalise(raw / raw.sum())

    def optimise(self, expected_returns: list[float], covariance: list[list[float]], views: list[float] | None = None) -> AllocationResult:
        cov = np.asarray(covariance, dtype=float)
        exp_ret = np.asarray(expected_returns, dtype=float)
        base = self.risk_parity(cov)
        weights = self.black_litterman(base, cov, np.asarray(views if views is not None else exp_ret), confidence=0.55)
        port_return = float(weights @ exp_ret)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        return AllocationResult(dict(zip(self.symbols, map(float, weights))), port_return, port_vol, float(np.sum(weights ** 2)))

    def _cap_and_normalise(self, weights: np.ndarray) -> np.ndarray:
        weights = np.minimum(np.maximum(weights, 0), self.max_weight)
        if weights.sum() == 0:
            return np.ones(len(self.symbols)) / len(self.symbols)
        return weights / weights.sum()


def _demo() -> None:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    cov = [[0.04, 0.028, 0.02], [0.028, 0.05, 0.024], [0.02, 0.024, 0.09]]
    opt = PortfolioOptimizer(symbols)
    print("Portfolio optimizer ready")
    print(opt.optimise([0.12, 0.14, 0.18], cov, views=[0.10, 0.16, 0.20]))


if __name__ == "__main__":
    _demo()
