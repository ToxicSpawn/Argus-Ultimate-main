"""Multi-asset regime detection using returns, volatility, and PCA-like breadth."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RegimeState:
    regime: str
    trend: float
    volatility: float
    breadth: float
    confidence: float


class MultiAssetRegimeDetector:
    def detect(self, returns_by_symbol: dict[str, list[float]]) -> RegimeState:
        usable = {k: np.asarray(v[-60:], dtype=float) for k, v in returns_by_symbol.items() if len(v) >= 10}
        if not usable:
            return RegimeState("unknown", 0.0, 0.0, 0.0, 0.0)
        min_len = min(len(v) for v in usable.values())
        matrix = np.vstack([v[-min_len:] for v in usable.values()])
        mean_returns = matrix.mean(axis=1)
        trend = float(np.mean(mean_returns))
        volatility = float(np.mean(np.std(matrix, axis=1)))
        breadth = float(np.mean(mean_returns > 0))
        corr = np.corrcoef(matrix) if matrix.shape[0] > 1 else np.array([[1.0]])
        avg_corr = float(np.nanmean(corr[np.triu_indices_from(corr, k=1)])) if matrix.shape[0] > 1 else 0.0
        if volatility > 0.035:
            regime = "volatile"
        elif trend > 0.004 and breadth > 0.6:
            regime = "bull"
        elif trend < -0.004 and breadth < 0.4:
            regime = "bear"
        elif avg_corr > 0.75:
            regime = "risk_on_clustered"
        else:
            regime = "range"
        confidence = float(np.clip(abs(trend) / 0.01 + volatility / 0.05 + abs(breadth - 0.5), 0, 1))
        return RegimeState(regime, trend, volatility, breadth, confidence)


def _demo() -> None:
    rng = np.random.default_rng(5)
    data = {s: list(rng.normal(0.002, 0.015, 80)) for s in ["BTC", "ETH", "SOL"]}
    print("Multi-asset regime detector ready")
    print(MultiAssetRegimeDetector().detect(data))


if __name__ == "__main__":
    _demo()
