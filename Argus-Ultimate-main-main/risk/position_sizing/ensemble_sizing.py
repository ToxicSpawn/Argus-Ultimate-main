"""
Ensemble Position Sizing
Weighted combination of Kelly, volatility, drawdown, and regime sizing methods.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class EnsembleSizer:
    """Combine multiple sizing methods with configurable weights."""

    def __init__(
        self,
        weights: Dict[str, float] = None,
        max_position_pct: float = 0.15,
        disagreement_penalty: bool = True,
    ):
        if weights is None:
            weights = {
                "kelly": 0.30,
                "volatility": 0.25,
                "drawdown": 0.25,
                "regime": 0.20,
            }
        total = sum(weights.values())
        self.weights = {k: v / max(total, 1e-9) for k, v in weights.items()}
        self.max_position_pct = float(max_position_pct)
        self.disagreement_penalty = disagreement_penalty

    def calculate(
        self,
        capital: float,
        risk_per_trade: float,
        confidence: float = 1.0,
        sub_results: Dict[str, Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cap = max(float(capital), 1.0)
        if not sub_results:
            base_size = cap * float(risk_per_trade) * float(confidence)
            return {
                "position_size": min(base_size, cap * self.max_position_pct),
                "pct_of_capital": min(base_size, cap * self.max_position_pct) / cap * 100,
                "method": "ensemble_sizing",
                "sub_methods": {},
                "disagreement": 0.0,
            }

        weighted_size = 0.0
        sizes: List[float] = []
        used_weights: Dict[str, float] = {}

        for method, w in self.weights.items():
            if method in sub_results and "position_size" in sub_results[method]:
                sz = float(sub_results[method]["position_size"])
                weighted_size += sz * w
                sizes.append(sz)
                used_weights[method] = w

        if not sizes:
            base_size = cap * float(risk_per_trade) * float(confidence)
            weighted_size = base_size
            disagreement = 0.0
        else:
            total_w = sum(used_weights.values())
            if total_w > 0 and total_w < 1.0:
                weighted_size /= total_w

            if len(sizes) >= 2:
                cv = float(np.std(sizes) / max(np.mean(sizes), 1e-9))
                disagreement = min(cv, 1.0)
            else:
                disagreement = 0.0

            if self.disagreement_penalty and disagreement > 0.3:
                penalty = max(0.5, 1.0 - (disagreement - 0.3) * 1.0)
                weighted_size *= penalty

        max_size = cap * self.max_position_pct
        final_size = min(weighted_size, max_size)
        final_size = max(final_size, 0.0)

        return {
            "position_size": final_size,
            "pct_of_capital": (final_size / cap) * 100,
            "method": "ensemble_sizing",
            "sub_methods": {k: sub_results.get(k, {}).get("position_size", 0) for k in self.weights},
            "disagreement": disagreement if sizes else 0.0,
            "weights": dict(self.weights),
        }


EnsembleSizingSizer = EnsembleSizer

__all__ = ["EnsembleSizer", "EnsembleSizingSizer"]
