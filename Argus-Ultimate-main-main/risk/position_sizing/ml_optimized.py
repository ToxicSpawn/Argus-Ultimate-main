"""
ML-optimized position sizing (import-safe placeholder).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MLOptimizedSizer:
    def calculate(self, capital: float, risk_per_trade: float, confidence: float = 1.0) -> Dict[str, Any]:
        cap = float(capital) if float(capital) > 0 else 1.0
        rpt = max(0.0, float(risk_per_trade))
        conf = max(0.0, float(confidence))
        base_size = cap * rpt
        adjusted_size = base_size * conf
        return {
            "position_size": adjusted_size,
            "pct_of_capital": (adjusted_size / cap) * 100.0,
            "method": "ml_optimized",
        }


# Backwards-compat casing / older name.
MlOptimizedSizer = MLOptimizedSizer

__all__ = ["MLOptimizedSizer", "MlOptimizedSizer"]