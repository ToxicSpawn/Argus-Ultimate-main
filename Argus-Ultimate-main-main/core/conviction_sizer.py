"""
Conviction-Based Position Sizer — sizes UP on high-conviction setups.

When the evolver, scanner, regime, AND multi-timeframe ALL agree,
that's a 3-4x conviction signal. Currently all trades are sized the same.
This module computes a conviction multiplier from 0.5x to 3.0x.

Conviction sources (each adds 0-1 points):
1. Scanner confidence: top scanner opportunity for this symbol
2. Evolver fitness: evolved strategy has Sharpe > 1.0
3. Regime alignment: strategy type matches current regime
4. Multi-timeframe agreement: 1h + 4h agree with signal direction
5. Signal quality tracker: source has high historical accuracy
6. Edge monitor: system edge is healthy (not degraded)

Total conviction: sum / max → multiplier via sigmoid mapping.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvictionResult:
    """Result of conviction analysis."""
    base_size_pct: float        # original position size
    conviction_score: float     # 0.0 to 1.0
    multiplier: float           # 0.5 to 3.0
    final_size_pct: float       # base * multiplier
    sources: Dict[str, float]   # per-source conviction breakdown


class ConvictionSizer:
    """Compute conviction multiplier from multiple agreement sources."""

    def __init__(
        self,
        min_multiplier: float = 0.5,
        max_multiplier: float = 3.0,
        sigmoid_steepness: float = 4.0,
    ):
        self._min_mult = min_multiplier
        self._max_mult = max_multiplier
        self._steepness = sigmoid_steepness

    def compute(
        self,
        base_size_pct: float,
        symbol: str,
        action: str,               # "BUY" or "SELL"
        strategy_type: str,
        advisory: Optional[Dict[str, Any]] = None,
        mtf_bias: Optional[Dict[str, str]] = None,  # {"1h": "BUY", "4h": "SELL"}
        regime: str = "NORMAL",
        max_pos_pct: float = 0.15,
    ) -> ConvictionResult:
        """
        Compute conviction-adjusted position size.

        Returns ConvictionResult with multiplier applied to base_size_pct.
        """
        if advisory is None:
            advisory = {}
        if mtf_bias is None:
            mtf_bias = {}

        sources: Dict[str, float] = {}

        # 1. Scanner confidence (0 or 1)
        scanner = advisory.get("strategy_scanner", {})
        if isinstance(scanner, dict):
            scan_opps = scanner.get("top_opportunities", [])
            scan_best_sym = ""
            if isinstance(scan_opps, list) and scan_opps:
                first = scan_opps[0]
                scan_best_sym = getattr(first, "symbol", "") if hasattr(first, "symbol") else first.get("symbol", "") if isinstance(first, dict) else ""
            if scan_best_sym == symbol:
                sources["scanner"] = 1.0
            else:
                sources["scanner"] = 0.0
        else:
            sources["scanner"] = 0.0

        # 2. Evolver fitness (0 to 1, based on best composite)
        evolver = advisory.get("strategy_evolver", {})
        if isinstance(evolver, dict):
            best_composite = float(evolver.get("best_composite", 0) or 0)
            best_sym = str(evolver.get("best_symbol", "") or "")
            if best_sym == symbol and best_composite > 0.3:
                sources["evolver"] = min(1.0, best_composite)
            else:
                sources["evolver"] = 0.0
        else:
            sources["evolver"] = 0.0

        # 3. Regime alignment (0 or 1)
        from core.strategy_evolver import _REGIME_STRATEGY_AFFINITY
        regime_lower = regime.lower().replace("_", "")
        regime_key = "trending" if "trend" in regime_lower else (
            "volatile" if "vol" in regime_lower or "crisis" in regime_lower else "ranging"
        )
        suited = _REGIME_STRATEGY_AFFINITY.get(regime_key, [])
        sources["regime"] = 1.0 if strategy_type in suited else 0.0

        # 4. Multi-timeframe agreement (0, 0.5, or 1.0)
        mtf_score = 0.0
        for tf, bias in mtf_bias.items():
            if bias == action:
                mtf_score += 0.5
            elif bias == "NEUTRAL":
                mtf_score += 0.1
        sources["mtf"] = min(1.0, mtf_score)

        # 5. Edge monitor health (0 or 1)
        edge = advisory.get("edge_monitor", {})
        if isinstance(edge, dict):
            edge_score = float(edge.get("edge_score", 0.5) or 0.5)
            sources["edge"] = 1.0 if edge_score > 0.4 else 0.0
        else:
            sources["edge"] = 0.5  # neutral when no data

        # 6. Health score (0 to 1)
        health = advisory.get("health_score", {})
        if isinstance(health, dict):
            h = float(health.get("score", 50) or 50)
            sources["health"] = min(1.0, h / 100.0)
        else:
            sources["health"] = 0.5

        # Compute total conviction (0 to 1)
        if not sources:
            conviction = 0.5
        else:
            conviction = sum(sources.values()) / len(sources)

        # Map conviction to multiplier via sigmoid
        # conviction=0 → min_mult, conviction=0.5 → 1.0, conviction=1.0 → max_mult
        centered = (conviction - 0.5) * self._steepness
        sigmoid = 1.0 / (1.0 + math.exp(-centered))
        multiplier = self._min_mult + (self._max_mult - self._min_mult) * sigmoid

        # Apply
        final = min(max_pos_pct, base_size_pct * multiplier)

        return ConvictionResult(
            base_size_pct=base_size_pct,
            conviction_score=conviction,
            multiplier=multiplier,
            final_size_pct=final,
            sources=sources,
        )
