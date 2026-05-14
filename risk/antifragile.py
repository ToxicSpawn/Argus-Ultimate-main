"""
Anti-Fragile Position Manager for ARGUS.

Benefits from volatility rather than just surviving it. Tracks P&L at
different volatility levels to determine if the system is fragile, robust,
or antifragile, and adjusts position sizing accordingly.

Usage:
    manager = AntifragileManager()
    manager.record(volatility=0.02, pnl=15.0)
    manager.record(volatility=0.08, pnl=25.0)
    multiplier = manager.get_position_multiplier(current_vol=0.06)
    score = manager.get_fragility_score()
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Volatility buckets
# ---------------------------------------------------------------------------
_VOL_BUCKETS = [
    ("very_low", 0.0, 0.01),
    ("low", 0.01, 0.02),
    ("medium", 0.02, 0.04),
    ("high", 0.04, 0.08),
    ("very_high", 0.08, float("inf")),
]


def _get_vol_bucket(volatility: float) -> str:
    """Return the volatility bucket label for a given volatility."""
    for label, lo, hi in _VOL_BUCKETS:
        if lo <= volatility < hi:
            return label
    return "very_high"


# ---------------------------------------------------------------------------
# AntifragileManager
# ---------------------------------------------------------------------------


class AntifragileManager:
    """
    Tracks P&L at different volatility levels and adjusts position sizing
    to benefit from volatility.

    Parameters
    ----------
    max_history : int
        Maximum observations to retain (default 5000).
    min_observations : int
        Minimum observations per bucket before making recommendations (default 10).
    """

    def __init__(
        self,
        max_history: int = 5000,
        min_observations: int = 10,
    ) -> None:
        self._max_history = max(100, int(max_history))
        self._min_obs = max(3, int(min_observations))

        self._vol_history: List[Tuple[float, float]] = []  # (volatility, pnl)
        self._pnl_at_vol: Dict[str, List[float]] = defaultdict(list)

    # ── Recording ─────────────────────────────────────────────────────────

    def record(self, volatility: float, pnl: float) -> None:
        """Record P&L at a given volatility level."""
        vol = float(volatility)
        p = float(pnl)

        self._vol_history.append((vol, p))
        if len(self._vol_history) > self._max_history:
            self._vol_history = self._vol_history[-self._max_history:]

        bucket = _get_vol_bucket(vol)
        self._pnl_at_vol[bucket].append(p)

        # Trim per-bucket history
        for b in self._pnl_at_vol:
            if len(self._pnl_at_vol[b]) > self._max_history:
                self._pnl_at_vol[b] = self._pnl_at_vol[b][-self._max_history:]

    # ── Vol-PnL curve ─────────────────────────────────────────────────────

    def get_vol_pnl_curve(self) -> dict:
        """
        Return P&L by volatility bucket.

        Shows if system is fragile/robust/antifragile:
        - antifragile = makes MORE money in high vol
        - fragile = loses money in high vol
        - robust = makes similar money in all vol levels

        Returns
        -------
        dict of {bucket_label: {avg_pnl, total_pnl, count, win_rate}}
        """
        result = {}
        for label, lo, hi in _VOL_BUCKETS:
            pnls = self._pnl_at_vol.get(label, [])
            if not pnls:
                result[label] = {
                    "avg_pnl": 0.0,
                    "total_pnl": 0.0,
                    "count": 0,
                    "win_rate": 0.0,
                }
                continue

            arr = np.array(pnls, dtype=float)
            wins = int(np.sum(arr > 0))
            result[label] = {
                "avg_pnl": round(float(np.mean(arr)), 4),
                "total_pnl": round(float(np.sum(arr)), 4),
                "count": len(pnls),
                "win_rate": round(wins / len(pnls), 4) if pnls else 0.0,
            }
        return result

    # ── Position multiplier ───────────────────────────────────────────────

    def get_position_multiplier(self, current_vol: float) -> float:
        """
        Get position size multiplier based on historical vol-pnl relationship.

        If historically profitable in high vol: increase size (up to 1.5x)
        If historically losing in high vol: decrease size (down to 0.5x)
        If insufficient data: return 1.0
        """
        bucket = _get_vol_bucket(current_vol)
        pnls = self._pnl_at_vol.get(bucket, [])

        if len(pnls) < self._min_obs:
            return 1.0

        avg_pnl = float(np.mean(pnls))
        win_rate = float(np.sum(np.array(pnls) > 0)) / len(pnls)

        # Score: combination of avg_pnl direction and win rate
        # Positive avg_pnl + high win rate = increase size
        if avg_pnl > 0 and win_rate > 0.55:
            # Scale up: 1.0 to 1.5 based on win rate
            multiplier = 1.0 + min(0.5, (win_rate - 0.5) * 2.0)
        elif avg_pnl < 0 or win_rate < 0.40:
            # Scale down: 1.0 to 0.5 based on how bad performance is
            loss_severity = min(1.0, max(0.0, 0.5 - win_rate))
            multiplier = max(0.5, 1.0 - loss_severity)
        else:
            multiplier = 1.0

        return round(float(np.clip(multiplier, 0.5, 1.5)), 4)

    # ── Fragility score ───────────────────────────────────────────────────

    def get_fragility_score(self) -> float:
        """
        Compute fragility score:
        -1.0 = fragile (loses in vol)
         0.0 = robust (unaffected by vol)
        +1.0 = antifragile (profits from vol)
        """
        # Compare low-vol performance with high-vol performance
        low_vol_pnls = (
            self._pnl_at_vol.get("very_low", [])
            + self._pnl_at_vol.get("low", [])
        )
        high_vol_pnls = (
            self._pnl_at_vol.get("high", [])
            + self._pnl_at_vol.get("very_high", [])
        )

        if len(low_vol_pnls) < self._min_obs or len(high_vol_pnls) < self._min_obs:
            return 0.0  # Insufficient data

        avg_low = float(np.mean(low_vol_pnls))
        avg_high = float(np.mean(high_vol_pnls))

        # Normalize the difference
        combined_std = float(np.std(low_vol_pnls + high_vol_pnls))
        if combined_std < 1e-10:
            return 0.0

        # Score: (high_vol_avg - low_vol_avg) / std
        raw_score = (avg_high - avg_low) / combined_std

        # Clip to [-1, 1]
        return round(float(np.clip(raw_score, -1.0, 1.0)), 4)

    # ── Strategy recommendation ───────────────────────────────────────────

    def recommend_vol_strategy(self) -> dict:
        """
        Based on fragility profile, recommend strategy adjustments.

        Returns
        -------
        dict with: fragility_score, category, recommendations
        """
        score = self.get_fragility_score()

        if score < -0.3:
            category = "fragile"
            recommendations = [
                "Reduce position size in high volatility periods",
                "Add tail hedges (OTM puts or vol products)",
                "Tighten stop losses during vol spikes",
                "Increase cash allocation in high-vol regimes",
            ]
        elif score > 0.3:
            category = "antifragile"
            recommendations = [
                "Increase position size during vol spikes",
                "Maintain current strategy — system benefits from disorder",
                "Consider selling puts during vol spikes (collect premium)",
                "Reduce hedging costs — natural vol profitability provides protection",
            ]
        else:
            category = "robust"
            recommendations = [
                "System is vol-neutral — maintain current sizing",
                "Monitor for changes in vol-PnL relationship",
                "Consider adding vol-targeting for consistent exposure",
            ]

        return {
            "fragility_score": score,
            "category": category,
            "recommendations": recommendations,
            "vol_pnl_curve": self.get_vol_pnl_curve(),
        }

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for dashboard/logging."""
        return {
            "total_observations": len(self._vol_history),
            "fragility_score": self.get_fragility_score(),
            "vol_pnl_curve": self.get_vol_pnl_curve(),
        }
