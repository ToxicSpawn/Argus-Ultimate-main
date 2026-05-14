"""
dip_analyser.py — OctoBot-inspired Dip Analyser trading mode.

Logic
-----
Detects local price bottoms using a configurable lookback window,
then places scaled buy orders at multiple dip depths. Take-profit
targets are set at proportional levels above entry.

OctoBot reference
-----------------
https://www.octobot.cloud/en/guides/octobot-trading-modes/dip-analyser-trading-mode

Key behaviours replicated
-------------------------
- Identify local minimum: close < min(close[-lookback:])
- Risk-scaled position sizing: larger dip = larger position fraction
- Multiple TP levels (TP1, TP2, TP3) with partial exits
- Cooldown after a buy to avoid over-averaging
- Signal output compatible with BaseTentacle EvalResult contract
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .base_tentacle import (
    BaseTentacle, EvalResult, TentacleType, register_tentacle,
    candles_close, candles_low, candles_volume,
)


@dataclass
class DipLevel:
    dip_pct: float        # price drop % that triggers this level
    size_pct: float       # fraction of available capital to deploy
    tp1_pct: float        # first take-profit % above entry
    tp2_pct: float        # second take-profit % above entry
    tp3_pct: float        # third take-profit % above entry


DEFAULT_DIP_LEVELS: List[DipLevel] = [
    DipLevel(dip_pct=2.0,  size_pct=0.10, tp1_pct=1.0, tp2_pct=2.0, tp3_pct=3.5),
    DipLevel(dip_pct=4.0,  size_pct=0.20, tp1_pct=1.5, tp2_pct=3.0, tp3_pct=5.0),
    DipLevel(dip_pct=7.0,  size_pct=0.35, tp1_pct=2.5, tp2_pct=5.0, tp3_pct=8.0),
    DipLevel(dip_pct=12.0, size_pct=0.50, tp1_pct=4.0, tp2_pct=8.0, tp3_pct=12.0),
]


@register_tentacle
class DipAnalyser(BaseTentacle):
    """
    Dip Analyser trading mode tentacle.

    Emits a buy signal (positive) when a local bottom is detected,
    scaled by dip severity. Emits 0 (neutral) otherwise.
    Emits a partial sell signal when price hits a TP level.

    Config keys
    -----------
    lookback        : int   bars to look back for local min detection (default 20)
    cooldown_bars   : int   bars to wait after a buy before buying again (default 5)
    dip_levels      : list  of DipLevel objects (uses DEFAULT_DIP_LEVELS if omitted)
    min_volume_ratio: float minimum volume ratio vs 20-bar avg to confirm dip (default 0.8)
    """

    name = "DipAnalyser"
    tentacle_type = TentacleType.TRADING_MODE
    version = "1.0.0"
    weight = 1.5

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._lookback      = int(self.config.get("lookback", 20))
        self._cooldown_bars = int(self.config.get("cooldown_bars", 5))
        self._dip_levels    = self.config.get("dip_levels", DEFAULT_DIP_LEVELS)
        self._min_vol_ratio = float(self.config.get("min_volume_ratio", 0.8))
        self._bars_since_buy: int = 999
        self._last_buy_price: Optional[float] = None
        self._active_tps: List[Dict] = []

    def evaluate(self, candles: np.ndarray, **kwargs: Any) -> EvalResult:
        if len(candles) < self._lookback + 2:
            return EvalResult(tentacle_name=self.name, signal=0.0, confidence=0.0)

        close = candles_close(candles)
        low   = candles_low(candles)
        current_price = float(close[-1])
        self._bars_since_buy += 1

        tp_signal = self._check_take_profits(current_price)
        if tp_signal != 0.0:
            return EvalResult(
                tentacle_name=self.name, signal=tp_signal, confidence=0.9,
                metadata={"mode": "take_profit", "price": current_price},
            )

        if self._bars_since_buy < self._cooldown_bars:
            return EvalResult(tentacle_name=self.name, signal=0.0,
                              metadata={"mode": "cooldown"})

        window_low  = float(np.min(low[-self._lookback - 1:-1]))
        prev_close  = float(close[-2])
        dip_pct     = (prev_close - current_price) / prev_close * 100.0 if prev_close > 0 else 0.0
        is_local_bottom = current_price <= window_low * 1.005 and dip_pct > 0

        if not is_local_bottom:
            return EvalResult(tentacle_name=self.name, signal=0.0,
                              metadata={"mode": "no_dip", "dip_pct": round(dip_pct, 3)})

        vol_ok        = self._check_volume(candles)
        matched_level = self._match_dip_level(dip_pct)
        if matched_level is None:
            return EvalResult(tentacle_name=self.name, signal=0.0,
                              metadata={"mode": "dip_below_threshold"})

        signal     = min(1.0, dip_pct / 12.0)
        confidence = 0.85 if vol_ok else 0.55
        self._register_tps(current_price, matched_level)
        self._last_buy_price = current_price
        self._bars_since_buy = 0

        return EvalResult(
            tentacle_name=self.name,
            signal=signal,
            confidence=confidence,
            metadata={
                "mode": "dip_buy",
                "dip_pct": round(dip_pct, 3),
                "dip_level": matched_level.dip_pct,
                "size_pct": matched_level.size_pct,
                "tp1": round(current_price * (1 + matched_level.tp1_pct / 100), 4),
                "tp2": round(current_price * (1 + matched_level.tp2_pct / 100), 4),
                "tp3": round(current_price * (1 + matched_level.tp3_pct / 100), 4),
                "volume_confirmed": vol_ok,
            },
        )

    def _match_dip_level(self, dip_pct: float) -> Optional[DipLevel]:
        matched = None
        for level in self._dip_levels:
            if dip_pct >= level.dip_pct:
                matched = level
        return matched

    def _check_volume(self, candles: np.ndarray) -> bool:
        vol = candles_volume(candles)
        if len(vol) < 20:
            return True
        avg_vol     = float(np.mean(vol[-20:-1]))
        current_vol = float(vol[-1])
        return (current_vol / avg_vol) >= self._min_vol_ratio if avg_vol > 0 else True

    def _register_tps(self, entry_price: float, level: DipLevel) -> None:
        self._active_tps = [
            {"target": entry_price * (1 + level.tp1_pct / 100), "fraction": 0.40, "hit": False},
            {"target": entry_price * (1 + level.tp2_pct / 100), "fraction": 0.35, "hit": False},
            {"target": entry_price * (1 + level.tp3_pct / 100), "fraction": 0.25, "hit": False},
        ]

    def _check_take_profits(self, price: float) -> float:
        for tp in self._active_tps:
            if not tp["hit"] and price >= tp["target"]:
                tp["hit"] = True
                return -tp["fraction"]
        return 0.0

    def reset(self) -> None:
        super().reset()
        self._bars_since_buy = 999
        self._last_buy_price = None
        self._active_tps = []
