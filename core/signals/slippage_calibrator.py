"""Push 96 — Slippage model calibrator (v8.32.0).

Calibrates actual historical slippage from TradeLedger fills
per-symbol per-side, replacing the ATR-scaled proxy in fee_adjuster.py.

Design:
  SlippageSample       dataclass
  SlippageCalibrator   per-symbol per-side rolling regression
  SlippageEstimator    high-level: estimate_bps(symbol, side, qty)
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import DefaultDict, Deque, Dict, Optional, Tuple


@dataclass
class SlippageSample:
    """A single slippage observation from a live fill."""
    symbol:     str
    side:       str    # "BUY" or "SELL"
    qty:        float  # order quantity
    notional:   float  # order notional in quote currency
    spread_bps: float  # spread at time of order
    vol_ratio:  float  # short/long vol at time of order
    slippage_bps: float  # actual observed slippage (mid to fill)
    ts:         float = field(default_factory=time.time)


class SlippageCalibrator:
    """Per-(symbol, side) online slippage calibrator.

    Maintains a rolling window of samples and fits a simple linear
    regression:  slippage_bps = a + b * spread_bps + c * vol_ratio

    Coefficients are updated every `retrain_every` samples.
    Falls back to spread/2 heuristic until enough data.
    """

    def __init__(self, window: int = 500, retrain_every: int = 50) -> None:
        self._buffers: DefaultDict[
            Tuple[str, str], Deque[SlippageSample]
        ] = defaultdict(lambda: deque(maxlen=window))
        self._coefs: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self._since_retrain: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self._retrain_every = retrain_every

    def record(self, sample: SlippageSample) -> None:
        key = (sample.symbol, sample.side)
        self._buffers[key].append(sample)
        self._since_retrain[key] += 1
        if self._since_retrain[key] >= self._retrain_every and len(self._buffers[key]) >= 20:
            self._fit(key)
            self._since_retrain[key] = 0

    def estimate(self, symbol: str, side: str, spread_bps: float, vol_ratio: float) -> float:
        """Return estimated slippage in bps."""
        key = (symbol, side)
        if key in self._coefs:
            a, b, c = self._coefs[key]
            return max(0.0, a + b * spread_bps + c * vol_ratio)
        return spread_bps * 0.5  # default half-spread heuristic

    @property
    def known_pairs(self) -> list:
        return list(self._coefs.keys())

    def _fit(self, key: Tuple[str, str]) -> None:
        """Ordinary least squares on [1, spread_bps, vol_ratio] -> slippage_bps."""
        samples = list(self._buffers[key])
        n = len(samples)
        if n < 20:
            return
        # Build matrices manually (no numpy required)
        # X cols: [1, spread_bps, vol_ratio]
        # y: slippage_bps
        sx, ss, sv, sy = 0.0, 0.0, 0.0, 0.0
        sss, ssv, svv = 0.0, 0.0, 0.0
        sxs, sxv, sxy = 0.0, 0.0, 0.0
        ssy, svy = 0.0, 0.0
        for s in samples:
            sx  += 1.0
            ss  += s.spread_bps
            sv  += s.vol_ratio
            sy  += s.slippage_bps
            sss += s.spread_bps ** 2
            ssv += s.spread_bps * s.vol_ratio
            svv += s.vol_ratio ** 2
            ssy += s.spread_bps * s.slippage_bps
            svy += s.vol_ratio  * s.slippage_bps
        # Normal equations via 3x3 solve (Cramer's rule)
        # Fallback to simple mean if singular
        mean_slip = sy / n
        mean_spd  = ss / n
        mean_vol  = sv / n
        b = (ssy - ss * sy / n) / max(1e-9, sss - ss * ss / n)
        c = (svy - sv * sy / n) / max(1e-9, svv - sv * sv / n)
        a = mean_slip - b * mean_spd - c * mean_vol
        self._coefs[key] = (max(0.0, a), max(0.0, b), max(0.0, c))


class SlippageEstimator:
    """High-level slippage estimator used by ExecutionEngine.

    Drop-in replacement for the ATR-scaled proxy in fee_adjuster.py.
    """

    def __init__(self) -> None:
        self._cal = SlippageCalibrator()

    def ingest_fill(
        self,
        symbol:       str,
        side:         str,
        qty:          float,
        notional:     float,
        spread_bps:   float,
        vol_ratio:    float,
        intended_mid: float,
        actual_fill:  float,
    ) -> None:
        """Record a completed fill to calibrate the model."""
        slippage_bps = abs(actual_fill - intended_mid) / intended_mid * 10_000
        self._cal.record(SlippageSample(
            symbol=symbol, side=side, qty=qty,
            notional=notional, spread_bps=spread_bps,
            vol_ratio=vol_ratio, slippage_bps=slippage_bps,
        ))

    def estimate_bps(self, symbol: str, side: str, spread_bps: float, vol_ratio: float) -> float:
        """Return calibrated slippage estimate in bps."""
        return self._cal.estimate(symbol, side, spread_bps, vol_ratio)

    @property
    def stats(self) -> dict:
        return {"known_pairs": self._cal.known_pairs}
