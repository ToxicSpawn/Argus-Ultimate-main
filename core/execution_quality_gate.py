#!/usr/bin/env python3
"""
Execution Quality Gate — rejects or warns on executions that fail
implementation-shortfall, slippage, or fill-rate thresholds.

Integrates with the execution loop as a pre-submit and post-fill gate.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QualityVerdict:
    allowed: bool
    reason: str = ""
    is_bps: float = 0.0
    slippage_bps: float = 0.0
    fill_rate: float = 1.0


class ExecutionQualityGate:
    """
    Rolling-window quality gate.

    Pre-submit: checks estimated IS against max_avg_is_bps.
    Post-fill:  records realized slippage and fill-rate; blocks future
                orders if rolling averages breach thresholds.
    """

    def __init__(
        self,
        *,
        max_avg_is_bps: float = 0.0,
        max_slippage_bps: float = 0.0,
        min_fill_rate: float = 0.0,
        window: int = 100,
        enabled: bool = True,
    ):
        self.max_avg_is_bps = float(max_avg_is_bps)
        self.max_slippage_bps = float(max_slippage_bps)
        self.min_fill_rate = float(min_fill_rate)
        self.window = max(10, int(window))
        self.enabled = bool(enabled)
        self._is_samples: deque = deque(maxlen=self.window)
        self._slippage_samples: deque = deque(maxlen=self.window)
        self._fill_rate_samples: deque = deque(maxlen=self.window)

    # ---------------------------------------------------------------- pre-submit

    def check_pre_submit(
        self,
        *,
        symbol: str,
        strategy: str = "",
        estimated_is_bps: float = 0.0,
    ) -> QualityVerdict:
        """Call before placing an order. Returns a QualityVerdict."""
        if not self.enabled:
            return QualityVerdict(allowed=True)
        if self.max_avg_is_bps > 0 and estimated_is_bps > self.max_avg_is_bps:
            return QualityVerdict(
                allowed=False,
                reason=f"IS {estimated_is_bps:.1f}bps > max {self.max_avg_is_bps:.1f}bps",
                is_bps=estimated_is_bps,
            )
        if self._is_samples and self.max_avg_is_bps > 0:
            avg_is = sum(self._is_samples) / len(self._is_samples)
            if avg_is > self.max_avg_is_bps:
                return QualityVerdict(
                    allowed=False,
                    reason=f"avg IS {avg_is:.1f}bps > max {self.max_avg_is_bps:.1f}bps",
                    is_bps=avg_is,
                )
        return QualityVerdict(allowed=True, is_bps=estimated_is_bps)

    # ---------------------------------------------------------------- post-fill

    def record_fill(
        self,
        *,
        symbol: str,
        realized_slippage_bps: float,
        fill_rate: float = 1.0,
        is_bps: float = 0.0,
    ) -> None:
        """Record a completed fill for rolling-window tracking."""
        self._is_samples.append(float(is_bps))
        self._slippage_samples.append(float(realized_slippage_bps))
        self._fill_rate_samples.append(max(0.0, min(1.0, float(fill_rate))))

    def check_post_fill(self, *, symbol: str = "") -> QualityVerdict:
        """Check rolling averages after fill recording. Returns verdict."""
        if not self.enabled:
            return QualityVerdict(allowed=True)
        avg_slip = sum(self._slippage_samples) / len(self._slippage_samples) if self._slippage_samples else 0.0
        avg_fill = sum(self._fill_rate_samples) / len(self._fill_rate_samples) if self._fill_rate_samples else 1.0
        if self.max_slippage_bps > 0 and avg_slip > self.max_slippage_bps:
            return QualityVerdict(
                allowed=False,
                reason=f"avg slippage {avg_slip:.1f}bps > max {self.max_slippage_bps:.1f}bps",
                slippage_bps=avg_slip,
                fill_rate=avg_fill,
            )
        if self.min_fill_rate > 0 and avg_fill < self.min_fill_rate:
            return QualityVerdict(
                allowed=False,
                reason=f"avg fill_rate {avg_fill:.2%} < min {self.min_fill_rate:.2%}",
                slippage_bps=avg_slip,
                fill_rate=avg_fill,
            )
        return QualityVerdict(allowed=True, slippage_bps=avg_slip, fill_rate=avg_fill)

    def stats(self) -> Dict[str, Any]:
        n_is = len(self._is_samples)
        n_sl = len(self._slippage_samples)
        n_fr = len(self._fill_rate_samples)
        return {
            "avg_is_bps": sum(self._is_samples) / n_is if n_is else 0.0,
            "avg_slippage_bps": sum(self._slippage_samples) / n_sl if n_sl else 0.0,
            "avg_fill_rate": sum(self._fill_rate_samples) / n_fr if n_fr else 1.0,
            "samples": n_sl,
        }
