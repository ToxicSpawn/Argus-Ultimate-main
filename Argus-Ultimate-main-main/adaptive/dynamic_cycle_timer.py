"""
adaptive/dynamic_cycle_timer.py — Dynamic Cycle Timing

Adjusts the main trading loop interval based on real-time volatility and
the current market regime.  High-volatility or crisis regimes trigger faster
cycles so the system can react more quickly; calm markets allow longer
intervals to conserve resources.

Usage::

    timer = DynamicCycleTimer(baseline_vol=0.01)
    interval = timer.get_cycle_interval(volatility=0.025, regime_label="trending")
    # → 30 (seconds)

Works standalone — no dependency on the rest of ARGUS.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default interval thresholds (seconds)
# ---------------------------------------------------------------------------
_INTERVAL_CRISIS = 15       # Crisis regime — fastest possible
_INTERVAL_HIGH_VOL = 30     # Volatility > 2x baseline
_INTERVAL_NORMAL = 60       # Normal volatility
_INTERVAL_LOW_VOL = 300     # Volatility < 0.5x baseline

# Price move threshold for forced cycle
_FORCE_CYCLE_PCT = 2.0      # Force immediate cycle if price moves > 2%

# EMA smoothing factor for baseline vol
_EMA_ALPHA = 0.05


class DynamicCycleTimer:
    """Adaptive cycle timer that adjusts interval based on market conditions.

    Parameters
    ----------
    baseline_vol : float
        Initial estimate of "normal" volatility (e.g. 0.01 = 1%).
        Updated via exponential moving average as new observations arrive.
    crisis_interval_s : int
        Cycle interval during crisis regime.
    high_vol_interval_s : int
        Cycle interval when volatility > 2x baseline.
    normal_interval_s : int
        Cycle interval under normal conditions.
    low_vol_interval_s : int
        Cycle interval when volatility < 0.5x baseline.
    force_cycle_pct : float
        Price move percentage that triggers an immediate forced cycle.
    ema_alpha : float
        Smoothing factor for the baseline volatility EMA (0 < alpha < 1).
    """

    def __init__(
        self,
        baseline_vol: float = 0.01,
        crisis_interval_s: int = _INTERVAL_CRISIS,
        high_vol_interval_s: int = _INTERVAL_HIGH_VOL,
        normal_interval_s: int = _INTERVAL_NORMAL,
        low_vol_interval_s: int = _INTERVAL_LOW_VOL,
        force_cycle_pct: float = _FORCE_CYCLE_PCT,
        ema_alpha: float = _EMA_ALPHA,
    ) -> None:
        self._baseline_vol = max(baseline_vol, 1e-9)
        self._crisis_s = crisis_interval_s
        self._high_vol_s = high_vol_interval_s
        self._normal_s = normal_interval_s
        self._low_vol_s = low_vol_interval_s
        self._force_pct = force_cycle_pct
        self._ema_alpha = ema_alpha
        self._last_interval: Optional[int] = None
        self._last_price: Optional[float] = None
        self._last_update_ts: float = time.time()

        log.info(
            "DynamicCycleTimer initialised — baseline_vol=%.4f  "
            "intervals=[crisis=%ds, high_vol=%ds, normal=%ds, low_vol=%ds]",
            self._baseline_vol,
            self._crisis_s,
            self._high_vol_s,
            self._normal_s,
            self._low_vol_s,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_cycle_interval(
        self,
        volatility: float,
        regime_label: str = "normal",
    ) -> int:
        """Determine the optimal cycle interval in seconds.

        Parameters
        ----------
        volatility : float
            Current realised volatility (same units as *baseline_vol*).
        regime_label : str
            Current market regime label (e.g. ``"crisis"``, ``"trending"``).

        Returns
        -------
        int
            Recommended cycle interval in seconds.
        """
        regime_label = regime_label.lower().strip()

        # Crisis overrides everything
        if regime_label in ("crisis", "bear"):
            interval = self._crisis_s
        elif self._baseline_vol > 0 and volatility > 2.0 * self._baseline_vol:
            interval = self._high_vol_s
        elif self._baseline_vol > 0 and volatility < 0.5 * self._baseline_vol:
            interval = self._low_vol_s
        else:
            interval = self._normal_s

        if self._last_interval is not None and interval != self._last_interval:
            log.info(
                "DynamicCycleTimer: interval changed %ds → %ds  "
                "(vol=%.4f, baseline=%.4f, regime=%s)",
                self._last_interval,
                interval,
                volatility,
                self._baseline_vol,
                regime_label,
            )
        self._last_interval = interval
        return interval

    # ------------------------------------------------------------------
    # Baseline maintenance
    # ------------------------------------------------------------------

    def update_baseline_vol(self, vol: float) -> float:
        """Update the baseline volatility with a new observation via EMA.

        Parameters
        ----------
        vol : float
            Latest volatility measurement.

        Returns
        -------
        float
            Updated baseline volatility.
        """
        if vol < 0:
            log.warning("DynamicCycleTimer: negative vol=%.4f ignored", vol)
            return self._baseline_vol

        self._baseline_vol = (
            self._ema_alpha * vol + (1.0 - self._ema_alpha) * self._baseline_vol
        )
        self._last_update_ts = time.time()
        log.debug(
            "DynamicCycleTimer: baseline_vol updated to %.6f (latest=%.6f)",
            self._baseline_vol,
            vol,
        )
        return self._baseline_vol

    # ------------------------------------------------------------------
    # Forced cycle detection
    # ------------------------------------------------------------------

    def should_force_cycle(self, price_change_pct: float) -> bool:
        """Return ``True`` if a sudden price move warrants an immediate cycle.

        Parameters
        ----------
        price_change_pct : float
            Absolute price change in percent since last check.

        Returns
        -------
        bool
            ``True`` if the price move exceeds the configured threshold.
        """
        if abs(price_change_pct) >= self._force_pct:
            log.warning(
                "DynamicCycleTimer: FORCE CYCLE — price move %.2f%% exceeds %.2f%% threshold",
                price_change_pct,
                self._force_pct,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def baseline_vol(self) -> float:
        """Current baseline volatility estimate."""
        return self._baseline_vol

    @property
    def last_interval(self) -> Optional[int]:
        """Most recently computed interval, or ``None`` if never called."""
        return self._last_interval
