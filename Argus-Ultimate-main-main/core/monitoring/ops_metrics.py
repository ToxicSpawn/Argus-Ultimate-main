"""
core/monitoring/ops_metrics.py

OpsMetrics — lightweight operational metrics tracker.

Tracks:
  * Rolling 24-hour fee drag in basis points
  * Portfolio heat (open risk as % of equity)
  * Alert flags for wire_all_on_cycle advisory

Designed to be attached once to the registry in _ensure_tier_patch
and then called on every fill + every cycle snapshot.

Usage::

    ops = OpsMetrics(fee_drag_alert_bps=25.0, heat_limit_pct=0.10)
    # on each fill:
    ops.record_fill(fee_usd=0.15, vol_usd=60.0)
    # on each cycle:
    snap = ops.snapshot(
        open_risk_usd=45.0,
        equity_usd=650.0,
        equity_aud=1000.0,
        capital_tier="MICRO",
    )
    # snap.fee_drag_alert == True if fee drag > threshold
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple


@dataclass(frozen=True)
class OpsSnapshot:
    """Point-in-time snapshot returned by OpsMetrics.snapshot()."""
    fee_drag_24h_bps:   float
    portfolio_heat_pct: float   # open_risk_usd / equity_usd
    heat_utilisation:   float   # heat_pct / heat_limit_pct  (1.0 = at limit)
    fee_drag_alert:     bool
    heat_alert:         bool
    capital_tier:       str
    equity_usd:         float
    equity_aud:         float
    open_risk_usd:      float


class OpsMetrics:
    """
    Thread-safe rolling operational metrics.

    Parameters
    ----------
    fee_drag_alert_bps:
        Alert threshold — if 24-h rolling fee drag exceeds this value
        (in bps), ``OpsSnapshot.fee_drag_alert`` will be True.
    heat_limit_pct:
        Portfolio heat limit (0.0–1.0).  When
        ``open_risk_usd / equity_usd`` exceeds this, ``heat_alert`` fires.
    window_seconds:
        Rolling window for fee-drag calculation (default 86 400 = 24 h).
    """

    _WINDOW = 86_400  # 24 h in seconds

    def __init__(
        self,
        fee_drag_alert_bps: float = 25.0,
        heat_limit_pct: float = 0.10,
        window_seconds: int = _WINDOW,
    ) -> None:
        self._alert_bps    = float(fee_drag_alert_bps)
        self._heat_limit   = float(heat_limit_pct)
        self._window       = int(window_seconds)

        # Deque of (timestamp_float, fee_usd, vol_usd) tuples
        self._fills: Deque[Tuple[float, float, float]] = deque()

        self._last_equity_usd: float = 0.0
        self._last_equity_aud: float = 0.0

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record_fill(
        self,
        fee_usd: float,
        vol_usd: float,
    ) -> None:
        """
        Record a single fill.  Call once per executed trade.

        Parameters
        ----------
        fee_usd:
            Absolute fee paid in USD for this fill.
        vol_usd:
            Gross notional of the fill in USD.
        """
        now = time.time()
        self._fills.append((now, max(0.0, float(fee_usd)), max(0.0, float(vol_usd))))
        self._evict(now)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def snapshot(
        self,
        open_risk_usd: float,
        equity_usd: float,
        equity_aud: float,
        capital_tier: str = "",
    ) -> OpsSnapshot:
        """
        Compute and return a point-in-time snapshot.

        Parameters
        ----------
        open_risk_usd:
            Current total open risk in USD (e.g. sum of notional × stop-loss
            distance for all open positions).
        equity_usd:
            Current portfolio value in USD.
        equity_aud:
            Current portfolio value in AUD.
        capital_tier:
            Tier string for logging ("MICRO" etc.).
        """
        now = time.time()
        self._evict(now)

        equity_usd  = max(1.0, float(equity_usd))
        equity_aud  = max(1.0, float(equity_aud))
        open_risk   = max(0.0, float(open_risk_usd))

        self._last_equity_usd = equity_usd
        self._last_equity_aud = equity_aud

        # Rolling fee drag
        total_fee_usd = sum(f for _, f, _ in self._fills)
        total_vol_usd = sum(v for _, _, v in self._fills)
        if total_vol_usd > 0:
            fee_drag_bps = (total_fee_usd / total_vol_usd) * 10_000.0
        else:
            fee_drag_bps = 0.0

        # Portfolio heat
        heat_pct         = open_risk / equity_usd
        heat_utilisation = heat_pct / self._heat_limit if self._heat_limit > 0 else 0.0

        return OpsSnapshot(
            fee_drag_24h_bps   = round(fee_drag_bps, 4),
            portfolio_heat_pct = round(heat_pct, 4),
            heat_utilisation   = round(min(heat_utilisation, 9.99), 4),
            fee_drag_alert     = fee_drag_bps  > self._alert_bps,
            heat_alert         = heat_pct      > self._heat_limit,
            capital_tier       = capital_tier,
            equity_usd         = round(equity_usd, 2),
            equity_aud         = round(equity_aud, 2),
            open_risk_usd      = round(open_risk, 2),
        )

    @property
    def total_fills(self) -> int:
        """Number of fills in the current rolling window."""
        return len(self._fills)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self, now: float) -> None:
        """Drop entries older than the rolling window."""
        cutoff = now - self._window
        while self._fills and self._fills[0][0] < cutoff:
            self._fills.popleft()
