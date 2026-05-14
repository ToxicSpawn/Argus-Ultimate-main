"""
execution/session_spread_schedule.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Session-aware dynamic spread scheduling tied to UTC liquidity profile.

Liquidity zones (UTC hours)
---------------------------
  peak_liquidity  : 13–17  → multiplier 0.7  (tight; high BTC/USD activity)
  european_open   : 08–12  → multiplier 0.9
  dead_zone       : 00–05  → multiplier 1.8  (wide; thin Asian overnight)
  normal          : all other hours → multiplier 1.0

Transitions between zones are linearly interpolated so there are no hard
step-changes in quoted spreads.

Override mechanism
------------------
`set_override(multiplier, duration_s)` locks in a specific multiplier for a
fixed duration (e.g., triggered externally when VPIN spikes).  `clear_override()`
removes it early.

Per-symbol base spreads
-----------------------
`register_symbol_override(symbol, base_spread_bps)` stores a per-symbol base
spread so `get_spread_bps(symbol)` returns symbol-specific values.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Zone definitions (immutable class constants)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _LiquidityZone:
    name: str
    start_hour_utc: float   # inclusive start (fractional hours OK)
    end_hour_utc: float     # exclusive end
    multiplier: float


class SessionSpreadSchedule:
    """
    Dynamically adjusts spread multipliers based on UTC session liquidity.

    Parameters
    ----------
    base_spread_bps : float
        Default base spread in basis points (used when no per-symbol override
        is registered).
    """

    # ------------------------------------------------------------------
    # Class-level zone definitions
    # ------------------------------------------------------------------

    PEAK_LIQUIDITY = _LiquidityZone(
        name="peak_liquidity",
        start_hour_utc=13.0,
        end_hour_utc=17.0,
        multiplier=0.7,
    )
    EUROPEAN_OPEN = _LiquidityZone(
        name="european_open",
        start_hour_utc=8.0,
        end_hour_utc=12.0,
        multiplier=0.9,
    )
    DEAD_ZONE = _LiquidityZone(
        name="dead_zone",
        start_hour_utc=0.0,
        end_hour_utc=5.0,
        multiplier=1.8,
    )
    NORMAL = _LiquidityZone(
        name="normal",
        start_hour_utc=5.0,
        end_hour_utc=8.0,
        multiplier=1.0,
    )

    # Ordered list used for interpolation lookups
    ZONES: Tuple[_LiquidityZone, ...] = (
        DEAD_ZONE,
        NORMAL,
        EUROPEAN_OPEN,
        PEAK_LIQUIDITY,
    )

    # Transition ramp duration in hours on each side of a zone boundary
    TRANSITION_RAMP_H: float = 0.5  # 30-minute linear ramp

    # Hours 17–24 are "normal" territory, treated as two implicit segments:
    # 17–24 and there's also 5–8 labelled NORMAL above.
    _IMPLICIT_NORMAL_AFTER_PEAK = _LiquidityZone(
        name="normal",
        start_hour_utc=17.0,
        end_hour_utc=24.0,
        multiplier=1.0,
    )

    def __init__(self, base_spread_bps: float = 5.0) -> None:
        self._base_spread_bps = base_spread_bps
        self._lock = threading.RLock()

        # Per-symbol base spread overrides
        self._symbol_base_spreads: Dict[str, float] = {}

        # Timed override state
        self._override_multiplier: Optional[float] = None
        self._override_expires_at: float = 0.0  # epoch seconds

    # ------------------------------------------------------------------
    # Core multiplier computation
    # ------------------------------------------------------------------

    def _utc_hour_now(self) -> float:
        """Return the current UTC time as a fractional hour (0.0 – 24.0)."""
        t = time.gmtime()
        return t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0

    def _multiplier_for_hour(self, utc_hour: float) -> float:
        """Compute the spread multiplier for a given UTC hour using smooth interpolation.

        The algorithm works by computing a weighted blend across all zone
        multipliers based on proximity.  Within a zone's core (more than
        TRANSITION_RAMP_H from its boundaries) the zone's multiplier dominates
        completely.  Near a boundary, the adjacent zone's multiplier is blended
        in linearly.
        """
        # Build the full 24-hour segment list including the implicit normal
        all_zones: List[_LiquidityZone] = [
            self.DEAD_ZONE,
            self.NORMAL,
            self.EUROPEAN_OPEN,
            self.PEAK_LIQUIDITY,
            self._IMPLICIT_NORMAL_AFTER_PEAK,
        ]

        ramp = self.TRANSITION_RAMP_H

        # Find which zone we are in
        current_zone: Optional[_LiquidityZone] = None
        for z in all_zones:
            if z.start_hour_utc <= utc_hour < z.end_hour_utc:
                current_zone = z
                break

        if current_zone is None:
            # Midnight wrap-around edge case (utc_hour == 24.0)
            return 1.0

        zone_start = current_zone.start_hour_utc
        zone_end = current_hour_end = current_zone.end_hour_utc
        zone_mult = current_zone.multiplier

        # Position within zone [0, duration]
        position = utc_hour - zone_start
        duration = zone_end - zone_start

        if duration <= 0:
            return zone_mult

        # Distance from each boundary
        dist_from_start = position
        dist_from_end = duration - position

        # --- Blend with previous zone near start boundary ---
        if dist_from_start < ramp:
            t = dist_from_start / ramp  # 0 at boundary, 1 at ramp distance
            prev_mult = self._prev_zone_multiplier(all_zones, zone_start)
            return _lerp(prev_mult, zone_mult, t)

        # --- Blend with next zone near end boundary ---
        if dist_from_end < ramp:
            t = (ramp - dist_from_end) / ramp  # 0 at ramp distance, 1 at boundary
            next_mult = self._next_zone_multiplier(all_zones, zone_end)
            return _lerp(zone_mult, next_mult, t)

        # Deep inside zone: return zone multiplier exactly
        return zone_mult

    def _prev_zone_multiplier(
        self, all_zones: List[_LiquidityZone], start: float
    ) -> float:
        """Return the multiplier of the zone that ends at *start*."""
        for z in all_zones:
            if abs(z.end_hour_utc - start) < 1e-9:
                return z.multiplier
        return 1.0  # default normal

    def _next_zone_multiplier(
        self, all_zones: List[_LiquidityZone], end: float
    ) -> float:
        """Return the multiplier of the zone that starts at *end*."""
        for z in all_zones:
            if abs(z.start_hour_utc - end) < 1e-9:
                return z.multiplier
        return 1.0  # default normal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_spread_multiplier(self) -> float:
        """Return the current spread multiplier.

        Respects a timed override if one is active; otherwise derives the
        multiplier from the current UTC hour using smooth zone interpolation.
        """
        with self._lock:
            if self._override_multiplier is not None:
                if time.monotonic() < self._override_expires_at:
                    return self._override_multiplier
                else:
                    # Override has expired; clear it
                    self._override_multiplier = None

        return self._multiplier_for_hour(self._utc_hour_now())

    def get_spread_bps(self, symbol: Optional[str] = None) -> float:
        """Return the current effective spread in basis points.

        If *symbol* is registered via `register_symbol_override`, that symbol's
        base spread is used; otherwise the engine-level base is used.
        """
        with self._lock:
            base = (
                self._symbol_base_spreads.get(symbol, self._base_spread_bps)
                if symbol is not None
                else self._base_spread_bps
            )
        multiplier = self.get_current_spread_multiplier()
        return base * multiplier

    def get_current_session(self) -> str:
        """Return the name of the currently active liquidity session.

        Returns one of: "peak_liquidity", "dead_zone", "european_open", "normal".
        """
        utc_hour = self._utc_hour_now()
        for zone in (self.DEAD_ZONE, self.EUROPEAN_OPEN, self.PEAK_LIQUIDITY):
            if zone.start_hour_utc <= utc_hour < zone.end_hour_utc:
                return zone.name
        return self.NORMAL.name

    def get_session_stats(self) -> dict:
        """Return a snapshot dict of the current scheduling state."""
        utc_h = self._utc_hour_now()
        multiplier = self.get_current_spread_multiplier()
        session = self.get_current_session()
        with self._lock:
            override_active = (
                self._override_multiplier is not None
                and time.monotonic() < self._override_expires_at
            )
            override_remaining_s = (
                max(0.0, self._override_expires_at - time.monotonic())
                if override_active
                else 0.0
            )
        return {
            "current_session": session,
            "multiplier": multiplier,
            "spread_bps": self._base_spread_bps * multiplier,
            "utc_hour": utc_h,
            "override_active": override_active,
            "override_remaining_s": override_remaining_s,
            "symbol_overrides": dict(self._symbol_base_spreads),
        }

    # ------------------------------------------------------------------
    # Override API
    # ------------------------------------------------------------------

    def set_override(self, multiplier: float, duration_s: float) -> None:
        """Temporarily force a specific multiplier for *duration_s* seconds.

        Useful when external signals (e.g. VPIN spike, news event) warrant
        immediate spread widening regardless of the clock-based schedule.

        Parameters
        ----------
        multiplier : float
            The override multiplier to apply (e.g. 2.0 = double the spread).
        duration_s : float
            How long the override should remain active, in seconds.
        """
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        with self._lock:
            self._override_multiplier = multiplier
            self._override_expires_at = time.monotonic() + duration_s

    def clear_override(self) -> None:
        """Remove any active override and revert to the schedule-based multiplier."""
        with self._lock:
            self._override_multiplier = None
            self._override_expires_at = 0.0

    # ------------------------------------------------------------------
    # Per-symbol base spread
    # ------------------------------------------------------------------

    def register_symbol_override(self, symbol: str, base_spread_bps: float) -> None:
        """Register a per-symbol base spread in basis points.

        The session multiplier will be applied on top of this base, just as
        with the default base spread.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. "BTC-USD", "ETH-USDT").
        base_spread_bps : float
            Base spread in basis points for this symbol.
        """
        if base_spread_bps <= 0:
            raise ValueError("base_spread_bps must be positive")
        with self._lock:
            self._symbol_base_spreads[symbol] = base_spread_bps

    def remove_symbol_override(self, symbol: str) -> bool:
        """Remove a previously registered per-symbol base spread.

        Returns True if removed, False if it was not registered.
        """
        with self._lock:
            if symbol in self._symbol_base_spreads:
                del self._symbol_base_spreads[symbol]
                return True
            return False

    # ------------------------------------------------------------------
    # Convenience / informational
    # ------------------------------------------------------------------

    def get_multiplier_at_hour(self, utc_hour: float) -> float:
        """Return what the multiplier *would be* at a hypothetical UTC hour.

        Useful for pre-trade analysis and backtesting.
        """
        return self._multiplier_for_hour(utc_hour)

    def get_zone_schedule(self) -> List[dict]:
        """Return the full 24-hour zone schedule as a list of dicts."""
        zones = [
            self.DEAD_ZONE,
            self.NORMAL,
            self.EUROPEAN_OPEN,
            self.PEAK_LIQUIDITY,
            self._IMPLICIT_NORMAL_AFTER_PEAK,
        ]
        return [
            {
                "name": z.name,
                "start_hour_utc": z.start_hour_utc,
                "end_hour_utc": z.end_hour_utc,
                "multiplier": z.multiplier,
                "base_spread_bps": self._base_spread_bps * z.multiplier,
            }
            for z in zones
        ]

    def __repr__(self) -> str:  # pragma: no cover
        stats = self.get_session_stats()
        return (
            f"<SessionSpreadSchedule session={stats['current_session']} "
            f"multiplier={stats['multiplier']:.3f} "
            f"spread_bps={stats['spread_bps']:.2f}>"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 → a, t=1 → b. Clipped to [0,1]."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t
