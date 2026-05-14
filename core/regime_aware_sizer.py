"""Push 96 — RegimeAwareSizer: per-regime position sizing with hot-update scalar tables.

Design
------
Each market regime maps to a set of *scalars* that modulate the base position size
returned by the upstream sizer.  Scalars are multiplicative and compose as:

    final_size = base_size
                 * risk_scalar
                 * leverage_scalar
                 * confidence_weight(confidence)
                 * volatility_damper(vol_ratio)

Public API
----------
    sizer = RegimeAwareSizer()
    size  = sizer.size_position(base_size, regime, confidence, vol_ratio)
    sizer.set_scalar(regime, scalar_name, value)   # hot-update, thread-safe
    sizer.scalars                                  # Dict[regime, ScalarSet]
    sizer.active_regime                            # str | None
    sizer.on_transition(prev_snap, curr_snap)      # plug into RegimeDetector callback

Scalar names (per regime)
-------------------------
    risk_scalar        — raw risk multiplier           (default 1.0)
    leverage_scalar    — leverage multiplier           (default 1.0)
    conf_weight        — confidence contribution       (default 1.0)
    vol_dampen_factor  — how hard vol dampens size     (default 1.0)
    max_size_cap       — absolute cap on final size    (default inf)

Default regime tables
---------------------
    TRENDING_UP   : risk=1.2, leverage=1.2, conf_weight=1.1, vol_dampen=0.8,  cap=inf
    TRENDING_DOWN : risk=0.8, leverage=0.8, conf_weight=0.9, vol_dampen=1.2,  cap=inf
    RANGING       : risk=1.0, leverage=1.0, conf_weight=1.0, vol_dampen=1.0,  cap=inf
    VOLATILE      : risk=0.5, leverage=0.5, conf_weight=0.7, vol_dampen=2.0,  cap=inf
    UNKNOWN       : risk=0.6, leverage=0.6, conf_weight=0.5, vol_dampen=1.5,  cap=inf
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scalar set
# ---------------------------------------------------------------------------

_SCALAR_FIELDS = {
    "risk_scalar",
    "leverage_scalar",
    "conf_weight",
    "vol_dampen_factor",
    "max_size_cap",
}


@dataclass
class ScalarSet:
    """Per-regime sizing scalars."""
    risk_scalar:       float = 1.0
    leverage_scalar:   float = 1.0
    conf_weight:       float = 1.0
    vol_dampen_factor: float = 1.0
    max_size_cap:      float = math.inf

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        # serialise inf as null-safe string for JSON
        d["max_size_cap"] = None if math.isinf(self.max_size_cap) else self.max_size_cap
        return d


# ---------------------------------------------------------------------------
# Default regime tables
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, ScalarSet] = {
    "trending_up":   ScalarSet(risk_scalar=1.2, leverage_scalar=1.2, conf_weight=1.1, vol_dampen_factor=0.8),
    "trending_down": ScalarSet(risk_scalar=0.8, leverage_scalar=0.8, conf_weight=0.9, vol_dampen_factor=1.2),
    "ranging":       ScalarSet(risk_scalar=1.0, leverage_scalar=1.0, conf_weight=1.0, vol_dampen_factor=1.0),
    "volatile":      ScalarSet(risk_scalar=0.5, leverage_scalar=0.5, conf_weight=0.7, vol_dampen_factor=2.0),
    "unknown":       ScalarSet(risk_scalar=0.6, leverage_scalar=0.6, conf_weight=0.5, vol_dampen_factor=1.5),
}


# ---------------------------------------------------------------------------
# RegimeAwareSizer
# ---------------------------------------------------------------------------

class RegimeAwareSizer:
    """Per-regime position sizer with hot-updatable scalar tables.

    Parameters
    ----------
    custom_scalars : dict mapping regime_name -> dict of scalar overrides
        E.g. {"volatile": {"risk_scalar": 0.3, "max_size_cap": 500.0}}
    min_size : float
        Floor on the final computed size (default 0.0 — no floor).
    """

    def __init__(
        self,
        custom_scalars: Optional[Dict[str, Dict[str, float]]] = None,
        min_size: float = 0.0,
    ) -> None:
        self._lock    = threading.Lock()
        self._min_size = float(min_size)
        self._active_regime: Optional[str] = None

        # Deep-copy defaults then apply overrides
        self._tables: Dict[str, ScalarSet] = {
            k: ScalarSet(**asdict(v)) for k, v in _DEFAULTS.items()
        }
        if custom_scalars:
            for regime, overrides in custom_scalars.items():
                self.set_scalars(regime, overrides)

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def size_position(
        self,
        base_size:  float,
        regime:     str,
        confidence: float = 1.0,
        vol_ratio:  float = 1.0,
    ) -> float:
        """Compute final position size for the given regime.

        Parameters
        ----------
        base_size  : raw size from upstream sizer (e.g. Kelly, fixed-frac)
        regime     : current regime label (case-insensitive)
        confidence : detector confidence [0, 1]
        vol_ratio  : current_vol / threshold_vol (> 1.0 means elevated vol)

        Returns
        -------
        float : final position size, floored at min_size and capped at max_size_cap
        """
        key = regime.lower()
        with self._lock:
            ss = self._tables.get(key) or self._tables.get("unknown") or ScalarSet()
            self._active_regime = key

        # Confidence contribution: lerp between conf_weight and 1.0
        # conf=1.0 -> full conf_weight; conf=0.0 -> neutral (1.0)
        conf_factor = 1.0 + (ss.conf_weight - 1.0) * max(0.0, min(1.0, confidence))

        # Volatility damper: size shrinks as vol_ratio rises
        # dampen = 1 / (1 + (vol_ratio - 1) * vol_dampen_factor)  for vol_ratio >= 1
        excess_vol  = max(0.0, vol_ratio - 1.0)
        vol_factor  = 1.0 / (1.0 + excess_vol * ss.vol_dampen_factor)

        raw = base_size * ss.risk_scalar * ss.leverage_scalar * conf_factor * vol_factor
        capped  = min(raw, ss.max_size_cap) if not math.isinf(ss.max_size_cap) else raw
        floored = max(capped, self._min_size)
        return round(floored, 8)

    # ------------------------------------------------------------------
    # Scalar hot-update
    # ------------------------------------------------------------------

    def set_scalar(self, regime: str, scalar_name: str, value: float) -> None:
        """Hot-update a single scalar for a regime.  Thread-safe.

        Raises
        ------
        ValueError  if scalar_name is not a recognised field.
        """
        if scalar_name not in _SCALAR_FIELDS:
            raise ValueError(
                f"Unknown scalar '{scalar_name}'. Valid: {sorted(_SCALAR_FIELDS)}"
            )
        key = regime.lower()
        with self._lock:
            if key not in self._tables:
                self._tables[key] = ScalarSet()
            setattr(self._tables[key], scalar_name, float(value))
        logger.debug("set_scalar(%s, %s) = %s", regime, scalar_name, value)

    def set_scalars(self, regime: str, overrides: Dict[str, float]) -> None:
        """Hot-update multiple scalars at once.  Thread-safe."""
        for k, v in overrides.items():
            self.set_scalar(regime, k, v)

    # ------------------------------------------------------------------
    # on_transition callback (plug into RegimeDetector)
    # ------------------------------------------------------------------

    def on_transition(
        self,
        prev_snap: Any,  # RegimeSnapshot
        curr_snap: Any,  # RegimeSnapshot
    ) -> None:
        """Callback wired to RegimeDetector.on_transition.

        Called automatically by RegimeDetector every time the regime changes.
        Updates `_active_regime` and optionally adapts scalars based on
        the incoming snapshot’s metrics.

        Current adaptation rules (conservative, additive):
        - If confidence < 0.3: halve risk_scalar for this regime (floor 0.1)
        - If vol_ratio > 2.0:  reduce leverage_scalar by 20% (floor 0.1)
        Both are one-shot adjustments; they do NOT compound across calls.
        """
        key = curr_snap.regime.value if hasattr(curr_snap.regime, "value") else str(curr_snap.regime)
        with self._lock:
            self._active_regime = key

        # Low-confidence detection — de-risk the table for this regime
        if curr_snap.confidence < 0.3:
            try:
                ss = self._get_table(key)
                new_risk = max(0.1, ss.risk_scalar * 0.5)
                self.set_scalar(key, "risk_scalar", new_risk)
                logger.info(
                    "on_transition: low confidence (%.2f) for %s — risk_scalar → %.4f",
                    curr_snap.confidence, key, new_risk,
                )
            except Exception as exc:
                logger.exception("on_transition risk adaptation error: %s", exc)

        # Elevated volatility — reduce leverage for this regime
        vol_ratio = getattr(curr_snap, "volatility", 0.0)
        # Normalise by vol_high_threshold proxy (0.03) if not pre-computed
        if vol_ratio > 0.06:  # > 2x the default 0.03 threshold
            try:
                ss = self._get_table(key)
                new_lev = max(0.1, ss.leverage_scalar * 0.8)
                self.set_scalar(key, "leverage_scalar", new_lev)
                logger.info(
                    "on_transition: high vol (%.4f) for %s — leverage_scalar → %.4f",
                    vol_ratio, key, new_lev,
                )
            except Exception as exc:
                logger.exception("on_transition leverage adaptation error: %s", exc)

    def _get_table(self, key: str) -> ScalarSet:
        with self._lock:
            return self._tables.get(key) or ScalarSet()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def scalars(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot of all regime scalar tables (used by GET /sizer API)."""
        with self._lock:
            return {k: v.to_dict() for k, v in self._tables.items()}

    @property
    def active_regime(self) -> Optional[str]:
        """Last regime seen via size_position() or on_transition()."""
        return self._active_regime

    def reset_regime(self, regime: str) -> None:
        """Reset a regime’s scalars back to factory defaults."""
        key = regime.lower()
        default = _DEFAULTS.get(key)
        with self._lock:
            self._tables[key] = ScalarSet(**asdict(default)) if default else ScalarSet()
        logger.debug("reset_regime(%s) — restored factory defaults", regime)

    def reset_all(self) -> None:
        """Reset every regime to factory defaults."""
        with self._lock:
            self._tables = {k: ScalarSet(**asdict(v)) for k, v in _DEFAULTS.items()}
        logger.debug("reset_all() — all regime scalars restored")

    def __repr__(self) -> str:
        return (
            f"RegimeAwareSizer("
            f"active={self._active_regime!r}, "
            f"regimes={list(self._tables.keys())})"
        )
