"""
Push 92 — RegimeAwareSizer v8.28.0

A thin scaling layer that wraps PositionSizer and multiplies the raw
position size by a regime-specific scalar sourced from LiveRegimeDetector
(or a plain string passed directly).

Regime scalars (defaults)
-------------------------
REGIME           SCALAR   RATIONALE
TRENDING_BULL    1.20     ride the trend, slightly wider size
TRENDING_BEAR    0.80     short-biased caution; reduce longs
RANGING          1.00     neutral baseline
HIGH_VOL         0.40     explosive vol — cut size sharply
UNKNOWN          0.60     startup / data gap — conservative

All scalars are clamped to [min_scalar, max_scalar] (defaults 0.10–1.50)
so no single regime can blow up or zero-out sizing entirely.

Usage
-----
# Standalone
sizer = RegimeAwareSizer()
qty = sizer.size(equity=10_000, price=50_000, strength=0.7,
                 regime="HIGH_VOL")

# Wired to LiveRegimeDetector (auto-reads regime)
from core.regime_detector import LiveRegimeDetector
detector = LiveRegimeDetector()
sizer = RegimeAwareSizer(detector=detector)
qty = sizer.size(equity=10_000, price=50_000, strength=0.7)

# Wired into ArgusSystem (call after Push 91 _build)
sizer = RegimeAwareSizer(detector=system.regime_detector)

Version: v8.28.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from core.risk.position_sizer import PositionSizer, SizerConfig, SizingMethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default regime scalars
# ---------------------------------------------------------------------------

DEFAULT_REGIME_SCALARS: Dict[str, float] = {
    "TRENDING_BULL": 1.20,
    "TRENDING_BEAR": 0.80,
    "RANGING":       1.00,
    "HIGH_VOL":      0.40,
    "UNKNOWN":       0.60,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RegimeSizerConfig:
    """Configuration for RegimeAwareSizer."""

    # Base sizer config (passed through to PositionSizer)
    sizer: SizerConfig = field(default_factory=SizerConfig)

    # Per-regime multipliers (override any subset you like)
    regime_scalars: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REGIME_SCALARS)
    )

    # Hard clamps applied after regime scaling
    min_scalar: float = 0.10   # never go below 10% of base size
    max_scalar: float = 1.50   # never exceed 150% of base size

    # Fallback scalar when regime is unrecognised
    fallback_scalar: float = 0.60


# ---------------------------------------------------------------------------
# RegimeAwareSizer
# ---------------------------------------------------------------------------

class RegimeAwareSizer:
    """
    Regime-aware position sizer.

    Delegates core sizing arithmetic to ``PositionSizer``, then multiplies
    the result by the scalar for the current market regime.

    Parameters
    ----------
    config:
        ``RegimeSizerConfig`` (or None for defaults).
    detector:
        Optional ``LiveRegimeDetector`` instance.  If supplied, the sizer
        reads the current regime automatically on every ``size()`` call.
        Passing ``regime=`` explicitly to ``size()`` always takes priority.
    """

    def __init__(
        self,
        config:   Optional[RegimeSizerConfig] = None,
        detector=None,
    ) -> None:
        self.config   = config or RegimeSizerConfig()
        self._sizer   = PositionSizer(config=self.config.sizer)
        self._detector = detector

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def size(
        self,
        equity:           float,
        price:            float,
        strength:         float = 0.5,
        atr:              float = 0.0,
        realised_vol_pct: float = 0.0,
        method:           Optional[SizingMethod] = None,
        regime:           Optional[str] = None,
    ) -> float:
        """
        Calculate regime-adjusted position size in base units.

        Parameters
        ----------
        equity:           Current portfolio equity ($).
        price:            Current asset price.
        strength:         Signal strength [0, 1].
        atr:              ATR value (used by FIXED_FRAC method).
        realised_vol_pct: Annualised realised vol % (used by VOL_ADJUSTED).
        method:           Override base sizing method for this call.
        regime:           Explicit regime string.  If None, reads from
                          attached ``detector`` (or falls back to UNKNOWN).

        Returns
        -------
        float
            Position size in base units, after regime scaling.
        """
        effective_regime = self._resolve_regime(regime)

        base_qty = self._sizer.size(
            equity=equity,
            price=price,
            strength=strength,
            atr=atr,
            realised_vol_pct=realised_vol_pct,
            method=method,
        )

        if base_qty <= 0:
            return 0.0

        scalar      = self._get_scalar(effective_regime)
        scaled_qty  = base_qty * scalar
        max_qty     = base_qty * self.config.max_scalar
        min_qty_val = base_qty * self.config.min_scalar
        scaled_qty  = max(min_qty_val, min(scaled_qty, max_qty))

        if scaled_qty < self.config.sizer.min_qty:
            return 0.0

        logger.debug(
            "[RegimeSizer] regime=%s scalar=%.2f base=%.8f scaled=%.8f",
            effective_regime, scalar, base_qty, scaled_qty,
        )
        return round(scaled_qty, 8)

    def scalar_for_regime(self, regime: Optional[str] = None) -> float:
        """Return the scalar that would be applied for *regime* (or current)."""
        return self._get_scalar(self._resolve_regime(regime))

    def attach_detector(self, detector) -> None:
        """Attach or swap the LiveRegimeDetector after construction."""
        self._detector = detector
        logger.info("[RegimeSizer] detector attached: %s", type(detector).__name__)

    def set_regime_scalar(self, regime: str, scalar: float) -> None:
        """
        Override the scalar for a specific regime at runtime.

        Example
        -------
        sizer.set_regime_scalar("HIGH_VOL", 0.25)  # more conservative
        """
        clamped = max(self.config.min_scalar,
                      min(scalar, self.config.max_scalar))
        self.config.regime_scalars[regime.upper()] = clamped
        logger.info("[RegimeSizer] set %s scalar → %.2f", regime, clamped)

    @property
    def current_regime(self) -> str:
        """The regime currently read from the attached detector (or UNKNOWN)."""
        return self._resolve_regime(None)

    @property
    def audit(self):
        """Passthrough to underlying PositionSizer audit log."""
        return self._sizer.audit

    def clear_audit(self) -> None:
        self._sizer.clear_audit()

    # ------------------------------------------------------------------
    # Regime summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Human-readable state snapshot."""
        regime = self._resolve_regime(None)
        return {
            "current_regime":  regime,
            "current_scalar":  self._get_scalar(regime),
            "regime_scalars":  dict(self.config.regime_scalars),
            "min_scalar":      self.config.min_scalar,
            "max_scalar":      self.config.max_scalar,
            "detector_wired":  self._detector is not None,
            "base_method":     self.config.sizer.method.value,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_regime(self, explicit: Optional[str]) -> str:
        """Return explicit regime, or read from detector, or UNKNOWN."""
        if explicit is not None:
            return explicit.upper()
        if self._detector is not None:
            try:
                snap = self._detector.snapshot()
                r = snap.regime
                return r.value if hasattr(r, "value") else str(r)
            except Exception as exc:
                logger.debug("[RegimeSizer] detector read failed: %s", exc)
        return "UNKNOWN"

    def _get_scalar(self, regime: str) -> float:
        """Look up scalar for regime, apply clamp."""
        raw = self.config.regime_scalars.get(
            regime.upper(), self.config.fallback_scalar
        )
        return max(self.config.min_scalar, min(raw, self.config.max_scalar))
