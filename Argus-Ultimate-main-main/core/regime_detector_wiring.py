"""
Push 90 — regime_detector_wiring.py
Drops a LiveRegimeDetector into an ArgusSystem and hooks it into the
existing tick pipeline so SystemConfig.market_regime stays current.

Usage
-----
from core.regime_detector_wiring import wire_regime_detector
detector = wire_regime_detector(system)

The returned detector can be queried at any time:
    snap = detector.snapshot()
    print(snap.regime, snap.vol_ratio, snap.confidence)

Version: v8.26.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.regime_detector import LiveRegimeDetector, RegimeDetectorConfig, Regime

if TYPE_CHECKING:
    pass  # avoid circular: ArgusSystem is not imported at runtime

logger = logging.getLogger(__name__)


def wire_regime_detector(
    system,
    detector_config: RegimeDetectorConfig | None = None,
) -> LiveRegimeDetector:
    """
    Wire a LiveRegimeDetector into *system*.

    Steps
    -----
    1. Instantiate detector with ``detector_config`` (or defaults).
    2. Attach it to ``system.config`` so regime updates are automatic.
    3. Monkey-patch ``system._on_tick_received`` to call
       ``detector.update()`` on every price tick.
    4. Store the detector as ``system.regime_detector`` for easy access.

    Parameters
    ----------
    system:
        An ArgusSystem instance (or compatible object with ``config``,
        ``_on_tick_received``, and a tick dispatch mechanism).
    detector_config:
        Optional custom RegimeDetectorConfig.  Defaults apply if None.

    Returns
    -------
    LiveRegimeDetector
        The wired detector instance.
    """
    detector = LiveRegimeDetector(config=detector_config)
    detector.attach_system_config(system.config)

    # Patch the tick handler --------------------------------------------------
    _original_on_tick = getattr(system, "_on_tick_received", None)

    def _patched_on_tick(tick, *args, **kwargs):
        # Extract OHLC fields — support both dict-style and attr-style ticks
        try:
            price = float(getattr(tick, "last",  None) or
                          getattr(tick, "price", None) or
                          tick.get("last",  tick.get("price", 0)))
            high  = float(getattr(tick, "high",  None) or
                          tick.get("high",  price))
            low   = float(getattr(tick, "low",   None) or
                          tick.get("low",   price))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[RegimeDetector] tick parse error: %s", exc)
            price = high = low = 0.0

        if price > 0:
            detector.update(price=price, high=high, low=low)

        if _original_on_tick is not None:
            return _original_on_tick(tick, *args, **kwargs)

    system._on_tick_received = _patched_on_tick

    # Also try to hook into the event bus if present -------------------------
    _bus = getattr(system, "bus", None) or getattr(system, "event_bus", None)
    if _bus is not None:
        try:
            _bus.subscribe("TICK", _patched_on_tick)
            logger.info("[RegimeDetector] subscribed to event_bus TICK channel")
        except Exception as exc:  # noqa: BLE001
            logger.debug("[RegimeDetector] event_bus subscribe skipped: %s", exc)

    # Expose on system -------------------------------------------------------
    system.regime_detector = detector

    logger.info(
        "[RegimeDetector] wired — warmup=%d ticks, "
        "trend_threshold=%.4f, high_vol_threshold=%.1f",
        detector.cfg.warmup_ticks,
        detector.cfg.trend_threshold,
        detector.cfg.high_vol_threshold,
    )
    return detector


def get_regime_summary(system) -> dict:
    """
    Convenience helper — returns a dict summary of the current regime state.
    Safe to call even if the detector is not wired.
    """
    det: LiveRegimeDetector | None = getattr(system, "regime_detector", None)
    if det is None:
        return {"regime": Regime.UNKNOWN.value, "wired": False}

    snap = det.snapshot()
    return {
        "regime":      snap.regime.value,
        "vol_ratio":   snap.vol_ratio,
        "trend_score": snap.trend_score,
        "bb_pos":      snap.bb_pos,
        "autocorr":    snap.autocorr,
        "confidence":  snap.confidence,
        "tick_count":  snap.tick_count,
        "wired":       True,
    }
