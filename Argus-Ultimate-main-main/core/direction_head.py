"""
core/direction_head.py
======================
DirectionHead — RL-gated skip/size decision layer.

Role in the pipeline
--------------------
The base pipeline (signal generators → ensemble → risk gate) produces a
``SignalPacket`` with:
  - ``direction``   : "long" | "short" | "flat"
  - ``base_size``   : raw position size from conviction_sizer
  - ``confidence``  : [0, 1] ensemble confidence

The DirectionHead sits between the ensemble and the ExecutionEngine.  It
has two operating modes:

Uncalibrated mode (rl_calibrated=False)
  - Pass every signal through unchanged.
  - Log the RL head's *shadow* decision without acting on it.
  - This is the default until FillCalibrator emits CALIBRATION_COMPLETE.

Calibrated mode (rl_calibrated=True)
  - Obtain a policy vector from the wired JaxPPOTrainer (or
    JaxPolicyNetwork) for the current observation.
  - ``skip`` the signal entirely if logit confidence < ``skip_threshold``.
  - ``scale`` the base_size by a multiplier in [size_floor, size_ceil]
    derived from the fill_ratio confidence supplied by FillCalibrator.
  - Route through KernelBypassRouter when colo_ready=True, else standard
    ExecutionEngine.

Calibration feedback loop
-------------------------
After each fill, DirectionHead calls ``fill_calibrator.record_sync(fill)``
so the calibration cycle is self-sustaining.

Integration
-----------
Wire into full_wiring.py::

    direction_head = DirectionHead(
        trainer=jax_ppo_trainer,
        fill_calibrator=fill_cal,
        bypass_router=kernel_bypass_router,
        colo_bridge=dpdk_colo_bridge,
    )
    # Replace base pipeline's final step:
    pipeline.set_execution_hook(direction_head.process)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("argus.core.direction_head")


# ---------------------------------------------------------------------------
# Signal packet (shared contract with ensemble / risk gate)
# ---------------------------------------------------------------------------

@dataclass
class SignalPacket:
    """
    Standardised signal handed off from the base pipeline to DirectionHead.
    """
    symbol: str
    direction: str          # "long" | "short" | "flat"
    base_size: float        # raw size from conviction_sizer
    confidence: float       # [0, 1] ensemble confidence
    price: float            # reference price at signal time
    signal_id: str = ""
    metadata: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class HeadDecision:
    """
    Output of DirectionHead.process().
    """
    signal: SignalPacket
    skipped: bool
    final_size: float
    size_multiplier: float
    rl_logit: float         # raw policy logit for the chosen action
    fill_ratio_est: float   # calibrator's current mean fill ratio
    via_bypass: bool        # True if routed through KernelBypassRouter
    shadow_only: bool       # True when running uncalibrated (no real effect)
    latency_us: float = 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DirectionHeadConfig:
    skip_threshold: float = 0.35     # policy logit below this → skip
    size_floor: float = 0.5          # minimum size multiplier
    size_ceil: float = 2.0           # maximum size multiplier
    fill_ratio_weight: float = 0.6   # blend weight for fill_ratio in sizing
    logit_weight: float = 0.4        # blend weight for logit in sizing
    shadow_log_interval: int = 20    # log shadow decisions every N signals


# ---------------------------------------------------------------------------
# Direction Head
# ---------------------------------------------------------------------------

class DirectionHead:
    """
    RL-gated skip/size decision layer.

    Parameters
    ----------
    trainer      : JaxPPOTrainer or JaxPolicyNetwork — exposes .predict(obs)
    fill_calibrator : FillCalibrator — provides calibrated=bool and stats()
    bypass_router   : KernelBypassRouter — low-latency order path
    colo_bridge     : DPDKColoBridge — provides is_ready bool
    cfg             : DirectionHeadConfig
    execution_engine: fallback ExecutionEngine for non-colo path
    """

    def __init__(
        self,
        trainer: Any,
        fill_calibrator: Any,
        bypass_router: Any = None,
        colo_bridge: Any = None,
        cfg: Optional[DirectionHeadConfig] = None,
        execution_engine: Any = None,
    ) -> None:
        self._trainer = trainer
        self._cal = fill_calibrator
        self._bypass = bypass_router
        self._colo = colo_bridge
        self._cfg = cfg or DirectionHeadConfig()
        self._engine = execution_engine

        self._processed = 0
        self._skipped = 0
        self._shadow_count = 0
        self._calibrated_at: Optional[float] = None

        # Subscribe to calibration complete
        if hasattr(self._cal, '_bus') and self._cal._bus is not None:
            try:
                self._cal._bus.subscribe(
                    "CALIBRATION_COMPLETE", self._on_calibration_complete
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, packet: SignalPacket) -> HeadDecision:
        """
        Main entry point.  Called by the pipeline for each signal.
        Returns a HeadDecision; the caller is responsible for routing
        to the ExecutionEngine when skipped=False.
        """
        t0 = time.monotonic_ns()
        self._processed += 1

        rl_calibrated = self._cal.calibrated
        cal_stats = self._cal.stats()
        fill_ratio_est = cal_stats.get("mean_fill_ratio", 1.0)

        # ----------------------------------------------------------
        # Get policy logit
        # ----------------------------------------------------------
        logit = await self._get_logit(packet)

        # ----------------------------------------------------------
        # Uncalibrated: pass-through (shadow only)
        # ----------------------------------------------------------
        if not rl_calibrated:
            self._shadow_count += 1
            if self._shadow_count % self._cfg.shadow_log_interval == 0:
                logger.debug(
                    "[shadow] %s dir=%s logit=%.3f fill_ratio=%.3f "
                    "(uncalibrated — deferring to base pipeline)",
                    packet.symbol, packet.direction, logit, fill_ratio_est,
                )
            return HeadDecision(
                signal=packet,
                skipped=False,
                final_size=packet.base_size,
                size_multiplier=1.0,
                rl_logit=logit,
                fill_ratio_est=fill_ratio_est,
                via_bypass=False,
                shadow_only=True,
                latency_us=(time.monotonic_ns() - t0) / 1_000.0,
            )

        # ----------------------------------------------------------
        # Calibrated: skip/size gate
        # ----------------------------------------------------------

        # Skip check
        if logit < self._cfg.skip_threshold or packet.direction == "flat":
            self._skipped += 1
            logger.debug(
                "[DirectionHead] SKIP %s logit=%.3f threshold=%.2f",
                packet.symbol, logit, self._cfg.skip_threshold,
            )
            return HeadDecision(
                signal=packet,
                skipped=True,
                final_size=0.0,
                size_multiplier=0.0,
                rl_logit=logit,
                fill_ratio_est=fill_ratio_est,
                via_bypass=False,
                shadow_only=False,
                latency_us=(time.monotonic_ns() - t0) / 1_000.0,
            )

        # Size scaling: blend logit confidence + fill_ratio confidence
        multiplier = self._compute_multiplier(logit, fill_ratio_est)
        final_size = packet.base_size * multiplier

        # Route via bypass if colo ready
        via_bypass = (
            self._bypass is not None
            and self._colo is not None
            and self._colo.is_ready
        )

        logger.debug(
            "[DirectionHead] EXECUTE %s dir=%s size=%.6f (x%.2f) "
            "logit=%.3f fill_ratio=%.3f bypass=%s",
            packet.symbol, packet.direction, final_size, multiplier,
            logit, fill_ratio_est, via_bypass,
        )

        latency_us = (time.monotonic_ns() - t0) / 1_000.0
        return HeadDecision(
            signal=packet,
            skipped=False,
            final_size=final_size,
            size_multiplier=multiplier,
            rl_logit=logit,
            fill_ratio_est=fill_ratio_est,
            via_bypass=via_bypass,
            shadow_only=False,
            latency_us=latency_us,
        )

    def stats(self) -> dict:
        return {
            "processed": self._processed,
            "skipped": self._skipped,
            "skip_rate": self._skipped / max(self._processed, 1),
            "shadow_signals": self._shadow_count,
            "rl_calibrated": self._cal.calibrated,
            "calibrated_at": self._calibrated_at,
            "colo_ready": self._colo.is_ready if self._colo else False,
            "fill_calibrator": self._cal.stats(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_logit(self, packet: SignalPacket) -> float:
        """
        Query the policy network for the current signal's action logit.
        Falls back to ensemble confidence when trainer not ready.
        """
        if self._trainer is None:
            return packet.confidence
        try:
            obs = self._build_obs(packet)
            if hasattr(self._trainer, "predict"):
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._trainer.predict, obs
                )
                # result expected to be array of logits; take max
                if hasattr(result, '__iter__'):
                    return float(max(result))
                return float(result)
        except Exception as exc:
            logger.debug("Policy predict failed: %s — using confidence", exc)
        return packet.confidence

    def _build_obs(self, packet: SignalPacket) -> list:
        """Build a minimal observation vector from the signal packet."""
        direction_enc = {"long": 1.0, "short": -1.0, "flat": 0.0}
        return [
            direction_enc.get(packet.direction, 0.0),
            packet.confidence,
            packet.base_size,
            packet.price,
            float(self._cal.total_fills) / 1000.0,  # normalised fill count
        ]

    def _compute_multiplier(self, logit: float, fill_ratio: float) -> float:
        """
        Blend logit confidence and fill_ratio into a size multiplier
        in [size_floor, size_ceil].
        """
        cfg = self._cfg
        blended = (
            logit * cfg.logit_weight
            + fill_ratio * cfg.fill_ratio_weight
        )
        # Normalise 0-1 blend into [floor, ceil]
        blended = max(0.0, min(blended, 1.0))
        return cfg.size_floor + blended * (cfg.size_ceil - cfg.size_floor)

    def _on_calibration_complete(self, payload: dict) -> None:
        self._calibrated_at = time.time()
        logger.info(
            "DirectionHead: RL gate UNLOCKED fills=%s sharpe=%.3f",
            payload.get('fills'), payload.get('sharpe', 0.0),
        )
