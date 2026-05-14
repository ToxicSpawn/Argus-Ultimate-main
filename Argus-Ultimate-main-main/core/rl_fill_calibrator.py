"""
core/rl_fill_calibrator.py
==========================
Real-fill feedback loop for RL gate calibration.

The DirectionHead is born uncalibrated — it defers every decision to
the base pipeline until it has seen enough live fill data to trust its
own policy logits.  This module closes that gap.

Design
------
1. ``FillEvent``         — lightweight record emitted by ExecutionEngine
                           after each order settles.
2. ``FillCalibrator``    — ingests FillEvents, computes per-fill reward
                           signals (fill_ratio, slippage_bps, PnL tick),
                           batches them into ``calibration_window`` chunks
                           and calls ``JaxPPOTrainer.ingest_live_batch()``.
3. Graduation condition  — once ``min_fills`` real fills have been
                           processed AND the rolling Sharpe of the
                           reward signal crosses ``sharpe_threshold``,
                           ``FillCalibrator.calibrated`` flips True and
                           a ``CALIBRATION_COMPLETE`` event fires on the
                           EventBus.
4. The DirectionHead subscribes to CALIBRATION_COMPLETE to unlock its
   skip/size logic.

Thread / async safety
---------------------
``FillCalibrator`` is fully async.  Ingest via ``await cal.record(fill)``
or call the synchronous ``cal.record_sync(fill)`` from a non-async context
(it schedules on the running loop).
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

logger = logging.getLogger("argus.core.rl_fill_calibrator")


# ---------------------------------------------------------------------------
# Fill event
# ---------------------------------------------------------------------------

@dataclass
class FillEvent:
    """
    Emitted by ExecutionEngine after an order is confirmed filled.

    Fields
    ------
    symbol         : e.g. "BTC/USDT"
    side           : "buy" | "sell"
    requested_qty  : quantity requested
    filled_qty     : quantity actually filled
    requested_px   : limit/market price at submission
    fill_px        : actual average fill price
    fee_bps        : fee paid in basis points
    latency_us     : submission-to-fill latency in microseconds
    signal_id      : opaque ID linking back to the originating signal
    ts             : Unix timestamp (float)
    """
    symbol: str
    side: str
    requested_qty: float
    filled_qty: float
    requested_px: float
    fill_px: float
    fee_bps: float = 0.0
    latency_us: float = 0.0
    signal_id: str = ""
    ts: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def fill_ratio(self) -> float:
        """Fraction of requested quantity that was filled (0-1)."""
        if self.requested_qty <= 0:
            return 0.0
        return min(self.filled_qty / self.requested_qty, 1.0)

    @property
    def slippage_bps(self) -> float:
        """Signed slippage in basis points vs requested price."""
        if self.requested_px <= 0:
            return 0.0
        raw = (self.fill_px - self.requested_px) / self.requested_px * 10_000
        return raw if self.side == "buy" else -raw

    @property
    def net_reward(self) -> float:
        """
        Scalar reward signal fed to the RL trainer:
        fill_ratio bonus − slippage penalty − fee penalty
        """
        return (
            self.fill_ratio * 1.0
            - max(self.slippage_bps, 0.0) * 0.01
            - self.fee_bps * 0.01
        )


# ---------------------------------------------------------------------------
# Calibration config
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    min_fills: int = 200             # minimum fills before graduation
    calibration_window: int = 50     # fills per batch fed to trainer
    sharpe_threshold: float = 0.5    # rolling Sharpe of net_reward to graduate
    sharpe_window: int = 100         # samples for rolling Sharpe
    reward_scale: float = 1.0        # scale factor before feeding trainer
    max_slippage_bps: float = 20.0   # fills above this are excluded as outliers


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class FillCalibrator:
    """
    Real-fill feedback loop.

    Usage
    -----
    ::

        cal = FillCalibrator(trainer=jax_ppo_trainer)
        # wire into execution engine:
        execution_engine.on_fill = cal.record_sync
        # or async:
        await cal.record(fill_event)
    """

    CALIBRATION_COMPLETE = "CALIBRATION_COMPLETE"

    def __init__(
        self,
        trainer: Any,                        # JaxPPOTrainer or compatible
        cfg: Optional[CalibrationConfig] = None,
        event_bus: Any = None,               # optional EventBus
        on_calibrated: Optional[Callable[[], None]] = None,
    ) -> None:
        self._trainer = trainer
        self._cfg = cfg or CalibrationConfig()
        self._bus = event_bus
        self._on_calibrated = on_calibrated or (lambda: None)

        self._fills: list[FillEvent] = []
        self._pending: list[FillEvent] = []
        self._reward_window: Deque[float] = deque(maxlen=self._cfg.sharpe_window)
        self._calibrated = False
        self._total_fills = 0
        self._batches_sent = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def total_fills(self) -> int:
        return self._total_fills

    async def record(self, fill: FillEvent) -> None:
        """Async ingest path — call from async execution engine hooks."""
        self._ingest(fill)
        if len(self._pending) >= self._cfg.calibration_window:
            await self._flush_batch()

    def record_sync(self, fill: FillEvent) -> None:
        """Sync ingest path — schedules on running event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.record(fill)))
            else:
                self._ingest(fill)
        except RuntimeError:
            self._ingest(fill)

    def stats(self) -> dict:
        rewards = list(self._reward_window)
        return {
            "calibrated": self._calibrated,
            "total_fills": self._total_fills,
            "batches_sent": self._batches_sent,
            "pending": len(self._pending),
            "rolling_sharpe": self._rolling_sharpe(),
            "mean_fill_ratio": (
                statistics.mean([f.fill_ratio for f in self._fills[-100:]])
                if self._fills else 0.0
            ),
            "mean_slippage_bps": (
                statistics.mean([f.slippage_bps for f in self._fills[-100:]])
                if self._fills else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ingest(self, fill: FillEvent) -> None:
        # Filter extreme outliers
        if abs(fill.slippage_bps) > self._cfg.max_slippage_bps:
            logger.debug("Fill outlier excluded: slippage=%.1fbps", fill.slippage_bps)
            return
        self._fills.append(fill)
        self._pending.append(fill)
        self._reward_window.append(fill.net_reward * self._cfg.reward_scale)
        self._total_fills += 1

    async def _flush_batch(self) -> None:
        """Send a calibration batch to the RL trainer."""
        batch = self._pending[:self._cfg.calibration_window]
        self._pending = self._pending[self._cfg.calibration_window:]

        rewards = [f.net_reward * self._cfg.reward_scale for f in batch]
        fill_ratios = [f.fill_ratio for f in batch]
        slippages = [f.slippage_bps for f in batch]

        # Feed into trainer if it exposes ingest_live_batch
        if hasattr(self._trainer, "ingest_live_batch"):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._trainer.ingest_live_batch,
                    rewards, fill_ratios, slippages,
                )
            except Exception as exc:
                logger.warning("Trainer ingest_live_batch failed: %s", exc)

        self._batches_sent += 1
        logger.debug(
            "Calibration batch #%d sent: n=%d mean_reward=%.4f",
            self._batches_sent, len(batch),
            statistics.mean(rewards) if rewards else 0.0,
        )

        # Graduation check
        if not self._calibrated:
            self._check_graduation()

    def _rolling_sharpe(self) -> float:
        rewards = list(self._reward_window)
        if len(rewards) < 10:
            return 0.0
        try:
            mu = statistics.mean(rewards)
            sigma = statistics.stdev(rewards)
            return mu / sigma if sigma > 1e-9 else 0.0
        except Exception:
            return 0.0

    def _check_graduation(self) -> None:
        if self._total_fills < self._cfg.min_fills:
            return
        sharpe = self._rolling_sharpe()
        if sharpe >= self._cfg.sharpe_threshold:
            self._calibrated = True
            logger.info(
                "FillCalibrator GRADUATED: fills=%d sharpe=%.3f",
                self._total_fills, sharpe,
            )
            self._on_calibrated()
            if self._bus is not None:
                try:
                    self._bus.emit(self.CALIBRATION_COMPLETE, {
                        "fills": self._total_fills,
                        "sharpe": sharpe,
                        "batches": self._batches_sent,
                    })
                except Exception as exc:
                    logger.warning("EventBus emit failed: %s", exc)
        else:
            logger.debug(
                "Graduation check: fills=%d sharpe=%.3f (need %.2f)",
                self._total_fills, sharpe, self._cfg.sharpe_threshold,
            )
