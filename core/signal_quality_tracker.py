"""
SignalQualityTracker — Per-source prediction accuracy tracker.

Records composite direction signals from advisory["ensemble"] sources,
compares them to actual price changes n_step_ahead cycles later, and
maintains EMA-decayed per-source reliability scores.

Every update_interval_cycles: calls ensemble_hub.update_source_weights()
with the current reliability map, so the EnsembleHub automatically
down-weights unreliable sources.

Output: advisory["signal_quality"]
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalQualityReport:
    source_accuracy: Dict[str, float]      # {source: 0–1 hit rate}
    recommended_weights: Dict[str, float]  # normalised, sum=1.0
    low_quality_sources: List[str]
    avg_reliability: float
    total_predictions: int
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# SignalQualityTracker
# ---------------------------------------------------------------------------

class SignalQualityTracker:
    """
    Tracks per-source signal prediction accuracy.

    Each cycle: records (source → predicted_direction, price) pairs.
    After n_step_ahead cycles: checks if actual price change matched
    the predicted direction → hit (1) or miss (0).
    EMA with `decay` smooths the per-source hit rate.

    Parameters
    ----------
    n_step_ahead         : cycles ahead to measure prediction outcome
    window               : rolling window for accuracy calculations
    min_samples          : minimum samples before reporting accuracy
    decay                : EMA decay factor (0.95 = recent data weighted)
    low_quality_threshold: accuracy below this → low_quality_sources list
    update_interval_cycles: cycles between weight updates to ensemble_hub
    ensemble_hub         : optional EnsembleSignalHub instance
    config               : optional config object
    """

    def __init__(
        self,
        n_step_ahead: int = 10,
        window: int = 50,
        min_samples: int = 10,
        decay: float = 0.95,
        low_quality_threshold: float = 0.40,
        update_interval_cycles: int = 100,
        ensemble_hub: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.n_step_ahead             = max(1, int(n_step_ahead))
        self.window                   = max(10, int(window))
        self.min_samples              = max(1, int(min_samples))
        self.decay                    = float(max(0.50, min(0.9999, decay)))
        self.low_quality_threshold    = float(low_quality_threshold)
        self.update_interval_cycles   = max(1, int(update_interval_cycles))
        self.ensemble_hub             = ensemble_hub
        self.config                   = config

        # Pending predictions: (cycle_number, source, direction, reference_price)
        self._pending: Deque[Tuple[int, str, int, float]] = deque(maxlen=5000)

        # EMA accuracy per source  {source: ema_value}
        self._ema_accuracy: Dict[str, float] = {}

        # Sample counts per source
        self._sample_counts: Dict[str, int] = {}

        self._last_report: Optional[SignalQualityReport] = None
        self._last_update_cycle: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def record_signals(
        self,
        advisory: Dict[str, Any],
        price: float,
        cycle: int,
    ) -> None:
        """
        Record current ensemble signals for future evaluation.
        Also evaluates any pending predictions that have matured.
        """
        if price is None or price <= 0:
            return

        # ── Evaluate matured predictions ──────────────────────────────────
        self._evaluate_matured(price, cycle)

        # ── Record new predictions from ensemble sources ──────────────────
        _ens = advisory.get("ensemble") or {}
        source_signals = _ens.get("source_signals") or {}  # {source: direction_sign}

        # Fallback: if ensemble has a "sources" list with direction fields
        if not source_signals:
            sources_list = _ens.get("sources") or []
            for src_entry in sources_list:
                if isinstance(src_entry, dict):
                    src_name = str(src_entry.get("name", "unknown"))
                    direction = src_entry.get("direction", 0)
                    if direction is not None:
                        source_signals[src_name] = int(direction)

        for source, direction in source_signals.items():
            if direction in (1, -1, 0):
                self._pending.append((cycle, str(source), int(direction), float(price)))

    def get_report(self) -> SignalQualityReport:
        """Return current signal quality report."""
        if not self._ema_accuracy:
            return SignalQualityReport(
                source_accuracy={},
                recommended_weights={},
                low_quality_sources=[],
                avg_reliability=1.0,
                total_predictions=0,
            )
        return self._build_report()

    def record_price(self, price: float, cycle: int) -> None:
        """Convenience method: evaluate matured predictions with latest price."""
        if price and price > 0:
            self._evaluate_matured(float(price), cycle)

    def maybe_update_ensemble_weights(self, cycle: int) -> bool:
        """
        If enough cycles have passed, push updated weights to ensemble_hub.
        Returns True if weights were updated.
        """
        if self.ensemble_hub is None:
            return False
        if cycle - self._last_update_cycle < self.update_interval_cycles:
            return False
        if not self._ema_accuracy:
            return False

        report = self._build_report()
        try:
            self.ensemble_hub.update_source_weights(report.recommended_weights)
            self._last_update_cycle = cycle
            logger.debug(
                "SignalQualityTracker: updated ensemble weights for %d sources",
                len(report.recommended_weights),
            )
            return True
        except Exception as exc:
            logger.debug("SignalQualityTracker: weight update failed: %s", exc)
            return False

    def snapshot(self) -> Dict[str, Any]:
        # Always use live sample count so it reflects records even before get_report()
        total = sum(self._sample_counts.values())
        r = self._last_report
        if r is None:
            return {
                "source_accuracy": {},
                "recommended_weights": {},
                "low_quality_sources": [],
                "avg_reliability": 1.0,
                "total_predictions": total,
            }
        return {
            "source_accuracy":     dict(r.source_accuracy),
            "recommended_weights": dict(r.recommended_weights),
            "low_quality_sources": list(r.low_quality_sources),
            "avg_reliability":     r.avg_reliability,
            "total_predictions":   total,  # live count
            "ts":                  r.ts,
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _evaluate_matured(self, current_price: float, current_cycle: int) -> None:
        """Score any pending predictions that have matured (>= n_step_ahead cycles)."""
        remaining: Deque[Tuple[int, str, int, float]] = deque(maxlen=5000)
        for entry in self._pending:
            pred_cycle, source, direction, ref_price = entry
            if current_cycle - pred_cycle >= self.n_step_ahead:
                # Evaluate: did price move in predicted direction?
                if ref_price > 0 and direction != 0:
                    price_change = current_price - ref_price
                    hit = 1.0 if (price_change * direction) > 0 else 0.0
                    self._update_ema(source, hit)
            else:
                remaining.append(entry)
        self._pending = remaining

    def _update_ema(self, source: str, hit: float) -> None:
        """Update EMA accuracy for a source."""
        if source not in self._ema_accuracy:
            self._ema_accuracy[source] = hit
            self._sample_counts[source] = 1
        else:
            self._ema_accuracy[source] = (
                self.decay * self._ema_accuracy[source]
                + (1.0 - self.decay) * hit
            )
            self._sample_counts[source] = self._sample_counts.get(source, 0) + 1

    def _build_report(self) -> SignalQualityReport:
        """Build a SignalQualityReport from current EMA accuracy values."""
        # Only include sources with enough samples
        valid = {
            src: acc
            for src, acc in self._ema_accuracy.items()
            if self._sample_counts.get(src, 0) >= self.min_samples
        }

        if not valid:
            rpt = SignalQualityReport(
                source_accuracy={},
                recommended_weights={},
                low_quality_sources=[],
                avg_reliability=1.0,
                total_predictions=sum(self._sample_counts.values()),
            )
            self._last_report = rpt
            return rpt

        low_quality = [
            src for src, acc in valid.items()
            if acc < self.low_quality_threshold
        ]

        avg_reliability = sum(valid.values()) / len(valid)

        # Normalised weights — proportional to accuracy, floor at 0.05
        total_acc = sum(max(0.05, acc) for acc in valid.values())
        if total_acc > 0:
            weights = {
                src: round(max(0.05, acc) / total_acc, 4)
                for src, acc in valid.items()
            }
        else:
            n = len(valid)
            weights = {src: round(1.0 / n, 4) for src in valid}

        rpt = SignalQualityReport(
            source_accuracy    = {src: round(acc, 4) for src, acc in valid.items()},
            recommended_weights= weights,
            low_quality_sources= low_quality,
            avg_reliability    = round(avg_reliability, 4),
            total_predictions  = sum(self._sample_counts.values()),
        )
        self._last_report = rpt
        return rpt
