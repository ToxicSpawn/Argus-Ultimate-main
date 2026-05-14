"""
MetaGate — Per-cycle "Should I trade?" decision gate.

Synthesises four self-knowledge signals into a single TradeGateDecision:
  1. regime_confidence  — how certain the ensemble is about current regime (0–1)
  2. staleness_score    — fraction of ML models that are stale (0–1)
  3. recent_sharpe      — rolling short-term Sharpe ratio (can be negative)
  4. drawdown_fraction  — current drawdown / max_drawdown_limit (0–1)

Decision hierarchy (first match wins):
  drawdown_fraction >= halt_dd_frac   → HALT
  drawdown_fraction >= pause_dd_frac  → PAUSE
  regime_confidence < 0.30 AND staleness > 0.70 → PAUSE
  regime_confidence < conf_threshold OR recent_sharpe < sharpe_floor → REDUCE
  staleness > staleness_threshold → REDUCE
  else → ALLOW
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Decision enum ─────────────────────────────────────────────────────────────

class TradeGateDecision(Enum):
    ALLOW  = "allow"   # trade at full size
    REDUCE = "reduce"  # trade at 50% size
    PAUSE  = "pause"   # no new positions; hold existing
    HALT   = "halt"    # same as circuit breaker — close all / no new trades


# ── MetaGate ──────────────────────────────────────────────────────────────────

@dataclass
class _GateRecord:
    decision: TradeGateDecision
    reasons: List[str]
    ts: float


class MetaGate:
    """
    Evaluates four scalar inputs and returns a TradeGateDecision.

    All thresholds are soft — can be overridden at construction time so
    they can be tuned via the YAML config without code changes.
    """

    def __init__(
        self,
        regime_conf_threshold: float = 0.45,
        regime_conf_panic: float = 0.30,
        staleness_threshold: float = 0.70,
        min_recent_sharpe: float = -0.50,
        pause_dd_fraction: float = 0.85,
        halt_dd_fraction: float = 0.95,
    ) -> None:
        self.regime_conf_threshold = float(regime_conf_threshold)
        self.regime_conf_panic = float(regime_conf_panic)
        self.staleness_threshold = float(staleness_threshold)
        self.min_recent_sharpe = float(min_recent_sharpe)
        self.pause_dd_fraction = float(pause_dd_fraction)
        self.halt_dd_fraction = float(halt_dd_fraction)

        self._last: Optional[_GateRecord] = None
        self._decision_counts: Dict[str, int] = {d.value: 0 for d in TradeGateDecision}

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        regime_confidence: float,
        staleness_score: float,
        recent_sharpe: float,
        drawdown_fraction: float,
    ) -> TradeGateDecision:
        """
        Evaluate current system state and return a gate decision.

        Parameters
        ----------
        regime_confidence : 0.0–1.0   (from RegimeEnsemble.snapshot()["confidence"])
        staleness_score   : 0.0–1.0   (stale_models / total_models)
        recent_sharpe     : float      (can be negative; rolling ~20-trade Sharpe)
        drawdown_fraction : 0.0–1.0   (current_drawdown / max_drawdown_limit)
        """
        rc  = float(regime_confidence)
        ss  = float(staleness_score)
        rs  = float(recent_sharpe)
        ddf = float(drawdown_fraction)

        decision = TradeGateDecision.ALLOW
        reasons: List[str] = []

        # 1. Drawdown-based hard stops (highest priority)
        if ddf >= self.halt_dd_fraction:
            decision = TradeGateDecision.HALT
            reasons.append(f"drawdown_fraction={ddf:.2f} >= halt_threshold={self.halt_dd_fraction:.2f}")

        elif ddf >= self.pause_dd_fraction:
            decision = TradeGateDecision.PAUSE
            reasons.append(f"drawdown_fraction={ddf:.2f} >= pause_threshold={self.pause_dd_fraction:.2f}")

        # 2. Regime panic + model staleness → PAUSE
        elif rc < self.regime_conf_panic and ss > self.staleness_threshold:
            decision = TradeGateDecision.PAUSE
            reasons.append(
                f"regime_confidence={rc:.2f} < panic={self.regime_conf_panic:.2f} "
                f"AND staleness={ss:.2f} > {self.staleness_threshold:.2f}"
            )

        # 3. Single-factor REDUCE
        elif rc < self.regime_conf_threshold:
            decision = TradeGateDecision.REDUCE
            reasons.append(f"regime_confidence={rc:.2f} < threshold={self.regime_conf_threshold:.2f}")

        elif rs < self.min_recent_sharpe:
            decision = TradeGateDecision.REDUCE
            reasons.append(f"recent_sharpe={rs:.3f} < floor={self.min_recent_sharpe:.3f}")

        elif ss > self.staleness_threshold:
            decision = TradeGateDecision.REDUCE
            reasons.append(f"staleness_score={ss:.2f} > {self.staleness_threshold:.2f}")

        self._last = _GateRecord(decision=decision, reasons=reasons, ts=time.time())
        self._decision_counts[decision.value] = self._decision_counts.get(decision.value, 0) + 1

        if decision != TradeGateDecision.ALLOW:
            logger.info(
                "MetaGate: %s — %s",
                decision.value.upper(),
                "; ".join(reasons),
            )

        return decision

    @property
    def last_decision(self) -> Optional[TradeGateDecision]:
        return self._last.decision if self._last else None

    @property
    def last_reasons(self) -> List[str]:
        return list(self._last.reasons) if self._last else []

    def snapshot(self) -> Dict[str, Any]:
        last = self._last
        return {
            "decision": last.decision.value if last else "allow",
            "reasons": list(last.reasons) if last else [],
            "decision_counts": dict(self._decision_counts),
            "thresholds": {
                "regime_conf_threshold": self.regime_conf_threshold,
                "regime_conf_panic":     self.regime_conf_panic,
                "staleness_threshold":   self.staleness_threshold,
                "min_recent_sharpe":     self.min_recent_sharpe,
                "pause_dd_fraction":     self.pause_dd_fraction,
                "halt_dd_fraction":      self.halt_dd_fraction,
            },
            "ts": last.ts if last else None,
        }
