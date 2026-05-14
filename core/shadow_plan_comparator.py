"""
Shadow Plan Comparator — Compare shadow (hypothetical) execution plans vs live.

Runs every cycle alongside the live trading pipeline. For each signal:
  1. Records what the live pipeline actually did (size, gates, outcome)
  2. Records what an alternative configuration would have done (shadow)
  3. Computes drift metrics between them

Use cases:
  - Detect execution quality degradation (shadow outperforms live consistently)
  - A/B test new gate configurations before deploying to live
  - Validate that intelligence gates improve rather than hurt performance
  - Track conviction sizer accuracy over time

The comparator does NOT place orders — it only records hypothetical decisions.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanSnapshot:
    """Snapshot of one signal's execution plan (live or shadow)."""
    symbol: str
    side: str
    strategy: str
    confidence: float
    size_pct: float            # final position size as % of portfolio
    size_aud: float            # final AUD amount
    gate_multiplier: float     # product of all gates
    gates_applied: int         # number of gates that modified size
    gates_blocked: bool        # True if any gate blocked entirely
    block_reason: str = ""
    regime: str = ""
    conviction: float = 0.0
    timestamp_ms: int = 0


@dataclass
class ComparisonResult:
    """Comparison between live and shadow plans for one signal."""
    symbol: str
    strategy: str
    cycle: int
    timestamp_ms: int

    # Live plan
    live_size_pct: float
    live_size_aud: float
    live_gate_mult: float
    live_blocked: bool

    # Shadow plan
    shadow_size_pct: float
    shadow_size_aud: float
    shadow_gate_mult: float
    shadow_blocked: bool

    # Drift metrics
    size_drift_pct: float      # (shadow - live) / live * 100
    gate_drift: float          # shadow_gate_mult - live_gate_mult
    agreement: bool            # both blocked or both passed
    shadow_would_trade: bool   # shadow says trade, live says no

    # After fill (updated later)
    live_pnl: float = 0.0
    shadow_pnl_estimate: float = 0.0


@dataclass
class DriftReport:
    """Aggregate drift statistics over a window."""
    total_comparisons: int = 0
    agreement_rate: float = 0.0           # % where both agree on block/pass
    avg_size_drift_pct: float = 0.0       # average size difference
    avg_gate_drift: float = 0.0           # average gate multiplier difference
    shadow_would_trade_count: int = 0     # times shadow would trade but live blocked
    live_would_trade_count: int = 0       # times live traded but shadow wouldn't
    shadow_advantage_pnl: float = 0.0     # shadow PnL - live PnL (positive = shadow better)
    symbols_with_drift: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Shadow Plan Comparator
# ─────────────────────────────────────────────────────────────────────────────

class ShadowPlanComparator:
    """
    Records live and shadow execution plans side-by-side,
    computes drift metrics, and flags when shadow consistently outperforms.

    Usage::

        comparator = ShadowPlanComparator(window=500)

        # In _execute_signals, for each signal:
        live_snap = PlanSnapshot(symbol="BTC/USD", side="BUY", ...)
        shadow_snap = comparator.compute_shadow(signal, advisory, shadow_config)
        comparator.record(live_snap, shadow_snap, cycle=42)

        # Periodic check:
        report = comparator.drift_report()
        if report.shadow_advantage_pnl > threshold:
            logger.warning("Shadow config outperforms live by %.2f AUD", ...)
    """

    def __init__(self, window: int = 500) -> None:
        self.window = window
        self._history: deque[ComparisonResult] = deque(maxlen=window)
        self._total_comparisons: int = 0
        self._shadow_pnl_sum: float = 0.0
        self._live_pnl_sum: float = 0.0
        logger.info("ShadowPlanComparator: initialized (window=%d)", window)

    def record(
        self,
        live: PlanSnapshot,
        shadow: PlanSnapshot,
        cycle: int,
    ) -> ComparisonResult:
        """
        Record a live vs shadow comparison for one signal.
        Returns the ComparisonResult.
        """
        ts = int(time.time() * 1000)

        # Size drift: how different is shadow sizing vs live
        if live.size_pct > 0:
            size_drift = (shadow.size_pct - live.size_pct) / live.size_pct * 100.0
        elif shadow.size_pct > 0:
            size_drift = 100.0  # shadow would trade, live wouldn't
        else:
            size_drift = 0.0

        gate_drift = shadow.gate_multiplier - live.gate_multiplier
        agreement = (live.gates_blocked == shadow.gates_blocked)
        shadow_would_trade = (not shadow.gates_blocked and live.gates_blocked)

        result = ComparisonResult(
            symbol=live.symbol,
            strategy=live.strategy,
            cycle=cycle,
            timestamp_ms=ts,
            live_size_pct=live.size_pct,
            live_size_aud=live.size_aud,
            live_gate_mult=live.gate_multiplier,
            live_blocked=live.gates_blocked,
            shadow_size_pct=shadow.size_pct,
            shadow_size_aud=shadow.size_aud,
            shadow_gate_mult=shadow.gate_multiplier,
            shadow_blocked=shadow.gates_blocked,
            size_drift_pct=size_drift,
            gate_drift=gate_drift,
            agreement=agreement,
            shadow_would_trade=shadow_would_trade,
        )

        self._history.append(result)
        self._total_comparisons += 1

        # Log significant disagreements
        if shadow_would_trade:
            logger.info(
                "ShadowComparator: shadow would trade %s %s (live blocked) — "
                "shadow_size=%.2f%%, gate_mult=%.3f",
                live.side, live.symbol, shadow.size_pct * 100, shadow.gate_multiplier,
            )

        return result

    def update_pnl(self, cycle: int, symbol: str, live_pnl: float, shadow_pnl: float) -> None:
        """
        Update P&L for a completed trade. Call from on_fill.
        Matches by cycle+symbol (most recent match).
        """
        for comp in reversed(self._history):
            if comp.cycle == cycle and comp.symbol == symbol:
                comp.live_pnl = live_pnl
                comp.shadow_pnl_estimate = shadow_pnl
                self._live_pnl_sum += live_pnl
                self._shadow_pnl_sum += shadow_pnl
                break

    def compute_shadow(
        self,
        signal: Any,
        advisory: Dict[str, Any],
        shadow_config: Dict[str, Any],
    ) -> PlanSnapshot:
        """
        Compute what a shadow (alternative) configuration would produce.

        shadow_config keys:
          - gate_floor: float (minimum gate multiplier, e.g. 0.15)
          - skip_gates: list of gate names to skip
          - conviction_override: float (override conviction score)
          - size_multiplier: float (global size adjustment)

        This is a simplified shadow — it doesn't rerun the full gate chain,
        but estimates what the shadow configuration would produce.
        """
        symbol = str(getattr(signal, "symbol", "UNKNOWN"))
        side = str(getattr(signal, "action", "UNKNOWN")).upper()
        strategy = str(getattr(signal, "source_strategy", "unknown"))
        confidence = float(getattr(signal, "confidence", 0.5))

        # Shadow sizing: start from base config
        gate_floor = shadow_config.get("gate_floor", 0.15)
        skip_gates = set(shadow_config.get("skip_gates", []))
        size_mult = shadow_config.get("size_multiplier", 1.0)
        conviction_override = shadow_config.get("conviction_override", None)

        # Base size from confidence
        base_size = confidence * 0.05  # 5% max at confidence=1.0
        base_size *= size_mult

        # Apply simplified gate chain (skip specified gates)
        shadow_gate_mult = 1.0
        gates_applied = 0
        blocked = False

        # Check advisory for gate decisions
        meta_gate = advisory.get("trade_gate", {})
        if meta_gate and "meta_gate" not in skip_gates:
            decision = meta_gate.get("decision", "ALLOW")
            if decision == "HALT":
                blocked = True
            elif decision == "PAUSE" and side == "BUY":
                blocked = True
            elif decision == "REDUCE":
                shadow_gate_mult *= 0.5
                gates_applied += 1

        escalating = advisory.get("escalating_gate", {})
        if escalating and "escalating_gate" not in skip_gates:
            if escalating.get("was_escalated"):
                esc_decision = escalating.get("decision", "ALLOW")
                if esc_decision == "HALT":
                    blocked = True
                elif esc_decision == "PAUSE" and side == "BUY":
                    blocked = True
                elif esc_decision == "REDUCE":
                    shadow_gate_mult *= 0.5
                    gates_applied += 1

        # Apply floor
        if shadow_gate_mult < gate_floor and not blocked:
            shadow_gate_mult = gate_floor

        final_size = base_size * shadow_gate_mult
        conviction = conviction_override if conviction_override is not None else confidence

        return PlanSnapshot(
            symbol=symbol,
            side=side,
            strategy=strategy,
            confidence=confidence,
            size_pct=final_size if not blocked else 0.0,
            size_aud=0.0,  # computed later with portfolio value
            gate_multiplier=shadow_gate_mult,
            gates_applied=gates_applied,
            gates_blocked=blocked,
            block_reason="shadow_gate_blocked" if blocked else "",
            regime=str(advisory.get("regime_label", "")),
            conviction=conviction,
            timestamp_ms=int(time.time() * 1000),
        )

    def drift_report(self) -> DriftReport:
        """
        Compute aggregate drift statistics over the sliding window.
        """
        history = list(self._history)
        if not history:
            return DriftReport()

        n = len(history)
        agreements = sum(1 for c in history if c.agreement)
        size_drifts = [c.size_drift_pct for c in history if not c.live_blocked]
        gate_drifts = [c.gate_drift for c in history]
        shadow_trades = sum(1 for c in history if c.shadow_would_trade)
        live_only = sum(
            1 for c in history
            if not c.live_blocked and c.shadow_blocked
        )

        # Symbols with consistent drift (>10% average size difference)
        sym_drifts: Dict[str, List[float]] = {}
        for c in history:
            if c.symbol not in sym_drifts:
                sym_drifts[c.symbol] = []
            sym_drifts[c.symbol].append(c.size_drift_pct)
        drifting = [
            sym for sym, drifts in sym_drifts.items()
            if abs(sum(drifts) / len(drifts)) > 10.0
        ]

        return DriftReport(
            total_comparisons=n,
            agreement_rate=agreements / n if n else 0.0,
            avg_size_drift_pct=(
                sum(size_drifts) / len(size_drifts) if size_drifts else 0.0
            ),
            avg_gate_drift=sum(gate_drifts) / n if n else 0.0,
            shadow_would_trade_count=shadow_trades,
            live_would_trade_count=live_only,
            shadow_advantage_pnl=self._shadow_pnl_sum - self._live_pnl_sum,
            symbols_with_drift=drifting,
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return current state as a dict for advisory/dashboard."""
        report = self.drift_report()
        return {
            "total_comparisons": report.total_comparisons,
            "agreement_rate": round(report.agreement_rate, 3),
            "avg_size_drift_pct": round(report.avg_size_drift_pct, 2),
            "avg_gate_drift": round(report.avg_gate_drift, 4),
            "shadow_would_trade": report.shadow_would_trade_count,
            "shadow_advantage_pnl": round(report.shadow_advantage_pnl, 2),
            "symbols_with_drift": report.symbols_with_drift,
        }
