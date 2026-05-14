"""
SelfIntelligenceOrchestrator — Master brain.

Reads ALL advisory signals and synthesises them into a single
IntelligenceDirective: one authoritative answer to
"what should ARGUS do right now?"

Quadrant mapping (health × critical count):
  health ≥ 80, 0 criticals → OPPORTUNISTIC / AGGRESSIVE / scale=1.00
  health 60–79, warnings ≤ 2   → NORMAL       / NORMAL     / scale=0.85
  health 40–59, OR ≥ 1 critical → CONSERVATIVE / PATIENT    / scale=0.60
  health < 40,  OR halt/budget  → CONSERVATIVE / PASSIVE    / scale=0.25

Execution modes (how aggressively to chase fills):
  AGGRESSIVE — market orders, full size
  NORMAL     — limit-at-mid preferred
  PATIENT    — wide limit, wait for fill
  PASSIVE    — TWAP / minimal market impact

Output: advisory["intelligence_directive"]
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExecutionMode(str, Enum):
    AGGRESSIVE = "aggressive"
    NORMAL     = "normal"
    PATIENT    = "patient"
    PASSIVE    = "passive"


class RiskMode(str, Enum):
    CONSERVATIVE  = "conservative"
    NORMAL        = "normal"
    OPPORTUNISTIC = "opportunistic"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceDirective:
    position_scale: float          # 0.0–1.5 global size multiplier
    max_new_positions: int         # 0 = halt all new entries
    execution_mode: ExecutionMode
    risk_mode: RiskMode
    confidence: float              # 0.0–1.0 overall confidence
    insights: List[str]            # 3–5 human-readable bullets
    override_count: int            # how many sub-systems are degraded
    health_score: int              # raw score that drove this decision
    health_label: str
    ts: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# SelfIntelligenceOrchestrator
# ---------------------------------------------------------------------------

class SelfIntelligenceOrchestrator:
    """
    Master brain — synthesises all advisory signals into one directive.

    Parameters
    ----------
    health_opportunistic_floor : int
        Health score at/above which we enter OPPORTUNISTIC mode (default 80).
    health_conservative_ceiling : int
        Health score below which we switch to CONSERVATIVE mode (default 40).
    max_new_positions_normal : int
        Maximum concurrent new positions in NORMAL mode (default 5).
    max_new_positions_opportunistic : int
        Maximum in OPPORTUNISTIC mode (default 8).
    config : optional config object
    """

    def __init__(
        self,
        health_opportunistic_floor: int = 80,
        health_conservative_ceiling: int = 40,
        max_new_positions_normal: int = 5,
        max_new_positions_opportunistic: int = 8,
        config: Optional[Any] = None,
    ) -> None:
        self.health_opportunistic_floor      = int(health_opportunistic_floor)
        self.health_conservative_ceiling     = int(health_conservative_ceiling)
        self.max_new_positions_normal        = int(max_new_positions_normal)
        self.max_new_positions_opportunistic = int(max_new_positions_opportunistic)
        self.config = config
        self._last: Optional[IntelligenceDirective] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def synthesize(self, advisory: Dict[str, Any]) -> IntelligenceDirective:
        """
        Read all advisory signals → IntelligenceDirective.
        Gracefully handles missing advisory keys.
        """
        insights: List[str] = []
        override_count: int = 0

        # ── 1. Health score ───────────────────────────────────────────────
        _hs = advisory.get("health_score") or {}
        health_score = int(_hs.get("score", 70) or 70)
        health_label = str(_hs.get("label", "GOOD") or "GOOD")

        # ── 2. Trade gate decision ────────────────────────────────────────
        _tg = advisory.get("trade_gate") or {}
        gate_decision = str(_tg.get("decision", "allow")).lower()
        is_halted  = gate_decision == "halt"
        is_paused  = gate_decision == "pause"
        is_reduced = gate_decision == "reduce"

        # ── 3. Critical / warning counts ──────────────────────────────────
        _sd = advisory.get("self_diagnosis") or {}
        critical_count = len(_sd.get("critical") or [])
        warning_count  = len(_sd.get("warnings") or [])

        _sa = advisory.get("system_status") or {}
        if str(_sa.get("status", "")).upper() in ("CRITICAL", "DEGRADED"):
            override_count += 1

        # ── 4. Consecutive-loss budget ────────────────────────────────────
        _clg = advisory.get("consecutive_loss_guard") or {}
        budget_exhausted = bool(_clg.get("daily_budget_exhausted", False))

        # ── 5. Transition risk ────────────────────────────────────────────
        _rt = advisory.get("regime_transition") or {}
        pre_hedge      = bool(_rt.get("pre_hedge_signal", False))
        transition_risk = float(_rt.get("transition_risk_score", 0.0) or 0.0)

        # ── 6. Regime / confidence ────────────────────────────────────────
        _re = advisory.get("regime_ensemble") or {}
        regime     = str(_re.get("regime", "UNKNOWN") or "UNKNOWN")
        regime_conf = float(_re.get("confidence", 0.5) or 0.5)

        # ── 6b. AI health signals ──────────────────────────────────────────
        _ah = advisory.get("ai_health") or {}
        ai_health_score  = float(_ah.get("ai_health_score", 100.0) or 100.0)
        ai_drifted       = list(_ah.get("drifted_models") or [])
        multi_drift      = bool(_ah.get("multi_model_drift", False))
        regime_agreement = float(_ah.get("regime_agreement", 1.0) or 1.0)
        retraining_queue = list(_ah.get("retraining_queue") or [])

        # Blend ensemble confidence with regime_conf for richer signal
        _ens = advisory.get("ensemble") or {}
        ens_conf = float(_ens.get("confidence", regime_conf) or regime_conf)
        blended_conf = (regime_conf + ens_conf) / 2.0

        # AI health degradation
        if ai_health_score < 50.0:
            override_count += 1

        # ── 7. EscalatingGate ─────────────────────────────────────────────
        _eg = advisory.get("escalating_gate") or {}
        eg_decision    = str(_eg.get("decision", "allow")).lower()
        was_escalated  = bool(_eg.get("was_escalated", False))
        if was_escalated:
            override_count += 1

        # ── 8. Edge monitor ───────────────────────────────────────────────
        _em = advisory.get("edge_monitor") or {}
        edge_degraded = bool(_em.get("is_degraded", False))
        edge_score    = float(_em.get("edge_score", 1.0) or 1.0)
        if edge_degraded:
            override_count += 1

        # ── 9. Signal quality ─────────────────────────────────────────────
        _sq = advisory.get("signal_quality") or {}
        avg_reliability = float(_sq.get("avg_reliability", 1.0) or 1.0)
        low_quality_sources = list(_sq.get("low_quality_sources") or [])
        if low_quality_sources:
            override_count += 1

        # ── 10. Self knowledge trend ──────────────────────────────────────
        _sk = advisory.get("self_knowledge") or {}
        health_trend = str(_sk.get("health_trend", "stable") or "stable")

        # ── 11. Risk intelligence aggregator ──────────────────────────────
        _ri = advisory.get("risk_intelligence") or {}
        risk_score_composite = float(_ri.get("risk_score_composite", 75.0) or 75.0)
        tail_hedge_urgency   = float(_ri.get("tail_hedge_urgency",   0.0)  or 0.0)
        stress_survival_rate = float(_ri.get("stress_survival_rate", 1.0)  or 1.0)
        cascade_risk         = float(_ri.get("cascade_risk",         0.0)  or 0.0)
        var_breach           = bool(_ri.get("var_breach", False))

        # ── 12. Signal intelligence aggregator ────────────────────────────
        _si = advisory.get("signal_intelligence") or {}
        signal_conviction  = float(_si.get("signal_conviction",  0.5)      or 0.5)
        quantum_anomaly    = bool(_si.get("quantum_anomaly", False))
        fear_greed_regime  = str(_si.get("fear_greed_regime", "NEUTRAL")   or "NEUTRAL")
        model_agreement    = float(_si.get("model_agreement",    0.5)       or 0.5)
        # Enhance blended_conf with model_agreement for richer signal
        blended_conf = (blended_conf + model_agreement) / 2.0

        # ── 13. Strategy intelligence aggregator ──────────────────────────
        _sti = advisory.get("strategy_intelligence") or {}
        bleeder_count             = int(_sti.get("bleeder_count",   0) or 0)
        validation_urgent         = bool(_sti.get("validation_urgent", False))
        strategy_health_composite = float(_sti.get("strategy_health_composite", 75.0) or 75.0)
        funding_opportunity_score = float(_sti.get("funding_opportunity_score", 0.0)  or 0.0)

        # ── 14. TCA score (direct) ────────────────────────────────────────
        tca_score = float(advisory.get("tca_score") or 100.0)

        # ── Blend confidence richer: include model_agreement ──────────────
        # (Applied after blended_conf is computed below, stored into blended_conf)

        # ── Derive quadrant ───────────────────────────────────────────────
        force_halt = (
            is_halted
            or eg_decision == "halt"
            or budget_exhausted
            or health_score < self.health_conservative_ceiling
            or tail_hedge_urgency > 0.90   # imminent tail event
        )
        force_conservative = (
            critical_count >= 1
            or is_paused
            or eg_decision == "pause"
            or pre_hedge
            or health_score < 60
            or multi_drift              # multi-model drift forces conservative
            or cascade_risk > 0.60      # causal cascade likely
            or stress_survival_rate < 0.70  # stress tests failing
            or var_breach               # VaR limit breached
            or validation_urgent        # models need urgent retraining
            or quantum_anomaly          # quantum anomaly detected
        )

        if force_halt:
            position_scale     = 0.25
            max_new_positions  = 0
            execution_mode     = ExecutionMode.PASSIVE
            risk_mode          = RiskMode.CONSERVATIVE
            confidence         = max(0.05, blended_conf * 0.3)
            insights.append(
                f"HALT: health={health_score} ({health_label}), "
                f"gate={gate_decision}, budget_exhausted={budget_exhausted}"
            )
        elif force_conservative:
            position_scale     = 0.60
            max_new_positions  = max(1, self.max_new_positions_normal // 2)
            execution_mode     = ExecutionMode.PATIENT
            risk_mode          = RiskMode.CONSERVATIVE
            confidence         = max(0.20, blended_conf * 0.6)
            insights.append(
                f"CONSERVATIVE: health={health_score}, criticals={critical_count}, "
                f"pre_hedge={pre_hedge}"
            )
        elif health_score >= self.health_opportunistic_floor and warning_count == 0:
            position_scale     = min(1.50, 1.0 + max(0, transition_risk) * 0)
            max_new_positions  = self.max_new_positions_opportunistic
            execution_mode     = ExecutionMode.AGGRESSIVE
            risk_mode          = RiskMode.OPPORTUNISTIC
            confidence         = min(1.0, blended_conf * 1.1)
            insights.append(
                f"OPPORTUNISTIC: health={health_score} ({health_label}), "
                f"regime={regime} ({blended_conf:.2f})"
            )
        elif warning_count <= 2:
            position_scale     = 0.85
            max_new_positions  = self.max_new_positions_normal
            execution_mode     = ExecutionMode.NORMAL
            risk_mode          = RiskMode.NORMAL
            confidence         = blended_conf
            insights.append(
                f"NORMAL: health={health_score}, warnings={warning_count}, "
                f"regime={regime}"
            )
        else:
            # health 60–79 with > 2 warnings
            position_scale     = 0.70
            max_new_positions  = max(2, self.max_new_positions_normal - 1)
            execution_mode     = ExecutionMode.PATIENT
            risk_mode          = RiskMode.NORMAL
            confidence         = blended_conf * 0.85
            insights.append(
                f"CAUTIOUS: health={health_score}, warnings={warning_count}"
            )

        # ── Modifiers ─────────────────────────────────────────────────────
        if is_reduced:
            position_scale = min(position_scale, 0.70)
            insights.append("MetaGate REDUCE active — size capped at 70%")

        if edge_degraded:
            position_scale = min(position_scale, position_scale * 0.85)
            insights.append(f"Edge degraded (score={edge_score:.2f}) — scale trimmed")

        if pre_hedge and not force_halt:
            scale_mod = max(0.10, 1.0 - transition_risk)
            position_scale *= scale_mod
            insights.append(
                f"Pre-hedge active (risk={transition_risk:.2f}) — scale ×{scale_mod:.2f}"
            )

        if avg_reliability < 0.55:
            position_scale *= 0.90
            insights.append(
                f"Signal reliability low ({avg_reliability:.2f}) — scale trimmed 10%"
            )

        if health_trend == "deteriorating":
            position_scale *= 0.95
            insights.append("Health trend deteriorating — 5% precautionary trim")

        # ── Risk intelligence modifiers ────────────────────────────────────
        if stress_survival_rate < 0.80:
            position_scale *= 0.85
            insights.append(
                f"Stress survival low ({stress_survival_rate:.0%}) — scale trimmed 15%"
            )

        if tail_hedge_urgency > 0.50 and not force_halt:
            scale_mod = max(0.10, 1.0 - tail_hedge_urgency)
            position_scale *= scale_mod
            insights.append(
                f"Tail-hedge urgency ({tail_hedge_urgency:.2f}) — scale ×{scale_mod:.2f}"
            )

        if cascade_risk > 0.40:
            scale_mod = 1.0 - cascade_risk * 0.30
            position_scale *= scale_mod
            insights.append(
                f"Cascade risk {cascade_risk:.2f} — scale trimmed"
            )

        # ── Signal intelligence modifiers ──────────────────────────────────
        if signal_conviction < 0.30:
            position_scale *= 0.90
            insights.append(
                f"Signal conviction low ({signal_conviction:.2f}) — scale trimmed 10%"
            )

        if tca_score < 50.0:
            position_scale *= 0.92
            insights.append(
                f"TCA score poor ({tca_score:.0f}/100) — scale trimmed 8%"
            )

        if fear_greed_regime == "EXTREME_GREED" and risk_mode == RiskMode.OPPORTUNISTIC:
            position_scale *= 0.95
            insights.append("Extreme greed — euphoria guard, scale trimmed 5%")
        elif fear_greed_regime == "EXTREME_FEAR" and risk_mode != RiskMode.CONSERVATIVE:
            position_scale = min(1.50, position_scale * 1.05)
            insights.append("Extreme fear — contrarian bonus +5%")

        # ── Strategy intelligence modifiers ────────────────────────────────
        if bleeder_count > 2:
            max_new_positions = max(1, max_new_positions - bleeder_count)
            insights.append(
                f"Bleeders detected ({bleeder_count}) — max positions reduced"
            )

        if validation_urgent:
            max_new_positions = max(0, max_new_positions - 2)
            insights.append("Urgent model retraining needed — positions cut")

        if funding_opportunity_score > 0.70 and risk_mode == RiskMode.OPPORTUNISTIC:
            insights.append(
                f"Funding opportunity ({funding_opportunity_score:.0%}) — consider harvesting"
            )

        # ── AI health modifiers ────────────────────────────────────────────
        if ai_health_score < 50.0:
            position_scale *= 0.85
            insights.append(
                f"AI health degraded ({ai_health_score:.0f}/100) — scale trimmed 15%"
            )

        if ai_drifted:
            insights.append(
                f"Model drift: {', '.join(ai_drifted[:3])}"
                + (f" (+{len(ai_drifted)-3} more)" if len(ai_drifted) > 3 else "")
            )

        if regime_agreement < 0.50:
            insights.append(
                f"Regime disagreement: sources={regime_agreement:.0%} agreement"
            )

        if retraining_queue:
            insights.append(
                f"Retraining queue: {', '.join(retraining_queue[:3])}"
            )

        # ── Clamp ─────────────────────────────────────────────────────────
        position_scale = round(max(0.0, min(1.50, position_scale)), 4)
        confidence     = round(max(0.0, min(1.0,  confidence)),     4)

        directive = IntelligenceDirective(
            position_scale    = position_scale,
            max_new_positions = max_new_positions,
            execution_mode    = execution_mode,
            risk_mode         = risk_mode,
            confidence        = confidence,
            insights          = insights[:5],
            override_count    = override_count,
            health_score      = health_score,
            health_label      = health_label,
        )
        self._last = directive
        return directive

    def snapshot(self) -> Dict[str, Any]:
        d = self._last
        if d is None:
            return {
                "position_scale": 1.0,
                "max_new_positions": 5,
                "execution_mode": ExecutionMode.NORMAL.value,
                "risk_mode": RiskMode.NORMAL.value,
                "confidence": 1.0,
                "insights": [],
                "override_count": 0,
                "health_score": 100,
                "health_label": "PEAK",
            }
        return {
            "position_scale":    d.position_scale,
            "max_new_positions": d.max_new_positions,
            "execution_mode":    d.execution_mode.value,
            "risk_mode":         d.risk_mode.value,
            "confidence":        d.confidence,
            "insights":          d.insights,
            "override_count":    d.override_count,
            "health_score":      d.health_score,
            "health_label":      d.health_label,
            "ts":                d.ts,
        }
