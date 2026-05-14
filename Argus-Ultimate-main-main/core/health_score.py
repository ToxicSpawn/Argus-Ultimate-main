"""
HealthScoreComputer — Single 0-100 composite system health metric.

Aggregates all advisory signals into one number:

  Component              Weight   Source key
  ──────────────────────────────────────────────────────────────────
  meta_gate              25%      advisory["trade_gate"]["decision"]
  edge_score             20%      advisory["edge_monitor"]["edge_score"]
  drawdown               20%      advisory["drawdown_controller"]["drawdown_pct"]
  regime_confidence      15%      advisory["regime_ensemble"]["confidence"]
  model_staleness        10%      advisory["trade_gate"]["staleness_score"]
  strategy_health        10%      advisory["strategy_enforcer"]["on_probation"] vs active count

When advisory["regime_transition"] is present, an additional
  transition_risk        10%      1 - advisory["regime_transition"]["transition_risk_score"]
component is added and all other weights are scaled by 0.9 so the
total remains 1.0.

When advisory["ai_health"] is present, an additional
  ai_model_health        10%      advisory["ai_health"]["ai_health_score"] / 100
component is added.

When advisory["risk_intelligence"] is present, an additional
  risk_intelligence       8%      advisory["risk_intelligence"]["risk_score_composite"] / 100
component is added.

When advisory["signal_intelligence"] is present, an additional
  signal_intelligence     7%      advisory["signal_intelligence"]["signal_conviction"] × 100
component is added.

Dynamic weight normalisation: base weights are scaled by
  (1.0 − sum_of_optional_weights_present)
so the grand total always equals exactly 1.0 regardless of which
optional components are active (0–4 of them).

Labels:
  80–100 = PEAK
  60–79  = GOOD
  40–59  = MARGINAL
  20–39  = POOR
  0–19   = CRITICAL

Output: advisory["health_score"]
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Gate decision → component score (0–100)
_GATE_SCORES: Dict[str, float] = {
    "allow":  100.0,
    "reduce":  60.0,
    "pause":   30.0,
    "halt":     0.0,
}

# Label thresholds (descending)
_LABELS = [
    (80, "PEAK"),
    (60, "GOOD"),
    (40, "MARGINAL"),
    (20, "POOR"),
    (0,  "CRITICAL"),
]

# Base weights (sum = 1.0)
_BASE_WEIGHTS: Dict[str, float] = {
    "meta_gate":         0.25,
    "edge_score":        0.20,
    "drawdown":          0.20,
    "regime_confidence": 0.15,
    "model_staleness":   0.10,
    "strategy_health":   0.10,
}
_TRANSITION_WEIGHT  = 0.10  # added when transition_risk key present
_AI_HEALTH_WEIGHT   = 0.10  # added when ai_health key present
_RISK_INTEL_WEIGHT  = 0.08  # added when risk_intelligence key present
_SIG_INTEL_WEIGHT   = 0.07  # added when signal_intelligence key present


@dataclass
class HealthScore:
    score: int                    # 0–100
    label: str                    # PEAK/GOOD/MARGINAL/POOR/CRITICAL
    breakdown: Dict[str, float]   # {component: weighted_contribution 0–100}
    weights_used: Dict[str, float]
    ts: float = field(default_factory=time.time)


class HealthScoreComputer:
    """
    Computes a composite 0-100 health score from the cycle advisory dict.

    Parameters
    ----------
    config : optional config object for threshold overrides
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self._max_dd_pct = float(
            getattr(config, "max_drawdown_pct", 0.12) or 0.12
        ) * 100.0  # convert to percent
        self._last: Optional[HealthScore] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, advisory: Dict[str, Any]) -> HealthScore:
        """
        Build a HealthScore from the current advisory dict.
        Gracefully handles missing keys — missing data contributes neutral score.
        """
        components: Dict[str, float] = {}

        # ── 1. MetaGate decision ──────────────────────────────────────────────
        _tg = (advisory.get("trade_gate") or {})
        gate_dec = str(_tg.get("decision", "allow")).lower()
        components["meta_gate"] = _GATE_SCORES.get(gate_dec, 60.0)

        # ── 2. Edge score ─────────────────────────────────────────────────────
        _em = (advisory.get("edge_monitor") or {})
        edge_sc = float(_em.get("edge_score", 1.0) or 1.0)
        components["edge_score"] = max(0.0, min(100.0, edge_sc * 100.0))

        # ── 3. Drawdown ───────────────────────────────────────────────────────
        _dc = (advisory.get("drawdown_controller") or {})
        dd_pct = float(_dc.get("drawdown_pct", 0.0) or 0.0)
        dd_score = max(0.0, 1.0 - dd_pct / max(self._max_dd_pct, 1.0)) * 100.0
        components["drawdown"] = dd_score

        # ── 4. Regime confidence ──────────────────────────────────────────────
        _re = (advisory.get("regime_ensemble") or {})
        reg_conf = float(_re.get("confidence", 0.5) or 0.5)
        components["regime_confidence"] = max(0.0, min(100.0, reg_conf * 100.0))

        # ── 5. Model staleness ────────────────────────────────────────────────
        staleness = float(_tg.get("staleness_score", 0.0) or 0.0)
        components["model_staleness"] = max(0.0, (1.0 - staleness) * 100.0)

        # ── 6. Strategy health ────────────────────────────────────────────────
        _se  = (advisory.get("strategy_enforcer") or {})
        _sr  = (advisory.get("strategy_router") or {})
        on_probation = len(_se.get("on_probation") or {})
        active       = len(_sr.get("active") or [])
        total        = active + on_probation
        if total > 0:
            strat_health = (active / total) * 100.0
        else:
            strat_health = 100.0  # no strategies registered — neutral
        components["strategy_health"] = strat_health

        # ── 7. Transition risk (optional) ─────────────────────────────────────
        _rt = (advisory.get("regime_transition") or {})
        has_transition = bool(_rt)
        if has_transition:
            risk = float(_rt.get("transition_risk_score", 0.0) or 0.0)
            components["transition_risk"] = max(0.0, (1.0 - risk) * 100.0)

        # ── 8. AI model health (optional) ─────────────────────────────────────
        _ah = (advisory.get("ai_health") or {})
        has_ai_health = bool(_ah)
        if has_ai_health:
            _ai_raw = _ah.get("ai_health_score")
            ai_score = float(_ai_raw) if _ai_raw is not None else 100.0
            components["ai_model_health"] = max(0.0, min(100.0, ai_score))

        # ── 9. Risk intelligence (optional) ───────────────────────────────────
        _ri = (advisory.get("risk_intelligence") or {})
        has_risk_intel = bool(_ri)
        if has_risk_intel:
            _ri_raw = _ri.get("risk_score_composite")
            ri_score = float(_ri_raw) if _ri_raw is not None else 75.0
            components["risk_intelligence"] = max(0.0, min(100.0, ri_score))

        # ── 10. Signal intelligence (optional) ────────────────────────────────
        _si = (advisory.get("signal_intelligence") or {})
        has_sig_intel = bool(_si)
        if has_sig_intel:
            _si_raw = _si.get("signal_conviction")
            si_score = float(_si_raw) if _si_raw is not None else 0.5
            components["signal_intelligence"] = max(0.0, min(100.0, si_score * 100.0))

        # ── Build weight map (dynamic normalisation) ──────────────────────────
        # Each optional component has a fixed weight.
        # Base weights are scaled by (1.0 - sum_of_optional_weights) so the
        # grand total is always exactly 1.0.
        optional_weights: dict = {}
        if has_transition:
            optional_weights["transition_risk"]   = _TRANSITION_WEIGHT
        if has_ai_health:
            optional_weights["ai_model_health"]   = _AI_HEALTH_WEIGHT
        if has_risk_intel:
            optional_weights["risk_intelligence"] = _RISK_INTEL_WEIGHT
        if has_sig_intel:
            optional_weights["signal_intelligence"] = _SIG_INTEL_WEIGHT

        base_scale = 1.0 - sum(optional_weights.values())
        weights = {k: v * base_scale for k, v in _BASE_WEIGHTS.items()}
        weights.update(optional_weights)

        # ── Weighted score ────────────────────────────────────────────────────
        raw_score = sum(
            components.get(k, 100.0) * w
            for k, w in weights.items()
        )
        score = max(0, min(100, int(round(raw_score))))

        # ── Label ─────────────────────────────────────────────────────────────
        label = "CRITICAL"
        for threshold, lbl in _LABELS:
            if score >= threshold:
                label = lbl
                break

        # ── Breakdown: per-component weighted contribution ────────────────────
        breakdown = {
            k: round(components.get(k, 100.0) * weights.get(k, 0.0), 2)
            for k in weights
        }

        hs = HealthScore(
            score=score,
            label=label,
            breakdown=breakdown,
            weights_used=dict(weights),
        )
        self._last = hs
        return hs

    def snapshot(self) -> Dict[str, Any]:
        h = self._last
        if h is None:
            return {"score": 100, "label": "PEAK", "breakdown": {}}
        return {
            "score":        h.score,
            "label":        h.label,
            "breakdown":    h.breakdown,
            "weights_used": h.weights_used,
            "ts":           h.ts,
        }
