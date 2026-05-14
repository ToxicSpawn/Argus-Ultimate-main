"""
RiskIntelligence — Aggregates all risk-domain advisory keys into one composite risk picture.

Consumes 12 previously-untapped advisory keys:
  tail_hedge, stress_test, risk_score, toxicity, fragility_score, antifragile_multiplier,
  causal_graph, avg_pairwise_correlation, correlation_penalty, regime_kelly,
  vol_target_scale, intraday_var

Output: advisory["risk_intelligence"]

Composite score formula (0–100, higher = healthier):
  no_tail_hedge       weight 0.20   1 - tail_hedge_urgency
  stress_survival     weight 0.20   survival rate from stress test
  no_toxicity         weight 0.15   1 - toxicity_severity
  no_cascade          weight 0.15   1 - cascade_risk
  no_fragility        weight 0.10   1 - fragility_level
  kelly_fraction      weight 0.10   higher kelly = healthier
  no_var_breach       weight 0.10   0.0 if var_breach else 1.0
Bonus:  antifragile_mult > 1.0 → +3 points
Penalty: correlation EXTREME → −5 points
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Correlation thresholds
_CORR_LOW      = 0.30
_CORR_MODERATE = 0.55
_CORR_HIGH     = 0.75

# Composite weights (must sum to 1.0)
_WEIGHTS = {
    "no_tail_hedge":    0.20,
    "stress_survival":  0.20,
    "no_toxicity":      0.15,
    "no_cascade":       0.15,
    "no_fragility":     0.10,
    "kelly_fraction":   0.10,
    "no_var_breach":    0.10,
}

_ANTIFRAGILE_BONUS         = 3.0
_EXTREME_CORR_PENALTY      = 5.0
_NEUTRAL_RISK_COMPOSITE    = 75.0


class RiskIntelligence:
    """
    Aggregates all risk-domain advisory signals into one risk-intelligence snapshot.

    Parameters
    ----------
    stress_tester : optional reference to stress tester (for summary())
    config        : optional config object
    """

    def __init__(
        self,
        stress_tester: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.stress_tester = stress_tester
        self.config = config
        self._last: Optional[Dict[str, Any]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, advisory: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk-intelligence snapshot from the current advisory dict."""

        # ── 1. Tail hedge urgency ─────────────────────────────────────────────
        _th = advisory.get("tail_hedge") or {}
        tail_hedge_urgency = 1.0 if bool(_th.get("should_hedge", False)) else 0.0

        # ── 2. Stress survival rate ───────────────────────────────────────────
        # advisory["stress_test"] is a string "ok" / summary string.
        # If stress_tester available, pull real pass_rate from summary().
        stress_survival = self._stress_survival_rate(advisory)

        # ── 3. Toxicity severity ──────────────────────────────────────────────
        toxicity_severity = self._toxicity_severity(advisory)

        # ── 4. Cascade risk ───────────────────────────────────────────────────
        cascade_risk = self._cascade_risk(advisory)

        # ── 5. Fragility level ────────────────────────────────────────────────
        fragility_level = self._fragility_composite(advisory)

        # ── 6. Kelly fraction ─────────────────────────────────────────────────
        kelly_fraction = float(advisory.get("regime_kelly") or 0.5)
        kelly_fraction = max(0.0, min(1.0, kelly_fraction))

        # ── 7. VaR ────────────────────────────────────────────────────────────
        _iv = advisory.get("intraday_var") or {}
        util_raw   = float(_iv.get("utilisation_pct", 0.0) or 0.0)
        var_breach = bool(_iv.get("breach", False))
        var_utilisation = max(0.0, min(1.0, util_raw / 100.0))

        # ── 8. Correlation regime ─────────────────────────────────────────────
        correlation_regime = self._correlation_regime(advisory)

        # ── 9. Antifragile multiplier ─────────────────────────────────────────
        antifragile_mult = float(advisory.get("antifragile_multiplier") or 1.0)

        # ── 10. Vol target scale ──────────────────────────────────────────────
        vol_scale = float(advisory.get("vol_target_scale") or 1.0)

        # ── Composite score ───────────────────────────────────────────────────
        components = {
            "no_tail_hedge":   1.0 - tail_hedge_urgency,
            "stress_survival": stress_survival,
            "no_toxicity":     1.0 - toxicity_severity,
            "no_cascade":      1.0 - cascade_risk,
            "no_fragility":    1.0 - fragility_level,
            "kelly_fraction":  kelly_fraction,
            "no_var_breach":   0.0 if var_breach else 1.0,
        }
        raw = sum(components[k] * _WEIGHTS[k] for k in _WEIGHTS)
        composite = raw * 100.0

        # Bonuses / penalties
        if antifragile_mult > 1.0:
            composite += _ANTIFRAGILE_BONUS
        if correlation_regime == "EXTREME":
            composite -= _EXTREME_CORR_PENALTY

        composite = max(0.0, min(100.0, composite))

        result: Dict[str, Any] = {
            "risk_score_composite": round(composite, 2),
            "tail_hedge_urgency":   round(tail_hedge_urgency, 4),
            "stress_survival_rate": round(stress_survival, 4),
            "toxicity_severity":    round(toxicity_severity, 4),
            "correlation_regime":   correlation_regime,
            "kelly_fraction":       round(kelly_fraction, 4),
            "fragility_level":      round(fragility_level, 4),
            "antifragile_mult":     round(antifragile_mult, 4),
            "cascade_risk":         round(cascade_risk, 4),
            "var_utilisation":      round(var_utilisation, 4),
            "var_breach":           var_breach,
            "vol_scale":            round(vol_scale, 4),
            "ts":                   time.time(),
        }
        self._last = result
        return result

    def snapshot(self) -> Dict[str, Any]:
        if self._last is None:
            return {
                "risk_score_composite": _NEUTRAL_RISK_COMPOSITE,
                "tail_hedge_urgency":   0.0,
                "stress_survival_rate": 1.0,
                "toxicity_severity":    0.0,
                "correlation_regime":   "LOW",
                "kelly_fraction":       0.5,
                "fragility_level":      0.0,
                "antifragile_mult":     1.0,
                "cascade_risk":         0.0,
                "var_utilisation":      0.0,
                "var_breach":           False,
                "vol_scale":            1.0,
            }
        return dict(self._last)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _stress_survival_rate(self, advisory: Dict[str, Any]) -> float:
        """Return 0–1 survival rate from stress tester."""
        # Try stress_tester.summary() first
        if self.stress_tester is not None:
            try:
                summary = self.stress_tester.summary()
                if isinstance(summary, dict):
                    passed = int(summary.get("passed", 0) or 0)
                    total  = int(summary.get("total",  1) or 1)
                    return max(0.0, min(1.0, passed / max(total, 1)))
            except Exception:  # noqa: BLE001
                pass
        # advisory["stress_test"] = "ok" means all passed
        _st = advisory.get("stress_test")
        if isinstance(_st, str) and _st.strip().lower() == "ok":
            return 1.0
        if isinstance(_st, dict):
            passed = int(_st.get("passed", 0) or 0)
            total  = int(_st.get("total",  1) or 1)
            return max(0.0, min(1.0, passed / max(total, 1)))
        # No data → neutral
        return 1.0

    def _toxicity_severity(self, advisory: Dict[str, Any]) -> float:
        """Return 0–1 toxicity level (0=clean, 1=maximally toxic)."""
        _tx = advisory.get("toxicity") or {}
        raw_score = float(_tx.get("score", 0.0) or 0.0)
        # score is flow_toxicity index (0+), normalise against ceiling of 2.0
        return max(0.0, min(1.0, raw_score / 2.0))

    def _cascade_risk(self, advisory: Dict[str, Any]) -> float:
        """Return cascade_probability from causal_graph (0–1)."""
        _cg = advisory.get("causal_graph") or {}
        return max(0.0, min(1.0, float(_cg.get("cascade_probability", 0.0) or 0.0)))

    def _fragility_composite(self, advisory: Dict[str, Any]) -> float:
        """Combine fragility_score + correlation_penalty into 0–1 fragility level."""
        raw_frag  = float(advisory.get("fragility_score") or 0.0)
        corr_pen  = float(advisory.get("correlation_penalty") or 0.0)
        # fragility_score is 0+ (no fixed ceiling); normalise against 5.0
        frag_norm = max(0.0, min(1.0, raw_frag / 5.0))
        # blend 70% fragility, 30% correlation penalty
        return max(0.0, min(1.0, frag_norm * 0.70 + corr_pen * 0.30))

    def _correlation_regime(self, advisory: Dict[str, Any]) -> str:
        """Classify correlation as LOW/MODERATE/HIGH/EXTREME."""
        corr = abs(float(advisory.get("avg_pairwise_correlation") or 0.0))
        if corr >= _CORR_HIGH:
            return "EXTREME"
        if corr >= _CORR_MODERATE:
            return "HIGH"
        if corr >= _CORR_LOW:
            return "MODERATE"
        return "LOW"
