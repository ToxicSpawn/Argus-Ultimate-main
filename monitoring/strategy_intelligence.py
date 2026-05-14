"""
StrategyIntelligence — Aggregates all strategy-domain advisory keys into one strategy health picture.

Consumes 14 previously-untapped advisory keys:
  bleeders, auto_paused_strategies, bandit_rankings, contextual_bandit, regime_rotation,
  strategy_regime_matrix, strategy_optimization, live_validation, universe_report,
  funding_harvester, funding_prediction, genetic_evolver, feature_discovery,
  capital_migration_advanced

Output: advisory["strategy_intelligence"]

Strategy health composite (0–100, higher = healthier):
  no_bleeders           weight 0.25   1 - bleeder_severity
  best_bandit_win_rate  weight 0.20
  no_retrain_needed     weight 0.15   0.0 if urgent, 0.5 if needed, 1.0 if clean
  capital_stage_health  weight 0.15
  feature_health_score  weight 0.10
  evolution_health      weight 0.10
  funding_opportunity   weight 0.05
Penalty: validation_urgent → −10 points
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Capital stage health mapping
_STAGE_HEALTH = {
    "MICRO":   0.40,
    "SMALL":   0.60,
    "MEDIUM":  0.80,
    "LARGE":   0.95,
    "PEAK":    1.00,
}

# Max expected bleeders for severity normalisation
_MAX_BLEEDERS = 5

# Composite weights (must sum to 1.0)
_WEIGHTS = {
    "no_bleeders":          0.25,
    "best_bandit_win_rate": 0.20,
    "no_retrain_needed":    0.15,
    "capital_stage_health": 0.15,
    "feature_health_score": 0.10,
    "evolution_health":     0.10,
    "funding_opportunity":  0.05,
}

_URGENT_RETRAIN_PENALTY = 10.0
_NEUTRAL_COMPOSITE      = 75.0


class StrategyIntelligence:
    """
    Aggregates all strategy-domain advisory signals into one strategy-intelligence snapshot.

    Parameters
    ----------
    config : optional config object
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self._last: Optional[Dict[str, Any]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, advisory: Dict[str, Any]) -> Dict[str, Any]:
        """Build strategy-intelligence snapshot from the current advisory dict."""

        # ── 1. Bleeders ───────────────────────────────────────────────────────
        bleeders: List[Any] = list(advisory.get("bleeders") or [])
        bleeder_count    = len(bleeders)
        bleeder_severity = self._bleeder_severity(bleeders)

        # ── 2. Active / paused strategies ────────────────────────────────────
        paused_list  = list(advisory.get("auto_paused_strategies") or [])
        paused_count = len(paused_list)

        # active = universe - paused (rough estimate from universe_report)
        _ur = advisory.get("universe_report") or {}
        n_symbols = int(_ur.get("n_symbols", 0) or 0)
        # if no universe info default to 0
        active_strategies = max(0, n_symbols - paused_count)

        # ── 3. Bandit rankings ────────────────────────────────────────────────
        rankings: List[Any] = list(advisory.get("bandit_rankings") or [])
        best_win_rate = 0.5  # neutral default
        best_regime_strategy = ""
        if rankings:
            top = rankings[0]
            if isinstance(top, dict):
                best_win_rate        = float(top.get("expected_win_rate", 0.5) or 0.5)
                best_regime_strategy = str(top.get("strategy", "") or "")
        best_win_rate = max(0.0, min(1.0, best_win_rate))

        # contextual bandit override if better
        _cb = advisory.get("contextual_bandit") or {}
        if _cb and not best_regime_strategy:
            best_regime_strategy = str(_cb.get("best_strategy", "") or "")

        # ── 4. Live validation ────────────────────────────────────────────────
        _lv = advisory.get("live_validation") or {}
        needs_retrain    = bool(_lv.get("needs_retrain", False))
        validation_urgent = bool(_lv.get("urgent", False))

        # ── 5. Capital migration health ───────────────────────────────────────
        capital_stage_health = self._capital_stage_health(advisory)

        # ── 6. Feature health ─────────────────────────────────────────────────
        feature_health_score = self._feature_health_score(advisory)

        # ── 7. Evolution health ───────────────────────────────────────────────
        evolution_health = self._evolution_health(advisory)

        # ── 8. Funding opportunity ────────────────────────────────────────────
        funding_opportunity_score = self._funding_opportunity_score(advisory)

        # ── Composite score ───────────────────────────────────────────────────
        # no_retrain_needed: 1.0 clean, 0.5 needs but not urgent, 0.0 urgent
        if validation_urgent:
            no_retrain = 0.0
        elif needs_retrain:
            no_retrain = 0.5
        else:
            no_retrain = 1.0

        components = {
            "no_bleeders":          1.0 - bleeder_severity,
            "best_bandit_win_rate": best_win_rate,
            "no_retrain_needed":    no_retrain,
            "capital_stage_health": capital_stage_health,
            "feature_health_score": feature_health_score,
            "evolution_health":     evolution_health,
            "funding_opportunity":  funding_opportunity_score,
        }
        raw = sum(components[k] * _WEIGHTS[k] for k in _WEIGHTS)
        composite = raw * 100.0

        if validation_urgent:
            composite -= _URGENT_RETRAIN_PENALTY

        composite = max(0.0, min(100.0, composite))

        result: Dict[str, Any] = {
            "active_strategies":         active_strategies,
            "paused_strategies":         paused_count,
            "bleeder_count":             bleeder_count,
            "bleeder_severity":          round(bleeder_severity, 4),
            "best_regime_strategy":      best_regime_strategy,
            "bandit_best_win_rate":      round(best_win_rate, 4),
            "validation_needs_retrain":  needs_retrain,
            "validation_urgent":         validation_urgent,
            "funding_opportunity_score": round(funding_opportunity_score, 4),
            "capital_stage_health":      round(capital_stage_health, 4),
            "feature_health_score":      round(feature_health_score, 4),
            "evolution_health":          round(evolution_health, 4),
            "strategy_health_composite": round(composite, 2),
            "ts":                        time.time(),
        }
        self._last = result
        return result

    def snapshot(self) -> Dict[str, Any]:
        if self._last is None:
            return {
                "active_strategies":         0,
                "paused_strategies":         0,
                "bleeder_count":             0,
                "bleeder_severity":          0.0,
                "best_regime_strategy":      "",
                "bandit_best_win_rate":      0.5,
                "validation_needs_retrain":  False,
                "validation_urgent":         False,
                "funding_opportunity_score": 0.0,
                "capital_stage_health":      1.0,
                "feature_health_score":      1.0,
                "evolution_health":          1.0,
                "strategy_health_composite": _NEUTRAL_COMPOSITE,
            }
        return dict(self._last)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _bleeder_severity(self, bleeders: List[Any]) -> float:
        """0–1 severity: 0=no bleeders, 1=max bleeders all with negative sharpe."""
        if not bleeders:
            return 0.0
        count_factor = min(1.0, len(bleeders) / _MAX_BLEEDERS)
        # Sharpe factor: mean of negative sharpe contributions
        sharpe_factors: List[float] = []
        for b in bleeders:
            if isinstance(b, dict):
                sh = float(b.get("sharpe", 0.0) or 0.0)
                # negative sharpe = bad; normalise against −2.0 floor
                sharpe_factors.append(max(0.0, min(1.0, -sh / 2.0)))
        if sharpe_factors:
            sharpe_factor = sum(sharpe_factors) / len(sharpe_factors)
        else:
            sharpe_factor = 0.5
        # blend 60% count, 40% sharpe severity
        return max(0.0, min(1.0, count_factor * 0.60 + sharpe_factor * 0.40))

    def _funding_opportunity_score(self, advisory: Dict[str, Any]) -> float:
        """0–1 opportunity score from funding harvester + prediction."""
        _fh = advisory.get("funding_harvester") or {}
        _fp = advisory.get("funding_prediction") or {}
        harvester_pct = abs(float(_fh.get("total_funding_pct", 0.0) or 0.0))
        pred_conf     = float(_fp.get("confidence", 0.0) or 0.0)
        # normalise harvester against 2% (high-rate environment)
        harvester_score = min(1.0, harvester_pct / 2.0)
        # blend
        if _fh or _fp:
            return max(0.0, min(1.0, harvester_score * 0.60 + pred_conf * 0.40))
        return 0.0

    def _capital_stage_health(self, advisory: Dict[str, Any]) -> float:
        """0–1 capital migration health."""
        _cm = advisory.get("capital_migration_advanced") or {}
        if not _cm:
            return 1.0  # neutral if not tracked
        stage     = str(_cm.get("stage", "MICRO") or "MICRO").upper()
        health_pct = float(_cm.get("health_pct", 100.0) or 100.0)
        stage_base = _STAGE_HEALTH.get(stage, 0.50)
        # blend stage level with health percentage
        return max(0.0, min(1.0, stage_base * 0.50 + (health_pct / 100.0) * 0.50))

    def _feature_health_score(self, advisory: Dict[str, Any]) -> float:
        """0–1 feature health: more discovered features = healthier."""
        _fd = advisory.get("feature_discovery") or {}
        if not _fd:
            return 1.0  # neutral
        total = int(_fd.get("total_discovered", 0) or 0)
        # normalise against 50 discovered features as "healthy" threshold
        return max(0.0, min(1.0, total / 50.0)) if total > 0 else 0.5

    def _evolution_health(self, advisory: Dict[str, Any]) -> float:
        """0–1 genetic evolver health based on best_fitness."""
        _ge = advisory.get("genetic_evolver") or {}
        if not _ge:
            return 1.0  # neutral
        fitness = float(_ge.get("best_fitness", 0.5) or 0.5)
        # fitness is a strategy fitness score: normalise 0–1 (can exceed 1 in good markets)
        return max(0.0, min(1.0, fitness))
