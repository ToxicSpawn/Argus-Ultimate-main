"""
AIHealthMonitor — Central AI/ML health aggregator.

Reads ALL ML advisory keys each cycle and produces a single
advisory["ai_health"] payload summarising:
  - Per-model health score (0–100)
  - Aggregate ai_health_score + label (HEALTHY/DEGRADED/CRITICAL)
  - Drifted model list + multi_model_drift flag
  - Regime agreement across hmm/autoencoder/regime_ensemble
  - Retraining queue (models with persistent low confidence)

Per-model scoring:
  base = confidence × 100
  × 0.70 if drifting
  × 0.80 if stale (from advisory["model_performance"])

Aggregate weights:
  ensemble=0.20, hmm=0.15, online=0.15, vol=0.10, alpha=0.10,
  llm=0.10, autoencoder=0.08, gnn=0.05, attention=0.05, rl=0.02

Output: advisory["ai_health"]
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_WEIGHTS: Dict[str, float] = {
    "ensemble":           0.20,
    "hmm_regime":         0.15,
    "online_learner":     0.15,
    "vol_forecaster":     0.10,
    "alpha_model":        0.10,
    "llm_signal":         0.10,
    "autoencoder_regime": 0.08,
    "gnn_asset_flow":     0.05,
    "attention_orderflow":0.05,
    "rl_portfolio":       0.02,
}

_HEALTH_LABEL_THRESHOLDS = [
    (70.0, "HEALTHY"),
    (45.0, "DEGRADED"),
    (0.0,  "CRITICAL"),
]


def _health_label(score: float) -> str:
    for threshold, label in _HEALTH_LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "CRITICAL"


# ---------------------------------------------------------------------------
# AIHealthMonitor
# ---------------------------------------------------------------------------

class AIHealthMonitor:
    """
    Aggregates AI/ML health from all advisory ML keys each cycle.

    Parameters
    ----------
    low_conf_threshold     : confidence below this → consecutive-low counter increments
    low_conf_cycles        : cycles of consecutive low confidence → add to retraining queue
    multi_drift_alert_count: number of drifted models to trigger multi_model_drift flag
    config                 : optional config object
    """

    def __init__(
        self,
        low_conf_threshold: float = 0.40,
        low_conf_cycles: int = 10,
        multi_drift_alert_count: int = 2,
        config: Optional[Any] = None,
    ) -> None:
        self.low_conf_threshold      = float(low_conf_threshold)
        self.low_conf_cycles         = max(1, int(low_conf_cycles))
        self.multi_drift_alert_count = max(1, int(multi_drift_alert_count))
        self.config                  = config

        # Consecutive low-confidence counter per model
        self._low_conf_streak: Dict[str, int] = {}
        # Current retraining queue
        self._retraining_queue: Set[str] = set()
        # Last advisory payload cached for snapshot()
        self._last_payload: Optional[Dict[str, Any]] = None
        # Cached LLM/attention confidence (these are slow to update)
        self._cached_conf: Dict[str, float] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def compute(self, advisory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute AI health from advisory dict.
        Returns advisory["ai_health"] payload.
        """
        readings = self._extract_readings(advisory)

        # Stale flags from model_performance
        stale_set: Set[str] = set()
        mp = advisory.get("model_performance") or {}
        per_model_perf = mp.get("per_model") or {}
        for mname, mdata in per_model_perf.items():
            if isinstance(mdata, dict) and mdata.get("is_stale"):
                stale_set.add(mname)

        # Score each model
        per_model_scores: Dict[str, float] = {}
        drifted: List[str] = []
        for name, reading in readings.items():
            conf        = reading.get("confidence", 0.0)
            is_drifting = bool(reading.get("is_drifting", False))
            is_stale    = name in stale_set

            score = self._score_model(conf, is_drifting, is_stale)
            per_model_scores[name] = round(score, 2)

            if is_drifting:
                drifted.append(name)

            # Update retraining queue
            self._update_retraining_queue(name, conf)

        # Aggregate
        agg_score = self._aggregate_score(per_model_scores)
        label     = _health_label(agg_score)

        # Regime agreement
        regime_agreement = self._compute_regime_agreement(advisory)

        multi_drift = len(drifted) >= self.multi_drift_alert_count

        payload: Dict[str, Any] = {
            "ai_health_score":   round(agg_score, 2),
            "ai_health_label":   label,
            "model_health":      per_model_scores,
            "drifted_models":    sorted(drifted),
            "multi_model_drift": multi_drift,
            "regime_agreement":  round(regime_agreement, 4),
            "retraining_queue":  sorted(self._retraining_queue),
            "models_evaluated":  len(readings),
            "ts":                time.time(),
        }
        self._last_payload = payload
        return payload

    def snapshot(self) -> Dict[str, Any]:
        """Return last computed payload."""
        if self._last_payload is None:
            return {
                "ai_health_score":   100.0,
                "ai_health_label":   "HEALTHY",
                "model_health":      {},
                "drifted_models":    [],
                "multi_model_drift": False,
                "regime_agreement":  1.0,
                "retraining_queue":  [],
                "models_evaluated":  0,
            }
        return dict(self._last_payload)

    # ── Internal ────────────────────────────────────────────────────────────

    def _extract_readings(
        self, advisory: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract (confidence, is_drifting) from each ML advisory key."""
        readings: Dict[str, Dict[str, Any]] = {}

        # ensemble
        _ens = advisory.get("ensemble") or {}
        if _ens:
            readings["ensemble"] = {
                "confidence":  float(_ens.get("confidence", 0.0) or 0.0),
                "is_drifting": False,
            }

        # hmm_regime
        _hmm = advisory.get("hmm_regime") or {}
        if _hmm:
            readings["hmm_regime"] = {
                "confidence":  float(_hmm.get("confidence", 0.0) or 0.0),
                "is_drifting": False,
            }

        # online_learner
        _ol = advisory.get("online_learner") or {}
        if _ol:
            conf_ol      = float(_ol.get("confidence", 0.0) or 0.0)
            drift_ol     = bool(_ol.get("drift_detected", False))
            readings["online_learner"] = {
                "confidence":  conf_ol,
                "is_drifting": drift_ol,
            }

        # vol_forecaster
        _vf = advisory.get("vol_forecasts") or {}
        if _vf:
            readings["vol_forecaster"] = {
                "confidence":  0.70,
                "is_drifting": False,
            }

        # alpha_model
        _alpha = advisory.get("alpha_scores") or {}
        if _alpha:
            # _alpha may be {symbol: {confidence: x}} or {symbol: confidence_float}
            confs: List[float] = []
            for sym_val in _alpha.values():
                if isinstance(sym_val, dict):
                    c = sym_val.get("confidence", None)
                    if c is not None:
                        try:
                            confs.append(float(c))
                        except (TypeError, ValueError):
                            pass
                elif isinstance(sym_val, (int, float)):
                    confs.append(float(sym_val))
            alpha_conf = (sum(confs) / len(confs)) if confs else 0.65
            readings["alpha_model"] = {
                "confidence":  alpha_conf,
                "is_drifting": False,
            }

        # llm_signal (use cached conf between updates)
        _llm = advisory.get("llm_analysis") or {}
        if _llm:
            conf_llm = float(_llm.get("confidence", 0.0) or 0.0)
            if conf_llm > 0:
                self._cached_conf["llm_signal"] = conf_llm
        cached_llm = self._cached_conf.get("llm_signal")
        if cached_llm is not None:
            readings["llm_signal"] = {
                "confidence":  cached_llm,
                "is_drifting": False,
            }
        elif _llm:
            readings["llm_signal"] = {
                "confidence":  0.0,
                "is_drifting": False,
            }

        # autoencoder_regime
        _ae = advisory.get("autoencoder_regime") or {}
        if _ae:
            recon_err = float(_ae.get("reconstruction_error", 0.0) or 0.0)
            ae_conf   = max(0.0, min(1.0, 1.0 - recon_err))
            readings["autoencoder_regime"] = {
                "confidence":  ae_conf,
                "is_drifting": False,
            }

        # gnn_asset_flow
        _gnn = advisory.get("gnn_asset_flow") or {}
        if _gnn:
            readings["gnn_asset_flow"] = {
                "confidence":  0.60,
                "is_drifting": False,
            }

        # attention_orderflow (use cached between updates)
        _att = advisory.get("attention_orderflow") or {}
        if _att:
            conf_att = float(_att.get("confidence", 0.0) or 0.0)
            if conf_att > 0:
                self._cached_conf["attention_orderflow"] = conf_att
        cached_att = self._cached_conf.get("attention_orderflow")
        if cached_att is not None:
            readings["attention_orderflow"] = {
                "confidence":  cached_att,
                "is_drifting": False,
            }
        elif _att:
            readings["attention_orderflow"] = {
                "confidence":  0.0,
                "is_drifting": False,
            }

        # rl_portfolio
        _rl = advisory.get("rl_portfolio_allocation") or {}
        if _rl:
            readings["rl_portfolio"] = {
                "confidence":  0.65,
                "is_drifting": False,
            }

        return readings

    def _score_model(
        self,
        confidence: float,
        is_drifting: bool,
        is_stale: bool,
    ) -> float:
        """Score a single model 0–100."""
        base = float(confidence) * 100.0
        if is_drifting:
            base *= 0.70
        if is_stale:
            base *= 0.80
        return max(0.0, min(100.0, base))

    def _compute_regime_agreement(self, advisory: Dict[str, Any]) -> float:
        """
        Compute fraction of regime labels that agree with the plurality.
        Uses hmm_regime, autoencoder_regime, regime_ensemble.
        Returns 1.0 when fewer than 2 sources available.
        """
        labels: List[str] = []

        _hmm = advisory.get("hmm_regime") or {}
        if _hmm:
            reg = str(_hmm.get("regime", "") or "")
            if reg:
                labels.append(reg)

        _ae = advisory.get("autoencoder_regime") or {}
        if _ae:
            reg = str(_ae.get("regime", "") or "")
            if reg:
                labels.append(reg)

        _re = advisory.get("regime_ensemble") or {}
        if _re:
            reg = str(_re.get("regime", "") or "")
            if reg:
                labels.append(reg)

        if len(labels) < 2:
            return 1.0

        # Count occurrences
        counts: Dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        plurality_count = max(counts.values())
        return round(plurality_count / len(labels), 4)

    def _update_retraining_queue(self, name: str, confidence: float) -> None:
        """
        Track consecutive low-confidence cycles per model.
        Add to retraining_queue after low_conf_cycles consecutive misses.
        Remove when confidence recovers above threshold.
        """
        if confidence < self.low_conf_threshold:
            self._low_conf_streak[name] = self._low_conf_streak.get(name, 0) + 1
            if self._low_conf_streak[name] >= self.low_conf_cycles:
                self._retraining_queue.add(name)
        else:
            # Recovery
            self._low_conf_streak[name] = 0
            self._retraining_queue.discard(name)

    def _aggregate_score(self, per_model_scores: Dict[str, float]) -> float:
        """Weighted average of per-model scores."""
        if not per_model_scores:
            return 100.0

        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in per_model_scores.items():
            w = _MODEL_WEIGHTS.get(name, 0.02)   # unknown models get small weight
            weighted_sum  += w * score
            total_weight  += w

        if total_weight <= 0:
            return sum(per_model_scores.values()) / len(per_model_scores)

        return max(0.0, min(100.0, weighted_sum / total_weight))
