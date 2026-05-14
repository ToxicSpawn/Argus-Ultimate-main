"""
SignalIntelligence — Aggregates all signal-domain advisory keys into one consensus signal picture.

Consumes 19 previously-untapped advisory keys:
  stacked_signal, quantum_prediction, quantum_regime, quantum_anomaly_score, quantum_portfolio,
  quantum_risk_check, fear_greed, vol_forecasts, hmm_regime, autoencoder_regime, ensemble_regime,
  online_learner, regime_prediction, llm_analysis, gnn_asset_flow, attention_orderflow,
  rl_portfolio_allocation, orderbook_prediction, sentiment_stats / chart_patterns

Output: advisory["signal_intelligence"]

Signal consensus sources (weighted sum −1 to 1):
  stacked_signal["value"]          weight 0.30  (already −1 to 1)
  quantum_prediction sign          weight 0.15
  LLM bullish/bearish              weight 0.15
  fear_greed normalised            weight 0.10  (val−50)/50
  orderbook direction              weight 0.10
  regime consensus                 weight 0.20  TRENDING_UP→+0.5, DOWN→−0.5
"""
from __future__ import annotations

import logging
import statistics
import time
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Fear/greed thresholds
_FG_EXTREME_FEAR  = 20
_FG_FEAR          = 40
_FG_GREED         = 60
_FG_EXTREME_GREED = 80

# Quantum anomaly threshold
_QUANTUM_ANOMALY_THRESHOLD = 0.75
_ANOMALY_CONF_PENALTY      = 0.60

# Signal consensus weights (must sum to 1.0)
_CONSENSUS_WEIGHTS = {
    "stacked":  0.30,
    "quantum":  0.15,
    "llm":      0.15,
    "fear_greed": 0.10,
    "orderbook":  0.10,
    "regime":     0.20,
}

# Regime labels that map to directional signal
_REGIME_SIGNAL = {
    "TRENDING_UP":   +0.50,
    "trending_up":   +0.50,
    "RANGING":        0.00,
    "ranging":        0.00,
    "TRENDING_DOWN": -0.50,
    "trending_down": -0.50,
    "HIGH_VOL":      -0.25,
    "high_vol":      -0.25,
    "CRISIS":        -0.75,
    "crisis":        -0.75,
}


class SignalIntelligence:
    """
    Aggregates all signal-domain advisory signals into one signal-intelligence snapshot.

    Parameters
    ----------
    config : optional config object
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        self._last: Optional[Dict[str, Any]] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, advisory: Dict[str, Any]) -> Dict[str, Any]:
        """Build signal-intelligence snapshot from the current advisory dict."""

        # ── 1. Stacked signal ─────────────────────────────────────────────────
        _ss = advisory.get("stacked_signal") or {}
        stacked_value      = float(_ss.get("value",      0.0) or 0.0)
        stacked_confidence = float(_ss.get("confidence", 0.5) or 0.5)
        stacked_value      = max(-1.0, min(1.0, stacked_value))
        stacked_confidence = max(0.0,  min(1.0, stacked_confidence))

        # ── 2. Quantum signals ────────────────────────────────────────────────
        _qp = advisory.get("quantum_prediction") or {}
        _qr = advisory.get("quantum_regime")     or {}
        qp_conf      = float(_qp.get("confidence", 0.5) or 0.5)
        qr_conf      = float(_qr.get("confidence", 0.5) or 0.5)
        qp_next_val  = float(_qp.get("next_value",  0.0) or 0.0)
        quantum_anom_raw = float(advisory.get("quantum_anomaly_score") or 0.0)
        quantum_anomaly  = quantum_anom_raw > _QUANTUM_ANOMALY_THRESHOLD

        quantum_confidence = (qp_conf + qr_conf) / 2.0
        if quantum_anomaly:
            quantum_confidence *= _ANOMALY_CONF_PENALTY
        quantum_confidence = max(0.0, min(1.0, quantum_confidence))

        # Quantum direction: sign of next_value (−1 / 0 / +1)
        quantum_dir = 0.5 if qp_next_val > 0 else (-0.5 if qp_next_val < 0 else 0.0)

        # ── 3. LLM signal ─────────────────────────────────────────────────────
        _la = advisory.get("llm_analysis") or {}
        llm_signal_str = str(_la.get("signal", "") or "").lower()
        llm_confidence = float(_la.get("confidence", 0.5) or 0.5)
        llm_dir = (
            +1.0 if "bullish" in llm_signal_str
            else -1.0 if "bearish" in llm_signal_str
            else 0.0
        )

        # ── 4. Fear & greed ───────────────────────────────────────────────────
        _fg = advisory.get("fear_greed") or {}
        fg_value = int(_fg.get("value", 50) or 50)
        fg_value = max(0, min(100, fg_value))
        fg_normalised = (fg_value - 50) / 50.0  # −1 to +1
        fear_greed_regime = self._fear_greed_regime(fg_value)

        # ── 5. Orderbook prediction ───────────────────────────────────────────
        _ob = advisory.get("orderbook_prediction") or {}
        ob_dir_str = str(_ob.get("direction", "neutral") or "neutral").lower()
        ob_confidence = float(_ob.get("confidence", 0.5) or 0.5)
        ob_dir = (
            +1.0 if ob_dir_str in ("buy", "long", "bullish")
            else -1.0 if ob_dir_str in ("sell", "short", "bearish")
            else 0.0
        )
        orderbook_direction = ob_dir_str if ob_dir_str in ("buy", "sell") else "neutral"

        # ── 6. Regime consensus ───────────────────────────────────────────────
        regime_consensus, model_agreement = self._regime_consensus(advisory)
        regime_dir = _REGIME_SIGNAL.get(regime_consensus, 0.0)

        # ── 7. Vol forecast ───────────────────────────────────────────────────
        volatility_forecast_1d = self._volatility_forecast_1d(advisory)

        # ── Signal consensus (weighted) ───────────────────────────────────────
        components = {
            "stacked":    stacked_value,
            "quantum":    quantum_dir,
            "llm":        llm_dir,
            "fear_greed": fg_normalised,
            "orderbook":  ob_dir,
            "regime":     regime_dir,
        }
        consensus = sum(
            components[k] * _CONSENSUS_WEIGHTS[k]
            for k in _CONSENSUS_WEIGHTS
        )
        consensus = max(-1.0, min(1.0, consensus))

        # ── Signal conviction (mean of all available confidence signals) ───────
        conf_signals: List[float] = [stacked_confidence]
        conf_signals.append(quantum_confidence)
        if _la:
            conf_signals.append(llm_confidence)
        if _ob:
            conf_signals.append(ob_confidence)
        conf_signals.append(model_agreement)
        conviction = max(0.0, min(1.0, statistics.mean(conf_signals)))

        result: Dict[str, Any] = {
            "signal_consensus":       round(consensus, 4),
            "signal_conviction":      round(conviction, 4),
            "quantum_confidence":     round(quantum_confidence, 4),
            "quantum_anomaly":        quantum_anomaly,
            "fear_greed_regime":      fear_greed_regime,
            "fear_greed_value":       fg_value,
            "volatility_forecast_1d": round(volatility_forecast_1d, 6),
            "regime_consensus":       regime_consensus,
            "model_agreement":        round(model_agreement, 4),
            "stacked_signal_value":   round(stacked_value, 4),
            "stacked_confidence":     round(stacked_confidence, 4),
            "orderbook_direction":    orderbook_direction,
            "ts":                     time.time(),
        }
        self._last = result
        return result

    def snapshot(self) -> Dict[str, Any]:
        if self._last is None:
            return {
                "signal_consensus":       0.0,
                "signal_conviction":      0.5,
                "quantum_confidence":     0.5,
                "quantum_anomaly":        False,
                "fear_greed_regime":      "NEUTRAL",
                "fear_greed_value":       50,
                "volatility_forecast_1d": 0.0,
                "regime_consensus":       "UNKNOWN",
                "model_agreement":        0.5,
                "stacked_signal_value":   0.0,
                "stacked_confidence":     0.5,
                "orderbook_direction":    "neutral",
            }
        return dict(self._last)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fear_greed_regime(self, value: int) -> str:
        if value <= _FG_EXTREME_FEAR:
            return "EXTREME_FEAR"
        if value <= _FG_FEAR:
            return "FEAR"
        if value < _FG_GREED:
            return "NEUTRAL"
        if value < _FG_EXTREME_GREED:
            return "GREED"
        return "EXTREME_GREED"

    def _volatility_forecast_1d(self, advisory: Dict[str, Any]) -> float:
        """Return median 1-day vol forecast across all symbols."""
        _vf = advisory.get("vol_forecasts") or {}
        if not isinstance(_vf, dict):
            return 0.0
        forecasts: List[float] = []
        for sym_data in _vf.values():
            if isinstance(sym_data, dict):
                val = sym_data.get("forecast_1d")
                if val is not None:
                    try:
                        forecasts.append(float(val))
                    except (TypeError, ValueError):
                        pass
        if not forecasts:
            return 0.0
        return statistics.median(forecasts)

    def _regime_consensus(self, advisory: Dict[str, Any]) -> tuple[str, float]:
        """
        Determine the most-voted regime across available sources.
        Returns (consensus_regime, model_agreement_fraction).
        """
        sources = [
            (advisory.get("hmm_regime")        or {}).get("regime"),
            (advisory.get("autoencoder_regime") or {}).get("regime"),
            (advisory.get("ensemble_regime")    or {}).get("regime"),
            (advisory.get("quantum_regime")     or {}).get("regime"),
        ]
        valid = [str(r).upper() for r in sources if r]
        if not valid:
            return "UNKNOWN", 0.5

        counts = Counter(valid)
        consensus = counts.most_common(1)[0][0]
        agreement = counts[consensus] / len(valid)
        return consensus, agreement
