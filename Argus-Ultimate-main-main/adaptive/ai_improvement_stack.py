from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _signal_get(signal: Any, field: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(field, default)
    return getattr(signal, field, default)


def _signal_set(signal: Any, field: str, value: Any) -> None:
    if isinstance(signal, dict):
        signal[field] = value
        return
    setattr(signal, field, value)


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp01((float(value) - lo) / (hi - lo))


class AIImprovementStack:
    """Deterministic advisory AI stack for candidate weighting.

    This module intentionally does not place orders. It only annotates
    candidates with auditable score/weight metadata and bounded confidence deltas.
    """

    MODEL_KEYS: Tuple[str, ...] = (
        "xgboost_edge",
        "lightgbm_edge",
        "catboost_edge",
        "sklearn_calibrated_classifier",
        "river_online_learner",
        "meta_labeling",
        "regime_classifier",
        "hmm_regime",
        "change_point_detection",
        "volatility_forecast",
        "liquidity_state_classifier",
        "adverse_selection_classifier",
        "maker_fill_probability",
        "slippage_quantile",
        "queue_time_to_fill",
        "spread_expansion_predictor",
        "order_book_imbalance_predictor",
        "trade_flow_toxicity",
        "execution_style_selector",
        "twap_slice_optimizer",
        "reconciliation_anomaly",
        "intent_failure_likelihood",
        "risk_breach_warning",
        "loss_streak_hazard",
        "fee_churn_predictor",
        "universe_ranking",
        "strategy_weighting",
        "strategy_disable_recover",
        "champion_challenger_scorer",
        "correlation_cluster_risk",
        "portfolio_target_scaler",
        "forecast_stacking",
        "feature_drift_detection",
        "data_quality_anomaly",
        "offline_llm_reviewer",
        "sentiment_finbert",
        "news_embedding",
        "microstructure_autoencoder_anomaly",
        "bayesian_model_averaging",
        "walk_forward_model_selector",
    )

    def __init__(self, config: Any):
        self.enabled = bool(getattr(config, "ai_improvement_stack_enabled", False))
        self.advisory_only = bool(getattr(config, "ai_improvement_stack_advisory_only", True))
        self.confidence_delta_cap = float(
            getattr(config, "ai_improvement_stack_confidence_delta_cap", 0.10) or 0.10
        )
        self.min_models_required = int(
            getattr(config, "ai_improvement_stack_min_models_required", 3) or 3
        )
        self.score_weights = dict(
            getattr(config, "ai_improvement_stack_score_weights", {"base_confidence": 1.0, "ai_stack": 1.0}) or {}
        )
        self.models_cfg = dict(getattr(config, "ai_improvement_stack_models", {}) or {})
        self.uncertainty_penalty_enabled = bool(
            getattr(config, "ai_improvement_stack_uncertainty_penalty_enabled", True)
        )
        self.max_uncertainty_penalty = float(
            getattr(config, "ai_improvement_stack_max_uncertainty_penalty", 0.35) or 0.35
        )
        self.regime_multipliers = dict(
            getattr(config, "ai_improvement_stack_regime_multipliers", {}) or {}
        )
        self.danger_liquidity_blocks_uplift = bool(
            getattr(config, "ai_improvement_stack_danger_liquidity_blocks_uplift", True)
        )
        self.max_adverse_selection_for_uplift = float(
            getattr(config, "ai_improvement_stack_max_adverse_selection_for_uplift", 0.75) or 0.75
        )
        self.min_liquidity_for_uplift = float(
            getattr(config, "ai_improvement_stack_min_liquidity_for_uplift", 0.20) or 0.20
        )
        self.drawdown_uplift_cutoff_pct = float(
            getattr(config, "ai_improvement_stack_drawdown_uplift_cutoff_pct", 15.0) or 15.0
        )

    def _model_enabled(self, key: str) -> bool:
        cfg = self.models_cfg.get(key, {}) or {}
        if not isinstance(cfg, dict):
            return False
        return bool(cfg.get("enabled", False))

    def _model_weight(self, key: str) -> float:
        cfg = self.models_cfg.get(key, {}) or {}
        if isinstance(cfg, dict):
            try:
                return max(0.0, float(cfg.get("weight", 1.0) or 1.0))
            except Exception:
                return 1.0
        return 1.0

    @staticmethod
    def _weighted_stddev(model_scores: List[Tuple[str, float, float]]) -> float:
        if not model_scores:
            return 0.0
        weights = [max(0.0, float(w)) for _, _, w in model_scores]
        total_w = sum(weights)
        if total_w <= 0.0:
            return 0.0
        mean = sum(float(s) * max(0.0, float(w)) for _, s, w in model_scores) / total_w
        var = sum((float(s) - mean) ** 2 * max(0.0, float(w)) for _, s, w in model_scores) / total_w
        return math.sqrt(max(0.0, var))

    def _regime_multiplier(self, signal: Any) -> float:
        label = str(_signal_get(signal, "regime_label", "") or "")
        if not label:
            return 1.0
        for key, raw in self.regime_multipliers.items():
            try:
                if str(key).strip().lower() in label.lower():
                    return max(0.0, float(raw))
            except Exception:
                continue
        return 1.0

    def _can_uplift(self, signal: Any) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        liq_state = str(_signal_get(signal, "liquidity_state", "") or "").lower()
        liq_score = float(_signal_get(signal, "liquidity_score", 0.0) or 0.0)
        adverse = float(_signal_get(signal, "adverse_selection_risk", 0.0) or 0.0)
        drawdown = float(_signal_get(signal, "strategy_drawdown_pct", 0.0) or 0.0)

        if self.danger_liquidity_blocks_uplift and liq_state in {"danger", "thin"}:
            reasons.append(f"liquidity_state:{liq_state}")
        if liq_score < self.min_liquidity_for_uplift:
            reasons.append(f"liquidity_score:{liq_score:.4f}")
        if adverse > self.max_adverse_selection_for_uplift:
            reasons.append(f"adverse_selection:{adverse:.4f}")
        if drawdown > self.drawdown_uplift_cutoff_pct:
            reasons.append(f"drawdown_pct:{drawdown:.2f}")

        return (len(reasons) == 0), reasons

    def _score_for_model(self, key: str, features: Dict[str, float]) -> float:
        c = _clamp01(features.get("confidence", 0.0))
        edge = _norm(features.get("expected_net_edge_bps", 0.0), 0.0, 40.0)
        spread_q = 1.0 - _norm(features.get("spread_bps", 0.0), 0.0, 30.0)
        liq = _clamp01(features.get("liquidity_score", 0.0))
        adv = _clamp01(features.get("adverse_selection_risk", 0.0))
        slippage = _norm(features.get("slippage_estimate_bps", 0.0), 0.0, 30.0)
        flow = _norm(abs(features.get("trade_velocity", 0.0)), 0.0, 20.0)
        imbalance = _norm(abs(features.get("order_book_imbalance", 0.0)), 0.0, 1.0)
        drawdown = _norm(features.get("strategy_drawdown_pct", 0.0), 0.0, 25.0)
        profit_factor = _norm(features.get("strategy_profit_factor", 1.0), 0.0, 3.0)
        expectancy = _norm(features.get("strategy_expectancy", 0.0), -10.0, 10.0)
        sharpe_like = _norm(features.get("strategy_sharpe_like", 0.0), -2.0, 3.0)
        fee_ratio = _norm(features.get("fee_ratio", 0.0), 0.0, 0.02)

        # Deterministic scoring families.
        if key in {"xgboost_edge", "lightgbm_edge", "catboost_edge"}:
            return _clamp01(0.45 * c + 0.45 * edge + 0.10 * spread_q)
        if key in {"sklearn_calibrated_classifier", "meta_labeling"}:
            return _clamp01(0.50 * c + 0.30 * edge + 0.20 * liq)
        if key in {"river_online_learner", "forecast_stacking", "bayesian_model_averaging"}:
            return _clamp01(0.35 * c + 0.35 * edge + 0.15 * profit_factor + 0.15 * expectancy)
        if key in {"regime_classifier", "hmm_regime", "change_point_detection"}:
            return _clamp01(0.30 * c + 0.25 * edge + 0.25 * flow + 0.20 * spread_q)
        if key in {"volatility_forecast", "correlation_cluster_risk", "portfolio_target_scaler"}:
            return _clamp01(0.40 * c + 0.25 * edge + 0.20 * liq - 0.15 * drawdown)
        if key in {"liquidity_state_classifier", "maker_fill_probability", "queue_time_to_fill"}:
            return _clamp01(0.20 * c + 0.25 * edge + 0.30 * liq + 0.25 * spread_q)
        if key in {"adverse_selection_classifier", "trade_flow_toxicity"}:
            return _clamp01(0.30 * c + 0.20 * edge + 0.20 * liq + 0.30 * (1.0 - adv))
        if key in {"slippage_quantile", "spread_expansion_predictor"}:
            return _clamp01(0.30 * c + 0.25 * edge + 0.20 * spread_q + 0.25 * (1.0 - slippage))
        if key in {"order_book_imbalance_predictor", "execution_style_selector"}:
            return _clamp01(0.30 * c + 0.30 * edge + 0.20 * imbalance + 0.20 * liq)
        if key in {"twap_slice_optimizer", "intent_failure_likelihood", "reconciliation_anomaly"}:
            return _clamp01(0.35 * c + 0.20 * edge + 0.20 * liq + 0.25 * (1.0 - slippage))
        if key in {"risk_breach_warning", "loss_streak_hazard", "fee_churn_predictor"}:
            return _clamp01(0.25 * c + 0.25 * edge + 0.20 * profit_factor + 0.15 * (1.0 - fee_ratio) + 0.15 * (1.0 - drawdown))
        if key in {"universe_ranking", "strategy_weighting", "strategy_disable_recover", "champion_challenger_scorer"}:
            return _clamp01(0.20 * c + 0.25 * edge + 0.20 * profit_factor + 0.20 * expectancy + 0.15 * sharpe_like)
        if key in {"feature_drift_detection", "data_quality_anomaly", "microstructure_autoencoder_anomaly"}:
            return _clamp01(0.30 * c + 0.20 * edge + 0.20 * liq + 0.30 * spread_q)
        if key in {"offline_llm_reviewer", "sentiment_finbert", "news_embedding", "walk_forward_model_selector"}:
            return _clamp01(0.30 * c + 0.30 * edge + 0.20 * liq + 0.20 * sharpe_like)
        return _clamp01(0.5 * c + 0.5 * edge)

    def _build_features(self, signal: Any) -> Dict[str, float]:
        return {
            "confidence": float(_signal_get(signal, "confidence", 0.0) or 0.0),
            "expected_net_edge_bps": float(_signal_get(signal, "expected_net_edge_bps", 0.0) or 0.0),
            "spread_bps": float(_signal_get(signal, "spread_bps", 0.0) or 0.0),
            "liquidity_score": float(_signal_get(signal, "liquidity_score", 0.0) or 0.0),
            "adverse_selection_risk": float(_signal_get(signal, "adverse_selection_risk", 0.0) or 0.0),
            "slippage_estimate_bps": float(_signal_get(signal, "slippage_estimate_bps", 0.0) or 0.0),
            "trade_velocity": float(_signal_get(signal, "trade_velocity", 0.0) or 0.0),
            "order_book_imbalance": float(_signal_get(signal, "order_book_imbalance", 0.0) or 0.0),
            "strategy_drawdown_pct": float(_signal_get(signal, "strategy_drawdown_pct", 0.0) or 0.0),
            "strategy_profit_factor": float(_signal_get(signal, "strategy_profit_factor", 1.0) or 1.0),
            "strategy_expectancy": float(_signal_get(signal, "strategy_expectancy", 0.0) or 0.0),
            "strategy_sharpe_like": float(_signal_get(signal, "strategy_sharpe_like", 0.0) or 0.0),
            "fee_ratio": float(_signal_get(signal, "fee_ratio", 0.0) or 0.0),
        }

    def annotate_candidates(self, signals: List[Any]) -> List[Any]:
        rows = list(signals or [])
        if not self.enabled or not rows:
            return rows

        base_w = float(self.score_weights.get("base_confidence", 1.0) or 1.0)
        ai_w = float(self.score_weights.get("ai_stack", 1.0) or 1.0)
        denom = max(1e-9, base_w + ai_w)

        for sig in rows:
            feats = self._build_features(sig)
            base_conf = _clamp01(feats.get("confidence", 0.0))

            model_scores: List[Tuple[str, float, float]] = []
            for key in self.MODEL_KEYS:
                if not self._model_enabled(key):
                    continue
                score = self._score_for_model(key, feats)
                weight = self._model_weight(key)
                model_scores.append((key, score, weight))

            if not model_scores or len(model_scores) < self.min_models_required:
                ai_score = base_conf
                used_models = ["baseline_fallback"]
            else:
                total_w = sum(max(0.0, w) for _, _, w in model_scores)
                if total_w <= 0.0:
                    ai_score = base_conf
                    used_models = ["baseline_fallback"]
                else:
                    ai_score = sum(s * max(0.0, w) for _, s, w in model_scores) / total_w
                    used_models = [k for k, _, _ in model_scores]

            uncertainty = float(self._weighted_stddev(model_scores))
            uncertainty_penalty = 0.0
            if self.uncertainty_penalty_enabled and len(model_scores) >= 2:
                # 0.0..0.5-ish stddev -> bounded penalty 0..max_uncertainty_penalty
                uncertainty_penalty = min(
                    max(0.0, self.max_uncertainty_penalty),
                    max(0.0, uncertainty * 2.0),
                )
                ai_score = _clamp01(ai_score * (1.0 - uncertainty_penalty))

            regime_mult = max(0.0, float(self._regime_multiplier(sig)))
            if regime_mult != 1.0:
                ai_score = _clamp01(ai_score * regime_mult)

            combined = _clamp01((base_w * base_conf + ai_w * ai_score) / denom)
            delta = max(
                -self.confidence_delta_cap,
                min(self.confidence_delta_cap, combined - base_conf),
            )
            uplift_allowed, uplift_block_reasons = self._can_uplift(sig)
            if not uplift_allowed and delta > 0.0:
                delta = 0.0
            new_conf = _clamp01(base_conf + delta)

            existing_adjust = float(_signal_get(sig, "meta_priority_adjustment", 0.0) or 0.0)
            _signal_set(sig, "confidence", new_conf)
            _signal_set(sig, "ai_stack_score", float(combined))
            _signal_set(sig, "ai_stack_base_confidence", float(base_conf))
            _signal_set(sig, "ai_stack_delta", float(delta))
            _signal_set(sig, "ai_stack_models_used", int(len(used_models)))
            _signal_set(sig, "ai_stack_models", ",".join(used_models))
            _signal_set(sig, "ai_stack_advisory_only", bool(self.advisory_only))
            _signal_set(sig, "ai_stack_reason", "deterministic_ensemble")
            _signal_set(sig, "ai_stack_uncertainty", float(uncertainty))
            _signal_set(sig, "ai_stack_uncertainty_penalty", float(uncertainty_penalty))
            _signal_set(sig, "ai_stack_regime_multiplier", float(regime_mult))
            _signal_set(sig, "ai_stack_uplift_blocked", bool(not uplift_allowed))
            _signal_set(sig, "ai_stack_uplift_block_reason", ",".join(uplift_block_reasons))
            _signal_set(sig, "meta_priority_adjustment", float(existing_adjust + delta))

        return rows
