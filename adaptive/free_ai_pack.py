from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    v = float(value)
    if v >= 0:
        z = math.exp(-v)
        return 1.0 / (1.0 + z)
    z = math.exp(v)
    return z / (1.0 + z)


def _signal_get(signal: Any, field: str, default: Any = None) -> Any:
    if isinstance(signal, dict):
        return signal.get(field, default)
    return getattr(signal, field, default)


def _signal_set(signal: Any, field: str, value: Any) -> None:
    if isinstance(signal, dict):
        signal[field] = value
        return
    setattr(signal, field, value)


class FreeAIPackEngine:
    """Optional open-source AI ensemble adapter.

    This engine is advisory-only by design in v1. It does not place orders;
    it only annotates/adjusts candidate confidence and priority.
    """

    def __init__(self, config: Any):
        self.config = config
        self.enabled = bool(getattr(config, "free_ai_pack_enabled", False))
        self.advisory_only = bool(getattr(config, "free_ai_pack_advisory_only", True))
        self.confidence_delta_cap = float(
            getattr(config, "free_ai_pack_confidence_delta_cap", 0.12) or 0.12
        )
        self.min_models_required = int(
            getattr(config, "free_ai_pack_min_models_required", 0) or 0
        )
        self.providers_cfg = dict(getattr(config, "free_ai_pack_providers", {}) or {})
        self.score_weights = dict(getattr(config, "free_ai_pack_score_weights", {}) or {})
        self._loaded_models: Dict[str, Any] = {}
        self._load_all_models()

    def _load_all_models(self) -> None:
        for provider in ("sklearn", "xgboost", "lightgbm", "river", "pytorch"):
            model = self._load_provider_model(provider)
            if model is not None:
                self._loaded_models[provider] = model

    def _provider_cfg(self, provider: str) -> Dict[str, Any]:
        cfg = self.providers_cfg.get(provider) or {}
        if isinstance(cfg, dict):
            return cfg
        return {}

    def _load_provider_model(self, provider: str) -> Optional[Any]:
        cfg = self._provider_cfg(provider)
        if not bool(cfg.get("enabled", False)):
            return None
        model_path = str(cfg.get("model_path", "") or "").strip()
        if not model_path:
            return None
        p = Path(model_path)
        if not p.exists():
            return None
        try:
            if provider == "sklearn":
                import joblib  # type: ignore

                return joblib.load(p)
            if provider == "xgboost":
                import xgboost as xgb  # type: ignore

                booster = xgb.Booster()
                booster.load_model(str(p))
                return booster
            if provider == "lightgbm":
                import lightgbm as lgb  # type: ignore

                return lgb.Booster(model_file=str(p))
            if provider == "river":
                with open(p, "rb") as f:
                    return pickle.load(f)
            if provider == "pytorch":
                import torch  # type: ignore

                # Prefer TorchScript modules for deterministic load in prod paths.
                return torch.jit.load(str(p))
        except Exception:
            return None
        return None

    def _build_features(self, signal: Any) -> Dict[str, float]:
        confidence = float(_signal_get(signal, "confidence", 0.0) or 0.0)
        score = float(_signal_get(signal, "score", confidence) or confidence)
        expected_net_edge_bps = float(_signal_get(signal, "expected_net_edge_bps", 0.0) or 0.0)
        spread_bps = float(_signal_get(signal, "spread_bps", 0.0) or 0.0)
        trade_velocity = float(_signal_get(signal, "trade_velocity", 0.0) or 0.0)
        imbalance = float(_signal_get(signal, "order_book_imbalance", 0.0) or 0.0)
        liquidity_score = float(_signal_get(signal, "liquidity_score", 0.0) or 0.0)
        adverse_selection_risk = float(_signal_get(signal, "adverse_selection_risk", 0.0) or 0.0)
        return {
            "confidence": confidence,
            "score": score,
            "expected_net_edge_bps": expected_net_edge_bps,
            "spread_bps": spread_bps,
            "trade_velocity": trade_velocity,
            "order_book_imbalance": imbalance,
            "liquidity_score": liquidity_score,
            "adverse_selection_risk": adverse_selection_risk,
        }

    @staticmethod
    def _to_vector(features: Dict[str, float]) -> List[float]:
        return [
            float(features.get("confidence", 0.0)),
            float(features.get("score", 0.0)),
            float(features.get("expected_net_edge_bps", 0.0)),
            float(features.get("spread_bps", 0.0)),
            float(features.get("trade_velocity", 0.0)),
            float(features.get("order_book_imbalance", 0.0)),
            float(features.get("liquidity_score", 0.0)),
            float(features.get("adverse_selection_risk", 0.0)),
        ]

    def _heuristic_score(self, features: Dict[str, float]) -> float:
        # Free fallback scorer when no model artifacts exist.
        confidence = _clamp01(features["confidence"])
        edge = max(0.0, min(100.0, features["expected_net_edge_bps"])) / 100.0
        spread_penalty = min(1.0, max(0.0, features["spread_bps"]) / 50.0)
        liquidity = _clamp01(features["liquidity_score"])
        adverse = _clamp01(features["adverse_selection_risk"])
        raw = 0.45 * confidence + 0.35 * edge + 0.25 * liquidity - 0.20 * spread_penalty - 0.20 * adverse
        return _clamp01(raw)

    def _predict_provider(self, provider: str, model: Any, features: Dict[str, float]) -> Optional[float]:
        vec = self._to_vector(features)
        try:
            if provider == "sklearn":
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba([vec])[0]
                    return _clamp01(float(pred[-1]))
                if hasattr(model, "decision_function"):
                    return _clamp01(_sigmoid(float(model.decision_function([vec])[0])))
                if hasattr(model, "predict"):
                    return _clamp01(float(model.predict([vec])[0]))
            elif provider == "xgboost":
                import numpy as np  # type: ignore
                import xgboost as xgb  # type: ignore

                arr = np.array([vec], dtype=float)
                pred = model.predict(xgb.DMatrix(arr))
                return _clamp01(float(pred[0]))
            elif provider == "lightgbm":
                import numpy as np  # type: ignore

                arr = np.array([vec], dtype=float)
                pred = model.predict(arr)
                return _clamp01(float(pred[0]))
            elif provider == "river":
                row = {
                    "confidence": vec[0],
                    "score": vec[1],
                    "expected_net_edge_bps": vec[2],
                    "spread_bps": vec[3],
                    "trade_velocity": vec[4],
                    "order_book_imbalance": vec[5],
                    "liquidity_score": vec[6],
                    "adverse_selection_risk": vec[7],
                }
                if hasattr(model, "predict_proba_one"):
                    p = model.predict_proba_one(row)
                    if isinstance(p, dict) and p:
                        # Assume positive class is max key if unknown; robust fallback.
                        return _clamp01(float(max(p.values())))
                if hasattr(model, "predict_one"):
                    return _clamp01(float(model.predict_one(row)))
            elif provider == "pytorch":
                import torch  # type: ignore

                tensor = torch.tensor([vec], dtype=torch.float32)
                with torch.no_grad():
                    out = model(tensor)
                if hasattr(out, "item"):
                    return _clamp01(float(out.item()))
                if hasattr(out, "__len__"):
                    return _clamp01(float(out[0]))
        except Exception:
            return None
        return None

    def annotate_candidates(self, signals: List[Any]) -> List[Any]:
        rows = list(signals or [])
        if not self.enabled or not rows:
            return rows

        w_base = float(self.score_weights.get("base_confidence", 1.0) or 1.0)
        w_ai = float(self.score_weights.get("ai_score", 1.0) or 1.0)
        denom = max(1e-9, w_base + w_ai)

        for sig in rows:
            features = self._build_features(sig)
            base_conf = _clamp01(features["confidence"])

            model_scores: List[float] = []
            model_names: List[str] = []
            for provider, model in self._loaded_models.items():
                pred = self._predict_provider(provider, model, features)
                if pred is None:
                    continue
                model_scores.append(_clamp01(pred))
                model_names.append(provider)

            if model_scores:
                ai_score = float(sum(model_scores) / len(model_scores))
            else:
                ai_score = float(self._heuristic_score(features))
                model_names.append("heuristic")

            if len(model_names) < self.min_models_required:
                # Not enough active models; keep baseline.
                ai_score = base_conf

            combined_score = _clamp01((w_base * base_conf + w_ai * ai_score) / denom)
            delta = max(
                -self.confidence_delta_cap,
                min(self.confidence_delta_cap, (combined_score - base_conf)),
            )
            new_conf = _clamp01(base_conf + delta)

            existing_adjust = float(_signal_get(sig, "meta_priority_adjustment", 0.0) or 0.0)
            _signal_set(sig, "confidence", new_conf)
            _signal_set(sig, "free_ai_score", combined_score)
            _signal_set(sig, "free_ai_base_confidence", base_conf)
            _signal_set(sig, "free_ai_delta", delta)
            _signal_set(sig, "free_ai_models_used", int(len(model_names)))
            _signal_set(sig, "free_ai_models", ",".join(model_names))
            _signal_set(sig, "free_ai_advisory_only", bool(self.advisory_only))
            _signal_set(sig, "meta_priority_adjustment", float(existing_adjust + delta))
            _signal_set(
                sig,
                "free_ai_reason",
                "model_ensemble" if model_scores else "heuristic_fallback",
            )

        return rows
