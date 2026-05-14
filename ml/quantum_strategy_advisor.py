"""Local quantum advisory signal for trading strategies.

This module adds quantum to strategies in a deliberately conservative way:
it calls the canonical local quantum facade, interprets simulated quantum-walk
asset weights as an advisory tilt, and emits a standard PredictionBundle.

No IBM/D-Wave/cloud hardware is contacted. If the quantum layer is unavailable
or input data is insufficient, the advisor returns a neutral hold decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from ml.prediction_bus import PredictionBundle


@dataclass
class QuantumWalkFeatureSummary:
    """Bounded diagnostics extracted from local quantum-walk output."""

    directional_bias: float
    dispersion: float
    concentration: float
    conviction: float
    entropy: Optional[float] = None
    mixing_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "directional_bias": round(self.directional_bias, 6),
            "dispersion": round(self.dispersion, 6),
            "concentration": round(self.concentration, 6),
            "conviction": round(self.conviction, 6),
            "entropy": self.entropy,
            "mixing_time": self.mixing_time,
        }


@dataclass
class QuantumTailRiskConsensus:
    """Conservative summary of local quantum-inspired tail-risk estimators."""

    var: float
    cvar: float
    confidence_level: float
    estimator_count: int
    disagreement: float
    risk_level: str
    qmc: Optional[Dict[str, Any]] = None
    mlqae: Optional[Dict[str, Any]] = None
    honest_claim: str = (
        "Conservative local tail-risk consensus from classical QMC/MLQAE simulations; "
        "no hardware quantum advantage is claimed."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var": round(self.var, 8),
            "cvar": round(self.cvar, 8),
            "confidence_level": round(self.confidence_level, 6),
            "estimator_count": self.estimator_count,
            "disagreement": round(self.disagreement, 8),
            "risk_level": self.risk_level,
            "qmc": self.qmc,
            "mlqae": self.mlqae,
            "honest_claim": self.honest_claim,
        }


@dataclass
class QuantumStrategyAdvice:
    """Strategy-safe interpretation of local quantum analysis."""

    symbol: str
    action: str
    direction: float
    strength: float
    confidence: float
    size_multiplier: float
    weights: Dict[str, float]
    execution_mode: str
    honest_claim: str
    feature_summary: Optional[QuantumWalkFeatureSummary] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_bundle(self, *, regime: str = "UNKNOWN", regime_confidence: float = 0.0) -> PredictionBundle:
        return PredictionBundle(
            symbol=self.symbol,
            action=self.action,
            direction=self.direction,
            strength=self.strength,
            confidence=self.confidence,
            regime=regime,
            regime_confidence=regime_confidence,
            size_multiplier=self.size_multiplier,
            sources={"quantum_walk": self.weights},
            ml_outputs={"quantum_strategy_advice": self.to_dict()},
            metadata={
                "execution_mode": self.execution_mode,
                "honest_claim": self.honest_claim,
                "quantum_features": self.feature_summary.to_dict() if self.feature_summary else None,
                **self.metadata,
            },
            timestamp=self.timestamp,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "direction": round(self.direction, 6),
            "strength": round(self.strength, 6),
            "confidence": round(self.confidence, 6),
            "size_multiplier": round(self.size_multiplier, 6),
            "weights": self.weights,
            "execution_mode": self.execution_mode,
            "honest_claim": self.honest_claim,
            "feature_summary": self.feature_summary.to_dict() if self.feature_summary else None,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class LocalQuantumStrategyAdvisor:
    """Generate optional local quantum-walk advisory bundles for strategies."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        facade: Optional[Any] = None,
        correlation_threshold: float = 0.3,
        max_steps: int = 50,
        strategy: str = "centrality",
        min_history: int = 3,
        neutral_band: float = 0.05,
    ) -> None:
        self.enabled = bool(enabled)
        self.facade = facade
        self.correlation_threshold = float(correlation_threshold)
        self.max_steps = int(max_steps)
        self.strategy = strategy
        self.min_history = max(int(min_history), 2)
        self.neutral_band = max(float(neutral_band), 0.0)

    def advise(
        self,
        symbol: str,
        returns: Dict[str, list[float]],
        *,
        regime: str = "UNKNOWN",
        regime_confidence: float = 0.0,
    ) -> QuantumStrategyAdvice:
        if not self.enabled:
            return self._neutral(symbol, "quantum_advisor_disabled")

        valid_returns = self._validate_returns(returns)
        if symbol not in valid_returns:
            return self._neutral(symbol, "symbol_missing_from_returns")
        if len(valid_returns) < 2:
            return self._neutral(symbol, "need_at_least_two_assets")

        try:
            facade = self.facade or self._local_facade()
            result = facade.run_quantum_walk(
                valid_returns,
                correlation_threshold=self.correlation_threshold,
                max_steps=self.max_steps,
                strategy=self.strategy,
            )
            return self._interpret_result(symbol, valid_returns, result)
        except Exception as exc:
            return self._neutral(symbol, "quantum_walk_failed", error=str(exc))

    def advise_bundle(
        self,
        symbol: str,
        returns: Dict[str, list[float]],
        *,
        regime: str = "UNKNOWN",
        regime_confidence: float = 0.0,
    ) -> PredictionBundle:
        return self.advise(
            symbol,
            returns,
            regime=regime,
            regime_confidence=regime_confidence,
        ).to_bundle(regime=regime, regime_confidence=regime_confidence)

    def assess_tail_risk(
        self,
        returns: list[float],
        *,
        confidence: float = 0.95,
        n_samples: int = 10000,
        n_qubits: int = 4,
    ) -> QuantumTailRiskConsensus:
        """Compare local QMC and MLQAE tail-risk estimates conservatively.

        The result uses the worse loss estimate from available estimators. This is
        intentionally defensive: strategy code can treat it as a risk overlay, not
        as a promise that either simulated estimator is superior to classical VaR.
        """
        valid_returns = [float(value) for value in returns if np.isfinite(float(value))]
        if not self.enabled or len(valid_returns) < self.min_history:
            return self._neutral_tail_risk(confidence, "tail_risk_unavailable")

        qmc_result: Optional[Dict[str, Any]] = None
        mlqae_result: Optional[Dict[str, Any]] = None
        qmc_var: Optional[float] = None
        qmc_cvar: Optional[float] = None
        mlqae_var: Optional[float] = None
        mlqae_cvar: Optional[float] = None

        try:
            facade = self.facade or self._local_facade()
            qmc_result = facade.estimate_tail_risk_qmc(
                valid_returns,
                n_samples=n_samples,
                confidence=confidence,
            )
            qmc_var, qmc_cvar = self._extract_tail_metrics(qmc_result, confidence)
        except Exception as exc:
            qmc_result = {"error": str(exc)}

        try:
            facade = self.facade or self._local_facade()
            mlqae_result = facade.estimate_tail_risk_mlqae(
                valid_returns,
                confidence=confidence,
                n_samples=n_samples,
                n_qubits=n_qubits,
            )
            mlqae_var, mlqae_cvar = self._extract_tail_metrics(mlqae_result, confidence)
        except Exception as exc:
            mlqae_result = {"error": str(exc)}

        vars_available = [value for value in (qmc_var, mlqae_var) if value is not None]
        cvars_available = [value for value in (qmc_cvar, mlqae_cvar) if value is not None]
        if not vars_available or not cvars_available:
            return self._neutral_tail_risk(
                confidence,
                "tail_risk_estimators_failed",
                qmc=qmc_result,
                mlqae=mlqae_result,
            )

        conservative_var = min(vars_available)
        conservative_cvar = min(cvars_available)
        disagreement = 0.0
        if qmc_var is not None and mlqae_var is not None:
            scale = max(abs(qmc_var), abs(mlqae_var), 1e-9)
            disagreement = abs(qmc_var - mlqae_var) / scale

        return QuantumTailRiskConsensus(
            var=float(conservative_var),
            cvar=float(conservative_cvar),
            confidence_level=float(confidence),
            estimator_count=len(vars_available),
            disagreement=float(np.clip(disagreement, 0.0, 1.0)),
            risk_level=self._tail_risk_level(conservative_var, conservative_cvar),
            qmc=qmc_result,
            mlqae=mlqae_result,
        )

    def _interpret_result(
        self,
        symbol: str,
        returns: Dict[str, list[float]],
        result: Dict[str, Any],
    ) -> QuantumStrategyAdvice:
        raw_weights = result.get("weights", {}) if isinstance(result, dict) else {}
        weights = {str(k): float(v) for k, v in raw_weights.items()}
        if not weights or symbol not in weights:
            return self._neutral(symbol, "quantum_walk_no_symbol_weight", weights=weights)

        equal_weight = 1.0 / max(len(weights), 1)
        target_weight = float(weights.get(symbol, 0.0))
        relative_tilt = (target_weight - equal_weight) / max(equal_weight, 1e-9)
        direction = float(np.clip(relative_tilt, -1.0, 1.0))
        feature_summary = self.summarize_quantum_walk(symbol, weights, result)

        if abs(direction) < self.neutral_band:
            action = "hold"
            strength = 0.0
            confidence = 0.35
            size_multiplier = 1.0
        else:
            action = "buy" if direction > 0 else "sell"
            strength = float(np.clip(abs(direction), 0.0, 1.0))
            confidence = float(np.clip(0.45 + strength * 0.25 + feature_summary.conviction * 0.30, 0.0, 1.0))
            size_multiplier = float(np.clip(1.0 + direction * 0.25, 0.5, 1.5))

        quantum_metadata = result.get("quantum_metadata", {}) if isinstance(result, dict) else {}
        return QuantumStrategyAdvice(
            symbol=symbol,
            action=action,
            direction=direction,
            strength=strength,
            confidence=confidence,
            size_multiplier=size_multiplier,
            weights=weights,
            execution_mode=str(quantum_metadata.get("execution_mode", "classical_statevector_simulation")),
            honest_claim=str(
                quantum_metadata.get(
                    "honest_claim",
                    "Local quantum-walk advisory from classical simulation; no hardware quantum advantage is claimed.",
                )
            ),
            feature_summary=feature_summary,
            metadata={
                "method": result.get("method", "quantum_walk") if isinstance(result, dict) else "quantum_walk",
                "walk_entropy": result.get("walk_entropy") if isinstance(result, dict) else None,
                "mixing_time": result.get("mixing_time") if isinstance(result, dict) else None,
                "relative_tilt": relative_tilt,
                "equal_weight": equal_weight,
                "target_weight": target_weight,
            },
        )

    def summarize_quantum_walk(
        self,
        symbol: str,
        weights: Dict[str, float],
        result: Optional[Dict[str, Any]] = None,
    ) -> QuantumWalkFeatureSummary:
        """Extract bounded, deterministic features from quantum-walk weights."""
        if not weights or symbol not in weights:
            return QuantumWalkFeatureSummary(0.0, 0.0, 0.0, 0.0)

        equal_weight = 1.0 / max(len(weights), 1)
        target_weight = float(weights.get(symbol, 0.0))
        directional_bias = float(np.clip((target_weight - equal_weight) / max(equal_weight, 1e-9), -1.0, 1.0))
        dispersion = self._weight_dispersion(weights)
        concentration = float(np.clip(max(weights.values()) if weights else 0.0, 0.0, 1.0))

        # Conviction rises only when the target tilt is meaningful and the walk
        # distribution is differentiated. Flat/equal weights remain low.
        conviction = float(np.clip(abs(directional_bias) * (0.65 + 0.35 * dispersion), 0.0, 1.0))
        entropy = result.get("walk_entropy") if isinstance(result, dict) else None
        mixing_time = result.get("mixing_time") if isinstance(result, dict) else None
        return QuantumWalkFeatureSummary(
            directional_bias=directional_bias,
            dispersion=dispersion,
            concentration=concentration,
            conviction=conviction,
            entropy=float(entropy) if entropy is not None else None,
            mixing_time=float(mixing_time) if mixing_time is not None else None,
        )

    def _validate_returns(self, returns: Dict[str, list[float]]) -> Dict[str, list[float]]:
        valid: Dict[str, list[float]] = {}
        for asset, values in (returns or {}).items():
            numeric = [float(value) for value in values if np.isfinite(float(value))]
            if len(numeric) >= self.min_history:
                valid[str(asset)] = numeric
        return valid

    @staticmethod
    def _weight_dispersion(weights: Dict[str, float]) -> float:
        values = np.asarray(list(weights.values()), dtype=float)
        if values.size <= 1:
            return 0.0
        return float(np.clip(np.std(values) / max(np.mean(values), 1e-9), 0.0, 1.0))

    @staticmethod
    def _extract_tail_metrics(result: Optional[Dict[str, Any]], confidence: float) -> tuple[Optional[float], Optional[float]]:
        if not isinstance(result, dict):
            return None, None
        suffix = str(int(round(confidence * 100)))
        var = result.get(f"var_{suffix}", result.get("var"))
        cvar = result.get(f"cvar_{suffix}", result.get("cvar"))
        if var is None or cvar is None:
            return None, None
        return float(var), float(cvar)

    @staticmethod
    def _tail_risk_level(var: float, cvar: float) -> str:
        loss = max(abs(min(var, 0.0)), abs(min(cvar, 0.0)))
        if loss >= 0.08:
            return "extreme"
        if loss >= 0.04:
            return "high"
        if loss >= 0.02:
            return "elevated"
        return "normal"

    @staticmethod
    def _neutral_tail_risk(
        confidence: float,
        reason: str,
        **metadata: Any,
    ) -> QuantumTailRiskConsensus:
        return QuantumTailRiskConsensus(
            var=0.0,
            cvar=0.0,
            confidence_level=float(confidence),
            estimator_count=0,
            disagreement=0.0,
            risk_level="unknown",
            honest_claim="Neutral fallback; no quantum tail-risk advisory was applied.",
            qmc=metadata.get("qmc", {"reason": reason}),
            mlqae=metadata.get("mlqae"),
        )

    @staticmethod
    def _local_facade() -> Any:
        import quantum

        return quantum.get_quantum_facade(hardware_enabled=False)

    @staticmethod
    def _neutral(symbol: str, reason: str, **metadata: Any) -> QuantumStrategyAdvice:
        return QuantumStrategyAdvice(
            symbol=symbol,
            action="hold",
            direction=0.0,
            strength=0.0,
            confidence=0.0,
            size_multiplier=1.0,
            weights=dict(metadata.pop("weights", {}) or {}),
            execution_mode="local_fallback",
            honest_claim="Neutral fallback; no quantum advisory was applied.",
            feature_summary=QuantumWalkFeatureSummary(0.0, 0.0, 0.0, 0.0),
            metadata={"reason": reason, **metadata},
        )
