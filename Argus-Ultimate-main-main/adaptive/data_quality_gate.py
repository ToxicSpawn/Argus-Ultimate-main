from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class DataQualityDecision:
    symbol: str
    quality_score: float
    stale_age_seconds: float
    spread_bps: float
    depth_estimate: float
    allowed: bool
    reasons: List[str]


class DataQualityGate:
    """Deterministic pre-gate for stale/wide/thin market data."""

    def __init__(self, config: Any):
        self.enabled = bool(getattr(config, "data_quality_gate_enabled", True))
        self.min_quality_score = float(
            getattr(config, "data_quality_gate_min_quality_score", 0.45) or 0.45
        )
        self.max_stale_age_seconds = float(
            getattr(config, "data_quality_gate_max_stale_age_seconds", 20.0) or 20.0
        )
        self.max_spread_bps = float(
            getattr(config, "data_quality_gate_max_spread_bps", 25.0) or 25.0
        )
        self.min_depth = float(getattr(config, "data_quality_gate_min_depth", 0.05) or 0.05)
        self.fail_closed = bool(getattr(config, "data_quality_gate_fail_closed", True))

    @staticmethod
    def _signal_get(signal: Any, field: str, default: Any = None) -> Any:
        if isinstance(signal, dict):
            return signal.get(field, default)
        return getattr(signal, field, default)

    @staticmethod
    def _signal_set(signal: Any, field: str, value: Any) -> None:
        if isinstance(signal, dict):
            signal[field] = value
        else:
            setattr(signal, field, value)

    def _stale_age_seconds(self, signal: Any) -> float:
        direct = self._signal_get(signal, "market_data_age_seconds", None)
        if direct is not None:
            return max(0.0, float(direct or 0.0))
        staleness_ms = self._signal_get(signal, "book_staleness_ms", None)
        if staleness_ms is not None:
            return max(0.0, float(staleness_ms or 0.0) / 1000.0)
        return 0.0

    def _depth_estimate(self, signal: Any) -> float:
        direct = self._signal_get(signal, "orderbook_depth_estimate", None)
        if direct is None:
            direct = self._signal_get(signal, "depth", None)
        if direct is not None:
            return max(0.0, float(direct or 0.0))
        bid = float(
            self._signal_get(signal, "top_of_book_bid_size", None)
            or self._signal_get(signal, "bid_size_1", None)
            or self._signal_get(signal, "bid_size", 0.0)
            or 0.0
        )
        ask = float(
            self._signal_get(signal, "top_of_book_ask_size", None)
            or self._signal_get(signal, "ask_size_1", None)
            or self._signal_get(signal, "ask_size", 0.0)
            or 0.0
        )
        return max(0.0, bid + ask)

    def evaluate(self, signal: Any) -> DataQualityDecision:
        symbol = str(self._signal_get(signal, "symbol", "") or "")
        stale_age = self._stale_age_seconds(signal)
        spread_bps = max(0.0, float(self._signal_get(signal, "spread_bps", 0.0) or 0.0))
        depth = self._depth_estimate(signal)

        reasons: List[str] = []
        stale_penalty = 0.0
        spread_penalty = 0.0
        depth_penalty = 0.0

        if stale_age > self.max_stale_age_seconds:
            reasons.append("stale_data")
            stale_penalty = min(1.0, (stale_age - self.max_stale_age_seconds) / max(self.max_stale_age_seconds, 1e-9))
        if spread_bps > self.max_spread_bps:
            reasons.append("wide_spread")
            spread_penalty = min(1.0, (spread_bps - self.max_spread_bps) / max(self.max_spread_bps, 1e-9))
        if depth < self.min_depth:
            reasons.append("thin_depth")
            depth_penalty = min(1.0, (self.min_depth - depth) / max(self.min_depth, 1e-9))

        quality_score = _clamp01(1.0 - (0.45 * stale_penalty + 0.30 * spread_penalty + 0.35 * depth_penalty))
        score_violation = quality_score < self.min_quality_score
        if score_violation and "low_quality_score" not in reasons:
            reasons.append("low_quality_score")

        allowed = not (self.fail_closed and (bool(reasons) or score_violation))
        return DataQualityDecision(
            symbol=symbol,
            quality_score=float(quality_score),
            stale_age_seconds=float(stale_age),
            spread_bps=float(spread_bps),
            depth_estimate=float(depth),
            allowed=bool(allowed),
            reasons=reasons,
        )

    def filter_candidates(self, signals: Iterable[Any]) -> Tuple[List[Any], List[Tuple[Any, DataQualityDecision]]]:
        rows = list(signals or [])
        if not self.enabled:
            return rows, []

        kept: List[Any] = []
        blocked: List[Tuple[Any, DataQualityDecision]] = []
        for sig in rows:
            decision = self.evaluate(sig)
            self._signal_set(sig, "data_quality_score", float(decision.quality_score))
            self._signal_set(sig, "data_quality_allowed", bool(decision.allowed))
            self._signal_set(sig, "data_quality_reject_reason", ",".join(decision.reasons))
            self._signal_set(sig, "data_quality_stale_age_seconds", float(decision.stale_age_seconds))
            self._signal_set(sig, "data_quality_depth_estimate", float(decision.depth_estimate))
            if decision.allowed:
                kept.append(sig)
            else:
                blocked.append((sig, decision))
        return kept, blocked
