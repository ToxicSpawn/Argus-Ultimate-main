from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(slots=True)
class SignalStabilityDecision:
    symbol: str
    side: str
    cycle_age: int
    suppressed: bool
    reason: str
    confidence_improvement: float
    edge_improvement_bps: float


class SignalStabilityEngine:
    """Suppress rapid flip-flop signals unless quality improves materially."""

    def __init__(self, config: Any):
        self.enabled = bool(getattr(config, "signal_stability_enabled", True))
        self.min_hold_cycles = int(getattr(config, "signal_stability_min_hold_cycles", 2) or 2)
        self.min_confidence_improvement = float(
            getattr(config, "signal_stability_min_confidence_improvement", 0.05) or 0.05
        )
        self.min_edge_improvement_bps = float(
            getattr(config, "signal_stability_min_edge_improvement_bps", 2.0) or 2.0
        )
        self.allow_opposite_side_with_strong_edge = bool(
            getattr(config, "signal_stability_allow_opposite_side_with_strong_edge", True)
        )
        self.fail_closed = bool(getattr(config, "signal_stability_fail_closed", True))
        self._last_by_symbol: Dict[str, Dict[str, Any]] = {}

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

    @staticmethod
    def _norm_side(raw: Any) -> str:
        side = str(raw or "").strip().upper()
        if side in {"BUY", "LONG"}:
            return "BUY"
        if side in {"SELL", "SHORT"}:
            return "SELL"
        return side

    def filter_candidates(
        self,
        signals: Iterable[Any],
        *,
        cycle_id: int,
    ) -> Tuple[List[Any], List[Tuple[Any, SignalStabilityDecision]]]:
        rows = list(signals or [])
        if not self.enabled:
            return rows, []

        kept: List[Any] = []
        blocked: List[Tuple[Any, SignalStabilityDecision]] = []

        for sig in rows:
            symbol = str(self._signal_get(sig, "symbol", "") or "")
            side = self._norm_side(
                self._signal_get(sig, "side", None) or self._signal_get(sig, "action", None)
            )
            confidence = float(self._signal_get(sig, "confidence", 0.0) or 0.0)
            edge_bps = float(self._signal_get(sig, "expected_net_edge_bps", 0.0) or 0.0)
            prev = dict(self._last_by_symbol.get(symbol, {}) or {})
            prev_side = self._norm_side(prev.get("side", ""))
            prev_conf = float(prev.get("confidence", 0.0) or 0.0)
            prev_edge = float(prev.get("edge_bps", 0.0) or 0.0)
            prev_cycle = int(prev.get("cycle_id", cycle_id - self.min_hold_cycles - 1) or 0)
            age_cycles = int(max(0, int(cycle_id) - prev_cycle))

            confidence_improvement = float(confidence - prev_conf)
            edge_improvement_bps = float(edge_bps - prev_edge)
            opposite_side = bool(prev_side and side and prev_side != side)
            within_hold = age_cycles < max(0, self.min_hold_cycles)
            strong_override = bool(
                self.allow_opposite_side_with_strong_edge
                and confidence_improvement >= self.min_confidence_improvement
                and edge_improvement_bps >= self.min_edge_improvement_bps
            )
            suppressed = bool(opposite_side and within_hold and not strong_override and self.fail_closed)
            reason = ""
            if suppressed:
                reason = "flip_flop_suppressed"
            elif opposite_side and within_hold and strong_override:
                reason = "flip_override_strong_edge"

            decision = SignalStabilityDecision(
                symbol=symbol,
                side=side,
                cycle_age=age_cycles,
                suppressed=suppressed,
                reason=reason,
                confidence_improvement=confidence_improvement,
                edge_improvement_bps=edge_improvement_bps,
            )

            self._signal_set(sig, "signal_stability_suppressed", bool(suppressed))
            self._signal_set(sig, "signal_stability_reason", str(reason))
            self._signal_set(sig, "signal_stability_age_cycles", int(age_cycles))
            self._signal_set(sig, "signal_stability_conf_improvement", float(confidence_improvement))
            self._signal_set(sig, "signal_stability_edge_improvement_bps", float(edge_improvement_bps))

            if suppressed:
                blocked.append((sig, decision))
                continue

            self._last_by_symbol[symbol] = {
                "side": side,
                "confidence": confidence,
                "edge_bps": edge_bps,
                "cycle_id": int(cycle_id),
            }
            kept.append(sig)
        return kept, blocked
