"""Adaptive configuration learner with guardrails.

This module lets Argus improve runtime settings without mutating the canonical
configuration file. It writes a small learned overlay that can be applied at
startup, rolled back instantly, and promoted only after paper validation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TunableParameter:
    path: str
    minimum: float
    maximum: float
    step: float
    default: float


@dataclass
class ConfigObservation:
    win_rate: float
    profit_factor: float
    drawdown: float
    average_slippage: float
    regime: str
    trades: int


@dataclass
class LearnedAdjustment:
    path: str
    old_value: float
    new_value: float
    reason: str
    confidence: float


class AdaptiveConfigLearner:
    """Learns bounded config overlays from paper/live performance metrics."""

    PARAMETERS = {
        "risk.stop_loss_pct": TunableParameter("risk.stop_loss_pct", 0.006, 0.05, 0.001, 0.012),
        "risk.take_profit_pct": TunableParameter("risk.take_profit_pct", 0.012, 0.12, 0.002, 0.035),
        "risk.max_daily_loss_pct": TunableParameter("risk.max_daily_loss_pct", 0.03, 0.25, 0.01, 0.10),
        "capital.max_position_pct": TunableParameter("capital.max_position_pct", 0.05, 0.35, 0.01, 0.20),
        "capital.max_total_exposure_pct": TunableParameter("capital.max_total_exposure_pct", 0.20, 0.95, 0.025, 0.60),
    }

    def __init__(self, overlay_path: str | Path = "config/runtime/adaptive_overlay.json"):
        self.overlay_path = Path(overlay_path)
        self.history_path = self.overlay_path.with_suffix(".history.jsonl")
        self.min_trades_for_adjustment = 30
        self.max_adjustments_per_cycle = 3

    def suggest(self, config: dict[str, Any], observation: ConfigObservation) -> list[LearnedAdjustment]:
        if observation.trades < self.min_trades_for_adjustment:
            return []

        candidates: list[LearnedAdjustment] = []

        if observation.drawdown > 0.10:
            candidates.append(self._adjust(config, "capital.max_position_pct", -1, "drawdown above 10%; reduce single-position exposure", 0.90))
            candidates.append(self._adjust(config, "capital.max_total_exposure_pct", -1, "drawdown above 10%; reduce total exposure", 0.90))
            candidates.append(self._adjust(config, "risk.max_daily_loss_pct", -1, "drawdown elevated; tighten daily loss guard", 0.85))

        if observation.average_slippage > 0.002:
            candidates.append(self._adjust(config, "capital.max_position_pct", -1, "slippage high; reduce order size pressure", 0.80))

        if observation.profit_factor > 1.35 and observation.drawdown < 0.06 and observation.win_rate > 0.53:
            candidates.append(self._adjust(config, "capital.max_position_pct", 1, "edge stable; cautiously increase single-position exposure", 0.70))
            candidates.append(self._adjust(config, "capital.max_total_exposure_pct", 1, "edge stable; cautiously increase total deployment", 0.65))

        if observation.regime == "volatile":
            candidates.append(self._adjust(config, "risk.stop_loss_pct", 1, "volatile regime; widen stop slightly to avoid noise exits", 0.70))
            candidates.append(self._adjust(config, "risk.take_profit_pct", 1, "volatile regime; widen target for larger moves", 0.65))
        elif observation.regime in {"range", "stable"} and observation.profit_factor < 1.10:
            candidates.append(self._adjust(config, "risk.take_profit_pct", -1, "range regime underperforming; take profits sooner", 0.65))

        unique: dict[str, LearnedAdjustment] = {}
        for adjustment in candidates:
            if adjustment.old_value != adjustment.new_value:
                existing = unique.get(adjustment.path)
                if existing is None or adjustment.confidence > existing.confidence:
                    unique[adjustment.path] = adjustment
        return sorted(unique.values(), key=lambda item: item.confidence, reverse=True)[: self.max_adjustments_per_cycle]

    def write_overlay(self, adjustments: list[LearnedAdjustment]) -> dict[str, Any]:
        overlay = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "candidate",
            "requires_paper_validation": True,
            "adjustments": [asdict(item) for item in adjustments],
        }
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        self.overlay_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(overlay) + "\n")
        return overlay

    def apply_overlay(self, config: dict[str, Any], overlay: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = json.loads(json.dumps(config))
        if overlay is None:
            if not self.overlay_path.exists():
                return merged
            overlay = json.loads(self.overlay_path.read_text(encoding="utf-8"))
        for item in overlay.get("adjustments", []):
            self._set_path(merged, str(item["path"]), float(item["new_value"]))
        return merged

    def promote_if_validated(self, validation: dict[str, float]) -> bool:
        """Mark overlay as promoted only if paper validation clears thresholds."""
        if not self.overlay_path.exists():
            return False
        passes = (
            validation.get("profit_factor", 0.0) >= 1.20
            and validation.get("max_drawdown", 1.0) <= 0.12
            and validation.get("trades", 0.0) >= self.min_trades_for_adjustment
        )
        overlay = json.loads(self.overlay_path.read_text(encoding="utf-8"))
        overlay["status"] = "promoted" if passes else "rejected"
        overlay["validation"] = validation
        overlay["validated_at"] = datetime.now(timezone.utc).isoformat()
        self.overlay_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
        return passes

    def rollback(self) -> None:
        if self.overlay_path.exists():
            self.overlay_path.unlink()

    def _adjust(self, config: dict[str, Any], path: str, direction: int, reason: str, confidence: float) -> LearnedAdjustment:
        parameter = self.PARAMETERS[path]
        old_value = float(self._get_path(config, path, parameter.default))
        new_value = self._clamp(old_value + direction * parameter.step, parameter.minimum, parameter.maximum)
        return LearnedAdjustment(path, old_value, new_value, reason, confidence)

    @staticmethod
    def _get_path(config: dict[str, Any], path: str, default: float) -> float:
        node: Any = config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return float(node)

    @staticmethod
    def _set_path(config: dict[str, Any], path: str, value: float) -> None:
        node = config
        parts = path.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return round(max(min(value, maximum), minimum), 6)


def _demo() -> None:
    config = {
        "risk": {"stop_loss_pct": 0.012, "take_profit_pct": 0.035, "max_daily_loss_pct": 0.25},
        "capital": {"max_position_pct": 0.35, "max_total_exposure_pct": 0.95},
    }
    learner = AdaptiveConfigLearner("reports/adaptive_overlay_demo.json")
    observation = ConfigObservation(win_rate=0.48, profit_factor=0.95, drawdown=0.13, average_slippage=0.0025, regime="volatile", trades=80)
    adjustments = learner.suggest(config, observation)
    overlay = learner.write_overlay(adjustments)
    merged = learner.apply_overlay(config, overlay)
    print("Adaptive config learner ready")
    print(json.dumps({"adjustments": overlay["adjustments"], "merged": merged}, indent=2))


if __name__ == "__main__":
    _demo()
