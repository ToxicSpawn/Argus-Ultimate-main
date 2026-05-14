"""
Advisory Bridge — converts existing ComponentRegistry advisory output
into structured proposals for the sealed runtime.

The existing system produces a dict of 54+ advisory keys every cycle via
ComponentRegistry.on_cycle(). This bridge:
1. Collects that advisory dict
2. Validates and classifies each key
3. Produces structured AdvisoryProposal objects
4. Feeds them to the sealed runtime's target engine

The bridge NEVER executes orders. It only produces proposals.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvisoryProposal:
    """A structured proposal from an existing ARGUS module."""
    source: str           # which advisory key produced this
    category: str         # "target", "risk", "execution_hint", "informational"
    symbol: str           # affected symbol (or "" for portfolio-wide)
    direction: str        # "buy", "sell", "hold", "reduce"
    strength: float       # 0.0 to 1.0
    confidence: float     # 0.0 to 1.0
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# Classification of advisory keys into authority levels
_TARGET_KEYS = {
    "ensemble", "stacked_signal", "kalman_pairs", "funding_harvester",
    "pretrained_alpha", "alpha_scores",
}

_RISK_KEYS = {
    "system_status", "risk_score", "quantum_anomaly_score", "market_anomaly",
    "correlation_penalty", "tail_hedge", "stress_test", "adaptive_risk",
    "quantum_risk_check",
}

_EXECUTION_HINT_KEYS = {
    "toxicity", "execution_intelligence", "tca_score", "session_effect",
    "market_impact",
}

_INFORMATIONAL_KEYS = {
    "vol_forecasts", "fear_greed", "llm_analysis", "sentiment_stats",
    "chart_patterns", "regime_rotation", "regime_prediction",
    "regime_pre_transition_signals", "bandit_rankings", "funding_prediction",
    "gnn_asset_flow", "autoencoder_regime", "attention_orderflow",
    "orderbook_prediction", "quantum_prediction", "quantum_regime",
    "quantum_signal_quality", "quantum_portfolio", "fragility_score",
    "antifragile_multiplier", "bleeders", "online_learner",
    "feature_discovery", "genetic_evolver", "strategy_optimization",
    "performance_scorecard", "portfolio_snapshot", "strategy_scanner",
    "liquidation_cascade",
}


def classify_advisory_key(key: str) -> str:
    """Classify an advisory key into its authority category."""
    if key in _TARGET_KEYS:
        return "target"
    if key in _RISK_KEYS:
        return "risk"
    if key in _EXECUTION_HINT_KEYS:
        return "execution_hint"
    return "informational"


class AdvisoryBridge:
    """
    Bridges the existing ComponentRegistry advisory dict into
    structured proposals for the sealed runtime.

    Usage:
        bridge = AdvisoryBridge()
        # After ComponentRegistry.on_cycle() produces advisory dict:
        proposals = bridge.process(advisory_dict)
        # Feed proposals to sealed runtime target engine
    """

    def __init__(self, blocked_keys: Optional[set] = None):
        self._blocked = blocked_keys or set()
        self._proposal_count = 0
        self._last_process_ts = 0.0

    def process(self, advisory: Dict[str, Any]) -> List[AdvisoryProposal]:
        """
        Convert raw advisory dict into structured proposals.

        Blocked keys are filtered out. Each key is classified by authority level.
        Only keys with actionable content produce proposals.
        """
        if not advisory:
            return []

        proposals: List[AdvisoryProposal] = []
        self._last_process_ts = time.time()

        for key, value in advisory.items():
            if key in self._blocked:
                continue
            if value is None:
                continue

            category = classify_advisory_key(key)

            # Extract proposals based on category
            try:
                extracted = self._extract_proposals(key, category, value)
                proposals.extend(extracted)
            except Exception as e:
                logger.debug("AdvisoryBridge: failed to extract from %s: %s", key, e)

        self._proposal_count += len(proposals)
        return proposals

    def _extract_proposals(self, key: str, category: str, value: Any) -> List[AdvisoryProposal]:
        """Extract structured proposals from a single advisory value."""
        proposals = []

        if isinstance(value, dict):
            # Dict-based advisories (most common)
            symbol = str(value.get("symbol", ""))
            direction = self._infer_direction(value)
            confidence = float(value.get("confidence", 0.5) or 0.5)
            strength = float(value.get("strength", 0.5) or 0.5)

            # Special handling for status-based risk advisories
            if key == "system_status":
                status = str(value.get("status", "HEALTHY")).upper()
                if status == "CRITICAL":
                    direction = "reduce"
                    strength = 1.0
                    confidence = 1.0
                elif status == "DEGRADED":
                    direction = "reduce"
                    strength = 0.5
                    confidence = 0.8

            proposals.append(AdvisoryProposal(
                source=key, category=category, symbol=symbol,
                direction=direction, strength=strength,
                confidence=confidence, payload=value,
            ))

        elif isinstance(value, (int, float)):
            # Scalar advisories (risk_score, tca_score, fear_greed, etc.)
            proposals.append(AdvisoryProposal(
                source=key, category=category, symbol="",
                direction="hold", strength=abs(float(value)),
                confidence=0.5, payload={"value": value},
            ))

        elif isinstance(value, list):
            # List advisories (bleeders, bandit_rankings, liquidation_cascade signals)
            for item in value:
                if isinstance(item, dict):
                    symbol = str(item.get("symbol", item.get("name", "")))
                    proposals.append(AdvisoryProposal(
                        source=key, category=category, symbol=symbol,
                        direction=self._infer_direction(item),
                        strength=float(item.get("confidence", 0.5) or 0.5),
                        confidence=float(item.get("confidence", 0.5) or 0.5),
                        payload=item,
                    ))

        return proposals

    def _infer_direction(self, data: Dict[str, Any]) -> str:
        """Infer trade direction from advisory data."""
        direction = str(data.get("direction", data.get("action", "hold"))).lower()
        if direction in ("up", "bullish", "buy", "long", "long_spread"):
            return "buy"
        if direction in ("down", "bearish", "sell", "short", "short_spread"):
            return "sell"
        if direction in ("reduce", "close", "flatten"):
            return "reduce"
        return "hold"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_proposals": self._proposal_count,
            "blocked_keys": len(self._blocked),
            "last_process_ts": self._last_process_ts,
        }
