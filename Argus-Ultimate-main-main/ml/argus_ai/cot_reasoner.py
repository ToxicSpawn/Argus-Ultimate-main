"""ChainOfThoughtReasoner — regime-aware structured reasoning for Argus-AI.

Produces a human-readable (and Redis-observable) reasoning chain:
  Step 1: Regime assessment
  Step 2: Volatility context
  Step 3: Spread / microstructure cost
  Step 4: Signal quality gate
  Step 5: Modality confidence check
  Step 6: Action decision

The scratchpad is stored to Redis under argus:ai:cot:<symbol> for
Grafana Loki ingestion and observability.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REGIME_NAMES = {0: "RANGING", 1: "TRENDING", 2: "VOLATILE", 3: "CRISIS"}

ACTION_NAMES = {0: "FLAT", 1: "LONG", 2: "SHORT"}


@dataclass
class CoTStep:
    step: int
    name: str
    observation: str
    conclusion: str
    passed: bool
    latency_us: float = 0.0


@dataclass
class CoTScratchpad:
    symbol: str
    timestamp: float
    regime: str
    steps: List[CoTStep] = field(default_factory=list)
    final_action: str = "FLAT"
    final_confidence: float = 0.0
    gated: bool = False
    gate_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


class ChainOfThoughtReasoner:
    """Regime-aware CoT reasoner for Argus-AI.

    Args:
        redis_client:        Optional redis.Redis instance for scratchpad publishing.
        vol_threshold:       Volatility above which VOLATILE logic applies.
        spread_threshold:    Spread above which signals are penalised.
        min_signal_quality:  Minimum signal quality score to proceed.
        min_confidence:      Minimum model confidence to emit a non-FLAT action.
        crisis_gate:         If True, always emit FLAT during CRISIS regime.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        vol_threshold: float = 0.03,
        spread_threshold: float = 0.0015,
        min_signal_quality: float = 0.5,
        min_confidence: float = 0.55,
        crisis_gate: bool = True,
    ) -> None:
        self.redis = redis_client
        self.vol_threshold = vol_threshold
        self.spread_threshold = spread_threshold
        self.min_signal_quality = min_signal_quality
        self.min_confidence = min_confidence
        self.crisis_gate = crisis_gate

    def reason(
        self,
        symbol: str,
        regime_id: int,
        volatility: float,
        spread: float,
        signal_quality: float,
        direction_probs: List[float],
        confidence_mean: float,
        confidence_std: float,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> CoTScratchpad:
        """Run 6-step reasoning chain and return structured scratchpad.

        Args:
            symbol:           Trading symbol, e.g. 'BTC/USDT'.
            regime_id:        Integer regime label [0-3].
            volatility:       Current volatility index.
            spread:           Current best spread.
            signal_quality:   Signal quality score in [0, 1].
            direction_probs:  [flat_prob, long_prob, short_prob].
            confidence_mean:  MC Dropout mean confidence.
            confidence_std:   MC Dropout uncertainty.
            modality_mask:    Dict[modality_name, available]. None = all available.

        Returns:
            Populated CoTScratchpad.
        """
        t0 = time.monotonic()
        regime_name = REGIME_NAMES.get(regime_id, "UNKNOWN")
        pad = CoTScratchpad(
            symbol=symbol,
            timestamp=time.time(),
            regime=regime_name,
        )

        # Step 1 — Regime
        t1 = time.monotonic()
        crisis = regime_id == 3
        pad.steps.append(CoTStep(
            step=1,
            name="regime_assessment",
            observation=f"Regime={regime_name} (id={regime_id})",
            conclusion="CRISIS detected — hard gate active" if crisis else f"Regime={regime_name} is tradeable",
            passed=not crisis or not self.crisis_gate,
            latency_us=(time.monotonic() - t1) * 1e6,
        ))
        if crisis and self.crisis_gate:
            pad.gated = True
            pad.gate_reason = "CRISIS regime"
            pad.final_action = "FLAT"
            self._publish(pad)
            return pad

        # Step 2 — Volatility
        t2 = time.monotonic()
        vol_ok = volatility <= self.vol_threshold * 3
        pad.steps.append(CoTStep(
            step=2,
            name="volatility_context",
            observation=f"vol={volatility:.5f}, threshold={self.vol_threshold:.5f}",
            conclusion="Volatility within tradeable bounds" if vol_ok else "Extreme volatility — reduce sizing",
            passed=vol_ok,
            latency_us=(time.monotonic() - t2) * 1e6,
        ))

        # Step 3 — Spread
        t3 = time.monotonic()
        spread_ok = spread <= self.spread_threshold * 4
        pad.steps.append(CoTStep(
            step=3,
            name="spread_microstructure",
            observation=f"spread={spread:.6f}, threshold={self.spread_threshold:.6f}",
            conclusion="Spread acceptable" if spread_ok else "Wide spread — signal penalised",
            passed=spread_ok,
            latency_us=(time.monotonic() - t3) * 1e6,
        ))

        # Step 4 — Signal Quality
        t4 = time.monotonic()
        sq_ok = signal_quality >= self.min_signal_quality
        pad.steps.append(CoTStep(
            step=4,
            name="signal_quality_gate",
            observation=f"signal_quality={signal_quality:.3f}, min={self.min_signal_quality:.3f}",
            conclusion="Signal quality sufficient" if sq_ok else "Signal quality too low — FLAT",
            passed=sq_ok,
            latency_us=(time.monotonic() - t4) * 1e6,
        ))
        if not sq_ok:
            pad.gated = True
            pad.gate_reason = f"signal_quality={signal_quality:.3f} < {self.min_signal_quality}"
            pad.final_action = "FLAT"
            self._publish(pad)
            return pad

        # Step 5 — Modality Confidence
        t5 = time.monotonic()
        missing = []
        if modality_mask:
            missing = [k for k, v in modality_mask.items() if not v]
        modal_ok = len(missing) <= 2
        pad.steps.append(CoTStep(
            step=5,
            name="modality_confidence",
            observation=f"missing_modalities={missing}, confidence_std={confidence_std:.4f}",
            conclusion="Modality coverage sufficient" if modal_ok else "Too many missing modalities — FLAT",
            passed=modal_ok,
            latency_us=(time.monotonic() - t5) * 1e6,
        ))
        if not modal_ok:
            pad.gated = True
            pad.gate_reason = f"missing modalities: {missing}"
            pad.final_action = "FLAT"
            self._publish(pad)
            return pad

        # Step 6 — Action Decision
        t6 = time.monotonic()
        conf_ok = confidence_mean >= self.min_confidence
        flat_p, long_p, short_p = direction_probs[0], direction_probs[1], direction_probs[2]
        if not conf_ok:
            action_name = "FLAT"
        elif long_p >= short_p and long_p > flat_p:
            action_name = "LONG"
        elif short_p > long_p and short_p > flat_p:
            action_name = "SHORT"
        else:
            action_name = "FLAT"
        pad.steps.append(CoTStep(
            step=6,
            name="action_decision",
            observation=(
                f"conf={confidence_mean:.3f}±{confidence_std:.3f}, "
                f"probs=[flat={flat_p:.3f}, long={long_p:.3f}, short={short_p:.3f}]"
            ),
            conclusion=f"Action={action_name} (confidence_ok={conf_ok})",
            passed=True,
            latency_us=(time.monotonic() - t6) * 1e6,
        ))
        pad.final_action = action_name
        pad.final_confidence = float(confidence_mean)
        pad.gated = False
        total_us = (time.monotonic() - t0) * 1e6
        logger.debug(
            "cot_complete symbol=%s action=%s conf=%.3f total_us=%.1f",
            symbol, action_name, confidence_mean, total_us,
        )
        self._publish(pad)
        return pad

    def _publish(self, pad: CoTScratchpad) -> None:
        if self.redis is None:
            return
        try:
            key = f"argus:ai:cot:{pad.symbol.replace('/', '_')}"
            self.redis.set(key, json.dumps(pad.to_dict()), ex=300)
        except Exception as exc:
            logger.warning("CoT Redis publish failed: %s", exc)
