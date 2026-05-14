"""Meta-learning adjustment computation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaLearningAdjustment:
    learning_rate_multiplier: float
    confidence_multiplier: float
    threshold_multiplier: float
    reason: str


def compute_meta_learning_adjustment(
    regime: str,
    signal_noise: float,
    recent_model_error_bps: float,
) -> MetaLearningAdjustment:
    """Compute meta-learning adjustments based on regime and model quality.

    - STRESSED regime -> faster learning (lr >= 1.0), tighter thresholds (>= 1.0)
    - High signal_noise -> lower confidence
    - High model error -> slower learning
    """
    regime_upper = regime.upper()

    # Base multipliers
    lr_mult = 1.0
    conf_mult = 1.0
    thresh_mult = 1.0

    reasons: list[str] = []

    # Regime adjustments
    if regime_upper == "STRESSED":
        lr_mult = 1.5
        thresh_mult = 1.3
        reasons.append(f"STRESSED: lr*1.5, thresh*1.3")
    elif regime_upper in ("HIGH_VOL", "CRISIS"):
        lr_mult = 1.3
        thresh_mult = 1.2
        reasons.append(f"{regime_upper}: lr*1.3, thresh*1.2")
    elif regime_upper == "TRENDING":
        lr_mult = 1.0
        thresh_mult = 0.9
        reasons.append("TRENDING: thresh*0.9")
    elif regime_upper == "RANGING":
        lr_mult = 0.8
        thresh_mult = 1.0
        reasons.append("RANGING: lr*0.8")

    # Noise adjustment — lower confidence when noisy
    if signal_noise > 0.5:
        noise_penalty = min(0.5, (signal_noise - 0.5) * 0.8)
        conf_mult = max(0.3, 1.0 - noise_penalty)
        reasons.append(f"noise={signal_noise:.2f}: conf*{conf_mult:.2f}")

    # Model error adjustment — high error slows learning
    if recent_model_error_bps > 10.0:
        error_penalty = min(0.5, (recent_model_error_bps - 10.0) / 100.0)
        lr_mult = max(0.5, lr_mult - error_penalty)
        reasons.append(f"error={recent_model_error_bps:.1f}bps: lr adjusted to {lr_mult:.2f}")

    reason = "; ".join(reasons) if reasons else "no adjustments"

    logger.debug("meta_learning: lr=%.2f conf=%.2f thresh=%.2f | %s",
                 lr_mult, conf_mult, thresh_mult, reason)

    return MetaLearningAdjustment(
        learning_rate_multiplier=lr_mult,
        confidence_multiplier=conf_mult,
        threshold_multiplier=thresh_mult,
        reason=reason,
    )
