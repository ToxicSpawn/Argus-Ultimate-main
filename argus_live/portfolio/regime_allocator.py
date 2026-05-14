from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class MarketRegime(Enum):
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    STRESSED = "STRESSED"
    LIQUID = "LIQUID"


@dataclass(frozen=True)
class RegimeAllocationPolicy:
    regime: MarketRegime
    gross_exposure_multiplier: float
    rebalance_threshold_multiplier: float
    maker_bias: float
    family_weight_overrides: Dict[str, float]
    reason: str


_REGIME_DEFAULTS: Dict[MarketRegime, dict] = {
    MarketRegime.TRENDING: dict(
        gross_exposure_multiplier=1.0,
        rebalance_threshold_multiplier=1.0,
        maker_bias=0.5,
        family_weight_overrides={},
    ),
    MarketRegime.MEAN_REVERTING: dict(
        gross_exposure_multiplier=0.9,
        rebalance_threshold_multiplier=0.8,
        maker_bias=0.7,
        family_weight_overrides={},
    ),
    MarketRegime.HIGH_VOL: dict(
        gross_exposure_multiplier=0.7,
        rebalance_threshold_multiplier=1.3,
        maker_bias=0.3,
        family_weight_overrides={},
    ),
    MarketRegime.LOW_VOL: dict(
        gross_exposure_multiplier=1.1,
        rebalance_threshold_multiplier=0.9,
        maker_bias=0.8,
        family_weight_overrides={},
    ),
    MarketRegime.STRESSED: dict(
        gross_exposure_multiplier=0.5,
        rebalance_threshold_multiplier=1.5,
        maker_bias=0.2,
        family_weight_overrides={},
    ),
    MarketRegime.LIQUID: dict(
        gross_exposure_multiplier=1.0,
        rebalance_threshold_multiplier=0.7,
        maker_bias=0.6,
        family_weight_overrides={},
    ),
}


def build_regime_policy(regime: MarketRegime) -> RegimeAllocationPolicy:
    defaults = _REGIME_DEFAULTS[regime]
    return RegimeAllocationPolicy(
        regime=regime,
        gross_exposure_multiplier=defaults["gross_exposure_multiplier"],
        rebalance_threshold_multiplier=defaults["rebalance_threshold_multiplier"],
        maker_bias=defaults["maker_bias"],
        family_weight_overrides=defaults["family_weight_overrides"],
        reason=f"regime policy for {regime.value}: exposure={defaults['gross_exposure_multiplier']}",
    )
