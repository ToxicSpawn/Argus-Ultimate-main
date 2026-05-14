"""Typed parameter space for Argus Optuna hyperopt — Push 51."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ParamSpace:
    """Defines search bounds for each tunable Argus parameter."""

    # Signal Gateway
    gateway_confidence: Tuple[float, float] = (0.30, 0.90)

    # HMM regime scalars
    hmm_bull_scalar: Tuple[float, float] = (1.00, 1.80)
    hmm_bear_scalar: Tuple[float, float] = (0.30, 0.70)

    # Spread calibration
    spread_bps: Tuple[float, float] = (1.0, 20.0)

    # HMM refit cadence
    regime_refit_bars: Tuple[int, int] = (50, 300)

    def as_dict(self) -> Dict[str, Tuple]:
        return {
            "gateway_confidence": self.gateway_confidence,
            "hmm_bull_scalar": self.hmm_bull_scalar,
            "hmm_bear_scalar": self.hmm_bear_scalar,
            "spread_bps": self.spread_bps,
            "regime_refit_bars": self.regime_refit_bars,
        }

    def validate(self) -> bool:
        """Return True if all lower bounds are strictly less than upper bounds."""
        for name, (lo, hi) in self.as_dict().items():
            if lo >= hi:
                raise ValueError(f"ParamSpace: {name} lower bound {lo} >= upper {hi}")
        return True


# Default search space used by HyperoptRunner unless overridden
ARGUS_DEFAULT_PARAM_SPACE = ParamSpace()
