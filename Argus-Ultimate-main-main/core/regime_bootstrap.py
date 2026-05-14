"""Bootstrap helper: instantiate RegimeManager from config and inject into the
execution engine + risk layer at startup.

Call `bootstrap_regime_manager(config)` from unified_trading_system.py
(or main.py) once the config is loaded, then pass the returned instance
through to whatever modules need it.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REGIME_MANAGER: Optional[Any] = None


def bootstrap_regime_manager(config: Any) -> Any:
    """Create (or re-use) the global RegimeManager instance.

    Parameters
    ----------
    config : Argus config object — reads the following attributes:
        regime_model_names    : list[str]   (default: see below)
        regime_cache_ttl_s    : float       (default: 60)
        regime_lookback       : int         (default: 60)
        regime_vol_threshold  : float       (default: 0.025)
        regime_trend_threshold: float       (default: 0.015)
        atr_period            : int         (default: 14)
        atr_vol_lookback      : int         (default: 100)
    """
    global _REGIME_MANAGER  # noqa: PLW0603
    if _REGIME_MANAGER is not None:
        return _REGIME_MANAGER

    from core.regime_manager import RegimeManager

    model_names = list(
        getattr(config, "regime_model_names", None)
        or [
            "lstm",
            "transformer",
            "xgboost",
            "rl_ppo",
            "ensemble",
        ]
    )

    _REGIME_MANAGER = RegimeManager(
        model_names=model_names,
        detector_kwargs={
            "lookback": int(getattr(config, "regime_lookback", 60) or 60),
            "vol_threshold": float(getattr(config, "regime_vol_threshold", 0.025) or 0.025),
            "trend_threshold": float(getattr(config, "regime_trend_threshold", 0.015) or 0.015),
        },
        consensus_kwargs={
            "ewm_alpha": float(getattr(config, "regime_ewm_alpha", 0.05) or 0.05),
            "softmax_temp": float(getattr(config, "regime_softmax_temp", 1.0) or 1.0),
        },
        atr_kwargs={
            "atr_period": int(getattr(config, "atr_period", 14) or 14),
            "vol_lookback": int(getattr(config, "atr_vol_lookback", 100) or 100),
        },
        cache_ttl_s=float(getattr(config, "regime_cache_ttl_s", 60.0) or 60.0),
    )
    logger.info("RegimeManager bootstrapped with models: %s", model_names)
    return _REGIME_MANAGER


def get_regime_manager() -> Optional[Any]:
    """Return the singleton RegimeManager (None if not yet bootstrapped)."""
    return _REGIME_MANAGER
