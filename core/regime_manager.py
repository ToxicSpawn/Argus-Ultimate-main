"""Regime manager — single source of truth for the current market regime.

Wires CrossAssetRegimeDetector → RegimeConsensusWeighter → AdaptiveATRStops
so every other module just calls RegimeManager.get() or RegimeManager.get_stops().
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from alpha.cross_asset_regime import CrossAssetRegimeDetector
from risk.adaptive_atr_stops import AdaptiveATRStops, StopLevels
from strategies.regime_consensus_weighter import RegimeConsensusWeighter

logger = logging.getLogger(__name__)


class RegimeManager:
    """Singleton-style facade that manages regime detection + consensus + ATR stops."""

    def __init__(
        self,
        model_names: List[str],
        detector_kwargs: Optional[dict] = None,
        consensus_kwargs: Optional[dict] = None,
        atr_kwargs: Optional[dict] = None,
        cache_ttl_s: float = 60.0,
    ) -> None:
        self._detector = CrossAssetRegimeDetector(**(detector_kwargs or {}))
        self._consensus = RegimeConsensusWeighter(model_names, **(consensus_kwargs or {}))
        self._atr_stops = AdaptiveATRStops(**(atr_kwargs or {}))
        self._cache_ttl = cache_ttl_s
        self._regime: str = "ranging"
        self._regime_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_regime(
        self,
        prices: Dict[str, pd.Series],
        primary: str = "BTC/USDT",
    ) -> str:
        """Detect and cache the current regime from cross-asset prices."""
        self._regime = self._detector.detect(prices, primary=primary)
        self._regime_ts = time.monotonic()
        logger.info("Regime updated: %s", self._regime)
        return self._regime

    def get(self) -> str:
        """Return the cached regime label."""
        return self._regime

    def is_stale(self) -> bool:
        return (time.monotonic() - self._regime_ts) > self._cache_ttl

    def get_weights(self, model_signals: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Return per-model weights for the current regime."""
        return self._consensus.get_weights(self._regime)

    def weighted_signal(self, model_signals: Dict[str, float]) -> float:
        """Return consensus signal [-1, 1] for the current regime."""
        return self._consensus.weighted_signal(self._regime, model_signals)

    def update_model_pnl(self, model_pnl: Dict[str, float]) -> None:
        """Record per-model realised PnL to update regime consensus weights."""
        self._consensus.update(self._regime, model_pnl)

    def compute_stops(
        self,
        ohlcv: pd.DataFrame,
        entry_price: float,
    ) -> StopLevels:
        """Compute adaptive ATR stops for the current regime."""
        return self._atr_stops.compute(ohlcv, entry_price, self._regime)
