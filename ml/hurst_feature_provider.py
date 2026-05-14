"""
Market Regime Analyzer — integrates HurstRegimeDetector and PermutationEntropy with RealTimeFeatureStore.

Provides hurst_exponent, hurst_regime, permutation_entropy, and entropy_regime features
for comprehensive regime-aware strategy selection.

Combined signals:
- Hurst: mean_reversion / momentum / avoid (persistence)
- Entropy: highly_predictable / normal / chaotic (complexity)

Together they provide:
- Trending + Predictable → Strong momentum signal
- Mean-reverting + Predictable → Strong mean reversion signal
- Any + Chaotic → Reduce position size (random noise)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """
    Provides Hurst exponent and Permutation Entropy features by wrapping
    HurstRegimeDetector and PermutationEntropy, then updating the RealTimeFeatureStore.
    
    Usage:
        analyzer = MarketRegimeAnalyzer(feature_store, symbols=["BTC/USD", "ETH/USD"])
        analyzer.update_price("BTC/USD", 50000.0)
        
        # Get individual signals
        hurst = analyzer.get_hurst("BTC/USD")
        entropy = analyzer.get_entropy("BTC/USD")
        
        # Get combined regime
        regime = analyzer.get_combined_regime("BTC/USD")
        scalar = analyzer.get_position_scalar("BTC/USD")
    """
    
    def __init__(
        self,
        feature_store: Any,
        symbols: Optional[List[str]] = None,
        window: int = 100,
        # Hurst parameters
        hurst_mean_reversion_threshold: float = 0.45,
        hurst_trending_threshold: float = 0.55,
        # Entropy parameters
        entropy_order: int = 3,
        entropy_delay: int = 1,
        entropy_low_threshold: float = 0.3,
        entropy_high_threshold: float = 0.8,
    ) -> None:
        self._feature_store = feature_store
        self._symbols = symbols or ["BTC/USD", "ETH/USD", "SOL/USD"]
        self._window = window
        
        # Import detectors
        from research.hurst_regime import HurstRegimeDetector
        from research.permutation_entropy import PermutationEntropy
        
        self._hurst_detector = HurstRegimeDetector(
            window=window,
            mean_reversion_threshold=hurst_mean_reversion_threshold,
            trending_threshold=hurst_trending_threshold,
        )
        
        self._entropy_detector = PermutationEntropy(
            window=window,
            order=entropy_order,
            delay=entropy_delay,
            low_threshold=entropy_low_threshold,
            high_threshold=entropy_high_threshold,
        )
        
        # Register features in the feature store
        self._register_features()
        
        logger.info(
            "MarketRegimeAnalyzer initialized: symbols=%s window=%d "
            "hurst_thresh=[%.2f, %.2f] entropy_order=%d entropy_thresh=[%.2f, %.2f]",
            self._symbols, window,
            hurst_mean_reversion_threshold, hurst_trending_threshold,
            entropy_order, entropy_low_threshold, entropy_high_threshold,
        )
    
    def _register_features(self) -> None:
        """Register all regime features in the feature store."""
        try:
            from ml.feature_store_realtime import FeatureConfig
            
            features = [
                # Hurst features
                FeatureConfig("hurst_exponent", "numeric", "derived", 300, 60),
                FeatureConfig("hurst_regime", "categorical", "derived", 300, 60),
                # Entropy features
                FeatureConfig("permutation_entropy", "numeric", "derived", 300, 60),
                FeatureConfig("entropy_regime", "categorical", "derived", 300, 60),
                FeatureConfig("predictability", "numeric", "derived", 300, 60),
                # Combined features
                FeatureConfig("regime_scalar", "numeric", "derived", 300, 60),
            ]
            
            for config in features:
                try:
                    self._feature_store.register_feature(config)
                except Exception as e:
                    logger.debug("Feature %s registration skipped: %s", config.feature_name, e)
            
            logger.info("Registered regime features")
        except Exception as exc:
            logger.warning("Failed to register regime features: %s", exc)
    
    def update_price(self, symbol: str, price: float) -> Dict[str, Optional[float]]:
        """
        Update both detectors with a new price and update the feature store.
        
        Returns dict with computed values, or None for each if insufficient data.
        """
        results = {
            "hurst": None,
            "entropy": None,
            "regime_scalar": None,
        }
        
        # Update Hurst detector
        hurst = self._hurst_detector.update(symbol, price)
        if hurst is not None:
            results["hurst"] = hurst
        
        # Update Entropy detector
        entropy = self._entropy_detector.update(symbol, price)
        if entropy is not None:
            results["entropy"] = entropy
        
        # Update feature store if we have data
        if hurst is not None or entropy is not None:
            self._update_feature_store(symbol, hurst, entropy)
            results["regime_scalar"] = self.get_position_scalar(symbol)
        
        return results
    
    def _update_feature_store(
        self,
        symbol: str,
        hurst: Optional[float],
        entropy: Optional[float],
    ) -> None:
        """Update the feature store with regime values."""
        try:
            # Hurst features
            if hurst is not None:
                self._feature_store.update_feature(symbol, "hurst_exponent", hurst)
                hurst_regime = self._classify_hurst_regime(hurst)
                hurst_code = self._hurst_regime_to_code(hurst_regime)
                self._feature_store.update_feature(symbol, "hurst_regime", hurst_code)
            
            # Entropy features
            if entropy is not None:
                self._feature_store.update_feature(symbol, "permutation_entropy", entropy)
                entropy_regime = self._classify_entropy_regime(entropy)
                entropy_code = self._entropy_regime_to_code(entropy_regime)
                self._feature_store.update_feature(symbol, "entropy_regime", entropy_code)
                predictability = 1.0 - entropy
                self._feature_store.update_feature(symbol, "predictability", predictability)
            
            # Combined scalar
            scalar = self.get_position_scalar(symbol)
            self._feature_store.update_feature(symbol, "regime_scalar", scalar)
            
            logger.debug(
                "Updated regime features for %s: hurst=%.4f entropy=%.4f scalar=%.2f",
                symbol,
                hurst if hurst is not None else 0.0,
                entropy if entropy is not None else 0.0,
                scalar,
            )
        except Exception as exc:
            logger.warning("Failed to update regime features for %s: %s", symbol, exc)
    
    def _classify_hurst_regime(self, hurst: float) -> str:
        """Classify Hurst exponent into regime string."""
        if hurst < self._hurst_detector.mean_reversion_threshold:
            return "mean_reversion"
        elif hurst > self._hurst_detector.trending_threshold:
            return "momentum"
        else:
            return "avoid"
    
    def _hurst_regime_to_code(self, regime: str) -> float:
        """Convert Hurst regime string to numeric code."""
        return {"mean_reversion": 0.0, "momentum": 1.0, "avoid": 2.0}.get(regime, 2.0)
    
    def _classify_entropy_regime(self, entropy: float) -> str:
        """Classify permutation entropy into regime string."""
        if entropy < self._entropy_detector.low_threshold:
            return "highly_predictable"
        elif entropy > self._entropy_detector.high_threshold:
            return "chaotic"
        else:
            return "normal"
    
    def _entropy_regime_to_code(self, regime: str) -> float:
        """Convert entropy regime string to numeric code."""
        return {"highly_predictable": 0.0, "normal": 1.0, "chaotic": 2.0}.get(regime, 1.0)
    
    # ------------------------------------------------------------------
    # Hurst queries
    # ------------------------------------------------------------------
    
    def get_hurst(self, symbol: str) -> float:
        """Get current Hurst exponent for symbol."""
        return self._hurst_detector.get_hurst(symbol)
    
    def get_hurst_regime(self, symbol: str) -> str:
        """Get current Hurst regime recommendation for symbol."""
        return self._hurst_detector.get_strategy_recommendation(symbol)
    
    # ------------------------------------------------------------------
    # Entropy queries
    # ------------------------------------------------------------------
    
    def get_entropy(self, symbol: str) -> float:
        """Get current permutation entropy for symbol."""
        return self._entropy_detector.get_entropy(symbol)
    
    def get_predictability(self, symbol: str) -> float:
        """Get current predictability score (1 - entropy) for symbol."""
        return self._entropy_detector.get_predictability(symbol)
    
    def get_entropy_regime(self, symbol: str) -> str:
        """Get current entropy regime recommendation for symbol."""
        return self._entropy_detector.get_recommendation(symbol)
    
    # ------------------------------------------------------------------
    # Combined regime
    # ------------------------------------------------------------------
    
    def get_combined_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Get combined regime analysis for symbol.
        
        Returns dict with:
        - hurst: float
        - hurst_regime: str
        - entropy: float
        - predictability: float
        - entropy_regime: str
        - combined_scalar: float (0-1.2)
        - should_trade: bool
        """
        hurst = self.get_hurst(symbol)
        hurst_regime = self.get_hurst_regime(symbol)
        entropy = self.get_entropy(symbol)
        predictability = self.get_predictability(symbol)
        entropy_regime = self.get_entropy_regime(symbol)
        scalar = self.get_position_scalar(symbol)
        should_trade = self.should_trade(symbol)
        
        return {
            "hurst": hurst,
            "hurst_regime": hurst_regime,
            "entropy": entropy,
            "predictability": predictability,
            "entropy_regime": entropy_regime,
            "combined_scalar": scalar,
            "should_trade": should_trade,
        }
    
    def get_regime_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get combined regime summary for all symbols."""
        summary = {}
        for symbol in self._hurst_detector.get_all_symbols():
            summary[symbol] = self.get_combined_regime(symbol)
        return summary
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols with price data."""
        return self._hurst_detector.get_all_symbols()
    
    # ------------------------------------------------------------------
    # Trading decisions
    # ------------------------------------------------------------------
    
    def should_trade(self, symbol: str) -> bool:
        """
        Check if trading is recommended based on combined regime.
        
        Returns False if:
        - Hurst says 'avoid' (random walk)
        - Entropy says 'chaotic' (too unpredictable)
        """
        hurst_regime = self.get_hurst_regime(symbol)
        entropy_regime = self.get_entropy_regime(symbol)
        
        # Don't trade in random walk
        if hurst_regime == "avoid":
            return False
        
        # Don't trade in chaotic markets
        if entropy_regime == "chaotic":
            return False
        
        return True
    
    def get_position_scalar(self, symbol: str) -> float:
        """
        Get combined position size scalar based on Hurst and Entropy.
        
        Logic:
        - Hurst avoid: 0.0 (no trading)
        - Entropy chaotic: 0.3 (high noise)
        - Hurst mean_reversion + Entropy highly_predictable: 1.2 (strong signal)
        - Hurst momentum + Entropy highly_predictable: 1.0 (good signal)
        - Hurst mean_reversion + Entropy normal: 0.9 (decent signal)
        - Hurst momentum + Entropy normal: 0.8 (standard trend following)
        - Default: 0.5
        """
        hurst_regime = self.get_hurst_regime(symbol)
        entropy_regime = self.get_entropy_regime(symbol)
        
        # Hurst says avoid → no trading
        if hurst_regime == "avoid":
            return 0.0
        
        # Entropy says chaotic → reduce significantly
        if entropy_regime == "chaotic":
            return 0.3
        
        # Combined matrix
        if hurst_regime == "mean_reversion":
            if entropy_regime == "highly_predictable":
                return 1.2  # Strong mean reversion signal
            else:  # normal
                return 0.9
        
        elif hurst_regime == "momentum":
            if entropy_regime == "highly_predictable":
                return 1.0  # Good momentum signal
            else:  # normal
                return 0.8  # Standard trend following
        
        return 0.5  # Default
    
    def get_trading_recommendation(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive trading recommendation.
        
        Returns dict with:
        - action: str ("trade", "reduce", "avoid")
        - scalar: float (position size multiplier)
        - strategy_hint: str (recommended strategy type)
        - confidence: float (0-1)
        """
        hurst_regime = self.get_hurst_regime(symbol)
        entropy_regime = self.get_entropy_regime(symbol)
        scalar = self.get_position_scalar(symbol)
        
        # Determine action
        if scalar == 0.0:
            action = "avoid"
        elif scalar < 0.5:
            action = "reduce"
        else:
            action = "trade"
        
        # Determine strategy hint
        if hurst_regime == "mean_reversion":
            strategy_hint = "mean_reversion"
        elif hurst_regime == "momentum":
            strategy_hint = "momentum"
        else:
            strategy_hint = "none"
        
        # Determine confidence
        if entropy_regime == "highly_predictable":
            confidence = 0.9
        elif entropy_regime == "normal":
            confidence = 0.6
        else:  # chaotic
            confidence = 0.2
        
        return {
            "action": action,
            "scalar": scalar,
            "strategy_hint": strategy_hint,
            "confidence": confidence,
            "hurst_regime": hurst_regime,
            "entropy_regime": entropy_regime,
        }
    
    def is_regime_change(self, symbol: str, lookback: int = 20) -> Dict[str, bool]:
        """Detect if a regime change occurred recently."""
        return {
            "hurst_change": False,  # HurstRegimeDetector doesn't track history changes
            "entropy_change": self._entropy_detector.is_regime_change(symbol, lookback),
        }
    
    def close(self) -> None:
        """Cleanup (no-op for now, kept for API compatibility)."""
        logger.info("MarketRegimeAnalyzer closed")


# Backward compatibility alias
HurstFeatureProvider = MarketRegimeAnalyzer


__all__ = ["MarketRegimeAnalyzer", "HurstFeatureProvider"]
