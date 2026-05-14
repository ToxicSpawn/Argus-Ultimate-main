# pyright: reportMissingImports=false
"""
ML Learning Integration
========================
Connects ML modules to the learning system for enhanced trading.

This module integrates:
1. Concept Drift Detector → Auto-reset learning on regime change
2. Meta Learner → Best model/strategy selection per regime
3. Online Stacking → ML-learned signal fusion
4. Transfer Learning → Cross-asset knowledge transfer

Architecture:
    Classical Learning System ←→ ML Enhancement Layer ←→ Trading Results
    
    Drift:      Basic threshold → ADWIN algorithm (auto-detect concept changes)
    Selection:  Fixed weights → Meta-learned (best model per regime)
    Fusion:     Simple averaging → Online stacking (learned optimal combination)
    Transfer:   Fresh start → Cross-asset knowledge (BTC → ETH)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque, defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MLLearningConfig:
    """Configuration for ML learning integration."""
    # Drift detection
    drift_window_size: int = 250
    drift_delta: float = 0.002
    drift_threshold: float = 0.1
    auto_reset_on_drift: bool = True
    
    # Meta learning
    meta_decay_hours: float = 168.0  # 1 week half-life
    meta_min_records: int = 3
    meta_db_path: str = "data/meta_learner.db"
    
    # Online stacking
    stacking_use_proba: bool = True
    stacking_add_features: bool = False
    
    # Transfer learning
    transfer_similarity_threshold: float = 0.7
    transfer_method: str = "domain_adaptation"
    
    # Enable/disable components
    enable_drift_detector: bool = True
    enable_meta_learner: bool = True
    enable_stacking: bool = True
    enable_transfer: bool = True


# ============================================================================
# ML Drift Detector (ADWIN-based)
# ============================================================================

class MLDetector:
    """
    ML-enhanced concept drift detection using ADWIN algorithm.
    
    Detects when market distribution changes significantly,
    triggering learning reset or model retraining.
    
    Much better than basic threshold because:
    - ADWIN adapts window size automatically
    - Detects gradual AND sudden drift
    - Provides drift magnitude estimate
    
    Usage:
        detector = MLDetector()
        
        # Check for drift every cycle
        if detector.check_drift(features, predictions, actual_returns):
            learning_orchestrator.reset_for_new_regime()
    """
    
    def __init__(self, config: Optional[MLLearningConfig] = None):
        self.config = config or MLLearningConfig()
        self._detector = None
        self._drift_history: deque = deque(maxlen=100)
        self._drift_count = 0
        self._last_drift_time: Optional[float] = None
        
        # Classical fallback ADWIN
        self._adwin_window: deque = deque(maxlen=self.config.drift_window_size)
        self._adwin_total: float = 0.0
        self._adwin_n: int = 0
        
        if self.config.enable_drift_detector:
            try:
                from ml.drift_detector import ConceptDriftDetector, ADWINState
                self._detector = ConceptDriftDetector(
                    model_name="learning_orchestrator",
                    feature_names=["volatility", "momentum", "trend", "volume_ratio"],
                    window_size=self.config.drift_window_size,
                    min_samples=30,
                    feature_drift_threshold=self.config.drift_threshold,
                    prediction_drift_threshold=0.2,
                )
                logger.info("ML Drift Detector: ADWIN algorithm enabled")
            except ImportError as e:
                logger.warning(f"ML Drift Detector: using fallback ADWIN: {e}")
    
    def check_drift(
        self,
        features: Dict[str, float],
        predictions: List[float],
        actuals: List[float]
    ) -> Dict[str, Any]:
        """
        Check for concept drift.
        
        Returns dict with:
        - drift_detected: bool
        - drift_type: "feature", "prediction", "concept", or "none"
        - magnitude: float 0.0 to 1.0
        - should_reset: bool (if drift is significant enough)
        """
        result = {
            "drift_detected": False,
            "drift_type": "none",
            "magnitude": 0.0,
            "should_reset": False,
            "adwin_drift_count": self._drift_count,
        }
        
        if self._detector is not None:
            # Use full ADWIN detector
            try:
                # Add samples
                for pred, actual in zip(predictions[-10:], actuals[-10:]):
                    error = abs(pred - actual)
                    self._detector.add_prediction(pred, error)
                
                # Check drift
                metrics = self._detector.compute_drift_metrics()
                
                if metrics.concept_drift_detected:
                    result["drift_detected"] = True
                    result["drift_type"] = "concept"
                    result["magnitude"] = min(metrics.drift_magnitude / 10.0, 1.0)
                    result["should_reset"] = metrics.drift_magnitude > 0.5
                    self._drift_count += 1
                    self._last_drift_time = time.time()
                    
            except Exception as e:
                logger.debug(f"Drift detector failed, using fallback: {e}")
                result = self._fallback_drift_check(features, predictions, actuals)
        else:
            # Use fallback ADWIN
            result = self._fallback_drift_check(features, predictions, actuals)
        
        self._drift_history.append(result)
        return result
    
    def _fallback_drift_check(
        self,
        features: Dict[str, float],
        predictions: List[float],
        actuals: List[float]
    ) -> Dict[str, Any]:
        """Fallback drift detection using simple ADWIN."""
        result = {
            "drift_detected": False,
            "drift_type": "none",
            "magnitude": 0.0,
            "should_reset": False,
            "adwin_drift_count": self._drift_count,
        }
        
        if len(predictions) < 10 or len(actuals) < 10:
            return result
        
        # Calculate recent errors
        recent_errors = [abs(p - a) for p, a in zip(predictions[-10:], actuals[-10:])]
        avg_error = np.mean(recent_errors)
        
        # Add to ADWIN window
        for error in recent_errors:
            self._adwin_window.append(error)
            self._adwin_total += error
            self._adwin_n += 1
        
        # Simple drift check: if error doubled recently
        if self._adwin_n >= 30:
            recent = list(self._adwin_window)[-10:]
            older = list(self._adwin_window)[-30:-10]
            
            if older:
                recent_mean = np.mean(recent)
                older_mean = np.mean(older)
                
                if older_mean > 0:
                    error_ratio = recent_mean / older_mean
                    
                    if error_ratio > 2.0:  # Error doubled
                        result["drift_detected"] = True
                        result["drift_type"] = "concept"
                        result["magnitude"] = min((error_ratio - 1.0), 1.0)
                        result["should_reset"] = error_ratio > 3.0
                        self._drift_count += 1
                        self._last_drift_time = time.time()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get drift detector statistics."""
        return {
            "detector_available": self._detector is not None,
            "drift_count": self._drift_count,
            "last_drift_time": self._last_drift_time,
            "recent_drifts": list(self._drift_history)[-5:] if self._drift_history else [],
            "window_size": self.config.drift_window_size,
        }


# ============================================================================
# ML Meta Learner (Best Model Selection)
# ============================================================================

class MLMetaLearner:
    """
    ML-enhanced meta-learning for best model/strategy selection.
    
    Learns which model/strategy performs best under each market regime
    by tracking historical performance with recency weighting.
    
    Better than fixed weights because:
    - Adapts to regime changes
    - Recency-weighted (recent performance matters more)
    - Feature-aware (considers volatility, trend, etc.)
    
    Usage:
        meta = MLMetaLearner()
        
        # Record performance
        meta.record_performance("momentum", "trending_up", {"vol": 0.02}, 0.72)
        meta.record_performance("mean_reversion", "trending_up", {"vol": 0.02}, 0.65)
        
        # Get best strategy for current regime
        best = meta.select_best("trending_up", {"vol": 0.02})
    """
    
    def __init__(self, config: Optional[MLLearningConfig] = None):
        self.config = config or MLLearningConfig()
        self._meta_learner = None
        self._performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self._selection_history: deque = deque(maxlen=100)
        
        if self.config.enable_meta_learner:
            try:
                from ml.meta_learner import MetaLearner
                import os
                os.makedirs("data", exist_ok=True)
                self._meta_learner = MetaLearner(
                    db_path=self.config.meta_db_path,
                    decay_half_life_hours=self.config.meta_decay_hours,
                    min_records=self.config.meta_min_records,
                )
                logger.info("ML Meta Learner: SQLite-backed model selection enabled")
            except ImportError as e:
                logger.warning(f"ML Meta Learner: using in-memory fallback: {e}")
    
    def record_performance(
        self,
        model_name: str,
        regime: str,
        features: Dict[str, float],
        performance: float
    ) -> None:
        """Record model performance for meta-learning."""
        # Always store in memory
        self._performance_history[model_name].append({
            "regime": regime,
            "features": features,
            "performance": performance,
            "timestamp": time.time()
        })
        
        # Also store in SQLite if available
        if self._meta_learner is not None:
            try:
                self._meta_learner.record_model_performance(
                    model_name=model_name,
                    regime=regime,
                    features=features,
                    accuracy=performance,
                )
            except Exception as e:
                logger.debug(f"Meta learner record failed: {e}")
    
    def select_best(
        self,
        regime: str,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Select best model/strategy for current conditions.
        
        Returns:
        - best_model: Name of best model
        - rankings: All models ranked by expected performance
        - confidence: Confidence in selection (0.0 to 1.0)
        """
        result = {
            "best_model": "",
            "rankings": [],
            "confidence": 0.0,
            "method": "fallback"
        }
        
        if self._meta_learner is not None:
            try:
                # Use SQLite-backed meta learner
                best = self._meta_learner.select_model(regime, features)
                rankings = self._meta_learner.get_model_rankings(regime)
                
                result["best_model"] = best
                result["rankings"] = rankings[:5]  # Top 5
                result["confidence"] = 0.8 if best else 0.0
                result["method"] = "meta_learner"
                
                self._selection_history.append(result)
                return result
                
            except Exception as e:
                logger.debug(f"Meta learner selection failed: {e}")
        
        # Fallback: use in-memory performance history
        model_scores = {}
        
        for model_name, history in self._performance_history.items():
            # Filter by regime
            regime_history = [h for h in history if h["regime"] == regime]
            
            if regime_history:
                # Recency-weighted average
                weights = []
                scores = []
                
                for i, record in enumerate(reversed(regime_history[-20:])):
                    age_weight = 0.9 ** i  # Exponential decay
                    weights.append(age_weight)
                    scores.append(record["performance"])
                
                if weights:
                    avg_score = np.average(scores, weights=weights)
                    model_scores[model_name] = avg_score
        
        if model_scores:
            rankings = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            result["best_model"] = rankings[0][0]
            result["rankings"] = rankings[:5]
            result["confidence"] = min(rankings[0][1], 1.0)
            result["method"] = "in_memory"
        
        self._selection_history.append(result)
        return result
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific model."""
        history = self._performance_history.get(model_name, [])
        
        if not history:
            return {"model": model_name, "records": 0}
        
        performances = [h["performance"] for h in history]
        regimes = set(h["regime"] for h in history)
        
        return {
            "model": model_name,
            "records": len(history),
            "avg_performance": float(np.mean(performances)),
            "best_performance": float(np.max(performances)),
            "regimes": list(regimes),
            "recent_performance": performances[-10:]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get meta learner statistics."""
        return {
            "meta_learner_available": self._meta_learner is not None,
            "tracked_models": len(self._performance_history),
            "total_records": sum(len(h) for h in self._performance_history.values()),
            "selections": len(self._selection_history),
            "models": list(self._performance_history.keys())
        }


# ============================================================================
# ML Signal Stacker (Ensemble Fusion)
# ============================================================================

class MLSignalStacker:
    """
    ML-enhanced signal stacking for ensemble fusion.
    
    Combines multiple strategy signals using learned weights
    instead of simple averaging.
    
    Better than simple averaging because:
    - Learns which strategies complement each other
    - Adapts weights based on recent performance
    - Handles strategy correlation
    
    Usage:
        stacker = MLSignalStacker()
        
        # Register strategies
        stacker.add_strategy("momentum", momentum_strategy)
        stacker.add_strategy("mean_reversion", mr_strategy)
        
        # Get stacked signal
        signal = stacker.stack_signals(prices, regime)
    """
    
    def __init__(self, config: Optional[MLLearningConfig] = None):
        self.config = config or MLLearningConfig()
        self._stacker = None
        self._strategies: Dict[str, Any] = {}
        self._strategy_weights: Dict[str, float] = {}
        self._signal_history: deque = deque(maxlen=100)
        
        if self.config.enable_stacking:
            try:
                from ml.online_stacking import OnlineStacker
                self._stacker = OnlineStacker(
                    use_proba=self.config.stacking_use_proba,
                    add_original_features=self.config.stacking_add_features,
                )
                logger.info("ML Signal Stacker: Online stacking enabled")
            except ImportError as e:
                logger.warning(f"ML Signal Stacker: using weighted averaging: {e}")
    
    def add_strategy(self, name: str, strategy: Any, initial_weight: float = 1.0) -> None:
        """Add a strategy to the stack."""
        self._strategies[name] = strategy
        self._strategy_weights[name] = initial_weight
    
    def stack_signals(
        self,
        signals: Dict[str, Dict[str, float]],
        regime: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Stack multiple strategy signals into one.
        
        Args:
            signals: Dict of {strategy_name: {"action": str, "confidence": float}}
            regime: Current market regime
            
        Returns:
        - action: "buy", "sell", or "hold"
        - confidence: 0.0 to 1.0
        - source_weights: Individual strategy contributions
        """
        if not signals:
            return {"action": "hold", "confidence": 0.0, "source_weights": {}}
        
        # Separate buy/sell votes
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        source_weights = {}
        
        for name, signal in signals.items():
            weight = self._strategy_weights.get(name, 1.0)
            confidence = signal.get("confidence", 0.0)
            action = signal.get("action", "hold")
            
            if action == "buy":
                buy_score += weight * confidence
            elif action == "sell":
                sell_score += weight * confidence
            
            total_weight += weight
            source_weights[name] = weight
        
        # Normalize
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final action
        if buy_score > sell_score and buy_score > 0.3:
            action = "buy"
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            action = "sell"
            confidence = sell_score
        else:
            action = "hold"
            confidence = 0.0
        
        result = {
            "action": action,
            "confidence": confidence,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "source_weights": source_weights,
            "n_strategies": len(signals),
            "regime": regime
        }
        
        self._signal_history.append(result)
        return result
    
    def update_weights(
        self,
        strategy_performance: Dict[str, float]
    ) -> None:
        """
        Update strategy weights based on recent performance.
        
        Args:
            strategy_performance: Dict of {strategy_name: recent_performance}
        """
        for name, perf in strategy_performance.items():
            if name in self._strategy_weights:
                # Softmax-like update
                old_weight = self._strategy_weights[name]
                new_weight = max(0.1, min(5.0, old_weight * (1.0 + perf * 0.1)))
                self._strategy_weights[name] = new_weight
        
        # Normalize weights
        total = sum(self._strategy_weights.values())
        if total > 0:
            for name in self._strategy_weights:
                self._strategy_weights[name] /= total
                self._strategy_weights[name] *= len(self._strategy_weights)  # Keep scale
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stacker statistics."""
        return {
            "stacker_available": self._stacker is not None,
            "n_strategies": len(self._strategies),
            "weights": dict(self._strategy_weights),
            "signals_stacked": len(self._signal_history),
            "recent_signals": list(self._signal_history)[-5:] if self._signal_history else []
        }


# ============================================================================
# ML Transfer Learner (Cross-Asset Knowledge)
# ============================================================================

class MLTransferLearner:
    """
    ML-enhanced transfer learning for cross-asset knowledge.
    
    Transfers learned patterns from one asset to another,
    enabling faster learning when trading new pairs.
    
    Better than starting fresh because:
    - BTC patterns often apply to ETH (similar crypto market)
    - Reduces cold start problem
    - Leverages domain similarity
    
    Usage:
        transfer = MLTransferLearner()
        
        # Learn from BTC
        transfer.register_asset("BTCUSDT", btc_features)
        
        # Transfer to ETH
        result = transfer.transfer_knowledge("BTCUSDT", "ETHUSDT", eth_features)
        if result["should_transfer"]:
            apply transferred_parameters to ETH strategy
    """
    
    def __init__(self, config: Optional[MLLearningConfig] = None):
        self.config = config or MLLearningConfig()
        self._transfer_analyzer = None
        self._asset_profiles: Dict[str, Dict[str, Any]] = {}
        self._transfer_history: deque = deque(maxlen=50)
        
        if self.config.enable_transfer:
            try:
                from ml.transfer_learning import (
                    TransferabilityAnalyzer,
                    AssetProfile,
                    TransferLearningConfig,
                    TransferMethod
                )
                self._transfer_analyzer = TransferabilityAnalyzer()
                self._transfer_method = TransferMethod.DOMAIN_ADAPTATION
                logger.info("ML Transfer Learner: Cross-asset transfer enabled")
            except ImportError as e:
                logger.warning(f"ML Transfer Learner: using fallback similarity: {e}")
    
    def register_asset(
        self,
        asset_id: str,
        features: Dict[str, float],
        asset_type: str = "crypto"
    ) -> None:
        """Register an asset for transfer learning."""
        self._asset_profiles[asset_id] = {
            "features": features,
            "asset_type": asset_type,
            "timestamp": time.time()
        }
        
        if self._transfer_analyzer is not None:
            try:
                from ml.transfer_learning import AssetProfile
                import numpy as np
                
                profile = AssetProfile(
                    asset_id=asset_id,
                    asset_type=asset_type,
                    volatility=features.get("volatility", 0.02),
                    volume_profile=np.array([features.get("volume_ratio", 1.0)]),
                    correlation_matrix=np.array([[1.0]]),
                    metadata=features
                )
                self._transfer_analyzer.register_asset(profile)
            except Exception as e:
                logger.debug(f"Transfer analyzer registration failed: {e}")
    
    def compute_similarity(self, source_id: str, target_id: str) -> float:
        """Compute similarity between two assets."""
        if source_id not in self._asset_profiles or target_id not in self._asset_profiles:
            return 0.0
        
        source_features = self._asset_profiles[source_id]["features"]
        target_features = self._asset_profiles[target_id]["features"]
        
        # Simple feature similarity
        common_keys = set(source_features.keys()) & set(target_features.keys())
        
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            s_val = source_features[key]
            t_val = target_features[key]
            
            # Normalize difference
            if max(abs(s_val), abs(t_val)) > 0:
                similarity = 1.0 - min(abs(s_val - t_val) / max(abs(s_val), abs(t_val)), 1.0)
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def transfer_knowledge(
        self,
        source_id: str,
        target_id: str,
        target_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Transfer learned knowledge from source to target asset.
        
        Returns:
        - should_transfer: bool
        - similarity: float 0.0 to 1.0
        - transferred_params: Dict of parameters to use
        - method: transfer method used
        """
        similarity = self.compute_similarity(source_id, target_id)
        should_transfer = similarity >= self.config.transfer_similarity_threshold
        
        result = {
            "should_transfer": should_transfer,
            "similarity": similarity,
            "source_asset": source_id,
            "target_asset": target_id,
            "transferred_params": {},
            "method": "none"
        }
        
        if should_transfer and source_id in self._asset_profiles:
            # Transfer proportional parameters
            source_features = self._asset_profiles[source_id]["features"]
            
            # Scale parameters by similarity
            for key, value in source_features.items():
                if key in target_features:
                    target_value = target_features[key]
                    # Weighted average by similarity
                    transferred = value * similarity + target_value * (1 - similarity)
                    result["transferred_params"][key] = transferred
            
            result["method"] = self.config.transfer_method
        
        self._transfer_history.append(result)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer learner statistics."""
        return {
            "transfer_available": self._transfer_analyzer is not None,
            "registered_assets": len(self._asset_profiles),
            "transfers": len(self._transfer_history),
            "recent_transfers": list(self._transfer_history)[-5:] if self._transfer_history else [],
            "similarity_threshold": self.config.transfer_similarity_threshold
        }


# ============================================================================
# ML Learning Manager
# ============================================================================

class MLLearningManager:
    """
    Manager that integrates all ML enhancements with the learning system.
    
    Connects:
    1. MLDetector → Drift detection (auto-reset on concept change)
    2. MLMetaLearner → Best model selection (per regime)
    3. MLSignalStacker → Ensemble fusion (learned weights)
    4. MLTransferLearner → Cross-asset knowledge
    
    Usage:
        manager = MLLearningManager()
        
        # Each cycle:
        drift = manager.check_drift(features, predictions, actuals)
        if drift["should_reset"]:
            learning_orchestrator.reset_for_new_regime()
        
        best = manager.select_best_strategy("trending_up", features)
        signal = manager.stack_signals(all_signals, regime)
        transfer = manager.transfer_knowledge("BTCUSDT", "ETHUSDT", features)
    """
    
    def __init__(self, config: Optional[MLLearningConfig] = None):
        self.config = config or MLLearningConfig()
        
        # Initialize ML components
        self.drift_detector = MLDetector(self.config)
        self.meta_learner = MLMetaLearner(self.config)
        self.signal_stacker = MLSignalStacker(self.config)
        self.transfer_learner = MLTransferLearner(self.config)
        
        # Integration state
        self._drift_resets: int = 0
        self._learning_cycle_count: int = 0
        self._last_drift_check: Optional[float] = None
        
        logger.info("=" * 60)
        logger.info("ML LEARNING MANAGER INITIALIZED")
        logger.info("=" * 60)
        logger.info("  1. Drift Detector: ADWIN algorithm (auto-reset)")
        logger.info("  2. Meta Learner: Best model per regime")
        logger.info("  3. Signal Stacking: ML-learned ensemble fusion")
        logger.info("  4. Transfer Learning: Cross-asset knowledge")
        logger.info("=" * 60)
    
    def check_drift(
        self,
        features: Dict[str, float],
        predictions: List[float],
        actuals: List[float]
    ) -> Dict[str, Any]:
        """Check for concept drift and auto-reset if needed."""
        self._learning_cycle_count += 1
        self._last_drift_check = time.time()
        
        result = self.drift_detector.check_drift(features, predictions, actuals)
        
        if result["should_reset"]:
            self._drift_resets += 1
            logger.warning(f"Concept drift detected (#{self._drift_resets}) - "
                          f"magnitude={result['magnitude']:.2f}, type={result['drift_type']}")
        
        return result
    
    def select_best_strategy(
        self,
        regime: str,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Select best strategy for current conditions."""
        return self.meta_learner.select_best(regime, features)
    
    def record_strategy_performance(
        self,
        strategy_name: str,
        regime: str,
        features: Dict[str, float],
        performance: float
    ) -> None:
        """Record strategy performance for meta-learning."""
        self.meta_learner.record_performance(strategy_name, regime, features, performance)
    
    def stack_signals(
        self,
        signals: Dict[str, Dict[str, float]],
        regime: str = "unknown"
    ) -> Dict[str, Any]:
        """Stack multiple strategy signals."""
        return self.signal_stacker.stack_signals(signals, regime)
    
    def update_strategy_weights(self, performance: Dict[str, float]) -> None:
        """Update strategy weights based on performance."""
        self.signal_stacker.update_weights(performance)
    
    def transfer_knowledge(
        self,
        source_asset: str,
        target_asset: str,
        target_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Transfer knowledge between assets."""
        return self.transfer_learner.transfer_knowledge(source_asset, target_asset, target_features)
    
    def register_asset(self, asset_id: str, features: Dict[str, float]) -> None:
        """Register asset for transfer learning."""
        self.transfer_learner.register_asset(asset_id, features)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive ML learning statistics."""
        return {
            "learning_cycles": self._learning_cycle_count,
            "drift_resets": self._drift_resets,
            "last_drift_check": self._last_drift_check,
            "drift_detector": self.drift_detector.get_stats(),
            "meta_learner": self.meta_learner.get_stats(),
            "signal_stacker": self.signal_stacker.get_stats(),
            "transfer_learner": self.transfer_learner.get_stats()
        }
    
    def log_stats(self) -> None:
        """Log ML learning statistics."""
        stats = self.get_stats()
        
        logger.info("ML LEARNING STATS:")
        logger.info(f"  Cycles: {stats['learning_cycles']}, Drift Resets: {stats['drift_resets']}")
        logger.info(f"  Tracked Models: {stats['meta_learner']['tracked_models']}")
        logger.info(f"  Strategies Stacked: {stats['signal_stacker']['n_strategies']}")
        logger.info(f"  Assets Registered: {stats['transfer_learner']['registered_assets']}")


# ============================================================================
# Global Instance
# ============================================================================

_global_ml_manager: Optional[MLLearningManager] = None


def get_ml_learning_manager(config: Optional[MLLearningConfig] = None) -> MLLearningManager:
    """Get or create the global ML learning manager."""
    global _global_ml_manager
    if _global_ml_manager is None:
        _global_ml_manager = MLLearningManager(config)
    return _global_ml_manager


def wire_ml_learning(config: Optional[MLLearningConfig] = None) -> MLLearningManager:
    """
    Wire ML modules to the learning system.
    
    This is the main entry point for ML-enhanced learning.
    """
    manager = get_ml_learning_manager(config)
    
    logger.info("=" * 60)
    logger.info("ML LEARNING WIRED TO LEARNING SYSTEM")
    logger.info("=" * 60)
    logger.info("  1. ADWIN Drift Detection → Auto-reset on concept change")
    logger.info("  2. Meta Learner → Best model per regime")
    logger.info("  3. Online Stacking → ML-learned signal fusion")
    logger.info("  4. Transfer Learning → Cross-asset knowledge")
    logger.info("=" * 60)
    
    return manager


def reset_ml_learning() -> None:
    """Reset the global ML learning manager (for testing)."""
    global _global_ml_manager
    _global_ml_manager = None


__all__ = [
    "MLLearningConfig",
    "MLDetector",
    "MLMetaLearner",
    "MLSignalStacker",
    "MLTransferLearner",
    "MLLearningManager",
    "get_ml_learning_manager",
    "wire_ml_learning",
    "reset_ml_learning",
]
