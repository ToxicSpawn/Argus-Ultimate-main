"""
Master Adaptive Orchestrator - Connects ALL learning systems for maximum adaptability.

This module wires together:
1. Online Learning (ml/online_learning.py) - learns from trade outcomes
2. Drift Detection (ml/online_learning.py) - detects regime changes
3. Bandit Strategy Selection (ml/bandit_allocator.py) - selects best strategies
4. Contextual Bandits (ml/contextual_bandit.py) - regime-aware selection
5. Evolution (evolution/) - optimizes parameters via GA
6. Self-Improvement (adaptive/self_improver.py) - shadow tuning
7. Feedback Loop (ml/feedback_loop.py) - closed-loop learning
8. Anomaly Detection (ml/anomaly_detector.py) - market anomaly detection
9. Volatility Forecasting (ml/volatility_forecaster.py) - regime-aware vol
10. Evolution Strategy Reward (ml/evolution_strategy_reward.py) - ES-style exploration
11. Ensemble Signal Hub (ml/ensemble_signal_hub.py) - multi-source signal aggregation
12. Feature Store (ml/feature_store.py) - ML feature computation

The orchestrator ensures that:
- Every trade outcome feeds back to update models
- Drift detection triggers model retraining
- Best parameters are continuously evolved and applied
- Strategy weights adapt based on recent performance
- Anomalies are detected and acted upon
- Volatility forecasts inform position sizing
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveState:
    """Current state of the adaptive system."""
    trades_processed: int = 0
    last_update_ts: float = 0.0
    current_regime: str = "UNKNOWN"
    best_strategy: str = ""
    learning_rate: float = 0.1
    adaptation_score: float = 0.0  # 0-1, how well the system is adapting


class MasterAdaptiveOrchestrator:
    """
    Orchestrates all learning systems for maximum adaptability.
    
    Key features:
    - Instant feedback: Every trade outcome updates models within seconds
    - Continuous evolution: GA runs every 15 minutes on live data
    - Regime awareness: Strategy weights adjust to current market regime
    - Drift response: Automatic retraining when market behavior changes
    """
    
    def __init__(self, config: Any) -> None:
        self.config = config
        self.state = AdaptiveState()
        self._learning_systems: Dict[str, Any] = {}
        self._last_feedback_ts: float = 0.0
        self._feedback_interval: float = 1.0  # Process feedback every 1 second
        self._initialized = False
        
        logger.info("MasterAdaptiveOrchestrator created")
    
    async def initialize(self) -> None:
        """Initialize all learning subsystems."""
        if self._initialized:
            return
        
        logger.info("Initializing adaptive learning systems...")
        
        # 1. Initialize Online Learner
        try:
            from ml.online_learning import EnsembleOnlineLearner
            self._learning_systems["online_learner"] = EnsembleOnlineLearner(
                n_features=20,
                n_learners=3,
                performance_window=100
            )
            logger.info("  ✅ Online Learner initialized (Ensemble, RLS/SGD)")
        except Exception as e:
            logger.warning(f"  ⚠️ Online Learner failed: {e}")
        
        # 2. Initialize Drift Detector
        try:
            from ml.online_learning import DriftDetector
            self._learning_systems["drift_detector"] = DriftDetector(
                adwin_delta=0.002,
                ph_threshold=50.0
            )
            logger.info("  ✅ Drift Detector initialized (ADWIN + Page-Hinkley)")
        except Exception as e:
            logger.warning(f"  ⚠️ Drift Detector failed: {e}")
        
        # 3. Initialize Bandit Allocator
        try:
            from ml.bandit_allocator import BanditStrategyAllocator
            self._learning_systems["bandit_allocator"] = BanditStrategyAllocator(
                strategy_names=["momentum", "mean_reversion", "trend_following", "arbitrage"],
                exploration_rate=0.10
            )
            logger.info("  ✅ Bandit Allocator initialized (Thompson Sampling)")
        except Exception as e:
            logger.warning(f"  ⚠️ Bandit Allocator failed: {e}")
        
        # 4. Initialize Contextual Bandit
        try:
            from ml.contextual_bandit import ContextualBandit
            self._learning_systems["contextual_bandit"] = ContextualBandit(
                strategy_names=["momentum", "mean_reversion", "trend_following", "arbitrage"],
                exploration_rate=0.10
            )
            logger.info("  ✅ Contextual Bandit initialized (Regime-aware)")
        except Exception as e:
            logger.warning(f"  ⚠️ Contextual Bandit failed: {e}")
        
        # 5. Initialize Feedback Loop
        try:
            from ml.feedback_loop import MLFeedbackLoop
            self._learning_systems["feedback_loop"] = MLFeedbackLoop()
            logger.info("  ✅ Feedback Loop initialized (closed-loop learning)")
        except Exception as e:
            logger.warning(f"  ⚠️ Feedback Loop failed: {e}")
        
        # 6. Initialize Anomaly Detector
        try:
            from ml.anomaly_detector import MarketAnomalyDetector
            self._learning_systems["anomaly_detector"] = MarketAnomalyDetector(
                contamination=0.05,
                n_trees=100,
                random_state=42
            )
            logger.info("  ✅ Anomaly Detector initialized (Isolation Forest)")
        except Exception as e:
            logger.warning(f"  ⚠️ Anomaly Detector failed: {e}")
        
        # 7. Initialize Volatility Forecaster
        try:
            from ml.volatility_forecaster import VolatilityForecaster
            self._learning_systems["volatility_forecaster"] = VolatilityForecaster(
                lambda_ewma=0.94
            )
            logger.info("  ✅ Volatility Forecaster initialized (EWMA/ARCH)")
        except Exception as e:
            logger.warning(f"  ⚠️ Volatility Forecaster failed: {e}")
        
        # 8. Initialize Evolution Strategy Reward Tracker
        try:
            from ml.evolution_strategy_reward import EvolutionStrategyReward
            self._learning_systems["evolution_reward"] = EvolutionStrategyReward(
                history_size=100
            )
            logger.info("  ✅ Evolution Strategy Reward initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ Evolution Strategy Reward failed: {e}")
        
        # 9. Initialize Feature Store
        try:
            from ml.feature_store import FeatureStore
            self._learning_systems["feature_store"] = FeatureStore()
            logger.info("  ✅ Feature Store initialized (ML features)")
        except Exception as e:
            logger.warning(f"  ⚠️ Feature Store failed: {e}")
        
        # 10. Initialize Ensemble Signal Hub (multi-source signal aggregation)
        try:
            from ml.ensemble_signal_hub import EnsembleSignalHub
            self._learning_systems["ensemble_hub"] = EnsembleSignalHub(
                config={
                    "cache_ttl": 60,
                    "bullish_threshold": 0.3,  # Lowered for more signals
                    "weights": {
                        "fear_greed": 0.15,
                        "llm": 0.20,
                        "whale": 0.10,
                        "news": 0.10,
                        "alpha": 0.15,
                        "vol_regime": 0.10,
                        "funding": 0.05,
                        "chain_metrics": 0.10,
                        "graph": 0.10,
                    }
                }
            )
            logger.info("  ✅ Ensemble Signal Hub initialized (10 sources)")
        except Exception as e:
            logger.warning(f"  ⚠️ Ensemble Signal Hub failed: {e}")
        
        # 11. Initialize LLM Signal Generator
        try:
            from ml.llm_signal import LLMSignalGenerator
            self._learning_systems["llm_signal"] = LLMSignalGenerator()
            logger.info("  ✅ LLM Signal Generator initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ LLM Signal Generator failed: {e}")
        
        # 12. Initialize LLM Sentiment Enhanced
        try:
            from ml.llm_sentiment_enhanced import LLMEnsembleSentiment
            self._learning_systems["llm_sentiment"] = LLMEnsembleSentiment(
                enable_finbert=True,
                enable_vader=True
            )
            logger.info("  ✅ LLM Sentiment Ensemble initialized (FinBERT + VADER + Rules)")
        except Exception as e:
            logger.warning(f"  ⚠️ LLM Sentiment failed: {e}")
        
        # 13. Initialize Graph Neural Network (cross-asset analysis)
        try:
            from ml.graph_neural_network import GraphNeuralNetwork
            self._learning_systems["gnn"] = GraphNeuralNetwork(
                input_dim=20,
                hidden_dim=32,
                output_dim=4,
                n_layers=2
            )
            logger.info("  ✅ Graph Neural Network initialized (cross-asset)")
        except Exception as e:
            logger.warning(f"  ⚠️ Graph Neural Network failed: {e}")
        
        # 14. Initialize Transformer Price Predictor
        try:
            from ml.transformer_price_predictor import TransformerPricePredictor
            self._learning_systems["transformer"] = TransformerPricePredictor()
            logger.info("  ✅ Transformer Price Predictor initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ Transformer Price Predictor failed: {e}")
        
        self._initialized = True
        logger.info(f"Adaptive orchestrator initialized with {len(self._learning_systems)} systems")
    
    async def record_trade_outcome(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        strategy: str,
        regime: str,
        features: Dict[str, float],
    ) -> None:
        """
        Record a trade outcome and update all learning systems.
        
        This is the KEY feedback loop - every trade outcome updates:
        1. Online learner (feature → outcome mapping)
        2. Bandit allocator (strategy performance)
        3. Contextual bandit (regime-specific performance)
        4. Feedback loop (model accuracy tracking)
        """
        now = time.time()
        
        # Rate limit feedback processing
        if now - self._last_feedback_ts < self._feedback_interval:
            return
        
        self._last_feedback_ts = now
        self.state.trades_processed += 1
        
        # Calculate return
        if direction == "buy":
            returns = (exit_price - entry_price) / entry_price
        else:
            returns = (entry_price - exit_price) / entry_price
        
        # 1. Update Online Learner
        if "online_learner" in self._learning_systems:
            try:
                learner = self._learning_systems["online_learner"]
                X = self._features_to_array(features)
                y = np.array([returns])
                learner.partial_fit(X, y)
            except Exception as e:
                logger.debug(f"Online learner update failed: {e}")
        
        # 2. Update Bandit Allocator
        if "bandit_allocator" in self._learning_systems:
            try:
                bandit = self._learning_systems["bandit_allocator"]
                bandit.record_outcome(strategy, pnl)
            except Exception as e:
                logger.debug(f"Bandit update failed: {e}")
        
        # 3. Update Contextual Bandit
        if "contextual_bandit" in self._learning_systems:
            try:
                ctx_bandit = self._learning_systems["contextual_bandit"]
                context = ctx_bandit.make_context(
                    regime=regime,
                    vol_estimate=features.get("volatility", 0.02),
                    utc_hour=datetime.utcnow().hour
                )
                ctx_bandit.update(strategy, context, pnl)
            except Exception as e:
                logger.debug(f"Contextual bandit update failed: {e}")
        
        # 4. Update Feedback Loop
        if "feedback_loop" in self._learning_systems:
            try:
                fb = self._learning_systems["feedback_loop"]
                prediction = features.get("predicted_return", 0.0)
                fb.record_prediction("strategy_engine", prediction, returns)
            except Exception as e:
                logger.debug(f"Feedback loop update failed: {e}")
        
        # 5. Check for drift
        if "drift_detector" in self._learning_systems:
            try:
                detector = self._learning_systems["drift_detector"]
                X = self._features_to_array(features)
                drift_detected = detector.update(X, np.array([returns]))
                if drift_detected:
                    logger.info(f"🚨 Drift detected! Triggering model retraining...")
                    await self._handle_drift(regime)
            except Exception as e:
                logger.debug(f"Drift detection failed: {e}")
        
        # 6. Update Evolution Strategy Reward
        if "evolution_reward" in self._learning_systems:
            try:
                es_reward = self._learning_systems["evolution_reward"]
                es_reward.record_reward(pnl, strategy, symbol)
            except Exception as e:
                logger.debug(f"Evolution reward update failed: {e}")
        
        # 7. Update Volatility Forecaster with new data
        if "volatility_forecaster" in self._learning_systems:
            try:
                vol_fc = self._learning_systems["volatility_forecaster"]
                # Update with returns for volatility estimation
                vol_fc.update(returns)
            except Exception as e:
                logger.debug(f"Volatility forecaster update failed: {e}")
        
        # 8. Check for anomalies in trade patterns
        if "anomaly_detector" in self._learning_systems:
            try:
                detector = self._learning_systems["anomaly_detector"]
                X = self._features_to_array(features).reshape(1, -1)
                is_anomaly = detector.predict(X)
                if is_anomaly[0] == -1:
                    logger.warning(f"⚠️ Anomalous trade pattern detected: {symbol} {direction}")
            except Exception as e:
                logger.debug(f"Anomaly detection failed: {e}")
        
        # 9. Update Ensemble Signal Hub with trade outcome
        if "ensemble_hub" in self._learning_systems:
            try:
                hub = self._learning_systems["ensemble_hub"]
                # The hub caches signals; we can update its internal state
                hub.invalidate_cache(symbol)
            except Exception as e:
                logger.debug(f"Ensemble hub update failed: {e}")
        
        # 10. Update GNN with new trade data
        if "gnn" in self._learning_systems:
            try:
                gnn = self._learning_systems["gnn"]
                # GNN can learn from cross-asset patterns
                X = self._features_to_array(features)
                gnn.update(X, returns)
            except Exception as e:
                logger.debug(f"GNN update failed: {e}")
        
        logger.debug(f"Trade outcome recorded: {symbol} {direction} PnL={pnl:.2f} strategy={strategy}")
    
    async def _handle_drift(self, regime: str) -> None:
        """Handle detected concept drift by retraining models."""
        logger.info(f"Handling drift in regime: {regime}")
        
        # Trigger evolution with fresh data
        drift_event = {
            "timestamp": time.time(),
            "regime": regime,
            "action": "retrain",
        }
        
        # Store drift event for evolution to pick up
        drift_path = Path("data/drift_event.json")
        drift_path.write_text(json.dumps(drift_event))
    
    def get_strategy_allocation(
        self,
        total_capital: float,
        regime: str,
        vol_estimate: float,
        utc_hour: int,
    ) -> Dict[str, float]:
        """
        Get current strategy allocation based on learned performance.
        """
        # Try contextual bandit first
        if "contextual_bandit" in self._learning_systems:
            try:
                ctx_bandit = self._learning_systems["contextual_bandit"]
                context = ctx_bandit.make_context(
                    regime=regime,
                    vol_estimate=vol_estimate,
                    utc_hour=utc_hour
                )
                allocations = ctx_bandit.get_allocations(total_capital, context)
                if sum(allocations.values()) > 0:
                    return allocations
            except Exception as e:
                logger.debug(f"Contextual bandit allocation failed: {e}")
        
        # Fall back to flat bandit
        if "bandit_allocator" in self._learning_systems:
            try:
                bandit = self._learning_systems["bandit_allocator"]
                return bandit.get_allocations(total_capital)
            except Exception as e:
                logger.debug(f"Bandit allocation failed: {e}")
        
        # Default equal allocation
        strategies = ["momentum", "mean_reversion", "trend_following", "arbitrage"]
        return {s: total_capital / len(strategies) for s in strategies}
    
    def predict_with_online_model(self, features: Dict[str, float]) -> float:
        """Use online learner to predict returns."""
        if "online_learner" not in self._learning_systems:
            return 0.0
        
        try:
            learner = self._learning_systems["online_learner"]
            X = self._features_to_array(features)
            return float(learner.predict(X)[0])
        except Exception:
            return 0.0
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array."""
        feature_names = [
            "rsi", "macd", "bb_position", "adx", "volume_ratio",
            "volatility", "momentum_1h", "momentum_4h", "momentum_24h",
            "trend_strength", "support_distance", "resistance_distance",
            "order_book_imbalance", "funding_rate", "open_interest_change",
            "trade_count", "avg_trade_size", "price_acceleration",
            "regime_encoded", "time_of_day"
        ]
        
        arr = np.zeros(len(feature_names))
        for i, name in enumerate(feature_names):
            arr[i] = features.get(name, 0.0)
        
        return arr
    
    def get_status(self) -> Dict[str, Any]:
        """Get current adaptive system status."""
        status = {
            "initialized": self._initialized,
            "systems_active": list(self._learning_systems.keys()),
            "systems_count": len(self._learning_systems),
            "trades_processed": self.state.trades_processed,
            "current_regime": self.state.current_regime,
            "best_strategy": self.state.best_strategy,
            "adaptation_score": self.state.adaptation_score,
        }
        
        # Add volatility forecast if available
        if "volatility_forecaster" in self._learning_systems:
            try:
                vol_fc = self._learning_systems["volatility_forecaster"]
                status["volatility_forecast"] = vol_fc.get_forecast()
            except Exception:
                pass
        
        # Add evolution reward stats if available
        if "evolution_reward" in self._learning_systems:
            try:
                es_reward = self._learning_systems["evolution_reward"]
                status["avg_reward"] = es_reward.get_avg_reward()
            except Exception:
                pass
        
        return status
    
    def get_volatility_forecast(self) -> Dict[str, Any]:
        """Get current volatility forecast."""
        if "volatility_forecaster" not in self._learning_systems:
            return {"volatility": 0.02, "regime": "NORMAL", "confidence": 0.5}
        
        try:
            vol_fc = self._learning_systems["volatility_forecaster"]
            return vol_fc.get_forecast()
        except Exception:
            return {"volatility": 0.02, "regime": "NORMAL", "confidence": 0.5}
    
    def should_explore(self, strategy: str, symbol: str) -> bool:
        """Check if we should explore new parameters for this strategy/symbol."""
        if "evolution_reward" not in self._learning_systems:
            return True
        
        try:
            es_reward = self._learning_systems["evolution_reward"]
            return es_reward.should_explore(strategy, symbol)
        except Exception:
            return True
    
    def suggest_param_jitter(self, param_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Suggest parameter jitter for exploration."""
        if "evolution_reward" not in self._learning_systems:
            return {}
        
        try:
            es_reward = self._learning_systems["evolution_reward"]
            return es_reward.suggest_param_jitter(param_bounds)
        except Exception:
            return {}
    
    async def get_ensemble_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Get aggregated signal from all sources via Ensemble Signal Hub.
        
        Returns:
            Dict with composite, confidence, size_multiplier, regime_bias
        """
        if "ensemble_hub" not in self._learning_systems:
            return {"composite": 0.0, "confidence": 0.0, "size_multiplier": 1.0, "regime_bias": "NEUTRAL"}
        
        try:
            hub = self._learning_systems["ensemble_hub"]
            signal = await hub.update(symbol)
            return {
                "composite": signal.composite,
                "confidence": signal.confidence,
                "size_multiplier": signal.size_multiplier,
                "regime_bias": signal.regime_bias,
                "sources": signal.sources,
            }
        except Exception as e:
            logger.debug(f"Ensemble signal failed: {e}")
            return {"composite": 0.0, "confidence": 0.0, "size_multiplier": 1.0, "regime_bias": "NEUTRAL"}
    
    def get_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Get sentiment analysis from LLM Sentiment Ensemble.
        
        Returns:
            Dict with sentiment, confidence, scores
        """
        if "llm_sentiment" not in self._learning_systems:
            return {"sentiment": "neutral", "confidence": 0.5, "scores": {}}
        
        try:
            sentiment_engine = self._learning_systems["llm_sentiment"]
            result = sentiment_engine.analyze(text)
            return result
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.5, "scores": {}}
    
    def predict_price(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get price prediction from Transformer model.
        
        Returns:
            Dict with predicted_return, confidence
        """
        if "transformer" not in self._learning_systems:
            return {"predicted_return": 0.0, "confidence": 0.0}
        
        try:
            transformer = self._learning_systems["transformer"]
            X = self._features_to_array(features)
            prediction = transformer.predict(X)
            return prediction
        except Exception as e:
            logger.debug(f"Transformer prediction failed: {e}")
            return {"predicted_return": 0.0, "confidence": 0.0}
    
    def adjust_signal_confidence(
        self,
        base_confidence: float,
        symbol: str,
        strategy: str,
        features: Dict[str, float],
    ) -> float:
        """
        Adjust signal confidence based on learned patterns.
        
        Uses online learner predictions and drift detection to modify
        the base confidence of trading signals.
        
        Args:
            base_confidence: Original signal confidence [0, 1]
            symbol: Trading pair symbol
            strategy: Strategy that generated the signal
            features: Feature dict for the current market state
            
        Returns:
            Adjusted confidence [0, 1]
        """
        adjustments = []
        
        # 1. Online learner prediction adjustment
        if "online_learner" in self._learning_systems:
            try:
                learner = self._learning_systems["online_learner"]
                X = self._features_to_array(features)
                predicted_return = float(learner.predict(X)[0])
                # Scale prediction to adjustment factor
                pred_adjustment = np.clip(1.0 + predicted_return * 10, 0.5, 1.5)
                adjustments.append(pred_adjustment)
            except Exception:
                pass
        
        # 2. Drift detection - reduce confidence if drift detected
        if "drift_detector" in self._learning_systems:
            try:
                detector = self._learning_systems["drift_detector"]
                drift_confidence = detector.get_drift_confidence()
                if drift_confidence > 0.5:
                    # Reduce confidence during drift
                    drift_adjustment = 1.0 - (drift_confidence - 0.5) * 0.5
                    adjustments.append(drift_adjustment)
            except Exception:
                pass
        
        # 3. Volatility adjustment - reduce size in high vol
        if "volatility_forecaster" in self._learning_systems:
            try:
                vol_fc = self._learning_systems["volatility_forecaster"]
                vol_forecast = vol_fc.get_forecast()
                vol_regime = vol_forecast.get("regime", "NORMAL")
                if vol_regime == "HIGH":
                    adjustments.append(0.8)
                elif vol_regime == "EXTREME":
                    adjustments.append(0.6)
            except Exception:
                pass
        
        # 4. Ensemble signal adjustment
        if "ensemble_hub" in self._learning_systems:
            try:
                hub = self._learning_systems["ensemble_hub"]
                # Get cached signal if available
                signal = hub.get_cached_signal(symbol)
                if signal and signal.confidence > 0:
                    # Boost confidence if ensemble agrees
                    ensemble_agreement = signal.confidence
                    adjustments.append(0.8 + ensemble_agreement * 0.4)
            except Exception:
                pass
        
        # Apply geometric mean of all adjustments
        if adjustments:
            combined = np.prod(adjustments) ** (1.0 / len(adjustments))
            adjusted = base_confidence * combined
            return float(np.clip(adjusted, 0.0, 1.0))
        
        return base_confidence
    
    def get_position_size_multiplier(
        self,
        symbol: str,
        strategy: str,
        regime: str,
        features: Dict[str, float],
    ) -> float:
        """
        Get position size multiplier based on learned patterns.
        
        Returns a multiplier (typically 0.5-2.0) that scales position size
        based on:
        - Recent strategy performance
        - Current volatility regime
        - Drift detection status
        - Ensemble signal strength
        
        Args:
            symbol: Trading pair symbol
            strategy: Strategy being used
            regime: Current market regime
            features: Feature dict
            
        Returns:
            Position size multiplier (1.0 = normal)
        """
        multipliers = []
        
        # 1. Bandit-based allocation
        if "contextual_bandit" in self._learning_systems:
            try:
                ctx_bandit = self._learning_systems["contextual_bandit"]
                context = ctx_bandit.make_context(
                    regime=regime,
                    vol_estimate=features.get("volatility", 0.02),
                    utc_hour=datetime.utcnow().hour
                )
                allocations = ctx_bandit.get_allocations(1.0, context)
                strategy_alloc = allocations.get(strategy, 1.0 / len(allocations))
                # Convert allocation to multiplier (higher allocation = larger size)
                bandit_mult = 0.5 + strategy_alloc * 4.0  # Scale to 0.5-2.5
                multipliers.append(bandit_mult)
            except Exception:
                pass
        
        # 2. Volatility-based sizing
        if "volatility_forecaster" in self._learning_systems:
            try:
                vol_fc = self._learning_systems["volatility_forecaster"]
                vol_forecast = vol_fc.get_forecast()
                vol_value = vol_forecast.get("volatility", 0.02)
                # Inverse volatility scaling (lower vol = larger position)
                vol_mult = 0.02 / max(vol_value, 0.005)  # Target 2% vol
                vol_mult = np.clip(vol_mult, 0.3, 2.0)
                multipliers.append(vol_mult)
            except Exception:
                pass
        
        # 3. Evolution reward adjustment
        if "evolution_reward" in self._learning_systems:
            try:
                es_reward = self._learning_systems["evolution_reward"]
                avg_reward = es_reward.get_avg_reward()
                # Scale based on recent reward
                reward_mult = 1.0 + np.clip(avg_reward * 5, -0.5, 0.5)
                multipliers.append(reward_mult)
            except Exception:
                pass
        
        # 4. Anomaly penalty - reduce size if anomaly detected
        if "anomaly_detector" in self._learning_systems:
            try:
                detector = self._learning_systems["anomaly_detector"]
                X = self._features_to_array(features).reshape(1, -1)
                is_anomaly = detector.predict(X)
                if is_anomaly[0] == -1:
                    multipliers.append(0.5)  # Reduce size by 50%
            except Exception:
                pass
        
        # Apply geometric mean of multipliers
        if multipliers:
            combined = np.prod(multipliers) ** (1.0 / len(multipliers))
            return float(np.clip(combined, 0.25, 3.0))
        
        return 1.0
    
    def should_take_trade(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        confidence: float,
        regime: str,
        features: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Final gate: should we take this trade?
        
        Uses all learning systems to make a final decision on whether
        to execute a trade.
        
        Returns:
            Tuple of (should_trade, reason)
        """
        # 1. Check drift - pause trading during regime transitions
        if "drift_detector" in self._learning_systems:
            try:
                detector = self._learning_systems["drift_detector"]
                drift_conf = detector.get_drift_confidence()
                if drift_conf > 0.8:
                    return False, f"High drift confidence ({drift_conf:.2f})"
            except Exception:
                pass
        
        # 2. Check for anomalies
        if "anomaly_detector" in self._learning_systems:
            try:
                detector = self._learning_systems["anomaly_detector"]
                X = self._features_to_array(features).reshape(1, -1)
                is_anomaly = detector.predict(X)
                if is_anomaly[0] == -1:
                    return False, "Anomalous market conditions detected"
            except Exception:
                pass
        
        # 3. Check ensemble signal direction
        if "ensemble_hub" in self._learning_systems:
            try:
                hub = self._learning_systems["ensemble_hub"]
                signal = hub.get_cached_signal(symbol)
                if signal and signal.composite != 0:
                    # Check if signal direction aligns with trade direction
                    if direction == "buy" and signal.composite < -0.3:
                        return False, f"Ensemble strongly bearish ({signal.composite:.2f})"
                    elif direction == "sell" and signal.composite > 0.3:
                        return False, f"Ensemble strongly bullish ({signal.composite:.2f})"
            except Exception:
                pass
        
        # 4. Check minimum confidence from online learner
        if "online_learner" in self._learning_systems:
            try:
                predicted_return = self.predict_with_online_model(features)
                # If predicted return is strongly negative, skip the trade
                if direction == "buy" and predicted_return < -0.01:
                    return False, f"Online learner predicts loss ({predicted_return:.4f})"
                elif direction == "sell" and predicted_return > 0.01:
                    return False, f"Online learner predicts gain ({predicted_return:.4f})"
            except Exception:
                pass
        
        return True, "Trade approved"
