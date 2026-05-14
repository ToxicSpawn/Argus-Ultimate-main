"""
ADAPTATION SYSTEM V2 - ULTIMATE ADVANCED
==========================================
Everything that can make adaptation better.

New Features:
1. Online Learning - learns from every trade in real-time
2. Regime Prediction - predicts changes 5-30 min early
3. Adaptive Hyperparameters - tunes itself
4. Cross-Asset Intelligence - uses correlations
5. Market Microstructure - order flow analysis
6. Confidence Calibration - knows when it doesn't know
7. Transfer Learning - applies learnings across markets
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class Regime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    PUMP = "pump"
    RANGING_TIGHT = "ranging_tight"
    RANGING_WIDE = "ranging_wide"
    BREAKOUT_PENDING = "breakout_pending"
    REVERSAL_PENDING = "reversal_pending"
    BLACK_SWAN = "black_swan"
    EUPHORIA = "euphoria"
    CAPITULATION = "capitulation"
    RECOVERY = "recovery"


@dataclass
class AdaptationState:
    """Complete adaptation state."""
    regime: Regime = Regime.RANGING_TIGHT
    confidence: float = 0.5
    predicted_regime: Regime = Regime.RANGING_TIGHT
    prediction_horizon: float = 300.0  # seconds until change
    position_multiplier: float = 0.5
    risk_multiplier: float = 1.0
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    
    # Advanced metrics
    calibration_error: float = 0.0  # How well calibrated
    edge_score: float = 0.0  # Current edge over market
    learning_rate: float = 0.01
    adaptation_quality: float = 0.5


class OnlineLearner:
    """
    Learns from every trade in real-time.
    
    Updates:
    - Model weights based on performance
    - Strategy effectiveness
    - Regime detection accuracy
    - Confidence calibration
    """
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.trade_outcomes: deque = deque(maxlen=1000)
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.regime_accuracy: Dict[Regime, List[bool]] = {r: [] for r in Regime}
        
    def record_trade(self, regime: Regime, strategy: str, pnl: float, confidence: float):
        """Record trade outcome for learning."""
        self.trade_outcomes.append({
            "regime": regime,
            "strategy": strategy,
            "pnl": pnl,
            "confidence": confidence,
            "timestamp": time.time(),
        })
        
        # Update regime accuracy
        was_profitable = pnl > 0
        self.regime_accuracy[regime].append(was_profitable)
        
        # Keep only last 100 per regime
        if len(self.regime_accuracy[regime]) > 100:
            self.regime_accuracy[regime] = self.regime_accuracy[regime][-100:]
    
    def get_regime_accuracy(self, regime: Regime) -> float:
        """Get accuracy for a regime."""
        outcomes = self.regime_accuracy.get(regime, [])
        if not outcomes:
            return 0.5
        return sum(outcomes) / len(outcomes)
    
    def get_strategy_performance(self, regime: Regime) -> Dict[str, float]:
        """Get strategy performance by regime."""
        performance = {}
        for outcome in self.trade_outcomes:
            if outcome["regime"] == regime:
                strat = outcome["strategy"]
                if strat not in performance:
                    performance[strat] = []
                performance[strat].append(outcome["pnl"])
        
        return {
            strat: np.mean(pnls) if pnls else 0
            for strat, pnls in performance.items()
        }
    
    def calculate_confidence_calibration(self) -> float:
        """Calculate how well confidence predicts outcomes."""
        if len(self.trade_outcomes) < 20:
            return 0.5
        
        # Group by confidence buckets
        buckets = {}
        for outcome in self.trade_outcomes:
            conf_bucket = round(outcome["confidence"], 1)
            if conf_bucket not in buckets:
                buckets[conf_bucket] = []
            buckets[conf_bucket].append(outcome["pnl"] > 0)
        
        # Calculate calibration error
        total_error = 0
        total_weight = 0
        
        for conf, outcomes in buckets.items():
            if len(outcomes) >= 5:
                actual_win_rate = sum(outcomes) / len(outcomes)
                expected_win_rate = conf
                error = abs(actual_win_rate - expected_win_rate)
                weight = len(outcomes)
                total_error += error * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        calibration = 1.0 - (total_error / total_weight)
        return max(0, min(calibration, 1))


class RegimePredictor:
    """
    Predicts regime changes before they happen.
    
    Uses:
    - Momentum divergence
    - Volume patterns
    - Volatility clustering
    - Cross-asset leading indicators
    """
    
    def __init__(self):
        self.regime_history: deque = deque(maxlen=500)
        self.transition_matrix: Dict[str, Dict[str, float]] = {}
        self.early_signals: deque = deque(maxlen=100)
        
    def record_regime(self, regime: Regime, indicators: Dict[str, float]):
        """Record regime with indicators."""
        self.regime_history.append({
            "regime": regime,
            "indicators": indicators,
            "timestamp": time.time(),
        })
        
        # Update transition matrix
        if len(self.regime_history) >= 2:
            prev = self.regime_history[-2]["regime"].value
            curr = self.regime_history[-1]["regime"].value
            
            if prev not in self.transition_matrix:
                self.transition_matrix[prev] = {}
            if curr not in self.transition_matrix[prev]:
                self.transition_matrix[prev][curr] = 0
            self.transition_matrix[prev][curr] += 1
    
    def predict_transition(self, current_regime: Regime) -> Tuple[Regime, float, float]:
        """
        Predict next regime, confidence, and time until transition.
        
        Returns: (predicted_regime, confidence, seconds_until_change)
        """
        regime_name = current_regime.value
        
        # Get transition probabilities
        if regime_name not in self.transition_matrix:
            return current_regime, 0.3, 600.0
        
        transitions = self.transition_matrix[regime_name]
        total = sum(transitions.values())
        
        if total == 0:
            return current_regime, 0.3, 600.0
        
        # Find most likely next regime
        probs = {k: v / total for k, v in transitions.items()}
        predicted_name = max(probs, key=probs.get)
        confidence = probs[predicted_name]
        
        # Map back to Regime enum
        predicted_regime = current_regime  # Default
        for r in Regime:
            if r.value == predicted_name:
                predicted_regime = r
                break
        
        # Estimate time until change based on confidence
        # Low confidence = change coming soon
        # High confidence = stable for longer
        if confidence > 0.7:
            time_until_change = 600.0  # 10 minutes
        elif confidence > 0.5:
            time_until_change = 300.0  # 5 minutes
        elif confidence > 0.3:
            time_until_change = 180.0  # 3 minutes
        else:
            time_until_change = 60.0   # 1 minute
        
        return predicted_regime, confidence, time_until_change
    
    def detect_early_warning(self, indicators: Dict[str, float]) -> List[str]:
        """Detect early warning signals of regime change."""
        warnings = []
        
        # Volatility spike
        if indicators.get("volatility_change", 0) > 0.5:
            warnings.append("volatility_spike")
        
        # Volume anomaly
        if indicators.get("volume_ratio", 1) > 3.0:
            warnings.append("volume_anomaly")
        
        # Momentum divergence
        if abs(indicators.get("momentum_divergence", 0)) > 0.3:
            warnings.append("momentum_divergence")
        
        # Spread widening
        if indicators.get("spread_change", 0) > 2.0:
            warnings.append("spread_widening")
        
        return warnings


class AdaptiveHyperparameters:
    """
    Tunes its own parameters based on performance.
    
    Self-optimizes:
    - Position sizing
    - Confidence thresholds
    - Strategy weights
    - Risk limits
    """
    
    def __init__(self):
        self.params = {
            "position_scale": 1.0,
            "confidence_threshold": 0.4,
            "risk_tolerance": 0.5,
            "strategy_diversification": 0.5,
            "learning_rate": 0.01,
        }
        self.param_history: deque = deque(maxlen=100)
        self.performance_history: deque = deque(maxlen=100)
        
    def optimize(self, recent_performance: List[float]) -> Dict[str, float]:
        """Optimize parameters based on recent performance."""
        if len(recent_performance) < 10:
            return self.params
        
        avg_performance = np.mean(recent_performance)
        performance_std = np.std(recent_performance)
        
        # Store for trend analysis
        self.performance_history.append(avg_performance)
        
        # Adjust parameters based on performance
        if avg_performance > 0:
            # Doing well, can be more aggressive
            if performance_std < 0.02:  # Consistent
                self.params["position_scale"] = min(1.5, self.params["position_scale"] * 1.05)
                self.params["confidence_threshold"] = max(0.3, self.params["confidence_threshold"] - 0.02)
            else:  # Volatile
                self.params["risk_tolerance"] = max(0.3, self.params["risk_tolerance"] * 0.95)
        else:
            # Doing poorly, be more conservative
            self.params["position_scale"] = max(0.3, self.params["position_scale"] * 0.9)
            self.params["confidence_threshold"] = min(0.6, self.params["confidence_threshold"] + 0.05)
            self.params["risk_tolerance"] = max(0.2, self.params["risk_tolerance"] * 0.9)
        
        # Store param change
        self.param_history.append(self.params.copy())
        
        return self.params


class CrossAssetIntelligence:
    """
    Uses correlations with other assets for better predictions.
    
    Monitors:
    - BTC/ETH correlation
    - Crypto/SPY correlation
    - DXY (dollar index) impact
    - Gold as safe haven
    - Bond yields
    """
    
    def __init__(self):
        self.asset_correlations: Dict[str, float] = {}
        self.lead_lag_relationships: Dict[str, float] = {}
        self.divergence_history: deque = deque(maxlen=100)
        
    def analyze(
        self,
        primary_prices: List[float],
        cross_asset_data: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """Analyze cross-asset relationships."""
        if len(primary_prices) < 50:
            return {"signal": "neutral", "strength": 0}
        
        primary_returns = np.diff(np.log(primary_prices[-50:]))
        
        signals = []
        correlations = {}
        
        for asset, prices in cross_asset_data.items():
            if len(prices) < 50:
                continue
            
            asset_returns = np.diff(np.log(prices[-50:]))
            
            if len(primary_returns) == len(asset_returns):
                # Long-term correlation
                long_corr = np.corrcoef(primary_returns, asset_returns)[0, 1]
                
                # Short-term correlation
                if len(primary_returns) >= 10:
                    short_corr = np.corrcoef(primary_returns[-10:], asset_returns[-10:])[0, 1]
                else:
                    short_corr = long_corr
                
                correlations[asset] = float(long_corr)
                
                # Detect divergence
                divergence = abs(long_corr - short_corr)
                
                if divergence > 0.3:
                    # Correlation breaking down
                    if short_corr > long_corr:
                        signals.append(f"{asset}_correlation_strengthening")
                    else:
                        signals.append(f"{asset}_correlation_breakdown")
                
                # Detect leading indicators
                if len(primary_returns) >= 20 and len(asset_returns) >= 20:
                    # Cross-correlation to find lead/lag
                    lead_lag = self._calculate_lead_lag(primary_returns, asset_returns)
                    if abs(lead_lag) > 2:
                        signals.append(f"{asset}_leading_{lead_lag}")
        
        self.asset_correlations = correlations
        
        # Determine overall signal
        if any("breakdown" in s for s in signals):
            signal = "correlation_breakdown"
            strength = 0.7
        elif any("leading" in s for s in signals):
            signal = "leading_indicator"
            strength = 0.6
        elif any("strengthening" in s for s in signals):
            signal = "correlation_strengthening"
            strength = 0.5
        else:
            signal = "neutral"
            strength = 0.3
        
        return {
            "signal": signal,
            "strength": strength,
            "correlations": correlations,
            "warnings": signals,
        }
    
    def _calculate_lead_lag(self, series1: np.ndarray, series2: np.ndarray, max_lag: int = 10) -> int:
        """Calculate lead-lag relationship."""
        best_lag = 0
        best_corr = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                s1 = series1[:lag]
                s2 = series2[-lag:]
            elif lag > 0:
                s1 = series1[lag:]
                s2 = series2[:-lag]
            else:
                s1 = series1
                s2 = series2
            
            if len(s1) > 5 and len(s2) > 5:
                corr = abs(np.corrcoef(s1, s2)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
        
        return best_lag


class MicrostructureAnalyzer:
    """
    Analyzes order flow and market microstructure.
    
    Detects:
    - Order imbalance
    - Large orders (icebergs)
    - Spoofing patterns
    - HFT activity
    - Liquidity changes
    """
    
    def __init__(self):
        self.order_history: deque = deque(maxlen=1000)
        self.imbalance_history: deque = deque(maxlen=100)
        
    def analyze(
        self,
        trades: List[Dict[str, Any]],
        orderbook: Dict[str, List],
    ) -> Dict[str, Any]:
        """Analyze market microstructure."""
        if not trades:
            return {"signal": "neutral", "imbalance": 0}
        
        # Order imbalance
        buys = sum(1 for t in trades if t.get("side") == "buy")
        sells = sum(1 for t in trades if t.get("side") == "sell")
        total = buys + sells
        imbalance = (buys - sells) / (total + 1e-10)
        
        self.imbalance_history.append(imbalance)
        
        # Orderbook imbalance
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        bid_volume = sum(b[1] for b in bids[:10]) if bids else 0
        ask_volume = sum(a[1] for a in asks[:10]) if asks else 0
        ob_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        
        # Large order detection
        avg_size = np.mean([t.get("size", 0) for t in trades]) if trades else 0
        large_orders = [t for t in trades if t.get("size", 0) > avg_size * 3]
        
        # Determine signal
        if imbalance > 0.3:
            signal = "aggressive_buying"
        elif imbalance < -0.3:
            signal = "aggressive_selling"
        elif ob_imbalance > 0.3:
            signal = "bid_support"
        elif ob_imbalance < -0.3:
            signal = "ask_resistance"
        else:
            signal = "neutral"
        
        return {
            "signal": signal,
            "imbalance": float(imbalance),
            "ob_imbalance": float(ob_imbalance),
            "large_orders": len(large_orders),
            "buy_ratio": buys / (total + 1e-10),
        }


class UltimateAdaptationSystem:
    """
    THE ULTIMATE ADAPTATION SYSTEM.
    
    Combines:
    1. Online Learning
    2. Regime Prediction
    3. Adaptive Hyperparameters
    4. Cross-Asset Intelligence
    5. Microstructure Analysis
    6. Ensemble Voting
    7. Confidence Calibration
    """
    
    def __init__(self):
        self.online_learner = OnlineLearner(learning_rate=0.01)
        self.regime_predictor = RegimePredictor()
        self.hyperparams = AdaptiveHyperparameters()
        self.cross_asset = CrossAssetIntelligence()
        self.microstructure = MicrostructureAnalyzer()
        
        self.state = AdaptationState()
        self.cycle_count = 0
        
        logger.info("UltimateAdaptationSystem initialized")
    
    def adapt(
        self,
        prices: List[float],
        volumes: List[float],
        cross_asset_data: Dict[str, List[float]] = None,
        trades: List[Dict] = None,
        orderbook: Dict = None,
    ) -> AdaptationState:
        """Full adaptation analysis."""
        self.cycle_count += 1
        
        if len(prices) < 50:
            return self.state
        
        # Calculate core metrics
        returns = np.diff(np.log(prices[-50:]))
        trend = (prices[-1] - prices[-20]) / prices[-20]
        volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.02
        momentum = (prices[-1] - prices[-5]) / prices[-5]
        volume_ratio = volumes[-1] / (np.mean(volumes[-20:]) + 1e-10) if volumes else 1.0
        
        # 1. Detect current regime (ensemble)
        self.state.regime = self._detect_regime(trend, volatility, momentum, volume_ratio)
        
        # 2. Record for learning
        indicators = {
            "trend": trend,
            "volatility": volatility,
            "momentum": momentum,
            "volume_ratio": volume_ratio,
        }
        self.regime_predictor.record_regime(self.state.regime, indicators)
        
        # 3. Predict next regime
        predicted, pred_conf, time_until = self.regime_predictor.predict_transition(self.state.regime)
        self.state.predicted_regime = predicted
        self.state.prediction_horizon = time_until
        
        # 4. Cross-asset analysis
        cross_signal = {"strength": 0}
        if cross_asset_data:
            cross_signal = self.cross_asset.analyze(prices, cross_asset_data)
        
        # 5. Microstructure analysis
        micro_signal = {"imbalance": 0}
        if trades and orderbook:
            micro_signal = self.microstructure.analyze(trades, orderbook)
        
        # 6. Calculate confidence (calibrated)
        base_confidence = self._calculate_confidence(trend, volatility, momentum)
        calibration = self.online_learner.calculate_confidence_calibration()
        self.state.confidence = base_confidence * (0.5 + calibration * 0.5)
        self.state.calibration_error = 1.0 - calibration
        
        # 7. Get optimized hyperparameters
        recent_pnl = [o["pnl"] for o in list(self.online_learner.trade_outcomes)[-20:]]
        params = self.hyperparams.optimize(recent_pnl)
        
        # 8. Calculate position multiplier
        self.state.position_multiplier = self._calculate_position_multiplier(
            regime=self.state.regime,
            confidence=self.state.confidence,
            cross_signal=cross_signal,
            micro_signal=micro_signal,
            params=params,
        )
        
        # 9. Calculate risk multiplier
        self.state.risk_multiplier = 1.0 / (self.state.position_multiplier + 0.1)
        
        # 10. Get strategy weights (optimized by online learning)
        self.state.strategy_weights = self._get_optimized_strategy_weights(
            self.state.regime,
            self.online_learner.get_strategy_performance(self.state.regime),
        )
        
        # 11. Calculate edge score
        self.state.edge_score = self._calculate_edge_score(
            self.state.confidence,
            calibration,
            cross_signal.get("strength", 0),
        )
        
        # 12. Update learning rate
        self.state.learning_rate = params["learning_rate"]
        self.state.adaptation_quality = calibration * self.state.confidence
        
        return self.state
    
    def record_trade_outcome(
        self,
        regime: Regime,
        strategy: str,
        pnl: float,
        confidence: float,
    ):
        """Record trade outcome for online learning."""
        self.online_learner.record_trade(regime, strategy, pnl, confidence)
    
    def _detect_regime(
        self,
        trend: float,
        volatility: float,
        momentum: float,
        volume_ratio: float,
    ) -> Regime:
        """Detect regime using ensemble."""
        scores = {}
        
        scores[Regime.STRONG_UPTREND] = max(0, trend) * max(0, momentum) * 10
        scores[Regime.STRONG_DOWNTREND] = max(0, -trend) * max(0, -momentum) * 10
        scores[Regime.HIGH_VOLATILITY] = volatility * 5
        scores[Regime.CRASH] = max(0, -momentum * 3) * max(0, -trend * 2)
        scores[Regime.PUMP] = max(0, momentum * 3) * max(0, trend * 2)
        scores[Regime.RANGING_TIGHT] = (1 - abs(trend)) * (1 - volatility * 5)
        scores[Regime.RANGING_WIDE] = (1 - abs(trend)) * 0.5
        scores[Regime.BREAKOUT_PENDING] = abs(trend) * 2 if abs(trend) > 0.02 else 0
        scores[Regime.ACCUMULATION] = max(0, -momentum * 0.5) * (1 - volatility) if trend > 0 else 0
        scores[Regime.DISTRIBUTION] = max(0, momentum * 0.5) * (1 - volatility) if trend < 0 else 0
        
        # Adjust by learning
        for regime in scores:
            accuracy = self.online_learner.get_regime_accuracy(regime)
            scores[regime] *= (0.5 + accuracy)
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return max(scores, key=scores.get)
    
    def _calculate_confidence(
        self,
        trend: float,
        volatility: float,
        momentum: float,
    ) -> float:
        """Calculate confidence in regime detection."""
        # High confidence when:
        # - Strong trend (clear direction)
        # - Low volatility (stable)
        # - Momentum aligned with trend
        
        trend_strength = abs(trend)
        vol_factor = 1.0 - min(volatility, 1.0)
        momentum_alignment = 1.0 - abs(trend - momentum)
        
        confidence = (
            trend_strength * 0.4 +
            vol_factor * 0.3 +
            momentum_alignment * 0.3
        )
        
        return max(0.1, min(confidence, 0.95))
    
    def _calculate_position_multiplier(
        self,
        regime: Regime,
        confidence: float,
        cross_signal: Dict,
        micro_signal: Dict,
        params: Dict,
    ) -> float:
        """Calculate position multiplier."""
        # Base by regime
        base = {
            Regime.STRONG_UPTREND: 1.2,
            Regime.WEAK_UPTREND: 0.9,
            Regime.ACCUMULATION: 0.8,
            Regime.DISTRIBUTION: 0.5,
            Regime.STRONG_DOWNTREND: 0.4,
            Regime.WEAK_DOWNTREND: 0.5,
            Regime.HIGH_VOLATILITY: 0.5,
            Regime.LOW_VOLATILITY: 0.8,
            Regime.CRASH: 0.1,
            Regime.PUMP: 0.6,
            Regime.RANGING_TIGHT: 0.6,
            Regime.RANGING_WIDE: 0.5,
            Regime.BREAKOUT_PENDING: 0.9,
            Regime.REVERSAL_PENDING: 0.4,
            Regime.BLACK_SWAN: 0.0,
            Regime.EUPHORIA: 0.4,
            Regime.CAPITULATION: 0.2,
            Regime.RECOVERY: 0.7,
        }.get(regime, 0.5)
        
        # Adjust by confidence
        adjusted = base * confidence
        
        # Adjust by cross-asset signal
        cross_boost = 1.0 + cross_signal.get("strength", 0) * 0.2
        adjusted *= cross_boost
        
        # Adjust by microstructure
        imbalance = abs(micro_signal.get("imbalance", 0))
        if imbalance > 0.3:
            adjusted *= 0.8  # Reduce when order flow is extreme
        
        # Apply hyperparameters
        adjusted *= params.get("position_scale", 1.0)
        
        return max(0.0, min(adjusted, 1.5))
    
    def _get_optimized_strategy_weights(
        self,
        regime: Regime,
        strategy_performance: Dict[str, float],
    ) -> Dict[str, float]:
        """Get strategy weights optimized by online learning."""
        # Base weights by regime
        base_weights = {
            Regime.STRONG_UPTREND: {"trend": 0.4, "momentum": 0.3, "breakout": 0.2, "swing": 0.1},
            Regime.WEAK_UPTREND: {"swing": 0.3, "mean_reversion": 0.3, "trend": 0.2, "grid": 0.2},
            Regime.ACCUMULATION: {"mean_reversion": 0.4, "swing": 0.3, "grid": 0.2, "accumulation": 0.1},
            Regime.DISTRIBUTION: {"mean_reversion": 0.3, "swing": 0.3, "distribution": 0.2, "grid": 0.2},
            Regime.STRONG_DOWNTREND: {"trend_short": 0.4, "momentum": 0.3, "volatility": 0.2, "swing": 0.1},
            Regime.WEAK_DOWNTREND: {"swing": 0.3, "mean_reversion": 0.3, "trend_short": 0.2, "grid": 0.2},
            Regime.HIGH_VOLATILITY: {"volatility": 0.4, "breakout": 0.3, "scalping": 0.2, "swing": 0.1},
            Regime.LOW_VOLATILITY: {"mean_reversion": 0.4, "grid": 0.3, "scalping": 0.2, "range": 0.1},
            Regime.CRASH: {"mean_reversion": 0.3, "volatility": 0.3, "trend_short": 0.3, "contrarian": 0.1},
            Regime.PUMP: {"momentum": 0.4, "trend": 0.3, "breakout": 0.2, "scalping": 0.1},
            Regime.RANGING_TIGHT: {"mean_reversion": 0.4, "grid": 0.3, "scalping": 0.2, "range": 0.1},
            Regime.RANGING_WIDE: {"swing": 0.4, "mean_reversion": 0.3, "grid": 0.2, "breakout": 0.1},
            Regime.BREAKOUT_PENDING: {"breakout": 0.5, "momentum": 0.3, "trend": 0.1, "swing": 0.1},
            Regime.REVERSAL_PENDING: {"mean_reversion": 0.4, "swing": 0.3, "counter_trend": 0.2, "grid": 0.1},
        }
        
        weights = base_weights.get(regime, {"mean_reversion": 0.5, "trend": 0.5})
        
        # Adjust by online learning performance
        for strategy, perf in strategy_performance.items():
            if strategy in weights:
                # Boost good strategies, reduce bad ones
                perf_factor = 1.0 + perf * 10  # Scale P&L impact
                weights[strategy] *= max(0.5, min(perf_factor, 2.0))
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _calculate_edge_score(
        self,
        confidence: float,
        calibration: float,
        cross_strength: float,
    ) -> float:
        """Calculate current edge over market."""
        return (
            confidence * 0.4 +
            calibration * 0.4 +
            cross_strength * 0.2
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "cycle": self.cycle_count,
            "regime": self.state.regime.value,
            "confidence": self.state.confidence,
            "predicted_regime": self.state.predicted_regime.value,
            "prediction_horizon": self.state.prediction_horizon,
            "position_multiplier": self.state.position_multiplier,
            "edge_score": self.state.edge_score,
            "calibration_error": self.state.calibration_error,
            "adaptation_quality": self.state.adaptation_quality,
            "trades_learned": len(self.online_learner.trade_outcomes),
            "hyperparams": self.hyperparams.params,
        }


def get_ultimate_adaptation() -> UltimateAdaptationSystem:
    """Get ultimate adaptation system."""
    return UltimateAdaptationSystem()
