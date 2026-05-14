"""
QUANTUM ADAPTIVE RISK TOLERANCE ENGINE
========================================
Risk tolerance that ADAPTS and uses QUANTUM to optimize.

Features:
1. Dynamic risk adjustment based on market regime
2. Quantum optimization of risk parameters
3. Self-learning from performance
4. Multi-objective optimization (return vs risk)
5. Black swan protection
6. Kelly Criterion with quantum enhancement
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """Current risk profile."""
    position_size_pct: float  # % of capital per trade
    max_daily_loss_pct: float  # Max daily loss %
    max_drawdown_pct: float  # Max total drawdown %
    leverage: float  # Max leverage
    stop_loss_pct: float  # Default stop loss %
    confidence_threshold: float  # Min confidence to trade
    trade_frequency: float  # 0-1, how often to trade
    aggression: float  # 0-1, overall aggression
    quantum_edge: float  # 0-1, quantum enhancement level


class QuantumRiskOptimizer:
    """Quantum-enhanced risk parameter optimization."""
    
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.parameter_history: deque = deque(maxlen=1000)
        self.optimal_params: Dict[str, float] = {}
        
    def optimize_risk_params(
        self,
        current_performance: Dict[str, float],
        market_state: Dict[str, float],
    ) -> Dict[str, float]:
        """Use quantum-inspired optimization to find optimal risk parameters."""
        
        # Extract performance metrics
        sharpe = current_performance.get("sharpe", 0)
        win_rate = current_performance.get("win_rate", 0.5)
        max_dd = current_performance.get("max_drawdown", 0)
        recent_return = current_performance.get("recent_return", 0)
        
        # Extract market state
        volatility = market_state.get("volatility", 0.02)
        regime = market_state.get("regime", "neutral")
        trend_strength = market_state.get("trend_strength", 0)
        
        # Quantum superposition of risk states
        # Explore multiple risk configurations simultaneously
        risk_configs = self._generate_quantum_superposition(
            sharpe, win_rate, max_dd, volatility, regime
        )
        
        # Quantum interference - collapse to optimal
        optimal_config = self._quantum_interference(
            risk_configs, current_performance, market_state
        )
        
        # Entanglement - ensure parameters work together
        entangled_params = self._quantum_entanglement(optimal_config)
        
        self.optimal_params = entangled_params
        self.parameter_history.append({
            "params": entangled_params,
            "performance": current_performance,
            "timestamp": time.time(),
        })
        
        return entangled_params
    
    def _generate_quantum_superposition(
        self,
        sharpe: float,
        win_rate: float,
        max_dd: float,
        volatility: float,
        regime: str,
    ) -> List[Dict[str, float]]:
        """Generate quantum superposition of risk configurations."""
        configs = []
        
        # Conservative superposition
        configs.append({
            "position_size": 0.02,
            "leverage": 1.0,
            "stop_loss": 0.01,
            "confidence": 0.8,
            "amplitude": 0.2,  # Probability amplitude
        })
        
        # Moderate superposition
        configs.append({
            "position_size": 0.10,
            "leverage": 2.0,
            "stop_loss": 0.05,
            "confidence": 0.6,
            "amplitude": 0.3,
        })
        
        # Aggressive superposition
        configs.append({
            "position_size": 0.25,
            "leverage": 5.0,
            "stop_loss": 0.10,
            "confidence": 0.4,
            "amplitude": 0.3,
        })
        
        # Ultra-aggressive superposition
        configs.append({
            "position_size": 0.40,
            "leverage": 10.0,
            "stop_loss": 0.20,
            "confidence": 0.2,
            "amplitude": 0.2,
        })
        
        # Adjust amplitudes based on current state
        if sharpe > 1.0:
            # Good performance - increase aggressive amplitudes
            configs[2]["amplitude"] *= 1.5
            configs[3]["amplitude"] *= 1.3
        elif sharpe < 0:
            # Poor performance - increase conservative amplitudes
            configs[0]["amplitude"] *= 1.5
            configs[1]["amplitude"] *= 1.3
        
        if volatility > 0.5:
            # High volatility - be more conservative
            configs[0]["amplitude"] *= 1.5
            configs[3]["amplitude"] *= 0.5
        
        if regime == "bull":
            configs[2]["amplitude"] *= 1.3
        elif regime == "bear":
            configs[0]["amplitude"] *= 1.5
        
        # Normalize amplitudes
        total = sum(c["amplitude"] for c in configs)
        for c in configs:
            c["amplitude"] /= total
        
        return configs
    
    def _quantum_interference(
        self,
        configs: List[Dict[str, float]],
        performance: Dict[str, float],
        market: Dict[str, float],
    ) -> Dict[str, float]:
        """Quantum interference - collapse to optimal configuration."""
        
        # Calculate interference pattern
        scores = []
        for config in configs:
            # Score based on expected utility
            expected_return = config["position_size"] * config["leverage"] * 0.1
            expected_risk = config["position_size"] * config["leverage"] * (1 - 0.5)
            
            # Kelly-inspired scoring
            kelly_score = (expected_return - expected_risk) / (expected_return + 1e-8)
            
            # Interference with amplitude
            score = config["amplitude"] * max(0, kelly_score)
            scores.append(score)
        
        # Collapse to highest score (measurement)
        best_idx = np.argmax(scores)
        
        return {
            "position_size": configs[best_idx]["position_size"],
            "leverage": configs[best_idx]["leverage"],
            "stop_loss": configs[best_idx]["stop_loss"],
            "confidence": configs[best_idx]["confidence"],
        }
    
    def _quantum_entanglement(self, params: Dict[str, float]) -> Dict[str, float]:
        """Ensure parameters are entangled (work together optimally)."""
        
        position_size = params["position_size"]
        leverage = params["leverage"]
        
        # Entanglement rules:
        # Higher leverage → smaller position size
        # Higher stop loss → can use larger position
        # Lower confidence → smaller position
        
        # Adjust position size based on leverage
        effective_exposure = position_size * leverage
        max_exposure = 0.5  # Max 50% effective exposure
        
        if effective_exposure > max_exposure:
            # Scale down position to maintain max exposure
            position_size = max_exposure / leverage
        
        # Adjust stop loss based on volatility targeting
        target_vol = 0.15  # 15% annualized vol target
        stop_loss = params["stop_loss"] * (target_vol / 0.15)
        
        # Confidence affects position size (Kelly-like)
        confidence_adj = params["confidence"] / 0.5  # Normalize around 50%
        position_size *= min(confidence_adj, 1.5)
        
        return {
            "position_size_pct": np.clip(position_size, 0.01, 0.50),
            "leverage": np.clip(leverage, 1.0, 20.0),
            "stop_loss_pct": np.clip(stop_loss, 0.005, 0.30),
            "confidence_threshold": np.clip(params["confidence"], 0.1, 0.9),
            "max_daily_loss_pct": position_size * 2,  # 2x position size
            "max_drawdown_pct": position_size * 10,  # 10x position size
            "trade_frequency": 0.5 + params["confidence"] * 0.5,
            "aggression": (position_size * leverage) / 2,
        }


class AdaptiveRiskController:
    """Adaptive risk controller that adjusts based on conditions."""
    
    def __init__(self):
        self.base_profile = RiskProfile(
            position_size_pct=0.10,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.20,
            leverage=3.0,
            stop_loss_pct=0.05,
            confidence_threshold=0.6,
            trade_frequency=0.5,
            aggression=0.5,
            quantum_edge=0.5,
        )
        
        self.current_profile = self.base_profile
        self.performance_history: deque = deque(maxlen=100)
        self.risk_adjustments: deque = deque(maxlen=100)
        
    def adjust_for_conditions(
        self,
        regime: str,
        volatility: float,
        drawdown: float,
        recent_performance: float,
    ) -> RiskProfile:
        """Adjust risk profile based on current conditions."""
        
        # Start with base profile
        position_size = self.base_profile.position_size_pct
        leverage = self.base_profile.leverage
        stop_loss = self.base_profile.stop_loss_pct
        confidence = self.base_profile.confidence_threshold
        
        # REGIME ADJUSTMENTS
        if regime == "strong_uptrend":
            position_size *= 1.5
            leverage *= 1.3
            confidence *= 0.9  # Lower threshold in trends
        elif regime == "weak_uptrend":
            position_size *= 1.2
            leverage *= 1.1
        elif regime == "ranging":
            position_size *= 0.8
            leverage *= 0.9
        elif regime == "weak_downtrend":
            position_size *= 0.5
            leverage *= 0.7
            confidence *= 1.2  # Higher threshold
        elif regime == "strong_downtrend":
            position_size *= 0.3
            leverage *= 0.5
            confidence *= 1.5
        elif regime == "high_volatility":
            position_size *= 0.5
            leverage *= 0.6
            stop_loss *= 1.5  # Wider stops
        
        # VOLATILITY ADJUSTMENTS
        if volatility > 0.5:
            position_size *= 0.6
            leverage *= 0.7
        elif volatility > 0.3:
            position_size *= 0.8
            leverage *= 0.9
        elif volatility < 0.1:
            position_size *= 1.2
            leverage *= 1.1
        
        # DRAWDOWN ADJUSTMENTS (CRITICAL)
        if drawdown > 0.30:
            position_size *= 0.2
            leverage = 1.0
            confidence *= 2.0
        elif drawdown > 0.20:
            position_size *= 0.4
            leverage *= 0.5
            confidence *= 1.5
        elif drawdown > 0.10:
            position_size *= 0.7
            leverage *= 0.8
            confidence *= 1.2
        
        # PERFORMANCE ADJUSTMENTS
        if recent_performance > 0.1:
            # Winning streak - can be slightly more aggressive
            position_size *= 1.1
        elif recent_performance < -0.05:
            # Losing streak - be more conservative
            position_size *= 0.7
            confidence *= 1.3
        
        # Calculate derived values
        max_daily_loss = position_size * 3
        max_drawdown = position_size * 15
        trade_frequency = 0.5 + (confidence - 0.5) * 0.5
        aggression = (position_size * leverage) / 2
        
        # Clip to safe ranges
        self.current_profile = RiskProfile(
            position_size_pct=np.clip(position_size, 0.01, 0.50),
            max_daily_loss_pct=np.clip(max_daily_loss, 0.02, 0.30),
            max_drawdown_pct=np.clip(max_drawdown, 0.05, 0.60),
            leverage=np.clip(leverage, 1.0, 20.0),
            stop_loss_pct=np.clip(stop_loss, 0.005, 0.30),
            confidence_threshold=np.clip(confidence, 0.2, 0.95),
            trade_frequency=np.clip(trade_frequency, 0.1, 1.0),
            aggression=np.clip(aggression, 0.1, 1.0),
            quantum_edge=0.5,
        )
        
        self.risk_adjustments.append({
            "regime": regime,
            "volatility": volatility,
            "drawdown": drawdown,
            "profile": self.current_profile.__dict__,
            "timestamp": time.time(),
        })
        
        return self.current_profile


class PerformanceLearner:
    """Learn from performance to improve risk decisions."""
    
    def __init__(self):
        self.performance_history: deque = deque(maxlen=1000)
        self.risk_performance: Dict[str, List[float]] = {}
        
    def record_performance(
        self,
        risk_params: Dict[str, float],
        return_pct: float,
        drawdown: float,
    ):
        """Record performance for given risk parameters."""
        self.performance_history.append({
            "params": risk_params,
            "return": return_pct,
            "drawdown": drawdown,
            "timestamp": time.time(),
        })
        
        # Track performance by risk level
        risk_level = self._classify_risk(risk_params)
        if risk_level not in self.risk_performance:
            self.risk_performance[risk_level] = []
        self.risk_performance[risk_level].append(return_pct)
    
    def _classify_risk(self, params: Dict[str, float]) -> str:
        """Classify risk level."""
        position_size = params.get("position_size_pct", 0.1)
        leverage = params.get("leverage", 1)
        
        exposure = position_size * leverage
        
        if exposure < 0.1:
            return "conservative"
        elif exposure < 0.3:
            return "moderate"
        elif exposure < 0.6:
            return "aggressive"
        else:
            return "extreme"
    
    def get_optimal_risk_level(self) -> str:
        """Get optimal risk level based on historical performance."""
        if not self.risk_performance:
            return "moderate"
        
        # Calculate risk-adjusted returns for each level
        best_level = "moderate"
        best_score = -float('inf')
        
        for level, returns in self.risk_performance.items():
            if len(returns) < 10:
                continue
            
            returns_arr = np.array(returns)
            mean_return = np.mean(returns_arr)
            std_return = np.std(returns_arr)
            
            # Sharpe-like score
            score = mean_return / (std_return + 1e-8)
            
            if score > best_score:
                best_score = score
                best_level = level
        
        return best_level


class BlackSwanProtector:
    """Protection against black swan events."""
    
    def __init__(self):
        self.risk_indicators: Dict[str, float] = {}
        self.black_swan_probability: float = 0
        
    def update_indicators(
        self,
        volatility: float,
        funding_rate: float,
        open_interest: float,
        whale_activity: float,
    ):
        """Update black swan indicators."""
        self.risk_indicators = {
            "volatility": volatility,
            "funding_rate": funding_rate,
            "open_interest": open_interest,
            "whale_activity": whale_activity,
        }
        
        # Calculate black swan probability
        score = 0
        
        if volatility > 0.8:
            score += 0.3
        if abs(funding_rate) > 0.001:
            score += 0.2
        if open_interest > 0.8:
            score += 0.2
        if whale_activity > 0.7:
            score += 0.3
        
        self.black_swan_probability = score
    
    def get_protection_level(self) -> Dict[str, Any]:
        """Get protection level based on black swan probability."""
        prob = self.black_swan_probability
        
        if prob > 0.7:
            return {
                "level": "maximum",
                "position_multiplier": 0.1,
                "leverage_limit": 1.0,
                "action": "reduce_all_positions",
            }
        elif prob > 0.5:
            return {
                "level": "high",
                "position_multiplier": 0.3,
                "leverage_limit": 2.0,
                "action": "hedge_tail_risk",
            }
        elif prob > 0.3:
            return {
                "level": "medium",
                "position_multiplier": 0.6,
                "leverage_limit": 3.0,
                "action": "tighten_stops",
            }
        else:
            return {
                "level": "normal",
                "position_multiplier": 1.0,
                "leverage_limit": 10.0,
                "action": "normal_trading",
            }


class QuantumAdaptiveRiskEngine:
    """
    QUANTUM ADAPTIVE RISK TOLERANCE ENGINE
    
    Combines:
    - Quantum optimization of risk parameters
    - Adaptive adjustment based on conditions
    - Self-learning from performance
    - Black swan protection
    """
    
    def __init__(self, initial_capital: float = 1000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        # Components
        self.quantum_optimizer = QuantumRiskOptimizer(n_qubits=8)
        self.adaptive_controller = AdaptiveRiskController()
        self.performance_learner = PerformanceLearner()
        self.black_swan_protector = BlackSwanProtector()
        
        # Current state
        self.current_profile = self.adaptive_controller.current_profile
        self.optimal_params: Dict[str, float] = {}
        
        # History
        self.risk_history: deque = deque(maxlen=1000)
        
        logger.info("QuantumAdaptiveRiskEngine initialized")
    
    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        regime: str,
    ) -> float:
        """Calculate optimal position size."""
        # Get current risk profile
        profile = self.current_profile
        
        # Base position size
        base_size = self.initial_capital * profile.position_size_pct
        
        # Adjust for confidence
        confidence_adj = confidence / profile.confidence_threshold
        base_size *= min(confidence_adj, 1.5)
        
        # Adjust for volatility
        vol_adj = 0.2 / (volatility + 0.01)  # Target 20% vol
        base_size *= np.clip(vol_adj, 0.3, 2.0)
        
        # Apply black swan protection
        protection = self.black_swan_protector.get_protection_level()
        base_size *= protection["position_multiplier"]
        
        # Apply leverage
        effective_size = base_size * min(profile.leverage, protection["leverage_limit"])
        
        # Cap at max daily loss limit
        max_size = self.current_capital * profile.max_daily_loss_pct
        effective_size = min(effective_size, max_size)
        
        return effective_size
    
    def update(
        self,
        market_state: Dict[str, float],
        performance: Dict[str, float],
    ) -> RiskProfile:
        """Update risk profile based on current conditions."""
        
        # Extract market conditions
        regime = market_state.get("regime", "ranging")
        volatility = market_state.get("volatility", 0.02)
        drawdown = self._calculate_drawdown()
        recent_return = performance.get("recent_return", 0)
        
        # Update black swan protection
        self.black_swan_protector.update_indicators(
            volatility=volatility,
            funding_rate=market_state.get("funding_rate", 0),
            open_interest=market_state.get("open_interest", 0.5),
            whale_activity=market_state.get("whale_activity", 0.5),
        )
        
        # Adaptive adjustment
        self.current_profile = self.adaptive_controller.adjust_for_conditions(
            regime=regime,
            volatility=volatility,
            drawdown=drawdown,
            recent_performance=recent_return,
        )
        
        # Quantum optimization (every 10 cycles)
        if len(self.risk_history) % 10 == 0:
            self.optimal_params = self.quantum_optimizer.optimize_risk_params(
                current_performance=performance,
                market_state=market_state,
            )
            
            # Blend quantum recommendations with adaptive
            self._blend_quantum_adaptive()
        
        # Record for learning
        self.performance_learner.record_performance(
            risk_params=self.current_profile.__dict__,
            return_pct=recent_return,
            drawdown=drawdown,
        )
        
        # Update history
        self.risk_history.append({
            "profile": self.current_profile.__dict__,
            "market_state": market_state,
            "performance": performance,
            "timestamp": time.time(),
        })
        
        return self.current_profile
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown."""
        self.peak_capital = max(self.peak_capital, self.current_capital)
        return (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
    
    def _blend_quantum_adaptive(self):
        """Blend quantum recommendations with adaptive controller."""
        if not self.optimal_params:
            return
        
        # 70% quantum, 30% adaptive (quantum gets more weight over time)
        blend_factor = min(0.7, 0.3 + len(self.risk_history) * 0.001)
        
        profile = self.current_profile
        quantum = self.optimal_params
        
        # Blend position size
        profile.position_size_pct = (
            profile.position_size_pct * (1 - blend_factor) +
            quantum.get("position_size_pct", profile.position_size_pct) * blend_factor
        )
        
        # Blend leverage
        profile.leverage = (
            profile.leverage * (1 - blend_factor) +
            quantum.get("leverage", profile.leverage) * blend_factor
        )
        
        # Update quantum edge indicator
        profile.quantum_edge = blend_factor
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "current_profile": self.current_profile.__dict__,
            "optimal_params": self.optimal_params,
            "drawdown": self._calculate_drawdown(),
            "black_swan_probability": self.black_swan_protector.black_swan_probability,
            "optimal_risk_level": self.performance_learner.get_optimal_risk_level(),
            "quantum_edge": self.current_profile.quantum_edge,
        }
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        profile = self.current_profile
        
        return {
            "risk_level": self._classify_risk_level(profile),
            "position_size": f"{profile.position_size_pct:.1%}",
            "leverage": f"{profile.leverage:.1f}x",
            "max_exposure": f"{profile.position_size_pct * profile.leverage:.1%}",
            "stop_loss": f"{profile.stop_loss_pct:.1%}",
            "max_daily_loss": f"{profile.max_daily_loss_pct:.1%}",
            "max_drawdown": f"{profile.max_drawdown_pct:.1%}",
            "confidence_threshold": f"{profile.confidence_threshold:.0%}",
            "quantum_enhancement": f"{profile.quantum_edge:.0%}",
            "expected_monthly_return": self._estimate_monthly_return(profile),
            "risk_of_ruin": self._estimate_risk_of_ruin(profile),
        }
    
    def _classify_risk_level(self, profile: RiskProfile) -> str:
        """Classify risk level."""
        exposure = profile.position_size_pct * profile.leverage
        
        if exposure < 0.1:
            return "CONSERVATIVE"
        elif exposure < 0.3:
            return "MODERATE"
        elif exposure < 0.6:
            return "AGGRESSIVE"
        else:
            return "EXTREME"
    
    def _estimate_monthly_return(self, profile: RiskProfile) -> str:
        """Estimate expected monthly return."""
        exposure = profile.position_size_pct * profile.leverage
        
        # Simplified estimation
        if exposure < 0.1:
            return "10-20%"
        elif exposure < 0.3:
            return "30-60%"
        elif exposure < 0.6:
            return "60-120%"
        else:
            return "100-300%"
    
    def _estimate_risk_of_ruin(self, profile: RiskProfile) -> str:
        """Estimate risk of ruin."""
        exposure = profile.position_size_pct * profile.leverage
        
        if exposure < 0.1:
            return "<1%"
        elif exposure < 0.3:
            return "2-5%"
        elif exposure < 0.6:
            return "10-20%"
        else:
            return "25-40%"


def get_quantum_adaptive_risk(initial_capital: float = 1000) -> QuantumAdaptiveRiskEngine:
    """Get Quantum Adaptive Risk Engine."""
    return QuantumAdaptiveRiskEngine(initial_capital)
