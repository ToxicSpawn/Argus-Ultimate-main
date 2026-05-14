"""
Regime-Specific Parameters - Real-Time Learning Component

This component adjusts trading parameters based on detected market regimes.
Key features:
- Volatility-based parameter tuning
- Drawdown and win-rate adjustments
- Regime classification (volatile, range-bound, trending)
- Integration with risk management
"""

from __future__ import annotations
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from .orchestrator import LearningComponent

logger = logging.getLogger(__name__)


@dataclass
class RegimeParameters:
    """Stores parameter adjustments for a specific regime"""
    
    regime: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[Dict] = field(default_factory=list)
    window_size: int = 20  # Number of performances to keep
    
    def add_performance(self, performance: Dict) -> None:
        """Add a new performance observation"""
        self.performance_history.append(performance)
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
    
    def get_average_performance(self) -> Dict:
        """Get average performance metrics"""
        if not self.performance_history:
            return {
                'sharpe_ratio': 0.0,
                'win_rate': 0.5,
                'max_drawdown': 0.0,
                'profit_factor': 1.0
            }
        
        # Calculate averages
        sharpe = np.mean([p['sharpe_ratio'] for p in self.performance_history if 'sharpe_ratio' in p])
        win_rate = np.mean([p['win_rate'] for p in self.performance_history if 'win_rate' in p])
        drawdown = np.mean([p['max_drawdown'] for p in self.performance_history if 'max_drawdown' in p])
        profit_factor = np.mean([p['profit_factor'] for p in self.performance_history if 'profit_factor' in p])
        
        return {
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': drawdown,
            'profit_factor': profit_factor
        }


class RegimeSpecificParameters(LearningComponent):
    """Adjusts trading parameters based on detected market regimes"""
    
    def __init__(self):
        super().__init__(
            name="regime_parameters",
            version="1.0",
            enabled=True,
            update_frequency=1  # Update every trade cycle
        )
        
        # Regime tracking
        self.current_regime: str = "stable"
        self.regime_history: List[str] = []
        self.max_history: int = 100
        
        # Parameter sets for each regime
        self.regime_parameters: Dict[str, RegimeParameters] = {}
        
        # Default parameter ranges
        self.default_parameters = {
            'position_size_pct': {'min': 0.01, 'max': 0.10, 'default': 0.05},
            'max_leverage': {'min': 1.0, 'max': 5.0, 'default': 2.0},
            'stop_loss_pct': {'min': 0.01, 'max': 0.05, 'default': 0.03},
            'take_profit_pct': {'min': 0.02, 'max': 0.10, 'default': 0.05},
            'trailing_stop_pct': {'min': 0.01, 'max': 0.05, 'default': 0.02},
            'entry_aggressiveness': {'min': 0.1, 'max': 0.9, 'default': 0.5},
            'exit_aggressiveness': {'min': 0.1, 'max': 0.9, 'default': 0.5},
            'risk_per_trade_pct': {'min': 0.005, 'max': 0.02, 'default': 0.01}
        }
        
        # Regime-specific adjustments
        self.regime_adjustments = {
            'volatile': {
                'position_size_pct': 0.7,  # Reduce position size
                'max_leverage': 0.5,      # Reduce leverage
                'stop_loss_pct': 1.5,     # Widen stop loss
                'take_profit_pct': 0.8,    # Reduce take profit
                'trailing_stop_pct': 1.2, # Widen trailing stop
                'entry_aggressiveness': 0.6, # Be more cautious on entry
                'exit_aggressiveness': 1.2, # Be more aggressive on exit
                'risk_per_trade_pct': 0.6   # Reduce risk per trade
            },
            'trending': {
                'position_size_pct': 1.3,  # Increase position size
                'max_leverage': 1.2,      # Slightly increase leverage
                'stop_loss_pct': 0.8,     # Tighter stop loss
                'take_profit_pct': 1.5,    # Increase take profit
                'trailing_stop_pct': 1.3, # Tighter trailing stop
                'entry_aggressiveness': 1.2, # Be more aggressive on entry
                'exit_aggressiveness': 0.8, # Be more patient on exit
                'risk_per_trade_pct': 1.2   # Increase risk per trade
            },
            'range': {
                'position_size_pct': 1.1,  # Slightly increase position size
                'max_leverage': 0.9,      # Slightly reduce leverage
                'stop_loss_pct': 0.9,     # Normal stop loss
                'take_profit_pct': 0.9,    # Normal take profit
                'trailing_stop_pct': 1.0, # Normal trailing stop
                'entry_aggressiveness': 0.9, # Be more aggressive on entry
                'exit_aggressiveness': 1.1, # Be more aggressive on exit
                'risk_per_trade_pct': 1.0   # Normal risk per trade
            },
            'stable': {
                'position_size_pct': 1.0,  # Default position size
                'max_leverage': 1.0,      # Default leverage
                'stop_loss_pct': 1.0,     # Default stop loss
                'take_profit_pct': 1.0,    # Default take profit
                'trailing_stop_pct': 1.0, # Default trailing stop
                'entry_aggressiveness': 1.0, # Default entry aggressiveness
                'exit_aggressiveness': 1.0, # Default exit aggressiveness
                'risk_per_trade_pct': 1.0   # Default risk per trade
            }
        }
        
        # Current parameters
        self.current_parameters: Dict[str, Any] = {}
        self._reset_to_defaults()
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.max_performance_history: int = 100

    def _reset_to_defaults(self) -> None:
        """Reset all parameters to their default values"""
        for param, config in self.default_parameters.items():
            self.current_parameters[param] = config['default']
    
    def initialize_regimes(self, regimes: List[str]) -> None:
        """Initialize tracking for all regimes"""
        for regime in regimes:
            if regime not in self.regime_parameters:
                self.regime_parameters[regime] = RegimeParameters(regime)
                
                # Initialize with default parameters
                for param, config in self.default_parameters.items():
                    self.regime_parameters[regime].parameters[param] = config['default']
    
    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from new market data and performance metrics"""
        
        # Update regime
        self._update_regime(data)
        
        # Update performance metrics
        if 'performance_metrics' in data:
            self._update_performance(data['performance_metrics'])
        
        # Adjust parameters based on current regime
        self._adjust_parameters_for_regime()
        
        return {
            'regime': self.current_regime,
            'parameters': self.current_parameters.copy(),
            'regime_performance': {
                r: p.get_average_performance() for r, p in self.regime_parameters.items()
            }
        }
    
    def _update_regime(self, data: Dict[str, Any]) -> None:
        """Update current market regime"""
        if 'market_data' in data:
            market_data = data['market_data']
            volatility = market_data.get('volatility', 0.01)
            trend_strength = market_data.get('trend_strength', 0.2)
            
            # Simple regime detection
            if volatility > 0.02:
                self.current_regime = 'volatile'
            elif trend_strength > 0.5:
                self.current_regime = 'trending'
            elif volatility < 0.008:
                self.current_regime = 'range'
            else:
                self.current_regime = 'stable'
            
            # Add to regime history
            self.regime_history.append(self.current_regime)
            if len(self.regime_history) > self.max_history:
                self.regime_history.pop(0)
    
    def _update_performance(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics for current regime"""
        if self.current_regime not in self.regime_parameters:
            return
        
        # Add performance to current regime
        self.regime_parameters[self.current_regime].add_performance(metrics)
        
        # Add to overall performance history
        self.performance_history.append({
            'regime': self.current_regime,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **metrics
        })
        
        if len(self.performance_history) > self.max_performance_history:
            self.performance_history.pop(0)
    
    def _adjust_parameters_for_regime(self) -> None:
        """Adjust parameters based on current regime"""
        if self.current_regime not in self.regime_adjustments:
            return
        
        adjustments = self.regime_adjustments[self.current_regime]
        
        # Apply adjustments to each parameter
        for param, config in self.default_parameters.items():
            if param in adjustments:
                # Get current value and adjustment factor
                current_value = self.current_parameters[param]
                adjustment_factor = adjustments[param]
                
                # Calculate new value
                new_value = current_value * adjustment_factor
                
                # Apply bounds
                min_val = config['min']
                max_val = config['max']
                bounded_value = max(min_val, min(max_val, new_value))
                
                # Update parameter
                self.current_parameters[param] = bounded_value
                
                # Update regime-specific parameter
                if self.current_regime in self.regime_parameters:
                    self.regime_parameters[self.current_regime].parameters[param] = bounded_value
    
    def get_regime_performance(self, regime: str) -> Dict:
        """Get performance metrics for a specific regime"""
        if regime in self.regime_parameters:
            return self.regime_parameters[regime].get_average_performance()
        return {
            'sharpe_ratio': 0.0,
            'win_rate': 0.5,
            'max_drawdown': 0.0,
            'profit_factor': 1.0
        }
    
    def get_parameter(self, param_name: str) -> Any:
        """Get the current value of a parameter"""
        return self.current_parameters.get(param_name)
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            'current_regime': self.current_regime,
            'current_parameters': self.current_parameters.copy(),
            'regime_adjustments': self.regime_adjustments.copy(),
            'regime_history': self.regime_history[-10:],  # Last 10 regimes
            'regime_performance': {
                r: p.get_average_performance() for r, p in self.regime_parameters.items()
            }
        }
    
    def rollback(self) -> None:
        """Revert to last known good state"""
        if len(self.performance_history) > 1:
            # Find the last regime change
            last_regime = None
            for metric in reversed(self.performance_history[:-1]):
                if metric['regime'] != self.current_regime:
                    last_regime = metric['regime']
                    break
            
            if last_regime and last_regime in self.regime_parameters:
                # Restore parameters from last regime
                for param, value in self.regime_parameters[last_regime].parameters.items():
                    self.current_parameters[param] = value
                
                self.current_regime = last_regime
                logger.info(f"Rolled back to {last_regime} regime parameters")
            else:
                # Fallback to defaults
                self._reset_to_defaults()
                logger.warning("No valid regime history - reset to defaults")
        else:
            # Fallback to defaults
            self._reset_to_defaults()
            logger.warning("No performance history - reset to defaults")
    
    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes"""
        if 'current_parameters' in new_params:
            for param, value in new_params['current_parameters'].items():
                if param in self.default_parameters:
                    config = self.default_parameters[param]
                    if value < config['min'] or value > config['max']:
                        logger.warning(f"Parameter {param} out of bounds: {value} (min: {config['min']}, max: {config['max']})")
                        return False
        
        if 'regime_adjustments' in new_params:
            for regime, adjustments in new_params['regime_adjustments'].items():
                for param, factor in adjustments.items():
                    if factor <= 0:
                        logger.warning(f"Invalid adjustment factor in {regime} for {param}: {factor}")
                        return False
        
        return True
    
    def learn_from_performance(self, performance: Dict[str, Any]) -> None:
        """Specialized learning from performance metrics"""
        self._update_performance(performance)
        self._adjust_parameters_for_regime()