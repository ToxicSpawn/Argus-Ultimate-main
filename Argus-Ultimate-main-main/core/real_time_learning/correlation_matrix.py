"""
Dynamic Correlation Matrix - Real-Time Learning Component

This component continuously updates asset correlations for optimal diversification.
Key features:
- Real-time correlation matrix updates
- Regime-aware correlation adjustments
- Portfolio concentration alerts
- Integration with position sizing
"""

from __future__ import annotations
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .orchestrator import LearningComponent

logger = logging.getLogger(__name__)


@dataclass
class AssetCorrelationData:
    """Tracks correlation data for asset pairs"""
    
    asset1: str
    asset2: str
    correlation_history: List[float] = field(default_factory=list)
    window_size: int = 30  # Number of observations to keep
    
    def add_observation(self, correlation: float) -> None:
        """Add a new correlation observation"""
        self.correlation_history.append(correlation)
        if len(self.correlation_history) > self.window_size:
            self.correlation_history.pop(0)
    
    def get_current_correlation(self) -> float:
        """Get the current correlation (average of recent observations)"""
        if not self.correlation_history:
            return 0.0
        return np.mean(self.correlation_history)
    
    def get_volatility(self) -> float:
        """Get volatility of correlation (standard deviation)"""
        if len(self.correlation_history) < 2:
            return 0.0
        return np.std(self.correlation_history)


class DynamicCorrelationMatrix(LearningComponent):
    """Dynamically updates asset correlations for optimal diversification"""
    
    def __init__(self):
        super().__init__(
            name="correlation_matrix",
            version="1.0",
            enabled=True,
            update_frequency=3  # Update every 3 trade cycles
        )
        
        # Correlation tracking
        self.assets: List[str] = []
        self.correlation_pairs: Dict[Tuple[str, str], AssetCorrelationData] = {}
        self.current_matrix: Dict[Tuple[str, str], float] = {}
        self.matrix_history: List[Dict] = []
        
        # Diversification metrics
        self.portfolio_concentration: float = 0.0
        self.max_concentration_threshold: float = 0.3  # 30% max in any single asset
        self.correlation_threshold: float = 0.7  # Warning threshold
        
        # Regime tracking
        self.current_regime: str = "stable"
        self.regime_correlation_adjustments: Dict[str, float] = {
            'stable': 1.0,
            'volatile': 1.2,  # Correlations tend to increase in volatile markets
            'trending': 0.9,   # Correlations may decrease in strong trends
            'range': 1.1      # Correlations may increase in range-bound markets
        }
        
        # State tracking
        self.last_updated: Optional[datetime] = None
    
    def initialize_assets(self, assets: List[str]) -> None:
        """Initialize tracking for all asset pairs"""
        self.assets = assets
        
        # Initialize correlation tracking for all pairs
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1 = assets[i]
                asset2 = assets[j]
                pair = (asset1, asset2)
                
                self.correlation_pairs[pair] = AssetCorrelationData(asset1, asset2)
                self.current_matrix[pair] = 0.0  # Initial correlation
                self.current_matrix[(asset2, asset1)] = 0.0  # Symmetric
    
    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from new market data and update correlations"""
        
        # Update regime
        self._update_regime(data)
        
        # Update correlations from returns data
        if 'asset_returns' in data:
            self._update_correlations(data['asset_returns'])
        
        # Calculate diversification metrics
        self._calculate_diversification_metrics()
        
        # Store current matrix in history
        self.matrix_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regime': self.current_regime,
            'matrix': self.current_matrix.copy(),
            'concentration': self.portfolio_concentration
        })
        
        # Keep only last 100 matrices
        if len(self.matrix_history) > 100:
            self.matrix_history.pop(0)
        
        self.last_updated = datetime.now(timezone.utc)
        
        return {
            'correlation_matrix': self.current_matrix.copy(),
            'portfolio_concentration': self.portfolio_concentration,
            'regime': self.current_regime
        }
    
    def _update_regime(self, data: Dict[str, Any]) -> None:
        """Update current market regime"""
        if 'market_data' in data:
            market_data = data['market_data']
            volatility = market_data.get('volatility', 0.01)
            trend_strength = market_data.get('trend_strength', 0.2)
            
            # Debug output
            logger.info(f"Regime detection - Volatility: {volatility:.3f}, Trend: {trend_strength:.3f}")
            
            # Simple regime detection (could be enhanced)
            if volatility > 0.02:
                self.current_regime = 'volatile'
            elif trend_strength > 0.5:
                self.current_regime = 'trending'
            elif volatility < 0.008:
                self.current_regime = 'range'
            else:
                self.current_regime = 'stable'
                
            logger.info(f"Detected regime: {self.current_regime}")
    
    def _update_correlations(self, returns_data: Dict[str, List[float]]) -> None:
        """Update correlation matrix from asset returns"""
        if not self.assets or not returns_data:
            return
        
        # Get the assets we have returns for
        available_assets = [a for a in self.assets if a in returns_data]
        
        if len(available_assets) < 2:
            return
        
        # Create a DataFrame for correlation calculation
        returns_df = pd.DataFrame()
        for asset in available_assets:
            returns_df[asset] = returns_data[asset][-30:]  # Use last 30 returns
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr(method='spearman')
        
        # Update our correlation tracking
        for i in range(len(available_assets)):
            for j in range(i+1, len(available_assets)):
                asset1 = available_assets[i]
                asset2 = available_assets[j]
                pair = (asset1, asset2)
                
                if pair in self.correlation_pairs:
                    # Apply regime adjustment
                    raw_corr = corr_matrix.loc[asset1, asset2]
                    adjusted_corr = raw_corr * self.regime_correlation_adjustments.get(self.current_regime, 1.0)
                    
                    # Add to history
                    self.correlation_pairs[pair].add_observation(adjusted_corr)
                    
                    # Update current matrix
                    self.current_matrix[pair] = adjusted_corr
                    self.current_matrix[(asset2, asset1)] = adjusted_corr
    
    def _calculate_diversification_metrics(self) -> None:
        """Calculate portfolio diversification metrics"""
        if not self.current_matrix:
            self.portfolio_concentration = 0.0
            return
        
        # Calculate average correlation
        correlations = [abs(corr) for corr in self.current_matrix.values()]
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Simple concentration metric (could be enhanced with actual portfolio weights)
        # For demo purposes, we'll use 1 - avg_correlation as a proxy
        self.portfolio_concentration = avg_correlation
        
        # Check for concentration warnings
        if self.portfolio_concentration > self.max_concentration_threshold:
            logger.warning(f"High portfolio concentration: {self.portfolio_concentration:.2f} > {self.max_concentration_threshold:.2f}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            'current_matrix': self.current_matrix.copy(),
            'portfolio_concentration': self.portfolio_concentration,
            'current_regime': self.current_regime,
            'max_concentration_threshold': self.max_concentration_threshold,
            'correlation_threshold': self.correlation_threshold,
            'regime_correlation_adjustments': self.regime_correlation_adjustments.copy()
        }
    
    def rollback(self) -> None:
        """Revert to last known good state"""
        if len(self.matrix_history) > 1:
            # Revert to previous matrix
            previous = self.matrix_history[-2]
            self.current_matrix = previous['matrix'].copy()
            self.portfolio_concentration = previous['concentration']
            self.current_regime = previous['regime']
            logger.info(f"Rolled back to previous correlation matrix from {previous['timestamp']}")
        else:
            # Fallback to zero correlations
            for pair in self.current_matrix:
                self.current_matrix[pair] = 0.0
            self.portfolio_concentration = 0.0
            logger.warning("No matrix history - reset to zero correlations")
    
    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes"""
        if 'current_matrix' not in new_params:
            return False
        
        # Check matrix symmetry
        matrix = new_params['current_matrix']
        for (a1, a2), corr in matrix.items():
            if (a2, a1) not in matrix or abs(matrix[(a2, a1)] - corr) > 0.001:
                logger.warning(f"Correlation matrix not symmetric for {a1}-{a2}")
                return False
        
        # Check correlation bounds
        for corr in matrix.values():
            if corr < -1.0 or corr > 1.0:
                logger.warning(f"Correlation out of bounds: {corr}")
                return False
        
        # Check concentration threshold
        if 'max_concentration_threshold' in new_params:
            threshold = new_params['max_concentration_threshold']
            if threshold < 0 or threshold > 1.0:
                logger.warning(f"Invalid concentration threshold: {threshold}")
                return False
        
        return True
    
    def get_concentration_alerts(self) -> List[Dict]:
        """Get any concentration alerts"""
        alerts = []
        
        if self.portfolio_concentration > self.max_concentration_threshold:
            alerts.append({
                'type': 'concentration',
                'level': 'warning',
                'message': f"Portfolio concentration {self.portfolio_concentration:.2f} exceeds threshold {self.max_concentration_threshold:.2f}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Check for high correlations
        for pair, corr_data in self.correlation_pairs.items():
            current_corr = corr_data.get_current_correlation()
            if abs(current_corr) > self.correlation_threshold:
                alerts.append({
                    'type': 'correlation',
                    'level': 'warning',
                    'message': f"High correlation between {pair[0]} and {pair[1]}: {current_corr:.2f} > {self.correlation_threshold:.2f}",
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'assets': list(pair)
                })
        
        return alerts
    
    def get_diversification_score(self) -> float:
        """Calculate a diversification score (0-1 where 1 is perfectly diversified)"""
        if not self.current_matrix:
            return 0.5  # Neutral score if no data
        
        # Calculate average absolute correlation
        correlations = [abs(corr) for corr in self.current_matrix.values()]
        avg_corr = np.mean(correlations) if correlations else 0.5
        
        # Diversification score (inverse of average correlation)
        return 1.0 - avg_corr
    
    def get_regime_adjusted_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets adjusted for current regime"""
        pair = (asset1, asset2)
        if pair in self.current_matrix:
            return self.current_matrix[pair]
        
        # Check reverse pair
        pair = (asset2, asset1)
        if pair in self.current_matrix:
            return self.current_matrix[pair]
        
        return 0.0  # Default if not found