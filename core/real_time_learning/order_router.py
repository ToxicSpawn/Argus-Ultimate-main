"""
Smart Order Router - Real-Time Learning Component

This component dynamically routes orders to optimal execution venues based on:
- Liquidity analysis
- Latency measurements
- Historical fill quality
- Market impact estimates

Key Features:
- Multi-venue execution routing
- Liquidity-aware order placement
- Latency optimization
- Adaptive to changing market conditions
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
class VenuePerformance:
    """Tracks performance metrics for a single execution venue"""
    
    venue_name: str
    liquidity_score: float = 0.5
    fill_ratio: float = 0.8
    avg_slippage: float = 0.001  # 0.1%
    latency: float = 50.0  # milliseconds
    success_rate: float = 0.95
    
    # Historical performance by order type and size
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    
    def update_metrics(self, execution_report: Dict) -> None:
        """Update performance metrics from an execution report"""
        order_type = execution_report.get('order_type', 'market')
        order_size = execution_report.get('order_size', 'medium')
        
        # Update liquidity score based on fill ratio and slippage
        if 'fill_ratio' in execution_report:
            self.fill_ratio = 0.9 * self.fill_ratio + 0.1 * execution_report['fill_ratio']
        
        if 'slippage' in execution_report:
            self.avg_slippage = 0.9 * self.avg_slippage + 0.1 * execution_report['slippage']
        
        if 'latency' in execution_report:
            self.latency = 0.9 * self.latency + 0.1 * execution_report['latency']
        
        if 'success' in execution_report:
            self.success_rate = 0.9 * self.success_rate + 0.1 * (1 if execution_report['success'] else 0)
        
        # Update liquidity score (composite metric)
        self.liquidity_score = (
            0.4 * self.fill_ratio +
            0.3 * (1 - min(self.avg_slippage * 100, 1.0)) +
            0.2 * (1 - min(self.latency / 100, 1.0)) +
            0.1 * self.success_rate
        )
        
        # Store performance by order type and size
        key = f"{order_type}_{order_size}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        # Store execution quality score (higher is better)
        quality_score = (execution_report.get('fill_ratio', 0.8) * 
                        (1 - min(execution_report.get('slippage', 0.001) * 100, 1.0)) *
                        (1 if execution_report.get('success', True) else 0))
        self.performance_history[key].append(quality_score)
        
        # Keep only last 100 executions
        if len(self.performance_history[key]) > 100:
            self.performance_history[key].pop(0)
    
    def get_performance_score(self, order_type: str = 'market', order_size: str = 'medium') -> float:
        """Get performance score for a specific order type and size"""
        key = f"{order_type}_{order_size}"
        if key in self.performance_history and self.performance_history[key]:
            return np.mean(self.performance_history[key])
        return self.liquidity_score  # Fallback to general liquidity score


@dataclass
class OrderRoutingDecision:
    """Represents an order routing decision"""
    
    order_id: str
    venue: str
    reason: str
    expected_fill_ratio: float
    expected_slippage: float
    expected_latency: float
    confidence: float


class SmartOrderRouter(LearningComponent):
    """Dynamically routes orders to optimal execution venues"""
    
    def __init__(self):
        super().__init__(
            name="order_router",
            version="1.0",
            enabled=True,
            update_frequency=1  # Update every trade cycle
        )
        
        # Venue performance tracking
        self.venues: Dict[str, VenuePerformance] = {}
        self.available_venues: List[str] = []
        
        # Routing history
        self.routing_history: List[OrderRoutingDecision] = []
        self.max_history = 1000
        
        # Market regime tracking
        self.current_regime: str = "stable"
        self.regime_adjustments: Dict[str, Dict] = {
            'stable': {
                'liquidity_weight': 0.5,
                'latency_weight': 0.3,
                'slippage_weight': 0.2
            },
            'volatile': {
                'liquidity_weight': 0.6,
                'latency_weight': 0.2,
                'slippage_weight': 0.2
            },
            'trending': {
                'liquidity_weight': 0.4,
                'latency_weight': 0.3,
                'slippage_weight': 0.3
            },
            'range': {
                'liquidity_weight': 0.5,
                'latency_weight': 0.2,
                'slippage_weight': 0.3
            }
        }
        
        # Order type performance
        self.order_type_performance: Dict[str, Dict] = defaultdict(dict)
        
        # State tracking
        self.last_routing_decision: Optional[OrderRoutingDecision] = None
    
    def initialize_venues(self, venues: List[str]) -> None:
        """Initialize tracking for all execution venues"""
        self.available_venues = venues
        for venue in venues:
            if venue not in self.venues:
                self.venues[venue] = VenuePerformance(venue)
    
    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from new market data and execution reports"""
        
        # Update regime
        self._update_regime(data)
        
        # Update venue performance from execution reports
        if 'execution_reports' in data:
            self._update_venue_performance(data['execution_reports'])
        
        # Update market conditions if available
        if 'market_conditions' in data:
            self._update_market_conditions(data['market_conditions'])
        
        return {
            'routing_strategy': 'adaptive',
            'regime': self.current_regime,
            'venue_performance': {v: ven.get_performance_score() for v, ven in self.venues.items()}
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
    
    def _update_venue_performance(self, execution_reports: List[Dict]) -> None:
        """Update venue performance metrics from execution reports"""
        for report in execution_reports:
            venue = report.get('venue')
            if venue and venue in self.venues:
                self.venues[venue].update_metrics(report)
                
                # Update order type performance
                order_type = report.get('order_type', 'market')
                order_size = report.get('order_size', 'medium')
                key = f"{venue}_{order_type}_{order_size}"
                
                if 'fill_ratio' in report and 'slippage' in report:
                    self.order_type_performance[key] = {
                        'fill_ratio': report['fill_ratio'],
                        'slippage': report['slippage'],
                        'latency': report.get('latency', 50),
                        'success': report.get('success', True)
                    }
    
    def _update_market_conditions(self, market_conditions: Dict[str, Any]) -> None:
        """Update market conditions that affect routing"""
        # This could include things like:
        # - Overall market liquidity
        # - Exchange outages
        # - Latency measurements
        # - Fee changes
        pass
    
    def route_order(self, order: Dict[str, Any]) -> OrderRoutingDecision:
        """Determine the optimal venue to route an order to"""
        
        order_id = order.get('order_id', 'unknown')
        order_type = order.get('order_type', 'market')
        order_size = order.get('order_size', 'medium')
        symbol = order.get('symbol', 'BTCUSDT')
        
        if not self.available_venues:
            logger.warning("No venues available for routing")
            return OrderRoutingDecision(
                order_id=order_id,
                venue=self.available_venues[0] if self.available_venues else "default",
                reason="no_venues_available",
                expected_fill_ratio=0.8,
                expected_slippage=0.001,
                expected_latency=50,
                confidence=0.5
            )
        
        # Get regime-specific weights
        regime_weights = self.regime_adjustments.get(self.current_regime, 
                                                     self.regime_adjustments['stable'])
        
        # Score each venue
        venue_scores = {}
        for venue_name, venue in self.venues.items():
            # Get base performance score
            performance_score = venue.get_performance_score(order_type, order_size)
            
            # Get specific metrics
            liquidity = venue.liquidity_score
            latency = venue.latency
            slippage = venue.avg_slippage
            
            # Calculate weighted score based on regime
            score = (
                regime_weights['liquidity_weight'] * liquidity +
                regime_weights['latency_weight'] * (1 - min(latency/200, 1.0)) +
                regime_weights['slippage_weight'] * (1 - min(slippage*100, 1.0))
            )
            
            # Add performance score
            score = 0.7 * score + 0.3 * performance_score
            
            venue_scores[venue_name] = score
        
        # Select best venue
        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        best_score = venue_scores[best_venue]
        
        # Get expected metrics for the best venue
        venue = self.venues[best_venue]
        
        # Create routing decision
        decision = OrderRoutingDecision(
            order_id=order_id,
            venue=best_venue,
            reason=f"best_score_{self.current_regime}",
            expected_fill_ratio=venue.fill_ratio,
            expected_slippage=venue.avg_slippage,
            expected_latency=venue.latency,
            confidence=best_score
        )
        
        # Store decision in history
        self.routing_history.append(decision)
        if len(self.routing_history) > self.max_history:
            self.routing_history.pop(0)
        
        self.last_routing_decision = decision
        
        return decision
    
    def get_venue_performance(self, venue: str) -> Dict[str, Any]:
        """Get performance metrics for a specific venue"""
        if venue in self.venues:
            return {
                'liquidity_score': self.venues[venue].liquidity_score,
                'fill_ratio': self.venues[venue].fill_ratio,
                'avg_slippage': self.venues[venue].avg_slippage,
                'latency': self.venues[venue].latency,
                'success_rate': self.venues[venue].success_rate
            }
        return {}
    
    def get_routing_history(self, limit: int = 10) -> List[OrderRoutingDecision]:
        """Get recent routing decisions"""
        return self.routing_history[-limit:]
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameters"""
        return {
            'current_regime': self.current_regime,
            'regime_adjustments': self.regime_adjustments.copy(),
            'available_venues': self.available_venues.copy(),
            'venue_performance': {
                venue: {
                    'liquidity_score': ven.liquidity_score,
                    'fill_ratio': ven.fill_ratio,
                    'avg_slippage': ven.avg_slippage,
                    'latency': ven.latency
                }
                for venue, ven in self.venues.items()
            },
            'last_routing_decision': {
                'venue': self.last_routing_decision.venue,
                'confidence': self.last_routing_decision.confidence
            } if self.last_routing_decision else None
        }
    
    def rollback(self) -> None:
        """Revert to last known good state"""
        if len(self.routing_history) > 1:
            # Revert to previous routing decision
            self.last_routing_decision = self.routing_history[-2]
            logger.info(f"Rolled back to previous routing decision")
        else:
            self.last_routing_decision = None
            logger.warning("No routing history - cleared last decision")
    
    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes"""
        if 'regime_adjustments' in new_params:
            for regime, weights in new_params['regime_adjustments'].items():
                total_weight = sum(weights.values())
                if not (0.99 <= total_weight <= 1.01):
                    logger.warning(f"Regime weights for {regime} don't sum to 1: {total_weight:.3f}")
                    return False
                
                for weight in weights.values():
                    if weight < 0 or weight > 1:
                        logger.warning(f"Invalid weight in {regime}: {weight}")
                        return False
        
        if 'available_venues' in new_params:
            for venue in new_params['available_venues']:
                if venue not in self.venues:
                    logger.warning(f"Unknown venue in available_venues: {venue}")
                    return False
        
        return True
    
    def learn_from_execution(self, execution_report: Dict[str, Any]) -> None:
        """Specialized learning from individual execution reports"""
        venue = execution_report.get('venue')
        if venue and venue in self.venues:
            self.venues[venue].update_metrics(execution_report)
            
            # Update order type performance
            order_type = execution_report.get('order_type', 'market')
            order_size = execution_report.get('order_size', 'medium')
            key = f"{venue}_{order_type}_{order_size}"
            
            if 'fill_ratio' in execution_report and 'slippage' in execution_report:
                self.order_type_performance[key] = {
                    'fill_ratio': execution_report['fill_ratio'],
                    'slippage': execution_report['slippage'],
                    'latency': execution_report.get('latency', 50),
                    'success': execution_report.get('success', True)
                }