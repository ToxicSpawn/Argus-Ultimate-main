"""
ALLOCATION OPTIMIZER - Adaptive Weight Tuning
==============================================
Analyzes edge performance and optimizes capital allocation.
Adjusts weights based on live results for maximum returns.
"""
import sys
sys.path.insert(0, '.')
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class EdgePerformance:
    """Performance metrics for a single edge."""
    name: str
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def expected_edge(self) -> float:
        """Calculate expected edge (risk-adjusted)."""
        if self.trades < 10:
            return 0.0
        return self.sharpe * 0.5 + self.win_rate * 0.3 + (1 - abs(self.max_drawdown)) * 0.2


@dataclass
class AllocationResult:
    """Optimized allocation result."""
    edge_name: str
    current_weight: float
    suggested_weight: float
    expected_return: float
    risk_score: float
    confidence: float
    reason: str


class AllocationOptimizer:
    """
    Optimizes capital allocation across trading edges.
    
    Uses:
    - Modern Portfolio Theory (MPT)
    - Kelly Criterion blending
    - Risk parity principles
    - Adaptive weight adjustment
    """
    
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Edge performance tracking
        self.edge_performance: Dict[str, EdgePerformance] = {}
        
        # Default edge configurations
        self.default_edges = {
            "funding_rate_arb": {
                "weight": 0.30,
                "expected_return": 0.15,
                "risk": 0.05,
                "sharpe": 2.5,
                "type": "risk_free"
            },
            "ml_momentum": {
                "weight": 0.25,
                "expected_return": 0.25,
                "risk": 0.15,
                "sharpe": 1.8,
                "type": "directional"
            },
            "market_making": {
                "weight": 0.20,
                "expected_return": 0.20,
                "risk": 0.10,
                "sharpe": 2.0,
                "type": "market_neutral"
            },
            "cross_exchange_arb": {
                "weight": 0.10,
                "expected_return": 0.10,
                "risk": 0.03,
                "sharpe": 3.0,
                "type": "risk_free"
            },
            "volatility_trading": {
                "weight": 0.10,
                "expected_return": 0.18,
                "risk": 0.12,
                "sharpe": 1.5,
                "type": "volatility"
            },
            "liquidation_hunting": {
                "weight": 0.05,
                "expected_return": 0.30,
                "risk": 0.20,
                "sharpe": 1.2,
                "type": "event"
            }
        }
        
        # Performance history
        self.allocation_history: deque = deque(maxlen=100)
        
        logger.info(f"AllocationOptimizer initialized: ${initial_capital}")
    
    def update_edge_performance(
        self,
        edge_name: str,
        pnl: float,
        return_pct: float,
        is_win: bool
    ):
        """Update performance metrics for an edge."""
        if edge_name not in self.edge_performance:
            self.edge_performance[edge_name] = EdgePerformance(name=edge_name)
        
        edge = self.edge_performance[edge_name]
        edge.trades += 1
        edge.total_pnl += pnl
        edge.total_return_pct += return_pct
        edge.last_updated = datetime.now()
        
        if is_win:
            edge.wins += 1
            edge.avg_win = (edge.avg_win * (edge.wins - 1) + return_pct) / edge.wins
        else:
            edge.avg_loss = (edge.avg_loss * (edge.trades - edge.wins) + abs(return_pct)) / (edge.trades - edge.wins)
        
        edge.win_rate = edge.wins / edge.trades if edge.trades > 0 else 0
        
        # Calculate profit factor
        if edge.avg_loss > 0:
            edge.profit_factor = (edge.win_rate * edge.avg_win) / ((1 - edge.win_rate) * edge.avg_loss)
        
        # Update drawdown
        if edge.total_return_pct < 0:
            edge.max_drawdown = min(edge.max_drawdown, edge.total_return_pct)
    
    def optimize_allocation(
        self,
        risk_tolerance: str = "moderate"
    ) -> List[AllocationResult]:
        """
        Optimize capital allocation across edges.
        
        Risk profiles:
        - conservative: Favor risk-free edges
        - moderate: Balanced approach
        - aggressive: Favor high-conviction directional
        """
        risk_multipliers = {
            "conservative": {"risk_free": 1.5, "market_neutral": 1.2, "directional": 0.5, "volatility": 0.6, "event": 0.3},
            "moderate": {"risk_free": 1.2, "market_neutral": 1.0, "directional": 1.0, "volatility": 0.8, "event": 0.5},
            "aggressive": {"risk_free": 0.8, "market_neutral": 0.8, "directional": 1.5, "volatility": 1.2, "event": 1.0}
        }
        
        multipliers = risk_multipliers.get(risk_tolerance, risk_multipliers["moderate"])
        
        results = []
        total_score = 0
        
        for edge_name, config in self.default_edges.items():
            # Get live performance if available
            perf = self.edge_performance.get(edge_name)
            
            # Calculate base score
            base_score = config["weight"]
            
            # Apply risk tolerance multiplier
            type_multiplier = multipliers.get(config["type"], 1.0)
            
            # Performance adjustment (if we have data)
            perf_multiplier = 1.0
            if perf and perf.trades >= 10:
                # Reward profitable edges, penalize losing ones
                if perf.profit_factor > 1.5:
                    perf_multiplier = 1.3
                elif perf.profit_factor > 1.0:
                    perf_multiplier = 1.1
                elif perf.profit_factor < 0.7:
                    perf_multiplier = 0.5
                elif perf.profit_factor < 1.0:
                    perf_multiplier = 0.8
            
            # Sharpe adjustment
            sharpe_multiplier = min(config["sharpe"] / 1.5, 2.0)
            
            # Calculate adjusted score
            adjusted_score = base_score * type_multiplier * perf_multiplier * sharpe_multiplier
            total_score += adjusted_score
            
            results.append({
                "edge_name": edge_name,
                "base_score": base_score,
                "type_multiplier": type_multiplier,
                "perf_multiplier": perf_multiplier,
                "sharpe_multiplier": sharpe_multiplier,
                "adjusted_score": adjusted_score,
                "config": config,
                "performance": perf
            })
        
        # Normalize to weights
        allocations = []
        for item in results:
            weight = item["adjusted_score"] / total_score if total_score > 0 else item["base_score"]
            
            # Calculate dollar allocation
            allocation_usd = self.current_capital * weight
            
            # Risk score (0-1, lower is better)
            risk_score = item["config"]["risk"] * (1 / item["config"]["sharpe"])
            
            # Generate reason
            reason = self._generate_reason(item, weight)
            
            allocations.append(AllocationResult(
                edge_name=item["edge_name"],
                current_weight=item["config"]["weight"],
                suggested_weight=weight,
                expected_return=item["config"]["expected_return"] * weight,
                risk_score=risk_score,
                confidence=min(item["config"]["sharpe"] / 2, 1.0),
                reason=reason
            ))
        
        # Sort by suggested weight
        allocations.sort(key=lambda x: x.suggested_weight, reverse=True)
        
        # Store in history
        self.allocation_history.append({
            "timestamp": datetime.now(),
            "risk_tolerance": risk_tolerance,
            "allocations": {a.edge_name: a.suggested_weight for a in allocations}
        })
        
        return allocations
    
    def _generate_reason(self, item: Dict, weight: float) -> str:
        """Generate human-readable reason for allocation."""
        edge_name = item["edge_name"]
        config = item["config"]
        perf = item.get("performance")
        
        reasons = []
        
        # Type-based reasoning
        if config["type"] == "risk_free":
            reasons.append("Risk-free income")
        elif config["sharpe"] > 2.0:
            reasons.append(f"High Sharpe ({config['sharpe']:.1f})")
        
        # Performance-based reasoning
        if perf and perf.trades >= 10:
            if perf.profit_factor > 1.5:
                reasons.append(f"Strong performance (PF: {perf.profit_factor:.2f})")
            elif perf.profit_factor < 0.8:
                reasons.append(f"Underperforming (PF: {perf.profit_factor:.2f})")
        
        return "; ".join(reasons) if reasons else "Base allocation"
    
    def calculate_kelly_allocation(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        edge_name: str
    ) -> float:
        """
        Calculate Kelly-optimal position size.
        
        Kelly Formula: f* = (p * b - q) / b
        Where:
        - p = win probability
        - q = loss probability (1 - p)
        - b = win/loss ratio
        """
        if avg_loss == 0:
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        # Use half-Kelly for safety
        half_kelly = kelly * 0.5
        
        # Cap at reasonable limits
        return max(0, min(half_kelly, 0.25))  # Max 25% per edge
    
    def get_risk_parity_allocation(self) -> Dict[str, float]:
        """
        Calculate risk parity allocation.
        
        Each edge contributes equal risk, not equal capital.
        """
        risk_contributions = {}
        
        for edge_name, config in self.default_edges.items():
            # Risk contribution = weight * volatility
            risk = config["weight"] * config["risk"]
            risk_contributions[edge_name] = risk
        
        # Normalize to equal risk
        total_risk = sum(risk_contributions.values())
        if total_risk == 0:
            return {name: 1/len(risk_contributions) for name in risk_contributions}
        
        # Inverse risk weighting
        risk_parity = {}
        for edge_name, risk in risk_contributions.items():
            if risk > 0:
                risk_parity[edge_name] = (1 / risk) / sum(1/r for r in risk_contributions.values() if r > 0)
            else:
                risk_parity[edge_name] = 1 / len(risk_contributions)
        
        return risk_parity
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive allocation report."""
        # Get allocations for different risk profiles
        conservative = self.optimize_allocation("conservative")
        moderate = self.optimize_allocation("moderate")
        aggressive = self.optimize_allocation("aggressive")
        
        # Get risk parity allocation
        risk_parity = self.get_risk_parity_allocation()
        
        # Calculate expected returns
        expected_returns = {}
        for profile, allocs in [("conservative", conservative), ("moderate", moderate), ("aggressive", aggressive)]:
            total_return = sum(a.expected_return for a in allocs)
            total_risk = sum(a.risk_score * a.suggested_weight for a in allocs)
            expected_returns[profile] = {
                "expected_return": total_return,
                "expected_risk": total_risk,
                "sharpe_estimate": total_return / total_risk if total_risk > 0 else 0
            }
        
        return {
            "capital": self.current_capital,
            "allocations": {
                "conservative": [(a.edge_name, f"{a.suggested_weight:.1%}") for a in conservative],
                "moderate": [(a.edge_name, f"{a.suggested_weight:.1%}") for a in moderate],
                "aggressive": [(a.edge_name, f"{a.suggested_weight:.1%}") for a in aggressive]
            },
            "risk_parity": {k: f"{v:.1%}" for k, v in risk_parity.items()},
            "expected_returns": expected_returns,
            "edge_count": len(self.default_edges),
            "performance_tracked": len(self.edge_performance)
        }


def print_allocation_report(optimizer: AllocationOptimizer):
    """Print formatted allocation report."""
    report = optimizer.generate_report()
    
    print("="*70)
    print("ALLOCATION OPTIMIZATION REPORT")
    print("="*70)
    
    print(f"\nCapital: ${report['capital']:,.2f}")
    print(f"Edges Tracked: {report['edge_count']}")
    print(f"Performance Data: {report['performance_tracked']} edges")
    
    print(f"\n{'='*70}")
    print("OPTIMAL ALLOCATIONS BY RISK PROFILE")
    print(f"{'='*70}")
    
    for profile in ["conservative", "moderate", "aggressive"]:
        allocs = report["allocations"][profile]
        metrics = report["expected_returns"][profile]
        
        print(f"\n{profile.upper()}:")
        print(f"  Expected Return: {metrics['expected_return']:.1%}")
        print(f"  Expected Risk: {metrics['expected_risk']:.1%}")
        print(f"  Sharpe Estimate: {metrics['sharpe_estimate']:.2f}")
        print(f"  Allocation:")
        for edge, weight in allocs:
            print(f"    {edge:25s}: {weight}")
    
    print(f"\n{'='*70}")
    print("RISK PARITY ALLOCATION")
    print(f"{'='*70}")
    for edge, weight in report["risk_parity"].items():
        print(f"  {edge:25s}: {weight}")
    
    print(f"\n{'='*70}")
    print("RECOMMENDED: MODERATE PROFILE")
    print(f"{'='*70}")
    moderate = report["allocations"]["moderate"]
    metrics = report["expected_returns"]["moderate"]
    print(f"  Expected Annual Return: {metrics['expected_return']:.1%}")
    print(f"  Expected Monthly Return: {metrics['expected_return']/12:.1%}")
    print(f"  Monthly ${report['capital']*metrics['expected_return']/12:,.2f} on ${report['capital']:,.2f}")
    print("="*70)


if __name__ == "__main__":
    # Demo with $1K capital
    optimizer = AllocationOptimizer(initial_capital=1000.0)
    
    # Simulate some performance data
    optimizer.update_edge_performance("funding_rate_arb", 50.0, 5.0, True)
    optimizer.update_edge_performance("funding_rate_arb", 45.0, 4.5, True)
    optimizer.update_edge_performance("funding_rate_arb", 52.0, 5.2, True)
    optimizer.update_edge_performance("ml_momentum", 100.0, 10.0, True)
    optimizer.update_edge_performance("ml_momentum", -30.0, -3.0, False)
    optimizer.update_edge_performance("ml_momentum", 80.0, 8.0, True)
    optimizer.update_edge_performance("market_making", 25.0, 2.5, True)
    optimizer.update_edge_performance("market_making", 30.0, 3.0, True)
    
    # Print report
    print_allocation_report(optimizer)
