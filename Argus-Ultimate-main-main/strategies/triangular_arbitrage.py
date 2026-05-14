"""
Triangular Arbitrage Strategy — Argus Ultimate v15.0.0
======================================================

Exploits pricing inefficiencies in triangular currency paths.

HOW IT WORKS:
1. Monitor 3 currency pairs that form a triangle (e.g., BTC/ETH, ETH/USDT, BTC/USDT)
2. When the calculated cross rate differs from direct rate, profit exists
3. Execute the full circle: A -> B -> C -> A

EXAMPLE:
- BTC/ETH = 20 ETH per BTC
- ETH/USDT = 500 USDT per ETH
- BTC/USDT = 9950 USDT per BTC (direct)
- Calculated: 20 * 500 = 10,000 USDT per BTC
- Deviation: 10,000 - 9,950 = 50 USDT (0.5% profit)

EXPECTED PERFORMANCE:
- 0.05-0.3% per cycle
- Low risk (if executed atomically)
- High frequency possible

Author: Argus Ultimate
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrianglePath:
    """Represents a triangular currency path."""
    base: str      # Starting/ending currency
    mid1: str      # First intermediate
    mid2: str      # Second intermediate
    
    @property
    def path(self) -> Tuple[str, str, str]:
        return (self.base, self.mid1, self.mid2)


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""
    triangle: TrianglePath
    direction: str  # "forward" or "reverse"
    direct_rate: float        # Direct rate from base to mid2
    calculated_rate: float   # Cross rate through triangle
    deviation_pct: float     # Deviation percentage
    estimated_profit_pct: float
    min_amount: float
    confidence: float
    timestamp: datetime


@dataclass
class ArbitrageResult:
    """Result of an arbitrage execution."""
    opportunity: ArbitrageOpportunity
    execution_prices: List[float]
    amount_in: float
    amount_out: float
    profit: float
    fees: float
    net_profit: float
    success: bool
    execution_time_ms: float


class TriangularArbitrageStrategy:
    """
    Triangular Arbitrage Strategy.
    
    Exploits pricing inefficiencies between related currency pairs.
    
    Common Triangles:
    - BTC/ETH + ETH/USDT + BTC/USDT
    - BTC/DAI + DAI/USDT + BTC/USDT
    - ETH/USD + USD/JPY + ETH/JPY
    - Arbitrum/ETH + ETH/USDC + Arbitrum/USDC
    
    The strategy:
    1. Continuously monitors pair prices
    2. Calculates implied rates through triangles
    3. When deviation > threshold, executes full cycle
    4. Profits from price convergence
    """
    
    # Predefined high-liquidity triangles
    DEFAULT_TRIANGLES = [
        TrianglePath("BTC", "ETH", "USDT"),      # BTC -> ETH -> USDT -> BTC
        TrianglePath("ETH", "BTC", "USDT"),       # ETH -> BTC -> USDT -> ETH
        TrianglePath("ETH", "USDT", "BTC"),       # ETH -> USDT -> BTC -> ETH
        TrianglePath("USDT", "BTC", "ETH"),       # USDT -> BTC -> ETH -> USDT
        TrianglePath("USDT", "ETH", "BTC"),       # USDT -> ETH -> BTC -> USDT
        TrianglePath("BTC", "USDT", "ETH"),       # BTC -> USDT -> ETH -> BTC
    ]
    
    def __init__(
        self,
        min_deviation_pct: float = 0.1,
        min_amount_usd: float = 1000.0,
        max_amount_usd: float = 50000.0,
        fee_per_trade: float = 0.003,
        execution_timeout_ms: float = 5000.0,
        triangles: Optional[List[TrianglePath]] = None,
    ):
        """
        Initialize Triangular Arbitrage Strategy.
        
        Args:
            min_deviation_pct: Minimum deviation to attempt arbitrage
            min_amount_usd: Minimum trade size
            max_amount_usd: Maximum trade size
            fee_per_trade: Fee per trade (0.003 = 0.3%)
            execution_timeout_ms: Max execution time
            triangles: Custom triangle paths
        """
        self.min_deviation_pct = min_deviation_pct
        self.min_amount_usd = min_amount_usd
        self.max_amount_usd = max_amount_usd
        self.fee_per_trade = fee_per_trade
        self.execution_timeout_ms = execution_timeout_ms
        self.triangles = triangles or self.DEFAULT_TRIANGLES
        
        # State
        self._opportunities: Deque[ArbitrageOpportunity] = deque(maxlen=500)
        self._results: Deque[ArbitrageResult] = deque(maxlen=2000)
        self._current_prices: Dict[str, float] = {}
        self._last_scan: Optional[datetime] = None
        
        logger.info(
            "TriangularArbitrageStrategy initialized: %d triangles, min_dev=%.2f%%",
            len(self.triangles),
            min_deviation_pct,
        )
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current market prices.
        
        Args:
            prices: Dict of pair_name -> price
                    e.g., {"BTC/USDT": 60000, "ETH/USDT": 3000, "BTC/ETH": 20}
        """
        self._current_prices.update(prices)
        self._last_scan = datetime.now(timezone.utc)
    
    def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Scan all triangles for arbitrage opportunities.
        
        Returns:
            List of detected opportunities
        """
        opportunities = []
        
        for triangle in self.triangles:
            opp = self._check_triangle(triangle)
            if opp:
                opportunities.append(opp)
        
        return opportunities
    
    def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity,
        amount: Optional[float] = None,
    ) -> ArbitrageResult:
        """
        Execute an arbitrage opportunity.
        
        Args:
            opportunity: Detected opportunity
            amount: Amount in base currency (optional)
        
        Returns:
            ArbitrageResult with execution details
        """
        import time
        
        start_time = time.time()
        
        # Determine amount
        if amount is None:
            # Auto-size based on opportunity
            amount = min(
                self.max_amount_usd,
                max(self.min_amount_usd, opportunity.min_amount),
            )
        
        base = opportunity.triangle.base
        mid1 = opportunity.triangle.mid1
        mid2 = opportunity.triangle.mid2
        
        # Calculate execution prices (simulated)
        execution_prices = []
        
        if opportunity.direction == "forward":
            # Base -> Mid1 -> Mid2 -> Base
            p1 = self._get_price(f"{base}/{mid1}")
            p2 = self._get_price(f"{mid1}/{mid2}")
            p3 = self._get_price(f"{mid2}/{base}")
            
            execution_prices = [p1, p2, p3]
            
            # Step 1: Base -> Mid1
            step1_out = amount / p1 if p1 > 0 else 0
            
            # Step 2: Mid1 -> Mid2
            step2_out = step1_out / p2 if p2 > 0 else 0
            
            # Step 3: Mid2 -> Base
            step3_out = step2_out * p3 if p3 > 0 else 0
            
            amount_out = step3_out
        else:
            # Reverse direction
            p1 = self._get_price(f"{base}/{mid2}")
            p2 = self._get_price(f"{mid2}/{mid1}")
            p3 = self._get_price(f"{mid1}/{base}")
            
            execution_prices = [p1, p2, p3]
            
            step1_out = amount / p1 if p1 > 0 else 0
            step2_out = step1_out / p2 if p2 > 0 else 0
            step3_out = step2_out * p3 if p3 > 0 else 0
            
            amount_out = step3_out
        
        # Calculate profits
        profit = amount_out - amount
        fees = amount * (3 * self.fee_per_trade)  # 3 trades
        net_profit = profit - fees
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        result = ArbitrageResult(
            opportunity=opportunity,
            execution_prices=execution_prices,
            amount_in=amount,
            amount_out=amount_out,
            profit=profit,
            fees=fees,
            net_profit=net_profit,
            success=net_profit > 0,
            execution_time_ms=execution_time_ms,
        )
        
        self._results.append(result)
        
        logger.info(
            "Arbitrage executed: %s %s, profit=%.2f, net=%.2f, success=%s",
            opportunity.direction,
            opportunity.triangle.path,
            profit,
            net_profit,
            result.success,
        )
        
        return result
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        if not self._results:
            return {
                "total_cycles": 0,
                "profitable_cycles": 0,
                "failed_cycles": 0,
                "total_profit": 0.0,
                "total_fees": 0.0,
                "net_profit": 0.0,
                "win_rate": 0.0,
            }
        
        total = len(self._results)
        profitable = sum(1 for r in self._results if r.success)
        total_profit = sum(r.profit for r in self._results)
        total_fees = sum(r.fees for r in self._results)
        net_profit = sum(r.net_profit for r in self._results)
        
        return {
            "total_cycles": total,
            "profitable_cycles": profitable,
            "failed_cycles": total - profitable,
            "total_profit": total_profit,
            "total_fees": total_fees,
            "net_profit": net_profit,
            "win_rate": profitable / total,
            "avg_profit_per_cycle": total_profit / total,
            "avg_execution_ms": sum(r.execution_time_ms for r in self._results) / total,
            "active_triangles": len(self.triangles),
        }
    
    def get_best_opportunity(self) -> Optional[ArbitrageOpportunity]:
        """Get the best current opportunity."""
        opportunities = self.scan_opportunities()
        if not opportunities:
            return None
        
        return max(opportunities, key=lambda x: x.estimated_profit_pct)
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _check_triangle(self, triangle: TrianglePath) -> Optional[ArbitrageOpportunity]:
        """Check a single triangle for opportunities."""
        base, mid1, mid2 = triangle.path
        
        # Get prices
        # Format: "BASE/QUOTE"
        p_base_mid1 = self._get_price(f"{base}/{mid1}")  # How much mid1 per base
        p_mid1_mid2 = self._get_price(f"{mid1}/{mid2}")  # How much mid2 per mid1
        p_mid2_base = self._get_price(f"{mid2}/{base}")  # How much base per mid2
        
        if not all([p_base_mid1, p_mid1_mid2, p_mid2_base]):
            return None
        
        # Calculate implied rate (forward direction)
        # base -> mid1 -> mid2 -> base
        # 1 base * p_base_mid1 * p_mid1_mid2 * p_mid2_base = final base
        # For profit: result > 1
        
        # Direct route: base -> mid2
        # Cross route: base -> mid1 -> mid2
        
        # Calculate cross rate for mid1 -> mid2 path
        # p_mid2_base is in terms of mid2 -> base
        # So 1/mid2_base = base -> mid2
        
        # Forward path: base -> mid1 -> mid2
        # Amount of mid2 you get: amount * p_base_mid1 * p_mid1_mid2
        
        # Reverse path: base -> mid2 -> mid1
        # Amount of base you get: amount / p_direct * p_mid2_base * p_mid1_mid2
        
        # Check forward opportunity: base -> mid1 -> mid2 -> base
        cross_rate_forward = p_base_mid1 * p_mid1_mid2 * p_mid2_base
        
        # Check reverse opportunity: base -> mid2 -> mid1 -> base
        # Need direct base -> mid2 rate
        p_base_mid2 = self._get_price(f"{base}/{mid2}")
        if p_base_mid2:
            cross_rate_reverse = (p_base_mid2 / p_mid2_base) * p_mid1_mid2 * (1 / p_base_mid1)
        else:
            cross_rate_reverse = 1.0
        
        # Calculate deviations
        dev_forward = (cross_rate_forward - 1.0) * 100
        dev_reverse = (cross_rate_reverse - 1.0) * 100
        
        # Determine best direction
        if abs(dev_forward) > abs(dev_reverse) and abs(dev_forward) > self.min_deviation_pct:
            # Forward is better
            deviation = dev_forward
            direction = "forward"
            direct_rate = 1.0  # We start and end with base
            calculated_rate = cross_rate_forward
            min_amount = self.min_amount_usd / self._get_price(f"{base}/USDT") if self._get_price(f"{base}/USDT") else self.min_amount_usd
        elif abs(dev_reverse) > self.min_deviation_pct:
            # Reverse is better
            deviation = dev_reverse
            direction = "reverse"
            direct_rate = 1.0
            calculated_rate = cross_rate_reverse
            min_amount = self.min_amount_usd / self._get_price(f"{base}/USDT") if self._get_price(f"{base}/USDT") else self.min_amount_usd
        else:
            return None
        
        opportunity = ArbitrageOpportunity(
            triangle=triangle,
            direction=direction,
            direct_rate=direct_rate,
            calculated_rate=calculated_rate,
            deviation_pct=deviation,
            estimated_profit_pct=abs(deviation) - (3 * self.fee_per_trade * 100),
            min_amount=min_amount,
            confidence=min(abs(deviation) / 1.0, 1.0),
            timestamp=datetime.now(timezone.utc),
        )
        
        if opportunity.estimated_profit_pct > 0:
            self._opportunities.append(opportunity)
            logger.debug(
                "Triangle opportunity: %s %s, deviation=%.3f%%",
                direction,
                triangle.path,
                deviation,
            )
        
        return opportunity if opportunity.estimated_profit_pct > 0 else None
    
    def _get_price(self, pair: str) -> Optional[float]:
        """Get price for a pair."""
        # Try exact match
        if pair in self._current_prices:
            return self._current_prices[pair]
        
        # Try reversed
        parts = pair.split("/")
        if len(parts) == 2:
            reversed_pair = f"{parts[1]}/{parts[0]}"
            if reversed_pair in self._current_prices:
                price = self._current_prices[reversed_pair]
                return 1.0 / price if price > 0 else None
        
        return None


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_triangular_arbitrage_strategy(
    min_deviation_pct: float = 0.1,
    min_amount_usd: float = 1000.0,
    max_amount_usd: float = 50000.0,
) -> TriangularArbitrageStrategy:
    """Factory to create configured TriangularArbitrageStrategy."""
    return TriangularArbitrageStrategy(
        min_deviation_pct=min_deviation_pct,
        min_amount_usd=min_amount_usd,
        max_amount_usd=max_amount_usd,
    )