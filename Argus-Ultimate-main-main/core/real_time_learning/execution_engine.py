"""
Smart Execution Engine - Adaptive Execution Component

Adapts execution parameters based on:
- Order book liquidity
- Market volatility
- Exchange latency
- Trade urgency
- Historical fill quality

Optimizes:
- Participation rate (TWAP/VWAP/POV)
- Aggressiveness (passive vs. aggressive)
- Order slicing (number of slices, slice size)
- Latency buffers (exchange-specific)
- Execution algorithms
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
from .orchestrator import LearningComponent


class SmartExecutionEngine(LearningComponent):
    """
    Adaptively optimizes execution parameters to minimize slippage and maximize fill quality.
    
    Key Adaptations:
    - Participation rate (TWAP/VWAP/POV)
    - Aggressiveness (passive vs. aggressive)
    - Order slicing (number of slices, slice size)
    - Latency buffers (exchange-specific)
    - Algorithm selection (TWAP, VWAP, POV, etc.)
    """

    def __init__(self):
        super().__init__(
            name="execution",
            version="1.0",
            update_frequency=1  # Update every cycle for execution-critical component
        )

        # Default execution parameters (conservative)
        self.params = {
            "base_participation": 0.20,      # 20% of volume
            "aggressiveness": 0.5,          # 0-1 scale (0=passive, 1=aggressive)
            "min_slice_size": 0.05,         # 5% of total order as minimum slice
            "max_slice_size": 0.20,         # 20% of total order as maximum slice
            "latency_buffer_ms": 10,        # 10ms latency buffer
            "algorithm_weights": {          # Algorithm selection probabilities
                "TWAP": 0.4,
                "VWAP": 0.3,
                "POV": 0.2,
                "Iceberg": 0.1
            },
            "exchange_latency": {            # Exchange-specific latency estimates
                "binance": 8,
                "bybit": 12,
                "okx": 10
            },
            "liquidity_thresholds": {       # Liquidity-based adaptation
                "low": 500000,              # USD volume
                "medium": 2000000,
                "high": 5000000
            }
        }

        # Execution performance tracking
        self.execution_history = []
        self.max_history = 1000
        self.current_exchange = "binance"  # Default exchange

        # Performance metrics
        self.avg_slippage = 0.0005  # 0.05% initial slippage estimate
        self.fill_ratio = 0.95      # 95% initial fill ratio estimate

    def learn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from market data and update execution parameters
        
        Args:
            data: Dictionary containing:
                - order_book: Dict (bids/asks)
                - recent_trades: List[Dict] (recent executions)
                - exchange: str (exchange name)
                - volatility: float (current volatility)
                - volume: float (current volume)
                - timestamp: datetime
        
        Returns:
            Updated parameters dictionary
        """
        # Extract market data
        order_book = data.get("order_book", {})
        recent_trades = data.get("recent_trades", [])
        exchange = data.get("exchange", self.current_exchange)
        volatility = data.get("volatility", 0.01)
        volume = data.get("volume", 1000000)
        timestamp = data.get("timestamp", datetime.now(timezone.utc))

        # Update current exchange
        self.current_exchange = exchange

        # Store execution data for learning
        self._store_execution_data(recent_trades, timestamp)

        # Only adapt if we have enough history
        if len(self.execution_history) < 10:
            self.last_updated = datetime.now(timezone.utc)
            return self.params

        # Calculate market liquidity
        liquidity = self._calculate_liquidity(order_book, volume)

        # Calculate current performance metrics
        self._update_performance_metrics()

        # Determine adaptation factors
        factors = self._calculate_adaptation_factors(liquidity, volatility)

        # Apply adaptations with bounds checking
        self.params = self._apply_adaptations(factors, exchange)

        self.last_updated = datetime.now(timezone.utc)
        return self.params

    def _store_execution_data(self, trades: List[Dict], timestamp: datetime):
        """Store execution data for analysis"""
        for trade in trades:
            self.execution_history.append({
                "timestamp": timestamp,
                "symbol": trade.get("symbol", ""),
                "size": trade.get("size", 0),
                "slippage": trade.get("slippage", 0),
                "fill_ratio": trade.get("fill_ratio", 1.0),
                "execution_time_ms": trade.get("execution_time_ms", 10),
                "algorithm": trade.get("algorithm", "TWAP"),
                "exchange": trade.get("exchange", self.current_exchange)
            })

        # Keep history bounded
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]

    def _calculate_liquidity(self, order_book: Dict, volume: float) -> str:
        """Calculate liquidity regime (low/medium/high)"""
        if not order_book or "bids" not in order_book or "asks" not in order_book:
            return "medium"

        # Calculate order book depth (USD)
        bid_depth = sum(qty * price for price, qty in order_book["bids"])
        ask_depth = sum(qty * price for price, qty in order_book["asks"])
        total_depth = bid_depth + ask_depth

        # Classify liquidity
        if total_depth < self.params["liquidity_thresholds"]["low"]:
            return "low"
        elif total_depth > self.params["liquidity_thresholds"]["high"]:
            return "high"
        else:
            return "medium"

    def _update_performance_metrics(self):
        """Update slippage and fill ratio estimates"""
        if not self.execution_history:
            return

        # Calculate recent slippage (last 100 trades)
        recent_trades = self.execution_history[-100:]
        if recent_trades:
            slippages = [t["slippage"] for t in recent_trades if "slippage" in t and t["slippage"] is not None]
            if slippages:
                self.avg_slippage = np.mean(slippages)

            fill_ratios = [t["fill_ratio"] for t in recent_trades if "fill_ratio" in t and t["fill_ratio"] is not None]
            if fill_ratios:
                self.fill_ratio = np.mean(fill_ratios)

    def _calculate_adaptation_factors(self, liquidity: str, volatility: float) -> Dict[str, float]:
        """
        Calculate adaptation factors based on current market conditions
        
        Args:
            liquidity: Liquidity regime (low/medium/high)
            volatility: Current market volatility
        
        Returns:
            Dictionary of adaptation factors
        """
        factors = {}

        # 1. Liquidity-based adaptations
        if liquidity == "low":
            factors.update({
                "base_participation": -0.05,  # Reduce participation in thin markets
                "aggressiveness": -0.1,       # Be more passive
                "min_slice_size": -0.02,      # Smaller slices
                "latency_buffer_ms": 5        # Tighter latency buffer
            })
        elif liquidity == "high":
            factors.update({
                "base_participation": 0.05,   # Can participate more in deep markets
                "aggressiveness": 0.1,        # Can be more aggressive
                "min_slice_size": 0.02,       # Larger slices
                "latency_buffer_ms": -2       # Can tolerate more latency
            })

        # 2. Volatility-based adaptations
        if volatility > 0.03:  # High volatility
            factors.update({
                "base_participation": -0.1,   # Reduce participation
                "aggressiveness": 0.1,        # Need to be more aggressive to get fills
                "algorithm_weights": {
                    "TWAP": -0.1,              # Less TWAP
                    "POV": 0.1,                # More POV
                    "Iceberg": 0.05            # More Iceberg
                }
            })
        elif volatility < 0.01:  # Low volatility
            factors.update({
                "base_participation": 0.05,   # Can participate more
                "aggressiveness": -0.1,       # Can be more passive
                "algorithm_weights": {
                    "TWAP": 0.1,               # More TWAP
                    "VWAP": 0.05,              # More VWAP
                    "POV": -0.05               # Less POV
                }
            })

        # 3. Performance-based adaptations
        if self.avg_slippage > 0.001:  # High slippage
            factors.update({
                "aggressiveness": -0.1,       # Be more passive to reduce slippage
                "base_participation": -0.05,  # Reduce participation
                "min_slice_size": -0.03       # Smaller slices
            })

        if self.fill_ratio < 0.9:  # Poor fill ratio
            factors.update({
                "aggressiveness": 0.1,        # Be more aggressive to improve fills
                "latency_buffer_ms": 2        # Increase latency buffer
            })

        # 4. Exchange-specific adaptations
        exchange_latency = self.params["exchange_latency"].get(self.current_exchange, 10)
        if exchange_latency > 12:  # High latency exchange
            factors.update({
                "latency_buffer_ms": 5,       # Increase buffer
                "aggressiveness": -0.1        # Be more passive
            })

        return factors

    def _apply_adaptations(self, factors: Dict[str, float], exchange: str) -> Dict[str, float]:
        """
        Apply adaptations with bounds checking
        
        Args:
            factors: Dictionary of adaptation factors
            exchange: Current exchange
        
        Returns:
            Updated parameters dictionary
        """
        new_params = self.params.copy()

        # Apply numeric parameter adaptations
        for param in ["base_participation", "aggressiveness", "min_slice_size", "max_slice_size"]:
            if param in factors:
                new_params[param] = max(0.01, min(1.0, new_params[param] + factors[param]))

        # Apply latency buffer adaptation
        if "latency_buffer_ms" in factors:
            new_params["latency_buffer_ms"] = max(5, min(50, new_params["latency_buffer_ms"] + factors["latency_buffer_ms"]))

        # Apply algorithm weight adaptations
        if "algorithm_weights" in factors:
            for algo, change in factors["algorithm_weights"].items():
                if algo in new_params["algorithm_weights"]:
                    new_val = max(0.05, min(0.9, new_params["algorithm_weights"][algo] + change))
                    new_params["algorithm_weights"][algo] = new_val

            # Renormalize weights
            total = sum(new_params["algorithm_weights"].values())
            for algo in new_params["algorithm_weights"]:
                new_params["algorithm_weights"][algo] /= total

        # Apply bounds checking
        new_params["base_participation"] = max(0.05, min(0.5, new_params["base_participation"]))
        new_params["aggressiveness"] = max(0.1, min(0.9, new_params["aggressiveness"]))
        new_params["min_slice_size"] = max(0.01, min(0.3, new_params["min_slice_size"]))
        new_params["max_slice_size"] = max(0.05, min(0.5, new_params["max_slice_size"]))
        new_params["latency_buffer_ms"] = max(5, min(50, new_params["latency_buffer_ms"]))

        # Update exchange-specific latency if needed
        if exchange in new_params["exchange_latency"]:
            # Update with 10% of observed latency (EWMA)
            observed_latency = self._calculate_observed_latency()
            if observed_latency:
                new_params["exchange_latency"][exchange] = (
                    0.9 * new_params["exchange_latency"][exchange] +
                    0.1 * observed_latency
                )

        return new_params

    def _calculate_observed_latency(self) -> Optional[float]:
        """Calculate observed exchange latency from execution history"""
        if not self.execution_history:
            return None

        # Get recent executions for current exchange
        recent = [
            t for t in self.execution_history[-100:]
            if t.get("exchange") == self.current_exchange and "execution_time_ms" in t
        ]

        if not recent:
            return None

        # Return 90th percentile of execution times as latency estimate
        latencies = [t["execution_time_ms"] for t in recent]
        return float(np.percentile(latencies, 90))

    def get_params(self) -> Dict[str, Any]:
        """Get current parameters with performance metrics"""
        return {
            **self.params,
            "current_exchange": self.current_exchange,
            "avg_slippage": self.avg_slippage,
            "fill_ratio": self.fill_ratio,
            "last_updated": self.last_updated
        }

    def rollback(self) -> None:
        """Revert to conservative defaults"""
        self.params = {
            "base_participation": 0.20,
            "aggressiveness": 0.5,
            "min_slice_size": 0.05,
            "max_slice_size": 0.20,
            "latency_buffer_ms": 10,
            "algorithm_weights": {
                "TWAP": 0.4,
                "VWAP": 0.3,
                "POV": 0.2,
                "Iceberg": 0.1
            },
            "exchange_latency": {
                "binance": 8,
                "bybit": 12,
                "okx": 10
            },
            "liquidity_thresholds": {
                "low": 500000,
                "medium": 2000000,
                "high": 5000000
            }
        }
        self.avg_slippage = 0.0005
        self.fill_ratio = 0.95

    def validate(self, new_params: Dict[str, Any]) -> bool:
        """Validate proposed parameter changes"""
        # Check participation rate bounds
        if new_params["base_participation"] < 0.05 or new_params["base_participation"] > 0.5:
            return False

        # Check aggressiveness bounds
        if new_params["aggressiveness"] < 0.1 or new_params["aggressiveness"] > 0.9:
            return False

        # Check slice size bounds
        if new_params["min_slice_size"] < 0.01 or new_params["min_slice_size"] > new_params["max_slice_size"]:
            return False

        if new_params["max_slice_size"] < 0.05 or new_params["max_slice_size"] > 0.5:
            return False

        # Check latency buffer bounds
        if new_params["latency_buffer_ms"] < 5 or new_params["latency_buffer_ms"] > 50:
            return False

        # Check algorithm weights sum to ~1
        if not (0.99 <= sum(new_params["algorithm_weights"].values()) <= 1.01):
            return False

        return True

    def learn_from_trade(self, trade: Dict[str, Any]) -> None:
        """
        Learn from individual trade execution data
        
        Args:
            trade: Dictionary containing:
                - symbol: str
                - size: float
                - slippage: float
                - fill_ratio: float (0-1)
                - execution_time_ms: float
                - algorithm: str
                - exchange: str
                - timestamp: datetime
        """
        # Store the trade execution data
        self._store_execution_data([trade], trade.get("timestamp", datetime.now(timezone.utc)))

        # Update performance metrics immediately for critical execution component
        self._update_performance_metrics()

        # If this was a problematic execution, trigger immediate adaptation
        if trade.get("slippage", 0) > 0.002 or trade.get("fill_ratio", 1.0) < 0.8:
            # Get current market data (simplified for this example)
            market_data = {
                "order_book": {},  # Would be real order book in production
                "recent_trades": [trade],
                "exchange": trade.get("exchange", self.current_exchange),
                "volatility": 0.01,  # Would be real volatility
                "volume": 1000000,   # Would be real volume
                "timestamp": trade.get("timestamp", datetime.now(timezone.utc))
            }

            # Run learning cycle with this problematic trade
            self.learn(market_data)

    def _restore_state(self, state: Dict[str, Any]) -> None:
        """Restore component state from saved data"""
        self.params = state["params"]
        self.current_exchange = state.get("current_exchange", "binance")
        self.avg_slippage = state.get("avg_slippage", 0.0005)
        self.fill_ratio = state.get("fill_ratio", 0.95)
        if "last_updated" in state:
            self.last_updated = datetime.fromisoformat(state["last_updated"])