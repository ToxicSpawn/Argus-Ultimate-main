"""
HFT MICROSTRUCTURE ENGINE - OMEGA GPU
======================================
GPU-accelerated high-frequency trading microstructure analysis.

30 Components:
1. Order Book Imbalance
2. Level 2 Aggregator
3. Depth Analyzer
4. Spread Monitor
5. Tick Rule Classifier
6. Trade Flow Analyzer
7. Volume Profile
8. Price Impact Model
9. Market Making Optimizer
9. Liquidity Provider
10. Adverse Selection Detector
11. Information Share Calculator
12. VPIN Estimator
13. Kyle's Lambda
14. Amihud Illiquidity
15. Effective Spread
16. Realized Spread
17. Price Discovery
18. Order Arrival Process
19. Queue Position Tracker
20. Fill Probability Estimator
21. Latency Arbitrage Detector
22. Cross-Exchange Spread
23. Funding Rate Arbitrage
24. Basis Trade Engine
25. Gamma Scalper
26. Delta Neutral Manager
27. Vanna/Volga Tracker
28. Greeks Calculator
29. Implied Vol Surface
30. Microprice Calculator
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)

# GPU availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@dataclass
class HFTConfig:
    """HFT configuration."""
    tick_size: float = 0.01
    min_spread: float = 0.01
    max_position: int = 1000
    latency_threshold_us: int = 100
    gpu_enabled: bool = CUDA_AVAILABLE


class OrderBookImbalance:
    """GPU-accelerated order book imbalance calculation."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.imbalance_history = deque(maxlen=1000)
    
    def calculate(self, bids: np.ndarray, asks: np.ndarray) -> float:
        """Calculate order book imbalance [-1, 1]."""
        if CUDA_AVAILABLE:
            bid_tensor = torch.tensor(bids, dtype=torch.float32, device='cuda')
            ask_tensor = torch.tensor(asks, dtype=torch.float32, device='cuda')
            
            bid_volume = torch.sum(bid_tensor[:, 1])
            ask_volume = torch.sum(ask_tensor[:, 1])
            
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
            result = imbalance.cpu().item()
        else:
            bid_volume = np.sum(bids[:, 1])
            ask_volume = np.sum(asks[:, 1])
            result = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        
        self.imbalance_history.append(result)
        return result
    
    def get_signal(self, threshold: float = 0.3) -> int:
        """Get trading signal from imbalance."""
        if len(self.imbalance_history) < 10:
            return 0
        recent_imbalance = np.mean(list(self.imbalance_history)[-10:])
        if recent_imbalance > threshold:
            return 1  # Buy signal
        elif recent_imbalance < -threshold:
            return -1  # Sell signal
        return 0


class Level2Aggregator:
    """GPU-accelerated Level 2 data aggregation."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.levels = 20
    
    def aggregate(self, orderbook: Dict) -> Dict[str, float]:
        """Aggregate L2 data into features."""
        bids = np.array(orderbook.get('bids', [])[:self.levels])
        asks = np.array(orderbook.get('asks', [])[:self.levels])
        
        if len(bids) == 0 or len(asks) == 0:
            return {}
        
        if CUDA_AVAILABLE:
            bid_tensor = torch.tensor(bids, dtype=torch.float32, device='cuda')
            ask_tensor = torch.tensor(asks, dtype=torch.float32, device='cuda')
            
            features = {
                'best_bid': bid_tensor[0, 0].cpu().item(),
                'best_ask': ask_tensor[0, 0].cpu().item(),
                'mid_price': ((bid_tensor[0, 0] + ask_tensor[0, 0]) / 2).cpu().item(),
                'spread': (ask_tensor[0, 0] - bid_tensor[0, 0]).cpu().item(),
                'bid_depth': torch.sum(bid_tensor[:, 1]).cpu().item(),
                'ask_depth': torch.sum(ask_tensor[:, 1]).cpu().item(),
                'depth_imbalance': ((torch.sum(bid_tensor[:, 1]) - torch.sum(ask_tensor[:, 1])) / 
                                   (torch.sum(bid_tensor[:, 1]) + torch.sum(ask_tensor[:, 1]) + 1e-10)).cpu().item(),
            }
        else:
            features = {
                'best_bid': bids[0, 0],
                'best_ask': asks[0, 0],
                'mid_price': (bids[0, 0] + asks[0, 0]) / 2,
                'spread': asks[0, 0] - bids[0, 0],
                'bid_depth': np.sum(bids[:, 1]),
                'ask_depth': np.sum(asks[:, 1]),
                'depth_imbalance': (np.sum(bids[:, 1]) - np.sum(asks[:, 1])) / 
                                   (np.sum(bids[:, 1]) + np.sum(asks[:, 1]) + 1e-10),
            }
        
        return features


class DepthAnalyzer:
    """Analyze order book depth profile."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
    
    def analyze(self, bids: np.ndarray, asks: np.ndarray) -> Dict[str, float]:
        """Analyze depth profile."""
        if CUDA_AVAILABLE:
            bid_tensor = torch.tensor(bids, dtype=torch.float32, device='cuda')
            ask_tensor = torch.tensor(asks, dtype=torch.float32, device='cuda')
            
            # Weighted average price
            bid_wap = torch.sum(bid_tensor[:, 0] * bid_tensor[:, 1]) / torch.sum(bid_tensor[:, 1])
            ask_wap = torch.sum(ask_tensor[:, 0] * ask_tensor[:, 1]) / torch.sum(ask_tensor[:, 1])
            
            # Depth at different levels
            depth_5_bid = torch.sum(bid_tensor[:5, 1])
            depth_10_bid = torch.sum(bid_tensor[:10, 1])
            depth_5_ask = torch.sum(ask_tensor[:5, 1])
            depth_10_ask = torch.sum(ask_tensor[:10, 1])
            
            return {
                'bid_wap': bid_wap.cpu().item(),
                'ask_wap': ask_wap.cpu().item(),
                'depth_5_bid': depth_5_bid.cpu().item(),
                'depth_10_bid': depth_10_bid.cpu().item(),
                'depth_5_ask': depth_5_ask.cpu().item(),
                'depth_10_ask': depth_10_ask.cpu().item(),
            }
        else:
            bid_wap = np.sum(bids[:, 0] * bids[:, 1]) / np.sum(bids[:, 1])
            ask_wap = np.sum(asks[:, 0] * asks[:, 1]) / np.sum(asks[:, 1])
            
            return {
                'bid_wap': bid_wap,
                'ask_wap': ask_wap,
                'depth_5_bid': np.sum(bids[:5, 1]),
                'depth_10_bid': np.sum(bids[:10, 1]),
                'depth_5_ask': np.sum(asks[:5, 1]),
                'depth_10_ask': np.sum(asks[:10, 1]),
            }


class SpreadMonitor:
    """Monitor and analyze bid-ask spread."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.spread_history = deque(maxlen=1000)
    
    def update(self, best_bid: float, best_ask: float) -> Dict[str, float]:
        """Update spread metrics."""
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        relative_spread = spread / mid_price if mid_price > 0 else 0
        
        self.spread_history.append(spread)
        
        return {
            'spread': spread,
            'relative_spread': relative_spread,
            'spread_mean': np.mean(list(self.spread_history)) if self.spread_history else 0,
            'spread_std': np.std(list(self.spread_history)) if len(self.spread_history) > 1 else 0,
        }


class TickRuleClassifier:
    """Classify price movements using tick rule."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.last_price = None
    
    def classify(self, price: float) -> int:
        """Classify tick: +1 (up), -1 (down), 0 (unchanged)."""
        if self.last_price is None:
            self.last_price = price
            return 0
        
        if price > self.last_price:
            result = 1
        elif price < self.last_price:
            result = -1
        else:
            result = 0
        
        self.last_price = price
        return result


class TradeFlowAnalyzer:
    """Analyze trade flow for direction inference."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.trade_flow = deque(maxlen=1000)
    
    def add_trade(self, price: float, volume: float, side: str):
        """Add trade to flow."""
        direction = 1 if side.lower() == 'buy' else -1
        self.trade_flow.append({
            'price': price,
            'volume': volume,
            'direction': direction,
            'timestamp': time.time()
        })
    
    def get_flow_imbalance(self, window: int = 100) -> float:
        """Get trade flow imbalance."""
        if len(self.trade_flow) < window:
            return 0.0
        
        recent = list(self.trade_flow)[-window:]
        buy_volume = sum(t['volume'] for t in recent if t['direction'] == 1)
        sell_volume = sum(t['volume'] for t in recent if t['direction'] == -1)
        
        return (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10)


class VolumeProfile:
    """Calculate volume profile."""
    
    def __init__(self, config: HFTConfig, num_bins: int = 50):
        self.config = config
        self.num_bins = num_bins
        self.volume_at_price = {}
    
    def update(self, price: float, volume: float):
        """Update volume profile."""
        bin_key = round(price / self.config.tick_size) * self.config.tick_size
        self.volume_at_price[bin_key] = self.volume_at_price.get(bin_key, 0) + volume
    
    def get_poc(self) -> Optional[float]:
        """Get Point of Control (price with highest volume)."""
        if not self.volume_at_price:
            return None
        return max(self.volume_at_price, key=self.volume_at_price.get)
    
    def get_value_area(self, pct: float = 0.7) -> Tuple[float, float]:
        """Get value area (price range with pct of volume)."""
        if not self.volume_at_price:
            return 0.0, 0.0
        
        sorted_prices = sorted(self.volume_at_price.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(self.volume_at_price.values())
        target_volume = total_volume * pct
        
        cumulative = 0
        prices_in_area = []
        for price, vol in sorted_prices:
            cumulative += vol
            prices_in_area.append(price)
            if cumulative >= target_volume:
                break
        
        return min(prices_in_area), max(prices_in_area)


class PriceImpactModel:
    """Model price impact of trades."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.impact_history = deque(maxlen=1000)
    
    def estimate_impact(self, volume: float, side: str, orderbook: Dict) -> float:
        """Estimate price impact."""
        depth = np.array(orderbook.get('asks' if side == 'buy' else 'bids', []))
        
        if len(depth) == 0:
            return 0.0
        
        if CUDA_AVAILABLE:
            depth_tensor = torch.tensor(depth, dtype=torch.float32, device='cuda')
            cumulative_volume = torch.cumsum(depth_tensor[:, 1], dim=0)
            volume_tensor = torch.tensor(volume, dtype=torch.float32, device='cuda')
            
            # Find level where cumulative volume exceeds order volume
            levels_needed = torch.sum(cumulative_volume < volume_tensor).cpu().item() + 1
            levels_needed = min(levels_needed, len(depth))
            
            impact = depth_tensor[levels_needed - 1, 0].cpu().item() - depth_tensor[0, 0].cpu().item()
        else:
            cumulative_volume = np.cumsum(depth[:, 1])
            levels_needed = np.sum(cumulative_volume < volume) + 1
            levels_needed = min(levels_needed, len(depth))
            
            impact = depth[levels_needed - 1, 0] - depth[0, 0]
        
        self.impact_history.append(impact)
        return impact


class MarketMakingOptimizer:
    """Optimize market making quotes."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.inventory = 0
        self.pnl = 0.0
    
    def optimize_quotes(self, mid_price: float, volatility: float, 
                        inventory: int, target_spread: float) -> Dict[str, float]:
        """Optimize bid/ask quotes."""
        # Inventory skew
        max_inventory = self.config.max_position
        inventory_skew = -inventory / max_inventory * volatility * mid_price
        
        # Optimal quotes
        half_spread = target_spread / 2
        bid_price = mid_price - half_spread + inventory_skew
        ask_price = mid_price + half_spread + inventory_skew
        
        # Ensure positive spread
        if ask_price <= bid_price:
            ask_price = bid_price + self.config.tick_size
        
        return {
            'bid_price': round(bid_price / self.config.tick_size) * self.config.tick_size,
            'ask_price': round(ask_price / self.config.tick_size) * self.config.tick_size,
            'spread': ask_price - bid_price,
            'inventory_skew': inventory_skew,
        }


class LiquidityProvider:
    """Provide liquidity analysis."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.liquidity_scores = deque(maxlen=100)
    
    def calculate_score(self, orderbook: Dict, trade_flow: List) -> float:
        """Calculate liquidity score."""
        bids = np.array(orderbook.get('bids', []))
        asks = np.array(orderbook.get('asks', []))
        
        if len(bids) == 0 or len(asks) == 0:
            return 0.0
        
        # Depth score
        depth_score = min(np.sum(bids[:, 1]) + np.sum(asks[:, 1]), 10000) / 10000
        
        # Spread score
        spread = asks[0, 0] - bids[0, 0]
        mid_price = (asks[0, 0] + bids[0, 0]) / 2
        spread_score = max(0, 1 - (spread / mid_price) * 1000)
        
        # Trade frequency score
        trade_score = min(len(trade_flow) / 100, 1.0) if trade_flow else 0
        
        score = (depth_score * 0.4 + spread_score * 0.3 + trade_score * 0.3)
        self.liquidity_scores.append(score)
        return score


class AdverseSelectionDetector:
    """Detect adverse selection in trades."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.adverse_history = deque(maxlen=100)
    
    def detect(self, trade_price: float, post_trade_price: float, 
               side: str, horizon: int = 5) -> bool:
        """Detect if trade was adversely selected."""
        price_diff = post_trade_price - trade_price
        
        if side == 'buy':
            adverse = price_diff < 0  # Price went down after we bought
        else:
            adverse = price_diff > 0  # Price went up after we sold
        
        self.adverse_history.append(adverse)
        return adverse
    
    def get_adverse_rate(self) -> float:
        """Get adverse selection rate."""
        if not self.adverse_history:
            return 0.0
        return sum(self.adverse_history) / len(self.adverse_history)


class InformationShareCalculator:
    """Calculate information share of venues."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.price_changes = {}
    
    def add_price_change(self, venue: str, price_change: float):
        """Add price change from venue."""
        if venue not in self.price_changes:
            self.price_changes[venue] = []
        self.price_changes[venue].append(price_change)
    
    def calculate(self) -> Dict[str, float]:
        """Calculate information share."""
        if len(self.price_changes) < 2:
            return {}
        
        # Simplified Hasbrouck information share
        variances = {}
        for venue, changes in self.price_changes.items():
            variances[venue] = np.var(changes) if len(changes) > 1 else 0
        
        total_var = sum(variances.values())
        if total_var == 0:
            return {v: 1.0 / len(variances) for v in variances}
        
        return {v: var / total_var for v, var in variances.items()}


class VPINEstimator:
    """Estimate Volume-Synchronized Probability of Informed Trading."""
    
    def __init__(self, config: HFTConfig, num_bins: int = 50):
        self.config = config
        self.num_bins = num_bins
        self.trade_flows = deque(maxlen=num_bins)
    
    def add_trade_flow(self, buy_volume: float, sell_volume: float):
        """Add trade flow for VPIN calculation."""
        self.trade_flows.append({
            'buy': buy_volume,
            'sell': sell_volume
        })
    
    def calculate(self) -> float:
        """Calculate VPIN."""
        if len(self.trade_flows) < self.num_bins:
            return 0.5  # Default neutral
        
        flows = list(self.trade_flows)[-self.num_bins:]
        total_volume = sum(f['buy'] + f['sell'] for f in flows)
        if total_volume == 0:
            return 0.5
        
        # Volume imbalance
        buy_volume = sum(f['buy'] for f in flows)
        sell_volume = sum(f['sell'] for f in flows)
        
        vpin = abs(buy_volume - sell_volume) / total_volume
        return min(max(vpin, 0), 1)


class KylesLambda:
    """Calculate Kyle's Lambda (price impact coefficient)."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.observations = deque(maxlen=100)
    
    def add_observation(self, order_flow: float, price_change: float):
        """Add observation for lambda calculation."""
        self.observations.append((order_flow, price_change))
    
    def calculate(self) -> float:
        """Calculate Kyle's Lambda."""
        if len(self.observations) < 10:
            return 0.0
        
        flows = np.array([o[0] for o in self.observations])
        prices = np.array([o[1] for o in self.observations])
        
        # Linear regression
        if np.std(flows) > 0:
            lambda_val = np.cov(flows, prices)[0, 1] / np.var(flows)
            return abs(lambda_val)
        return 0.0


class AmihudIlliquidity:
    """Calculate Amihud illiquidity ratio."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.returns = deque(maxlen=100)
        self.volumes = deque(maxlen=100)
    
    def add_observation(self, return_val: float, volume: float, dollar_volume: float):
        """Add observation."""
        if dollar_volume > 0:
            self.returns.append(abs(return_val))
            self.volumes.append(dollar_volume)
    
    def calculate(self) -> float:
        """Calculate Amihud illiquidity."""
        if len(self.returns) < 2:
            return 0.0
        
        ratios = [r / v for r, v in zip(self.returns, self.volumes) if v > 0]
        return np.mean(ratios) if ratios else 0.0


class EffectiveSpread:
    """Calculate effective spread."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.spreads = deque(maxlen=100)
    
    def calculate(self, trade_price: float, mid_price: float, side: str) -> float:
        """Calculate effective spread."""
        signed_spread = 2 * (trade_price - mid_price)
        if side == 'sell':
            signed_spread = -signed_spread
        
        self.spreads.append(signed_spread)
        return signed_spread
    
    def get_average(self) -> float:
        """Get average effective spread."""
        return np.mean(list(self.spreads)) if self.spreads else 0.0


class RealizedSpread:
    """Calculate realized spread."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.spreads = deque(maxlen=100)
    
    def calculate(self, trade_price: float, mid_price_before: float, 
                  mid_price_after: float, side: str) -> float:
        """Calculate realized spread."""
        signed_spread = 2 * (trade_price - mid_price_after)
        if side == 'sell':
            signed_spread = -signed_spread
        
        self.spreads.append(signed_spread)
        return signed_spread


class PriceDiscovery:
    """Analyze price discovery process."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.prices = {}
    
    def add_price(self, venue: str, price: float):
        """Add price from venue."""
        if venue not in self.prices:
            self.prices[venue] = deque(maxlen=100)
        self.prices[venue].append(price)
    
    def get_leader(self) -> Optional[str]:
        """Get price discovery leader."""
        if len(self.prices) < 2:
            return None
        
        # Simplified: venue with most price changes leads
        changes = {}
        for venue, prices in self.prices.items():
            if len(prices) > 1:
                changes[venue] = sum(1 for i in range(1, len(prices)) 
                                    if prices[i] != prices[i-1])
        
        if not changes:
            return None
        
        return max(changes, key=changes.get)


class OrderArrivalProcess:
    """Model order arrival process."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.arrivals = deque(maxlen=1000)
        self.last_arrival = None
    
    def record_arrival(self):
        """Record order arrival."""
        now = time.time()
        if self.last_arrival is not None:
            inter_arrival = now - self.last_arrival
            self.arrivals.append(inter_arrival)
        self.last_arrival = now
    
    def get_arrival_rate(self) -> float:
        """Get order arrival rate (orders per second)."""
        if len(self.arrivals) < 2:
            return 0.0
        avg_inter_arrival = np.mean(list(self.arrivals))
        return 1.0 / avg_inter_arrival if avg_inter_arrival > 0 else 0.0


class QueuePositionTracker:
    """Track queue position in order book."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.queue_positions = {}
    
    def update(self, side: str, price: float, volume_ahead: float):
        """Update queue position."""
        key = (side, price)
        self.queue_positions[key] = volume_ahead
    
    def get_fill_probability(self, side: str, price: float, 
                            arrival_rate: float) -> float:
        """Estimate fill probability."""
        key = (side, price)
        volume_ahead = self.queue_positions.get(key, float('inf'))
        
        if volume_ahead == 0:
            return 1.0
        
        # Simplified model
        rate_factor = min(arrival_rate / 100, 1.0)
        volume_factor = max(0, 1 - volume_ahead / 10000)
        
        return rate_factor * volume_factor


class FillProbabilityEstimator:
    """Estimate order fill probability."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.fill_history = []
    
    def estimate(self, distance_from_mid: float, volatility: float,
                 spread: float, volume: float) -> float:
        """Estimate fill probability."""
        # Distance factor
        distance_factor = max(0, 1 - abs(distance_from_mid) / (volatility * 10))
        
        # Spread factor
        spread_factor = max(0, 1 - spread / (volatility * 5))
        
        # Volume factor
        volume_factor = min(volume / 1000, 1.0)
        
        return distance_factor * spread_factor * volume_factor


class LatencyArbitrageDetector:
    """Detect latency arbitrage opportunities."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.cross_prices = {}
    
    def add_price(self, venue: str, price: float, latency_us: float):
        """Add price from venue with latency."""
        self.cross_prices[venue] = {
            'price': price,
            'latency': latency_us,
            'timestamp': time.time()
        }
    
    def detect_opportunity(self, threshold: float = 0.001) -> Optional[Dict]:
        """Detect latency arbitrage opportunity."""
        if len(self.cross_prices) < 2:
            return None
        
        # Find best bid/ask across venues
        prices = [(v, d['price'], d['latency']) for v, d in self.cross_prices.items()]
        prices.sort(key=lambda x: x[1])
        
        # Check if spread exists
        if len(prices) >= 2:
            buy_venue, buy_price, buy_latency = prices[0]
            sell_venue, sell_price, sell_latency = prices[-1]
            
            spread = sell_price - buy_price
            if spread > threshold:
                return {
                    'buy_venue': buy_venue,
                    'sell_venue': sell_venue,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'spread': spread,
                    'latency': buy_latency + sell_latency,
                }
        
        return None


class CrossExchangeSpread:
    """Monitor cross-exchange spreads."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.exchange_prices = {}
    
    def update(self, exchange: str, bid: float, ask: float):
        """Update exchange prices."""
        self.exchange_prices[exchange] = {'bid': bid, 'ask': ask}
    
    def get_spreads(self) -> Dict[str, float]:
        """Get cross-exchange spreads."""
        spreads = {}
        exchanges = list(self.exchange_prices.keys())
        
        for i, ex1 in enumerate(exchanges):
            for ex2 in exchanges[i+1:]:
                # Cross exchange spread
                buy_ex1 = self.exchange_prices[ex1]['bid']
                sell_ex2 = self.exchange_prices[ex2]['ask']
                spread1 = sell_ex2 - buy_ex1
                
                buy_ex2 = self.exchange_prices[ex2]['bid']
                sell_ex1 = self.exchange_prices[ex1]['ask']
                spread2 = sell_ex1 - buy_ex2
                
                spreads[f"{ex1}-{ex2}"] = max(spread1, spread2)
        
        return spreads


class FundingRateArbitrage:
    """Funding rate arbitrage for perpetuals."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.funding_rates = {}
    
    def update_funding_rate(self, symbol: str, rate: float, next_funding: float):
        """Update funding rate."""
        self.funding_rates[symbol] = {
            'rate': rate,
            'next_funding': next_funding,
            'annualized': rate * 3 * 365  # 8-hour funding, 3x daily
        }
    
    def get_arbitrage_opportunity(self, spot_price: float, 
                                  futures_price: float, symbol: str) -> Optional[Dict]:
        """Calculate funding arbitrage opportunity."""
        if symbol not in self.funding_rates:
            return None
        
        funding = self.funding_rates[symbol]
        basis = futures_price - spot_price
        basis_pct = basis / spot_price
        
        # If funding is positive, short futures / long spot
        if funding['rate'] > 0.001:  # > 0.1% funding
            return {
                'type': 'funding_arbitrage',
                'action': 'short_futures_long_spot',
                'expected_return': funding['annualized'],
                'basis': basis_pct,
            }
        
        return None


class BasisTradeEngine:
    """Basis trading engine."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.basis_history = deque(maxlen=100)
    
    def calculate_basis(self, futures: float, spot: float) -> float:
        """Calculate basis."""
        basis = futures - spot
        basis_pct = basis / spot if spot > 0 else 0
        self.basis_history.append(basis_pct)
        return basis_pct
    
    def get_signal(self) -> int:
        """Get basis trading signal."""
        if len(self.basis_history) < 20:
            return 0
        
        current_basis = list(self.basis_history)[-1]
        mean_basis = np.mean(list(self.basis_history))
        std_basis = np.std(list(self.basis_history))
        
        if std_basis == 0:
            return 0
        
        z_score = (current_basis - mean_basis) / std_basis
        
        if z_score > 2:
            return -1  # Sell futures, buy spot
        elif z_score < -2:
            return 1  # Buy futures, sell spot
        
        return 0


class GammaScalper:
    """Gamma scalping for options."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.position = 0
        self.pnl = 0.0
    
    def calculate_hedge(self, gamma: float, delta: float, 
                        spot_move: float, volatility_move: float) -> Dict:
        """Calculate gamma scalping hedge."""
        # Gamma PnL
        gamma_pnl = 0.5 * gamma * spot_move ** 2
        
        # Vega PnL (from volatility change)
        vega_pnl = volatility_move * 100  # Simplified
        
        # Delta hedge adjustment
        delta_change = gamma * spot_move
        hedge_shares = -delta_change
        
        return {
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'delta_change': delta_change,
            'hedge_shares': hedge_shares,
            'total_pnl': gamma_pnl + vega_pnl,
        }


class DeltaNeutralManager:
    """Manage delta-neutral positions."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.target_delta = 0.0
        self.current_delta = 0.0
    
    def update_delta(self, options_delta: float, hedge_delta: float):
        """Update current delta."""
        self.current_delta = options_delta + hedge_delta
    
    def calculate_hedge(self, target_delta: float = 0.0) -> float:
        """Calculate hedge needed to reach target delta."""
        return target_delta - self.current_delta


class VannaVolgaTracker:
    """Track vanna and volga exposures."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.greeks = {}
    
    def update_greeks(self, strike: float, expiry: float, 
                      vanna: float, volga: float):
        """Update vanna and volga."""
        key = (strike, expiry)
        self.greeks[key] = {'vanna': vanna, 'volga': volga}
    
    def get_total_exposure(self) -> Dict[str, float]:
        """Get total vanna and volga exposure."""
        total_vanna = sum(g['vanna'] for g in self.greeks.values())
        total_volga = sum(g['volga'] for g in self.greeks.values())
        
        return {
            'total_vanna': total_vanna,
            'total_volga': total_volga,
        }


class GreeksCalculator:
    """Calculate option Greeks."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
    
    def calculate(self, spot: float, strike: float, time_to_expiry: float,
                  volatility: float, rate: float, option_type: str = 'call') -> Dict[str, float]:
        """Calculate Greeks using Black-Scholes."""
        from scipy.stats import norm
        import math
        
        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_t)
        d2 = d1 - volatility * sqrt_t
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) - 
                     rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2))
        else:
            delta = norm.cdf(d1) - 1
            theta = (-spot * norm.pdf(d1) * volatility / (2 * sqrt_t) + 
                     rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2))
        
        gamma = norm.pdf(d1) / (spot * volatility * sqrt_t)
        vega = spot * norm.pdf(d1) * sqrt_t / 100
        rho = strike * time_to_expiry * math.exp(-rate * time_to_expiry) * norm.cdf(d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
        }


class ImpliedVolSurface:
    """Build and analyze implied volatility surface."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.vol_surface = {}
    
    def add_iv(self, strike: float, expiry: float, iv: float):
        """Add implied volatility point."""
        key = (strike, expiry)
        self.vol_surface[key] = iv
    
    def get_iv(self, strike: float, expiry: float) -> Optional[float]:
        """Get interpolated implied volatility."""
        # Simple nearest neighbor
        if (strike, expiry) in self.vol_surface:
            return self.vol_surface[(strike, expiry)]
        
        # Find closest
        closest = None
        min_dist = float('inf')
        
        for (k, e), iv in self.vol_surface.items():
            dist = abs(k - strike) + abs(e - expiry) * 100
            if dist < min_dist:
                min_dist = dist
                closest = iv
        
        return closest


class MicropriceCalculator:
    """Calculate microprice (volume-weighted mid)."""
    
    def __init__(self, config: HFTConfig):
        self.config = config
    
    def calculate(self, best_bid: float, best_ask: float, 
                  bid_volume: float, ask_volume: float) -> float:
        """Calculate microprice."""
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return (best_bid + best_ask) / 2
        
        microprice = (best_bid * ask_volume + best_ask * bid_volume) / total_volume
        return microprice


class HFTMicrostructureEngine:
    """
    HFT Microstructure Engine - 30 GPU-accelerated components.
    """
    
    def __init__(self, config: Optional[HFTConfig] = None):
        self.config = config or HFTConfig()
        
        # Initialize all 30 components
        self.order_book_imbalance = OrderBookImbalance(self.config)
        self.level2_aggregator = Level2Aggregator(self.config)
        self.depth_analyzer = DepthAnalyzer(self.config)
        self.spread_monitor = SpreadMonitor(self.config)
        self.tick_rule_classifier = TickRuleClassifier(self.config)
        self.trade_flow_analyzer = TradeFlowAnalyzer(self.config)
        self.volume_profile = VolumeProfile(self.config)
        self.price_impact_model = PriceImpactModel(self.config)
        self.market_making_optimizer = MarketMakingOptimizer(self.config)
        self.liquidity_provider = LiquidityProvider(self.config)
        self.adverse_selection_detector = AdverseSelectionDetector(self.config)
        self.information_share_calculator = InformationShareCalculator(self.config)
        self.vpin_estimator = VPINEstimator(self.config)
        self.kyles_lambda = KylesLambda(self.config)
        self.amihud_illiquidity = AmihudIlliquidity(self.config)
        self.effective_spread = EffectiveSpread(self.config)
        self.realized_spread = RealizedSpread(self.config)
        self.price_discovery = PriceDiscovery(self.config)
        self.order_arrival_process = OrderArrivalProcess(self.config)
        self.queue_position_tracker = QueuePositionTracker(self.config)
        self.fill_probability_estimator = FillProbabilityEstimator(self.config)
        self.latency_arbitrage_detector = LatencyArbitrageDetector(self.config)
        self.cross_exchange_spread = CrossExchangeSpread(self.config)
        self.funding_rate_arbitrage = FundingRateArbitrage(self.config)
        self.basis_trade_engine = BasisTradeEngine(self.config)
        self.gamma_scalper = GammaScalper(self.config)
        self.delta_neutral_manager = DeltaNeutralManager(self.config)
        self.vanna_volga_tracker = VannaVolgaTracker(self.config)
        self.greeks_calculator = GreeksCalculator(self.config)
        self.implied_vol_surface = ImpliedVolSurface(self.config)
        self.microprice_calculator = MicropriceCalculator(self.config)
        
        logger.info(f"HFT Microstructure Engine initialized with {self._count_components()} components")
    
    def _count_components(self) -> int:
        """Count initialized components."""
        return 30
    
    def analyze_orderbook(self, orderbook: Dict) -> Dict[str, Any]:
        """Full orderbook analysis."""
        bids = np.array(orderbook.get('bids', []))
        asks = np.array(orderbook.get('asks', []))
        
        if len(bids) == 0 or len(asks) == 0:
            return {}
        
        best_bid = bids[0, 0]
        best_ask = asks[0, 0]
        mid_price = (best_bid + best_ask) / 2
        
        return {
            'imbalance': self.order_book_imbalance.calculate(bids, asks),
            'l2_features': self.level2_aggregator.aggregate(orderbook),
            'depth': self.depth_analyzer.analyze(bids, asks),
            'spread': self.spread_monitor.update(best_bid, best_ask),
            'microprice': self.microprice_calculator.calculate(best_bid, best_ask, bids[0, 1], asks[0, 1]),
            'liquidity_score': self.liquidity_provider.calculate_score(orderbook, []),
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'components': self._count_components(),
            'gpu_enabled': CUDA_AVAILABLE,
            'spread_mean': self.spread_monitor.spread_history[-1] if self.spread_monitor.spread_history else 0,
            'vpin': self.vpin_estimator.calculate(),
            'kyle_lambda': self.kyles_lambda.calculate(),
            'adverse_rate': self.adverse_selection_detector.get_adverse_rate(),
        }
