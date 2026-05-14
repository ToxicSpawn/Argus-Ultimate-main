"""
Multi-Pair Liquidity Scanner for Argus Ultimate.

Scans exchanges for the most liquid trading pairs and dynamically selects
the best pairs for trading based on:
- 24h volume (USD)
- Bid-ask spread
- Order book depth
- Volatility
- Funding rates (for perpetuals)
- Correlation with existing positions
- Historical performance

Features:
- Real-time liquidity ranking across all exchange pairs
- Dynamic pair rotation based on market conditions
- Multi-exchange scanning (Kraken, Coinbase, Binance)
- Automatic pair discovery and filtering
- Integration with quantum adapter for optimal selection
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import deque
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class PairStatus(Enum):
    """Status of a trading pair."""
    ACTIVE = "active"           # Currently trading
    MONITORING = "monitoring"   # Watching for entry
    PAUSED = "paused"           # Temporarily disabled
    BLACKLISTED = "blacklisted" # Permanently disabled


@dataclass
class PairLiquidityMetrics:
    """Comprehensive liquidity metrics for a trading pair."""
    symbol: str
    exchange: str
    
    # Volume metrics
    volume_24h_usd: float
    volume_1h_usd: float
    volume_trend: float  # -1 to 1 (declining to growing)
    
    # Spread metrics
    spread_pct: float  # (ask - bid) / mid
    spread_bps: float  # spread in basis points
    
    # Order book metrics
    bid_depth_usd: float  # Total bid value in top N levels
    ask_depth_usd: float  # Total ask value in top N levels
    orderbook_imbalance: float  # -1 to 1 (bid heavy to ask heavy)
    
    # Volatility metrics
    volatility_24h: float  # Standard deviation of returns
    atr_pct: float  # Average True Range as % of price
    
    # Price metrics
    price: float
    price_change_24h_pct: float
    
    # Composite scores
    liquidity_score: float  # 0-100
    trading_score: float  # 0-100 (includes volatility)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: PairStatus = PairStatus.MONITORING
    
    @property
    def is_liquid(self) -> bool:
        return self.liquidity_score > 50
    
    @property
    def is_tradeable(self) -> bool:
        return (
            self.status == PairStatus.ACTIVE and
            self.liquidity_score > 40 and
            self.volume_24h_usd > 100_000 and
            self.spread_pct < 0.5
        )


@dataclass
class PairSelection:
    """Selected pair with allocation and reasoning."""
    symbol: str
    exchange: str
    allocation_pct: float  # % of capital to allocate
    expected_return_pct: float
    risk_score: float  # 0-1 (lower = safer)
    confidence: float  # 0-1
    
    # Selection reasons
    reasons: List[str] = field(default_factory=list)
    
    # Metrics at selection time
    liquidity_score: float = 0.0
    volume_24h_usd: float = 0.0
    spread_pct: float = 0.0


class MultiPairLiquidityScanner:
    """
    Multi-Pair Liquidity Scanner.
    
    Continuously scans exchanges for the most liquid pairs and selects
    the best ones for trading based on multiple criteria.
    """
    
    # Default trading pairs (major cryptos) - Kraken uses /USD not /USDT
    DEFAULT_PAIRS = [
        "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD",
        "DOGE/USD", "AVAX/USD", "DOT/USD", "MATIC/USD", "LINK/USD",
        "UNI/USD", "ATOM/USD", "LTC/USD", "BCH/USD", "XLM/USD",
        "ARB/USD", "OP/USD", "INJ/USD", "NEAR/USD", "FIL/USD",
    ]
    
    # Liquidity thresholds
    MIN_VOLUME_USD = 100_000  # Minimum 24h volume
    MAX_SPREAD_PCT = 0.5  # Maximum spread %
    MIN_LIQUIDITY_SCORE = 40  # Minimum liquidity score
    
    def __init__(
        self,
        exchanges: List[str] = None,
        max_pairs: int = 20,
        min_volume_usd: float = 100_000,
        max_spread_pct: float = 0.5,
        scan_interval_seconds: float = 60.0,
        enable_auto_discovery: bool = True,
    ):
        """
        Initialize Multi-Pair Liquidity Scanner.
        
        Args:
            exchanges: List of exchange IDs to scan
            max_pairs: Maximum number of pairs to track
            min_volume_usd: Minimum 24h volume in USD
            max_spread_pct: Maximum acceptable spread %
            scan_interval_seconds: How often to rescan
            enable_auto_discovery: Auto-discover new pairs
        """
        self.exchanges = exchanges or ["kraken", "coinbase", "bybit"]
        self.max_pairs = max_pairs
        self.min_volume_usd = min_volume_usd
        self.max_spread_pct = max_spread_pct
        self.scan_interval = scan_interval_seconds
        self.enable_auto_discovery = enable_auto_discovery
        
        # State
        self.pair_metrics: Dict[str, PairLiquidityMetrics] = {}
        self.selected_pairs: List[PairSelection] = []
        self.blacklisted_pairs: Set[str] = set()
        self.last_scan_time: Optional[datetime] = None
        
        # Performance tracking
        self.scan_count = 0
        self.pairs_found = 0
        self.rotations = 0
        
        # History
        self.liquidity_history: Dict[str, deque] = {}
        
        # Exchange connections (lazy loaded)
        self._exchanges: Dict[str, Any] = {}
        
        logger.info(
            f"MultiPairLiquidityScanner initialized: "
            f"exchanges={self.exchanges}, "
            f"max_pairs={max_pairs}"
        )
    
    async def scan_all_pairs(self) -> Dict[str, PairLiquidityMetrics]:
        """
        Scan all exchanges for liquid pairs.
        
        Returns:
            Dict mapping symbol to PairLiquidityMetrics
        """
        self.scan_count += 1
        all_metrics = {}
        
        for exchange_id in self.exchanges:
            try:
                exchange_metrics = await self._scan_exchange(exchange_id)
                # Merge keeping highest volume for duplicates
                for key, metrics in exchange_metrics.items():
                    if key in all_metrics and all_metrics[key].volume_24h_usd >= metrics.volume_24h_usd:
                        continue  # Keep existing higher-volume entry
                    all_metrics[key] = metrics
            except Exception as e:
                logger.warning(f"Error scanning {exchange_id}: {e}")
        
        # Update stored metrics
        self.pair_metrics.update(all_metrics)
        self.pairs_found = len(all_metrics)
        self.last_scan_time = datetime.utcnow()
        
        # Update history
        self._update_history(all_metrics)
        
        logger.info(
            f"Scan complete: {len(all_metrics)} pairs found across "
            f"{len(self.exchanges)} exchanges"
        )
        
        return all_metrics
    
    async def _scan_exchange(self, exchange_id: str) -> Dict[str, PairLiquidityMetrics]:
        """Scan a single exchange for liquid pairs."""
        metrics = {}
        
        try:
            exchange = await self._get_exchange(exchange_id)
            
            # Get all tickers
            tickers = await exchange.fetch_tickers()
            
            # Filter and score pairs
            for symbol, ticker in tickers.items():
                # Skip non-USD pairs for now
                if not self._is_valid_pair(symbol):
                    continue
                
                # Skip blacklisted
                if symbol in self.blacklisted_pairs:
                    continue
                
                # Calculate metrics
                pair_metrics = self._calculate_metrics(exchange_id, symbol, ticker)
                
                if pair_metrics and pair_metrics.liquidity_score >= self.MIN_LIQUIDITY_SCORE:
                    # Use normalized symbol as key, but keep highest volume if duplicate
                    key = pair_metrics.symbol
                    if key in metrics and metrics[key].volume_24h_usd >= pair_metrics.volume_24h_usd:
                        # Keep existing higher-volume entry
                        continue
                    metrics[key] = pair_metrics
                    
        except Exception as e:
            logger.error(f"Error scanning exchange {exchange_id}: {e}")
        
        return metrics
    
    def _is_valid_pair(self, symbol: str) -> bool:
        """Check if a pair is valid for trading.
        
        Supports multiple quote currencies:
        - Kraken/Coinbase: /USD
        - Bybit/Binance: /USDT or /USDT:USDT
        
        Filters out:
        - Futures contracts (suffixes like -260501, -260508)
        - Stablecoin bases (USDT, USDC, etc as base)
        """
        # First check: skip futures contracts (have date suffixes like -260501)
        # Check in the full symbol before any normalization
        if "-" in symbol:
            # Futures contracts have format: BTC/USDT:USDT-260501 or BTC/USDT-260501
            # The date suffix comes after the quote currency
            return False
        
        # Normalize: strip duplicate quote suffix (e.g., BTC/USDT:USDT -> BTC/USDT)
        normalized = symbol.split(":")[0] if ":" in symbol else symbol
        
        # Must have a valid quote currency
        if not any(normalized.endswith(q) for q in ("/USD", "/USDT", "/USDC")):
            return False
        
        # Skip stablecoin pairs (but allow USDT as quote)
        base = normalized.split("/")[0]
        if base in ("USDT", "USDC", "BUSD", "DAI", "TUSD"):
            return False
        
        return True
    
    def _calculate_metrics(
        self,
        exchange_id: str,
        symbol: str,
        ticker: Dict[str, Any],
    ) -> Optional[PairLiquidityMetrics]:
        """Calculate liquidity metrics for a pair."""
        try:
            # Normalize symbol (strip duplicate quote suffix only)
            normalized = symbol.split(":")[0] if ":" in symbol else symbol
            
            # Extract basic data
            volume_24h = ticker.get("quoteVolume", ticker.get("baseVolume", 0)) or 0
            bid = ticker.get("bid", 0) or 0
            ask = ticker.get("ask", 0) or 0
            last = ticker.get("last", ticker.get("close", 0)) or 0
            high = ticker.get("high", 0) or 0
            low = ticker.get("low", 0) or 0
            
            if last <= 0 or bid <= 0 or ask <= 0:
                return None
            
            # Calculate spread
            mid = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0
            spread_bps = spread_pct * 100  # Convert to basis points
            
            # Skip if spread too wide
            if spread_pct > self.max_spread_pct:
                return None
            
            # Skip if volume too low
            if volume_24h < self.min_volume_usd:
                return None
            
            # Calculate volatility (using daily range as proxy)
            daily_range = (high - low) / last if last > 0 else 0
            
            # Estimate order book depth from volume
            # This is a rough estimate - real depth requires order book fetch
            estimated_depth = volume_24h * 0.01  # 1% of daily volume
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(
                volume_24h=volume_24h,
                spread_pct=spread_pct,
                estimated_depth=estimated_depth,
            )
            
            # Calculate trading score (includes volatility)
            trading_score = self._calculate_trading_score(
                liquidity_score=liquidity_score,
                volatility=daily_range,
                volume_trend=0,  # Would need historical data
            )
            
            return PairLiquidityMetrics(
                symbol=normalized,
                exchange=exchange_id,
                volume_24h_usd=volume_24h,
                volume_1h_usd=volume_24h / 24,  # Rough estimate
                volume_trend=0.0,
                spread_pct=spread_pct,
                spread_bps=spread_bps,
                bid_depth_usd=estimated_depth,
                ask_depth_usd=estimated_depth,
                orderbook_imbalance=0.0,
                volatility_24h=daily_range,
                atr_pct=daily_range * 0.5,  # Rough ATR estimate
                price=last,
                price_change_24h_pct=ticker.get("percentage", 0) or 0,
                liquidity_score=liquidity_score,
                trading_score=trading_score,
            )
            
        except Exception as e:
            logger.debug(f"Error calculating metrics for {symbol}: {e}")
            return None
    
    def _calculate_liquidity_score(
        self,
        volume_24h: float,
        spread_pct: float,
        estimated_depth: float,
    ) -> float:
        """Calculate composite liquidity score (0-100)."""
        # Volume score (0-40 points)
        # Log scale: $100K = 10, $1M = 20, $10M = 30, $100M+ = 40
        if volume_24h > 0:
            vol_score = min(40, np.log10(volume_24h / 100_000) * 10 + 10)
        else:
            vol_score = 0
        
        # Spread score (0-30 points)
        # Lower spread = higher score
        if spread_pct <= 0.01:
            spread_score = 30
        elif spread_pct <= 0.1:
            spread_score = 25
        elif spread_pct <= 0.2:
            spread_score = 20
        elif spread_pct <= 0.3:
            spread_score = 15
        elif spread_pct <= 0.5:
            spread_score = 10
        else:
            spread_score = max(0, 10 - spread_pct * 10)
        
        # Depth score (0-30 points)
        # Higher depth = higher score
        if estimated_depth > 1_000_000:
            depth_score = 30
        elif estimated_depth > 500_000:
            depth_score = 25
        elif estimated_depth > 100_000:
            depth_score = 20
        elif estimated_depth > 50_000:
            depth_score = 15
        else:
            depth_score = 10
        
        return vol_score + spread_score + depth_score
    
    def _calculate_trading_score(
        self,
        liquidity_score: float,
        volatility: float,
        volume_trend: float,
    ) -> float:
        """Calculate trading score (includes volatility for opportunity)."""
        # Base liquidity score (60%)
        base = liquidity_score * 0.6
        
        # Volatility bonus (30%)
        # Moderate volatility is good for trading
        if 0.02 <= volatility <= 0.05:
            vol_bonus = 30  # Ideal volatility
        elif 0.01 <= volatility <= 0.08:
            vol_bonus = 20  # Acceptable
        elif volatility > 0.1:
            vol_bonus = 10  # Too volatile
        else:
            vol_bonus = 15  # Low volatility
        
        # Volume trend bonus (10%)
        trend_bonus = (volume_trend + 1) * 5  # -1 to 1 -> 0 to 10
        
        return base + vol_bonus + trend_bonus
    
    async def select_best_pairs(
        self,
        n_pairs: int = None,
        exclude_existing: List[str] = None,
    ) -> List[PairSelection]:
        """
        Select the best pairs for trading.
        
        Args:
            n_pairs: Number of pairs to select (default: max_pairs)
            exclude_existing: Pairs to exclude (e.g., already trading)
            
        Returns:
            List of PairSelection sorted by score
        """
        n_pairs = n_pairs or self.max_pairs
        exclude = set(exclude_existing or [])
        
        # Get all tradeable pairs
        tradeable = [
            (symbol, metrics)
            for symbol, metrics in self.pair_metrics.items()
            if metrics.is_tradeable and symbol not in exclude
        ]
        
        # Sort by trading score
        tradeable.sort(key=lambda x: x[1].trading_score, reverse=True)
        
        # Select top N
        selections = []
        total_score = sum(m.trading_score for _, m in tradeable[:n_pairs])
        
        for symbol, metrics in tradeable[:n_pairs]:
            # Calculate allocation based on score
            if total_score > 0:
                allocation = (metrics.trading_score / total_score) * 100
            else:
                allocation = 100 / n_pairs
            
            # Determine reasons for selection
            reasons = self._get_selection_reasons(metrics)
            
            selection = PairSelection(
                symbol=symbol,
                exchange=metrics.exchange,
                allocation_pct=allocation,
                expected_return_pct=self._estimate_return(metrics),
                risk_score=self._calculate_risk_score(metrics),
                confidence=min(metrics.liquidity_score / 100, 0.95),
                reasons=reasons,
                liquidity_score=metrics.liquidity_score,
                volume_24h_usd=metrics.volume_24h_usd,
                spread_pct=metrics.spread_pct,
            )
            selections.append(selection)
        
        self.selected_pairs = selections
        self.rotations += 1
        
        logger.info(
            f"Selected {len(selections)} pairs for trading: "
            f"{[s.symbol for s in selections[:5]]}..."
        )
        
        return selections
    
    def _get_selection_reasons(self, metrics: PairLiquidityMetrics) -> List[str]:
        """Get reasons why a pair was selected."""
        reasons = []
        
        if metrics.volume_24h_usd > 10_000_000:
            reasons.append("High volume (> $10M)")
        elif metrics.volume_24h_usd > 1_000_000:
            reasons.append("Good volume (> $1M)")
        
        if metrics.spread_pct < 0.1:
            reasons.append("Tight spread (< 0.1%)")
        elif metrics.spread_pct < 0.2:
            reasons.append("Reasonable spread (< 0.2%)")
        
        if metrics.liquidity_score > 80:
            reasons.append("Excellent liquidity")
        elif metrics.liquidity_score > 60:
            reasons.append("Good liquidity")
        
        if 0.02 <= metrics.volatility_24h <= 0.06:
            reasons.append("Optimal volatility")
        
        return reasons
    
    def _estimate_return(self, metrics: PairLiquidityMetrics) -> float:
        """Estimate expected return based on metrics."""
        # Base estimate from volatility (more volatility = more opportunity)
        base_return = metrics.volatility_24h * 100 * 0.5  # 50% of daily vol
        
        # Adjust for volume trend
        if metrics.volume_trend > 0:
            base_return *= 1.2
        
        # Adjust for spread (wider spread = harder to profit)
        if metrics.spread_pct > 0.3:
            base_return *= 0.8
        
        return base_return
    
    def _calculate_risk_score(self, metrics: PairLiquidityMetrics) -> float:
        """Calculate risk score (0-1, lower = safer)."""
        risk = 0.0
        
        # Volatility risk
        if metrics.volatility_24h > 0.1:
            risk += 0.4
        elif metrics.volatility_24h > 0.05:
            risk += 0.2
        else:
            risk += 0.1
        
        # Spread risk
        if metrics.spread_pct > 0.3:
            risk += 0.3
        elif metrics.spread_pct > 0.1:
            risk += 0.15
        else:
            risk += 0.05
        
        # Volume risk (lower volume = higher risk)
        if metrics.volume_24h_usd < 500_000:
            risk += 0.3
        elif metrics.volume_24h_usd < 1_000_000:
            risk += 0.15
        else:
            risk += 0.05
        
        return min(risk, 1.0)
    
    async def _get_exchange(self, exchange_id: str) -> Any:
        """Get or create exchange connection."""
        if exchange_id not in self._exchanges:
            try:
                import ccxt.async_support as ccxt
                
                exchange_class = getattr(ccxt, exchange_id, None)
                if exchange_class is None:
                    raise ValueError(f"Unknown exchange: {exchange_id}")
                
                self._exchanges[exchange_id] = exchange_class({
                    "enableRateLimit": True,
                    "timeout": 30000,
                })
                
            except ImportError:
                logger.error("ccxt not installed - cannot connect to exchanges")
                raise
        
        return self._exchanges[exchange_id]
    
    def _update_history(self, metrics: Dict[str, PairLiquidityMetrics]):
        """Update liquidity history for trend analysis."""
        for symbol, m in metrics.items():
            if symbol not in self.liquidity_history:
                self.liquidity_history[symbol] = deque(maxlen=100)
            
            self.liquidity_history[symbol].append({
                "timestamp": m.timestamp,
                "volume": m.volume_24h_usd,
                "spread": m.spread_pct,
                "score": m.liquidity_score,
            })
    
    def get_pair_rankings(self, limit: int = 20) -> List[Tuple[str, float]]:
        """Get pairs ranked by liquidity score."""
        rankings = [
            (symbol, metrics.liquidity_score)
            for symbol, metrics in self.pair_metrics.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:limit]
    
    def get_volume_rankings(self, limit: int = 20) -> List[Tuple[str, float]]:
        """Get pairs ranked by 24h volume."""
        rankings = [
            (symbol, metrics.volume_24h_usd)
            for symbol, metrics in self.pair_metrics.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:limit]
    
    def get_spread_rankings(self, limit: int = 20) -> List[Tuple[str, float]]:
        """Get pairs ranked by spread (tightest first)."""
        rankings = [
            (symbol, metrics.spread_pct)
            for symbol, metrics in self.pair_metrics.items()
        ]
        rankings.sort(key=lambda x: x[1])
        return rankings[:limit]
    
    def blacklist_pair(self, symbol: str, reason: str = ""):
        """Blacklist a pair from future selection."""
        self.blacklisted_pairs.add(symbol)
        logger.info(f"Blacklisted {symbol}: {reason}")
    
    def unblacklist_pair(self, symbol: str):
        """Remove a pair from blacklist."""
        self.blacklisted_pairs.discard(symbol)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        return {
            "scan_count": self.scan_count,
            "pairs_found": self.pairs_found,
            "pairs_selected": len(self.selected_pairs),
            "blacklisted": len(self.blacklisted_pairs),
            "rotations": self.rotations,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "exchanges": self.exchanges,
            "tracked_symbols": len(self.pair_metrics),
        }
    
    async def continuous_scan(
        self,
        callback=None,
        stop_event: Optional[asyncio.Event] = None,
    ):
        """
        Continuously scan for liquid pairs.
        
        Args:
            callback: Function to call with new selections
            stop_event: Event to signal stopping
        """
        logger.info(f"Starting continuous scan (interval: {self.scan_interval}s)")
        
        while True:
            try:
                # Scan all pairs
                await self.scan_all_pairs()
                
                # Select best pairs
                selections = await self.select_best_pairs()
                
                # Call callback if provided
                if callback:
                    await callback(selections)
                
            except Exception as e:
                logger.error(f"Error in continuous scan: {e}")
            
            # Wait or stop
            if stop_event:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.scan_interval)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    pass  # Continue scanning
            else:
                await asyncio.sleep(self.scan_interval)
        
        logger.info("Continuous scan stopped")
    
    async def close(self):
        """Close all exchange connections."""
        for exchange in self._exchanges.values():
            try:
                await exchange.close()
            except Exception:
                pass
        self._exchanges.clear()


# Factory function
def create_liquidity_scanner(
    exchanges: List[str] = None,
    max_pairs: int = 20,
) -> MultiPairLiquidityScanner:
    """Create a configured Multi-Pair Liquidity Scanner."""
    return MultiPairLiquidityScanner(
        exchanges=exchanges or ["kraken", "coinbase", "bybit"],
        max_pairs=max_pairs,
        min_volume_usd=100_000,
        max_spread_pct=0.5,
        enable_auto_discovery=True,
    )
