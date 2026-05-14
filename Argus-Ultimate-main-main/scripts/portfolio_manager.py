"""
Portfolio Manager - Complete Portfolio Management System

Features:
- Position sizing (Kelly, fixed fraction, volatility-based)
- Portfolio allocation across symbols
- Risk management (max drawdown, position limits)
- Portfolio rebalancing (threshold-based, time-based)
- Integration with learning system

Usage:
    from scripts.portfolio_manager import PortfolioManager
    
    # Create
    pm = PortfolioManager(
        capital=10000,
        max_risk=0.02,
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    )
    
    # Process signal
    signal = pm.process_signal(signal, 'BTC/USDT', 50000)
    
    # Get allocation
    allocation = pm.get_allocation()
    
    # Rebalance
    pm.rebalance()
    
    # Get risk
    risk = pm.get_risk_metrics()

Run: py scripts/portfolio_manager.py
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# POSITION SIZING STRATEGIES
# ============================================================================

class PositionSizer:
    """Position sizing with multiple strategies."""
    
    @staticmethod
    def kelly(win_rate: float, avg_win: float, avg_loss: float, 
              fraction: float = 0.25) -> float:
        """
        Kelly Criterion.
        
        f* = (bp - q) / b
        where: b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        kelly = max(0, min(kelly, fraction))
        
        return kelly
    
    @staticmethod
    def fixed_fraction(risk: float, capital: float, 
                       stop_pct: float = 0.02) -> float:
        """
        Fixed fraction risk per trade.
        
        size = (capital * risk) / stop_distance
        """
        if stop_pct <= 0:
            return 0
        
        position_size = (capital * risk) / stop_pct
        return position_size / capital
    
    @staticmethod
    def volatility_based(target_vol: float, historical_vol: float,
                         capital: float) -> float:
        """
        Volatility-based position sizing.
        
        size = (target_vol / historical_vol) * capital
        """
        if historical_vol <= 0:
            return 0.01
        
        vol_scalar = target_vol / historical_vol
        return min(max(vol_scalar, 0.25), 4.0)
    
    @staticmethod
    def equal_weight(n_symbols: int) -> float:
        """Equal weight across N symbols."""
        return 1.0 / max(n_symbols, 1)


# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    """Risk management with limits and safeguards."""
    
    def __init__(
        self,
        max_position_size: float = 0.2,
        max_daily_loss: float = 0.05,
        max_drawdown: float = 0.15,
        max_correlation: float = 0.7,
        max_leverage: float = 1.0
    ):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_correlation = max_correlation
        self.max_leverage = max_leverage
        
        self.peak_capital = 0
        self.daily_pnl = 0
        self.daily_start = 0
        
        logger.info(f"Risk Manager: max_pos={max_position_size}, max_dd={max_drawdown}")
    
    def check_position(self, position_size: float, capital: float) -> bool:
        """Check if position size is within limits."""
        if position_size > self.max_position_size:
            logger.warning(f"Position {position_size:.1%} exceeds max {self.max_position_size:.1%}")
            return False
        return True
    
    def check_daily_loss(self, capital: float) -> bool:
        """Check if daily loss limit reached."""
        if self.peak_capital == 0:
            self.peak_capital = capital
            self.daily_start = capital
        
        self.daily_pnl = capital - self.daily_start
        loss_pct = -self.daily_pnl / self.daily_start
        
        if loss_pct >= self.max_daily_loss:
            logger.warning(f"Daily loss {loss_pct:.1%} exceeds max {self.max_daily_loss:.1%}")
            return False
        
        return True
    
    def check_drawdown(self, capital: float) -> bool:
        """Check if drawdown limit reached."""
        if capital > self.peak_capital:
            self.peak_capital = capital
        
        dd = (self.peak_capital - capital) / self.peak_capital
        
        if dd >= self.max_drawdown:
            logger.warning(f"Drawdown {dd:.1%} exceeds max {self.max_drawdown:.1%}")
            return False
        
        return True
    
    def get_available_capital(self, capital: float, current_exposure: float) -> float:
        """Get available capital for new positions."""
        available = capital * (self.max_position_size - current_exposure)
        return max(available, 0)


# ============================================================================
# PORTFOLIO REBALANCER
# ============================================================================

class PortfolioRebalancer:
    """Portfolio rebalancing strategies."""
    
    def __init__(
        self,
        threshold: float = 0.05,
        rebalance_interval: int = 0,
        method: str = 'threshold'
    ):
        self.threshold = threshold
        self.rebalance_interval = rebalance_interval
        self.method = method
        self.last_rebalance = datetime.now(timezone.utc)
        
        logger.info(f"Rebalancer: method={method}, threshold={threshold}")
    
    def should_rebalance(self, current_weights: Dict[str, float], 
                        target_weights: Dict[str, float]) -> bool:
        """Check if rebalancing needed."""
        if self.method == 'threshold':
            return self._check_threshold(current_weights, target_weights)
        elif self.method == 'time':
            return self._check_time()
        return False
    
    def _check_threshold(self, current: Dict[str, float], 
                       target: Dict[str, float]) -> bool:
        """Check if any weight deviation exceeds threshold."""
        for symbol in target:
            curr = current.get(symbol, 0)
            tgt = target[symbol]
            
            if abs(curr - tgt) > self.threshold:
                return True
        return False
    
    def _check_time(self) -> bool:
        """Check if time-based rebalance needed."""
        if self.rebalance_interval <= 0:
            return False
        
        elapsed = (datetime.now(timezone.utc) - self.last_rebalance).total_seconds()
        return elapsed > self.rebalance_interval
    
    def get_new_weights(self, current: Dict[str, float], target: Dict[str, float],
                      capital: float) -> Dict[str, float]:
        """Calculate new weights to rebalance."""
        return target.copy()
    
    def mark_rebalanced(self):
        """Mark that rebalancing occurred."""
        self.last_rebalance = datetime.now(timezone.utc)


# ============================================================================
# MAIN PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """
    Complete Portfolio Manager.
    
    Features:
    - Position sizing (multiple strategies)
    - Portfolio allocation
    - Risk management
    - Rebalancing
    - Learning integration
    """
    
    def __init__(
        self,
        capital: float = 10000,
        max_risk: float = 0.02,
        symbols: List[str] = None,
        position_strategy: str = 'kelly',
        rebalance_method: str = 'threshold',
        rebalance_threshold: float = 0.05
    ):
        self.capital = capital
        self.initial_capital = capital
        self.max_risk = max_risk
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # Position sizing strategy
        self.position_strategy = position_strategy
        
        # Track positions and performance
        self.positions = {s: {'size': 0, 'entry': 0, 'pnl': 0} for s in self.symbols}
        self.weights = {s: 1.0/len(self.symbols) for s in self.symbols}
        self.target_weights = {s: 1.0/len(self.symbols) for s in self.symbols}
        
        # Performance tracking
        self.trade_history = deque(maxlen=100)
        self.returns = deque(maxlen=500)
        
        # Risk manager
        self.risk_manager = RiskManager()
        
        # Rebalancer
        self.rebalancer = PortfolioRebalancer(
            method=rebalance_method,
            threshold=rebalance_threshold
        )
        
        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0
        self.daily_start = capital
        
        logger.info("=" * 60)
        logger.info("PORTFOLIO MANAGER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Capital: ${capital:,.2f}")
        logger.info(f"Max Risk: {max_risk:.1%}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Position Strategy: {position_strategy}")
        logger.info("=" * 60)
    
    # ========================================
    # POSITION SIZING
    # ========================================
    
    def calculate_position_size(
        self,
        symbol: str,
        confidence: float,
        stop_pct: float = 0.02,
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = 1.0
    ) -> float:
        """Calculate position size for symbol."""
        
        if self.position_strategy == 'kelly':
            kelly_size = PositionSizer.kelly(win_rate, avg_win, avg_loss)
            size = kelly_size * confidence
        elif self.position_strategy == 'fixed':
            size = PositionSizer.fixed_fraction(self.max_risk, self.capital, stop_pct)
        elif self.position_strategy == 'volatility':
            vol = self._get_historical_vol(symbol)
            size = PositionSizer.volatility_based(0.02, vol, self.capital)
        else:  # equal
            size = PositionSizer.equal_weight(len(self.symbols))
        
        # Apply confidence
        size *= confidence
        
        # Limit to max position
        size = min(size, self.risk_manager.max_position_size)
        
        return size
    
    def _get_historical_vol(self, symbol: str) -> float:
        """Get historical volatility for symbol."""
        return 0.02  # 2% default
    
    # ========================================
    # SIGNAL PROCESSING
    # ========================================
    
    def process_signal(
        self,
        signal,
        symbol: str,
        price: float,
        win_rate: float = 0.55
    ) -> Any:
        """Process trading signal with risk management."""
        
        if symbol not in self.symbols:
            return signal
        
        # Get confidence
        confidence = getattr(signal, 'confidence', 0.5)
        
        # Calculate position size
        position_size = self.calculate_position_size(
            symbol, confidence, win_rate=win_rate
        )
        
        # Check risk limits
        current_exposure = sum(p['size'] for p in self.positions.values())
        
        if not self.risk_manager.check_position(position_size, self.capital):
            position_size = self.risk_manager.get_available_capital(
                self.capital, current_exposure
            )
        
        if not self.risk_manager.check_daily_loss(self.capital):
            signal.action = 'hold'
            signal.reasoning = "daily_loss_limit"
            return signal
        
        if not self.risk_manager.check_drawdown(self.capital):
            signal.action = 'hold'
            signal.reasoning = "drawdown_limit"
            return signal
        
        # Update position size
        self.positions[symbol]['size'] = position_size
        self.positions[symbol]['entry'] = price
        
        # Update signal
        signal.position_size = position_size
        signal.risk_adjusted = True
        
        return signal
    
    # ========================================
    # PORTFOLIO ALLOCATION
    # ========================================
    
    def get_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation."""
        return self.weights.copy()
    
    def get_target_allocation(self) -> Dict[str, float]:
        """Get target allocation."""
        return self.target_weights.copy()
    
    def set_target_allocation(self, weights: Dict[str, float]):
        """Set target allocation."""
        total = sum(weights.values())
        if total > 0:
            self.target_weights = {s: w/total for s, w in weights.items()}
            logger.info(f"Target allocation: {self.target_weights}")
    
    def update_weights(self, prices: Dict[str, float]):
        """Update current weights based on prices."""
        total_value = 0
        
        for symbol in self.symbols:
            pos = self.positions[symbol]
            if pos['size'] > 0 and pos['entry'] > 0:
                pnl = (prices.get(symbol, pos['entry']) / pos['entry'] - 1) * pos['size']
                pos['pnl'] = pnl
                total_value += pos['size'] * pos['entry']
        
        # Update weights
        if total_value > 0:
            for symbol in self.symbols:
                pos = self.positions[symbol]
                value = pos['size'] * pos.get('entry', prices.get(symbol, 1))
                self.weights[symbol] = value / total_value
    
    # ========================================
    # REBALANCING
    # ========================================
    
    def rebalance(self, prices: Dict[str, float] = None) -> bool:
        """Rebalance portfolio if needed."""
        prices = prices or {s: 1 for s in self.symbols}
        
        should_rebalance = self.rebalancer.should_rebalance(
            self.weights, self.target_weights
        )
        
        if should_rebalance:
            self.rebalancer.mark_rebalanced()
            
            # Adjust positions to target
            for symbol in self.symbols:
                target = self.target_weights.get(symbol, 0)
                current = self.weights.get(symbol, 0)
                diff = target - current
                
                pos = self.positions[symbol]
                if diff > 0.05:  # Buy
                    pos['size'] += diff
                elif diff < -0.05:  # Sell
                    pos['size'] += diff  # diff is negative
            
            self.update_weights(prices)
            logger.info("Portfolio rebalanced")
            return True
        
        return False
    
    # ========================================
    # TRADE MANAGEMENT
    # ========================================
    
    def on_trade(
        self,
        symbol: str,
        pnl: float,
        entry_price: float,
        exit_price: float
    ):
        """Record trade result."""
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'entry': entry_price,
            'exit': exit_price,
            'time': datetime.now(timezone.utc)
        })
        
        self.capital += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Reset position
        self.positions[symbol] = {'size': 0, 'entry': 0, 'pnl': 0}
        
        # Track returns
        ret = pnl / self.initial_capital
        self.returns.append(ret)
        
        logger.info(f"Trade: {symbol} PnL: ${pnl:,.2f} | Capital: ${self.capital:,.2f}")
    
    def on_bar_close(self, symbol: str, price: float):
        """Update position on bar close."""
        pos = self.positions[symbol]
        if pos['size'] > 0 and pos['entry'] > 0:
            pnl = (price / pos['entry'] - 1) * pos['size'] * pos['entry']
            pos['pnl'] = pnl
    
    # ========================================
    # RISK METRICS
    # ========================================
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        
        returns_arr = np.array(list(self.returns)) if self.returns else np.array([0])
        
        if len(returns_arr) > 1:
            volatility = np.std(returns_arr) * np.sqrt(252)
            sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 0.0001) * np.sqrt(252)
        else:
            volatility = 0
            sharpe = 0
        
        max_dd = self._calculate_max_drawdown()
        
        return {
            'capital': self.capital,
            'total_return': (self.capital / self.initial_capital - 1),
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'exposure': sum(p['size'] for p in self.positions.values())
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.returns:
            return 0
        
        cumulative = np.cumsum([0] + list(self.returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1)
        
        return np.max(drawdown) if len(drawdown) > 0 else 0
    
    # ========================================
    # PERFORMANCE
    # ========================================
    
    def get_performance(self) -> Dict:
        """Get performance summary."""
        metrics = self.get_risk_metrics()
        
        return {
            'capital': self.capital,
            'return': metrics['total_return'],
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown'],
            'positions': {s: p['size'] for s, p in self.positions.items()}
        }
    
    def print_summary(self):
        """Print portfolio summary."""
        perf = self.get_performance()
        
        print("=" * 60)
        print("PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"Capital:        ${perf['capital']:,.2f}")
        print(f"Return:        {perf['return']:.2%}")
        print(f"Trades:        {perf['trades']}")
        print(f"Win Rate:      {perf['win_rate']:.1%}")
        print(f"Sharpe Ratio:  {perf['sharpe']:.2f}")
        print(f"Max Drawdown: {perf['max_dd']:.2%}")
        print("-" * 60)
        
        print("Positions:")
        for symbol, size in perf['positions'].items():
            if size > 0:
                print(f"  {symbol}: {size:.1%}")
        
        print("=" * 60)


# ============================================================================
# DEMO
# ============================================================================

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("PORTFOLIO MANAGER DEMO")
    print("=" * 60)
    print()
    
    # Create portfolio manager
    pm = PortfolioManager(
        capital=10000,
        max_risk=0.02,
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        position_strategy='kelly',
        rebalance_threshold=0.05
    )
    
    print()
    
    # Simulate some trades
    trades = [
        ('BTC/USDT', 100, 50000, 51000),
        ('ETH/USDT', 50, 3000, 3100),
        ('SOL/USDT', -25, 100, 95),
        ('BTC/USDT', 150, 51000, 52500),
        ('ETH/USDT', 75, 3100, 3200),
    ]
    
    print("Simulated Trades:")
    for symbol, pnl, entry, exit in trades:
        pm.on_trade(symbol, pnl, entry, exit)
    
    print()
    
    # Print summary
    pm.print_summary()
    
    print()
    
    # Get risk metrics
    risk = pm.get_risk_metrics()
    print("Risk Metrics:")
    print(f"  Volatility: {risk['volatility']:.2%}")
    print(f"  Sharpe:     {risk['sharpe_ratio']:.2f}")
    print(f"  Max DD:     {risk['max_drawdown']:.2%}")
    print(f"  Exposure:  {risk['exposure']:.1%}")
    
    print()
    print("=" * 60)
    print("PORTFOLIO MANAGER COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()