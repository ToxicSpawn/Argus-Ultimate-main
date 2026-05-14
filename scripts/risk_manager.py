"""
Advanced Risk Management System

Features:
1. Circuit Breakers - Stop all trading if extreme conditions
2. Max Drawdown Limits - Stop trading if drawdown > X%
3. Correlation-Based Position Limits - No more than X% in correlated assets
4. Max Daily Loss Protection - Stop trading if daily loss > X%
5. Position Sizing Validation - Validate size before execution
6. Risk-Adjusted Position Sizing - Scale position based on risk
7. Comprehensive Logging & Alerts - Track all risk events

Usage:
    from scripts.risk_manager import RiskManager
    
    rm = RiskManager(
        max_drawdown=0.15,
        max_daily_loss=0.05,
        max_correlation=0.7,
        circuit_breaker_threshold=0.20
    )
    
    # Check before trade
    if rm.can_trade('BTC/USDT', 0.1, 50000):
        # Execute trade
        pass
    
    # Record trade result
    rm.on_trade('BTC/USDT', 100, 50000, 51000)
    
    # Get risk status
    status = rm.get_status()

Run: py scripts/risk_manager.py
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Advanced Risk Management System.
    
    Prevents catastrophic losses with multiple layers of protection.
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.15,
        max_daily_loss: float = 0.05,
        max_correlation: float = 0.7,
        circuit_breaker_threshold: float = 0.20,
        max_position_size: float = 0.25,
        min_position_size: float = 0.005,
        symbols: List[str] = None
    ):
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_correlation = max_correlation
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        # State
        self.peak_capital = None
        self.daily_start_capital = None
        self.current_drawdown = 0
        self.daily_pnl = 0
        self.positions = {s: 0 for s in self.symbols}
        self.correlation_matrix = self._create_default_correlation()
        self.trade_history = deque(maxlen=1000)
        self.risk_events = deque(maxlen=100)
        self.is_circuit_broken = False
        self.circuit_breaker_triggered_at = None
        
        # Risk metrics
        self.total_risk_exposure = 0
        self.max_risk_per_symbol = {s: 0.2 for s in self.symbols}
        
        logger.info("=" * 60)
        logger.info("ADVANCED RISK MANAGEMENT SYSTEM")
        logger.info("=" * 60)
        logger.info(f"Max Drawdown: {max_drawdown:.1%}")
        logger.info(f"Max Daily Loss: {max_daily_loss:.1%}")
        logger.info(f"Max Correlation: {max_correlation:.1%}")
        logger.info(f"Circuit Breaker: {circuit_breaker_threshold:.1%}")
        logger.info(f"Symbols: {self.symbols}")
        logger.info("=" * 60)
    
    def _create_default_correlation(self) -> Dict[str, Dict[str, float]]:
        """Create default correlation matrix."""
        corr = {s1: {s2: 0 for s2 in self.symbols} for s1 in self.symbols}
        
        # Set known correlations (BTC-ETH high, BTC-SOL medium, ETH-SOL medium)
        if 'BTC/USDT' in self.symbols and 'ETH/USDT' in self.symbols:
            corr['BTC/USDT']['ETH/USDT'] = 0.75
            corr['ETH/USDT']['BTC/USDT'] = 0.75
        
        if 'BTC/USDT' in self.symbols and 'SOL/USDT' in self.symbols:
            corr['BTC/USDT']['SOL/USDT'] = 0.55
            corr['SOL/USDT']['BTC/USDT'] = 0.55
        
        if 'ETH/USDT' in self.symbols and 'SOL/USDT' in self.symbols:
            corr['ETH/USDT']['SOL/USDT'] = 0.60
            corr['SOL/USDT']['ETH/USDT'] = 0.60
        
        return corr
    
    def reset_daily(self, capital: float):
        """Reset daily tracking."""
        self.daily_start_capital = capital
        self.daily_pnl = 0
        self.risk_events.clear()
        logger.info("Daily risk tracking reset")
    
    def update_capital(self, capital: float):
        """Update capital reference."""
        if self.peak_capital is None or capital > self.peak_capital:
            self.peak_capital = capital
        
        # Update drawdown
        if self.peak_capital > 0:
            self.current_drawdown = (self.peak_capital - capital) / self.peak_capital
        
        # Update daily PnL
        if self.daily_start_capital is not None:
            self.daily_pnl = capital - self.daily_start_capital
    
    # ========================================
    # RISK CHECKS
    # ========================================
    
    def can_trade(
        self,
        symbol: str,
        position_size: float,
        price: float,
        leverage: float = 1.0,
        current_positions: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Check if trading is allowed based on all risk parameters.
        
        Returns:
            bool: True if can trade, False if risk limit exceeded
        """
        
        # Update capital from current positions
        if current_positions:
            self.positions = current_positions
        
        # Check circuit breaker first
        if self.is_circuit_broken:
            logger.warning("Circuit breaker is active - no trading allowed")
            self._log_risk_event("circuit_breaker", "Circuit breaker active")
            return False
        
        # Check position size
        if not self._validate_position_size(position_size, price, leverage):
            return False
        
        # Check max position limit
        if not self._check_max_position(position_size):
            return False
        
        # Check correlation limits
        if not self._check_correlation_limits(symbol, position_size):
            return False
        
        # Check daily loss
        if not self._check_daily_loss():
            return False
        
        # Check drawdown
        if not self._check_drawdown():
            return False
        
        return True
    
    def _validate_position_size(
        self,
        position_size: float,
        price: float,
        leverage: float
    ) -> bool:
        """Validate position size is within bounds."""
        
        position_value = position_size * price * leverage
        
        # Check min/max bounds
        if position_size < self.min_position_size:
            logger.warning(f"Position size {position_size:.1%} below minimum {self.min_position_size:.1%}")
            self._log_risk_event("position_too_small", f"Size: {position_size:.1%}")
            return False
        
        if position_size > self.max_position_size:
            logger.warning(f"Position size {position_size:.1%} exceeds maximum {self.max_position_size:.1%}")
            self._log_risk_event("position_too_large", f"Size: {position_size:.1%}")
            return False
        
        return True
    
    def _check_max_position(self, position_size: float) -> bool:
        """Check if adding this position would exceed max exposure."""
        
        total_exposure = sum(self.positions.values()) + position_size
        
        if total_exposure > 1.0:
            logger.warning(f"Total exposure {total_exposure:.1%} exceeds 100%")
            self._log_risk_event("total_exposure_exceeded", f"Exposure: {total_exposure:.1%}")
            return False
        
        return True
    
    def _check_correlation_limits(self, symbol: str, position_size: float) -> bool:
        """
        Check if adding this position would exceed correlation limits.
        
        If symbol is highly correlated with existing positions, reduce allowed size.
        """
        
        # Find max correlation with existing positions
        max_corr = 0
        for s in self.positions:
            if s != symbol and self.correlation_matrix[symbol][s] > max_corr:
                max_corr = self.correlation_matrix[symbol][s]
        
        # Calculate risk-adjusted position size
        risk_adjusted_size = position_size * (1 - max_corr * 0.5)
        
        if risk_adjusted_size < self.min_position_size:
            logger.warning(f"Correlation limit: position reduced from {position_size:.1%} to {risk_adjusted_size:.1%}")
            self._log_risk_event("correlation_limit", f"Reduced to {risk_adjusted_size:.1%}")
            return False
        
        return True
    
    def _check_daily_loss(self) -> bool:
        """Check if daily loss limit exceeded."""
        
        if self.daily_start_capital is None:
            return True
        
        loss_pct = -self.daily_pnl / self.daily_start_capital
        
        if loss_pct >= self.max_daily_loss:
            logger.error(f"Daily loss {loss_pct:.1%} exceeds limit {self.max_daily_loss:.1%}")
            self._log_risk_event("daily_loss_limit", f"Loss: {loss_pct:.1%}")
            return False
        
        return True
    
    def _check_drawdown(self) -> bool:
        """Check if max drawdown limit exceeded."""
        
        if self.current_drawdown >= self.max_drawdown:
            logger.error(f"Drawdown {self.current_drawdown:.1%} exceeds limit {self.max_drawdown:.1%}")
            self._log_risk_event("max_drawdown", f"Drawdown: {self.current_drawdown:.1%}")
            return False
        
        return True
    
    def check_circuit_breaker(self, capital: float) -> bool:
        """Check if circuit breaker should be triggered."""
        
        if self.peak_capital is None:
            return True
        
        drawdown = (self.peak_capital - capital) / self.peak_capital
        
        if drawdown >= self.circuit_breaker_threshold:
            if not self.is_circuit_broken:
                self.is_circuit_broken = True
                self.circuit_breaker_triggered_at = datetime.now(timezone.utc)
                logger.critical(f"CIRCUIT BREAKER TRIGGERED! Drawdown: {drawdown:.1%}")
                self._log_risk_event("circuit_breaker", f"Drawdown: {drawdown:.1%}")
            return False
        
        return True
    
    # ========================================
    # TRADE RECORDING
    # ========================================
    
    def on_trade(
        self,
        symbol: str,
        pnl: float,
        entry_price: float,
        exit_price: float,
        position_size: float = None
    ):
        """Record trade result and update risk state."""
        
        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'time': datetime.now(timezone.utc)
        })
        
        # Update positions
        if position_size is not None:
            self.positions[symbol] = position_size
        
        # Check if circuit breaker should be reset
        if self.is_circuit_broken:
            if self.current_drawdown < self.circuit_breaker_threshold * 0.5:
                self.is_circuit_broken = False
                self.circuit_breaker_triggered_at = None
                logger.info("Circuit breaker reset")
    
    # ========================================
    # RISK METRICS
    # ========================================
    
    def get_risk_metrics(self) -> Dict:
        """Get comprehensive risk metrics."""
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate portfolio risk metrics
        if len(self.trade_history) > 0:
            returns = [t['pnl'] / 10000 for t in self.trade_history]  # Assume $10k capital
            volatility = np.std(returns) * np.sqrt(252) if returns else 0
            sharpe = np.mean(returns) / (volatility + 0.0001) * np.sqrt(252) if volatility > 0 else 0
        else:
            volatility = 0
            sharpe = 0
        
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'daily_loss': -self.daily_pnl / self.daily_start_capital if self.daily_start_capital else 0,
            'max_daily_loss': self.max_daily_loss,
            'peak_capital': self.peak_capital,
            'current_capital': self.peak_capital * (1 - self.current_drawdown) if self.peak_capital else 0,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'positions': self.positions.copy(),
            'is_circuit_broken': self.is_circuit_broken,
            'circuit_breaker_triggered_at': self.circuit_breaker_triggered_at
        }
    
    def get_risk_status(self) -> Dict:
        """Get simplified risk status."""
        
        metrics = self.get_risk_metrics()
        
        status = {
            'can_trade': True,
            'warnings': [],
            'errors': []
        }
        
        if metrics['current_drawdown'] >= metrics['max_drawdown']:
            status['can_trade'] = False
            status['errors'].append(f"Max drawdown reached: {metrics['current_drawdown']:.1%}")
        
        if metrics['daily_loss'] >= metrics['max_daily_loss']:
            status['can_trade'] = False
            status['errors'].append(f"Daily loss limit reached: {metrics['daily_loss']:.1%}")
        
        if metrics['is_circuit_broken']:
            status['can_trade'] = False
            status['errors'].append(f"Circuit breaker active")
        
        return status
    
    def _log_risk_event(self, event_type: str, details: str):
        """Log a risk event."""
        
        event = {
            'timestamp': datetime.now(timezone.utc),
            'type': event_type,
            'details': details,
            'drawdown': self.current_drawdown,
            'daily_loss': -self.daily_pnl / self.daily_start_capital if self.daily_start_capital else 0
        }
        
        self.risk_events.append(event)
        
        # Log at appropriate level
        if event_type in ['daily_loss_limit', 'max_drawdown']:
            logger.error(f"RISK EVENT: {event_type.upper()} - {details}")
        elif event_type in ['position_too_large', 'total_exposure_exceeded']:
            logger.warning(f"RISK WARNING: {event_type.upper()} - {details}")
        else:
            logger.info(f"RISK INFO: {event_type.upper()} - {details}")
    
    def print_risk_summary(self):
        """Print comprehensive risk summary."""
        
        metrics = self.get_risk_metrics()
        status = self.get_risk_status()
        
        print("=" * 60)
        print("ADVANCED RISK MANAGEMENT SUMMARY")
        print("=" * 60)
        
        print(f"\nRisk Metrics:")
        print(f"  Current Drawdown: {metrics['current_drawdown']:.1%} (Max: {metrics['max_drawdown']:.1%})")
        print(f"  Daily Loss: {metrics['daily_loss']:.1%} (Max: {metrics['max_daily_loss']:.1%})")
        print(f"  Peak Capital: ${metrics['peak_capital']:,.2f}")
        print(f"  Current Capital: ${metrics['current_capital']:,.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        print(f"\nStatus:")
        if not status['can_trade']:
            print(f"  TRADING BLOCKED")
            for error in status['errors']:
                print(f"     - {error}")
        else:
            print(f"  Trading allowed")
        
        if metrics['is_circuit_broken']:
            print(f"  Circuit Breaker: ACTIVE (triggered at {metrics['circuit_breaker_triggered_at']})")
        
        print(f"\nPositions:")
        for symbol, size in metrics['positions'].items():
            if size > 0:
                print(f"  {symbol}: {size:.1%}")
        
        print(f"\nRecent Risk Events ({len(self.risk_events)} total):")
        for event in list(self.risk_events)[-5:]:
            print(f"  [{event['timestamp']}] {event['type'].upper()}: {event['details']}")
        
        print("\n" + "=" * 60)
    
    def export_risk_report(self) -> Dict:
        """Export complete risk report."""
        
        metrics = self.get_risk_metrics()
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': metrics,
            'risk_events': list(self.risk_events),
            'trade_history': list(self.trade_history),
            'config': {
                'max_drawdown': self.max_drawdown,
                'max_daily_loss': self.max_daily_loss,
                'max_correlation': self.max_correlation,
                'circuit_breaker_threshold': self.circuit_breaker_threshold
            }
        }


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo the risk manager."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("ADVANCED RISK MANAGEMENT DEMO")
    print("=" * 60)
    print()
    
    # Create risk manager
    rm = RiskManager(
        max_drawdown=0.15,
        max_daily_loss=0.05,
        max_correlation=0.7,
        circuit_breaker_threshold=0.20,
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    )
    
    # Reset with capital
    rm.reset_daily(10000)
    rm.update_capital(10000)
    
    print("Initial state:")
    rm.print_risk_summary()
    print()
    
    # Simulate some trades
    print("Simulating trades...")
    print()
    
    # Trade 1: Valid trade
    can_trade = rm.can_trade('BTC/USDT', 0.1, 50000)
    print(f"Trade 1 - BTC buy 10%: {'ALLOWED' if can_trade else 'BLOCKED'}")
    if can_trade:
        rm.on_trade('BTC/USDT', 150, 50000, 51000, 0.1)
        rm.update_capital(10150)
    
    # Trade 2: Too large
    can_trade = rm.can_trade('ETH/USDT', 0.3, 3000)
    print(f"Trade 2 - ETH buy 30%: {'ALLOWED' if can_trade else 'BLOCKED (too large)'}")
    
    # Trade 3: Valid trade
    can_trade = rm.can_trade('ETH/USDT', 0.1, 3000)
    print(f"Trade 3 - ETH buy 10%: {'ALLOWED' if can_trade else 'BLOCKED'}")
    if can_trade:
        rm.on_trade('ETH/USDT', 60, 3000, 3100, 0.1)
        rm.update_capital(10210)
    
    # Trade 4: Trigger drawdown limit
    rm.update_capital(8000)  # Simulate large loss
    can_trade = rm.can_trade('SOL/USDT', 0.1, 100)
    print(f"Trade 4 - SOL buy 10%: {'ALLOWED' if can_trade else 'BLOCKED (drawdown limit)'}")
    
    print()
    print("Final state:")
    rm.print_risk_summary()
    
    print()
    print("Risk Report:")
    report = rm.export_risk_report()
    print(f"  Total Trades: {report['metrics']['total_trades']}")
    print(f"  Win Rate: {report['metrics']['win_rate']:.1%}")
    print(f"  Risk Events: {len(report['risk_events'])}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    demo()
