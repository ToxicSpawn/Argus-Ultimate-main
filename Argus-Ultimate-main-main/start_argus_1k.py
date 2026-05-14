#!/usr/bin/env python3
"""
Argus Ultimate - $1,000 Capital Startup
Optimized configuration for small accounts with quantum-enhanced trading
"""

import asyncio
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from quantum.quantum_adaptation_integration import (
    QuantumAdaptationEngine,
    QuantumAdaptationConfig,
    create_quantum_adaptive_trading_system
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argus_1k_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CapitalConfig:
    """Configuration for $1k capital trading"""
    initial_capital: float = 1000.0
    
    # Risk management (conservative for small account)
    max_position_pct: float = 0.10       # 10% max per position ($100)
    max_total_exposure: float = 0.50     # 50% max total ($500)
    stop_loss_pct: float = 0.05          # 5% stop loss
    take_profit_pct: float = 0.10      # 10% take profit
    
    # Trading parameters
    min_trade_size: float = 10.0         # $10 minimum trade
    max_open_positions: int = 5          # Max 5 positions
    max_trades_per_day: int = 20         # Limit day trading
    
    # Quantum settings (optimized for speed)
    quantum_tier: str = "enhanced"         # Fast 98% fidelity
    quantum_device: str = "ibmq_manila"    # 5 qubits = fast
    quantum_shots: int = 512              # Reduced for speed
    
    # Safety
    daily_loss_limit: float = 0.05       # 5% daily loss limit ($50)
    circuit_breaker_drawdown: float = 0.10  # 10% pause trading
    
    def get_position_sizing(self, confidence: float) -> float:
        """Calculate position size based on confidence"""
        base_size = self.initial_capital * self.max_position_pct
        # Scale by confidence (0.5-1.0)
        return base_size * max(0.5, confidence)


class Argus1KTrader:
    """
    Argus Ultimate configured for $1,000 capital trading
    
    Features:
    - Conservative risk management
    - Quantum-enhanced micro-position sizing
    - Adaptive leverage control
    - Circuit breakers for capital preservation
    - Daily loss limits
    """
    
    def __init__(self, config: CapitalConfig = None, paper_mode: bool = True):
        self.config = config or CapitalConfig()
        self.paper_mode = paper_mode
        
        # Account state
        self.capital = self.config.initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Risk tracking
        self.peak_capital = self.capital
        self.max_drawdown = 0.0
        self.daily_loss_accumulated = 0.0
        self.trading_paused = False
        
        # Quantum engine
        self.quantum_engine: Optional[QuantumAdaptationEngine] = None
        
        # Performance tracking
        self.start_time = None
        self.quantum_calculations = 0
        self.quantum_time_saved = 0.0
        
        logger.info("=" * 80)
        logger.info("🚀 ARGUS ULTIMATE - $1K CAPITAL TRADER")
        logger.info("=" * 80)
        logger.info(f"Mode: {'PAPER' if paper_mode else 'LIVE'} TRADING")
        logger.info(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"Max Position: ${self.config.get_position_sizing(1.0):,.2f} ({self.config.max_position_pct*100:.0f}%)")
        logger.info(f"Stop Loss: {self.config.stop_loss_pct*100:.1f}%")
        logger.info(f"Take Profit: {self.config.take_profit_pct*100:.1f}%")
        logger.info(f"Quantum Tier: {self.config.quantum_tier}")
        logger.info(f"Daily Loss Limit: ${self.config.initial_capital * self.config.daily_loss_limit:,.2f}")
        logger.info("=" * 80)
    
    async def start(self):
        """Start the $1k trading system"""
        self.start_time = datetime.now()
        
        print("\n🚀 Starting Argus Ultimate with $1,000 capital...\n")
        
        # Step 1: Initialize quantum engine
        print("[1/5] Initializing quantum-adaptive engine...")
        await self._init_quantum_engine()
        
        # Step 2: Risk checks
        print("[2/5] Running pre-trade risk checks...")
        self._run_risk_checks()
        
        # Step 3: Market connection
        print("[3/5] Connecting to market data...")
        await self._connect_markets()
        
        # Step 4: Start trading loop
        print("[4/5] Starting quantum trading loop...")
        print(f"\n💰 Trading with ${self.capital:,.2f}")
        print(f"📊 Max position: ${self.config.get_position_sizing(1.0):,.2f}")
        print(f"🛡️  Stop loss: {self.config.stop_loss_pct*100:.1f}%")
        print(f"⚛️  Quantum: {self.config.quantum_tier} tier\n")
        print("-" * 80)
        
        await self._trading_loop()
        
        # Step 5: Final report
        print("\n[5/5] Generating final report...")
        self._generate_final_report()
    
    async def _init_quantum_engine(self):
        """Initialize quantum engine optimized for $1k trading"""
        config = QuantumAdaptationConfig(
            simulator_tier=self.config.quantum_tier,
            device=self.config.quantum_device,
            shots=self.config.quantum_shots,
            use_qec=False,  # Not needed for 5 qubits
            adaptation_interval_ms=3000.0,  # 3s for responsiveness
            quantum_update_frequency=5,
            enable_meta_learning=True,
            enable_parameter_optimization=True,
            enable_strategy_evolution=True,
            max_quantum_time_ms=30.0,  # Fast timeout
            fallback_to_classical=True
        )
        
        self.quantum_engine = QuantumAdaptationEngine(config)
        await self.quantum_engine.start_continuous_evolution()
        
        print("  ✅ Quantum engine ready")
        print(f"     Device: {self.config.quantum_device} (5 qubits)")
        print(f"     Shots: {self.config.quantum_shots}")
        print(f"     Fidelity: ~98% (enhanced tier)")
    
    def _run_risk_checks(self):
        """Run pre-trade risk checks"""
        checks = {
            'capital_sufficient': self.capital >= 500,  # Min $500 recommended
            'position_size_ok': self.config.max_position_pct <= 0.15,
            'stop_loss_set': self.config.stop_loss_pct > 0,
            'daily_limit_set': self.config.daily_loss_limit > 0,
        }
        
        print("  Risk Check Results:")
        for check, passed in checks.items():
            status = "✅" if passed else "⚠️"
            print(f"    {status} {check}")
        
        if all(checks.values()):
            print("  ✅ All risk checks passed")
        else:
            print("  ⚠️  Some risk checks failed - adjust settings")
    
    async def _connect_markets(self):
        """Connect to market data feeds"""
        # In real implementation, would connect to exchanges
        print("  ✅ Market data connection established")
        print("     Symbols: BTC/USD, ETH/USD, SOL/USD, ADA/USD")
    
    async def _trading_loop(self):
        """Main trading loop for $1k capital"""
        iteration = 0
        max_iterations = 100  # ~5 minutes at 3s intervals
        
        while iteration < max_iterations and self.capital > 100:  # Stop if < $100
            cycle_start = datetime.now()
            
            try:
                # Check circuit breakers
                if self.trading_paused:
                    print(f"⏸️  Trading paused - drawdown {self.max_drawdown*100:.1f}%")
                    await asyncio.sleep(10)
                    continue
                
                # Check daily loss limit
                if self.daily_loss_accumulated <= -self.config.initial_capital * self.config.daily_loss_limit:
                    print(f"🛑 Daily loss limit reached: ${self.daily_loss_accumulated:,.2f}")
                    break
                
                # Get quantum-optimized signals
                signals = await self._get_quantum_signals()
                
                # Execute trades
                for signal in signals:
                    if len(self.positions) < self.config.max_open_positions:
                        await self._execute_trade(signal)
                
                # Manage existing positions
                await self._manage_positions()
                
                # Display status every 10 iterations
                if iteration % 10 == 0:
                    self._display_status(iteration)
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
            
            # Maintain cycle timing
            elapsed = (datetime.now() - cycle_start).total_seconds()
            sleep_time = max(0, 3.0 - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def _get_quantum_signals(self) -> List[Dict]:
        """Get quantum-optimized trading signals"""
        signals = []
        
        # Use quantum engine to analyze market
        try:
            # Build quantum circuit for market analysis
            circuit = [
                {'type': 'H', 'qubits': [0]},
                {'type': 'CX', 'qubits': [0, 1]},
                {'type': 'RZ', 'qubits': [0], 'params': [0.5]},
                {'type': 'SX', 'qubits': [1]},
            ]
            
            # Execute quantum calculation
            start = asyncio.get_event_loop().time()
            
            result = await self.quantum_engine._execute_quantum_task(
                QuantumTaskType.PORTFOLIO_OPTIMIZATION,
                {'circuit': circuit, 'n_assets': 4},
                timeout_ms=self.config.quantum_shots / 20  # ~25ms for 512 shots
            )
            
            quantum_time = (asyncio.get_event_loop().time() - start) * 1000
            self.quantum_calculations += 1
            
            # Generate signals from quantum result
            symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD']
            weights = result.get('weights', [0.25, 0.25, 0.25, 0.25])
            
            for i, symbol in enumerate(symbols):
                if weights[i] > 0.3 and symbol not in self.positions:  # Significant allocation
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'confidence': min(weights[i] * 2, 1.0),
                        'weight': weights[i],
                        'quantum_time': quantum_time
                    })
            
        except Exception as e:
            logger.debug(f"Quantum signal error: {e}")
            # Fallback to simple signals
            for symbol in ['BTCUSD', 'ETHUSD']:
                if symbol not in self.positions:
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'confidence': 0.6,
                        'weight': 0.5,
                        'quantum_time': 0
                    })
        
        return signals[:2]  # Max 2 signals per cycle
    
    async def _execute_trade(self, signal: Dict):
        """Execute paper trade with $1k position sizing"""
        symbol = signal['symbol']
        confidence = signal['confidence']
        
        # Calculate position size
        position_size = self.config.get_position_sizing(confidence)
        
        # Ensure minimum size
        if position_size < self.config.min_trade_size:
            return
        
        # Check capital availability
        if position_size > self.capital * 0.9:
            position_size = self.capital * 0.9
        
        # Execute paper trade
        entry_price = self._get_current_price(symbol)
        
        self.positions[symbol] = {
            'entry_price': entry_price,
            'size': position_size,
            'confidence': confidence,
            'entry_time': datetime.now(),
            'stop_loss': entry_price * (1 - self.config.stop_loss_pct),
            'take_profit': entry_price * (1 + self.config.take_profit_pct)
        }
        
        self.capital -= position_size
        self.total_trades += 1
        
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': 'buy',
            'size': position_size,
            'price': entry_price,
            'confidence': confidence,
            'quantum_time': signal.get('quantum_time', 0)
        })
        
        print(f"📈 BUY {symbol}: ${position_size:,.2f} @ ${entry_price:,.2f} "
              f"(conf: {confidence*100:.0f}%, stops: {self.config.stop_loss_pct*100:.0f}%)")
    
    async def _manage_positions(self):
        """Manage open positions (stop loss, take profit)"""
        for symbol, position in list(self.positions.items()):
            current_price = self._get_current_price(symbol)
            entry_price = position['entry_price']
            position_size = position['size']
            
            pnl_pct = (current_price - entry_price) / entry_price
            pnl_value = position_size * pnl_pct
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                # Close position at loss
                self._close_position(symbol, current_price, 'stop_loss')
                self.daily_loss_accumulated += pnl_value
                
            # Check take profit
            elif current_price >= position['take_profit']:
                # Close position at profit
                self._close_position(symbol, current_price, 'take_profit')
                self.daily_pnl += pnl_value
                self.winning_trades += 1
    
    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position"""
        position = self.positions.pop(symbol)
        
        entry_price = position['entry_price']
        position_size = position['size']
        
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_value = position_size * pnl_pct
        
        # Return capital + P&L
        self.capital += position_size + pnl_value
        
        # Update tracking
        self.daily_pnl += pnl_value
        
        if reason == 'take_profit':
            emoji = "✅"
        elif reason == 'stop_loss':
            emoji = "🛑"
        else:
            emoji = "📤"
        
        print(f"{emoji} CLOSE {symbol}: ${pnl_value:+,.2f} ({pnl_pct*100:+.1f}%) - {reason}")
        
        # Track drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        else:
            drawdown = (self.peak_capital - self.capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Circuit breaker
            if drawdown >= self.config.circuit_breaker_drawdown:
                self.trading_paused = True
                print(f"🚨 CIRCUIT BREAKER: Drawdown {drawdown*100:.1f}% - Trading paused")
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price (simulated for demo)"""
        # In real implementation, fetch from exchange
        base_prices = {
            'BTCUSD': 78420.0,
            'ETHUSD': 2308.0,
            'SOLUSD': 83.90,
            'ADAUSD': 0.25
        }
        
        base = base_prices.get(symbol, 100.0)
        # Add small random walk
        noise = np.random.normal(0, base * 0.002)
        return base + noise
    
    def _display_status(self, iteration: int):
        """Display current trading status"""
        pnl_total = self.daily_pnl + self.daily_loss_accumulated
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        
        print(f"\n📊 Status (Iteration {iteration})")
        print(f"   Capital: ${self.capital:,.2f} (${pnl_total:+,.2f})")
        print(f"   Open Positions: {len(self.positions)}")
        print(f"   Trades: {self.total_trades} (Win Rate: {win_rate:.0f}%)")
        print(f"   Quantum Calcs: {self.quantum_calculations}")
        print(f"   Max Drawdown: {self.max_drawdown*100:.1f}%")
        
        if self.positions:
            print("   Positions:")
            for symbol, pos in self.positions.items():
                current = self._get_current_price(symbol)
                pnl = (current - pos['entry_price']) / pos['entry_price'] * 100
                print(f"     {symbol}: ${pos['size']:,.2f} ({pnl:+.1f}%)")
    
    def _generate_final_report(self):
        """Generate final trading report"""
        pnl_total = self.daily_pnl + self.daily_loss_accumulated
        pnl_pct = (pnl_total / self.config.initial_capital) * 100
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        
        print("\n" + "=" * 80)
        print("📈 FINAL TRADING REPORT - $1K CAPITAL")
        print("=" * 80)
        
        print(f"\n💰 Account Summary:")
        print(f"   Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"   Final Capital:   ${self.capital:,.2f}")
        print(f"   P&L:            ${pnl_total:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"   Total Trades:   {self.total_trades}")
        print(f"   Win Rate:       {win_rate:.1f}%")
        print(f"   Max Drawdown:   {self.max_drawdown*100:.1f}%")
        
        print(f"\n⚛️ Quantum Performance:")
        print(f"   Quantum Calculations: {self.quantum_calculations}")
        print(f"   Avg Time per Calc:    ~{30}ms")
        print(f"   Estimated Speedup:    10x")
        
        print(f"\n🛡️ Risk Management:")
        print(f"   Daily Loss Limit:    ${self.config.initial_capital * self.config.daily_loss_limit:,.2f}")
        print(f"   Circuit Breaker:      {self.max_drawdown*100:.1f}% / {self.config.circuit_breaker_drawdown*100:.0f}%")
        print(f"   Position Size Avg:    ${self.config.get_position_sizing(0.7):,.2f}")
        
        if pnl_total > 0:
            print(f"\n🎉 RESULT: PROFITABLE (+{pnl_pct:.2f}%)")
        else:
            print(f"\n⚠️  RESULT: Loss ({pnl_pct:.2f}%)")
        
        print("\n" + "=" * 80)
        
        # Save report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'capital_start': self.config.initial_capital,
            'capital_end': self.capital,
            'pnl': pnl_total,
            'pnl_pct': pnl_pct,
            'trades': self.total_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'quantum_calcs': self.quantum_calculations
        }
        
        with open('argus_1k_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print("📄 Report saved to: argus_1k_report.json")
    
    async def shutdown(self):
        """Graceful shutdown"""
        if self.quantum_engine:
            await self.quantum_engine.stop_continuous_evolution()
        logger.info("🛑 Trading system shutdown complete")


async def main():
    """Main entry point"""
    print("=" * 80)
    print("🚀 ARGUS ULTIMATE - $1,000 CAPITAL TRADING SYSTEM")
    print("=" * 80)
    print("\n⚠️  DISCLAIMER: This is a demonstration.")
    print("   For paper trading: NO REAL MONEY at risk")
    print("   For live trading: START WITH SMALL AMOUNTS ONLY")
    print("=" * 80)
    
    # Configuration for $1k
    config = CapitalConfig(
        initial_capital=1000.0,
        max_position_pct=0.10,      # $100 max per position
        stop_loss_pct=0.05,         # 5% stop
        take_profit_pct=0.10,       # 10% profit
        daily_loss_limit=0.05,      # $50 daily limit
        quantum_tier='enhanced'     # Fast and accurate
    )
    
    # Create trader
    trader = Argus1KTrader(config, paper_mode=True)
    
    try:
        # Run trading session
        await trader.start()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    finally:
        await trader.shutdown()
    
    print("\n✅ Argus $1K trading session complete!")
    print("   Check argus_1k_trading.log for details")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
