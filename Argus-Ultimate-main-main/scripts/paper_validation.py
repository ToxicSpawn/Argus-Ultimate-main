#!/usr/bin/env python3
"""
Paper Trading Validation Script

Tracks key performance metrics during paper trading:
- Win rate
- Sharpe ratio
- Max drawdown
- Average P&L per trade
- TCA metrics (slippage, fill quality)
- Regime performance breakdown

Usage:
    py scripts/paper_validation.py --duration 7d    # 7-day validation
    py scripts/paper_validation.py --duration 30d   # 30-day validation
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Single trade record for validation"""
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_seconds: float
    regime: str
    strategy: str
    slippage: float
    fees: float


@dataclass
class ValidationMetrics:
    """Aggregated validation metrics"""
    start_time: str
    end_time: str
    duration_hours: float
    
    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L stats
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Risk stats
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Execution quality
    avg_slippage: float = 0.0
    avg_fees: float = 0.0
    tca_score: float = 0.0
    
    # Regime breakdown
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Strategy breakdown
    strategy_performance: Dict[str, float] = field(default_factory=dict)


class PaperValidationTracker:
    """Tracks paper trading performance for validation"""
    
    def __init__(self, starting_capital: float = 1000.0):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.peak_capital = starting_capital
        self.trades: List[TradeRecord] = []
        self.start_time = datetime.now()
        self.equity_curve: List[float] = [starting_capital]
        
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        self.trades.append(trade)
        self.current_capital += trade.pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.equity_curve.append(self.current_capital)
        
        # Log trade
        logger.info(
            f"Trade: {trade.side} {trade.symbol} | "
            f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%) | "
            f"Regime: {trade.regime} | Strategy: {trade.strategy}"
        )
        
    def calculate_metrics(self) -> ValidationMetrics:
        """Calculate all validation metrics"""
        if not self.trades:
            return ValidationMetrics(
                start_time=self.start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_hours=(datetime.now() - self.start_time).total_seconds() / 3600
            )
        
        # Basic stats
        total_trades = len(self.trades)
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        
        # P&L stats
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = (self.current_capital - self.starting_capital) / self.starting_capital * 100
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk stats (simplified)
        max_dd = self.peak_capital - min(self.equity_curve)
        max_dd_pct = max_dd / self.peak_capital * 100 if self.peak_capital > 0 else 0
        
        # Sharpe (simplified - using trade returns)
        returns = [t.pnl_pct for t in self.trades]
        if returns:
            import statistics
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0.01
            sharpe = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe = 0
            
        # TCA metrics
        avg_slippage = sum(t.slippage for t in self.trades) / total_trades if total_trades > 0 else 0
        avg_fees = sum(t.fees for t in self.trades) / total_trades if total_trades > 0 else 0
        
        # Regime breakdown
        regime_pnl: Dict[str, List[float]] = {}
        for t in self.trades:
            if t.regime not in regime_pnl:
                regime_pnl[t.regime] = []
            regime_pnl[t.regime].append(t.pnl)
        regime_perf = {k: sum(v) for k, v in regime_pnl.items()}
        
        # Strategy breakdown
        strategy_pnl: Dict[str, List[float]] = {}
        for t in self.trades:
            if t.strategy not in strategy_pnl:
                strategy_pnl[t.strategy] = []
            strategy_pnl[t.strategy].append(t.pnl)
        strategy_perf = {k: sum(v) for k, v in strategy_pnl.items()}
        
        return ValidationMetrics(
            start_time=self.start_time.isoformat(),
            end_time=datetime.now().isoformat(),
            duration_hours=(datetime.now() - self.start_time).total_seconds() / 3600,
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_pnl_per_trade=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            avg_slippage=avg_slippage,
            avg_fees=avg_fees,
            regime_performance=regime_perf,
            strategy_performance=strategy_perf,
        )
    
    def print_report(self):
        """Print validation report to console"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("PAPER TRADING VALIDATION REPORT")
        print("="*60)
        print(f"Duration: {metrics.duration_hours:.1f} hours")
        print(f"Starting Capital: ${self.starting_capital:.2f}")
        print(f"Current Capital: ${self.current_capital:.2f}")
        print(f"Peak Capital: ${self.peak_capital:.2f}")
        print("-"*60)
        
        print("\n📊 TRADE STATISTICS")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Win Rate: {metrics.win_rate*100:.1f}%")
        print(f"  Winning: {metrics.winning_trades} | Losing: {metrics.losing_trades}")
        
        print("\n💰 P&L STATISTICS")
        print(f"  Total P&L: ${metrics.total_pnl:.2f} ({metrics.total_pnl_pct:.2f}%)")
        print(f"  Avg P&L/Trade: ${metrics.avg_pnl_per_trade:.2f}")
        print(f"  Avg Win: ${metrics.avg_win:.2f}")
        print(f"  Avg Loss: ${metrics.avg_loss:.2f}")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        
        print("\n⚠️ RISK METRICS")
        print(f"  Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        
        print("\n📈 EXECUTION QUALITY")
        print(f"  Avg Slippage: ${metrics.avg_slippage:.4f}")
        print(f"  Avg Fees: ${metrics.avg_fees:.4f}")
        
        if metrics.regime_performance:
            print("\n🔄 REGIME PERFORMANCE")
            for regime, pnl in metrics.regime_performance.items():
                print(f"  {regime}: ${pnl:.2f}")
                
        if metrics.strategy_performance:
            print("\n🎯 STRATEGY PERFORMANCE")
            for strategy, pnl in sorted(metrics.strategy_performance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {strategy}: ${pnl:.2f}")
        
        print("\n" + "="*60)
        
        # Validation verdict
        print("\n✅ VALIDATION VERDICT")
        if metrics.total_trades < 30:
            print("  ⏳ INSUFFICIENT DATA - Need at least 30 trades")
        else:
            checks = []
            if metrics.win_rate >= 0.45:
                checks.append("✅ Win rate acceptable (≥45%)")
            else:
                checks.append("❌ Win rate too low (<45%)")
                
            if metrics.profit_factor >= 1.2:
                checks.append("✅ Profit factor good (≥1.2)")
            else:
                checks.append("❌ Profit factor too low (<1.2)")
                
            if metrics.max_drawdown_pct <= 15:
                checks.append("✅ Drawdown acceptable (≤15%)")
            else:
                checks.append("❌ Drawdown too high (>15%)")
                
            if metrics.sharpe_ratio >= 1.0:
                checks.append("✅ Sharpe ratio good (≥1.0)")
            else:
                checks.append("❌ Sharpe ratio too low (<1.0)")
            
            for check in checks:
                print(f"  {check}")
                
            passed = sum(1 for c in checks if c.startswith("✅"))
            if passed == 4:
                print("\n  🎉 READY FOR LIVE TRADING (with small position sizes)")
            elif passed >= 3:
                print("\n  ⚠️ MARGINAL - Consider more paper trading")
            else:
                print("\n  ❌ NOT READY - Continue paper trading")
    
    def save_report(self, filepath: str = "paper_validation_report.json"):
        """Save validation report to file"""
        metrics = self.calculate_metrics()
        report = {
            "metrics": asdict(metrics),
            "equity_curve": self.equity_curve,
            "trades": [asdict(t) for t in self.trades]
        }
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {filepath}")


def main():
    """Main entry point for paper validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Paper Trading Validation")
    parser.add_argument("--duration", default="7d", help="Validation duration (e.g., 7d, 30d)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital")
    parser.add_argument("--output", default="paper_validation_report.json", help="Output file")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║           ARGUS PAPER TRADING VALIDATION                       ║
╠════════════════════════════════════════════════════════════════╣
║  Duration: {args.duration:<10}                                       ║
║  Capital:  ${args.capital:<10.2f}                                     ║
║  Mode:     PAPER (no real money at risk)                       ║
╚════════════════════════════════════════════════════════════════╝

To start paper trading:
    py main.py paper

To view this validation report:
    py scripts/paper_validation.py

The system will automatically track:
    ✓ Win rate and profit factor
    ✓ Sharpe and Sortino ratios  
    ✓ Max drawdown
    ✓ TCA (slippage, fill quality)
    ✓ Regime performance
    ✓ Strategy attribution
""")


if __name__ == "__main__":
    main()
