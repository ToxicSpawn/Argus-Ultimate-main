#!/usr/bin/env python3
"""Quick paper trading test to demonstrate learning performance."""

import asyncio
from main import Argus

async def test():
    print('=' * 70)
    print('ARGUS ULTIMATE - MARKET-SPEED LEARNING PERFORMANCE TEST')
    print('=' * 70)
    print()
    
    system = Argus(mode='paper', capital=1000)
    
    # Run for 60 cycles
    print('Running 60 trading cycles with MARKET-SPEED learning...')
    print('-' * 70)
    
    for i in range(60):
        result = await system.run_cycle()
        if (i + 1) % 15 == 0:
            val = result['value']
            trades = result['trades']
            print(f'  Cycle {i+1:3d} | Value: ${val:,.2f} | Trades: {trades}')
    
    print('-' * 70)
    print()
    
    # Final results
    final_value = system._portfolio_value(50000)
    total_pnl = final_value - 1000
    return_pct = (final_value / 1000 - 1) * 100
    
    winning = [t for t in system.trades if t.pnl > 0]
    losing = [t for t in system.trades if t.pnl <= 0]
    win_rate = len(winning) / max(len(system.trades), 1)
    
    print('=' * 70)
    print('PERFORMANCE RESULTS')
    print('=' * 70)
    print(f'Initial Capital:    $1,000.00')
    print(f'Final Value:        ${final_value:,.2f}')
    print(f'Total P&L:          ${total_pnl:+,.2f}')
    print(f'Return:             {return_pct:+.2f}%')
    print(f'Max Drawdown:       {system.max_drawdown:.2%}')
    print()
    print(f'Total Trades:       {len(system.trades)}')
    print(f'Winning Trades:     {len(winning)}')
    print(f'Losing Trades:      {len(losing)}')
    print(f'Win Rate:           {win_rate:.1%}')
    
    if winning:
        avg_win = sum(t.pnl for t in winning) / len(winning)
        print(f'Avg Win:            ${avg_win:+,.2f}')
    if losing:
        avg_loss = sum(t.pnl for t in losing) / len(losing)
        print(f'Avg Loss:           ${avg_loss:+,.2f}')
    
    print()
    print('LEARNING STATS')
    print('-' * 70)
    
    if system.learning_orchestrator:
        stats = system.learning_orchestrator.get_stats()
        print(f'Learning Updates:   {stats["total_updates"]}')
        print(f'Learning Rate:      {stats["learning_rate"]:.4f}')
        print(f'Exploration Rate:   {stats["exploration_rate"]:.4f}')
        print(f'Best Algorithm:     {stats["best_algorithm"]}')
        print(f'Avg Latency:        {stats["avg_latency_ms"]:.3f}ms')
    
    if system.strategy_learning_manager:
        print()
        print('STRATEGY PERFORMANCE')
        print('-' * 70)
        strat_stats = system.strategy_learning_manager.get_all_stats()
        for name, st in strat_stats.items():
            if st['total_trades'] > 0:
                w = system.strategy_learning_manager.strategy_weights.get(name, 1.0)
                print(f'{name:20s} | Trades: {st["total_trades"]:3d} | Win: {st["win_rate"]:5.1%} | PnL: ${st["total_pnl"]:+8,.2f} | Weight: {w:.2f}')
    
    print('=' * 70)

if __name__ == "__main__":
    asyncio.run(test())
