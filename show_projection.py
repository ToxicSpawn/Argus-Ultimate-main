import sys
sys.path.insert(0, '.')
from earnings.aggressive_earnings_maximizer import create_aggressive_trader

trader = create_aggressive_trader(capital=1000, leverage=3)

print('='*70)
print('AGGRESSIVE MONTHLY EARNINGS MAXIMIZATION PLAN')
print('='*70)

projection = trader.generate_monthly_projection(
    win_rate=0.60,
    avg_win=2.5,
    avg_loss=1.5,
    trades_per_day=10
)

print()
print('PROJECTIONS (Based on $1,000 Capital, 3x Leverage):')
print('-'*70)
print(f"  Target Monthly Return:    {projection['target_monthly_return']}")
print(f"  Target Monthly Profit:    {projection['target_monthly_profit']}")
print(f"  Daily Target:             {projection['daily_target']}")
print(f"  Trades Per Day:           {projection['trades_per_day']}")
print(f"  Win Rate Required:        {projection['win_rate_required']}")
print(f"  Avg Win:                  {projection['avg_win_pct']}")
print(f"  Avg Loss:                 {projection['avg_loss_pct']}")
print(f"  Expected Sharpe:          {projection['expected_sharpe']}")
print(f"  Max Daily Loss:           {projection['max_daily_loss']}")
print(f"  Max Drawdown:             {projection['max_drawdown']}")

print()
print('CAPITAL ALLOCATION:')
print('-'*70)
plan = trader.get_aggressive_strategy_plan()
for strategy, details in plan['allocation'].items():
    print(f"  {strategy}:")
    print(f"    Capital: {details['capital_pct']}%")
    print(f"    Expected Monthly: {details['expected_monthly']}")
    print(f"    Risk: {details['risk']}")
    print()

print('KEY RULES FOR MAXIMUM PROFITS:')
print('-'*70)
for i, rule in enumerate(plan['key_rules'], 1):
    print(f'  {i}. {rule}')

print()
print('OPTIMAL TRADING HOURS (UTC):')
print('-'*70)
for hour in plan['optimal_trading_hours']:
    print(f'  - {hour}')

print()
print('BEST PAIRS FOR LEVERAGE:')
print('-'*70)
for pair in plan['best_pairs_for_leverage']:
    print(f'  - {pair}')

print()
print('='*70)
print(f"EXPECTED MONTHLY PROFIT: {plan['expected_monthly_profit']}")
print('='*70)
