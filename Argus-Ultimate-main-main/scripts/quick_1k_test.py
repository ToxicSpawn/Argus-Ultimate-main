#!/usr/bin/env py
"""Quick $1K capital backtest for Argus"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math

random.seed(42)

initial_capital = 1000.0
cash = initial_capital
position = 0.0
entry_price = 0.0
total_trades = 0
winning_trades = 0
equity_curve = [initial_capital]
peak_equity = initial_capital
max_drawdown = 0.0

# Simulate 30 days of minute-level trading
n_minutes = 30 * 24 * 60
price = 65000.0
annual_drift = 0.5
annual_vol = 0.65
dt = 1.0 / (365.25 * 24 * 60)

# Position limits for $1K account
max_position_usd = 500.0
position_size_usd = 100.0
taker_fee = 7.0 / 10000.0

price_history = []

for i in range(n_minutes):
    shock = random.gauss(0, 1)
    ret = annual_drift * dt + annual_vol * math.sqrt(dt) * shock
    price = price * math.exp(ret)
    price_history.append(price)
    
    if len(price_history) < 20:
        continue
    if len(price_history) > 20:
        price_history.pop(0)
    
    returns = [math.log(price_history[j] / price_history[j-1]) for j in range(1, len(price_history))]
    momentum = (sum(returns) / len(returns)) * 10000
    
    if i % 120 != 0:
        continue
    
    current_value = abs(position * price)
    if current_value >= max_position_usd:
        continue
    
    fill_price = price
    
    if momentum > 3.0 and position <= 0:
        size = position_size_usd / fill_price
        fee = fill_price * size * taker_fee
        
        if position < 0:
            pnl = (entry_price - fill_price) * min(size, abs(position)) - fee
            if pnl > 0:
                winning_trades += 1
            remaining = abs(position) - size
            if remaining > 0:
                position = -remaining
            else:
                position = size - abs(position) if size > abs(position) else 0
                entry_price = fill_price if position > 0 else 0
        else:
            if position == 0:
                entry_price = fill_price
            else:
                total_cost = entry_price * position + fill_price * size
                position += size
                entry_price = total_cost / position
            position += size
            cash -= (fill_price * size + fee)
        total_trades += 1
        
    elif momentum < -3.0 and position >= 0:
        size = position_size_usd / fill_price
        fee = fill_price * size * taker_fee
        
        if position > 0:
            pnl = (fill_price - entry_price) * min(size, position) - fee
            if pnl > 0:
                winning_trades += 1
            remaining = position - size
            if remaining > 0:
                position = remaining
            else:
                position = size - position if size > position else 0
                entry_price = fill_price if position < 0 else 0
        else:
            if position == 0:
                entry_price = fill_price
            else:
                total_cost = entry_price * abs(position) + fill_price * size
                position -= size
                entry_price = total_cost / abs(position) if position != 0 else 0
            position -= size
            cash += (fill_price * size - fee)
        total_trades += 1
    
    equity = cash + position * price
    equity_curve.append(equity)
    if equity > peak_equity:
        peak_equity = equity
    dd = (peak_equity - equity) / peak_equity * 100
    if dd > max_drawdown:
        max_drawdown = dd

# Close position
if abs(position) > 1e-9:
    fill_price = price * 0.99965
    fee = fill_price * abs(position) * taker_fee
    if position > 0:
        pnl = (fill_price - entry_price) * position - fee
    else:
        pnl = (entry_price - fill_price) * abs(position) - fee
    cash += pnl

final_equity = cash
total_pnl = final_equity - initial_capital
return_30d = (final_equity / initial_capital - 1) * 100
fees_paid = total_trades * position_size_usd * taker_fee
weekly_return = return_30d / 4.3

print("=" * 60)
print("  ARGUS $1,000 CAPITAL BACKTEST (30 days)")
print("=" * 60)
print(f"  Starting Capital:    ${initial_capital:,.2f}")
print(f"  Final Equity:        ${final_equity:,.2f}")
print(f"  Total PnL (30 days): ${total_pnl:,.2f}")
print(f"  Return (30 days):    {return_30d:.1f}%")
print()
print(f"  Total Trades:        {total_trades}")
print(f"  Win Rate:            {winning_trades/total_trades*100:.1f}%")
print(f"  Max Drawdown:        {max_drawdown:.1f}%")
print()
print(f"  FEE IMPACT:")
print(f"  Fees paid:           ${fees_paid:.2f}")
print(f"  Fee as % of capital: {fees_paid / initial_capital * 100:.1f}%")
print()
print("  WEEKLY GROWTH PROJECTION:")
for week in range(1, 9):
    projected = initial_capital * ((1 + weekly_return/100) ** week)
    print(f"  Week {week}: ${projected:,.2f}")
print()
print("  VERDICT:")
if return_30d > 10:
    print(f"  YES - Can grow ${initial_capital:.0f} to ${final_equity:,.0f} in 30 days ({return_30d:.0f}% return)")
    print(f"  At this rate, $1K becomes ${initial_capital * (1 + return_30d/100)**12:,.0f} in 1 year")
else:
    print(f"  SLOW - Only {return_30d:.1f}% return in 30 days")
print("=" * 60)
