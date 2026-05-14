# How Much Can the Bot Earn Using Everything?

There is **no fixed dollar amount**. Earnings depend on capital, market conditions, timeframe, and which features are enabled. Here’s what the system is set up for and what to expect.

---

## 1. Config target (what the bot aims for)

From **unified_config.yaml** → **targets**:

| Target | Value | Meaning |
|--------|--------|--------|
| **Monthly return** | **12%** | Conservative monthly return target. |
| Win rate | 65% | Target win rate. |
| Max drawdown | 15% | Target cap on drawdown. |

So on **$1,000 AUD**, the **target** is about **$120 per month** (12% × $1,000).  
That is a goal, not a guarantee. Real results will be above or below depending on markets and execution.

---

## 2. What “using everything” means

When you run the **unified path** (paper or live) with **everything** enabled you get:

- **Signal sources:** AI brain (strategy engine) + strategy engine directly + **strategy library** (tier + algorithmic) + HFT (if not disabled).
- **Strategies:** All whitelisted strategies (unified_engine, regime_aligned, volume_momentum, mean_reversion, breakout_bb, macd_trend, akashic_tier, etc. if in whitelist).
- **Execution:** Same execution engine; paper can **simulate live** (same config, same disabled list, simulated slippage).
- **Risk:** Max drawdown, daily loss, circuit breaker, position limits from config.

So “using everything” = **maximum signal diversity and all enabled strategies**, within your risk and sizing limits. That can mean **more trades** and **more diversification**, but also more exposure – results can be better or worse than a minimal setup in any given period.

---

## 3. Example 30-day backtest (unified engine only)

The **30-day backtest** (`scripts/run_30d_backtest.py`) uses **only the unified strategy engine** (no scanner, no strategy library). It gives one possible outcome on historical data:

- **Latest run (BTC/USD, $1k, 30 days):** about **-$13 AUD** (~**-1.33%**), 2 trades, 1 closed (loss).
- **Earlier run (same script):** about **+$42 AUD** (~**+4.24%**).

So the **same bot and config** can show a loss or a gain over 30 days depending on the data and the exact run. The backtest does **not** include the strategy library or the full scanner; it’s a lower bound of “everything” in terms of signal sources.

---

## 4. Rough scale by capital (if target is hit)

If the bot were to hit the **12% monthly** target:

| Capital | ~Monthly (12%) | ~Year (if compounded) |
|---------|----------------|------------------------|
| $1,000  | ~$120          | ~$1,120 (year 1)       |
| $5,000  | ~$600          | ~$5,600                |
| $10,000 | ~$1,200        | ~$11,200               |

Again: these are **targets**, not promises. Drawdowns, losing streaks, and fees/slippage will reduce or outweigh these in bad periods.

---

## 5. How to get a number for your setup

1. **Backtest (unified engine only)**  
   ```bash
   py -3 scripts/run_30d_backtest.py --symbol BTC/USD --capital 1000 --days 30
   ```  
   Use the printed “Estimated 30-day earnings” as one data point (one symbol, one history slice).

2. **Paper trading with everything on**  
   Run paper for 30 days (or as long as you can) with:
   - `strategy_library.enabled: true`
   - `strategy_library.modes` including your mode (e.g. paper/live)
   - All desired strategies in `strategy_whitelist`  
   Then check **data/paper_results.json** (or your paper logs) for actual PnL over that period.

3. **Live (when ready)**  
   Only after you’re comfortable with backtest and paper: run live with the same “everything” config and track real PnL.

---

## 6. Short answer

- **Target with everything:** **~12% per month** (e.g. **~$120/month on $1k**).
- **Actual earnings:** Unknown in advance; the last 30-day backtest was about **-$13** on $1k; an earlier one was about **+$42**.
- **To get a number for your case:** Run the 30-day backtest and/or a long paper run with “everything” enabled and read the reported PnL.
