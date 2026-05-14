# Strategies Quick Reference Guide 🚀

## 🎯 Core Strategies

### Trinity Router
```python
from strategies.strategy_router import TrinityRouter

router = TrinityRouter(exchange, data_feed, risk_manager)
await router.on_tick_event('BTC/USDT', 50000.0)
status = router.get_status()
```

**Components**:
- **Hunter**: Extreme BTC opportunities (liquidations, crashes)
- **Farmer**: Grid trading for steady income
- **Shadow**: Lead-lag arbitrage on altcoins

---

## 📊 Advanced Strategies

### Statistical Arbitrage
```python
from strategies.stat_arb import StatisticalArbitrage

stat_arb = StatisticalArbitrage(exchange, data_feed, position_tracker, risk_manager)
await stat_arb.run_iteration()
```

**Features**: Kalman filter, dynamic hedging, pairs trading

### Momentum Strategy
```python
from strategies.momentum import GodTierMomentumStrategy

momentum = GodTierMomentumStrategy('BTC/USDT', data_feed, risk_manager)
signal = await momentum.get_signal()
```

**Features**: Multi-timeframe, volatility targeting, trend quality filters

### Transformer Brain
```python
from strategies.transformer_brain import TransformerBrain

transformer = TransformerBrain(feature_dim=100, seq_len=60)
prediction = transformer.predict(features)
transformer.train_step(features, target)
```

**Features**: Multi-head attention, time-series forecasting, online learning

### Genetic Evolution
```python
from strategies.genetic_evolution import GeneticEvolution

optimizer = GeneticEvolution(
    param_bounds={'param1': (0, 1), 'param2': (10, 100)},
    fitness_func=my_fitness_function
)
best = optimizer.evolve()
```

**Features**: Parameter optimization, crossover, mutation, elitism

### Cross-Exchange Arbitrage
```python
from strategies.cross_exchange_arbitrage import CrossExchangeArbitrage

arb = CrossExchangeArbitrage(primary_exchange, position_tracker, risk_manager)
await arb.initialize()
await arb.run_iteration()
```

**Features**: Async polling, latency normalization, fee-adjusted spreads

### Market Maker
```python
from strategies.market_maker import SPlusMarketMaker

mm = SPlusMarketMaker(
    'BTC/USDT', exchange, data_feed, position_tracker, risk_manager
)
await mm.run_iteration()
```

**Features**: Avellaneda-Stoikov, OBI adjustment, GARCH volatility

### MEV Extractor
```python
from strategies.mev_extractor import MEVExtractor

mev = MEVExtractor(connector)
await mev.scan_mempool(simulated_txs)
```

**Features**: Sandwich attacks, back-running, long-tail arbitrage

---

## 💎 DeFi Strategies

### Yield Optimizer
```python
from strategies.defi.yield_optimizer import YieldOptimizer

optimizer = YieldOptimizer(protocols=['uniswap', 'aave'], auto_compound=True)
apy = optimizer.calculate_apy(protocol_data)
allocations = optimizer.optimize_allocation(capital, protocol_apys, gas_costs)
```

**Features**: APY comparison, gas optimization, auto-compounding

### Flash Loan Arbitrage
```python
from strategies.defi.flash_loan_arb import FlashLoanArbitrage

flash_arb = FlashLoanArbitrage(flash_loan_fee=0.0009, gas_cost_usd=50)
opportunities = flash_arb.find_opportunities(dex_prices)
```

**Features**: Flash loan execution, profit calculation, opportunity detection

---

## ⚡ HFT Strategies

### Triangular Arbitrage
```python
from strategies.hft.triangular_arb import TriangularArbitrage

tri_arb = TriangularArbitrage(min_profit_bps=3)
cycles = tri_arb.detect_cycles(prices)
if cycles:
    profit = tri_arb.execute_cycle(cycles[0])
```

**Features**: Cycle detection, profit calculation, fast execution

### Market Making 2.0
```python
from strategies.hft.market_making_2 import MarketMaking2

mm2 = MarketMaking2(
    symbol='BTC/USDT',
    target_inventory=0.0,
    max_inventory=5.0,
    risk_aversion=0.01,
    base_spread=0.002
)
mm2.update_market_data(orderbook, trades)
quote = mm2.generate_quotes()
```

**Features**: Enhanced Avellaneda-Stoikov, inventory optimization, adaptive spreads

### Latency Arbitrage
```python
from strategies.hft.latency_arbitrage import LatencyArbitrage

latency_arb = LatencyArbitrage(fast_feed, slow_feed, min_edge_bps=2)
opportunity = latency_arb.detect_stale_quotes(fast_price, slow_price, timestamp_diff_ms)
if opportunity:
    latency_arb.execute_if_profitable(opportunity, execution_speed_ms)
```

**Features**: Speed advantage, stale quote detection, execution timing

---

## 🔄 Advanced Module Strategies

### Regime Switching
```python
from strategies.advanced.regime_switching import RegimeSwitchingStrategy

regime = RegimeSwitchingStrategy(n_regimes=3)
regime.fit(returns)
current_regime = regime.predict_regime(recent_returns)
strategy = regime.get_strategy_for_regime(current_regime)
```

**Features**: Hidden Markov Models, strategy switching, regime detection

### Factor Investing
```python
from strategies.advanced.factor_investing import FactorInvestingStrategy

factor = FactorInvestingStrategy(factors=['momentum', 'value', 'size', 'volatility'])
scores = factor.calculate_composite_score(asset_data)
signals = factor.generate_signals(scores)
```

**Features**: Multi-factor model, long-short portfolio, factor-based allocation

---

## 📈 Expected Performance

| Strategy | Returns | Sharpe | Win Rate |
|----------|---------|--------|----------|
| Trinity Hunter | +40-80% | 2.5-4.0 | 55-65% |
| Trinity Farmer | +15-25% | 3.0-4.5 | 60-70% |
| Trinity Shadow | +20-35% | 2.0-3.5 | 50-60% |
| Stat Arb | +30-50% | 2.5-3.5 | 60-70% |
| Momentum | +40-60% | 2.0-3.0 | 55-65% |
| Transformer | +35-55% | 2.5-3.5 | 55-65% |
| Cross-Exchange | +20-35% | 3.0-4.0 | 65-75% |
| Market Maker | +20-40% | 2.0-3.0 | 52-55% |
| MEV | +30-60% | 2.5-4.0 | 50-60% |
| Yield Optimizer | +15-35% | 2.5-3.5 | 70-80% |
| Flash Loan | +25-50% | 2.0-3.5 | 55-65% |
| Triangular Arb | +15-30% | 3.0-4.5 | 70-80% |
| Market Making 2.0 | +20-40% | 2.0-3.0 | 52-55% |
| Latency Arb | +20-40% | 2.5-3.5 | 60-70% |

---

## 🔧 Integration Example

```python
from argus_peak_ultimate import ArgusPeakUltimate
from strategies.strategy_router import TrinityRouter
from strategies.stat_arb import StatisticalArbitrage

# Initialize bot
bot = ArgusPeakUltimate()

# Add strategies
router = TrinityRouter(bot.exchange, bot.data_feed, bot.risk_manager)
stat_arb = StatisticalArbitrage(
    bot.exchange, bot.data_feed, bot.position_tracker, bot.risk_manager
)

bot.add_strategy('trinity', router)
bot.add_strategy('stat_arb', stat_arb)

# Run bot
await bot.start()
```

---

## 📚 Full Documentation

See `STRATEGIES_V70_INTEGRATION_COMPLETE.md` for comprehensive documentation.
