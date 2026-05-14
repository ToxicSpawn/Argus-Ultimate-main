# Monolith Split Plan

`unified_trading_system.py` is 639KB / ~11,500 lines. This document tracks the extraction plan.

## Status

| Module | Status | Extracted To |
|--------|--------|-------------|
| TradingEngine base | ✅ Done | `core/trading_engine.py` |
| SignalPipeline | ✅ Done | `core/signal_pipeline.py` |
| OrderRouter | ✅ Done | `core/order_router.py` |
| PositionTracker | ✅ Done | `core/position_tracker.py` |
| RegimeDetector | ✅ Done | `core/regime_detector.py` |
| EnsembleController | ✅ Done | `core/ensemble_controller.py` |
| RiskManager | ✅ Previously done | `risk/unified_risk_manager.py` |
| HFT Infrastructure | ✅ Previously done | `hft_engine/` |
| Backtesting | ✅ Previously done | `backtesting/` |
| Exchange connectors | ⏳ Pending | `exchanges/` (already exists) |
| ML models | ⏳ Pending | `ml/` (already exists) |
| Dashboard/metrics | ⏳ Pending | `monitoring/` (already exists) |
| Strategy engine | ⏳ Pending | `strategies/` (already exists) |

## Migration Strategy

The monolith is NOT deleted until all functionality is verified in the extracted modules.

```
Phase 1 (Done):  Extract pure logic classes with zero external deps
Phase 2 (Next):  Wire extracted classes into unified_trading_system.py via imports
Phase 3 (Later): Gradually gut unified_trading_system.py, replacing with imports
Phase 4 (Final): Delete unified_trading_system.py when all refs point to core/
```

## How to Use the Extracted Core

```python
from core import TradingEngine, SignalPipeline, OrderRouter, PositionTracker
from core import RegimeDetector, EnsembleController, MarketRegime

# In your engine subclass:
class ArgusLiveEngine(TradingEngine):
    async def on_startup(self):
        self.tracker = PositionTracker(starting_cash=1000.0)
        self.router = OrderRouter(self.config)
        self.regime = RegimeDetector()
        self.ensemble = EnsembleController(model_ids=["momentum", "mean_rev", "ml_dqn"])

    async def on_tick(self):
        # signals come from SignalPipeline
        # regime from RegimeDetector.detect(closes, highs, lows)
        # routing from OrderRouter.route()
        # positions from PositionTracker.snapshot()
        pass

    async def on_shutdown(self):
        pass
```
