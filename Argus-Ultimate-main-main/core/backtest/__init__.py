"""Argus backtest engine — Push 76."""
from core.backtest.metrics import BacktestMetrics, compute_metrics
from core.backtest.walk_forward import WalkForwardEngine, WalkForwardResult
from core.backtest.monte_carlo import MonteCarloSimulator, MonteCarloResult
from core.backtest.backtest_runner import BacktestRunner

try:
    from core.backtest.hft_backtest import (
        HFTBacktestEngine,
        FillProbabilityModel,
        MarketImpactModel,
        BacktestStats,
        TradeRecord,
        HFTOrder,
        OrderSide,
        FillStatus,
        OrderType,
    )
    _HFT_AVAILABLE = True
except Exception:
    _HFT_AVAILABLE = False

__all__ = [
    "BacktestMetrics", "compute_metrics",
    "WalkForwardEngine", "WalkForwardResult",
    "MonteCarloSimulator", "MonteCarloResult",
    "BacktestRunner",
]
if _HFT_AVAILABLE:
    __all__ += [
        "HFTBacktestEngine",
        "FillProbabilityModel",
        "MarketImpactModel",
        "BacktestStats",
        "TradeRecord",
        "HFTOrder",
        "OrderSide",
        "FillStatus",
        "OrderType",
    ]
