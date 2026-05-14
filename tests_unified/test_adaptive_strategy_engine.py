import unittest

import numpy as np
import pandas as pd

from strategies.unified.strategy_engine import StrategyEngine
from unified_trading_system import UnifiedConfig


class _MDS:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    async def fetch_ohlcv_df(self, symbol: str, timeframe: str = "1m", limit: int = 200):
        return self._df.tail(int(limit)).copy()

    async def fetch_ticker(self, symbol: str):
        return None

    async def fetch_order_book(self, symbol: str, limit: int = 20):
        return None


def _df_trend_up(n: int = 260) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    close = 100 + (t * 0.2) + (np.sin(t / 7.0) * 0.5)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0})
    return df


def _df_range(n: int = 260) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    close = 100 + (np.sin(t / 3.0) * 2.0)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0})
    return df


class TestAdaptiveStrategyEngine(unittest.IsolatedAsyncioTestCase):
    async def test_trend_market_produces_buy_signal(self):
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.min_signal_confidence = 0.55
        eng = StrategyEngine(cfg)
        mds = _MDS(_df_trend_up())

        signals = await eng._signal_for_symbol("BTC/USD", mds)  # noqa: SLF001 (test internal)
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, list)
        if signals:
            self.assertIn(signals[0].action, ("BUY", "SELL"))

    async def test_online_tuner_updates_after_trade(self):
        cfg = UnifiedConfig()
        cfg.run_mode = "paper"
        cfg.min_signal_confidence = 0.55
        cfg.adaptive_min_trades_before_bias = 1  # speed up for test
        eng = StrategyEngine(cfg)
        mds = _MDS(_df_range())

        sig1 = await eng._signal_for_symbol("BTC/USD", mds)  # noqa: SLF001 (test internal)
        self.assertIsNotNone(sig1)
        # Simulate a losing trade to force selectivity/weight adjustment
        eng.on_realized_pnl(symbol="BTC/USD", pnl_pct=-2.0)
        status = eng.get_adaptation_status()
        self.assertIn("tuner", status)
        self.assertIsInstance(status["tuner"], dict)
        self.assertIn("BTC/USD", status["tuner"])


if __name__ == "__main__":
    unittest.main()

