"""
Ultimate Comprehensive Strategy – production signal generator.

Generates signals from trend + mean-reversion hybrid (RSI/EMA).
Wired into StrategyAllocator via strategy pack; param space in strategy_param_space.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UltimateComprehensiveStrategy:
    """Unified-style trend + mean-reversion; analyze(md) for strategy pack."""

    def __init__(self, config: Any = None) -> None:
        self.config = config or {}
        self._cfg = self.config if isinstance(self.config, dict) else getattr(self.config, "__dict__", {})

    def analyze(self, md: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        md: symbol, price, ohlcv_df, tickers.
        Returns dict with action (BUY/SELL), symbol, confidence, price, source or None.
        """
        symbol = str(md.get("symbol") or "")
        price = float(md.get("price") or 0.0)
        df = md.get("ohlcv_df")
        if not symbol or price <= 0:
            return None
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
            return None
        try:
            close = df["close"].astype(float)
            if len(close) < 30:
                return None
            rsi = self._rsi(close, 14)
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=48, adjust=False).mean()
            last_rsi = float(rsi.iloc[-1])
            trend = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
            # Buy: RSI oversold or trend turning up
            if last_rsi < 35 or (last_rsi < 50 and trend > 0):
                return {
                    "action": "BUY",
                    "symbol": symbol,
                    "confidence": 0.55 + (35 - min(last_rsi, 35)) / 100,
                    "price": price,
                    "source": "ultimate",
                }
            if last_rsi > 65 or (last_rsi > 50 and trend < 0):
                return {
                    "action": "SELL",
                    "symbol": symbol,
                    "confidence": 0.55 + (last_rsi - 65) / 100,
                    "price": price,
                    "source": "ultimate",
                }
        except Exception as e:
            logger.debug("Ultimate analyze: %s", e)
        return None

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs)).fillna(100)
