#!/usr/bin/env python3
"""
Technical Indicators - S+ Tier
Comprehensive technical analysis indicators for trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical Indicators - S+ Tier
    Calculates various technical indicators for trading analysis.
    """

    def __init__(self):
        self.indicators = {
            "ema": self.calculate_ema,
            "sma": self.calculate_sma,
            "rsi": self.calculate_rsi,
            "macd": self.calculate_macd,
            "atr": self.calculate_atr,
            "bollinger": self.calculate_bollinger_bands,
            "stochastic": self.calculate_stochastic,
            "williams_r": self.calculate_williams_r,
        }

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to DataFrame.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with indicators added
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to add_indicators().")
            return df

        logger.info("Adding indicators (EMA, SMA, RSI, MACD, ATR, Bollinger Bands)")

        # Ensure we have the required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.error("DataFrame missing required OHLCV columns")
            return df

        # Make a copy to avoid modifying original
        df = df.copy()

        try:
            # Add basic indicators
            df = self.calculate_ema(df, periods=[9, 21, 50])
            df = self.calculate_sma(df, periods=[20, 50, 200])
            df = self.calculate_rsi(df, period=14)
            df = self.calculate_macd(df)
            df = self.calculate_atr(df, period=14)
            df = self.calculate_bollinger_bands(df, period=20, std_dev=2)
            df = self.calculate_stochastic(df, k_period=14, d_period=3)
            df = self.calculate_williams_r(df, period=14)

            logger.info("Successfully added all technical indicators")
            return df

        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df

    def calculate_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50]) -> pd.DataFrame:
        """Calculate Simple Moving Average"""
        df = df.copy()
        for period in periods:
            df[f"SMA_{period}"] = df["close"].rolling(window=period).mean()
        return df

    def calculate_ema(self, df: pd.DataFrame, periods: List[int] = [9, 21, 50]) -> pd.DataFrame:
        """Calculate Exponential Moving Average"""
        df = df.copy()
        for period in periods:
            df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        return df

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        df = df.copy()

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

        return df

    def calculate_macd(
        self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        df = df.copy()

        fast_ema = df["close"].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df["close"].ewm(span=slow_period, adjust=False).mean()

        df["MACD"] = fast_ema - slow_ema
        df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        df = df.copy()

        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f"ATR_{period}"] = true_range.rolling(window=period).mean()

        return df

    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = df.copy()

        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()

        df[f"BB_Upper_{period}"] = sma + (std * std_dev)
        df[f"BB_Lower_{period}"] = sma - (std * std_dev)
        df[f"BB_Middle_{period}"] = sma

        # Bollinger Band Position
        df[f"BB_Position_{period}"] = (df["close"] - df[f"BB_Lower_{period}"]) / (
            df[f"BB_Upper_{period}"] - df[f"BB_Lower_{period}"]
        )

        return df

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df = df.copy()

        lowest_low = df["low"].rolling(window=k_period).min()
        highest_high = df["high"].rolling(window=k_period).max()

        df["%K"] = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low))
        df["%D"] = df["%K"].rolling(window=d_period).mean()

        return df

    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R"""
        df = df.copy()

        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()

        df[f"Williams_R_{period}"] = -100 * ((highest_high - df["close"]) / (highest_high - lowest_low))

        return df

    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price"""
        df = df.copy()

        df["VWAP"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on technical indicators.

        Args:
            df: DataFrame with indicators

        Returns:
            Series of signals (-1, 0, 1)
        """
        signals = pd.Series(0, index=df.index)

        try:
            # RSI signals
            if "RSI_14" in df.columns:
                signals[df["RSI_14"] < 30] = 1  # Oversold - Buy
                signals[df["RSI_14"] > 70] = -1  # Overbought - Sell

            # MACD signals
            if "MACD" in df.columns and "MACD_Signal" in df.columns:
                macd_cross_up = (df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
                macd_cross_down = (df["MACD"] < df["MACD_Signal"]) & (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))

                signals[macd_cross_up] = 1
                signals[macd_cross_down] = -1

            # Bollinger Band signals
            if "BB_Position_20" in df.columns:
                signals[df["BB_Position_20"] < 0.1] = 1  # Near lower band - Buy
                signals[df["BB_Position_20"] > 0.9] = -1  # Near upper band - Sell

            # Stochastic signals
            if "%K" in df.columns and "%D" in df.columns:
                stochastic_buy = (df["%K"] < 20) & (df["%D"] < 20)
                stochastic_sell = (df["%K"] > 80) & (df["%D"] > 80)

                signals[stochastic_buy] = 1
                signals[stochastic_sell] = -1

        except Exception as e:
            logger.error(f"Error generating signals: {e}")

        return signals

    def get_indicator_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available indicators"""
        return {
            "SMA": {"description": "Simple Moving Average", "parameters": ["period"]},
            "EMA": {"description": "Exponential Moving Average", "parameters": ["period"]},
            "RSI": {"description": "Relative Strength Index", "parameters": ["period"]},
            "MACD": {
                "description": "Moving Average Convergence Divergence",
                "parameters": ["fast_period", "slow_period", "signal_period"],
            },
            "ATR": {"description": "Average True Range", "parameters": ["period"]},
            "Bollinger Bands": {"description": "Bollinger Bands", "parameters": ["period", "std_dev"]},
            "Stochastic": {"description": "Stochastic Oscillator", "parameters": ["k_period", "d_period"]},
            "Williams %R": {"description": "Williams Percent Range", "parameters": ["period"]},
        }


class SignalGenerator:
    """
    Legacy compatibility signal generator used by `core.argus`.

    The canonical Unified System uses `strategies/` + the router/initializer.
    This class exists so older modules can import without optional dependencies.
    """

    def __init__(self) -> None:
        self._ti = TechnicalIndicators()

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._ti.add_indicators(df)

    def compute_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "RSI_14" not in out.columns:
            return out
        out["signal"] = "HOLD"
        try:
            out.loc[out["RSI_14"] < 30, "signal"] = "BUY"
            out.loc[out["RSI_14"] > 70, "signal"] = "SELL"
        except Exception as _e:
            logger.debug("indicators error: %s", _e)
        return out
