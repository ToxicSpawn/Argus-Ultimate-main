"""Historical Data Fetcher and Backtesting Engine.

Fetches historical crypto data and runs backtests.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class OHLCV:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestResult:
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)


class HistoricalDataFetcher:
    """Fetch historical crypto data."""

    def __init__(self):
        self._cache: Dict[str, List[OHLCV]] = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 1000,
    ) -> List[OHLCV]:
        cache_key = f"{symbol}_{interval}_{limit}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, using synthetic data")
            return self._generate_synthetic_data(symbol, limit)

        base_symbol = symbol.replace("USDT", "").upper() + "USDT"
        
        all_candles = []
        max_per_request = 1000
        chunks = (limit + max_per_request - 1) // max_per_request
        
        try:
            for chunk in range(chunks):
                url = f"https://api.binance.com/api/v3/klines"
                params = {
                    "symbol": base_symbol,
                    "interval": interval,
                    "limit": max_per_request,
                }
                
                if chunk > 0 and len(all_candles) > 0:
                    last_time = int(all_candles[-1].timestamp * 1000) + 1
                    params["startTime"] = last_time
                
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if not data:
                    break
                
                for item in data:
                    all_candles.append(OHLCV(
                        timestamp=item[0] / 1000,
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                    ))
                
                if len(data) < max_per_request:
                    break
            
            if len(all_candles) == 0:
                logger.warning("No data from Binance, using synthetic data")
                return self._generate_synthetic_data(symbol, limit)
            
            result = all_candles[:limit]
            self._cache[cache_key] = result
            logger.info(f"Fetched {len(result)} candles for {symbol} via {chunks} API calls")
            return result

        except Exception as e:
            logger.warning(f"Failed to fetch data from Binance: {e}, using synthetic data")
            return self._generate_synthetic_data(symbol, limit)

    def _generate_synthetic_data(
        self,
        symbol: str,
        limit: int,
    ) -> List[OHLCV]:
        np.random.seed(hash(symbol) % 2**32)
        
        base_price = 50000 if "BTC" in symbol else 3000
        data = []
        
        current_price = base_price
        current_time = time.time() - (limit * 3600)
        
        for i in range(limit):
            trend = np.random.randn() * 0.02
            volatility = np.random.uniform(0.01, 0.03)
            
            open_price = current_price
            close_price = open_price * (1 + trend + np.random.randn() * volatility)
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * volatility)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * volatility)
            
            volume = np.random.uniform(1000, 10000)
            
            data.append(OHLCV(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            ))
            
            current_price = close_price
            current_time += 3600
        
        return data

    def calculate_features(
        self,
        data: List[OHLCV],
    ) -> pd.DataFrame:
        df = pd.DataFrame([
            {
                "timestamp": o.timestamp,
                "open": o.open,
                "high": o.high,
                "low": o.low,
                "close": o.close,
                "volume": o.volume,
            }
            for o in data
        ])

        df["returns"] = df["close"].pct_change()
        
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        
        df["adf"] = (df["close"] - df["close"].shift(1)).fillna(0)
        df["adf"] = df["adf"].where(df["close"] > df["close"].shift(1), 0)
        df["adf"] = df["adf"].rolling(14).sum()
        
        df["cci"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
        
        df["stoch_k"] = 100 * (df["close"] - df["low"].rolling(14).min()) / (df["high"].rolling(14).max() - df["low"].rolling(14).min())
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        
        df["momentum"] = df["close"] / df["close"].shift(10) - 1
        
        df["roc"] = df["close"].pct_change(10)
        
        df["willr"] = -100 * (df["high"].rolling(14).max() - df["close"]) / (df["high"].rolling(14).max() - df["low"].rolling(14).min())
        
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        df["obv"] = df["obv"].rolling(20).mean()
        
        for lag in [1, 2, 3, 5, 10]:
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
        
        df["price_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        df = df.dropna()
        
        return df


class BacktestEngine:
    """Run backtests on historical data."""

    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.001,
        stop_loss: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.stop_loss = stop_loss

    def run(
        self,
        data: List[OHLCV],
        signals: List[int],
        position_size: float = 0.1,
        stop_loss: float = 0.02,
    ) -> BacktestResult:
        
        if len(data) != len(signals):
            min_len = min(len(data), len(signals))
            data = data[:min_len]
            signals = signals[:min_len]

        capital = self.initial_capital
        position = 0
        entry_price = 0
        
        equity_curve = [capital]
        trades = []
        
        stop_loss_pct = stop_loss if stop_loss else self.stop_loss
            
        for i in range(1, len(data)):
            current_price = data[i].close
            
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct < -stop_loss_pct:
                    proceeds = position * current_price * (1 - self.commission)
                    pnl = proceeds - (position * entry_price)
                    
                    trades.append({
                        "type": "sell",
                        "price": current_price,
                        "shares": position,
                        "pnl": pnl,
                        "reason": "stop_loss",
                    })
                    
                    capital += proceeds
                    position = 0
                    entry_price = 0
            
            if position == 0 and signals[i] == 1:
                buy_amount = capital * position_size
                shares = buy_amount / current_price
                cost = shares * current_price * (1 + self.commission)
                
                if cost <= capital:
                    position = shares
                    entry_price = current_price
                    capital -= cost
                    
                    trades.append({
                        "type": "buy",
                        "price": current_price,
                        "shares": shares,
                        "timestamp": data[i].timestamp,
                    })
            
            elif position > 0 and signals[i] == -1:
                proceeds = position * current_price * (1 - self.commission)
                pnl = proceeds - (position * entry_price)
                
                trades.append({
                    "type": "sell",
                    "price": current_price,
                    "shares": position,
                    "pnl": pnl,
                    "reason": "signal",
                })
                
                capital += proceeds
                position = 0
                entry_price = 0
            
            total_value = capital + (position * current_price)
            equity_curve.append(total_value)
        
        if position > 0:
            final_price = data[-1].close
            proceeds = position * final_price * (1 - self.commission)
            trades.append({
                "type": "sell",
                "price": final_price,
                "shares": position,
                "pnl": proceeds - (position * entry_price),
                "timestamp": data[-1].timestamp,
            })
            capital = proceeds
        
        return self._calculate_metrics(capital, equity_curve, trades)

    def _calculate_metrics(
        self,
        final_capital: float,
        equity_curve: List[float],
        trades: List[Dict],
    ) -> BacktestResult:
        
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)
        
        if returns:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        avg_win = np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd * 100,
            win_rate=win_rate,
            total_trades=len(trades),
            profitable_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_curve,
            trades=trades,
        )


class MLTrainingPipeline:
    """Train ML models on historical data."""

    def __init__(self):
        self.data_fetcher = HistoricalDataFetcher()
        self.backtest_engine = BacktestEngine()
        self._df = None

    def _momentum_signals(self, df: pd.DataFrame) -> np.ndarray:
        signals = np.zeros(len(df), dtype=int)
        
        df = df.copy()
        
        rsi = df["rsi"].fillna(50)
        close = df["close"]
        sma20 = df["sma_20"].fillna(close)
        sma50 = df["sma_50"].fillna(close)
        
        trend_up = close > sma20
        strong_trend = close > sma50
        
        volatility = df["atr"].fillna(df["atr"].mean()) / close
        low_vol = volatility < volatility.quantile(0.5)
        
        volume_ok = df["volume_ratio"].fillna(1) > 0.8
        
        buy_condition = trend_up & (strong_trend | low_vol) & volume_ok
        sell_condition = ~trend_up & (~strong_trend | ~low_vol)
        
        signals = np.where(buy_condition, 1, 0)
        signals = np.where(sell_condition, -1, signals)
        
        return signals
    
    def _filter_by_regime(self, df: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        df = df.copy()
        
        close = df["close"]
        sma20 = df["sma_20"].fillna(close)
        sma50 = df["sma_50"].fillna(close)
        
        trend_up = close > sma20
        strong_trend = close > sma50
        
        volatility = df["atr"].fillna(df["atr"].mean()) / close
        vol_percentile = volatility.rank(pct=True)
        
        regime = np.zeros(len(df), dtype=int)
        regime = np.where(trend_up & strong_trend & (vol_percentile < 0.6), 1, regime)
        regime = np.where(~trend_up & ~strong_trend & (vol_percentile > 0.4), -1, regime)
        
        filtered_signals = np.where(regime != 0, signals, 0)
        
        logger.info(f"Regime filter: trending={np.sum(regime==1)}, ranging={np.sum(regime==0)}, down={np.sum(regime==-1)}")
        
        return filtered_signals

    def prepare_data(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 2000,
    ) -> Tuple[pd.DataFrame, List[OHLCV]]:
        ohlcv_data = self.data_fetcher.fetch_ohlcv(symbol, interval, limit)
        df = self.data_fetcher.calculate_features(ohlcv_data)
        
        return df, ohlcv_data

    def create_labels(
        self,
        df: pd.DataFrame,
        lookahead: int = 5,
        threshold: float = 0.005,
    ) -> np.ndarray:
        labels = []
        
        for i in range(len(df) - lookahead):
            future_return = (df["close"].iloc[i + lookahead] - df["close"].iloc[i]) / df["close"].iloc[i]
            
            if future_return > threshold:
                labels.append(1)
            elif future_return < -threshold:
                labels.append(-1)
            else:
                labels.append(0)
        
        return np.array(labels)

    def prepare_features(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        feature_cols = [
            "returns", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", 
            "volume_ratio", "atr", "stoch_k", "momentum",
        ]
        
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if not available_cols:
            available_cols = ["returns", "rsi", "macd"]
        
        while len(available_cols) < 10:
            available_cols.append("returns")
        
        X = df[available_cols].values
        
        X = np.nan_to_num(X, nan=0.0)
        
        return X

    def train_and_backtest(
        self,
        symbol: str = "BTCUSDT",
        model_type: str = "lstm",
        epochs: int = 20,
        limit: int = 5000,
        interval: str = "1h",
    ) -> Dict[str, Any]:
        
        logger.info(f"Preparing data for {symbol} {interval}...")
        df, ohlcv_data = self.prepare_data(symbol, interval=interval, limit=limit)
        
        logger.info("Creating labels...")
        labels = self.create_labels(df)
        
        logger.info("Preparing features...")
        X = self.prepare_features(df)
        
        min_len = min(len(X), len(labels))
        X = X[:min_len]
        labels = labels[:min_len]
        
        self._df = df
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = labels[:train_size], labels[train_size:]
        
        logger.info(f"Training size: {len(X_train)}, Test size: {len(X_test)}, Features: {X_train.shape[1] if len(X_train.shape) > 1 else 1}")
        logger.info(f"Train labels: {np.unique(y_train, return_counts=True)}")
        logger.info(f"Test labels: {np.unique(y_test, return_counts=True)}")
        
        X_train_reshaped = X_train.reshape(-1, 1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, 1, X_test.shape[-1])
        
        all_predictions = []
        
        if len(X_train) < 10 or len(np.unique(y_train)) < 2:
            logger.warning("Insufficient training data, using simple signals")
            signals = np.zeros(len(X_test), dtype=int)
        else:
            try:
                from ml.real_ml_engine import create_ml_engine
                
                model_types = ["pytorch_lstm"]
                all_predictions = []
                
                for mt in model_types:
                    try:
                        logger.info(f"Training {mt}...")
                        from ml.real_ml_engine import create_ml_engine
                        
                        engine = create_ml_engine({"input_size": X_train.shape[-1]})
                        
                        if len(np.unique(y_train)) > 1:
                            engine.train_price_prediction(
                                X_train_reshaped.astype(np.float32),
                                y_train.astype(np.float32),
                                X_test_reshaped.astype(np.float32),
                                y_test.astype(np.float32),
                                model_type=mt,
                                epochs=epochs,
                            )
                        
                        predictions = engine.predict_price(X_test_reshaped.astype(np.float32))
                        all_predictions.append(predictions.flatten())
                    except Exception as e:
                        logger.warning(f"Model {mt} failed: {e}")
                
                if all_predictions:
                    ensemble_pred = np.mean(all_predictions, axis=0)
                else:
                    ensemble_pred = np.zeros(len(X_test))
                
                predictions_flat = ensemble_pred
                std_pred = np.std(predictions_flat)
                logger.info(f"Ensemble predictions: min={predictions_flat.min():.4f}, max={predictions_flat.max():.4f}, mean={predictions_flat.mean():.4f}, std={std_pred:.6f}")
                
                df_test = self._df.iloc[-len(X_test):].copy()
                
                mom_signals = self._momentum_signals(df_test)
                
                if len(np.unique(predictions_flat)) > 1 and np.std(predictions_flat) >= 0.005:
                    ml_confirm = np.where(predictions_flat > 0.0, 1, -1)
                    raw_signals = np.where(mom_signals == ml_confirm, mom_signals, 0)
                    logger.info(f"ML confirmed: buy={np.sum(raw_signals==1)}, sell={np.sum(raw_signals==-1)}, hold={np.sum(raw_signals==0)}")
                else:
                    raw_signals = mom_signals
                    logger.info(f"Using momentum only: buy={np.sum(raw_signals==1)}, sell={np.sum(raw_signals==-1)}")
                
                signals = self._filter_by_regime(df_test, raw_signals)
                
                logger.info(f"Signals: buy={np.sum(signals==1)}, sell={np.sum(signals==-1)}, hold={np.sum(signals==0)}")
                
            except Exception as e:
                logger.warning(f"ML training failed: {e}")
                signals = np.zeros(len(X_test), dtype=int)
        
        if len(ohlcv_data) > len(signals):
            test_ohlcv = ohlcv_data[-(len(signals)):]
        else:
            test_ohlcv = ohlcv_data
        
        logger.info("Running backtest...")
        result = self.backtest_engine.run(test_ohlcv, signals.tolist())
        
        return {
            "symbol": symbol,
            "model_type": model_type,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "profitable_trades": result.profitable_trades,
            "losing_trades": result.losing_trades,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
        }


def run_full_backtest(
    symbol: str = "BTCUSDT",
    model_type: str = "lstm",
    epochs: int = 20,
    limit: int = 5000,
    interval: str = "1h",
) -> Dict[str, Any]:
    """Run a full backtest with ML training."""
    
    pipeline = MLTrainingPipeline()
    results = pipeline.train_and_backtest(symbol, model_type, epochs, limit, interval)
    
    return results


def run_multi_timeframe_backtest(
    symbol: str = "BTCUSDT",
    intervals: list = None,
) -> Dict[str, Dict]:
    if intervals is None:
        intervals = ["1h", "4h", "1d"]
    
    all_results = {}
    
    for interval in intervals:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing timeframe: {interval}")
        logging.info(f"{'='*50}")
        
        results = run_full_backtest(symbol, interval=interval, limit=1000, epochs=30)
        all_results[interval] = results
    
    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    multi_results = run_multi_timeframe_backtest("BTCUSDT")
    
    print("\n" + "="*50)
    print("MULTI-TIMEFRAME BACKTEST RESULTS")
    print("="*50)
    for timeframe, results in multi_results.items():
        print(f"\n{timeframe}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")