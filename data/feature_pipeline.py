"""
feature_pipeline.py — FreqAI-inspired multi-timeframe feature pipeline.

Auto-expands a base OHLCV candle array into a rich feature matrix by:
  1. Resampling to multiple timeframes (1m, 5m, 15m, 1h, 4h, 1d)
  2. Computing TA features per timeframe (RSI, EMA, MACD, BB, ATR, volume)
  3. Adding lag-shifted versions of each feature (1, 2, 3, 5, 10 bars back)
  4. Adding rolling statistics (mean, std, min, max over configurable windows)
  5. Optionally injecting correlated pair features (ETH, BNB price ratios)
  6. Normalising all features to zero-mean unit-variance (online scaler)

Output
------
  FeatureMatrix
    .X          : np.ndarray (N, F) normalised feature matrix
    .feature_names : List[str] column labels
    .timestamp  : np.ndarray (N,) timestamps

Compatible with MetaLearner, RL training env, and all tentacle evaluators.

Usage
-----
    pipeline = FeaturePipeline(
        timeframes=[1, 5, 60],
        lag_periods=[1, 2, 3, 5],
        rolling_windows=[5, 10, 20],
    )
    fm = pipeline.build(candles_1m)  # candles shape (N, 6)
    X  = fm.X                        # (N, F) feature matrix
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TIMEFRAMES    = [1, 5, 15, 60, 240]    # minutes
DEFAULT_LAG_PERIODS   = [1, 2, 3, 5, 10]
DEFAULT_ROLL_WINDOWS  = [5, 10, 20, 50]
DEFAULT_BASE_INTERVAL = 1                       # base candle interval in minutes


@dataclass
class FeatureMatrix:
    X: np.ndarray                          # (N, F) normalised features
    feature_names: List[str]               # length F
    timestamp: np.ndarray                  # (N,) unix timestamps
    raw: Optional[np.ndarray] = None       # (N, F) un-normalised
    n_features: int = field(init=False)

    def __post_init__(self):
        self.n_features = self.X.shape[1] if self.X.ndim == 2 else 0

    def latest(self) -> np.ndarray:
        """Return most recent feature row (1, F)."""
        return self.X[-1:]

    def tail(self, n: int) -> np.ndarray:
        return self.X[-n:]


class OnlineScaler:
    """Welford online mean/variance scaler. No look-ahead bias."""

    def __init__(self, n_features: int, eps: float = 1e-8) -> None:
        self._n   = np.zeros(n_features)
        self._mean= np.zeros(n_features)
        self._M2  = np.zeros(n_features)
        self._eps = eps

    def update(self, x: np.ndarray) -> np.ndarray:
        """Update stats with row x (1D), return normalised value."""
        self._n   += 1
        delta      = x - self._mean
        self._mean += delta / self._n
        delta2     = x - self._mean
        self._M2  += delta * delta2
        std = np.sqrt(self._M2 / np.maximum(self._n - 1, 1)) + self._eps
        return (x - self._mean) / std

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Normalise full matrix X row-by-row (respects time order)."""
        out = np.zeros_like(X, dtype=np.float64)
        for i, row in enumerate(X):
            out[i] = self.update(row)
        return out

    def reset(self) -> None:
        self._n[:] = 0
        self._mean[:] = 0
        self._M2[:] = 0


class FeaturePipeline:
    """
    Multi-timeframe feature expansion pipeline.

    Parameters
    ----------
    base_interval  : int   interval of input candles in minutes (default 1)
    timeframes     : list  timeframes to resample to in minutes
    lag_periods    : list  bar lags to compute per feature
    rolling_windows: list  rolling stat window sizes
    corr_pairs     : dict  {name: close_price_array} for ratio features
    external_features : dict/array optional aligned external signals
    normalise      : bool  apply OnlineScaler (default True)
    fill_nan       : float value to fill NaN with (default 0.0)
    """

    def __init__(
        self,
        base_interval: int = DEFAULT_BASE_INTERVAL,
        timeframes: Optional[List[int]] = None,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        corr_pairs: Optional[Dict[str, np.ndarray]] = None,
        external_features: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
        external_feature_names: Optional[List[str]] = None,
        normalise: bool = True,
        fill_nan: float = 0.0,
    ) -> None:
        self._base_interval  = base_interval
        self._timeframes     = timeframes or DEFAULT_TIMEFRAMES
        self._lag_periods    = lag_periods or DEFAULT_LAG_PERIODS
        self._roll_windows   = rolling_windows or DEFAULT_ROLL_WINDOWS
        self._corr_pairs     = corr_pairs or {}
        self._external_features = external_features
        self._external_feature_names = external_feature_names
        self._normalise      = normalise
        self._fill_nan       = fill_nan
        self._scaler: Optional[OnlineScaler] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(
        self,
        candles: np.ndarray,
        external_features: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None,
        external_feature_names: Optional[List[str]] = None,
    ) -> FeatureMatrix:
        """
        Build feature matrix from base-interval OHLCV candle array.

        Parameters
        ----------
        candles : np.ndarray (N, 6) [ts, open, high, low, close, volume]

        Returns
        -------
        FeatureMatrix with .X shape (N, F)
        """
        if candles.ndim != 2 or candles.shape[1] < 6:
            raise ValueError("candles must be shape (N, 6): [ts, open, high, low, close, volume]")

        N = len(candles)
        ts= candles[:, 0]
        feature_cols: List[np.ndarray] = []
        feature_names: List[str] = []

        # --- Base OHLCV features ---
        base_feats, base_names = self._base_features(candles)
        feature_cols.extend(base_feats)
        feature_names.extend(base_names)

        # --- Multi-timeframe features ---
        for tf in self._timeframes:
            ratio = tf // self._base_interval
            if ratio <= 1:
                continue
            if len(candles) < ratio * 2:
                continue
            tf_candles = self._resample(candles, ratio)
            tf_feats, tf_names = self._base_features(tf_candles, prefix=f"tf{tf}m")
            # Reindex back to base timeframe
            for feat, name in zip(tf_feats, tf_names):
                feat_reindexed = self._reindex(feat, len(tf_candles), N, ratio)
                feature_cols.append(feat_reindexed)
                feature_names.append(name)

        # --- Correlated pair features ---
        for pair_name, pair_close in self._corr_pairs.items():
            close = candles[:, 4].astype(np.float64)
            if len(pair_close) >= N:
                pair_close_aligned = pair_close[-N:].astype(np.float64)
                ratio_feat = np.where(
                    pair_close_aligned > 0,
                    close / pair_close_aligned,
                    1.0,
                )
                feature_cols.append(ratio_feat)
                feature_names.append(f"ratio_{pair_name}")

        # --- Optional enriched external signals ---
        # Examples: sentiment, on-chain, macro, order-flow, funding, open interest.
        # They are aligned to the latest N bars and treated as first-class features
        # before lag/rolling expansion so models can learn persistence and changes.
        ext = external_features if external_features is not None else self._external_features
        ext_names = external_feature_names or self._external_feature_names
        ext_cols, ext_col_names = self._prepare_external_features(ext, ext_names, N)
        feature_cols.extend(ext_cols)
        feature_names.extend(ext_col_names)

        # --- Assemble raw matrix ---
        raw_X = np.column_stack(feature_cols).astype(np.float64)
        raw_X = np.nan_to_num(raw_X, nan=self._fill_nan, posinf=self._fill_nan, neginf=self._fill_nan)

        # --- Lag features ---
        lag_X, lag_names = self._lag_features(raw_X, feature_names)
        raw_X = np.hstack([raw_X, lag_X])
        feature_names = feature_names + lag_names

        # --- Rolling stat features ---
        roll_X, roll_names = self._rolling_features(raw_X[:, :len(feature_names) - len(lag_names)],
                                                     feature_names[:len(feature_names) - len(lag_names)])
        raw_X = np.hstack([raw_X, roll_X])
        feature_names = feature_names + roll_names

        # Final NaN fill
        raw_X = np.nan_to_num(raw_X, nan=self._fill_nan, posinf=self._fill_nan, neginf=self._fill_nan)

        # --- Normalise ---
        if self._normalise:
            if self._scaler is None or self._scaler._n.shape[0] != raw_X.shape[1]:
                self._scaler = OnlineScaler(raw_X.shape[1])
            norm_X = self._scaler.transform(raw_X)
        else:
            norm_X = raw_X.copy()

        logger.debug(
            "FeaturePipeline: built %d features for %d bars",
            norm_X.shape[1], N,
        )

        return FeatureMatrix(
            X=norm_X,
            feature_names=feature_names,
            timestamp=ts,
            raw=raw_X,
        )

    def reset_scaler(self) -> None:
        """Reset online scaler state (call between independent backtests)."""
        if self._scaler:
            self._scaler.reset()

    @property
    def scaler(self) -> Optional[OnlineScaler]:
        return self._scaler

    def set_external_features(
        self,
        external_features: Optional[Union[Dict[str, np.ndarray], np.ndarray]],
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Set reusable external features for subsequent ``build`` calls."""
        self._external_features = external_features
        self._external_feature_names = feature_names

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _base_features(
        self,
        candles: np.ndarray,
        prefix: str = "",
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Compute base TA features from OHLCV candle array."""
        p = f"{prefix}_" if prefix else ""
        close  = candles[:, 4].astype(np.float64)
        high   = candles[:, 2].astype(np.float64)
        low    = candles[:, 3].astype(np.float64)
        volume = candles[:, 5].astype(np.float64)
        N      = len(close)

        feats: List[np.ndarray] = []
        names: List[str] = []

        # Returns
        ret = np.diff(close, prepend=close[0]) / (np.abs(close) + 1e-10)
        feats.append(ret);  names.append(f"{p}return")

        # Log returns
        log_ret = np.log(close / (np.roll(close, 1) + 1e-10))
        log_ret[0] = 0.0
        feats.append(log_ret); names.append(f"{p}log_return")

        # EMA ratios (price / EMA)
        for period in [10, 20, 50, 100]:
            if N >= period:
                e = self._ema(close, period)
                feats.append(close / (e + 1e-10) - 1.0)
                names.append(f"{p}ema{period}_ratio")

        # RSI
        for period in [7, 14, 21]:
            if N >= period + 1:
                feats.append(self._rsi(close, period) / 100.0 - 0.5)
                names.append(f"{p}rsi{period}")

        # MACD histogram (normalised by price)
        if N >= 35:
            macd_hist = self._macd_hist(close)
            feats.append(macd_hist / (close + 1e-10))
            names.append(f"{p}macd_hist")

        # Bollinger %B
        if N >= 20:
            pct_b = self._bollinger_pctb(close, 20, 2.0)
            feats.append(pct_b - 0.5)
            names.append(f"{p}bb_pctb")

        # ATR normalised by price
        if N >= 15:
            atr = self._atr(high, low, close, 14)
            feats.append(atr / (close + 1e-10))
            names.append(f"{p}atr_ratio")

        # Volume ratio vs 20-bar mean
        if N >= 20:
            vol_mean = np.convolve(volume, np.ones(20) / 20, mode="same")
            feats.append(volume / (vol_mean + 1e-10) - 1.0)
            names.append(f"{p}vol_ratio")

        # High-Low range normalised
        hl_range = (high - low) / (close + 1e-10)
        feats.append(hl_range); names.append(f"{p}hl_range")

        # Close position within bar (0=low, 1=high)
        bar_pos = np.where(
            high - low > 1e-10,
            (close - low) / (high - low),
            0.5,
        )
        feats.append(bar_pos); names.append(f"{p}bar_position")

        return feats, names

    def _prepare_external_features(
        self,
        external_features: Optional[Union[Dict[str, np.ndarray], np.ndarray]],
        feature_names: Optional[List[str]],
        n_rows: int,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Align optional external features to the base candle length.

        The method is deliberately tolerant: missing/short arrays are left-padded
        with ``fill_nan`` and long arrays use their most recent ``n_rows`` values.
        This keeps external adapters optional and avoids breaking offline training
        when one source is unavailable.
        """
        if external_features is None:
            return [], []

        cols: List[np.ndarray] = []
        names: List[str] = []

        if isinstance(external_features, dict):
            items = list(external_features.items())
        else:
            arr = np.asarray(external_features, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            supplied = feature_names or [f"external_{i}" for i in range(arr.shape[1])]
            items = [(name, arr[:, i]) for i, name in enumerate(supplied[: arr.shape[1]])]

        for raw_name, raw_values in items:
            values = np.asarray(raw_values, dtype=np.float64).ravel()
            if values.size == 0:
                aligned = np.full(n_rows, self._fill_nan, dtype=np.float64)
            elif values.size >= n_rows:
                aligned = values[-n_rows:]
            else:
                aligned = np.full(n_rows, self._fill_nan, dtype=np.float64)
                aligned[-values.size:] = values

            clean_name = str(raw_name).strip().lower().replace(" ", "_")
            clean_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in clean_name)
            cols.append(np.nan_to_num(aligned, nan=self._fill_nan, posinf=self._fill_nan, neginf=self._fill_nan))
            names.append(f"ext_{clean_name or 'feature'}")

        return cols, names

    def _lag_features(
        self,
        X: np.ndarray,
        names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Append lagged versions of all base features."""
        lag_cols, lag_names = [], []
        for lag in self._lag_periods:
            lagged = np.roll(X, lag, axis=0)
            lagged[:lag] = 0.0
            lag_cols.append(lagged)
            lag_names.extend([f"{n}_lag{lag}" for n in names])
        if not lag_cols:
            return np.empty((len(X), 0)), []
        return np.hstack(lag_cols), lag_names

    def _rolling_features(
        self,
        X: np.ndarray,
        names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute rolling mean and std for each feature over configured windows."""
        roll_cols, roll_names = [], []
        for win in self._roll_windows:
            kernel = np.ones(win) / win
            for i, name in enumerate(names):
                col = X[:, i]
                roll_mean = np.convolve(col, kernel, mode="same")
                roll_std  = np.array([
                    np.std(col[max(0, j - win):j + 1]) if j >= win else 0.0
                    for j in range(len(col))
                ])
                roll_cols.extend([roll_mean, roll_std])
                roll_names.extend([
                    f"{name}_rmean{win}",
                    f"{name}_rstd{win}",
                ])
        if not roll_cols:
            return np.empty((len(X), 0)), []
        return np.column_stack(roll_cols), roll_names

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(candles: np.ndarray, ratio: int) -> np.ndarray:
        """
        Resample base candles to a higher timeframe by aggregating `ratio` bars.
        OHLCV aggregation: open=first, high=max, low=min, close=last, vol=sum.
        """
        N = len(candles)
        n_out = N // ratio
        if n_out == 0:
            return candles
        trimmed = candles[:n_out * ratio].reshape(n_out, ratio, 6)
        resampled = np.column_stack([
            trimmed[:, 0, 0],             # timestamp: first bar
            trimmed[:, 0, 1],             # open: first
            trimmed[:, :, 2].max(axis=1), # high: max
            trimmed[:, :, 3].min(axis=1), # low: min
            trimmed[:, -1, 4],            # close: last
            trimmed[:, :, 5].sum(axis=1), # volume: sum
        ])
        return resampled

    @staticmethod
    def _reindex(feat: np.ndarray, n_tf: int, n_base: int, ratio: int) -> np.ndarray:
        """
        Expand a higher-timeframe feature array back to base timeframe length
        by forward-filling each value for `ratio` bars.
        """
        out = np.zeros(n_base)
        for i in range(min(n_tf, n_base // ratio)):
            start = i * ratio
            end   = min(start + ratio, n_base)
            out[start:end] = feat[i]
        return out

    # ------------------------------------------------------------------
    # TA helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> np.ndarray:
        out = np.zeros_like(prices)
        k   = 2.0 / (period + 1)
        out[0] = prices[0]
        for i in range(1, len(prices)):
            out[i] = prices[i] * k + out[i - 1] * (1 - k)
        return out

    @staticmethod
    def _rsi(prices: np.ndarray, period: int) -> np.ndarray:
        delta = np.diff(prices, prepend=prices[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = np.convolve(gain, np.ones(period) / period, mode="same")
        avg_l = np.convolve(loss, np.ones(period) / period, mode="same")
        rs    = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-10))
        return 100.0 - (100.0 / (1.0 + rs))

    @classmethod
    def _macd_hist(cls, prices: np.ndarray) -> np.ndarray:
        ema_fast = cls._ema(prices, 12)
        ema_slow = cls._ema(prices, 26)
        macd     = ema_fast - ema_slow
        signal   = cls._ema(macd, 9)
        return macd - signal

    @staticmethod
    def _bollinger_pctb(
        prices: np.ndarray, period: int = 20, std_dev: float = 2.0
    ) -> np.ndarray:
        pct_b = np.full(len(prices), 0.5)
        for i in range(period, len(prices)):
            win   = prices[i - period:i]
            mid   = np.mean(win)
            std   = np.std(win, ddof=1)
            upper = mid + std_dev * std
            lower = mid - std_dev * std
            bw    = upper - lower
            pct_b[i] = (prices[i] - lower) / bw if bw > 0 else 0.5
        return pct_b

    @staticmethod
    def _atr(
        high: np.ndarray, low: np.ndarray,
        close: np.ndarray, period: int = 14,
    ) -> np.ndarray:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:]  - close[:-1]),
            ),
        )
        tr = np.concatenate([[tr[0]], tr])
        atr = np.zeros_like(close)
        atr[0] = tr[0]
        alpha  = 1.0 / period
        for i in range(1, len(tr)):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
        return atr
