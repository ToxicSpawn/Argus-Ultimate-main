"""
Feature Auto-Discovery for ARGUS.

Automatically generates and evaluates candidate features from raw OHLCV
price data. Ranks features by information coefficient (correlation with
forward returns) and prunes stale features that have lost predictive power.

Usage:
    discoverer = FeatureDiscoverer(max_features=50)
    candidates = discoverer.generate_candidates(ohlcv_df)
    forward_ret = ohlcv_df["close"].pct_change().shift(-1).dropna().values
    ic_scores = discoverer.evaluate_predictive_power(candidates, forward_ret)
    top = discoverer.get_top_features(n=10)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FeatureDiscoverer
# ---------------------------------------------------------------------------


class FeatureDiscoverer:
    """
    Auto-discover predictive features from OHLCV data.

    Parameters
    ----------
    max_features : int
        Maximum number of discovered features to retain (default 50).
    """

    def __init__(self, max_features: int = 50) -> None:
        self._max_features = max(5, int(max_features))
        # name -> {func_name, importance, last_evaluated, ic_score}
        self._discovered_features: Dict[str, Dict[str, Any]] = {}

    # ── Candidate generation ──────────────────────────────────────────────

    def generate_candidates(self, ohlcv_df) -> Dict[str, np.ndarray]:
        """
        Generate candidate features from OHLCV data.

        Parameters
        ----------
        ohlcv_df : DataFrame with columns: open, high, low, close, volume

        Returns
        -------
        dict of {feature_name: array_of_values}
        """
        candidates: Dict[str, np.ndarray] = {}
        n = len(ohlcv_df)
        if n < 10:
            return candidates

        close = np.asarray(ohlcv_df["close"], dtype=float)
        high = np.asarray(ohlcv_df["high"], dtype=float)
        low = np.asarray(ohlcv_df["low"], dtype=float)
        volume = np.asarray(ohlcv_df["volume"], dtype=float)

        # Handle open column (may not exist)
        if "open" in ohlcv_df.columns:
            open_ = np.asarray(ohlcv_df["open"], dtype=float)
        else:
            open_ = close.copy()

        # ── Price ratios ──────────────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            candidates["close_open_ratio"] = np.where(open_ > 0, close / open_, 1.0)
            candidates["high_low_ratio"] = np.where(low > 0, high / low, 1.0)

        for period in [5, 10, 20, 50, 100, 200]:
            if n > period:
                sma = self._rolling_mean(close, period)
                with np.errstate(divide="ignore", invalid="ignore"):
                    candidates[f"close_sma{period}_ratio"] = np.where(
                        sma > 0, close / sma, 1.0
                    )

        # ── Volatility features ───────────────────────────────────────────
        for window in [5, 10, 20, 50]:
            if n > window:
                # Rolling standard deviation
                candidates[f"vol_std_{window}"] = self._rolling_std(close, window)

                # Parkinson volatility (using high/low)
                with np.errstate(divide="ignore", invalid="ignore"):
                    hl_ratio = np.where(low > 0, np.log(high / low), 0.0)
                parkinson = self._rolling_mean(hl_ratio ** 2, window)
                parkinson = np.sqrt(parkinson / (4.0 * np.log(2.0)))
                candidates[f"vol_parkinson_{window}"] = parkinson

        # ── Volume features ───────────────────────────────────────────────
        for window in [5, 10, 20]:
            if n > window:
                vol_sma = self._rolling_mean(volume, window)
                with np.errstate(divide="ignore", invalid="ignore"):
                    candidates[f"volume_ratio_{window}"] = np.where(
                        vol_sma > 0, volume / vol_sma, 1.0
                    )

        # OBV slope
        if n > 5:
            price_dir = np.sign(np.diff(close, prepend=close[0]))
            obv = np.cumsum(volume * price_dir)
            candidates["obv_slope_10"] = self._rolling_slope(obv, min(10, n - 1))

        # ── Momentum features ─────────────────────────────────────────────
        for period in [1, 3, 5, 10, 20]:
            if n > period:
                roc = np.zeros(n)
                roc[period:] = (close[period:] - close[:-period]) / np.maximum(
                    close[:-period], 1e-10
                )
                candidates[f"roc_{period}"] = roc

        # RSI divergence (14-period)
        if n > 14:
            rsi = self._compute_rsi(close, 14)
            candidates["rsi_14"] = rsi

        # MACD histogram slope
        if n > 26:
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd = ema12 - ema26
            signal = self._ema(macd, 9)
            hist = macd - signal
            candidates["macd_hist"] = hist
            candidates["macd_hist_slope"] = self._rolling_slope(hist, 3)

        # ── Calendar features ─────────────────────────────────────────────
        # These need a datetime index; generate synthetic if absent
        if hasattr(ohlcv_df, "index") and hasattr(ohlcv_df.index, "hour"):
            try:
                candidates["hour_of_day"] = np.asarray(ohlcv_df.index.hour, dtype=float)
                candidates["day_of_week"] = np.asarray(ohlcv_df.index.dayofweek, dtype=float)
                candidates["is_weekend"] = (
                    np.asarray(ohlcv_df.index.dayofweek, dtype=float) >= 5
                ).astype(float)
            except Exception as _e:
                logger.debug("feature_discoverer error: %s", _e)

        # Clean up NaN/inf
        for name in list(candidates.keys()):
            arr = candidates[name]
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            candidates[name] = arr

        return candidates

    # ── Predictive power evaluation ───────────────────────────────────────

    def evaluate_predictive_power(
        self,
        features: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Rank features by information coefficient (correlation with forward returns).
        Apply Bonferroni correction for multiple testing.

        Parameters
        ----------
        features : dict of {name: array}
        forward_returns : 1D array of forward returns

        Returns
        -------
        dict of {feature_name: ic_score} sorted by abs(ic)
        """
        fwd = np.asarray(forward_returns, dtype=float)
        n_tests = len(features)
        if n_tests == 0:
            return {}

        # Bonferroni threshold: p < 0.05 / n_tests
        # For large-sample correlation, |r| > z_crit / sqrt(n)
        # z_crit at Bonferroni-adjusted alpha
        from scipy.stats import norm
        alpha_adj = 0.05 / max(n_tests, 1)
        z_crit = norm.ppf(1.0 - alpha_adj / 2.0)

        results: Dict[str, float] = {}
        now = time.time()

        for name, values in features.items():
            arr = np.asarray(values, dtype=float)
            # Align lengths
            min_len = min(len(arr), len(fwd))
            if min_len < 10:
                continue

            x = arr[:min_len]
            y = fwd[:min_len]

            # Remove NaN pairs
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            if len(x) < 10:
                continue

            # Information coefficient = Pearson correlation
            x_std = np.std(x)
            y_std = np.std(y)
            if x_std < 1e-12 or y_std < 1e-12:
                continue

            ic = float(np.corrcoef(x, y)[0, 1])

            # Significance check (Bonferroni)
            r_threshold = z_crit / np.sqrt(max(len(x), 1))
            if abs(ic) >= r_threshold:
                results[name] = round(ic, 6)
                self._discovered_features[name] = {
                    "func_name": name,
                    "importance": abs(ic),
                    "ic_score": ic,
                    "last_evaluated": now,
                    "n_samples": len(x),
                }

        # Sort by absolute IC
        results = dict(sorted(results.items(), key=lambda x: abs(x[1]), reverse=True))

        # Trim to max_features
        if len(self._discovered_features) > self._max_features:
            # Keep only top features by importance
            sorted_feats = sorted(
                self._discovered_features.items(),
                key=lambda x: x[1]["importance"],
                reverse=True,
            )
            self._discovered_features = dict(sorted_feats[:self._max_features])

        return results

    # ── Top features ──────────────────────────────────────────────────────

    def get_top_features(self, n: int = 10) -> List[dict]:
        """Return top N features by predictive power with metadata."""
        sorted_feats = sorted(
            self._discovered_features.items(),
            key=lambda x: x[1]["importance"],
            reverse=True,
        )
        return [
            {
                "name": name,
                "ic_score": info["ic_score"],
                "importance": info["importance"],
                "last_evaluated": info["last_evaluated"],
                "n_samples": info.get("n_samples", 0),
            }
            for name, info in sorted_feats[:n]
        ]

    # ── Pruning ───────────────────────────────────────────────────────────

    def prune_stale_features(self, max_age_days: int = 30) -> int:
        """
        Remove features that haven't been evaluated recently.

        Returns number of features pruned.
        """
        cutoff = time.time() - max_age_days * 86400
        to_remove = [
            name for name, info in self._discovered_features.items()
            if info["last_evaluated"] < cutoff
        ]
        for name in to_remove:
            del self._discovered_features[name]
        return len(to_remove)

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Simple rolling mean using cumsum."""
        n = len(arr)
        result = np.full(n, np.nan)
        if n < window:
            return result
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        result[window - 1:] = (cs[window:] - cs[:-window]) / window
        # Forward fill NaN at start
        result[:window - 1] = result[window - 1] if not np.isnan(result[window - 1]) else 0.0
        return result

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation."""
        n = len(arr)
        result = np.zeros(n)
        for i in range(window - 1, n):
            result[i] = float(np.std(arr[max(0, i - window + 1):i + 1]))
        return result

    @staticmethod
    def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling linear regression slope."""
        n = len(arr)
        result = np.zeros(n)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)
        if x_var < 1e-12:
            return result
        for i in range(window - 1, n):
            y = arr[max(0, i - window + 1):i + 1]
            if len(y) < window:
                continue
            y_mean = y.mean()
            result[i] = float(np.sum((x - x_mean) * (y - y_mean)) / x_var)
        return result

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        n = len(close)
        rsi = np.full(n, 50.0)
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        if len(gains) < period:
            return rsi

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss > 1e-10:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
            else:
                rsi[i + 1] = 100.0

        return rsi

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        """Exponential moving average."""
        n = len(arr)
        result = np.zeros(n)
        alpha = 2.0 / (span + 1.0)
        result[0] = arr[0]
        for i in range(1, n):
            result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
        return result

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return current state for dashboard/logging."""
        return {
            "discovered_features": len(self._discovered_features),
            "top_features": self.get_top_features(5),
        }
