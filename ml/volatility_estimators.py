"""
Advanced Volatility Estimators — range-based and return-based methods.

Implements five classical volatility estimators used in quantitative finance:

  1. **Garman-Klass** (1980) — uses OHLC data, ~7.4x more efficient than
     close-to-close for continuous processes.
  2. **Yang-Zhang** (2000) — handles overnight jumps + drift, best all-round
     estimator for daily bars.
  3. **Parkinson** (1980) — high-low range estimator, ~5.2x more efficient
     than close-to-close.
  4. **Rogers-Satchell** (1991) — drift-independent, handles trending markets.
  5. **Realized Volatility** — standard return-based (close-to-close) with
     optional annualisation.

All estimators work on numpy arrays and return annualised volatility by default
(assuming 252 trading days).  Pass ``annualize=False`` for per-bar variance.

Pure Python + numpy.  No exchange or config dependencies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# Crypto trades 365 days, but traditional finance uses 252.
# Defaulting to 365 for crypto context; caller can override.
DEFAULT_TRADING_DAYS = 365


@dataclass
class OHLCVBar:
    """Single OHLCV bar for volatility estimation."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class VolatilityEstimator:
    """Collection of advanced volatility estimators.

    Parameters
    ----------
    trading_days_per_year : int
        Number of trading days for annualisation (default 365 for crypto).
    """

    def __init__(self, trading_days_per_year: int = DEFAULT_TRADING_DAYS) -> None:
        self.trading_days = trading_days_per_year
        logger.info(
            "VolatilityEstimator initialised (trading_days=%d)", self.trading_days,
        )

    # ------------------------------------------------------------------
    # Garman-Klass (1980)
    # ------------------------------------------------------------------

    def garman_klass(
        self,
        high: Sequence[float],
        low: Sequence[float],
        open_: Sequence[float],
        close: Sequence[float],
        annualize: bool = True,
    ) -> float:
        """Garman-Klass volatility estimator.

        Uses the full OHLC information.  Under geometric Brownian motion
        assumptions it is ~7.4x more efficient than close-to-close.

        .. math::

            \\sigma_{GK}^2 = \\frac{1}{n} \\sum_i
            \\left[ \\frac{1}{2} (\\ln H_i - \\ln L_i)^2
            - (2\\ln 2 - 1)(\\ln C_i - \\ln O_i)^2 \\right]

        Parameters
        ----------
        high, low, open_, close : array-like
            OHLC price arrays (same length, each bar).
        annualize : bool
            If True, multiply variance by trading_days and take sqrt.

        Returns
        -------
        float
            Annualised (or per-bar) volatility.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)
        o = np.asarray(open_, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)

        self._validate_ohlc(h, l, o, c)

        ln_hl = np.log(h / l)
        ln_co = np.log(c / o)

        variance = np.mean(0.5 * ln_hl**2 - (2.0 * math.log(2) - 1.0) * ln_co**2)
        variance = max(variance, 0.0)

        if annualize:
            return float(math.sqrt(variance * self.trading_days))
        return float(math.sqrt(variance))

    # ------------------------------------------------------------------
    # Yang-Zhang (2000)
    # ------------------------------------------------------------------

    def yang_zhang(
        self,
        high: Sequence[float],
        low: Sequence[float],
        open_: Sequence[float],
        close: Sequence[float],
        annualize: bool = True,
    ) -> float:
        """Yang-Zhang volatility estimator.

        Combines overnight (open-to-close) variance, close-to-open variance,
        and the Rogers-Satchell estimator.  It is unbiased and handles both
        opening jumps and intraday drift.

        Parameters
        ----------
        high, low, open_, close : array-like
            OHLC price arrays.
        annualize : bool
            If True, annualise.

        Returns
        -------
        float
            Yang-Zhang volatility.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)
        o = np.asarray(open_, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)

        self._validate_ohlc(h, l, o, c)
        n = len(h)

        # Overnight returns: ln(O_i / C_{i-1})
        ln_oc = np.log(o[1:] / c[:-1])
        # Close-to-open returns: ln(C_i / O_i)
        ln_co = np.log(c / o)

        # Overnight variance
        sigma_o_sq = np.var(ln_oc, ddof=1) if len(ln_oc) > 1 else 0.0
        # Close-to-open variance
        sigma_c_sq = np.var(ln_co, ddof=1) if len(ln_co) > 1 else 0.0

        # Rogers-Satchell component
        rs_var = self._rogers_satchell_variance(h, l, o, c)

        # Yang-Zhang weighting: k = 0.34 / (1.34 + (n+1)/(n-1))
        k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))

        variance = sigma_o_sq + k * sigma_c_sq + (1.0 - k) * rs_var
        variance = max(variance, 0.0)

        if annualize:
            return float(math.sqrt(variance * self.trading_days))
        return float(math.sqrt(variance))

    # ------------------------------------------------------------------
    # Parkinson (1980)
    # ------------------------------------------------------------------

    def parkinson(
        self,
        high: Sequence[float],
        low: Sequence[float],
        annualize: bool = True,
    ) -> float:
        """Parkinson volatility estimator (high-low range).

        Uses only high and low prices.  Under GBM it is ~5.2x more
        efficient than close-to-close.

        .. math::

            \\sigma_P^2 = \\frac{1}{4 n \\ln 2} \\sum_i (\\ln H_i - \\ln L_i)^2

        Parameters
        ----------
        high, low : array-like
            High and low price arrays.
        annualize : bool
            If True, annualise.

        Returns
        -------
        float
            Parkinson volatility.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)

        if len(h) != len(l):
            raise ValueError("high and low must be the same length")
        if len(h) < 1:
            raise ValueError("Need at least 1 bar for Parkinson estimator")
        if np.any(l <= 0) or np.any(h <= 0):
            raise ValueError("Prices must be positive")

        ln_hl = np.log(h / l)
        variance = np.mean(ln_hl**2) / (4.0 * math.log(2))
        variance = max(variance, 0.0)

        if annualize:
            return float(math.sqrt(variance * self.trading_days))
        return float(math.sqrt(variance))

    # ------------------------------------------------------------------
    # Rogers-Satchell (1991)
    # ------------------------------------------------------------------

    def rogers_satchell(
        self,
        high: Sequence[float],
        low: Sequence[float],
        open_: Sequence[float],
        close: Sequence[float],
        annualize: bool = True,
    ) -> float:
        """Rogers-Satchell volatility estimator.

        Drift-independent: works well in trending markets where close-to-close
        and Parkinson estimators are biased.

        .. math::

            \\sigma_{RS}^2 = \\frac{1}{n} \\sum_i
            \\left[ \\ln\\frac{H_i}{C_i} \\ln\\frac{H_i}{O_i}
            + \\ln\\frac{L_i}{C_i} \\ln\\frac{L_i}{O_i} \\right]

        Parameters
        ----------
        high, low, open_, close : array-like
            OHLC price arrays.
        annualize : bool
            If True, annualise.

        Returns
        -------
        float
            Rogers-Satchell volatility.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)
        o = np.asarray(open_, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)

        self._validate_ohlc(h, l, o, c)

        variance = self._rogers_satchell_variance(h, l, o, c)
        variance = max(variance, 0.0)

        if annualize:
            return float(math.sqrt(variance * self.trading_days))
        return float(math.sqrt(variance))

    # ------------------------------------------------------------------
    # Realized Volatility (close-to-close)
    # ------------------------------------------------------------------

    def realized_vol(
        self,
        returns: Sequence[float],
        annualize: bool = True,
    ) -> float:
        """Standard realized volatility from a return series.

        Parameters
        ----------
        returns : array-like
            Log returns or simple returns.
        annualize : bool
            If True, multiply std by sqrt(trading_days).

        Returns
        -------
        float
            Realized volatility.
        """
        r = np.asarray(returns, dtype=np.float64)
        if len(r) < 2:
            logger.warning("Need >= 2 returns for realized_vol, returning 0")
            return 0.0

        std = float(np.std(r, ddof=1))
        if annualize:
            return std * math.sqrt(self.trading_days)
        return std

    # ------------------------------------------------------------------
    # Compare all estimators
    # ------------------------------------------------------------------

    def compare_estimators(
        self,
        ohlcv_data: Sequence[Union[OHLCVBar, dict]],
        annualize: bool = True,
    ) -> Dict[str, float]:
        """Run all estimators on the same OHLCV dataset and return results.

        Parameters
        ----------
        ohlcv_data : list[OHLCVBar | dict]
            Each element must have open, high, low, close fields.
            Dicts must have keys: 'open', 'high', 'low', 'close'.
        annualize : bool
            If True, annualise all estimators.

        Returns
        -------
        dict[str, float]
            Estimator name → volatility value.
        """
        opens, highs, lows, closes = [], [], [], []
        for bar in ohlcv_data:
            if isinstance(bar, dict):
                opens.append(bar["open"])
                highs.append(bar["high"])
                lows.append(bar["low"])
                closes.append(bar["close"])
            else:
                opens.append(bar.open)
                highs.append(bar.high)
                lows.append(bar.low)
                closes.append(bar.close)

        # Close-to-close log returns for realized vol
        c = np.asarray(closes, dtype=np.float64)
        log_returns = np.diff(np.log(c))

        results: Dict[str, float] = {}
        try:
            results["garman_klass"] = self.garman_klass(
                highs, lows, opens, closes, annualize=annualize,
            )
        except Exception as e:
            logger.warning("Garman-Klass failed: %s", e)
            results["garman_klass"] = float("nan")

        try:
            results["yang_zhang"] = self.yang_zhang(
                highs, lows, opens, closes, annualize=annualize,
            )
        except Exception as e:
            logger.warning("Yang-Zhang failed: %s", e)
            results["yang_zhang"] = float("nan")

        try:
            results["parkinson"] = self.parkinson(highs, lows, annualize=annualize)
        except Exception as e:
            logger.warning("Parkinson failed: %s", e)
            results["parkinson"] = float("nan")

        try:
            results["rogers_satchell"] = self.rogers_satchell(
                highs, lows, opens, closes, annualize=annualize,
            )
        except Exception as e:
            logger.warning("Rogers-Satchell failed: %s", e)
            results["rogers_satchell"] = float("nan")

        try:
            results["realized_vol"] = self.realized_vol(
                log_returns.tolist(), annualize=annualize,
            )
        except Exception as e:
            logger.warning("Realized vol failed: %s", e)
            results["realized_vol"] = float("nan")

        logger.info("Volatility comparison: %s", results)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ohlc(
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        close: np.ndarray,
    ) -> None:
        """Validate OHLC arrays: same length, positive, high >= low."""
        if not (len(high) == len(low) == len(open_) == len(close)):
            raise ValueError("OHLC arrays must all be the same length")
        if len(high) < 1:
            raise ValueError("Need at least 1 bar")
        if np.any(high <= 0) or np.any(low <= 0) or np.any(open_ <= 0) or np.any(close <= 0):
            raise ValueError("All prices must be positive")

    @staticmethod
    def _rogers_satchell_variance(
        h: np.ndarray,
        l: np.ndarray,
        o: np.ndarray,
        c: np.ndarray,
    ) -> float:
        """Compute the per-bar Rogers-Satchell variance (not annualised)."""
        ln_hc = np.log(h / c)
        ln_ho = np.log(h / o)
        ln_lc = np.log(l / c)
        ln_lo = np.log(l / o)
        return float(np.mean(ln_hc * ln_ho + ln_lc * ln_lo))
