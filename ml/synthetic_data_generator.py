"""
GAN-Style Synthetic Market Data Generator — bootstrap-based OHLCV synthesis.

Learns statistical properties from real OHLCV data (return distribution,
volatility clustering, autocorrelation structure) and generates synthetic
bars that preserve these properties using block bootstrap resampling.

Supports regime-conditioned generation (normal, bull, bear, crisis,
flash_crash, low_vol) and hardcoded historical stress scenarios.

Usage:
    from ml.synthetic_data_generator import SyntheticDataGenerator

    gen = SyntheticDataGenerator()
    gen.fit(real_ohlcv)
    synthetic = gen.generate(n_bars=1000, regime="bull")
    stress = gen.generate_stress_scenario("2020_covid_crash")
    validation = gen.validate_synthetic(real_ohlcv, synthetic)
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime parameters — mean daily return, vol multiplier, autocorrelation bias
# ---------------------------------------------------------------------------

_REGIME_PARAMS: Dict[str, Dict[str, float]] = {
    "normal":      {"mean_mult": 1.0,  "vol_mult": 1.0, "autocorr_bias": 0.0},
    "bull":        {"mean_mult": 2.5,  "vol_mult": 0.8, "autocorr_bias": 0.15},
    "bear":        {"mean_mult": -2.0, "vol_mult": 1.3, "autocorr_bias": 0.10},
    "crisis":      {"mean_mult": -4.0, "vol_mult": 3.0, "autocorr_bias": 0.25},
    "flash_crash":  {"mean_mult": -8.0, "vol_mult": 5.0, "autocorr_bias": 0.40},
    "low_vol":     {"mean_mult": 0.3,  "vol_mult": 0.3, "autocorr_bias": -0.05},
}

# ---------------------------------------------------------------------------
# Historical stress scenario return patterns (approximate daily returns)
# ---------------------------------------------------------------------------

_STRESS_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "2020_covid_crash": {
        "description": "COVID-19 crypto crash, March 2020 (BTC dropped ~50% in 2 days)",
        "daily_returns": [
            -0.02, -0.01, 0.01, -0.03, -0.05, -0.08, -0.12,
            -0.25, -0.18, 0.10, 0.15, -0.05, 0.08, 0.12,
            0.06, 0.04, 0.03, -0.02, 0.05, 0.07, 0.04,
        ],
        "initial_price": 9000.0,
    },
    "2022_luna_collapse": {
        "description": "LUNA/UST collapse, May 2022 (BTC dropped ~30% over a week)",
        "daily_returns": [
            -0.01, -0.03, -0.05, -0.08, -0.10, -0.07, -0.04,
            -0.06, -0.03, 0.02, -0.02, -0.05, 0.01, 0.03,
            0.02, -0.01, 0.04, 0.03, 0.02, 0.01, 0.02,
        ],
        "initial_price": 40000.0,
    },
    "2021_china_ban": {
        "description": "China crypto ban, May-June 2021 (BTC dropped ~50% over weeks)",
        "daily_returns": [
            -0.03, -0.05, -0.02, -0.08, -0.04, 0.02, -0.06,
            -0.10, -0.07, 0.05, -0.03, -0.04, -0.02, 0.03,
            -0.05, -0.03, 0.02, 0.04, -0.02, 0.01, -0.01,
            0.03, 0.05, 0.02, 0.04, 0.03, 0.01, -0.01,
        ],
        "initial_price": 58000.0,
    },
    "flash_crash_10pct": {
        "description": "Generic 10% flash crash followed by V-shaped recovery",
        "daily_returns": [
            0.01, 0.00, -0.02, -0.10, 0.06, 0.04, 0.02,
            0.01, 0.00, -0.01, 0.01, 0.00, 0.01,
        ],
        "initial_price": 50000.0,
    },
}


class SyntheticDataGenerator:
    """Generate synthetic OHLCV data using block bootstrap resampling.

    Learns the statistical properties of real market data and generates
    new bars that preserve autocorrelation structure, volatility clustering,
    and return distribution characteristics.

    Parameters
    ----------
    block_size : int
        Block length for bootstrap resampling (preserves autocorrelation).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, block_size: int = 10, seed: Optional[int] = None) -> None:
        self.block_size = max(2, block_size)
        self._seed = seed
        self._rng = random.Random(seed)

        # Learned properties (populated by fit())
        self._fitted = False
        self._returns: List[float] = []
        self._vol_ratios: List[float] = []  # high-low range / close
        self._volume_ratios: List[float] = []
        self._mean_return: float = 0.0
        self._std_return: float = 0.01
        self._mean_volume: float = 1000.0
        self._autocorr_lag1: float = 0.0
        self._last_close: float = 100.0

    # ------------------------------------------------------------------
    # Fit — learn from real data
    # ------------------------------------------------------------------

    def fit(self, ohlcv_data: List[Dict[str, Any]]) -> None:
        """Learn statistical properties from real OHLCV data.

        Extracts return distribution, volatility clustering, volume patterns,
        and first-order autocorrelation from the input data.

        Parameters
        ----------
        ohlcv_data : list of dict
            Each dict must have keys ``o``/``open``, ``h``/``high``,
            ``l``/``low``, ``c``/``close``, ``v``/``volume``.

        Raises
        ------
        ValueError
            If fewer than 3 bars are provided.
        """
        if len(ohlcv_data) < 3:
            raise ValueError("Need at least 3 bars to fit; got %d" % len(ohlcv_data))

        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        volumes: List[float] = []

        for bar in ohlcv_data:
            c = float(bar.get("c", bar.get("close", 0)))
            h = float(bar.get("h", bar.get("high", c)))
            lo = float(bar.get("l", bar.get("low", c)))
            v = float(bar.get("v", bar.get("volume", 0)))
            closes.append(c)
            highs.append(h)
            lows.append(lo)
            volumes.append(v)

        # Returns
        self._returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                self._returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

        if not self._returns:
            raise ValueError("Could not compute any returns from the data")

        self._mean_return = sum(self._returns) / len(self._returns)
        self._std_return = (
            sum((r - self._mean_return) ** 2 for r in self._returns) / len(self._returns)
        ) ** 0.5
        if self._std_return < 1e-12:
            self._std_return = 0.01

        # Volatility ratios (intrabar range / close)
        self._vol_ratios = []
        for i in range(len(closes)):
            if closes[i] > 0:
                self._vol_ratios.append((highs[i] - lows[i]) / closes[i])
        if not self._vol_ratios:
            self._vol_ratios = [0.02]

        # Volume
        self._volume_ratios = [v for v in volumes if v > 0]
        self._mean_volume = sum(volumes) / len(volumes) if volumes else 1000.0

        # Autocorrelation (lag-1)
        if len(self._returns) > 2:
            mean_r = self._mean_return
            numerator = sum(
                (self._returns[i] - mean_r) * (self._returns[i - 1] - mean_r)
                for i in range(1, len(self._returns))
            )
            denominator = sum((r - mean_r) ** 2 for r in self._returns)
            self._autocorr_lag1 = numerator / denominator if abs(denominator) > 1e-12 else 0.0
        else:
            self._autocorr_lag1 = 0.0

        self._last_close = closes[-1] if closes else 100.0
        self._fitted = True

        logger.info(
            "SyntheticDataGenerator: fitted on %d bars — mean_ret=%.6f std_ret=%.6f autocorr=%.4f",
            len(ohlcv_data), self._mean_return, self._std_return, self._autocorr_lag1,
        )

    # ------------------------------------------------------------------
    # Generate — produce synthetic bars
    # ------------------------------------------------------------------

    def generate(self, n_bars: int = 1000, regime: str = "normal") -> List[Dict[str, Any]]:
        """Generate synthetic OHLCV bars using block bootstrap resampling.

        Parameters
        ----------
        n_bars : int
            Number of bars to generate.
        regime : str
            Market regime: ``normal``, ``bull``, ``bear``, ``crisis``,
            ``flash_crash``, ``low_vol``.

        Returns
        -------
        list of dict
            Each dict has keys: ``timestamp``, ``open``, ``high``, ``low``,
            ``close``, ``volume``.

        Raises
        ------
        ValueError
            If not fitted or regime is unknown.
        """
        if not self._fitted:
            raise ValueError("Must call fit() before generate()")
        if regime not in _REGIME_PARAMS:
            raise ValueError(f"Unknown regime '{regime}'; valid: {sorted(_REGIME_PARAMS)}")

        params = _REGIME_PARAMS[regime]
        mean_mult = params["mean_mult"]
        vol_mult = params["vol_mult"]
        autocorr_bias = params["autocorr_bias"]

        # Block bootstrap: sample blocks of returns from fitted data
        resampled_returns = self._block_bootstrap(n_bars, mean_mult, vol_mult, autocorr_bias)

        # Build OHLCV bars from returns
        bars: List[Dict[str, Any]] = []
        price = self._last_close
        base_ts = time.time()

        for i, ret in enumerate(resampled_returns):
            open_price = price
            close_price = open_price * (1 + ret)
            close_price = max(close_price, 0.01)  # Floor at 1 cent

            # Intrabar range from learned vol ratios
            vol_ratio_idx = self._rng.randint(0, len(self._vol_ratios) - 1)
            intrabar_range = close_price * self._vol_ratios[vol_ratio_idx] * vol_mult
            mid = (open_price + close_price) / 2
            high_price = mid + intrabar_range / 2
            low_price = mid - intrabar_range / 2
            high_price = max(high_price, max(open_price, close_price))
            low_price = min(low_price, min(open_price, close_price))
            low_price = max(low_price, 0.01)

            # Volume
            if self._volume_ratios:
                vol_idx = self._rng.randint(0, len(self._volume_ratios) - 1)
                volume = self._volume_ratios[vol_idx] * (0.5 + self._rng.random())
            else:
                volume = self._mean_volume * (0.5 + self._rng.random())

            bars.append({
                "timestamp": datetime.fromtimestamp(base_ts + i * 3600, tz=timezone.utc).isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
            })

            price = close_price

        logger.info(
            "SyntheticDataGenerator: generated %d bars regime=%s final_price=%.2f",
            len(bars), regime, price,
        )
        return bars

    def _block_bootstrap(
        self,
        n: int,
        mean_mult: float,
        vol_mult: float,
        autocorr_bias: float,
    ) -> List[float]:
        """Resample returns using circular block bootstrap with regime adjustments.

        Parameters
        ----------
        n : int
            Number of returns to generate.
        mean_mult : float
            Multiplier applied to mean return.
        vol_mult : float
            Multiplier applied to return volatility.
        autocorr_bias : float
            Additional autocorrelation injected via AR(1) filter.

        Returns
        -------
        list of float
        """
        if not self._returns:
            return [0.0] * n

        source = self._returns
        block_size = min(self.block_size, len(source))
        result: List[float] = []
        prev_ret = 0.0

        while len(result) < n:
            # Pick a random start index (circular)
            start = self._rng.randint(0, len(source) - 1)
            for j in range(block_size):
                if len(result) >= n:
                    break
                idx = (start + j) % len(source)
                raw = source[idx]

                # Regime adjustment
                adjusted_mean = self._mean_return * mean_mult
                centered = raw - self._mean_return
                scaled = centered * vol_mult
                regime_ret = adjusted_mean + scaled

                # AR(1) autocorrelation injection
                ac = self._autocorr_lag1 + autocorr_bias
                ac = max(-0.9, min(0.9, ac))
                final_ret = ac * prev_ret + (1 - abs(ac)) * regime_ret

                result.append(final_ret)
                prev_ret = final_ret

        return result[:n]

    # ------------------------------------------------------------------
    # Stress scenarios
    # ------------------------------------------------------------------

    def generate_stress_scenario(self, scenario_name: str) -> List[Dict[str, Any]]:
        """Generate OHLCV bars for a hardcoded historical stress scenario.

        Parameters
        ----------
        scenario_name : str
            One of: ``2020_covid_crash``, ``2022_luna_collapse``,
            ``2021_china_ban``, ``flash_crash_10pct``.

        Returns
        -------
        list of dict
            OHLCV bars matching the stress scenario's return pattern.

        Raises
        ------
        ValueError
            If scenario_name is not recognised.
        """
        if scenario_name not in _STRESS_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'; valid: {sorted(_STRESS_SCENARIOS)}"
            )

        scenario = _STRESS_SCENARIOS[scenario_name]
        daily_returns = scenario["daily_returns"]
        initial_price = scenario["initial_price"]

        bars: List[Dict[str, Any]] = []
        price = initial_price
        base_ts = time.time()

        for i, ret in enumerate(daily_returns):
            open_price = price
            close_price = open_price * (1 + ret)
            close_price = max(close_price, 0.01)

            # Stress scenarios have wider intrabar ranges
            abs_move = abs(open_price - close_price)
            overshoot = abs_move * (0.3 + self._rng.random() * 0.5)
            if ret < 0:
                high_price = open_price + overshoot * 0.3
                low_price = close_price - overshoot
            else:
                high_price = close_price + overshoot
                low_price = open_price - overshoot * 0.3
            high_price = max(high_price, max(open_price, close_price))
            low_price = min(low_price, min(open_price, close_price))
            low_price = max(low_price, 0.01)

            # Volume spikes during stress
            vol_spike = 1.0 + abs(ret) * 20  # More volume on larger moves
            volume = 50000 * vol_spike * (0.8 + self._rng.random() * 0.4)

            bars.append({
                "timestamp": datetime.fromtimestamp(base_ts + i * 86400, tz=timezone.utc).isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
            })
            price = close_price

        logger.info(
            "SyntheticDataGenerator: generated stress scenario '%s' — %d bars, final_price=%.2f",
            scenario_name, len(bars), price,
        )
        return bars

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_synthetic(
        self,
        real_data: List[Dict[str, Any]],
        synthetic_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare statistical properties of real and synthetic data.

        Parameters
        ----------
        real_data : list of dict
            Real OHLCV bars.
        synthetic_data : list of dict
            Synthetic OHLCV bars.

        Returns
        -------
        dict
            Comparison of mean return, std return, skewness, kurtosis,
            autocorrelation, and a quality score in [0, 1].
        """
        def _extract_returns(bars: List[Dict[str, Any]]) -> List[float]:
            closes = [float(b.get("c", b.get("close", 0))) for b in bars]
            returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
            return returns

        def _stats(rets: List[float]) -> Dict[str, float]:
            if not rets:
                return {"mean": 0, "std": 0, "skew": 0, "kurtosis": 0, "autocorr": 0}
            n = len(rets)
            mean = sum(rets) / n
            var = sum((r - mean) ** 2 for r in rets) / n
            std = var ** 0.5
            if std < 1e-12:
                return {"mean": mean, "std": 0, "skew": 0, "kurtosis": 0, "autocorr": 0}

            # Skewness
            skew = sum((r - mean) ** 3 for r in rets) / (n * std ** 3) if n > 2 else 0.0

            # Excess kurtosis
            kurt = sum((r - mean) ** 4 for r in rets) / (n * std ** 4) - 3.0 if n > 3 else 0.0

            # Autocorrelation (lag-1)
            if n > 2:
                num = sum((rets[i] - mean) * (rets[i - 1] - mean) for i in range(1, n))
                den = sum((r - mean) ** 2 for r in rets)
                ac = num / den if abs(den) > 1e-12 else 0.0
            else:
                ac = 0.0

            return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt, "autocorr": ac}

        real_rets = _extract_returns(real_data)
        synth_rets = _extract_returns(synthetic_data)

        real_stats = _stats(real_rets)
        synth_stats = _stats(synth_rets)

        # Quality score: penalise divergence in each statistic
        penalties = 0.0
        comparisons = {}
        for key in ("mean", "std", "skew", "kurtosis", "autocorr"):
            rv = real_stats[key]
            sv = synth_stats[key]
            denom = max(abs(rv), 1e-6)
            rel_diff = abs(rv - sv) / denom
            comparisons[key] = {
                "real": round(rv, 6),
                "synthetic": round(sv, 6),
                "relative_diff": round(rel_diff, 4),
            }
            penalties += min(rel_diff, 1.0)

        quality_score = max(0.0, 1.0 - penalties / 5.0)

        result = {
            "real_bar_count": len(real_data),
            "synthetic_bar_count": len(synthetic_data),
            "comparisons": comparisons,
            "quality_score": round(quality_score, 4),
        }

        logger.info("SyntheticDataGenerator: validation quality_score=%.4f", quality_score)
        return result
