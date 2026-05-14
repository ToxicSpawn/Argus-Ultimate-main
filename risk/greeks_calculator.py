"""
Options Greeks Calculator — real-time delta, gamma, vega, theta, rho for crypto options.

Uses Black-Scholes-Merton (BSM) model adapted for crypto (no dividends, European
exercise). Supports calls and puts as traded on Deribit for BTC/ETH.

Greeks computed:
  delta : dV/dS     — directional exposure per $1 move in underlying
  gamma : d²V/dS²   — rate of delta change; convexity
  vega  : dV/dσ     — sensitivity to a 1% (absolute) vol change
  theta : dV/dt     — daily time decay in USD (negative for long options)
  rho   : dV/dr     — interest rate sensitivity (small for crypto)

BSM formulae (no dividend yield):
  T   = (expiry_ts - now) / 31_536_000   [in years]
  d1  = (ln(S/K) + (r + σ²/2)*T) / (σ*√T)
  d2  = d1 - σ*√T
  C   = S*N(d1) - K*e^{-rT}*N(d2)
  P   = K*e^{-rT}*N(-d2) - S*N(-d1)

Pure-Python normal CDF via Abramowitz & Stegun rational approximation (§26.2.17),
accurate to ±7.5e-8.  scipy.stats.norm is used when scipy is available.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from scipy.stats import norm as _scipy_norm  # type: ignore
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.debug("scipy not available — using pure-Python normal approximation")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionSpec:
    """Specification of a single option contract."""
    symbol: str          # e.g. "BTC-28MAR25-80000-C"
    underlying: str      # e.g. "BTC"
    strike: float        # strike price in USD
    expiry_ts: float     # Unix timestamp of expiry (UTC)
    option_type: OptionType
    position_size: float = 1.0  # number of contracts (negative = short)


@dataclass
class Greeks:
    """Complete Greeks snapshot for one option position."""
    delta: float
    gamma: float
    vega: float           # per 1% absolute vol move
    theta: float          # per calendar day (USD)
    rho: float
    iv: float             # implied vol used for computation
    intrinsic_value: float
    time_value: float
    timestamp: float = field(default_factory=time.time)

    def scale(self, qty: float) -> "Greeks":
        """Return Greeks scaled by position quantity."""
        return Greeks(
            delta=self.delta * qty,
            gamma=self.gamma * qty,
            vega=self.vega * qty,
            theta=self.theta * qty,
            rho=self.rho * qty,
            iv=self.iv,
            intrinsic_value=self.intrinsic_value * qty,
            time_value=self.time_value * qty,
            timestamp=self.timestamp,
        )

    def __add__(self, other: "Greeks") -> "Greeks":
        """Aggregate Greeks (for portfolio summation)."""
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
            iv=(self.iv + other.iv) / 2.0,  # average iv — informational only
            intrinsic_value=self.intrinsic_value + other.intrinsic_value,
            time_value=self.time_value + other.time_value,
            timestamp=max(self.timestamp, other.timestamp),
        )


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class GreeksCalculator:
    """
    Black-Scholes-Merton Greeks for European crypto options.

    Usage:
        calc = GreeksCalculator(risk_free_rate=0.05)
        spec = OptionSpec("BTC-25APR25-90000-C", "BTC", 90_000,
                          expiry_ts=..., option_type=OptionType.CALL)
        greeks = calc.compute(spec, spot_price=85_000, implied_vol=0.65)
    """

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        """
        Parameters
        ----------
        risk_free_rate:
            Annualised continuously-compounded rate (e.g. 0.05 = 5%).
        """
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute(
        self,
        spec: OptionSpec,
        spot_price: float,
        implied_vol: float,
    ) -> Greeks:
        """
        Compute full Greeks for a single option.

        Parameters
        ----------
        spec:
            Option specification.
        spot_price:
            Current underlying price in USD.
        implied_vol:
            Implied volatility as decimal (e.g. 0.65 = 65%).

        Returns
        -------
        Greeks
            Scaled by spec.position_size. Theta is per calendar day.
        """
        S = spot_price
        K = spec.strike
        r = self.risk_free_rate
        sigma = implied_vol
        now = time.time()
        T = max((spec.expiry_ts - now) / 31_536_000, 1e-6)  # avoid T≤0

        try:
            d1 = self._d1(S, K, T, r, sigma)
            d2 = self._d2(S, K, T, r, sigma)
        except (ValueError, ZeroDivisionError):
            logger.warning(
                "BSM d1/d2 computation failed for %s (S=%.2f K=%.2f T=%.6f σ=%.4f)",
                spec.symbol, S, K, T, sigma,
            )
            return self._zero_greeks(implied_vol)

        Nd1 = self._norm_cdf(d1)
        Nd2 = self._norm_cdf(d2)
        Nnd1 = self._norm_cdf(-d1)
        Nnd2 = self._norm_cdf(-d2)
        nd1 = self._norm_pdf(d1)

        sqrt_T = math.sqrt(T)
        exp_rT = math.exp(-r * T)

        # Option price
        if spec.option_type == OptionType.CALL:
            price = S * Nd1 - K * exp_rT * Nd2
            delta = Nd1
            intrinsic = max(S - K, 0.0)
        else:
            price = K * exp_rT * Nnd2 - S * Nnd1
            delta = Nd1 - 1.0       # N(d1) - 1
            intrinsic = max(K - S, 0.0)

        time_val = max(price - intrinsic, 0.0)

        # Common Greeks
        gamma = nd1 / (S * sigma * sqrt_T)
        vega = S * nd1 * sqrt_T / 100.0  # per 1% vol

        if spec.option_type == OptionType.CALL:
            theta = (-(S * nd1 * sigma) / (2.0 * sqrt_T)
                     - r * K * exp_rT * Nd2) / 365.0
            rho = K * T * exp_rT * Nd2 / 100.0  # per 1% rate move
        else:
            theta = (-(S * nd1 * sigma) / (2.0 * sqrt_T)
                     + r * K * exp_rT * Nnd2) / 365.0
            rho = -K * T * exp_rT * Nnd2 / 100.0

        raw = Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            iv=sigma,
            intrinsic_value=intrinsic,
            time_value=time_val,
        )
        return raw.scale(spec.position_size)

    def portfolio_greeks(
        self,
        positions: List[Tuple[OptionSpec, float, float]],
    ) -> Greeks:
        """
        Aggregate Greeks across a portfolio.

        Parameters
        ----------
        positions:
            List of (OptionSpec, spot_price, implied_vol) tuples.

        Returns
        -------
        Greeks
            Sum of all position Greeks (weighted by position_size in each spec).
        """
        if not positions:
            return self._zero_greeks(0.0)

        total = self._zero_greeks(0.0)
        for spec, spot, iv in positions:
            try:
                g = self.compute(spec, spot, iv)
                total = total + g
            except Exception:
                logger.exception(
                    "Failed to compute Greeks for %s — skipping", spec.symbol
                )
        return total

    def delta_hedge_size(
        self,
        spec: OptionSpec,
        spot: float,
        iv: float,
    ) -> float:
        """
        Compute underlying quantity needed to delta-hedge the position.

        A long call (delta 0.6, qty 1) requires selling 0.6 BTC.
        Returns negative when the hedge requires selling the underlying.
        """
        g = self.compute(spec, spot, iv)
        # Hedge: sell delta * qty of underlying
        return -g.delta  # already scaled by position_size

    # ------------------------------------------------------------------
    # BSM internals
    # ------------------------------------------------------------------

    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """BSM d1 coefficient."""
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    def _d2(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """BSM d2 coefficient."""
        return self._d1(S, K, T, r, sigma) - sigma * math.sqrt(T)

    def _norm_cdf(self, x: float) -> float:
        """
        Standard normal CDF N(x).

        Uses scipy when available; otherwise Abramowitz & Stegun §26.2.17
        rational approximation (max error ±7.5e-8).
        """
        if _SCIPY_AVAILABLE:
            return float(_scipy_norm.cdf(x))
        return _abramowitz_stegun_cdf(x)

    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF n(x) = exp(-x²/2) / sqrt(2π)."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_greeks(iv: float) -> Greeks:
        return Greeks(
            delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0,
            iv=iv, intrinsic_value=0.0, time_value=0.0,
        )


# ---------------------------------------------------------------------------
# Pure-Python normal CDF (Abramowitz & Stegun 26.2.17)
# ---------------------------------------------------------------------------

def _abramowitz_stegun_cdf(x: float) -> float:
    """
    Standard normal CDF approximation.

    Abramowitz & Stegun §26.2.17 (1964), rational approximation.
    Absolute error <= 7.5e-8 over entire real line.
    """
    p = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    neg = x < 0.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    cdf = 1.0 - pdf * poly
    return 1.0 - cdf if neg else cdf
