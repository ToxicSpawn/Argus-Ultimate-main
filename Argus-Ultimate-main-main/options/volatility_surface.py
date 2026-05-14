"""
Volatility Surface Modeling for Options Pricing.

Implements SVI (Stochastic Volatility Inspired) parameterization,
volatility surface construction, arbitrage detection, and Greeks computation
for institutional-grade options trading.

SVI Parameterization (Gatheral, 2004):
    w(k) = a + b * {rho * (k - m) + sqrt((k - m)^2 + sigma^2)}
where w = sigma^2 * T is total variance and k = log(K/F) is log-moneyness.

Usage:
    surface = VolatilitySurface.build_surface(options)
    iv = surface.get_iv(strike=50000, expiry_days=30)
    analyzer = SurfaceAnalyzer(surface)
    skew = analyzer.compute_skew(strike_range=(0.9, 1.1))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class OptionQuote:
    """Single option market quote."""
    strike: float
    expiry_days: int
    bid: float
    ask: float
    iv: float
    option_type: str  # "call" or "put"

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def expiry_years(self) -> float:
        return self.expiry_days / 365.0


@dataclass
class Greeks:
    """Option Greeks from Black-Scholes model."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


@dataclass
class ArbitrageViolation:
    """Represents a detected arbitrage opportunity."""
    violation_type: str
    description: str
    severity: float
    details: Dict[str, float] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# SVI Parameterization
# ═════════════════════════════════════════════════════════════════════════════


class SVIParameterization:
    """
    Stochastic Volatility Inspired (SVI) parameterization of the
    implied volatility smile.

    Parameters (Gatheral 2004):
        a: ATM total variance level
        b: overall smile steepness
        rho: correlation between spot and vol
        m: ATM log-moneyness shift
        sigma: smile curvature
    """

    def __init__(
        self,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ):
        self.params: Dict[str, float] = {
            "a": a,
            "b": b,
            "rho": rho,
            "m": m,
            "sigma": sigma,
        }

    @property
    def a(self) -> float:
        return self.params["a"]

    @property
    def b(self) -> float:
        return self.params["b"]

    @property
    def rho(self) -> float:
        return self.params["rho"]

    @property
    def m(self) -> float:
        return self.params["m"]

    @property
    def sigma(self) -> float:
        return self.params["sigma"]

    def compute_total_variance(self, k: float) -> float:
        """
        Compute total variance w(k) = a + b * {rho*(k-m) + sqrt((k-m)^2 + sigma^2)}.

        Args:
            k: log-moneyness = log(K/F)

        Returns:
            Total variance (sigma^2 * T)
        """
        km = k - self.m
        return self.a + self.b * (self.rho * km + np.sqrt(km ** 2 + self.sigma ** 2))

    def compute_implied_vol(self, k: float, t: float) -> float:
        """
        Compute implied volatility from total variance.

        Args:
            k: log-moneyness
            t: time to expiry in years

        Returns:
            Implied volatility (annualized)
        """
        if t <= 0:
            return 0.0
        w = self.compute_total_variance(k)
        if w < 0:
            logger.warning(
                "Negative total variance at k=%.4f, t=%.4f: w=%.6f", k, t, w
            )
            return 0.0
        return np.sqrt(w / t)

    def check_arbitrage(self) -> bool:
        """
        Check for static arbitrage conditions in the SVI parameterization.

        Conditions (Gatheral & Jacquier 2014):
            1. b >= 0
            2. b * (1 - |rho|) >= 0
            3. sigma > 0

        Returns:
            True if arbitrage-free, False otherwise
        """
        conditions = [
            self.b >= 0,
            self.b * (1 - abs(self.rho)) >= 0,
            self.sigma > 0,
            self.a >= 0,
        ]
        return all(conditions)

    @classmethod
    def fit(cls, market_data: List[OptionQuote]) -> SVIParameterization:
        """
        Fit SVI parameters to market option quotes.

        Args:
            market_data: list of OptionQuote with implied volatilities

        Returns:
            Fitted SVIParameterization
        """
        if len(market_data) < 5:
            logger.warning(
                "Insufficient market data for SVI fitting (%d points), using defaults",
                len(market_data),
            )
            return cls(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.5)

        forward = cls._estimate_forward(market_data)

        strikes = np.array([q.strike for q in market_data])
        expiries = np.array([q.expiry_years for q in market_data])
        ivs = np.array([q.iv for q in market_data])

        k = np.log(strikes / forward)
        w_market = ivs ** 2 * expiries

        def objective(params: np.ndarray) -> float:
            a, b, rho, m, sigma = params
            try:
                km = k - m
                w_model = a + b * (rho * km + np.sqrt(km ** 2 + sigma ** 2))
                w_model = np.maximum(w_model, 1e-8)
                return np.sum((w_market - w_model) ** 2)
            except (ValueError, FloatingPointError):
                return 1e10

        x0 = np.array([0.04, 0.1, -0.3, 0.0, 0.5])
        bounds = [
            (1e-6, 2.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (-0.5, 0.5),
            (1e-6, 2.0),
        ]

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if not result.success:
            logger.warning("SVI fitting did not converge: %s", result.message)

        a, b, rho, m, sigma = result.x
        logger.info(
            "SVI fitted: a=%.4f, b=%.4f, rho=%.4f, m=%.4f, sigma=%.4f (loss=%.6f)",
            a, b, rho, m, sigma, result.fun,
        )
        return cls(a=a, b=b, rho=rho, m=m, sigma=sigma)

    @staticmethod
    def _estimate_forward(quotes: List[OptionQuote]) -> float:
        """Estimate forward price from put-call parity."""
        call_quotes = [q for q in quotes if q.option_type == "call"]
        put_quotes = [q for q in quotes if q.option_type == "put"]

        if not call_quotes or not put_quotes:
            return np.mean([q.strike for q in quotes])

        for c in call_quotes:
            for p in put_quotes:
                if c.strike == p.strike and c.expiry_days == p.expiry_days:
                    return c.strike

        return np.mean([q.strike for q in quotes])


# ═════════════════════════════════════════════════════════════════════════════
# Volatility Surface
# ═════════════════════════════════════════════════════════════════════════════


class VolatilitySurface:
    """
    Implied volatility surface built from market option quotes.

    Uses piecewise SVI parameterization across expiry buckets with
    interpolation for arbitrary (strike, expiry) queries.
    """

    def __init__(
        self,
        options: List[OptionQuote],
        svi_by_expiry: Optional[Dict[int, SVIParameterization]] = None,
        forward: Optional[float] = None,
    ):
        self.options: List[OptionQuote] = options
        self.svi_by_expiry: Dict[int, SVIParameterization] = svi_by_expiry or {}
        self.forward: float = forward or self._estimate_forward()

    def _estimate_forward(self) -> float:
        if not self.options:
            return 0.0
        return np.mean([q.strike for q in self.options])

    @classmethod
    def build_surface(cls, options: List[OptionQuote]) -> VolatilitySurface:
        """
        Build a volatility surface from market option quotes.

        Args:
            options: list of OptionQuote

        Returns:
            VolatilitySurface with fitted SVI per expiry bucket
        """
        if not options:
            logger.warning("No options provided for surface building")
            return cls(options=[])

        forward = SVIParameterization._estimate_forward(options)

        expiry_groups: Dict[int, List[OptionQuote]] = {}
        for q in options:
            expiry_groups.setdefault(q.expiry_days, []).append(q)

        svi_by_expiry: Dict[int, SVIParameterization] = {}
        for expiry_days, group in expiry_groups.items():
            try:
                svi = SVIParameterization.fit(group)
                svi_by_expiry[expiry_days] = svi
            except Exception as e:
                logger.error(
                    "Failed to fit SVI for expiry %d: %s", expiry_days, e
                )

        logger.info(
            "Volatility surface built: %d options, %d expiry buckets",
            len(options), len(svi_by_expiry),
        )
        return cls(
            options=options, svi_by_expiry=svi_by_expiry, forward=forward
        )

    def get_iv(self, strike: float, expiry: float) -> float:
        """
        Get implied volatility for a given strike and expiry.

        Args:
            strike: option strike price
            expiry: expiry in days

        Returns:
            Implied volatility (annualized)
        """
        if self.forward <= 0:
            return 0.0

        k = np.log(strike / self.forward)
        t = expiry / 365.0

        expiries = sorted(self.svi_by_expiry.keys())
        if not expiries:
            return self._fallback_iv(strike, expiry)

        if expiry in self.svi_by_expiry:
            return self.svi_by_expiry[expiry].compute_implied_vol(k, t)

        if expiry <= expiries[0]:
            return self.svi_by_expiry[expiries[0]].compute_implied_vol(k, t)
        if expiry >= expiries[-1]:
            return self.svi_by_expiry[expiries[-1]].compute_implied_vol(k, t)

        lower_exp = max(e for e in expiries if e < expiry)
        upper_exp = min(e for e in expiries if e > expiry)

        lower_svi = self.svi_by_expiry[lower_exp]
        upper_svi = self.svi_by_expiry[upper_exp]

        lower_iv = lower_svi.compute_implied_vol(k, lower_exp / 365.0)
        upper_iv = upper_svi.compute_implied_vol(k, upper_exp / 365.0)

        weight = (expiry - lower_exp) / (upper_exp - lower_exp)
        return lower_iv * (1 - weight) + upper_iv * weight

    def get_total_variance(self, k: float, t: float) -> float:
        """
        Get total variance for log-moneyness k and time t.

        Args:
            k: log-moneyness = log(K/F)
            t: time to expiry in years

        Returns:
            Total variance (sigma^2 * T)
        """
        expiry_days = int(round(t * 365))
        expiries = sorted(self.svi_by_expiry.keys())
        if not expiries:
            return 0.04 * t

        if expiry_days in self.svi_by_expiry:
            return self.svi_by_expiry[expiry_days].compute_total_variance(k)

        if expiry_days <= expiries[0]:
            return self.svi_by_expiry[expiries[0]].compute_total_variance(k)
        if expiry_days >= expiries[-1]:
            return self.svi_by_expiry[expiries[-1]].compute_total_variance(k)

        lower_exp = max(e for e in expiries if e < expiry_days)
        upper_exp = min(e for e in expiries if e > expiry_days)

        lower_var = self.svi_by_expiry[lower_exp].compute_total_variance(k)
        upper_var = self.svi_by_expiry[upper_exp].compute_total_variance(k)

        weight = (expiry_days - lower_exp) / (upper_exp - lower_exp)
        return lower_var * (1 - weight) + upper_var * weight

    def interpolate(
        self, strikes: np.ndarray, expiries: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate IV surface over a grid of strikes and expiries.

        Args:
            strikes: array of strike prices
            expiries: array of expiry days

        Returns:
            2D array of implied volatilities (shape: len(expiries) x len(strikes))
        """
        iv_grid = np.zeros((len(expiries), len(strikes)))
        for i, exp in enumerate(expiries):
            for j, strike in enumerate(strikes):
                iv_grid[i, j] = self.get_iv(strike, exp)
        return iv_grid

    def smooth(self, window: int = 3) -> VolatilitySurface:
        """
        Smooth the volatility surface using moving average filter.

        Args:
            window: smoothing window size

        Returns:
            New VolatilitySurface with smoothed quotes
        """
        if not self.options:
            return self

        sorted_options = sorted(self.options, key=lambda q: (q.expiry_days, q.strike))
        smoothed: List[OptionQuote] = []

        half = window // 2
        for i, opt in enumerate(sorted_options):
            start = max(0, i - half)
            end = min(len(sorted_options), i + half + 1)
            window_options = sorted_options[start:end]

            same_type = [q for q in window_options if q.option_type == opt.option_type]
            if same_type:
                avg_iv = np.mean([q.iv for q in same_type])
            else:
                avg_iv = opt.iv

            smoothed.append(
                OptionQuote(
                    strike=opt.strike,
                    expiry_days=opt.expiry_days,
                    bid=opt.bid,
                    ask=opt.ask,
                    iv=float(avg_iv),
                    option_type=opt.option_type,
                )
            )

        return VolatilitySurface.build_surface(smoothed)

    def _fallback_iv(self, strike: float, expiry: float) -> float:
        if not self.options:
            return 0.5
        return np.mean([q.iv for q in self.options])


# ═════════════════════════════════════════════════════════════════════════════
# Surface Analyzer
# ═════════════════════════════════════════════════════════════════════════════


class SurfaceAnalyzer:
    """Analyze volatility surface characteristics."""

    def __init__(self, surface: VolatilitySurface):
        self.surface = surface

    def compute_skew(self, strike_range: Tuple[float, float] = (0.9, 1.1)) -> float:
        """
        Compute volatility skew (25-delta risk reversal proxy).

        Args:
            strike_range: (lower, upper) as fraction of forward

        Returns:
            Skew = IV(OTM put) - IV(OTM call)
        """
        if self.surface.forward <= 0:
            return 0.0

        k_low = np.log(strike_range[0])
        k_high = np.log(strike_range[1])

        expiry_days = self._most_liquid_expiry()
        t = expiry_days / 365.0

        svi = self.surface.svi_by_expiry.get(expiry_days)
        if not svi:
            return 0.0

        iv_low = svi.compute_implied_vol(k_low, t)
        iv_high = svi.compute_implied_vol(k_high, t)

        return iv_low - iv_high

    def compute_kurtosis(self, strike_range: Tuple[float, float] = (0.85, 1.15)) -> float:
        """
        Compute volatility kurtosis (butterfly spread proxy).

        Args:
            strike_range: (lower, upper) as fraction of forward

        Returns:
            Kurtosis = IV(ATM) - (IV(OTM put) + IV(OTM call)) / 2
        """
        if self.surface.forward <= 0:
            return 0.0

        k_center = 0.0
        k_low = np.log(strike_range[0])
        k_high = np.log(strike_range[1])

        expiry_days = self._most_liquid_expiry()
        t = expiry_days / 365.0

        svi = self.surface.svi_by_expiry.get(expiry_days)
        if not svi:
            return 0.0

        iv_center = svi.compute_implied_vol(k_center, t)
        iv_low = svi.compute_implied_vol(k_low, t)
        iv_high = svi.compute_implied_vol(k_high, t)

        return iv_center - (iv_low + iv_high) / 2.0

    def detect_smile_pattern(self) -> str:
        """
        Detect the volatility smile pattern.

        Returns:
            "smirk" (negative skew), "smile" (symmetric), or "flat"
        """
        skew = self.compute_skew()
        kurtosis = self.compute_kurtosis()

        if abs(skew) < 0.02 and kurtosis < 0.02:
            return "flat"
        if skew > 0.03:
            return "smirk"
        if kurtosis > 0.03:
            return "smile"
        return "smirk" if skew > 0 else "flat"

    def compute_term_structure(
        self, expiries: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Compute the volatility term structure (IV vs time).

        Args:
            expiries: list of expiry days (default: from surface data)

        Returns:
            Array of ATM implied volatilities for each expiry
        """
        if expiries is None:
            expiries = sorted(self.surface.svi_by_expiry.keys())

        if not expiries:
            return np.array([])

        atm_ivs = np.zeros(len(expiries))
        for i, exp in enumerate(expiries):
            svi = self.surface.svi_by_expiry.get(exp)
            if svi:
                atm_ivs[i] = svi.compute_implied_vol(0.0, exp / 365.0)
            else:
                atm_ivs[i] = self.surface.get_iv(self.surface.forward, exp)

        return atm_ivs

    def _most_liquid_expiry(self) -> int:
        if not self.surface.svi_by_expiry:
            return 30
        return max(
            self.surface.svi_by_expiry.keys(),
            key=lambda e: len(
                [q for q in self.surface.options if q.expiry_days == e]
            ),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Arbitrage Checker
# ═════════════════════════════════════════════════════════════════════════════


class ArbitrageChecker:
    """Detect arbitrage opportunities in volatility surfaces."""

    def check_calendar_arbitrage(
        self, surface: VolatilitySurface
    ) -> List[ArbitrageViolation]:
        """
        Check for calendar spread arbitrage (total variance must increase with T).

        Args:
            surface: VolatilitySurface to check

        Returns:
            List of arbitrage violations
        """
        violations: List[ArbitrageViolation] = []
        expiries = sorted(surface.svi_by_expiry.keys())

        if len(expiries) < 2:
            return violations

        strikes = np.linspace(0.8, 1.2, 21)

        for i in range(len(expiries) - 1):
            t1 = expiries[i] / 365.0
            t2 = expiries[i + 1] / 365.0
            svi1 = surface.svi_by_expiry[expiries[i]]
            svi2 = surface.svi_by_expiry[expiries[i + 1]]

            for k in strikes:
                log_k = np.log(k)
                w1 = svi1.compute_total_variance(log_k)
                w2 = svi2.compute_total_variance(log_k)

                if w2 < w1 - 1e-6:
                    violations.append(
                        ArbitrageViolation(
                            violation_type="calendar_arbitrage",
                            description=(
                                f"Total variance decreases from T={t1:.3f} to T={t2:.3f} "
                                f"at k={log_k:.4f}"
                            ),
                            severity=float(abs(w1 - w2)),
                            details={
                                "k": float(log_k),
                                "t1": float(t1),
                                "t2": float(t2),
                                "w1": float(w1),
                                "w2": float(w2),
                            },
                        )
                    )

        if violations:
            logger.warning(
                "Found %d calendar arbitrage violations", len(violations)
            )
        return violations

    def check_butterfly_arbitrage(
        self, surface: VolatilitySurface
    ) -> List[ArbitrageViolation]:
        """
        Check for butterfly arbitrage (call prices must be convex in strike).

        Args:
            surface: VolatilitySurface to check

        Returns:
            List of arbitrage violations
        """
        violations: List[ArbitrageViolation] = []

        for expiry_days, svi in surface.svi_by_expiry.items():
            t = expiry_days / 365.0
            strikes = np.linspace(0.7, 1.3, 61)

            for i in range(1, len(strikes) - 1):
                k_prev = np.log(strikes[i - 1])
                k_curr = np.log(strikes[i])
                k_next = np.log(strikes[i + 1])

                w_prev = svi.compute_total_variance(k_prev)
                w_curr = svi.compute_total_variance(k_curr)
                w_next = svi.compute_total_variance(k_next)

                dk1 = k_curr - k_prev
                dk2 = k_next - k_curr

                second_deriv = (
                    (w_next - w_curr) / dk2 - (w_curr - w_prev) / dk1
                ) / ((dk1 + dk2) / 2)

                if second_deriv < -1e-6:
                    violations.append(
                        ArbitrageViolation(
                            violation_type="butterfly_arbitrage",
                            description=(
                                f"Negative convexity at k={k_curr:.4f}, T={t:.3f}"
                            ),
                            severity=float(abs(second_deriv)),
                            details={
                                "k": float(k_curr),
                                "t": float(t),
                                "second_derivative": float(second_deriv),
                            },
                        )
                    )

        if violations:
            logger.warning(
                "Found %d butterfly arbitrage violations", len(violations)
            )
        return violations

    def check_put_call_parity(
        self, calls: List[OptionQuote], puts: List[OptionQuote]
    ) -> List[ArbitrageViolation]:
        """
        Check put-call parity: C - P = S - K*exp(-r*T).

        Args:
            calls: list of call quotes
            puts: list of put quotes

        Returns:
            List of arbitrage violations
        """
        violations: List[ArbitrageViolation] = []
        r = 0.05

        call_dict: Dict[Tuple[float, int], OptionQuote] = {
            (q.strike, q.expiry_days): q for q in calls
        }

        for put in puts:
            key = (put.strike, put.expiry_days)
            if key not in call_dict:
                continue

            call = call_dict[key]
            T = put.expiry_days / 365.0
            discount = np.exp(-r * T)

            lhs = call.mid - put.mid
            rhs = call.strike * (1 - discount)

            parity_diff = abs(lhs - rhs)
            threshold = 0.01 * call.strike

            if parity_diff > threshold:
                violations.append(
                    ArbitrageViolation(
                        violation_type="put_call_parity",
                        description=(
                            f"Put-call parity violation at K={put.strike}, "
                            f"T={put.expiry_days}d: diff={parity_diff:.4f}"
                        ),
                        severity=float(parity_diff / call.strike),
                        details={
                            "strike": put.strike,
                            "expiry_days": put.expiry_days,
                            "call_mid": call.mid,
                            "put_mid": put.mid,
                            "parity_diff": float(parity_diff),
                        },
                    )
                )

        if violations:
            logger.warning(
                "Found %d put-call parity violations", len(violations)
            )
        return violations

    def is_arbitrage_free(self, surface: VolatilitySurface) -> bool:
        """
        Check if surface is free of static arbitrage.

        Args:
            surface: VolatilitySurface to check

        Returns:
            True if no arbitrage detected
        """
        calendar = self.check_calendar_arbitrage(surface)
        butterfly = self.check_butterfly_arbitrage(surface)

        svi_arb_free = all(
            svi.check_arbitrage() for svi in surface.svi_by_expiry.values()
        )

        return len(calendar) == 0 and len(butterfly) == 0 and svi_arb_free


# ═════════════════════════════════════════════════════════════════════════════
# Volatility Calculator
# ═════════════════════════════════════════════════════════════════════════════


class VolatilityCalculator:
    """Black-Scholes implied volatility and pricing calculator."""

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))

    @staticmethod
    def _norm_pdf(x: float) -> float:
        return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)

    def black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """
        Compute Black-Scholes option price.

        Args:
            S: spot price
            K: strike price
            T: time to expiry in years
            r: risk-free rate
            sigma: implied volatility
            option_type: "call" or "put"

        Returns:
            Option price
        """
        if T <= 0 or sigma <= 0:
            if option_type == "call":
                return max(S - K, 0.0)
            return max(K - S, 0.0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2)
        else:
            price = K * np.exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

        return float(price)

    def black_scholes_iv(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> float:
        """
        Compute implied volatility via Newton-Raphson.

        Args:
            price: market option price
            S: spot price
            K: strike price
            T: time to expiry in years
            r: risk-free rate
            option_type: "call" or "put"
            tol: convergence tolerance
            max_iter: maximum iterations

        Returns:
            Implied volatility
        """
        if T <= 0:
            logger.warning("Time to expiry must be positive")
            return 0.0

        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        if price < intrinsic - 1e-6:
            logger.warning(
                "Option price %.4f below intrinsic %.4f", price, intrinsic
            )
            return 0.0

        sigma = 0.5

        for _ in range(max_iter):
            bs_price = self.black_scholes_price(S, K, T, r, sigma, option_type)
            vega = self._vega(S, K, T, r, sigma)

            if abs(vega) < 1e-12:
                sigma += 0.01
                continue

            diff = bs_price - price
            sigma = sigma - diff / vega

            if sigma <= 0:
                sigma = 0.01

            if abs(diff) < tol:
                return float(sigma)

        logger.warning(
            "IV did not converge after %d iterations (diff=%.6f)",
            max_iter, abs(bs_price - price),
        )
        return float(sigma)

    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> Greeks:
        """
        Compute all Black-Scholes Greeks.

        Args:
            S: spot price
            K: strike price
            T: time to expiry in years
            r: risk-free rate
            sigma: implied volatility
            option_type: "call" or "put"

        Returns:
            Greeks dataclass
        """
        if T <= 0 or sigma <= 0:
            sign = 1 if option_type == "call" else -1
            return Greeks(
                delta=float(sign if S > K else 0.0),
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
            )

        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        nd1 = self._norm_pdf(d1)
        Nd1 = self._norm_cdf(d1)
        Nd2 = self._norm_cdf(d2)

        if option_type == "call":
            delta = float(Nd1)
            rho = float(K * T * np.exp(-r * T) * Nd2 / 100)
            theta_num = -S * nd1 * sigma / (2 * sqrt_t) - r * K * np.exp(-r * T) * Nd2
        else:
            delta = float(Nd1 - 1)
            rho = float(-K * T * np.exp(-r * T) * self._norm_cdf(-d2) / 100)
            theta_num = -S * nd1 * sigma / (2 * sqrt_t) + r * K * np.exp(-r * T) * self._norm_cdf(-d2)

        gamma = float(nd1 / (S * sigma * sqrt_t))
        vega = float(S * nd1 * sqrt_t / 100)
        theta = float(theta_num / 365.0)

        return Greeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
        )

    @staticmethod
    def _vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
        nd1 = np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi)
        return float(S * nd1 * sqrt_t)
