"""
Argus Trading System - Portfolio Risk Management
===============================================

Portfolio-level risk metrics and management.

Features:
- Value at Risk (VaR) calculation
- Conditional VaR (Expected Shortfall)
- Correlation monitoring
- Portfolio exposure limits
- Diversification scoring
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

from core.types import Position, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics snapshot."""
    # Value at Risk
    var_95: float  # 95% VaR (1-day)
    var_99: float  # 99% VaR (1-day)
    cvar_95: float  # Conditional VaR (Expected Shortfall)

    # Exposure
    total_exposure: float  # Total notional exposure
    net_exposure: float    # Long - Short exposure
    gross_exposure: float  # Long + Short exposure
    leverage: float        # Gross exposure / capital

    # Concentration
    max_position_pct: float  # Largest position as % of portfolio
    herfindahl_index: float  # Concentration index (0-1)
    num_positions: int

    # Correlation
    avg_correlation: float  # Average pairwise correlation
    max_correlation: float  # Maximum pairwise correlation

    # Performance
    portfolio_beta: float   # Market beta
    sharpe_ratio: float
    sortino_ratio: float

    # Risk level
    risk_level: RiskLevel
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioLimits:
    """Portfolio risk limits configuration."""
    # Exposure limits
    max_leverage: float = 3.0
    max_gross_exposure_pct: float = 0.80  # 80% of capital
    max_net_exposure_pct: float = 0.50    # 50% of capital

    # Concentration limits
    max_single_position_pct: float = 0.15  # 15% max per position
    max_correlated_exposure_pct: float = 0.30  # 30% in correlated assets
    min_positions: int = 3  # Minimum diversification

    # Correlation limits
    max_avg_correlation: float = 0.70
    correlation_lookback_days: int = 30

    # VaR limits
    max_var_95_pct: float = 0.05  # 5% daily VaR limit
    max_var_99_pct: float = 0.10  # 10% daily VaR limit

    # Drawdown
    max_drawdown_pct: float = 0.15  # 15% max drawdown


class PortfolioRiskManager:
    """
    Portfolio-level risk management.

    Tracks portfolio composition, calculates risk metrics,
    and enforces portfolio-level limits.
    """

    def __init__(
        self,
        initial_capital: float,
        limits: Optional[PortfolioLimits] = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.peak_capital = float(initial_capital)
        self.limits = limits or PortfolioLimits()

        # Position tracking
        self._positions: Dict[str, Position] = {}

        # Returns history for VaR calculation
        self._returns_history: deque = deque(maxlen=252)  # 1 year of daily returns
        self._last_portfolio_value = float(initial_capital)

        # Price history for correlation calculation
        self._price_history: Dict[str, deque] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._correlation_updated: Optional[datetime] = None

        logger.info(
            "PortfolioRiskManager initialized with capital=%.2f",
            initial_capital,
        )

    def update_capital(self, new_capital: float) -> None:
        """Update current capital and track returns."""
        old_capital = self.current_capital
        self.current_capital = float(new_capital)

        if old_capital > 0:
            daily_return = (new_capital - old_capital) / old_capital
            self._returns_history.append(daily_return)

        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        self._last_portfolio_value = new_capital

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        self._positions[position.symbol] = position

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position."""
        return self._positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a position by symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all current positions."""
        return list(self._positions.values())

    def update_price(self, symbol: str, price: float) -> None:
        """Update price for a symbol (for correlation calculation)."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.limits.correlation_lookback_days)
        self._price_history[symbol].append(price)

        # Update position current price if we have the position
        if symbol in self._positions:
            self._positions[symbol].current_price = price

    def calculate_metrics(self) -> PortfolioMetrics:
        """Calculate all portfolio risk metrics."""
        positions = list(self._positions.values())

        # Exposure calculations
        total_long = sum(p.notional_value for p in positions if p.side.value == "buy")
        total_short = sum(p.notional_value for p in positions if p.side.value == "sell")

        gross_exposure = total_long + total_short
        net_exposure = total_long - total_short
        total_exposure = gross_exposure

        leverage = gross_exposure / self.current_capital if self.current_capital > 0 else 0

        # VaR calculations
        var_95, var_99, cvar_95 = self._calculate_var()

        # Concentration metrics
        max_position_pct, herfindahl = self._calculate_concentration(positions)

        # Correlation metrics
        avg_corr, max_corr = self._calculate_correlation_metrics()

        # Performance metrics
        sharpe, sortino = self._calculate_risk_adjusted_returns()
        beta = self._calculate_beta()

        # Determine risk level
        risk_level, warnings = self._assess_risk_level(
            var_95=var_95,
            leverage=leverage,
            max_position_pct=max_position_pct,
            avg_correlation=avg_corr,
        )

        return PortfolioMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            leverage=leverage,
            max_position_pct=max_position_pct,
            herfindahl_index=herfindahl,
            num_positions=len(positions),
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            portfolio_beta=beta,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            risk_level=risk_level,
            warnings=warnings,
        )

    def check_new_position(
        self,
        symbol: str,
        notional_value: float,
        side: str,
    ) -> Tuple[bool, List[str]]:
        """
        Check if a new position would violate portfolio limits.

        Args:
            symbol: Asset symbol
            notional_value: Proposed position value
            side: 'buy' or 'sell'

        Returns:
            Tuple of (approved, list of reasons if rejected)
        """
        limits = self.limits
        warnings = []

        # Current exposure
        positions = list(self._positions.values())
        current_long = sum(p.notional_value for p in positions if p.side.value == "buy")
        current_short = sum(p.notional_value for p in positions if p.side.value == "sell")

        # Projected exposure
        if side.lower() == "buy":
            new_long = current_long + notional_value
            new_short = current_short
        else:
            new_long = current_long
            new_short = current_short + notional_value

        new_gross = new_long + new_short
        new_net = new_long - new_short

        # Check leverage limit
        new_leverage = new_gross / self.current_capital if self.current_capital > 0 else 0
        if new_leverage > limits.max_leverage:
            warnings.append(
                f"Leverage would exceed limit: {new_leverage:.2f}x > {limits.max_leverage:.2f}x"
            )

        # Check gross exposure limit
        gross_pct = new_gross / self.current_capital if self.current_capital > 0 else 0
        if gross_pct > limits.max_gross_exposure_pct:
            warnings.append(
                f"Gross exposure would exceed limit: {gross_pct:.1%} > {limits.max_gross_exposure_pct:.1%}"
            )

        # Check net exposure limit
        net_pct = abs(new_net) / self.current_capital if self.current_capital > 0 else 0
        if net_pct > limits.max_net_exposure_pct:
            warnings.append(
                f"Net exposure would exceed limit: {net_pct:.1%} > {limits.max_net_exposure_pct:.1%}"
            )

        # Check single position concentration
        position_pct = notional_value / self.current_capital if self.current_capital > 0 else 0
        if position_pct > limits.max_single_position_pct:
            warnings.append(
                f"Position size would exceed limit: {position_pct:.1%} > {limits.max_single_position_pct:.1%}"
            )

        # Check correlation with existing positions
        if symbol in self._price_history and len(self._price_history) >= 2:
            correlated_exposure = self._calculate_correlated_exposure(symbol, notional_value)
            corr_pct = correlated_exposure / self.current_capital if self.current_capital > 0 else 0
            if corr_pct > limits.max_correlated_exposure_pct:
                warnings.append(
                    f"Correlated exposure would exceed limit: {corr_pct:.1%} > {limits.max_correlated_exposure_pct:.1%}"
                )

        approved = len(warnings) == 0
        return approved, warnings

    def _calculate_var(self) -> Tuple[float, float, float]:
        """Calculate VaR and CVaR from historical returns."""
        if len(self._returns_history) < 10:
            logger.warning(
                "_calculate_var: insufficient data (%d records, need ≥10); "
                "returning 0.0 sentinel — risk manager may underestimate risk.",
                len(self._returns_history),
            )
            return 0.0, 0.0, 0.0

        returns = np.array(list(self._returns_history))

        # Historical VaR (parametric)
        var_95 = float(np.percentile(returns, 5)) * self.current_capital
        var_99 = float(np.percentile(returns, 1)) * self.current_capital

        # CVaR (Expected Shortfall) - average of returns below VaR
        var_95_pct = np.percentile(returns, 5)
        tail_returns = returns[returns <= var_95_pct]
        if len(tail_returns) > 0:
            cvar_95 = float(np.mean(tail_returns)) * self.current_capital
        else:
            cvar_95 = var_95

        # Return absolute values (losses are positive)
        return abs(var_95), abs(var_99), abs(cvar_95)

    def _calculate_concentration(
        self,
        positions: List[Position],
    ) -> Tuple[float, float]:
        """Calculate concentration metrics."""
        if not positions or self.current_capital <= 0:
            return 0.0, 0.0

        # Position weights
        weights = [p.notional_value / self.current_capital for p in positions]

        # Max position percentage
        max_pct = max(weights) if weights else 0.0

        # Herfindahl-Hirschman Index (sum of squared weights)
        # 1/n for equal weights, 1.0 for single position
        hhi = sum(w * w for w in weights)

        return max_pct, hhi

    def _calculate_correlation_metrics(self) -> Tuple[float, float]:
        """Calculate portfolio correlation metrics."""
        # Need at least 2 assets with price history
        symbols_with_data = [
            s for s, h in self._price_history.items()
            if len(h) >= 10
        ]

        if len(symbols_with_data) < 2:
            return 0.0, 0.0

        # Build returns DataFrame
        returns_dict = {}
        for symbol in symbols_with_data:
            prices = list(self._price_history[symbol])
            if len(prices) >= 2:
                returns = pd.Series(prices).pct_change().dropna().values
                returns_dict[symbol] = returns[-min(len(returns), 30):]  # Last 30 returns

        if len(returns_dict) < 2:
            return 0.0, 0.0

        # Find minimum length
        min_len = min(len(r) for r in returns_dict.values())
        if min_len < 5:
            return 0.0, 0.0

        # Trim to equal length
        returns_df = pd.DataFrame({
            s: r[-min_len:] for s, r in returns_dict.items()
        })

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        # Guard: identical return series produce NaN via division-by-zero; treat as uncorrelated
        corr_matrix = corr_matrix.fillna(0.0)
        self._correlation_matrix = corr_matrix
        self._correlation_updated = datetime.utcnow()

        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corrs = corr_matrix.values[mask]
        # Belt-and-suspenders NaN guard after fillna
        upper_corrs = upper_corrs[~np.isnan(upper_corrs)]

        if len(upper_corrs) == 0:
            return 0.0, 0.0

        avg_corr = float(np.mean(upper_corrs))
        max_corr = float(np.max(upper_corrs))

        return avg_corr, max_corr

    def _calculate_correlated_exposure(
        self,
        new_symbol: str,
        new_notional: float,
    ) -> float:
        """Calculate total exposure to correlated assets."""
        if self._correlation_matrix is None or new_symbol not in self._correlation_matrix.columns:
            return new_notional

        correlated_exposure = new_notional

        for symbol, position in self._positions.items():
            if symbol == new_symbol:
                continue
            if symbol not in self._correlation_matrix.columns:
                continue

            corr = self._correlation_matrix.loc[new_symbol, symbol]
            if abs(corr) > 0.5:  # Consider >0.5 as correlated
                correlated_exposure += position.notional_value * abs(corr)

        return correlated_exposure

    def _calculate_risk_adjusted_returns(self) -> Tuple[float, float]:
        """Calculate Sharpe and Sortino ratios."""
        if len(self._returns_history) < 20:
            return 0.0, 0.0

        returns = np.array(list(self._returns_history))
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Sharpe ratio (annualized, assuming daily returns)
        if std_return > 0:
            sharpe = (mean_return / std_return) * math.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio (using downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                sortino = (mean_return / downside_std) * math.sqrt(252)
            else:
                sortino = 0.0
        else:
            sortino = sharpe  # No negative returns

        return float(sharpe), float(sortino)

    def _calculate_beta(self, market_returns: Optional[pd.Series] = None) -> float:
        """
        Calculate portfolio beta (market sensitivity).

        Uses BTC as market proxy for crypto.  Beta = cov(portfolio, market) / var(market).

        Args:
            market_returns: Series of market (BTC) returns aligned with
                portfolio returns.  If None, attempts to derive from BTC
                price history in ``_price_history``.

        Returns:
            Portfolio beta.  Falls back to 1.0 if insufficient data (<20 samples).
        """
        MIN_SAMPLES = 20

        # Build portfolio returns array
        port_returns = np.array(list(self._returns_history), dtype=float)
        if len(port_returns) < MIN_SAMPLES:
            return 1.0

        # Resolve market returns
        if market_returns is not None:
            mkt = np.asarray(market_returns, dtype=float)
        else:
            # Attempt to derive from BTC price history
            btc_key = None
            for key in self._price_history:
                if "BTC" in key.upper():
                    btc_key = key
                    break
            if btc_key is None or len(self._price_history[btc_key]) < MIN_SAMPLES + 1:
                return 1.0
            btc_prices = list(self._price_history[btc_key])
            mkt = pd.Series(btc_prices).pct_change().dropna().values

        # Align lengths (take the most recent overlapping portion)
        n = min(len(port_returns), len(mkt))
        if n < MIN_SAMPLES:
            return 1.0
        p = port_returns[-n:]
        m = mkt[-n:]

        var_market = float(np.var(m, ddof=0))
        if var_market < 1e-18:
            return 1.0

        cov_pm = float(np.cov(p, m, ddof=0)[0, 1])
        beta = cov_pm / var_market
        return float(beta)

    def _assess_risk_level(
        self,
        var_95: float,
        leverage: float,
        max_position_pct: float,
        avg_correlation: float,
    ) -> Tuple[RiskLevel, List[str]]:
        """Assess overall portfolio risk level."""
        limits = self.limits
        warnings = []
        risk_score = 0

        # VaR check
        var_pct = var_95 / self.current_capital if self.current_capital > 0 else 0
        if var_pct > limits.max_var_99_pct:
            risk_score += 3
            warnings.append(f"VaR95 very high: {var_pct:.1%}")
        elif var_pct > limits.max_var_95_pct:
            risk_score += 2
            warnings.append(f"VaR95 elevated: {var_pct:.1%}")

        # Leverage check
        if leverage > limits.max_leverage:
            risk_score += 3
            warnings.append(f"Leverage exceeded: {leverage:.2f}x")
        elif leverage > limits.max_leverage * 0.8:
            risk_score += 1
            warnings.append(f"Leverage high: {leverage:.2f}x")

        # Concentration check
        if max_position_pct > limits.max_single_position_pct:
            risk_score += 2
            warnings.append(f"Position concentration: {max_position_pct:.1%}")

        # Correlation check
        if avg_correlation > limits.max_avg_correlation:
            risk_score += 2
            warnings.append(f"High correlation: {avg_correlation:.2f}")

        # Drawdown check
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > limits.max_drawdown_pct:
            risk_score += 3
            warnings.append(f"Drawdown exceeded: {drawdown:.1%}")
        elif drawdown > limits.max_drawdown_pct * 0.7:
            risk_score += 1
            warnings.append(f"Drawdown elevated: {drawdown:.1%}")

        # Determine risk level
        if risk_score >= 5:
            return RiskLevel.CRITICAL, warnings
        elif risk_score >= 3:
            return RiskLevel.HIGH, warnings
        elif risk_score >= 1:
            return RiskLevel.MEDIUM, warnings
        else:
            return RiskLevel.LOW, warnings

    @property
    def drawdown(self) -> float:
        """Current drawdown as decimal."""
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as percentage."""
        return self.drawdown * 100


__all__ = [
    "PortfolioRiskManager",
    "PortfolioMetrics",
    "PortfolioLimits",
]
