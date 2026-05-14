"""
Adaptive position sizing based on market conditions.

Implements multi-factor position sizing with adjustments for:
- Volatility regime
- Kelly criterion
- Portfolio correlation
- Drawdown state
- Signal confidence
- Liquidity conditions
- Risk budget allocation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketConditions:
    """Snapshot of current market state."""
    volatility: float = 0.0
    volatility_regime: str = "normal"
    trend_strength: float = 0.0
    liquidity_score: float = 0.5
    correlation_regime: str = "normal"
    market_regime: str = "neutral"
    vix_level: float = 20.0


@dataclass
class PositionSizingConfig:
    """Configuration for adaptive position sizing."""
    base_position_pct: float = 0.02
    max_position_pct: float = 0.10
    min_position_pct: float = 0.005
    kelly_fraction: float = 0.25
    volatility_target: float = 0.15
    max_drawdown_multiplier: float = 0.5


@dataclass
class PositionSize:
    """Result of a position sizing calculation."""
    symbol: str
    base_size: float
    adjusted_size: float
    size_pct: float
    notional_value: float
    sizing_factors: Dict[str, float]
    final_multiplier: float
    max_allowed_size: float
    reason: str


@dataclass
class SizingBreakdown:
    """Detailed breakdown of sizing adjustments."""
    base_size: float
    volatility_adjustment: float
    correlation_adjustment: float
    drawdown_adjustment: float
    confidence_adjustment: float
    liquidity_adjustment: float
    kelly_adjustment: float
    final_size: float


@dataclass
class PortfolioState:
    """Current portfolio snapshot."""
    total_capital: float
    current_positions: Dict[str, float] = field(default_factory=dict)
    current_drawdown: float = 0.0
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    open_risk: float = 0.0


# ---------------------------------------------------------------------------
# VolatilityAdjuster
# ---------------------------------------------------------------------------

class VolatilityAdjuster:
    """Adjust position sizes based on realized vs target volatility."""

    @staticmethod
    def compute_volatility(returns: np.ndarray, window: int = 20) -> float:
        """Compute rolling realized volatility."""
        if len(returns) < window:
            return float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        return float(np.std(returns[-window:], ddof=1))

    @staticmethod
    def annualize_volatility(daily_vol: float, periods_per_year: int = 252) -> float:
        """Annualize a daily volatility figure."""
        return daily_vol * np.sqrt(periods_per_year)

    @staticmethod
    def get_volatility_regime(volatility: float) -> str:
        """Classify volatility into a regime."""
        if volatility < 0.10:
            return "low"
        elif volatility < 0.25:
            return "normal"
        elif volatility < 0.50:
            return "high"
        else:
            return "extreme"

    @staticmethod
    def adjust_for_volatility(base_size: float, current_vol: float, target_vol: float) -> float:
        """Scale position size inversely to volatility ratio."""
        if current_vol <= 0 or target_vol <= 0:
            return base_size
        scalar = target_vol / current_vol
        return base_size * scalar

    @staticmethod
    def compute_volatility_scalar(current_vol: float, target_vol: float) -> float:
        """Return the volatility adjustment scalar (0-2 range)."""
        if current_vol <= 0 or target_vol <= 0:
            return 1.0
        scalar = target_vol / current_vol
        return float(np.clip(scalar, 0.0, 2.0))


# ---------------------------------------------------------------------------
# KellyCalculator
# ---------------------------------------------------------------------------

class KellyCalculator:
    """Kelly criterion position sizing."""

    @staticmethod
    def compute_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute full Kelly fraction: W - (1-W)/R."""
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0
        win_rate = float(np.clip(win_rate, 0.0, 1.0))
        r = avg_win / avg_loss
        kelly = win_rate - (1.0 - win_rate) / r
        return float(kelly)

    @staticmethod
    def compute_fractional_kelly(full_kelly: float, fraction: float = 0.25) -> float:
        """Apply fractional Kelly scaling."""
        return float(np.clip(full_kelly * fraction, 0.0, 1.0))

    @staticmethod
    def compute_modified_kelly(returns: np.ndarray, confidence: float = 1.0) -> float:
        """Compute Kelly from return series with confidence weighting."""
        if len(returns) < 2:
            return 0.0
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        win_rate = len(wins) / len(returns)
        avg_win = float(np.mean(wins))
        avg_loss = float(np.abs(np.mean(losses)))
        full_kelly = KellyCalculator.compute_kelly(win_rate, avg_win, avg_loss)
        fractional = KellyCalculator.compute_fractional_kelly(full_kelly, 0.25)
        return float(fractional * confidence)

    @staticmethod
    def kelly_with_bounds(kelly_pct: float, min_pct: float, max_pct: float) -> float:
        """Clamp Kelly percentage to bounds."""
        return float(np.clip(kelly_pct, min_pct, max_pct))


# ---------------------------------------------------------------------------
# CorrelationAdjuster
# ---------------------------------------------------------------------------

class CorrelationAdjuster:
    """Adjust sizes based on portfolio correlation and diversification."""

    @staticmethod
    def compute_portfolio_correlation(positions: Dict[str, float], returns: np.ndarray) -> float:
        """Compute average pairwise correlation across position returns."""
        if len(positions) < 2 or returns.size < 2:
            return 0.0
        n = min(len(positions), returns.shape[0])
        if n < 2:
            return 0.0
        corr_matrix = np.corrcoef(returns[:n])
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diag = corr_matrix[mask]
        return float(np.mean(off_diag))

    @staticmethod
    def adjust_for_correlation(base_size: float, correlation: float) -> float:
        """Reduce size when correlation is high."""
        correlation = float(np.clip(correlation, -1.0, 1.0))
        if correlation <= 0:
            return base_size
        reduction = 1.0 - (correlation * 0.5)
        return base_size * reduction

    @staticmethod
    def compute_diversification_score(positions: Dict[str, float]) -> float:
        """Score from 0 (concentrated) to 1 (well diversified)."""
        if not positions:
            return 0.0
        total = sum(abs(v) for v in positions.values())
        if total == 0:
            return 0.0
        weights = np.array([abs(v) / total for v in positions.values()])
        n = len(weights)
        if n <= 1:
            return 0.0
        herfindahl = float(np.sum(weights ** 2))
        return float((1.0 - herfindahl) / (1.0 - 1.0 / n))

    @staticmethod
    def reduce_for_correlation(
        positions: Dict[str, float],
        base_sizes: Dict[str, float],
    ) -> Dict[str, float]:
        """Reduce sizes for highly correlated positions."""
        if not positions or not base_sizes:
            return base_sizes
        symbols = list(set(positions.keys()) & set(base_sizes.keys()))
        if len(symbols) < 2:
            return base_sizes
        adjusted = dict(base_sizes)
        for sym in symbols:
            overlap = sum(
                1.0 for other in symbols
                if other != sym and abs(positions.get(other, 0)) > 0
            )
            if overlap > 0:
                factor = 1.0 / (1.0 + overlap * 0.25)
                adjusted[sym] = base_sizes[sym] * factor
        return adjusted


# ---------------------------------------------------------------------------
# DrawdownAdjuster
# ---------------------------------------------------------------------------

class DrawdownAdjuster:
    """Adjust sizes based on current and historical drawdown."""

    @staticmethod
    def compute_current_drawdown(equity_curve: np.ndarray) -> float:
        """Compute current drawdown from peak."""
        if len(equity_curve) == 0:
            return 0.0
        peak = float(np.max(equity_curve))
        current = float(equity_curve[-1])
        if peak <= 0:
            return 0.0
        return float((peak - current) / peak)

    @staticmethod
    def compute_max_drawdown(equity_curve: np.ndarray) -> float:
        """Compute maximum drawdown over the entire curve."""
        if len(equity_curve) < 2:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / np.where(running_max > 0, running_max, 1.0)
        return float(np.max(drawdowns))

    @staticmethod
    def adjust_for_drawdown(base_size: float, current_dd: float, max_dd: float) -> float:
        """Scale down size proportionally to drawdown severity."""
        multiplier = DrawdownAdjuster.get_drawdown_multiplier(current_dd, max_dd)
        return base_size * multiplier

    @staticmethod
    def get_drawdown_multiplier(current_dd: float, max_dd: float) -> float:
        """Return a multiplier in [0, 1] based on drawdown ratio."""
        if max_dd <= 0:
            return 1.0
        ratio = current_dd / max_dd
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return float(1.0 - ratio * 0.5)


# ---------------------------------------------------------------------------
# ConfidenceAdjuster
# ---------------------------------------------------------------------------

class ConfidenceAdjuster:
    """Adjust sizes based on signal confidence."""

    @staticmethod
    def compute_signal_confidence(signal: Dict) -> float:
        """Extract confidence from a signal dict."""
        if isinstance(signal, dict):
            return float(np.clip(signal.get("confidence", 0.5), 0.0, 1.0))
        return 0.5

    @staticmethod
    def adjust_for_confidence(base_size: float, confidence: float) -> float:
        """Scale size by confidence multiplier."""
        multiplier = ConfidenceAdjuster.confidence_to_multiplier(confidence)
        return base_size * multiplier

    @staticmethod
    def confidence_to_multiplier(confidence: float) -> float:
        """Map confidence [0, 1] to multiplier [0.5, 1.5]."""
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return float(0.5 + confidence * 1.0)


# ---------------------------------------------------------------------------
# LiquidityAdjuster
# ---------------------------------------------------------------------------

class LiquidityAdjuster:
    """Adjust sizes based on orderbook liquidity."""

    @staticmethod
    def compute_liquidity_score(orderbook: Dict, order_size: float) -> float:
        """Compute liquidity score from orderbook depth."""
        if not orderbook:
            return 0.0
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids and not asks:
            return 0.0
        total_bid_depth = sum(float(b[1]) for b in bids) if bids else 0.0
        total_ask_depth = sum(float(a[1]) for a in asks) if asks else 0.0
        total_depth = total_bid_depth + total_ask_depth
        if total_depth <= 0:
            return 0.0
        score = min(order_size / total_depth, 1.0) if order_size > 0 else 1.0
        return float(1.0 - score)

    @staticmethod
    def adjust_for_liquidity(base_size: float, liquidity_score: float) -> float:
        """Scale size by liquidity score."""
        liquidity_score = float(np.clip(liquidity_score, 0.0, 1.0))
        if liquidity_score >= 0.8:
            return base_size
        elif liquidity_score >= 0.5:
            return base_size * 0.75
        elif liquidity_score >= 0.2:
            return base_size * 0.5
        else:
            return base_size * 0.25

    @staticmethod
    def compute_max_size_for_liquidity(orderbook: Dict, max_impact_bps: float = 10.0) -> float:
        """Compute maximum order size given a max slippage constraint."""
        if not orderbook:
            return 0.0
        bids = orderbook.get("bids", [])
        if not bids:
            return 0.0
        mid_price = (float(bids[0][0]) + float(orderbook.get("asks", [[0]])[0][0])) / 2
        if mid_price <= 0:
            return 0.0
        max_impact_price = mid_price * (1.0 - max_impact_bps / 10000.0)
        cumulative_size = 0.0
        for price, size in bids:
            if float(price) <= max_impact_price:
                break
            cumulative_size += float(size)
        return float(cumulative_size)


# ---------------------------------------------------------------------------
# RiskBudgetAllocator
# ---------------------------------------------------------------------------

class RiskBudgetAllocator:
    """Allocate capital by risk budget and risk parity."""

    @staticmethod
    def allocate_by_risk_budget(
        assets: List[str],
        risk_budgets: Dict[str, float],
    ) -> Dict[str, float]:
        """Allocate risk budgets across assets (normalized to sum to 1)."""
        if not assets:
            return {}
        total = sum(risk_budgets.get(a, 0.0) for a in assets)
        if total <= 0:
            equal = 1.0 / len(assets)
            return {a: equal for a in assets}
        return {a: risk_budgets.get(a, 0.0) / total for a in assets}

    @staticmethod
    def equal_risk_contribution(
        volatilities: np.ndarray,
        correlations: Optional[np.ndarray] = None,
    ) -> Dict[int, float]:
        """Compute weights for equal risk contribution."""
        if len(volatilities) == 0:
            return {}
        vols = np.asarray(volatilities, dtype=float)
        vols = np.where(vols > 0, vols, 1e-8)
        if correlations is not None and correlations.shape[0] == len(vols):
            cov = correlations * np.outer(vols, vols)
        else:
            cov = np.diag(vols ** 2)
        inv_vol = 1.0 / vols
        weights = inv_vol / np.sum(inv_vol)
        return {i: float(w) for i, w in enumerate(weights)}

    @staticmethod
    def risk_parity_weights(
        volatilities: np.ndarray,
        correlations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute risk parity weights via inverse volatility."""
        if len(volatilities) == 0:
            return np.array([])
        vols = np.asarray(volatilities, dtype=float)
        vols = np.where(vols > 0, vols, 1e-8)
        inv_vol = 1.0 / vols
        weights = inv_vol / np.sum(inv_vol)
        if correlations is not None and correlations.shape[0] == len(vols):
            cov = correlations * np.outer(vols, vols)
            for _ in range(10):
                marginal_risk = cov @ weights
                risk_contrib = weights * marginal_risk
                total_risk = np.sum(risk_contrib)
                if total_risk > 0:
                    adjustment = np.sqrt(risk_contrib / total_risk)
                    weights = weights * adjustment
                    weights = weights / np.sum(weights)
        return weights


# ---------------------------------------------------------------------------
# AdaptivePositionSizer
# ---------------------------------------------------------------------------

class AdaptivePositionSizer:
    """Main adaptive position sizing engine.

    Combines volatility, Kelly, correlation, drawdown, confidence,
    and liquidity adjustments to compute optimal position sizes.
    """

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        self.volatility_adjuster = VolatilityAdjuster()
        self.kelly_calculator = KellyCalculator()
        self.correlation_adjuster = CorrelationAdjuster()
        self.drawdown_adjuster = DrawdownAdjuster()
        self.confidence_adjuster = ConfidenceAdjuster()
        self.liquidity_adjuster = LiquidityAdjuster()
        self.risk_budget_allocator = RiskBudgetAllocator()
        self._sizing_history: List[PositionSize] = []

    def compute_position_size(
        self,
        signal: Dict,
        market_conditions: MarketConditions,
        portfolio_state: PortfolioState,
    ) -> PositionSize:
        """Compute adaptive position size for a signal.

        Args:
            signal: Signal dict with at least 'symbol' and 'confidence'.
            market_conditions: Current market state.
            portfolio_state: Current portfolio snapshot.

        Returns:
            PositionSize with base, adjusted, and breakdown.
        """
        symbol = signal.get("symbol", "UNKNOWN")
        price = signal.get("price", 0.0)
        confidence = self.confidence_adjuster.compute_signal_confidence(signal)

        base_size_pct = self.config.base_position_pct
        base_size = portfolio_state.total_capital * base_size_pct

        sizing_factors: Dict[str, float] = {}

        vol_scalar = self.volatility_adjuster.compute_volatility_scalar(
            market_conditions.volatility,
            self.config.volatility_target,
        )
        sizing_factors["volatility"] = vol_scalar
        size_after_vol = base_size * vol_scalar

        kelly_pct = self.kelly_calculator.compute_modified_kelly(
            np.array(signal.get("returns", [])),
            confidence,
        )
        kelly_scalar = float(np.clip(kelly_pct / base_size_pct, 0.0, 2.0)) if base_size_pct > 0 else 1.0
        sizing_factors["kelly"] = kelly_scalar
        size_after_kelly = size_after_vol * kelly_scalar

        corr_scalar = 1.0
        if market_conditions.correlation_regime == "high":
            corr_scalar = 0.6
        elif market_conditions.correlation_regime == "low":
            corr_scalar = 1.2
        sizing_factors["correlation"] = corr_scalar
        size_after_corr = size_after_kelly * corr_scalar

        dd_multiplier = self.drawdown_adjuster.get_drawdown_multiplier(
            portfolio_state.current_drawdown,
            portfolio_state.peak_equity * self.config.max_drawdown_multiplier if portfolio_state.peak_equity > 0 else 0.0,
        )
        sizing_factors["drawdown"] = dd_multiplier
        size_after_dd = size_after_corr * dd_multiplier

        conf_multiplier = self.confidence_adjuster.confidence_to_multiplier(confidence)
        sizing_factors["confidence"] = conf_multiplier
        size_after_conf = size_after_dd * conf_multiplier

        liq_scalar = 1.0
        if market_conditions.liquidity_score < 0.3:
            liq_scalar = 0.5
        elif market_conditions.liquidity_score < 0.6:
            liq_scalar = 0.75
        sizing_factors["liquidity"] = liq_scalar
        size_after_liq = size_after_conf * liq_scalar

        final_multiplier = (
            vol_scalar * kelly_scalar * corr_scalar * dd_multiplier * conf_multiplier * liq_scalar
        )
        adjusted_size = base_size * final_multiplier

        max_allowed = portfolio_state.total_capital * self.config.max_position_pct
        min_allowed = portfolio_state.total_capital * self.config.min_position_pct
        adjusted_size = float(np.clip(adjusted_size, min_allowed, max_allowed))

        size_pct = adjusted_size / portfolio_state.total_capital if portfolio_state.total_capital > 0 else 0.0
        notional_value = adjusted_size * price if price > 0 else adjusted_size

        regime = market_conditions.market_regime
        vol_regime = market_conditions.volatility_regime
        reason = (
            f"regime={regime}, vol_regime={vol_regime}, "
            f"confidence={confidence:.2f}, dd={portfolio_state.current_drawdown:.4f}"
        )

        result = PositionSize(
            symbol=symbol,
            base_size=base_size,
            adjusted_size=adjusted_size,
            size_pct=size_pct,
            notional_value=notional_value,
            sizing_factors=sizing_factors,
            final_multiplier=final_multiplier,
            max_allowed_size=max_allowed,
            reason=reason,
        )
        self._sizing_history.append(result)
        logger.debug(
            "Position sized: %s base=%.2f adjusted=%.2f pct=%.4f multiplier=%.3f",
            symbol, base_size, adjusted_size, size_pct, final_multiplier,
        )
        return result

    def get_sizing_breakdown(
        self,
        signal: Dict,
        market_conditions: MarketConditions,
        portfolio_state: PortfolioState,
    ) -> SizingBreakdown:
        """Get detailed breakdown of sizing adjustments."""
        price = signal.get("price", 0.0)
        confidence = self.confidence_adjuster.compute_signal_confidence(signal)
        base_size = portfolio_state.total_capital * self.config.base_position_pct

        vol_scalar = self.volatility_adjuster.compute_volatility_scalar(
            market_conditions.volatility,
            self.config.volatility_target,
        )
        vol_adjustment = base_size * vol_scalar - base_size

        kelly_pct = self.kelly_calculator.compute_modified_kelly(
            np.array(signal.get("returns", [])),
            confidence,
        )
        kelly_scalar = float(np.clip(kelly_pct / self.config.base_position_pct, 0.0, 2.0)) if self.config.base_position_pct > 0 else 1.0
        kelly_adjustment = vol_adjustment * kelly_scalar

        corr_scalar = 0.6 if market_conditions.correlation_regime == "high" else (1.2 if market_conditions.correlation_regime == "low" else 1.0)
        corr_adjustment = kelly_adjustment * (corr_scalar - 1.0)

        dd_multiplier = self.drawdown_adjuster.get_drawdown_multiplier(
            portfolio_state.current_drawdown,
            portfolio_state.peak_equity * self.config.max_drawdown_multiplier if portfolio_state.peak_equity > 0 else 0.0,
        )
        dd_adjustment = corr_adjustment * (dd_multiplier - 1.0)

        conf_multiplier = self.confidence_adjuster.confidence_to_multiplier(confidence)
        conf_adjustment = dd_adjustment * conf_multiplier

        liq_scalar = 0.5 if market_conditions.liquidity_score < 0.3 else (0.75 if market_conditions.liquidity_score < 0.6 else 1.0)
        liq_adjustment = conf_adjustment * liq_scalar

        final_size = base_size * vol_scalar * kelly_scalar * corr_scalar * dd_multiplier * conf_multiplier * liq_scalar
        max_allowed = portfolio_state.total_capital * self.config.max_position_pct
        min_allowed = portfolio_state.total_capital * self.config.min_position_pct
        final_size = float(np.clip(final_size, min_allowed, max_allowed))

        return SizingBreakdown(
            base_size=base_size,
            volatility_adjustment=vol_adjustment,
            correlation_adjustment=corr_adjustment,
            drawdown_adjustment=dd_adjustment,
            confidence_adjustment=conf_adjustment,
            liquidity_adjustment=liq_adjustment,
            kelly_adjustment=kelly_adjustment,
            final_size=final_size,
        )

    def update_config(self, config: PositionSizingConfig) -> None:
        """Update the sizing configuration."""
        self.config = config
        logger.info("Position sizing config updated: base_pct=%.4f max_pct=%.4f", config.base_position_pct, config.max_position_pct)

    def get_sizing_history(self, limit: Optional[int] = None) -> List[PositionSize]:
        """Return sizing history, optionally limited to last N entries."""
        if limit is not None:
            return self._sizing_history[-limit:]
        return list(self._sizing_history)

    def clear_history(self) -> None:
        """Clear sizing history."""
        self._sizing_history.clear()
        logger.debug("Sizing history cleared")
