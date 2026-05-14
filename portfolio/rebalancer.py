"""
Portfolio Rebalancing Optimizer - Ultimate Edge Module

Provides portfolio optimization:
- Target allocation calculation
- Deviation detection
- Optimal rebalancing triggers
- Tax-loss harvesting
- Transaction cost-aware rebalancing

This module maximizes capital efficiency through intelligent rebalancing.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    quantity: float
    current_price: float
    current_value: float
    target_weight: float
    current_weight: float
    deviation_pct: float


@dataclass
class RebalanceOrder:
    """Rebalancing trade order."""
    symbol: str
    action: str
    quantity: float
    estimated_value: float
    priority: int
    reason: str


@dataclass
class RebalancePlan:
    """Complete rebalancing plan."""
    timestamp: datetime
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    orders: List[RebalanceOrder]
    total_trades: int
    estimated_cost_bps: float
    deviation_before: float
    deviation_after: float


@dataclass
class TaxLot:
    """Tax lot for cost basis tracking."""
    symbol: str
    purchase_date: datetime
    quantity: float
    cost_basis: float
    current_value: float
    unrealized_gain: float
    unrealized_gain_pct: float
    holding_period_days: int
    is_long_term: bool


@dataclass
class TaxLossHarvestOpportunity:
    """Tax-loss harvesting opportunity."""
    symbol: str
    unrealized_loss: float
    tax_savings: float
    replacement_candidates: List[str]
    swap_pairs: List[Tuple[str, str]]


@dataclass
class RebalancerConfig:
    """Configuration for rebalancing."""
    rebalance_threshold_pct: float = 5.0
    min_trade_value: float = 10.0
    max_turnover_pct: float = 25.0
    tax_loss_harvest_enabled: bool = True
    tax_rate: float = 0.25
    short_term_threshold_days: int = 365
    cost_per_trade_bps: float = 10.0


class PortfolioRebalancer:
    """
    Portfolio rebalancing optimizer.

    Features:
    - Target weight calculation
    - Drift detection
    - Optimal rebalancing
    - Tax-loss harvesting
    - Cost-aware rebalancing
    """

    def __init__(
        self,
        config: Optional[RebalancerConfig] = None,
    ):
        self.config = config or RebalancerConfig()
        self._target_weights: Dict[str, float] = {}
        self._positions: Dict[str, Position] = {}
        self._tax_lots: List[TaxLot] = []
        self._rebalance_history: Deque[RebalancePlan] = deque(maxlen=50)

    def set_target_weights(self, weights: Dict[str, float]) -> None:
        """
        Set target allocation weights.

        Args:
            weights: Dict mapping symbol to target weight (0-1)
        """
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            logger.warning("Target weights sum to %.3f, normalizing", total)
            weights = {k: v / total for k, v in weights.items()}

        self._target_weights = weights
        logger.info("Target weights set: %s", weights)

    def update_positions(
        self,
        positions: Dict[str, Tuple[float, float]],
    ) -> List[Position]:
        """
        Update current positions.

        Args:
            positions: Dict mapping symbol to (quantity, current_price)

        Returns:
            List of Position objects with current state
        """
        total_value = sum(qty * price for qty, price in positions.values())
        current_weights = {}

        for symbol, (qty, price) in positions.items():
            value = qty * price
            weight = value / total_value if total_value > 0 else 0.0
            current_weights[symbol] = weight

        self._positions = {}

        all_symbols = set(self._target_weights.keys()) | set(positions.keys())

        for symbol in all_symbols:
            if symbol in positions:
                qty, price = positions[symbol]
                value = qty * price
                weight = current_weights.get(symbol, 0.0)
            else:
                qty, price = 0.0, 0.0
                value = 0.0
                weight = 0.0

            target = self._target_weights.get(symbol, 0.0)
            deviation = (weight - target) if target > 0 else weight

            pos = Position(
                symbol=symbol,
                quantity=qty,
                current_price=price,
                current_value=value,
                target_weight=target,
                current_weight=weight,
                deviation_pct=deviation * 100,
            )
            self._positions[symbol] = pos

        return list(self._positions.values())

    def needs_rebalancing(self) -> Tuple[bool, float]:
        """
        Check if portfolio needs rebalancing.

        Returns:
            (needs_rebalance, max_deviation)
        """
        if not self._positions:
            return False, 0.0

        max_deviation = max(abs(p.deviation_pct) for p in self._positions.values())
        needs_rebalance = max_deviation > self.config.rebalance_threshold_pct

        if needs_rebalance:
            logger.info(
                "Rebalancing needed: max deviation=%.2f%% > threshold=%.2f%%",
                max_deviation, self.config.rebalance_threshold_pct
            )

        return needs_rebalance, max_deviation

    def get_rebalance_orders(
        self,
        portfolio_value: float,
        exclude_symbols: Optional[List[str]] = None,
    ) -> RebalancePlan:
        """
        Generate rebalancing orders.

        Args:
            portfolio_value: Total portfolio value
            exclude_symbols: Symbols to exclude from rebalancing

        Returns:
            RebalancePlan with orders
        """
        if exclude_symbols is None:
            exclude_symbols = []

        needs, deviation_before = self.needs_rebalancing()
        if not needs:
            return RebalancePlan(
                timestamp=datetime.now(),
                current_weights={},
                target_weights=self._target_weights,
                orders=[],
                total_trades=0,
                estimated_cost_bps=0.0,
                deviation_before=0.0,
                deviation_after=0.0,
            )

        orders = []
        total_trade_value = 0.0

        for symbol, pos in self._positions.items():
            if symbol in exclude_symbols:
                continue

            target_value = self._target_weights.get(symbol, 0.0) * portfolio_value
            current_value = pos.current_value
            diff_value = target_value - current_value

            if abs(diff_value) < self.config.min_trade_value:
                continue

            action = "buy" if diff_value > 0 else "sell"
            priority = abs(diff_value) / portfolio_value * 100

            order = RebalanceOrder(
                symbol=symbol,
                action=action,
                quantity=abs(diff_value / pos.current_price) if pos.current_price > 0 else 0,
                estimated_value=abs(diff_value),
                priority=int(priority * 100),
                reason=f"Rebalance {action}: weight {pos.current_weight*100:.2f}% -> {pos.target_weight*100:.2f}%",
            )
            orders.append(order)
            total_trade_value += abs(diff_value)

        orders.sort(key=lambda x: x.priority, reverse=True)

        turnover = total_trade_value / portfolio_value if portfolio_value > 0 else 0.0
        if turnover > self.config.max_turnover_pct:
            scale_factor = self.config.max_turnover_pct / turnover
            for order in orders:
                order.estimated_value *= scale_factor
                order.quantity *= scale_factor
            total_trade_value *= scale_factor
            logger.warning(
                "Turnover %.2f%% exceeds max %.2f%%, scaling orders by %.2f%%",
                turnover * 100, self.config.max_turnover_pct * 100, scale_factor * 100
            )

        current_weights = {s: p.current_weight for s, p in self._positions.items()}
        target_weights = self._target_weights.copy()

        n_trades = len(orders)
        cost_bps = (n_trades * self.config.cost_per_trade_bps) / max(turnover * 100, 1) if turnover > 0 else 0.0

        deviation_after = deviation_before * (1 - turnover) if orders else deviation_before

        plan = RebalancePlan(
            timestamp=datetime.now(),
            current_weights=current_weights,
            target_weights=target_weights,
            orders=orders,
            total_trades=n_trades,
            estimated_cost_bps=cost_bps,
            deviation_before=deviation_before,
            deviation_after=deviation_after,
        )

        self._rebalance_history.append(plan)
        return plan

    def add_tax_lot(
        self,
        symbol: str,
        quantity: float,
        cost_basis: float,
        purchase_date: Optional[datetime] = None,
    ) -> None:
        """Add a tax lot for tracking."""
        if purchase_date is None:
            purchase_date = datetime.now()

        current_value = quantity * self._positions.get(symbol, Position(symbol, 0, 0, 0, 0, 0, 0)).current_price
        unrealized_gain = current_value - cost_basis
        unrealized_gain_pct = unrealized_gain / cost_basis if cost_basis > 0 else 0.0

        holding_days = (datetime.now() - purchase_date).days
        is_long_term = holding_days > self.config.short_term_threshold_days

        lot = TaxLot(
            symbol=symbol,
            purchase_date=purchase_date,
            quantity=quantity,
            cost_basis=cost_basis,
            current_value=current_value,
            unrealized_gain=unrealized_gain,
            unrealized_gain_pct=unrealized_gain_pct,
            holding_period_days=holding_days,
            is_long_term=is_long_term,
        )
        self._tax_lots.append(lot)

    def find_tax_loss_harvest_opportunities(
        self,
        min_loss_pct: float = 0.05,
    ) -> List[TaxLossHarvestOpportunity]:
        """
        Find tax-loss harvesting opportunities.

        Args:
            min_loss_pct: Minimum loss percentage to qualify

        Returns:
            List of TaxLossHarvestOpportunity
        """
        opportunities = []

        losing_lots = [
            lot for lot in self._tax_lots
            if lot.unrealized_gain_pct < -min_loss_pct
        ]

        for lot in losing_lots:
            unrealized_loss = abs(lot.unrealized_gain)
            tax_savings = unrealized_loss * self.config.tax_rate

            replacement_candidates = [
                s for s in self._target_weights.keys()
                if s != lot.symbol and s not in [l.symbol for l in losing_lots]
            ]

            swap_pairs = []
            for candidate in replacement_candidates:
                if candidate in self._positions:
                    swap_pairs.append((lot.symbol, candidate))

            opp = TaxLossHarvestOpportunity(
                symbol=lot.symbol,
                unrealized_loss=unrealized_loss,
                tax_savings=tax_savings,
                replacement_candidates=replacement_candidates,
                swap_pairs=swap_pairs,
            )
            opportunities.append(opp)

        opportunities.sort(key=lambda x: x.unrealized_loss, reverse=True)
        return opportunities

    def calculate_optimal_allocation(
        self,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
        risk_aversion: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate optimal allocation using mean-variance optimization.

        Args:
            expected_returns: Dict of expected annual returns
            volatilities: Dict of annual volatilities
            correlations: Dict of (symbol1, symbol2) -> correlation
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            Dict of optimal weights
        """
        symbols = list(expected_returns.keys())
        n = len(symbols)

        if n == 0:
            return {}
        if n == 1:
            return {symbols[0]: 1.0}

        returns_arr = np.array([expected_returns.get(s, 0.0) for s in symbols])
        vol_arr = np.array([volatilities.get(s, 0.1) for s in symbols])

        corr_matrix = np.eye(n)
        for i, si in enumerate(symbols):
            for j, sj in enumerate(symbols):
                if i != j:
                    corr_matrix[i, j] = correlations.get((si, sj), correlations.get((sj, si), 0.0))

        cov_matrix = np.outer(vol_arr, vol_arr) * corr_matrix

        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n)

            numerator = inv_cov @ returns_arr
            denominator = ones @ inv_cov @ returns_arr

            if denominator == 0:
                return {s: 1.0 / n for s in symbols}

            raw_weights = inv_cov @ ones / (ones @ inv_cov @ ones)

            weights = raw_weights - (risk_aversion / n) * (raw_weights - 1.0 / n)

            weights = np.maximum(weights, 0.0)
            weights = weights / weights.sum()

            return {symbols[i]: float(weights[i]) for i in range(n)}

        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in optimization, using equal weights")
            return {s: 1.0 / n for s in symbols}

    def get_drift_analysis(self) -> Dict[str, float]:
        """Get detailed drift analysis."""
        drifts = {}

        for symbol, pos in self._positions.items():
            drifts[symbol] = pos.deviation_pct

        return drifts

    def get_rebalance_history(self, limit: int = 10) -> List[RebalancePlan]:
        """Get recent rebalancing history."""
        return list(self._rebalance_history)[-limit:]

    def reset(self) -> None:
        """Reset rebalancer state."""
        self._target_weights.clear()
        self._positions.clear()
        self._tax_lots.clear()
        self._rebalance_history.clear()
        logger.info("PortfolioRebalancer reset")
