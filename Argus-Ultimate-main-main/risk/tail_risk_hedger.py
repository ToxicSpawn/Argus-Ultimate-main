"""Tail risk hedging utilities for portfolio crash protection.

This module sizes and rebalances tail hedges using deep out-of-the-money
puts, VIX calls, and a Constant Proportion Portfolio Insurance (CPPI)
overlay. It is designed to integrate with the wider Argus risk stack via
dependency injection rather than direct hard coupling to one runtime.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class SupportsPortfolio(Protocol):
    """Protocol for portfolio integrations."""

    def get_positions(self) -> Any:
        """Return current positions."""

    def get_total_value(self) -> float:
        """Return total marked portfolio value."""


class SupportsOptionsFeed(Protocol):
    """Protocol for options chain integrations."""

    def get_option_chain(self, underlying: str) -> Any:
        """Return an option chain for the requested underlying."""


class SupportsVolatilityFeed(Protocol):
    """Protocol for volatility data integrations."""

    def get_vix_level(self) -> float:
        """Return the current VIX or synthetic volatility index."""


class SupportsRiskManager(Protocol):
    """Protocol for risk-system integrations."""

    def get_risk_metrics(self) -> Any:
        """Return current risk metrics."""


@dataclass
class TailHedgeConfig:
    """Configuration for tail risk hedging."""

    hedge_allocation_pct: float = 0.02
    otm_percentage: float = 0.15
    hedge_instruments: List[str] = field(
        default_factory=lambda: ["puts", "vix_calls", "cppi"]
    )
    rebalance_frequency: str = "weekly"
    max_hedge_cost_pct: float = 0.02

    def __post_init__(self) -> None:
        if not 0.01 <= float(self.hedge_allocation_pct) <= 0.05:
            raise ValueError("hedge_allocation_pct must be between 0.01 and 0.05")
        if not 0.10 <= float(self.otm_percentage) <= 0.30:
            raise ValueError("otm_percentage must be between 0.10 and 0.30")
        if float(self.max_hedge_cost_pct) <= 0:
            raise ValueError("max_hedge_cost_pct must be positive")

        allowed_instruments = {"puts", "vix_calls", "cppi"}
        normalized_instruments = [str(item).lower() for item in self.hedge_instruments]
        invalid = sorted(set(normalized_instruments) - allowed_instruments)
        if invalid:
            raise ValueError(f"unsupported hedge instruments: {', '.join(invalid)}")
        if not normalized_instruments:
            raise ValueError("hedge_instruments cannot be empty")

        allowed_frequencies = {"daily", "weekly", "monthly"}
        frequency = str(self.rebalance_frequency).lower()
        if frequency not in allowed_frequencies:
            raise ValueError("rebalance_frequency must be daily, weekly, or monthly")

        self.hedge_instruments = normalized_instruments
        self.rebalance_frequency = frequency
        self.hedge_allocation_pct = float(self.hedge_allocation_pct)
        self.otm_percentage = float(self.otm_percentage)
        self.max_hedge_cost_pct = float(self.max_hedge_cost_pct)


@dataclass
class HedgeOrder:
    """Execution-neutral hedge order instruction."""

    instrument_type: str
    symbol: str
    side: str
    quantity: float
    notional_usd: float
    estimated_cost_usd: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_type": self.instrument_type,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "notional_usd": float(self.notional_usd),
            "estimated_cost_usd": float(self.estimated_cost_usd),
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
            "generated_at": self.generated_at.isoformat(),
        }


class TailRiskHedger:
    """Tail risk hedging engine for deep drawdown protection."""

    _MIN_OBSERVATIONS = 10
    _EPSILON = 1e-12

    def __init__(
        self,
        config: Optional[TailHedgeConfig] = None,
        portfolio: Optional[SupportsPortfolio] = None,
        options_data_feed: Optional[SupportsOptionsFeed] = None,
        volatility_data_feed: Optional[SupportsVolatilityFeed] = None,
        risk_manager: Optional[SupportsRiskManager] = None,
    ) -> None:
        self.config = config or TailHedgeConfig()
        self.portfolio = portfolio
        self.options_data_feed = options_data_feed
        self.volatility_data_feed = volatility_data_feed
        self.risk_manager = risk_manager

        self._last_orders: List[HedgeOrder] = []
        self._last_report: Dict[str, Any] = {}
        self._last_rebalance_at: Optional[datetime] = None
        self._last_metrics: Dict[str, float] = {}

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.99) -> float:
        """Calculate historical Value at Risk from returns."""
        arr = self._normalize_returns(returns)
        if arr.size < self._MIN_OBSERVATIONS:
            return 0.0
        percentile = (1.0 - float(confidence)) * 100.0
        return max(0.0, -float(np.percentile(arr, percentile)))

    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.99) -> float:
        """Calculate Conditional Value at Risk from returns."""
        arr = self._normalize_returns(returns)
        if arr.size < self._MIN_OBSERVATIONS:
            return 0.0
        var_threshold = -self.calculate_var(arr, confidence=confidence)
        tail_losses = arr[arr <= var_threshold]
        if tail_losses.size == 0:
            return self.calculate_var(arr, confidence=confidence)
        return max(0.0, -float(np.mean(tail_losses)))

    def calculate_expected_shortfall(self, returns: np.ndarray) -> float:
        """Calculate expected shortfall using the configuration default tail level."""
        return self.calculate_cvar(returns, confidence=0.99)

    def size_put_hedge(self, portfolio_value: float, var: float, otm_pct: float) -> dict:
        """Size a rolling deep OTM put hedge."""
        portfolio_value = max(float(portfolio_value), 0.0)
        var = max(float(var), 0.0)
        otm_pct = max(float(otm_pct), 0.0)

        hedge_ratio = self._dynamic_hedge_ratio(var=var)
        target_notional = portfolio_value * self.config.hedge_allocation_pct * hedge_ratio
        strike_pct = max(0.01, 1.0 - otm_pct)
        premium_rate = self._estimate_put_premium_rate(var=var, otm_pct=otm_pct)
        estimated_cost = target_notional * premium_rate
        max_cost = portfolio_value * self.config.max_hedge_cost_pct

        if estimated_cost > max_cost and premium_rate > self._EPSILON:
            target_notional = max_cost / premium_rate
            estimated_cost = max_cost

        contracts = self._safe_divide(target_notional, portfolio_value * strike_pct)
        return {
            "instrument": "puts",
            "target_notional_usd": float(target_notional),
            "estimated_cost_usd": float(estimated_cost),
            "strike_pct": float(strike_pct),
            "contracts": float(max(0.0, contracts)),
            "hedge_ratio": float(hedge_ratio),
            "roll_tenor_days": 30,
            "premium_rate": float(premium_rate),
        }

    def size_vix_hedge(self, portfolio_value: float, vix_level: float) -> dict:
        """Size a VIX call hedge allocation."""
        portfolio_value = max(float(portfolio_value), 0.0)
        vix_level = max(float(vix_level), 0.0)

        vix_stress = min(2.0, max(0.25, vix_level / 20.0 if vix_level > 0 else 0.25))
        hedge_ratio = self._dynamic_hedge_ratio(vix_level=vix_level)
        target_notional = portfolio_value * self.config.hedge_allocation_pct * 0.5 * hedge_ratio
        premium_rate = min(0.12, 0.025 + 0.0025 * max(vix_level - 15.0, 0.0))
        estimated_cost = target_notional * premium_rate
        max_cost = portfolio_value * self.config.max_hedge_cost_pct

        if estimated_cost > max_cost and premium_rate > self._EPSILON:
            target_notional = max_cost / premium_rate
            estimated_cost = max_cost

        contracts = self._safe_divide(target_notional, max(vix_level * 100.0, 100.0))
        return {
            "instrument": "vix_calls",
            "target_notional_usd": float(target_notional),
            "estimated_cost_usd": float(estimated_cost),
            "contracts": float(max(0.0, contracts)),
            "hedge_ratio": float(hedge_ratio),
            "vol_multiplier": float(vix_stress),
            "premium_rate": float(premium_rate),
        }

    def calculate_cppi_floor(self, portfolio_value: float, floor_pct: float) -> float:
        """Calculate the CPPI floor value."""
        portfolio_value = max(float(portfolio_value), 0.0)
        floor_pct = min(max(float(floor_pct), 0.0), 1.0)
        return portfolio_value * floor_pct

    def rebalance_hedges(
        self,
        current_positions: Any,
        market_data: Mapping[str, Any],
    ) -> List[HedgeOrder]:
        """Rebalance tail hedges using portfolio, options, vol, and risk inputs."""
        positions = self._coerce_positions(current_positions)
        portfolio_value = self._resolve_portfolio_value(positions, market_data)
        historical_returns = self._extract_returns(market_data)
        vix_level = self._resolve_vix_level(market_data)
        current_hedges = self._extract_current_hedges(positions)

        var_99 = self.calculate_var(historical_returns, confidence=0.99)
        cvar_99 = self.calculate_cvar(historical_returns, confidence=0.99)
        hedge_ratio = self._dynamic_hedge_ratio(var=var_99, cvar=cvar_99, vix_level=vix_level)

        self._last_metrics = {
            "portfolio_value": portfolio_value,
            "var_99": var_99,
            "cvar_99": cvar_99,
            "vix_level": vix_level,
            "hedge_ratio": hedge_ratio,
        }

        candidate_orders: List[HedgeOrder] = []

        if "puts" in self.config.hedge_instruments:
            put_spec = self.size_put_hedge(portfolio_value, var_99, self.config.otm_percentage)
            put_order = self._build_put_order(put_spec, current_hedges, market_data)
            if put_order is not None:
                candidate_orders.append(put_order)

        if "vix_calls" in self.config.hedge_instruments:
            vix_spec = self.size_vix_hedge(portfolio_value, vix_level)
            vix_order = self._build_vix_order(vix_spec, current_hedges, market_data)
            if vix_order is not None:
                candidate_orders.append(vix_order)

        if "cppi" in self.config.hedge_instruments:
            cppi_order = self._build_cppi_order(
                portfolio_value=portfolio_value,
                hedge_ratio=hedge_ratio,
                current_hedges=current_hedges,
                market_data=market_data,
            )
            if cppi_order is not None:
                candidate_orders.append(cppi_order)

        optimized_orders = self._optimize_hedge_costs(candidate_orders, portfolio_value)

        self._last_orders = optimized_orders
        self._last_rebalance_at = datetime.now(timezone.utc)
        self._last_report = self.generate_hedge_report()

        logger.info(
            "Tail hedge rebalance completed with %d order(s); portfolio=%.2f var99=%.4f cvar99=%.4f vix=%.2f",
            len(optimized_orders),
            portfolio_value,
            var_99,
            cvar_99,
            vix_level,
        )
        return optimized_orders

    def calculate_hedge_effectiveness(
        self,
        returns: np.ndarray,
        hedge_returns: np.ndarray,
    ) -> float:
        """Measure variance reduction from applying a hedge overlay."""
        base = self._normalize_returns(returns)
        hedge = self._normalize_returns(hedge_returns)
        n_obs = min(base.size, hedge.size)
        if n_obs < self._MIN_OBSERVATIONS:
            return 0.0

        base = base[-n_obs:]
        hedge = hedge[-n_obs:]
        hedged = base + hedge

        base_var = float(np.var(base))
        hedged_var = float(np.var(hedged))
        if base_var <= self._EPSILON:
            return 0.0

        effectiveness = 1.0 - (hedged_var / base_var)
        return float(max(-1.0, min(1.0, effectiveness)))

    def generate_hedge_report(self) -> dict:
        """Generate a report covering latest hedge state and integration status."""
        total_cost = sum(order.estimated_cost_usd for order in self._last_orders)
        total_notional = sum(order.notional_usd for order in self._last_orders)
        portfolio_value = max(self._last_metrics.get("portfolio_value", 0.0), 0.0)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "hedge_allocation_pct": self.config.hedge_allocation_pct,
                "otm_percentage": self.config.otm_percentage,
                "hedge_instruments": list(self.config.hedge_instruments),
                "rebalance_frequency": self.config.rebalance_frequency,
                "max_hedge_cost_pct": self.config.max_hedge_cost_pct,
            },
            "latest_metrics": dict(self._last_metrics),
            "hedges": [order.to_dict() for order in self._last_orders],
            "summary": {
                "hedge_count": len(self._last_orders),
                "total_notional_usd": float(total_notional),
                "total_estimated_cost_usd": float(total_cost),
                "cost_pct_of_portfolio": self._safe_divide(total_cost, portfolio_value),
                "rebalance_frequency": self.config.rebalance_frequency,
                "last_rebalance_at": self._last_rebalance_at.isoformat() if self._last_rebalance_at else None,
            },
            "integrations": {
                "portfolio": self.portfolio is not None,
                "options_data_feed": self.options_data_feed is not None,
                "volatility_data_feed": self.volatility_data_feed is not None,
                "risk_manager": self.risk_manager is not None,
            },
        }

    def _normalize_returns(self, returns: np.ndarray) -> np.ndarray:
        arr = np.asarray(returns, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.array([], dtype=float)
        return arr[np.isfinite(arr)]

    def _dynamic_hedge_ratio(
        self,
        var: float = 0.0,
        cvar: float = 0.0,
        vix_level: float = 0.0,
    ) -> float:
        var_signal = min(1.5, max(0.0, float(var) / 0.02))
        cvar_signal = min(1.5, max(0.0, float(cvar) / 0.03))
        vix_signal = min(1.5, max(0.0, (float(vix_level) - 15.0) / 15.0))
        ratio = 0.35 + 0.35 * var_signal + 0.20 * cvar_signal + 0.10 * vix_signal
        return float(max(0.25, min(1.50, ratio)))

    def _estimate_put_premium_rate(self, var: float, otm_pct: float) -> float:
        base_premium = 0.01 + 0.25 * max(var, 0.0)
        otm_discount = max(0.35, 1.0 - max(otm_pct, 0.0))
        return float(min(0.08, max(0.005, base_premium * otm_discount)))

    def _resolve_portfolio_value(
        self,
        positions: Sequence[Mapping[str, Any]],
        market_data: Mapping[str, Any],
    ) -> float:
        if "portfolio_value" in market_data:
            return max(float(market_data["portfolio_value"]), 0.0)

        if self.portfolio is not None:
            get_total_value = getattr(self.portfolio, "get_total_value", None)
            if callable(get_total_value):
                try:
                    return max(float(get_total_value()), 0.0)
                except Exception as exc:
                    logger.debug("Portfolio value lookup failed: %s", exc)

        total_value = 0.0
        for position in positions:
            quantity = abs(float(position.get("quantity", 0.0) or 0.0))
            price = float(position.get("mark_price", position.get("price", 0.0)) or 0.0)
            total_value += quantity * price
        return total_value

    def _extract_returns(self, market_data: Mapping[str, Any]) -> np.ndarray:
        returns = market_data.get("historical_returns")
        if returns is not None:
            return self._normalize_returns(np.asarray(returns, dtype=float))

        if self.risk_manager is not None:
            metrics = self._safe_call(self.risk_manager, "get_risk_metrics")
            if isinstance(metrics, Mapping):
                history = metrics.get("returns_history")
                if history is not None:
                    return self._normalize_returns(np.asarray(history, dtype=float))
            history = getattr(self.risk_manager, "returns_history", None)
            if history is not None:
                return self._normalize_returns(np.asarray(list(history), dtype=float))

        return np.array([], dtype=float)

    def _resolve_vix_level(self, market_data: Mapping[str, Any]) -> float:
        if "vix_level" in market_data:
            return max(float(market_data["vix_level"]), 0.0)
        if "volatility_index" in market_data:
            return max(float(market_data["volatility_index"]), 0.0)
        if self.volatility_data_feed is not None:
            try:
                return max(float(self.volatility_data_feed.get_vix_level()), 0.0)
            except Exception as exc:
                logger.debug("Volatility data lookup failed: %s", exc)
        return 20.0

    def _build_put_order(
        self,
        put_spec: Mapping[str, Any],
        current_hedges: Mapping[str, float],
        market_data: Mapping[str, Any],
    ) -> Optional[HedgeOrder]:
        target_notional = float(put_spec.get("target_notional_usd", 0.0))
        if target_notional <= 0:
            return None

        option_symbol = self._select_put_symbol(market_data, put_spec)
        current_notional = current_hedges.get("puts", 0.0)
        delta_notional = target_notional - current_notional
        if abs(delta_notional) < max(50.0, target_notional * 0.05):
            return None

        side = "BUY" if delta_notional > 0 else "SELL"
        unit_notional = max(target_notional / max(float(put_spec.get("contracts", 0.0)), 1.0), 1.0)
        quantity = abs(delta_notional) / unit_notional
        estimated_cost = abs(delta_notional) * float(put_spec.get("premium_rate", 0.0))

        return HedgeOrder(
            instrument_type="puts",
            symbol=option_symbol,
            side=side,
            quantity=float(quantity),
            notional_usd=abs(delta_notional),
            estimated_cost_usd=float(estimated_cost),
            rationale="Roll deep OTM puts to maintain target crash protection notional.",
            metadata={
                "strike_pct": put_spec.get("strike_pct"),
                "roll_tenor_days": put_spec.get("roll_tenor_days"),
                "target_notional_usd": target_notional,
            },
        )

    def _build_vix_order(
        self,
        vix_spec: Mapping[str, Any],
        current_hedges: Mapping[str, float],
        market_data: Mapping[str, Any],
    ) -> Optional[HedgeOrder]:
        target_notional = float(vix_spec.get("target_notional_usd", 0.0))
        if target_notional <= 0:
            return None

        symbol = str(market_data.get("vix_call_symbol", "VIX-CALL-30D"))
        current_notional = current_hedges.get("vix_calls", 0.0)
        delta_notional = target_notional - current_notional
        if abs(delta_notional) < max(50.0, target_notional * 0.10):
            return None

        side = "BUY" if delta_notional > 0 else "SELL"
        unit_notional = max(target_notional / max(float(vix_spec.get("contracts", 0.0)), 1.0), 1.0)
        quantity = abs(delta_notional) / unit_notional
        estimated_cost = abs(delta_notional) * float(vix_spec.get("premium_rate", 0.0))

        return HedgeOrder(
            instrument_type="vix_calls",
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            notional_usd=abs(delta_notional),
            estimated_cost_usd=float(estimated_cost),
            rationale="Adjust VIX call overlay to match volatility-spike protection target.",
            metadata={
                "vol_multiplier": vix_spec.get("vol_multiplier"),
                "target_notional_usd": target_notional,
            },
        )

    def _build_cppi_order(
        self,
        portfolio_value: float,
        hedge_ratio: float,
        current_hedges: Mapping[str, float],
        market_data: Mapping[str, Any],
    ) -> Optional[HedgeOrder]:
        floor_pct = float(market_data.get("cppi_floor_pct", 1.0 - self.config.hedge_allocation_pct * 5.0))
        floor_value = self.calculate_cppi_floor(portfolio_value, floor_pct)
        cushion = max(0.0, portfolio_value - floor_value)
        multiplier = min(5.0, max(1.0, 2.0 + hedge_ratio))
        risky_allocation = min(portfolio_value, cushion * multiplier)
        target_de_risking = max(0.0, portfolio_value - risky_allocation)
        current_notional = current_hedges.get("cppi", 0.0)
        delta_notional = target_de_risking - current_notional

        if abs(delta_notional) < max(50.0, portfolio_value * 0.01):
            return None

        symbol = str(market_data.get("cppi_cash_symbol", "USD-CASH"))
        side = "BUY" if delta_notional > 0 else "SELL"
        return HedgeOrder(
            instrument_type="cppi",
            symbol=symbol,
            side=side,
            quantity=abs(delta_notional),
            notional_usd=abs(delta_notional),
            estimated_cost_usd=0.0,
            rationale="Apply CPPI overlay by shifting exposure between risky assets and cash floor.",
            metadata={
                "floor_pct": floor_pct,
                "floor_value": floor_value,
                "target_de_risking_usd": target_de_risking,
            },
        )

    def _optimize_hedge_costs(
        self,
        orders: Sequence[HedgeOrder],
        portfolio_value: float,
    ) -> List[HedgeOrder]:
        max_total_cost = max(0.0, portfolio_value * self.config.max_hedge_cost_pct)
        if max_total_cost <= 0:
            return []

        def efficiency(order: HedgeOrder) -> float:
            cost = max(order.estimated_cost_usd, self._EPSILON)
            weight = {"puts": 1.35, "vix_calls": 1.15, "cppi": 0.90}.get(order.instrument_type, 1.0)
            return weight * order.notional_usd / cost

        optimized: List[HedgeOrder] = []
        used_cost = 0.0
        for order in sorted(orders, key=efficiency, reverse=True):
            if used_cost + order.estimated_cost_usd <= max_total_cost + self._EPSILON:
                optimized.append(order)
                used_cost += order.estimated_cost_usd
                continue

            remaining_budget = max_total_cost - used_cost
            if remaining_budget <= self._EPSILON or order.estimated_cost_usd <= self._EPSILON:
                if order.estimated_cost_usd <= self._EPSILON:
                    optimized.append(order)
                continue

            scale = remaining_budget / order.estimated_cost_usd
            if scale < 0.10:
                continue

            optimized.append(
                HedgeOrder(
                    instrument_type=order.instrument_type,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity * scale,
                    notional_usd=order.notional_usd * scale,
                    estimated_cost_usd=order.estimated_cost_usd * scale,
                    rationale=f"{order.rationale} Scaled for hedge budget.",
                    metadata=dict(order.metadata),
                )
            )
            used_cost = max_total_cost

        return optimized

    def _coerce_positions(self, current_positions: Any) -> List[Mapping[str, Any]]:
        if current_positions is None and self.portfolio is not None:
            current_positions = self._safe_call(self.portfolio, "get_positions")

        if current_positions is None:
            return []

        if isinstance(current_positions, Mapping):
            if all(isinstance(v, Mapping) for v in current_positions.values()):
                return [dict(v) for v in current_positions.values()]
            return [dict(current_positions)]

        if isinstance(current_positions, Iterable) and not isinstance(current_positions, (str, bytes)):
            positions: List[Mapping[str, Any]] = []
            for item in current_positions:
                if isinstance(item, Mapping):
                    positions.append(dict(item))
                else:
                    positions.append(vars(item))
            return positions

        return [vars(current_positions)]

    def _extract_current_hedges(self, positions: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        hedge_notional = {"puts": 0.0, "vix_calls": 0.0, "cppi": 0.0}
        for position in positions:
            hedge_type = str(position.get("instrument_type", "")).lower()
            symbol = str(position.get("symbol", "")).upper()
            quantity = abs(float(position.get("quantity", 0.0) or 0.0))
            price = abs(float(position.get("mark_price", position.get("price", 0.0)) or 0.0))
            notional = float(position.get("notional_usd", 0.0) or quantity * price)

            if hedge_type in hedge_notional:
                hedge_notional[hedge_type] += notional
            elif "PUT" in symbol:
                hedge_notional["puts"] += notional
            elif "VIX" in symbol:
                hedge_notional["vix_calls"] += notional
            elif hedge_type == "cash":
                hedge_notional["cppi"] += notional
        return hedge_notional

    def _select_put_symbol(
        self,
        market_data: Mapping[str, Any],
        put_spec: Mapping[str, Any],
    ) -> str:
        chain = market_data.get("options_chain")
        if chain is None and self.options_data_feed is not None:
            chain = self._safe_call(self.options_data_feed, "get_option_chain", market_data.get("underlying", "BTC"))

        underlying = str(market_data.get("underlying", "BTC"))
        strike_pct = float(put_spec.get("strike_pct", 0.85))
        spot = float(market_data.get("spot_price", market_data.get("underlying_price", 0.0)) or 0.0)
        strike_target = spot * strike_pct if spot > 0 else None

        if isinstance(chain, Sequence) and not isinstance(chain, (str, bytes)):
            best_symbol: Optional[str] = None
            best_distance = math.inf
            for contract in chain:
                if not isinstance(contract, Mapping):
                    continue
                option_type = str(contract.get("type", contract.get("option_type", ""))).lower()
                if option_type not in {"put", "puts", "p"}:
                    continue
                strike = float(contract.get("strike", 0.0) or 0.0)
                if strike_target is None:
                    return str(contract.get("symbol", f"{underlying}-PUT"))
                distance = abs(strike - strike_target)
                if distance < best_distance:
                    best_distance = distance
                    best_symbol = str(contract.get("symbol", f"{underlying}-PUT"))
            if best_symbol:
                return best_symbol

        if strike_target is not None and strike_target > 0:
            return f"{underlying}-{int(round(strike_target))}-PUT-30D"
        return f"{underlying}-PUT-30D"

    def _safe_call(self, obj: Any, method_name: str, *args: Any) -> Any:
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return method(*args)
            except Exception as exc:
                logger.debug("%s.%s failed: %s", type(obj).__name__, method_name, exc)
        return None

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        if abs(denominator) <= self._EPSILON:
            return 0.0
        return float(numerator / denominator)


__all__ = ["HedgeOrder", "TailHedgeConfig", "TailRiskHedger"]
