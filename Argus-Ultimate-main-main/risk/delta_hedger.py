"""Delta hedging engine for options portfolios.

This module provides Black-Scholes Greeks, portfolio aggregation, delta-neutral
hedge generation, gamma scalping support, vega monitoring, and a lightweight
delta-neutral backtest interface. Integrations are dependency-injected to keep
the hedger usable across paper, live, and research workflows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from scipy.stats import norm

logger = logging.getLogger(__name__)


class SupportsOptionPositions(Protocol):
    """Protocol for options-position integrations."""

    def get_option_positions(self) -> Any:
        """Return current option positions."""


class SupportsUnderlyingPrices(Protocol):
    """Protocol for underlying spot-price integrations."""

    def get_price(self, symbol: str) -> float:
        """Return current underlying price for a symbol."""


class SupportsVolatilitySurface(Protocol):
    """Protocol for implied-volatility integrations."""

    def get_implied_volatility(
        self,
        symbol: str,
        strike: float,
        expiry: datetime,
        option_type: str,
        underlying_price: float,
    ) -> float:
        """Return implied volatility for an option definition."""


class SupportsRiskManager(Protocol):
    """Protocol for risk approval integrations."""

    def approve_hedge(self, order: Mapping[str, Any]) -> bool:
        """Return whether the hedge order is acceptable."""


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    implied_volatility: float

    def scale(self, quantity: float) -> "Greeks":
        return Greeks(
            delta=self.delta * quantity,
            gamma=self.gamma * quantity,
            vega=self.vega * quantity,
            theta=self.theta * quantity,
            rho=self.rho * quantity,
            implied_volatility=self.implied_volatility,
        )

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
            implied_volatility=(self.implied_volatility + other.implied_volatility) / 2.0,
        )


@dataclass
class OptionPosition:
    symbol: str
    option_type: str
    strike: float
    expiry: datetime
    quantity: int
    current_price: float
    underlying_price: float
    greeks: Greeks


@dataclass
class HedgeOrder:
    symbol: str
    side: str
    quantity: float
    reference_price: float
    estimated_cost_usd: float
    target_delta: float
    post_trade_delta: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "reference_price": float(self.reference_price),
            "estimated_cost_usd": float(self.estimated_cost_usd),
            "target_delta": float(self.target_delta),
            "post_trade_delta": float(self.post_trade_delta),
            "rationale": self.rationale,
            "metadata": dict(self.metadata),
            "generated_at": self.generated_at.isoformat(),
        }


class DeltaHedger:
    """Options delta hedging engine with optional integration hooks."""

    _MIN_T = 1e-6
    _MIN_SIGMA = 1e-6
    _MIN_PRICE = 1e-9

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        transaction_cost_bps: float = 5.0,
        contract_multiplier: float = 1.0,
        dynamic_threshold_factor: float = 0.25,
        underlying_symbol: str = "UNDERLYING",
        option_positions_provider: Optional[SupportsOptionPositions] = None,
        underlying_price_provider: Optional[SupportsUnderlyingPrices] = None,
        volatility_surface: Optional[SupportsVolatilitySurface] = None,
        risk_manager: Optional[SupportsRiskManager] = None,
        transaction_cost_model: Optional[Any] = None,
    ) -> None:
        self.risk_free_rate = float(risk_free_rate)
        self.transaction_cost_bps = max(0.0, float(transaction_cost_bps))
        self.contract_multiplier = max(1e-9, float(contract_multiplier))
        self.dynamic_threshold_factor = max(0.0, float(dynamic_threshold_factor))
        self.underlying_symbol = underlying_symbol
        self.option_positions_provider = option_positions_provider
        self.underlying_price_provider = underlying_price_provider
        self.volatility_surface = volatility_surface
        self.risk_manager = risk_manager
        self.transaction_cost_model = transaction_cost_model

        self._last_portfolio_greeks: Optional[Greeks] = None
        self._last_rebalance_threshold: float = 0.1

    def calculate_black_scholes_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
    ) -> Greeks:
        """Calculate Black-Scholes Greeks for a European option."""
        option_type_normalized = str(option_type).lower().strip()
        if option_type_normalized not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        S = max(float(S), self._MIN_PRICE)
        K = max(float(K), self._MIN_PRICE)
        T = max(float(T), self._MIN_T)
        sigma = max(float(sigma), self._MIN_SIGMA)
        r = float(r)

        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        pdf_d1 = norm.pdf(d1)

        if option_type_normalized == "call":
            delta = float(norm.cdf(d1))
            theta = (
                -(S * pdf_d1 * sigma) / (2.0 * sqrt_t)
                - r * K * math.exp(-r * T) * float(norm.cdf(d2))
            ) / 365.0
            rho = K * T * math.exp(-r * T) * float(norm.cdf(d2)) / 100.0
        else:
            delta = float(norm.cdf(d1) - 1.0)
            theta = (
                -(S * pdf_d1 * sigma) / (2.0 * sqrt_t)
                + r * K * math.exp(-r * T) * float(norm.cdf(-d2))
            ) / 365.0
            rho = -K * T * math.exp(-r * T) * float(norm.cdf(-d2)) / 100.0

        gamma = pdf_d1 / (S * sigma * sqrt_t)
        vega = S * pdf_d1 * sqrt_t / 100.0
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
            implied_volatility=float(sigma),
        )

    def calculate_portfolio_greeks(self, positions: List[OptionPosition]) -> Greeks:
        """Aggregate portfolio Greeks across option positions."""
        total = self._zero_greeks()
        weighted_iv = 0.0
        total_abs_qty = 0.0

        for position in positions:
            quantity = float(position.quantity) * self.contract_multiplier
            position_greeks = position.greeks.scale(quantity)
            total = total + position_greeks
            total_abs_qty += abs(quantity)
            weighted_iv += abs(quantity) * float(position.greeks.implied_volatility)

        if total_abs_qty > 0:
            total.implied_volatility = weighted_iv / total_abs_qty

        self._last_portfolio_greeks = total
        return total

    def calculate_hedge_ratio(self, portfolio_delta: float, target_delta: float) -> float:
        """Return the underlying quantity needed to move toward target delta."""
        return float(target_delta) - float(portfolio_delta)

    def generate_hedge_trades(
        self,
        positions: List[OptionPosition],
        target_delta: float = 0.0,
    ) -> List[HedgeOrder]:
        """Generate cost-aware hedge orders for the underlying asset."""
        if not positions:
            return []

        portfolio_greeks = self.calculate_portfolio_greeks(positions)
        current_delta = portfolio_greeks.delta
        rebalance_threshold = self._calculate_dynamic_threshold(
            base_threshold=0.1,
            greeks=portfolio_greeks,
        )
        self._last_rebalance_threshold = rebalance_threshold

        if not self.should_rebalance(current_delta - target_delta, rebalance_threshold):
            logger.debug(
                "Delta hedge skipped: current_delta=%.6f target_delta=%.6f threshold=%.6f",
                current_delta,
                target_delta,
                rebalance_threshold,
            )
            return []

        underlying_symbol = self._resolve_underlying_symbol(positions)
        reference_price = self._resolve_underlying_price(positions, underlying_symbol)
        hedge_quantity = self.calculate_hedge_ratio(current_delta, target_delta)
        if abs(hedge_quantity) <= self._MIN_PRICE:
            return []

        side = "buy" if hedge_quantity > 0 else "sell"
        quantity = abs(hedge_quantity)
        estimated_cost_usd = self._estimate_transaction_cost(
            symbol=underlying_symbol,
            quantity=quantity,
            side=side,
            current_price=reference_price,
        )
        post_trade_delta = current_delta + hedge_quantity

        vega_flag = abs(portfolio_greeks.vega) > max(1.0, abs(current_delta))
        gamma_scalp_bias = self._gamma_scalping_bias(portfolio_greeks)
        rationale = (
            f"Rebalance portfolio delta from {current_delta:.4f} to {target_delta:.4f}; "
            f"dynamic threshold={rebalance_threshold:.4f}"
        )

        order = HedgeOrder(
            symbol=underlying_symbol,
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            estimated_cost_usd=estimated_cost_usd,
            target_delta=float(target_delta),
            post_trade_delta=float(post_trade_delta),
            rationale=rationale,
            metadata={
                "portfolio_delta": float(current_delta),
                "portfolio_gamma": float(portfolio_greeks.gamma),
                "portfolio_vega": float(portfolio_greeks.vega),
                "portfolio_theta": float(portfolio_greeks.theta),
                "implied_volatility": float(portfolio_greeks.implied_volatility),
                "dynamic_threshold": float(rebalance_threshold),
                "gamma_scalping_bias": gamma_scalp_bias,
                "vega_risk_warning": bool(vega_flag),
                "transaction_cost_bps": float(self.transaction_cost_bps),
            },
        )

        if self._risk_manager_blocks(order):
            logger.warning("Risk manager rejected hedge order for %s", order.symbol)
            return []

        logger.info(
            "Generated hedge order %s %.6f %s at %.4f (cost %.4f USD)",
            order.side,
            order.quantity,
            order.symbol,
            order.reference_price,
            order.estimated_cost_usd,
        )
        return [order]

    def should_rebalance(self, current_delta: float, threshold: float = 0.1) -> bool:
        """Return whether current delta breach warrants rebalancing."""
        effective_threshold = float(threshold)
        if math.isclose(effective_threshold, 0.1, rel_tol=0.0, abs_tol=1e-12):
            effective_threshold = self._calculate_dynamic_threshold(
                base_threshold=effective_threshold,
                greeks=self._last_portfolio_greeks,
            )
        self._last_rebalance_threshold = effective_threshold
        return abs(float(current_delta)) >= effective_threshold

    def calculate_gamma_pnl(
        self,
        positions: List[OptionPosition],
        price_change: float,
    ) -> float:
        """Approximate gamma PnL for an absolute move in the underlying."""
        portfolio_greeks = self.calculate_portfolio_greeks(positions)
        return 0.5 * float(portfolio_greeks.gamma) * float(price_change) ** 2

    def run_delta_neutral_backtest(self, historical_data: Any) -> dict:
        """Run a simple delta-neutral hedging backtest over historical snapshots."""
        snapshots = self._normalize_historical_data(historical_data)
        if not snapshots:
            return {
                "periods": 0,
                "hedge_trades": 0,
                "transaction_costs_usd": 0.0,
                "gamma_scalping_pnl": 0.0,
                "net_pnl": 0.0,
                "average_abs_delta": 0.0,
                "max_abs_delta": 0.0,
                "vega_breach_count": 0,
                "rebalance_rate": 0.0,
            }

        total_costs = 0.0
        gamma_pnl_total = 0.0
        hedge_trade_count = 0
        abs_deltas: List[float] = []
        vega_breach_count = 0

        for snapshot in snapshots:
            positions = self._positions_from_snapshot(snapshot)
            if not positions:
                continue

            portfolio_greeks = self.calculate_portfolio_greeks(positions)
            abs_deltas.append(abs(portfolio_greeks.delta))

            if abs(portfolio_greeks.vega) > snapshot.get("vega_limit", float("inf")):
                vega_breach_count += 1

            price_change = float(snapshot.get("price_change", 0.0))
            gamma_pnl_total += self.calculate_gamma_pnl(positions, price_change)

            hedge_orders = self.generate_hedge_trades(
                positions=positions,
                target_delta=float(snapshot.get("target_delta", 0.0)),
            )
            hedge_trade_count += len(hedge_orders)
            total_costs += sum(order.estimated_cost_usd for order in hedge_orders)

        periods = len(snapshots)
        average_abs_delta = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
        max_abs_delta = max(abs_deltas) if abs_deltas else 0.0
        net_pnl = gamma_pnl_total - total_costs

        return {
            "periods": periods,
            "hedge_trades": hedge_trade_count,
            "transaction_costs_usd": float(total_costs),
            "gamma_scalping_pnl": float(gamma_pnl_total),
            "net_pnl": float(net_pnl),
            "average_abs_delta": float(average_abs_delta),
            "max_abs_delta": float(max_abs_delta),
            "vega_breach_count": int(vega_breach_count),
            "rebalance_rate": float(hedge_trade_count / periods) if periods > 0 else 0.0,
        }

    def get_live_positions(self) -> List[OptionPosition]:
        """Fetch live option positions via the injected provider."""
        provider = self.option_positions_provider
        if provider is None:
            return []
        try:
            raw_positions = provider.get_option_positions()
        except Exception:
            logger.exception("Failed to fetch option positions from provider")
            return []
        return self._coerce_option_positions(raw_positions)

    def enrich_positions_from_vol_surface(
        self,
        positions: Sequence[OptionPosition],
    ) -> List[OptionPosition]:
        """Refresh position Greeks using the injected volatility surface."""
        if self.volatility_surface is None:
            return list(positions)

        enriched: List[OptionPosition] = []
        for position in positions:
            implied_vol = self._resolve_implied_volatility(position)
            time_to_expiry = self._time_to_expiry(position.expiry)
            greeks = self.calculate_black_scholes_greeks(
                S=position.underlying_price,
                K=position.strike,
                T=time_to_expiry,
                r=self.risk_free_rate,
                sigma=implied_vol,
                option_type=position.option_type,
            )
            enriched.append(
                OptionPosition(
                    symbol=position.symbol,
                    option_type=position.option_type,
                    strike=position.strike,
                    expiry=position.expiry,
                    quantity=position.quantity,
                    current_price=position.current_price,
                    underlying_price=position.underlying_price,
                    greeks=greeks,
                )
            )
        return enriched

    def _coerce_option_positions(self, raw_positions: Any) -> List[OptionPosition]:
        if isinstance(raw_positions, list) and all(
            isinstance(item, OptionPosition) for item in raw_positions
        ):
            return list(raw_positions)

        positions: List[OptionPosition] = []
        for item in self._iterable(raw_positions):
            if isinstance(item, OptionPosition):
                positions.append(item)
                continue
            if not isinstance(item, Mapping):
                continue

            expiry = self._parse_datetime(item.get("expiry"))
            option_type = str(item.get("option_type", "call")).lower()
            underlying_price = float(item.get("underlying_price", 0.0))
            strike = float(item.get("strike", 0.0))
            implied_vol = float(item.get("implied_volatility", item.get("iv", 0.0)) or 0.0)
            greeks_data = item.get("greeks") if isinstance(item.get("greeks"), Mapping) else {}

            if expiry is None:
                continue

            if greeks_data:
                greeks = Greeks(
                    delta=float(greeks_data.get("delta", 0.0)),
                    gamma=float(greeks_data.get("gamma", 0.0)),
                    vega=float(greeks_data.get("vega", 0.0)),
                    theta=float(greeks_data.get("theta", 0.0)),
                    rho=float(greeks_data.get("rho", 0.0)),
                    implied_volatility=float(
                        greeks_data.get("implied_volatility", implied_vol)
                    ),
                )
            else:
                sigma = implied_vol if implied_vol > 0 else self._resolve_default_iv(underlying_price, strike)
                greeks = self.calculate_black_scholes_greeks(
                    S=underlying_price,
                    K=strike,
                    T=self._time_to_expiry(expiry),
                    r=self.risk_free_rate,
                    sigma=sigma,
                    option_type=option_type,
                )

            positions.append(
                OptionPosition(
                    symbol=str(item.get("symbol", "")),
                    option_type=option_type,
                    strike=strike,
                    expiry=expiry,
                    quantity=int(item.get("quantity", 0)),
                    current_price=float(item.get("current_price", 0.0)),
                    underlying_price=underlying_price,
                    greeks=greeks,
                )
            )
        return positions

    def _positions_from_snapshot(self, snapshot: Mapping[str, Any]) -> List[OptionPosition]:
        if isinstance(snapshot.get("positions"), list):
            return self.enrich_positions_from_vol_surface(
                self._coerce_option_positions(snapshot.get("positions"))
            )

        live_positions = self.get_live_positions()
        return self.enrich_positions_from_vol_surface(live_positions)

    def _normalize_historical_data(self, historical_data: Any) -> List[Mapping[str, Any]]:
        if isinstance(historical_data, Mapping):
            if isinstance(historical_data.get("snapshots"), list):
                return [item for item in historical_data["snapshots"] if isinstance(item, Mapping)]
            return [historical_data]
        if isinstance(historical_data, list):
            return [item for item in historical_data if isinstance(item, Mapping)]
        return []

    def _resolve_underlying_symbol(self, positions: Sequence[OptionPosition]) -> str:
        symbols = [self._extract_underlying_from_symbol(position.symbol) for position in positions]
        for symbol in symbols:
            if symbol:
                return symbol
        return self.underlying_symbol

    def _resolve_underlying_price(
        self,
        positions: Sequence[OptionPosition],
        underlying_symbol: str,
    ) -> float:
        if self.underlying_price_provider is not None:
            try:
                price = float(self.underlying_price_provider.get_price(underlying_symbol))
                if price > 0:
                    return price
            except Exception:
                logger.debug("Underlying price provider failed for %s", underlying_symbol, exc_info=True)

        for position in positions:
            if position.underlying_price > 0:
                return float(position.underlying_price)
        return 0.0

    def _estimate_transaction_cost(
        self,
        symbol: str,
        quantity: float,
        side: str,
        current_price: float,
    ) -> float:
        if self.transaction_cost_model is not None:
            try:
                estimate = self.transaction_cost_model.estimate_cost(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    current_price=current_price,
                )
                return float(getattr(estimate, "total_usd", 0.0))
            except Exception:
                logger.debug("TransactionCostModel estimate failed for %s", symbol, exc_info=True)

        notional = abs(float(quantity) * float(current_price))
        return notional * (self.transaction_cost_bps / 10000.0)

    def _risk_manager_blocks(self, order: HedgeOrder) -> bool:
        if self.risk_manager is None:
            return False

        payload = order.to_dict()

        approve_hedge = getattr(self.risk_manager, "approve_hedge", None)
        if callable(approve_hedge):
            try:
                return not bool(approve_hedge(payload))
            except Exception:
                logger.debug("Risk manager approve_hedge failed", exc_info=True)

        validate_order = getattr(self.risk_manager, "validate_order", None)
        if callable(validate_order):
            try:
                return not bool(validate_order(payload))
            except Exception:
                logger.debug("Risk manager validate_order failed", exc_info=True)

        return False

    def _resolve_implied_volatility(self, position: OptionPosition) -> float:
        if self.volatility_surface is not None:
            try:
                iv = float(
                    self.volatility_surface.get_implied_volatility(
                        symbol=position.symbol,
                        strike=position.strike,
                        expiry=position.expiry,
                        option_type=position.option_type,
                        underlying_price=position.underlying_price,
                    )
                )
                if iv > 0:
                    return iv
            except Exception:
                logger.debug("Vol surface lookup failed for %s", position.symbol, exc_info=True)
        if position.greeks.implied_volatility > 0:
            return float(position.greeks.implied_volatility)
        return self._resolve_default_iv(position.underlying_price, position.strike)

    def _calculate_dynamic_threshold(
        self,
        base_threshold: float,
        greeks: Optional[Greeks],
    ) -> float:
        threshold = max(float(base_threshold), 1e-6)
        if greeks is None:
            return threshold

        gamma_adjustment = 1.0 / (1.0 + abs(greeks.gamma) * self.dynamic_threshold_factor)
        iv_adjustment = 1.0 / (1.0 + max(greeks.implied_volatility, 0.0) * 0.5)
        vega_adjustment = 1.0 / (1.0 + abs(greeks.vega) * 0.01)
        dynamic_threshold = threshold * gamma_adjustment * iv_adjustment * vega_adjustment
        return max(0.01, dynamic_threshold)

    def _gamma_scalping_bias(self, greeks: Greeks) -> str:
        if greeks.gamma > 0:
            return "positive_gamma_scalping_enabled"
        if greeks.gamma < 0:
            return "negative_gamma_reduce_rehedging"
        return "neutral_gamma"

    def _extract_underlying_from_symbol(self, symbol: str) -> str:
        if not symbol:
            return ""
        if "-" in symbol:
            return symbol.split("-", 1)[0].upper()
        if "_" in symbol:
            return symbol.split("_", 1)[0].upper()
        return symbol.upper()

    def _resolve_default_iv(self, underlying_price: float, strike: float) -> float:
        if underlying_price <= 0 or strike <= 0:
            return 0.5
        moneyness = abs(math.log(max(underlying_price, self._MIN_PRICE) / max(strike, self._MIN_PRICE)))
        return min(2.0, max(0.2, 0.45 + 0.15 * moneyness))

    def _time_to_expiry(self, expiry: datetime) -> float:
        expiry_utc = expiry if expiry.tzinfo is not None else expiry.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = max((expiry_utc - now).total_seconds(), 0.0)
        return max(seconds / 31_536_000.0, self._MIN_T)

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            normalized = value.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalized)
            except ValueError:
                return None
        return None

    def _iterable(self, value: Any) -> Iterable[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return value
        return [value]

    def _zero_greeks(self) -> Greeks:
        return Greeks(
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
            implied_volatility=0.0,
        )


__all__ = [
    "DeltaHedger",
    "Greeks",
    "HedgeOrder",
    "OptionPosition",
    "SupportsOptionPositions",
    "SupportsRiskManager",
    "SupportsUnderlyingPrices",
    "SupportsVolatilitySurface",
]
