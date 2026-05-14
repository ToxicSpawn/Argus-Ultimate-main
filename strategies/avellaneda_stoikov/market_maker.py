"""Core Avellaneda-Stoikov market-making logic."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from .inventory风险管理 import InventoryRiskConfig, InventoryRiskManager
from .volatility_estimator import VolatilityEstimator, VolatilitySnapshot

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MarketMakerConfig:
    symbol: str = "BTCUSDT"
    gamma: float = 0.1
    kappa: float = 1.5
    sigma_floor: float = 1e-4
    horizon_seconds: float = 300.0
    min_spread_bps: float = 4.0
    max_spread_bps: float = 80.0
    base_order_size: float = 0.01
    min_order_size: float = 0.001
    max_order_size: float = 1.0
    inventory_target: float = 0.0
    max_inventory: float = 5.0
    order_size_decay: float = 0.75
    tick_size: float = 0.01
    use_injective_reservation_adjustment: bool = True
    volatility_window: int = 120
    ewma_lambda: float = 0.94
    inventory_soft_limit_fraction: float = 0.7
    max_skew_bps: float = 20.0

    def __post_init__(self) -> None:
        numeric_positive = {
            "gamma": self.gamma,
            "kappa": self.kappa,
            "sigma_floor": self.sigma_floor,
            "horizon_seconds": self.horizon_seconds,
            "base_order_size": self.base_order_size,
            "min_order_size": self.min_order_size,
            "max_order_size": self.max_order_size,
            "max_inventory": self.max_inventory,
            "order_size_decay": self.order_size_decay,
            "tick_size": self.tick_size,
        }
        for name, value in numeric_positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_spread_bps < 0 or self.max_spread_bps <= self.min_spread_bps:
            raise ValueError("spread bounds are invalid")


@dataclass(slots=True)
class MarketSnapshot:
    mid_price: float
    best_bid: float
    best_ask: float
    timestamp: float = field(default_factory=time.time)
    market_depth: float = 1.0
    order_book_imbalance: float = 0.0
    last_trade_price: float = 0.0
    volume: float = 0.0


@dataclass(slots=True)
class OptimalQuote:
    reservation_price: float
    reservation_price_adjustment: float
    optimal_spread: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    volatility: float
    time_remaining: float
    inventory: float
    inventory_penalty: float


class AvellanedaStoikovMarketMaker:
    """Implements optimal spread and inventory-aware quote generation."""

    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.inventory_manager = InventoryRiskManager(
            InventoryRiskConfig(
                gamma=config.gamma,
                max_position=config.max_inventory,
                soft_limit_fraction=config.inventory_soft_limit_fraction,
                inventory_target=config.inventory_target,
                max_skew_bps=config.max_skew_bps,
            )
        )
        self.volatility_estimator = VolatilityEstimator(
            window_size=config.volatility_window,
            ewma_lambda=config.ewma_lambda,
        )
        self._start_time = time.time()

    def elapsed_seconds(self, timestamp: float) -> float:
        return max(0.0, timestamp - self._start_time)

    def time_remaining(self, timestamp: float) -> float:
        remaining = max(0.0, self.config.horizon_seconds - self.elapsed_seconds(timestamp))
        return remaining

    def reservation_price_adjustment(self, volatility: float, timestamp: float) -> float:
        time_remaining = self.time_remaining(timestamp)
        if not self.config.use_injective_reservation_adjustment:
            return 0.0
        gamma = self.config.gamma
        kappa = self.config.kappa
        first_term = math.log(1.0 + gamma / kappa) / (2.0 * gamma)
        second_term = ((volatility ** 2) * self.config.horizon_seconds / (2.0 * gamma)) * (
            1.0 - min(1.0, self.elapsed_seconds(timestamp) / self.config.horizon_seconds)
        )
        return first_term + second_term if time_remaining > 0 else first_term

    def reservation_price(self, market: MarketSnapshot, volatility: float) -> float:
        inventory_adjustment = self.inventory_manager.reservation_price_adjustment(
            volatility=volatility,
            time_remaining=self.time_remaining(market.timestamp),
        )
        injective_adjustment = self.reservation_price_adjustment(volatility, market.timestamp)
        direction = -1.0 if self.inventory_manager.position > self.config.inventory_target else 1.0
        if self.inventory_manager.position == self.config.inventory_target:
            direction = 0.0
        return market.mid_price + inventory_adjustment + direction * injective_adjustment

    def optimal_spread(self, volatility: float, timestamp: float) -> float:
        gamma = self.config.gamma
        kappa = self.config.kappa
        horizon = max(self.time_remaining(timestamp), self.config.tick_size)
        raw_spread = gamma * (volatility ** 2) * horizon + (2.0 / gamma) * math.log(1.0 + gamma / kappa)
        return raw_spread

    def clamp_spread(self, mid_price: float, spread: float, vol_snapshot: VolatilitySnapshot) -> float:
        spread *= vol_snapshot.spread_multiplier
        min_spread = mid_price * self.config.min_spread_bps / 10_000.0
        max_spread = mid_price * self.config.max_spread_bps / 10_000.0
        return max(min_spread, min(max_spread, spread))

    def order_sizes(self) -> tuple[float, float]:
        ratio = abs(self.inventory_manager.inventory_ratio())
        scaling = max(0.0, 1.0 - ratio * self.config.order_size_decay)
        size = max(self.config.min_order_size, min(self.config.max_order_size, self.config.base_order_size * scaling))
        bid_size = self.inventory_manager.clamp_order_size(size, "buy")
        ask_size = self.inventory_manager.clamp_order_size(size, "sell")
        return bid_size, ask_size

    def round_to_tick(self, price: float) -> float:
        tick = self.config.tick_size
        return round(price / tick) * tick

    def update_volatility(self, market: MarketSnapshot) -> VolatilitySnapshot:
        reference_price = market.last_trade_price if market.last_trade_price > 0 else market.mid_price
        return self.volatility_estimator.update(reference_price, market.timestamp)

    def generate_quotes(self, market: MarketSnapshot) -> OptimalQuote:
        if market.mid_price <= 0 or market.best_bid <= 0 or market.best_ask <= 0:
            raise ValueError("market snapshot prices must be positive")

        vol_snapshot = self.update_volatility(market)
        volatility = max(self.config.sigma_floor, vol_snapshot.selected_volatility)
        reservation = self.reservation_price(market, volatility)
        spread = self.clamp_spread(market.mid_price, self.optimal_spread(volatility, market.timestamp), vol_snapshot)
        half_spread = spread / 2.0
        bid_skew, ask_skew = self.inventory_manager.quote_skew(market.mid_price)
        bid_size, ask_size = self.order_sizes()

        bid_price = self.round_to_tick(max(self.config.tick_size, reservation - half_spread + bid_skew))
        ask_price = self.round_to_tick(max(bid_price + self.config.tick_size, reservation + half_spread + ask_skew))

        inventory_penalty = self.inventory_manager.inventory_penalty(volatility, self.time_remaining(market.timestamp))
        return OptimalQuote(
            reservation_price=reservation,
            reservation_price_adjustment=self.reservation_price_adjustment(volatility, market.timestamp),
            optimal_spread=spread,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            volatility=volatility,
            time_remaining=self.time_remaining(market.timestamp),
            inventory=self.inventory_manager.position,
            inventory_penalty=inventory_penalty,
        )
