"""
Market Making Strategy with Inventory Management.

Implements institutional-grade market making with:
- Dynamic quote generation with inventory skew
- Adverse selection detection
- Real-time PnL computation
- Backtesting framework
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Side(str, Enum):
    BID = "bid"
    ASK = "ask"


@dataclass
class MarketMakerConfig:
    """Configuration for market making strategy."""
    spread_bps: float = 10.0
    order_size: float = 1.0
    max_inventory: float = 10.0
    skew_factor: float = 0.5
    quote_update_interval_ms: int = 100
    min_spread_bps: float = 5.0
    max_spread_bps: float = 50.0
    inventory_skew_limit: float = 0.8
    adverse_selection_threshold: float = 0.7
    pnl_update_interval_ms: int = 1000
    risk_limit_pct: float = 0.02


@dataclass
class Quote:
    """Represents a market quote."""
    price: float
    size: float
    side: str
    timestamp: datetime
    quote_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.side not in (Side.BID, Side.ASK):
            raise ValueError(f"Invalid side: {self.side}. Must be 'bid' or 'ask'")
        if self.price <= 0:
            raise ValueError(f"Price must be positive: {self.price}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive: {self.size}")


@dataclass
class OrderBook:
    """Represents the current order book state."""
    bids: List[Tuple[float, float]] = field(default_factory=list)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    mid_price: float = 0.0
    spread: float = 0.0
    imbalance: float = 0.0

    def compute_imbalance(self) -> float:
        """Compute order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        bid_vol = sum(size for _, size in self.bids) if self.bids else 0.0
        ask_vol = sum(size for _, size in self.asks) if self.asks else 0.0
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return 0.0
        self.imbalance = (bid_vol - ask_vol) / total_vol
        return self.imbalance

    def update_mid_price(self) -> float:
        """Update mid price from best bid/ask."""
        if self.bids and self.asks:
            best_bid = max(price for price, _ in self.bids)
            best_ask = min(price for price, _ in self.asks)
            self.mid_price = (best_bid + best_ask) / 2.0
            self.spread = best_ask - best_bid
        return self.mid_price


@dataclass
class Trade:
    """Represents a trade execution."""
    price: float
    size: float
    side: str
    timestamp: datetime
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class PnLResult:
    """PnL computation result."""
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    position_value: float = 0.0
    inventory: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BacktestResult:
    """Backtest result."""
    total_trades: int = 0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    total_quotes: int = 0
    fill_rate: float = 0.0
    inventory_violations: int = 0
    duration_seconds: float = 0.0


@dataclass
class Metrics:
    """Trading metrics."""
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_spread: float = 0.0
    avg_inventory: float = 0.0
    inventory_volatility: float = 0.0
    quote_count: int = 0
    trade_count: int = 0
    fill_rate: float = 0.0


class InventoryManager:
    """Manages market maker inventory and position risk."""

    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.position: float = 0.0
        self.avg_entry_price: float = 0.0
        self.trades: List[Trade] = []
        self._position_history: List[float] = []

    def update_position(self, trade: Trade) -> None:
        """Update position based on executed trade."""
        old_position = self.position

        if trade.side == Side.BID:
            self.position += trade.size
        else:
            self.position -= trade.size

        if self.avg_entry_price == 0:
            self.avg_entry_price = trade.price
        else:
            total_cost = self.avg_entry_price * old_position + trade.price * trade.size
            new_position = old_position + (trade.size if trade.side == Side.BID else -trade.size)
            if new_position != 0:
                self.avg_entry_price = total_cost / new_position
            else:
                self.avg_entry_price = 0.0

        self.trades.append(trade)
        self._position_history.append(self.position)

        logger.debug(
            f"Position updated: {old_position:.4f} -> {self.position:.4f} "
            f"at {trade.price:.6f}"
        )

    def get_inventory_skew(self) -> float:
        """
        Compute inventory skew in range [-1, 1].

        Positive skew means long inventory (more bids filled).
        Negative skew means short inventory (more asks filled).
        """
        if self.config.max_inventory == 0:
            return 0.0

        skew = self.position / self.config.max_inventory
        skew = np.clip(skew, -1.0, 1.0)
        return float(skew)

    def should_reduce_position(self) -> bool:
        """Check if position should be reduced based on limits."""
        return abs(self.position) >= self.config.max_inventory * self.config.inventory_skew_limit

    def get_target_position(self) -> float:
        """
        Compute target position to revert to neutral.

        Returns the size and side needed to reduce inventory.
        """
        if abs(self.position) < 0.01:
            return 0.0

        target = -self.position * self.config.skew_factor
        return float(np.clip(target, -self.config.order_size, self.config.order_size))

    def get_position_value(self, current_price: float) -> float:
        """Compute current position value."""
        return self.position * current_price

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Compute unrealized PnL."""
        if self.position == 0:
            return 0.0
        return self.position * (current_price - self.avg_entry_price)


class AdverseSelectionDetector:
    """Detects adverse selection and toxic order flow."""

    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self._trade_history: List[Trade] = []
        self._quote_history: List[Quote] = []
        self._toxicity_scores: List[float] = []

    def detect_toxic_flow(self, trades: List[Trade], quotes: List[Quote]) -> bool:
        """
        Detect toxic order flow using VPIN-like methodology.

        Returns True if order flow is considered toxic.
        """
        if len(trades) < 10:
            return False

        self._trade_history = trades[-100:]
        self._quote_history = quotes[-100:]

        buy_volume = sum(t.size for t in trades if t.side == Side.BID)
        sell_volume = sum(t.size for t in trades if t.side == Side.ASK)
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return False

        order_imbalance = abs(buy_volume - sell_volume) / total_volume

        price_impact = self._compute_price_impact(trades)

        toxicity_score = 0.6 * order_imbalance + 0.4 * min(price_impact, 1.0)
        self._toxicity_scores.append(toxicity_score)

        is_toxic = toxicity_score > self.config.adverse_selection_threshold

        if is_toxic:
            logger.warning(f"Toxic flow detected: toxicity={toxicity_score:.4f}")

        return is_toxic

    def compute_adverse_selection_cost(
        self, trades: List[Trade], quotes: List[Quote]
    ) -> float:
        """
        Compute adverse selection cost from trades and quotes.

        Measures the cost of trading against informed traders.
        """
        if len(trades) < 5 or len(quotes) < 5:
            return 0.0

        price_changes = []
        for i in range(1, len(trades)):
            change = abs(trades[i].price - trades[i-1].price) / trades[i-1].price
            price_changes.append(change)

        if not price_changes:
            return 0.0

        avg_price_change = np.mean(price_changes)

        trade_directions = [1 if t.side == Side.BID else -1 for t in trades]
        autocorrelation = self._compute_autocorrelation(trade_directions)

        adverse_selection_cost = avg_price_change * abs(autocorrelation)
        return float(adverse_selection_cost)

    def should_widen_spread(self, toxicity_score: float) -> bool:
        """Determine if spread should be widened based on toxicity."""
        return toxicity_score > self.config.adverse_selection_threshold

    def _compute_price_impact(self, trades: List[Trade]) -> float:
        """Compute price impact of trades."""
        if len(trades) < 2:
            return 0.0

        price_range = max(t.price for t in trades) - min(t.price for t in trades)
        avg_price = np.mean([t.price for t in trades])

        if avg_price == 0:
            return 0.0

        return price_range / avg_price

    def _compute_autocorrelation(self, series: List[float], lag: int = 1) -> float:
        """Compute autocorrelation of trade directions."""
        if len(series) < lag + 2:
            return 0.0

        series_np = np.array(series)
        mean = np.mean(series_np)
        var = np.var(series_np)

        if var == 0:
            return 0.0

        autocov = np.mean(
            (series_np[:-lag] - mean) * (series_np[lag:] - mean)
        )

        return float(autocov / var)


class QuoteGenerator:
    """Generates market making quotes with dynamic adjustments."""

    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.adverse_selection = AdverseSelectionDetector(config)

    def compute_bid_ask(
        self,
        mid_price: float,
        spread: float,
        inventory_skew: float,
    ) -> Tuple[float, float]:
        """
        Compute bid and ask prices with inventory skew.

        Inventory skew adjusts quotes to encourage position reduction.
        """
        half_spread = spread / 2.0

        skew_adjustment = inventory_skew * self.config.skew_factor * half_spread

        bid = mid_price - half_spread - skew_adjustment
        ask = mid_price + half_spread - skew_adjustment

        if bid >= ask:
            mid = (bid + ask) / 2.0
            min_half = mid_price * self.config.min_spread_bps / 10000.0 / 2.0
            bid = mid - min_half
            ask = mid + min_half

        return float(bid), float(ask)

    def adjust_for_volatility(self, base_spread: float, volatility: float) -> float:
        """
        Adjust spread based on market volatility.

        Higher volatility -> wider spread to compensate for risk.
        """
        if volatility <= 0:
            return base_spread

        vol_multiplier = 1.0 + np.sqrt(volatility) * 2.0
        adjusted_spread = base_spread * vol_multiplier

        min_spread = base_spread * self.config.min_spread_bps / self.config.spread_bps
        max_spread = base_spread * self.config.max_spread_bps / self.config.spread_bps

        return float(np.clip(adjusted_spread, min_spread, max_spread))

    def adjust_for_imbalance(self, order_imbalance: float) -> float:
        """
        Adjust spread based on order book imbalance.

        Positive imbalance (more bids) -> tighten spread on ask side.
        Negative imbalance (more asks) -> tighten spread on bid side.
        """
        if abs(order_imbalance) < 0.1:
            return 1.0

        imbalance_factor = 1.0 - abs(order_imbalance) * 0.3
        return float(np.clip(imbalance_factor, 0.5, 1.0))

    def compute_optimal_spread(
        self, volatility: float, adverse_selection: float
    ) -> float:
        """
        Compute optimal spread using Avellaneda-Stoikov framework.

        Considers volatility and adverse selection risk.
        """
        base_spread_bps = self.config.spread_bps / 10000.0

        vol_component = volatility * 2.0
        adverse_component = adverse_selection * 3.0

        optimal_spread_bps = base_spread_bps + vol_component + adverse_component

        min_spread = self.config.min_spread_bps / 10000.0
        max_spread = self.config.max_spread_bps / 10000.0

        optimal_spread_bps = np.clip(optimal_spread_bps, min_spread, max_spread)

        return float(optimal_spread_bps)


class MarketMakingStrategy:
    """
    Main market making strategy coordinator.

    Manages quote generation, inventory, and risk.
    """

    def __init__(self, config: Optional[MarketMakerConfig] = None):
        self.config = config or MarketMakerConfig()
        self.inventory = InventoryManager(self.config)
        self.quote_generator = QuoteGenerator(self.config)
        self.active_quotes: List[Quote] = []
        self.filled_trades: List[Trade] = []
        self._quote_history: List[Quote] = []
        self._trade_history: List[Trade] = []
        self._pnl_snapshots: List[float] = []
        self._current_mid_price: float = 0.0
        self._last_quote_time: Optional[datetime] = None
        self._is_active: bool = True

        logger.info(f"MarketMakingStrategy initialized: spread={self.config.spread_bps}bps")

    def on_order_book_update(self, orderbook: OrderBook) -> Optional[Quote]:
        """
        Handle order book update and generate new quote if needed.

        Returns new quote or None if quoting is paused.
        """
        if not self._is_active:
            return None

        orderbook.update_mid_price()
        orderbook.compute_imbalance()

        self._current_mid_price = orderbook.mid_price

        if self.inventory.should_reduce_position():
            logger.warning("Inventory limit reached, pausing quotes")
            return None

        if self._should_update_quotes():
            quote = self._generate_single_quote(orderbook)
            if quote:
                self.active_quotes.append(quote)
                self._quote_history.append(quote)
                self._last_quote_time = quote.timestamp

            return quote

        return None

    def on_trade_update(self, trade: Trade) -> None:
        """Handle trade execution update."""
        self.inventory.update_position(trade)
        self.filled_trades.append(trade)
        self._trade_history.append(trade)

        logger.info(
            f"Trade filled: {trade.side} {trade.size:.4f} @ {trade.price:.6f}"
        )

    def on_quote_timeout(self, quote: Quote) -> None:
        """Handle quote expiration."""
        if quote in self.active_quotes:
            self.active_quotes.remove(quote)
            logger.debug(f"Quote expired: {quote.quote_id}")

    def generate_quotes(
        self, orderbook: OrderBook
    ) -> Tuple[Optional[Quote], Optional[Quote]]:
        """
        Generate bid and ask quotes from order book.

        Returns tuple of (bid_quote, ask_quote).
        """
        if not self._is_active or self._current_mid_price == 0:
            return None, None

        if self.inventory.should_reduce_position():
            return None, None

        spread_bps = self._compute_current_spread(orderbook)
        spread = self._current_mid_price * spread_bps

        inventory_skew = self.inventory.get_inventory_skew()

        bid_price, ask_price = self.quote_generator.compute_bid_ask(
            self._current_mid_price, spread, inventory_skew
        )

        now = datetime.utcnow()

        bid_quote = Quote(
            price=bid_price,
            size=self.config.order_size,
            side=Side.BID,
            timestamp=now,
        )

        ask_quote = Quote(
            price=ask_price,
            size=self.config.order_size,
            side=Side.ASK,
            timestamp=now,
        )

        self.active_quotes.extend([bid_quote, ask_quote])
        self._quote_history.extend([bid_quote, ask_quote])
        self._last_quote_time = now

        return bid_quote, ask_quote

    def compute_pnl(self) -> PnLResult:
        """Compute current PnL."""
        realized_pnl = self._compute_realized_pnl()
        unrealized_pnl = self.inventory.get_unrealized_pnl(self._current_mid_price)
        total_pnl = realized_pnl + unrealized_pnl

        position_value = self.inventory.get_position_value(self._current_mid_price)

        result = PnLResult(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            position_value=position_value,
            inventory=self.inventory.position,
        )

        self._pnl_snapshots.append(total_pnl)

        return result

    def pause(self) -> None:
        """Pause market making."""
        self._is_active = False
        self.active_quotes.clear()
        logger.info("Market making paused")

    def resume(self) -> None:
        """Resume market making."""
        self._is_active = True
        logger.info("Market making resumed")

    def _should_update_quotes(self) -> bool:
        """Check if quotes should be updated based on interval."""
        if self._last_quote_time is None:
            return True

        now = datetime.utcnow()
        elapsed_ms = (now - self._last_quote_time).total_seconds() * 1000
        return elapsed_ms >= self.config.quote_update_interval_ms

    def _generate_single_quote(self, orderbook: OrderBook) -> Optional[Quote]:
        """Generate a single quote based on current state."""
        spread_bps = self._compute_current_spread(orderbook)
        spread = self._current_mid_price * spread_bps

        inventory_skew = self.inventory.get_inventory_skew()
        target_position = self.inventory.get_target_position()

        if target_position > 0:
            side = Side.BID
        elif target_position < 0:
            side = Side.ASK
        else:
            side = Side.BID if inventory_skew <= 0 else Side.ASK

        bid_price, ask_price = self.quote_generator.compute_bid_ask(
            self._current_mid_price, spread, inventory_skew
        )

        price = bid_price if side == Side.BID else ask_price

        return Quote(
            price=price,
            size=self.config.order_size,
            side=side,
            timestamp=datetime.utcnow(),
        )

    def _compute_current_spread(self, orderbook: OrderBook) -> float:
        """Compute current spread with all adjustments."""
        base_spread = self.config.spread_bps / 10000.0

        volatility = self._estimate_volatility()
        adverse_selection = self._estimate_adverse_selection()

        optimal_spread = self.quote_generator.compute_optimal_spread(
            volatility, adverse_selection
        )

        adjusted_spread = self.quote_generator.adjust_for_volatility(
            optimal_spread, volatility
        )

        imbalance_factor = self.quote_generator.adjust_for_imbalance(
            orderbook.imbalance
        )

        return float(adjusted_spread * imbalance_factor)

    def _estimate_volatility(self, window: int = 50) -> float:
        """Estimate recent price volatility."""
        if len(self._trade_history) < window:
            return 0.001

        prices = [t.price for t in self._trade_history[-window:]]
        returns = np.diff(np.log(prices))

        if len(returns) == 0:
            return 0.001

        return float(np.std(returns))

    def _estimate_adverse_selection(self) -> float:
        """Estimate adverse selection risk."""
        if len(self._trade_history) < 10:
            return 0.0

        return self.quote_generator.adverse_selection.compute_adverse_selection_cost(
            self._trade_history[-50:],
            self._quote_history[-50:],
        )

    def _compute_realized_pnl(self) -> float:
        """Compute realized PnL from closed positions."""
        if not self.filled_trades:
            return 0.0

        buys = [t for t in self.filled_trades if t.side == Side.BID]
        sells = [t for t in self.filled_trades if t.side == Side.ASK]

        if not buys or not sells:
            return 0.0

        total_bought = sum(t.price * t.size for t in buys)
        total_sold = sum(t.price * t.size for t in sells)
        total_buy_size = sum(t.size for t in buys)
        total_sell_size = sum(t.size for t in sells)

        closed_size = min(total_buy_size, total_sell_size)

        if closed_size == 0:
            return 0.0

        avg_buy_price = total_bought / total_buy_size
        avg_sell_price = total_sold / total_sell_size

        return float(closed_size * (avg_sell_price - avg_buy_price))


class MarketMakerBacktester:
    """Backtesting framework for market making strategies."""

    def __init__(self):
        self.results: List[BacktestResult] = []

    def backtest(
        self,
        strategy: MarketMakingStrategy,
        historical_data: List[dict],
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        historical_data: List of dicts with keys:
            - timestamp: datetime
            - bids: List[Tuple[float, float]]
            - asks: List[Tuple[float, float]]
            - trades: Optional[List[dict]]
        """
        logger.info(f"Starting backtest with {len(historical_data)} data points")

        quotes_generated = 0
        trades_executed = 0
        inventory_violations = 0
        pnl_history = []

        start_time = historical_data[0]["timestamp"] if historical_data else datetime.utcnow()

        for data_point in historical_data:
            orderbook = OrderBook(
                bids=data_point.get("bids", []),
                asks=data_point.get("asks", []),
            )

            quotes = strategy.generate_quotes(orderbook)
            if quotes[0]:
                quotes_generated += 1
            if quotes[1]:
                quotes_generated += 1

            trades = data_point.get("trades", [])
            for trade_data in trades:
                trade = Trade(
                    price=trade_data["price"],
                    size=trade_data["size"],
                    side=trade_data["side"],
                    timestamp=trade_data.get("timestamp", data_point["timestamp"]),
                )
                strategy.on_trade_update(trade)
                trades_executed += 1

                if strategy.inventory.should_reduce_position():
                    inventory_violations += 1

            pnl = strategy.compute_pnl()
            pnl_history.append(pnl.total_pnl)

        end_time = historical_data[-1]["timestamp"] if historical_data else datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        metrics = self._compute_metrics(
            strategy.filled_trades,
            strategy._quote_history,
            pnl_history,
        )

        result = BacktestResult(
            total_trades=trades_executed,
            total_pnl=pnl_history[-1] if pnl_history else 0.0,
            realized_pnl=pnl.realized_pnl,
            unrealized_pnl=pnl.unrealized_pnl,
            max_drawdown=metrics.max_drawdown,
            sharpe_ratio=metrics.sharpe_ratio,
            win_rate=metrics.win_rate,
            avg_trade_pnl=metrics.total_pnl / max(trades_executed, 1),
            total_quotes=quotes_generated,
            fill_rate=trades_executed / max(quotes_generated, 1),
            inventory_violations=inventory_violations,
            duration_seconds=duration,
        )

        self.results.append(result)

        logger.info(
            f"Backtest complete: {trades_executed} trades, "
            f"PnL={result.total_pnl:.4f}, Sharpe={result.sharpe_ratio:.2f}"
        )

        return result

    def _compute_metrics(
        self,
        trades: List[Trade],
        quotes: List[Quote],
        pnl_history: List[float],
    ) -> Metrics:
        """Compute comprehensive trading metrics."""
        if not trades:
            return Metrics()

        trade_pnls = self._compute_trade_pnls(trades)
        winning_trades = sum(1 for p in trade_pnls if p > 0)

        total_pnl = sum(trade_pnls) if trade_pnls else 0.0
        win_rate = winning_trades / max(len(trade_pnls), 1)

        max_drawdown = self._compute_max_drawdown(pnl_history)
        sharpe = self._compute_sharpe_ratio(pnl_history)

        inventory_values = [abs(t.size) for t in trades]
        avg_inventory = np.mean(inventory_values) if inventory_values else 0.0
        inventory_vol = np.std(inventory_values) if inventory_values else 0.0

        return Metrics(
            total_pnl=total_pnl,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            quote_count=len(quotes),
            trade_count=len(trades),
            fill_rate=len(trades) / max(len(quotes), 1),
            avg_inventory=float(avg_inventory),
            inventory_volatility=float(inventory_vol),
        )

    def _compute_trade_pnls(self, trades: List[Trade]) -> List[float]:
        """Compute PnL for each trade pair."""
        if len(trades) < 2:
            return []

        pnls = []
        position = 0.0
        entry_price = 0.0

        for trade in trades:
            if trade.side == Side.BID:
                if position < 0:
                    pnl = (entry_price - trade.price) * min(abs(position), trade.size)
                    pnls.append(pnl)
                position += trade.size
                if position > 0:
                    entry_price = trade.price
            else:
                if position > 0:
                    pnl = (trade.price - entry_price) * min(position, trade.size)
                    pnls.append(pnl)
                position -= trade.size
                if position < 0:
                    entry_price = trade.price

        return pnls

    def _compute_max_drawdown(self, pnl_history: List[float]) -> float:
        """Compute maximum drawdown from PnL history."""
        if not pnl_history:
            return 0.0

        peak = pnl_history[0]
        max_dd = 0.0

        for pnl in pnl_history:
            if pnl > peak:
                peak = pnl
            dd = peak - pnl
            if dd > max_dd:
                max_dd = dd

        return float(max_dd)

    def _compute_sharpe_ratio(
        self, pnl_history: List[float], risk_free_rate: float = 0.0
    ) -> float:
        """Compute annualized Sharpe ratio."""
        if len(pnl_history) < 2:
            return 0.0

        returns = np.diff(pnl_history)
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        daily_sharpe = (avg_return - risk_free_rate) / std_return
        annualized_sharpe = daily_sharpe * np.sqrt(252 * 24 * 60)

        return float(annualized_sharpe)
