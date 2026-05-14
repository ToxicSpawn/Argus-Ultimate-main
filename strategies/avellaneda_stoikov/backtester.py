"""Backtesting utilities for Avellaneda-Stoikov market making."""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence

from .market_maker import AvellanedaStoikovMarketMaker, MarketSnapshot, OptimalQuote
from .profit_calculator import FillRecord, ProfitCalculator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BacktestMetrics:
    sharpe_ratio: float
    max_drawdown: float
    fill_rate: float
    total_fills: int
    total_quotes: int
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    average_spread: float
    average_inventory: float


@dataclass(slots=True)
class BacktestResult:
    metrics: BacktestMetrics
    equity_curve: list[float] = field(default_factory=list)
    fills: list[FillRecord] = field(default_factory=list)


class AvellanedaStoikovBacktester:
    """Simulates maker quotes against a stream of market snapshots."""

    def __init__(
        self,
        market_maker: AvellanedaStoikovMarketMaker,
        fee_bps: float = 0.0,
        slippage_bps: float = 1.0,
        market_impact_bps: float = 0.5,
        queue_position: float = 0.5,
    ) -> None:
        if not 0.0 <= queue_position <= 1.0:
            raise ValueError("queue_position must be in [0, 1]")
        self.market_maker = market_maker
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.market_impact_bps = market_impact_bps
        self.queue_position = queue_position
        self.profit_calculator = ProfitCalculator()

    def _execution_price(self, quote_price: float, side: str) -> float:
        impact = (self.slippage_bps + self.market_impact_bps) / 10_000.0
        if side == "buy":
            return quote_price * (1.0 + impact)
        return quote_price * (1.0 - impact)

    def _fill_probability(self, quote: OptimalQuote, snapshot: MarketSnapshot, side: str) -> float:
        if side == "buy":
            distance = max(0.0, snapshot.mid_price - quote.bid_price)
        else:
            distance = max(0.0, quote.ask_price - snapshot.mid_price)
        normalized = distance / max(snapshot.mid_price, 1e-12)
        aggressiveness = math.exp(-normalized * max(self.market_maker.config.kappa, 1e-6) * 10_000.0)
        touch_bonus = 1.0 if ((side == "buy" and snapshot.best_bid <= quote.bid_price) or (side == "sell" and snapshot.best_ask >= quote.ask_price)) else 0.5
        queue_factor = 1.0 - 0.5 * self.queue_position
        return max(0.0, min(1.0, aggressiveness * touch_bonus * queue_factor))

    def _simulate_fills(self, quote: OptimalQuote, snapshot: MarketSnapshot) -> list[FillRecord]:
        fills: list[FillRecord] = []
        for side, price, quantity in (
            ("buy", quote.bid_price, quote.bid_size),
            ("sell", quote.ask_price, quote.ask_size),
        ):
            if quantity <= 0:
                continue
            probability = self._fill_probability(quote, snapshot, side)
            threshold = 0.45 + 0.10 * (1.0 - min(snapshot.market_depth, 1.0))
            if probability >= threshold:
                execution_price = self._execution_price(price, side)
                fee = execution_price * quantity * self.fee_bps / 10_000.0
                fills.append(
                    FillRecord(
                        side=side,
                        price=execution_price,
                        quantity=quantity,
                        reference_mid=snapshot.mid_price,
                        timestamp=snapshot.timestamp,
                        fee=fee,
                    )
                )
        return fills

    @staticmethod
    def _max_drawdown(equity_curve: Sequence[float]) -> float:
        peak = -float("inf")
        max_dd = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_dd = min(max_dd, (equity - peak) / peak)
        return abs(max_dd)

    @staticmethod
    def _sharpe(returns: Sequence[float]) -> float:
        if len(returns) < 2:
            return 0.0
        mean_ret = statistics.fmean(returns)
        std_ret = statistics.pstdev(returns)
        if std_ret == 0:
            return 0.0
        return mean_ret / std_ret * math.sqrt(len(returns))

    def run(self, snapshots: Iterable[MarketSnapshot]) -> BacktestResult:
        snapshot_list = list(snapshots)
        equity_curve: list[float] = []
        fills: list[FillRecord] = []
        spreads: list[float] = []
        inventories: list[float] = []
        total_quotes = 0
        latest_mid = 1.0

        for snapshot in snapshot_list:
            quote = self.market_maker.generate_quotes(snapshot)
            total_quotes += 1
            spreads.append(quote.optimal_spread)
            inventories.append(quote.inventory)
            latest_mid = snapshot.mid_price

            step_fills = self._simulate_fills(quote, snapshot)
            for fill in step_fills:
                self.market_maker.inventory_manager.update_fill(fill.side, fill.quantity, fill.price)
                self.profit_calculator.process_fill(fill)
                fills.append(fill)

            pnl = self.profit_calculator.snapshot(snapshot.mid_price)
            equity_curve.append(pnl.total_pnl)

        returns = [equity_curve[i] - equity_curve[i - 1] for i in range(1, len(equity_curve))]
        pnl_snapshot = self.profit_calculator.snapshot(latest_mid)
        metrics = BacktestMetrics(
            sharpe_ratio=self._sharpe(returns),
            max_drawdown=self._max_drawdown(equity_curve),
            fill_rate=(len(fills) / (total_quotes * 2)) if total_quotes else 0.0,
            total_fills=len(fills),
            total_quotes=total_quotes,
            total_pnl=pnl_snapshot.total_pnl,
            realized_pnl=pnl_snapshot.realized_pnl,
            unrealized_pnl=pnl_snapshot.unrealized_pnl,
            average_spread=statistics.fmean(spreads) if spreads else 0.0,
            average_inventory=statistics.fmean(inventories) if inventories else 0.0,
        )
        return BacktestResult(metrics=metrics, equity_curve=equity_curve, fills=fills)
