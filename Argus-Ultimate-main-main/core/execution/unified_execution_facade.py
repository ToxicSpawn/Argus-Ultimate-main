from __future__ import annotations

from dataclasses import dataclass

from unified_execution_engine import UnifiedExecutionEngine, UnifiedExecutionRequest


@dataclass(frozen=True)
class ExecutionFacadeRequest:
    symbol: str
    side: str
    quantity: float
    strategy_id: str
    price: float
    equity: float
    symbol_notional_after: float
    cluster_notional_after: float
    gross_notional_after: float
    top_of_book_notional: float
    spread_bps: float
    volatility_bps: float
    fee_rate: float
    allow_market_orders: bool = False


class UnifiedExecutionFacade:
    def __init__(self, engine: UnifiedExecutionEngine) -> None:
        self.engine = engine

    def submit(self, request: ExecutionFacadeRequest) -> str:
        return self.engine.execute(
            UnifiedExecutionRequest(
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                strategy_id=request.strategy_id,
                price=request.price,
                equity=request.equity,
                symbol_notional_after=request.symbol_notional_after,
                cluster_notional_after=request.cluster_notional_after,
                gross_notional_after=request.gross_notional_after,
                top_of_book_notional=request.top_of_book_notional,
                spread_bps=request.spread_bps,
                volatility_bps=request.volatility_bps,
                fee_rate=request.fee_rate,
                allow_market_orders=request.allow_market_orders,
            )
        )
