"""Order book analysis for liquidity, spread, imbalance, and execution choice."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class OrderBookSignal:
    spread_pct: float
    imbalance: float
    depth_score: float
    liquidity_score: float
    execution_style: str


class OrderBookAnalyzer:
    def analyse(self, bids: list[tuple[float, float]], asks: list[tuple[float, float]], order_notional: float = 1000) -> OrderBookSignal:
        if not bids or not asks:
            return OrderBookSignal(0.01, 0.0, 0.0, 0.0, "market")
        best_bid, best_ask = bids[0][0], asks[0][0]
        mid = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / max(mid, 1e-9)
        bid_depth = sum(price * qty for price, qty in bids[:10])
        ask_depth = sum(price * qty for price, qty in asks[:10])
        imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-9)
        depth_score = float(np.clip((bid_depth + ask_depth) / max(order_notional * 20, 1), 0, 1))
        liquidity_score = float(np.clip(depth_score * (1 - min(spread_pct / 0.01, 1)), 0, 1))
        execution_style = "limit" if spread_pct < 0.0007 and liquidity_score > 0.6 else "vwap" if liquidity_score > 0.35 else "iceberg"
        return OrderBookSignal(float(spread_pct), float(imbalance), depth_score, liquidity_score, execution_style)

    def estimate_market_impact(self, side: str, amount: float, levels: list[tuple[float, float]]) -> float:
        remaining = amount
        cost = 0.0
        filled = 0.0
        for price, qty in levels:
            take = min(remaining, qty)
            cost += take * price
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        if filled == 0:
            return 0.0
        avg = cost / filled
        reference = levels[0][0]
        return float((avg - reference) / reference if side == "buy" else (reference - avg) / reference)


def _demo() -> None:
    bids = [(49990 - i * 5, 1.5 + i * 0.1) for i in range(20)]
    asks = [(50010 + i * 5, 1.2 + i * 0.08) for i in range(20)]
    print("Order book analyzer ready")
    print(OrderBookAnalyzer().analyse(bids, asks, order_notional=5000))


if __name__ == "__main__":
    _demo()
