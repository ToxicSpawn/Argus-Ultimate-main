"""
Arbitrage Engine
Cross-exchange basis trading and funding rate arbitrage detection.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ArbitrageOpportunity:
    pair: str
    exchange_a: str
    exchange_b: str
    spread_pct: float
    direction: str
    estimated_profit_pct: float
    confidence: float
    timestamp: float = 0.0


class CrossExchangeArbitrage:
    """Detect and track cross-exchange price discrepancies."""

    def __init__(self, min_spread_pct: float = 0.001, fee_estimate_pct: float = 0.0006, lookback: int = 100):
        self.min_spread_pct = float(min_spread_pct)
        self.fee_estimate_pct = float(fee_estimate_pct)
        self._prices: Dict[str, Dict[str, deque]] = {}
        self.lookback = int(lookback)

    def update_price(self, symbol: str, exchange: str, price: float) -> None:
        if symbol not in self._prices:
            self._prices[symbol] = {}
        if exchange not in self._prices[symbol]:
            self._prices[symbol][exchange] = deque(maxlen=self.lookback)
        self._prices[symbol][exchange].append(float(price))

    def scan(self, symbol: str) -> List[ArbitrageOpportunity]:
        if symbol not in self._prices:
            return []
        exchanges = self._prices[symbol]
        if len(exchanges) < 2:
            return []
        opportunities = []
        exchange_names = list(exchanges.keys())
        for i in range(len(exchange_names)):
            for j in range(i + 1, len(exchange_names)):
                ex_a, ex_b = exchange_names[i], exchange_names[j]
                if not exchanges[ex_a] or not exchanges[ex_b]:
                    continue
                price_a = float(exchanges[ex_a][-1])
                price_b = float(exchanges[ex_b][-1])
                mid = (price_a + price_b) / 2
                if mid < 1e-12:
                    continue
                spread_pct = (price_a - price_b) / mid
                net_spread = abs(spread_pct) - 2 * self.fee_estimate_pct
                if net_spread > self.min_spread_pct:
                    direction = "buy_b_sell_a" if spread_pct > 0 else "buy_a_sell_b"
                    if len(exchanges[ex_a]) >= 10 and len(exchanges[ex_b]) >= 10:
                        hist_a = np.array(list(exchanges[ex_a])[-10:])
                        hist_b = np.array(list(exchanges[ex_b])[-10:])
                        hist_spreads = (hist_a - hist_b) / ((hist_a + hist_b) / 2)
                        consistency = float(np.mean(np.abs(hist_spreads) > self.min_spread_pct))
                    else:
                        consistency = 0.5
                    opportunities.append(ArbitrageOpportunity(
                        pair=symbol, exchange_a=ex_a, exchange_b=ex_b,
                        spread_pct=abs(spread_pct), direction=direction,
                        estimated_profit_pct=net_spread, confidence=consistency,
                    ))
        return opportunities


class FundingRateArbitrage:
    """Detect funding rate arbitrage between perp and spot."""

    def __init__(self, min_annualized_yield: float = 0.10, lookback: int = 100):
        self.min_annualized_yield = float(min_annualized_yield)
        self._rates: Dict[str, deque] = {}
        self.lookback = int(lookback)

    def update_rate(self, symbol: str, rate: float) -> None:
        if symbol not in self._rates:
            self._rates[symbol] = deque(maxlen=self.lookback)
        self._rates[symbol].append(float(rate))

    def check(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._rates or len(self._rates[symbol]) < 3:
            return {"opportunity": False}
        rates = np.array(list(self._rates[symbol]))
        current = float(rates[-1])
        avg = float(np.mean(rates))
        annualized = abs(current) * 3 * 365
        if annualized < self.min_annualized_yield:
            return {"opportunity": False, "annualized_yield": annualized}
        direction = "short_perp_long_spot" if current > 0 else "long_perp_short_spot"
        stability = 1.0 - float(np.std(rates) / max(abs(avg), 1e-12))
        return {
            "opportunity": True, "direction": direction,
            "current_rate": current, "annualized_yield": annualized,
            "avg_rate": avg, "stability": max(0.0, stability),
        }


class BasisTracker:
    """Track spot-futures basis for basis trading signals."""

    def __init__(self, lookback: int = 200):
        self._spot: deque = deque(maxlen=lookback)
        self._futures: deque = deque(maxlen=lookback)

    def update(self, spot: float, futures: float) -> Dict[str, Any]:
        self._spot.append(float(spot))
        self._futures.append(float(futures))
        if len(self._spot) < 5:
            return {"basis_pct": 0.0, "signal": "neutral"}
        basis = (float(self._futures[-1]) - float(self._spot[-1])) / max(float(self._spot[-1]), 1e-12)
        basis_arr = np.array(list(self._futures)) - np.array(list(self._spot))
        spot_arr = np.array(list(self._spot))
        basis_pcts = basis_arr / np.maximum(spot_arr, 1e-12)
        mean_basis = float(np.mean(basis_pcts))
        std_basis = float(np.std(basis_pcts))
        z_score = (basis - mean_basis) / max(std_basis, 1e-12) if std_basis > 1e-12 else 0.0
        signal = "neutral"
        if z_score > 2.0:
            signal = "short_basis"
        elif z_score < -2.0:
            signal = "long_basis"
        return {"basis_pct": basis, "mean_basis": mean_basis, "z_score": z_score, "signal": signal}


class ArbitrageEngine:
    """Unified arbitrage detection engine."""

    def __init__(self):
        self.cross_exchange = CrossExchangeArbitrage()
        self.funding_arb = FundingRateArbitrage()
        self.basis_tracker = BasisTracker()

    def on_price(self, symbol: str, exchange: str, price: float) -> None:
        self.cross_exchange.update_price(symbol, exchange, price)

    def on_funding_rate(self, symbol: str, rate: float) -> None:
        self.funding_arb.update_rate(symbol, rate)

    def on_spot_futures(self, spot: float, futures: float) -> None:
        self.basis_tracker.update(spot, futures)

    def scan_all(self, symbol: str) -> Dict[str, Any]:
        cross_opps = self.cross_exchange.scan(symbol)
        funding = self.funding_arb.check(symbol)
        basis = self.basis_tracker.update(
            self.basis_tracker._spot[-1] if self.basis_tracker._spot else 0,
            self.basis_tracker._futures[-1] if self.basis_tracker._futures else 0,
        )
        return {
            "cross_exchange": [
                {"pair": o.pair, "spread": o.spread_pct, "profit": o.estimated_profit_pct, "direction": o.direction}
                for o in cross_opps
            ],
            "funding_arb": funding,
            "basis": basis,
            "any_opportunity": len(cross_opps) > 0 or funding.get("opportunity", False),
        }
