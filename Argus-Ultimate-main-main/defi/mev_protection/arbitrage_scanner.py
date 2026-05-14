"""Cross-DEX arbitrage monitoring and route discovery."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DexQuote:
    dex: str
    pair: str
    token_in: str
    token_out: str
    price: float
    liquidity_usd: float
    fee_pct: float = 0.003
    gas_cost_usd: float = 8.0
    chain: str = "ethereum"
    timestamp: float = field(default_factory=time.time)


@dataclass
class ArbitrageRoute:
    route_id: str
    hops: List[str]
    dexes: List[str]
    expected_profit_usd: float
    trade_size_usd: float
    confidence: float
    chain: str = "ethereum"
    metadata: Dict[str, float] = field(default_factory=dict)


class ArbitrageScanner:
    """Track DEX quotes and detect direct or multi-hop arbitrage."""

    def __init__(self, min_profit_usd: float = 20.0) -> None:
        self.min_profit_usd = min_profit_usd
        self._quotes_by_pair: Dict[str, List[DexQuote]] = defaultdict(list)

    def update_quote(self, quote: DexQuote) -> None:
        quotes = [q for q in self._quotes_by_pair[quote.pair] if q.dex != quote.dex or q.chain != quote.chain]
        quotes.append(quote)
        self._quotes_by_pair[quote.pair] = sorted(quotes, key=lambda q: q.timestamp, reverse=True)[:20]

    def bulk_update(self, quotes: Iterable[DexQuote]) -> None:
        for quote in quotes:
            self.update_quote(quote)

    def monitor_prices_across_dexes(self, pair: Optional[str] = None) -> Dict[str, List[DexQuote]]:
        if pair is not None:
            return {pair: list(self._quotes_by_pair.get(pair, []))}
        return {symbol: list(quotes) for symbol, quotes in self._quotes_by_pair.items()}

    def detect_price_discrepancies(self, pair: str) -> List[ArbitrageRoute]:
        quotes = self._quotes_by_pair.get(pair, [])
        opportunities: List[ArbitrageRoute] = []

        for buy_quote in quotes:
            for sell_quote in quotes:
                if buy_quote.dex == sell_quote.dex or buy_quote.price <= 0:
                    continue
                spread = (sell_quote.price - buy_quote.price) / buy_quote.price
                if spread <= 0:
                    continue
                trade_size = self.calculate_optimal_trade_size(buy_quote, sell_quote)
                gross_profit = trade_size * spread
                fees = trade_size * (buy_quote.fee_pct + sell_quote.fee_pct)
                gas = buy_quote.gas_cost_usd + sell_quote.gas_cost_usd
                net_profit = gross_profit - fees - gas
                if net_profit < self.min_profit_usd:
                    continue
                opportunities.append(
                    ArbitrageRoute(
                        route_id=f"{pair}:{buy_quote.dex}->{sell_quote.dex}",
                        hops=[buy_quote.token_in, buy_quote.token_out],
                        dexes=[buy_quote.dex, sell_quote.dex],
                        expected_profit_usd=net_profit,
                        trade_size_usd=trade_size,
                        confidence=min(0.95, 0.4 + spread * 5),
                        chain=buy_quote.chain,
                        metadata={"spread_pct": spread * 100, "gross_profit_usd": gross_profit},
                    )
                )

        return sorted(opportunities, key=lambda route: route.expected_profit_usd, reverse=True)

    def calculate_optimal_trade_size(self, buy_quote: DexQuote, sell_quote: DexQuote) -> float:
        constrained_liquidity = min(buy_quote.liquidity_usd, sell_quote.liquidity_usd)
        spread = max((sell_quote.price - buy_quote.price) / max(buy_quote.price, 1e-9), 0.0)
        size_factor = min(0.2, 0.05 + spread)
        return max(1_000.0, constrained_liquidity * size_factor)

    def find_multi_hop_routes(self, start_token: str, end_token: str, max_hops: int = 3) -> List[ArbitrageRoute]:
        graph: Dict[str, List[Tuple[str, DexQuote]]] = defaultdict(list)
        for quotes in self._quotes_by_pair.values():
            for quote in quotes:
                graph[quote.token_in].append((quote.token_out, quote))

        routes: List[ArbitrageRoute] = []
        self._dfs_routes(graph, start_token, end_token, max_hops, [], [], set(), routes)
        return sorted(routes, key=lambda route: route.expected_profit_usd, reverse=True)

    def _dfs_routes(
        self,
        graph: Dict[str, List[Tuple[str, DexQuote]]],
        current: str,
        target: str,
        remaining_hops: int,
        tokens: List[str],
        dexes: List[str],
        visited: Set[str],
        routes: List[ArbitrageRoute],
    ) -> None:
        if remaining_hops < 0:
            return
        if current == target and len(tokens) > 1:
            efficiency = 1.0 / max(len(dexes), 1)
            profit = max(self.min_profit_usd, 50.0 * efficiency)
            routes.append(
                ArbitrageRoute(
                    route_id="->".join(tokens),
                    hops=list(tokens),
                    dexes=list(dexes),
                    expected_profit_usd=profit,
                    trade_size_usd=5_000.0 * efficiency,
                    confidence=max(0.3, efficiency),
                )
            )
            return

        visited.add(current)
        for next_token, quote in graph.get(current, []):
            if next_token in visited:
                continue
            self._dfs_routes(
                graph,
                next_token,
                target,
                remaining_hops - 1,
                tokens + [current, next_token] if not tokens else tokens + [next_token],
                dexes + [quote.dex],
                set(visited),
                routes,
            )
