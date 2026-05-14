"""
Cross-DEX Arbitrage Engine — exploits price discrepancies between decentralised
exchanges (DEXs) on the same or different chains.

Flow:
  1. Monitor prices across multiple DEXs via registered quote functions
  2. Detect price discrepancies after accounting for gas, fees, slippage
  3. Compute triangular arbitrage paths (A → B → C → A)
  4. Build, simulate, and execute atomic bundle transactions
  5. Verify execution and log results

All features are disabled by default (``dex.cross_dex_arb_enabled: false``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Stale quote threshold
_STALE_SECONDS = 15.0

# Default expiry for arbitrage opportunities
_DEFAULT_EXPIRY_SECONDS = 5.0

# Default gas estimates by network
_DEFAULT_GAS_COSTS: Dict[str, float] = {
    "ethereum": 25.0,
    "arbitrum": 0.5,
    "optimism": 0.3,
    "base": 0.3,
    "polygon": 0.05,
    "bnb": 0.15,
    "avalanche": 0.2,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DEXQuote:
    """A single DEX price observation."""

    dex_name: str
    pair: str
    price: float
    liquidity: float
    slippage_estimate: float
    gas_estimate: float
    timestamp: datetime

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        if now is None:
            now = datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = now - self.timestamp
        return delta.total_seconds()

    def is_stale(self, now: Optional[datetime] = None) -> bool:
        return self.age_seconds(now) > _STALE_SECONDS


@dataclass
class PriceUpdate:
    """A streaming price update emitted by the monitor."""

    pair: str
    quotes: List[DEXQuote]
    best_buy: Optional[DEXQuote]
    best_sell: Optional[DEXQuote]
    spread_bps: float
    timestamp: datetime


@dataclass
class ArbOpportunity:
    """A detected cross-DEX arbitrage opportunity."""

    pair: str
    buy_dex: str
    sell_dex: str
    buy_price: float
    sell_price: float
    spread_pct: float
    estimated_profit: float
    estimated_gas: float
    net_profit: float
    expiry_timestamp: datetime


@dataclass
class TriangularPath:
    """A triangular arbitrage path: A → B → C → A."""

    tokens: List[str]
    pairs: List[str]
    dexes: List[str]
    input_amount: float
    output_amount: float
    gross_profit: float
    net_profit: float
    total_fees: float
    total_gas: float


@dataclass
class Bundle:
    """An atomic bundle of transactions for arbitrage execution."""

    opportunity: ArbOpportunity
    amount: float
    transactions: List[Dict[str, Any]]
    expected_profit: float
    max_slippage: float
    deadline: int  # Unix timestamp


@dataclass
class SimulationResult:
    """Result of a bundle simulation."""

    success: bool
    simulated_profit: float
    simulated_gas: float
    price_impact: float
    failure_reason: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of a bundle execution."""

    success: bool
    tx_hash: Optional[str]
    actual_profit: float
    actual_gas: float
    opportunity: ArbOpportunity
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# DEX Price Monitor
# ---------------------------------------------------------------------------


class DEXPriceMonitor:
    """
    Monitors prices across multiple DEXs and provides unified quote access.

    Parameters
    ----------
    stale_seconds : float
        Seconds after which a quote is considered stale.
    """

    def __init__(self, stale_seconds: float = _STALE_SECONDS) -> None:
        self._stale_seconds = stale_seconds
        self._dexes: Dict[str, Callable[[str], Optional[DEXQuote]]] = {}
        self._cache: Dict[str, List[DEXQuote]] = {}
        self._last_update: Dict[str, datetime] = {}
        logger.info("DEXPriceMonitor initialised (stale=%ds)", stale_seconds)

    def add_dex(self, name: str, quote_fn: Callable[[str], Optional[DEXQuote]]) -> None:
        """
        Register a DEX with a quote-fetching function.

        Parameters
        ----------
        name : str
            DEX name (e.g. "uniswap_v3", "sushiswap").
        quote_fn : callable
            Function that takes a pair string and returns a DEXQuote or None.
        """
        self._dexes[name.lower()] = quote_fn
        logger.debug("Registered DEX: %s", name)

    def remove_dex(self, name: str) -> None:
        """Unregister a DEX."""
        name = name.lower()
        self._dexes.pop(name, None)
        self._cache.pop(name, None)
        logger.debug("Removed DEX: %s", name)

    def get_quotes(self, pair: str, refresh: bool = True) -> List[DEXQuote]:
        """
        Get current quotes for a pair from all registered DEXs.

        Parameters
        ----------
        pair : str
            Trading pair (e.g. "ETH/USDC").
        refresh : bool
            If True, fetch fresh quotes from each DEX.

        Returns
        -------
        list of DEXQuote
            All available quotes, sorted by price ascending.
        """
        pair = pair.upper()
        quotes: List[DEXQuote] = []

        if refresh:
            for dex_name, quote_fn in self._dexes.items():
                try:
                    quote = quote_fn(pair)
                    if quote is not None:
                        quotes.append(quote)
                except Exception:
                    logger.warning("Failed to fetch quote from %s for %s", dex_name, pair)

            self._cache[pair] = quotes
            self._last_update[pair] = datetime.now(timezone.utc)
        else:
            quotes = self._cache.get(pair, [])

        quotes.sort(key=lambda q: q.price)
        return quotes

    def find_best_price(self, pair: str, side: str) -> Optional[DEXQuote]:
        """
        Find the best price for a pair on a given side.

        Parameters
        ----------
        pair : str
            Trading pair.
        side : str
            "buy" for best ask (lowest price), "sell" for best bid (highest).

        Returns
        -------
        DEXQuote or None
            Best quote for the requested side.
        """
        quotes = self.get_quotes(pair)
        if not quotes:
            return None

        if side.lower() == "buy":
            return min(quotes, key=lambda q: q.price)
        return max(quotes, key=lambda q: q.price)

    async def monitor_prices(
        self,
        pair: str,
        interval_seconds: float = 1.0,
    ) -> AsyncIterator[PriceUpdate]:
        """
        Continuously monitor prices and yield updates.

        Parameters
        ----------
        pair : str
            Trading pair to monitor.
        interval_seconds : float
            Seconds between updates.

        Yields
        ------
        PriceUpdate
            Latest price update with best buy/sell quotes and spread.
        """
        logger.info("Starting price monitor for %s (interval=%.1fs)", pair, interval_seconds)
        try:
            while True:
                quotes = self.get_quotes(pair)
                if quotes:
                    best_buy = min(quotes, key=lambda q: q.price)
                    best_sell = max(quotes, key=lambda q: q.price)

                    spread_bps = 0.0
                    if best_buy.price > 0:
                        spread_bps = (best_sell.price - best_buy.price) / best_buy.price * 10_000.0

                    update = PriceUpdate(
                        pair=pair,
                        quotes=quotes,
                        best_buy=best_buy if best_buy.dex_name != best_sell.dex_name else None,
                        best_sell=best_sell if best_buy.dex_name != best_sell.dex_name else None,
                        spread_bps=spread_bps,
                        timestamp=datetime.now(timezone.utc),
                    )
                    yield update

                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("Price monitor cancelled for %s", pair)
            raise

    def get_dex_count(self) -> int:
        """Return the number of registered DEXs."""
        return len(self._dexes)

    def get_stats(self) -> Dict[str, Any]:
        """Return monitor statistics."""
        return {
            "dex_count": len(self._dexes),
            "cached_pairs": len(self._cache),
            "dexes": list(self._dexes.keys()),
        }


# ---------------------------------------------------------------------------
# Arbitrage Calculator
# ---------------------------------------------------------------------------


class ArbitrageCalculator:
    """
    Computes arbitrage profitability and optimal trade parameters.

    Parameters
    ----------
    default_fee_bps : float
        Default DEX fee in basis points.
    min_profit_usd : float
        Minimum net profit in USD to consider an opportunity viable.
    network : str
        Target blockchain network for gas estimation.
    """

    def __init__(
        self,
        default_fee_bps: float = 30.0,
        min_profit_usd: float = 1.0,
        network: str = "arbitrum",
    ) -> None:
        self.default_fee_bps = default_fee_bps
        self.min_profit_usd = min_profit_usd
        self.network = network.lower()
        logger.info(
            "ArbitrageCalculator: fee=%.1fbps min_profit=$%.2f network=%s",
            default_fee_bps,
            min_profit_usd,
            network,
        )

    def compute_arb_opportunity(
        self,
        buy_quote: DEXQuote,
        sell_quote: DEXQuote,
        gas_cost: Optional[float] = None,
    ) -> Optional[ArbOpportunity]:
        """
        Compute an arbitrage opportunity from buy/sell quotes.

        Parameters
        ----------
        buy_quote : DEXQuote
            Quote from the DEX to buy on (lower price).
        sell_quote : DEXQuote
            Quote from the DEX to sell on (higher price).
        gas_cost : float, optional
            Gas cost in USD. Defaults to network estimate.

        Returns
        -------
        ArbOpportunity or None
            The computed opportunity, or None if not profitable.
        """
        if buy_quote.pair != sell_quote.pair:
            logger.warning("Pair mismatch: %s vs %s", buy_quote.pair, sell_quote.pair)
            return None

        if buy_quote.price <= 0 or sell_quote.price <= 0:
            return None

        if buy_quote.price >= sell_quote.price:
            return None

        pair = buy_quote.pair
        buy_price = buy_quote.price
        sell_price = sell_quote.price

        spread_pct = (sell_price - buy_price) / buy_price * 100.0

        fees = self._compute_fees(buy_price, sell_price)
        gas = gas_cost if gas_cost is not None else self.estimate_gas_cost("swap")

        profit = self.compute_profit(buy_price, sell_price, 1.0, fees, gas)

        if profit < self.min_profit_usd:
            return None

        expiry = datetime.now(timezone.utc) + timedelta(seconds=_DEFAULT_EXPIRY_SECONDS)

        return ArbOpportunity(
            pair=pair,
            buy_dex=buy_quote.dex_name,
            sell_dex=sell_quote.dex_name,
            buy_price=buy_price,
            sell_price=sell_price,
            spread_pct=spread_pct,
            estimated_profit=profit,
            estimated_gas=gas,
            net_profit=profit,
            expiry_timestamp=expiry,
        )

    def compute_profit(
        self,
        buy_price: float,
        sell_price: float,
        amount: float,
        fees: float,
        gas: float,
    ) -> float:
        """
        Compute net profit for an arbitrage trade.

        Parameters
        ----------
        buy_price : float
            Price at which to buy.
        sell_price : float
            Price at which to sell.
        amount : float
            Trade amount in base currency.
        fees : float
            Total trading fees in USD.
        gas : float
            Gas cost in USD.

        Returns
        -------
        float
            Net profit in USD.
        """
        if buy_price <= 0 or sell_price <= 0 or amount <= 0:
            return 0.0

        buy_cost = buy_price * amount
        sell_revenue = sell_price * amount
        gross_profit = sell_revenue - buy_cost

        return gross_profit - fees - gas

    def compute_min_profitable_amount(
        self,
        spread: float,
        fees: float,
        gas: float,
    ) -> float:
        """
        Compute the minimum trade amount to be profitable.

        Parameters
        ----------
        spread : float
            Price spread as a decimal (e.g. 0.001 for 0.1%).
        fees : float
            Total fees per unit traded.
        gas : float
            Gas cost in USD.

        Returns
        -------
        float
            Minimum amount in base currency.
        """
        if spread <= 0:
            return float("inf")

        net_spread_per_unit = spread - fees
        if net_spread_per_unit <= 0:
            return float("inf")

        return gas / net_spread_per_unit

    def estimate_gas_cost(self, tx_type: str = "swap") -> float:
        """
        Estimate gas cost for a transaction type.

        Parameters
        ----------
        tx_type : str
            Transaction type: "swap", "bundle", "triangular".

        Returns
        -------
        float
            Estimated gas cost in USD.
        """
        base = _DEFAULT_GAS_COSTS.get(self.network, 1.0)

        multipliers = {
            "swap": 1.0,
            "bundle": 1.5,
            "triangular": 2.0,
            "flash_loan": 2.5,
        }

        return base * multipliers.get(tx_type.lower(), 1.0)

    def _compute_fees(self, buy_price: float, sell_price: float) -> float:
        """Compute total fees for a buy-sell pair."""
        avg_price = (buy_price + sell_price) / 2.0
        return avg_price * self.default_fee_bps / 10_000.0 * 2.0


# ---------------------------------------------------------------------------
# Atomic Arb Executor
# ---------------------------------------------------------------------------


class AtomicArbExecutor:
    """
    Builds, simulates, and executes atomic arbitrage bundles.

    Parameters
    ----------
    max_slippage_bps : float
        Maximum acceptable slippage in basis points.
    dry_run : bool
        If True, only simulate (do not execute on-chain).
    """

    def __init__(
        self,
        max_slippage_bps: float = 50.0,
        dry_run: bool = True,
    ) -> None:
        self.max_slippage_bps = max_slippage_bps
        self.dry_run = dry_run
        self._execution_history: List[ExecutionResult] = []
        logger.info(
            "AtomicArbExecutor: max_slippage=%.1fbps dry_run=%s",
            max_slippage_bps,
            dry_run,
        )

    def build_bundle(
        self,
        opportunity: ArbOpportunity,
        amount: float,
    ) -> Bundle:
        """
        Build an atomic bundle of transactions for an opportunity.

        Parameters
        ----------
        opportunity : ArbOpportunity
            The arbitrage opportunity to execute.
        amount : float
            Trade amount in base currency.

        Returns
        -------
        Bundle
            The constructed transaction bundle.
        """
        deadline = int(time.time()) + 60  # 60 second deadline

        transactions = [
            {
                "type": "swap",
                "dex": opportunity.buy_dex,
                "action": "buy",
                "pair": opportunity.pair,
                "amount": amount,
                "min_output": amount * opportunity.buy_price * (1 - self.max_slippage_bps / 10_000.0),
            },
            {
                "type": "swap",
                "dex": opportunity.sell_dex,
                "action": "sell",
                "pair": opportunity.pair,
                "amount": amount,
                "min_output": amount * opportunity.sell_price * (1 - self.max_slippage_bps / 10_000.0),
            },
        ]

        return Bundle(
            opportunity=opportunity,
            amount=amount,
            transactions=transactions,
            expected_profit=opportunity.net_profit * amount,
            max_slippage=self.max_slippage_bps,
            deadline=deadline,
        )

    def simulate_bundle(self, bundle: Bundle) -> SimulationResult:
        """
        Simulate a bundle to estimate profit and risks.

        Parameters
        ----------
        bundle : Bundle
            The bundle to simulate.

        Returns
        -------
        SimulationResult
            Simulation outcome with estimated profit and risks.
        """
        opp = bundle.opportunity

        if opp.expiry_timestamp <= datetime.now(timezone.utc):
            return SimulationResult(
                success=False,
                simulated_profit=0.0,
                simulated_gas=0.0,
                price_impact=0.0,
                failure_reason="Opportunity expired",
            )

        if opp.net_profit <= 0:
            return SimulationResult(
                success=False,
                simulated_profit=0.0,
                simulated_gas=0.0,
                price_impact=0.0,
                failure_reason="Negative expected profit",
            )

        slippage_impact = opp.buy_price * (opp.spread_pct / 100.0) * (self.max_slippage_bps / 10_000.0)
        simulated_profit = opp.net_profit * bundle.amount - slippage_impact * bundle.amount

        gas_cost = opp.estimated_gas * 1.5

        if simulated_profit <= 0:
            return SimulationResult(
                success=False,
                simulated_profit=simulated_profit,
                simulated_gas=gas_cost,
                price_impact=slippage_impact / opp.buy_price if opp.buy_price > 0 else 0.0,
                failure_reason="Simulation shows negative profit after slippage",
            )

        return SimulationResult(
            success=True,
            simulated_profit=simulated_profit,
            simulated_gas=gas_cost,
            price_impact=slippage_impact / opp.buy_price if opp.buy_price > 0 else 0.0,
        )

    def execute_bundle(self, bundle: Bundle) -> ExecutionResult:
        """
        Execute an arbitrage bundle.

        Parameters
        ----------
        bundle : Bundle
            The bundle to execute.

        Returns
        -------
        ExecutionResult
            Execution outcome with transaction hash and actual profit.
        """
        if self.dry_run:
            sim = self.simulate_bundle(bundle)
            result = ExecutionResult(
                success=sim.success,
                tx_hash=None,
                actual_profit=sim.simulated_profit,
                actual_gas=sim.simulated_gas,
                opportunity=bundle.opportunity,
                error=sim.failure_reason,
            )
            self._execution_history.append(result)
            logger.info(
                "Dry-run execution: %s profit=$%.4f",
                "SUCCESS" if sim.success else "FAILED",
                sim.simulated_profit,
            )
            return result

        if int(time.time()) > bundle.deadline:
            result = ExecutionResult(
                success=False,
                tx_hash=None,
                actual_profit=0.0,
                actual_gas=0.0,
                opportunity=bundle.opportunity,
                error="Bundle deadline exceeded",
            )
            self._execution_history.append(result)
            return result

        try:
            tx_hash = self._send_bundle_tx(bundle)
            result = ExecutionResult(
                success=True,
                tx_hash=tx_hash,
                actual_profit=bundle.expected_profit,
                actual_gas=bundle.opportunity.estimated_gas,
                opportunity=bundle.opportunity,
            )
            self._execution_history.append(result)
            logger.info(
                "Bundle executed: tx=%s profit=$%.4f",
                tx_hash,
                bundle.expected_profit,
            )
            return result
        except Exception as e:
            result = ExecutionResult(
                success=False,
                tx_hash=None,
                actual_profit=0.0,
                actual_gas=0.0,
                opportunity=bundle.opportunity,
                error=str(e),
            )
            self._execution_history.append(result)
            logger.error("Bundle execution failed: %s", e)
            return result

    def verify_execution(self, tx_hash: str) -> bool:
        """
        Verify that a transaction was successfully executed on-chain.

        Parameters
        ----------
        tx_hash : str
            Transaction hash to verify.

        Returns
        -------
        bool
            True if the transaction succeeded.
        """
        try:
            receipt = self._get_tx_receipt(tx_hash)
            if receipt is None:
                return False
            return receipt.get("status", 0) == 1
        except Exception as e:
            logger.error("Failed to verify execution %s: %s", tx_hash, e)
            return False

    def get_execution_history(self) -> List[ExecutionResult]:
        """Return recent execution history."""
        return list(self._execution_history)

    def _send_bundle_tx(self, bundle: Bundle) -> str:
        """Send the bundle transaction on-chain. Stub for real implementation."""
        return f"0x{'0' * 64}"

    def _get_tx_receipt(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Get transaction receipt. Stub for real implementation."""
        return {"status": 1, "gasUsed": 150000}


# ---------------------------------------------------------------------------
# Triangular Arb Finder
# ---------------------------------------------------------------------------


class TriangularArbFinder:
    """
    Finds triangular arbitrage paths across DEXs.

    Parameters
    ----------
    min_profit_usd : float
        Minimum net profit to consider a triangular path viable.
    calculator : ArbitrageCalculator, optional
        Calculator for fee and gas estimation.
    """

    def __init__(
        self,
        min_profit_usd: float = 1.0,
        calculator: Optional[ArbitrageCalculator] = None,
    ) -> None:
        self.min_profit_usd = min_profit_usd
        self.calculator = calculator or ArbitrageCalculator()
        logger.info("TriangularArbFinder initialised (min_profit=$%.2f)", min_profit_usd)

    def find_triangular_paths(
        self,
        pairs: List[str],
        quotes: Dict[str, List[DEXQuote]],
    ) -> List[TriangularPath]:
        """
        Find triangular arbitrage paths from a set of pairs and quotes.

        Parameters
        ----------
        pairs : list of str
            Trading pairs (e.g. ["ETH/USDC", "BTC/ETH", "BTC/USDC"]).
        quotes : dict
            Pair → list of DEXQuote.

        Returns
        -------
        list of TriangularPath
            Profitable triangular paths, sorted by net profit descending.
        """
        tokens = self._extract_tokens(pairs)
        if len(tokens) < 3:
            return []

        paths: List[TriangularPath] = []

        for i, token_a in enumerate(tokens):
            for j, token_b in enumerate(tokens):
                if i == j:
                    continue
                for k, token_c in enumerate(tokens):
                    if k == i or k == j:
                        continue

                    pair_ab = self._find_pair(token_a, token_b, pairs)
                    pair_bc = self._find_pair(token_b, token_c, pairs)
                    pair_ca = self._find_pair(token_c, token_a, pairs)

                    if not pair_ab or not pair_bc or not pair_ca:
                        continue

                    quote_ab = self._best_quote_for_direction(pair_ab, token_a, token_b, quotes)
                    quote_bc = self._best_quote_for_direction(pair_bc, token_b, token_c, quotes)
                    quote_ca = self._best_quote_for_direction(pair_ca, token_c, token_a, quotes)

                    if not quote_ab or not quote_bc or not quote_ca:
                        continue

                    path = self._compute_path(token_a, token_b, token_c, quote_ab, quote_bc, quote_ca)
                    if path and path.net_profit >= self.min_profit_usd:
                        paths.append(path)

        paths.sort(key=lambda p: p.net_profit, reverse=True)
        return self._deduplicate_paths(paths)

    def compute_path_profit(self, path: TriangularPath, amount: float) -> float:
        """
        Compute profit for a triangular path at a given amount.

        Parameters
        ----------
        path : TriangularPath
            The triangular path.
        amount : float
            Input amount.

        Returns
        -------
        float
            Net profit in USD.
        """
        if path.input_amount <= 0:
            return 0.0

        scale = amount / path.input_amount
        return path.net_profit * scale

    def find_best_path(self, paths: List[TriangularPath]) -> Optional[TriangularPath]:
        """
        Find the best triangular path by net profit.

        Parameters
        ----------
        paths : list of TriangularPath
            Candidate paths.

        Returns
        -------
        TriangularPath or None
            Best path, or None if no paths provided.
        """
        if not paths:
            return None
        return max(paths, key=lambda p: p.net_profit)

    def _extract_tokens(self, pairs: List[str]) -> List[str]:
        """Extract unique tokens from a list of pairs."""
        tokens = set()
        for pair in pairs:
            parts = pair.upper().split("/")
            if len(parts) == 2:
                tokens.add(parts[0])
                tokens.add(parts[1])
        return sorted(tokens)

    @staticmethod
    def _find_pair(token_a: str, token_b: str, pairs: List[str]) -> Optional[str]:
        """Find a pair connecting two tokens."""
        for pair in pairs:
            parts = pair.upper().split("/")
            if len(parts) == 2:
                if (parts[0] == token_a and parts[1] == token_b) or \
                   (parts[0] == token_b and parts[1] == token_a):
                    return pair
        return None

    @staticmethod
    def _best_quote_for_direction(
        pair: str,
        token_from: str,
        token_to: str,
        quotes: Dict[str, List[DEXQuote]],
    ) -> Optional[DEXQuote]:
        """Get the best quote for a specific direction."""
        pair_quotes = quotes.get(pair.upper(), [])
        if not pair_quotes:
            return None
        return min(pair_quotes, key=lambda q: q.price)

    def _compute_path(
        self,
        token_a: str,
        token_b: str,
        token_c: str,
        quote_ab: DEXQuote,
        quote_bc: DEXQuote,
        quote_ca: DEXQuote,
    ) -> Optional[TriangularPath]:
        """Compute a triangular path from three quotes."""
        test_amount = 1.0

        amount_b = test_amount / quote_ab.price
        amount_c = amount_b / quote_bc.price
        amount_a_out = amount_c * quote_ca.price

        gross_profit = amount_a_out - test_amount

        fees = self.calculator._compute_fees(quote_ab.price, quote_ca.price) * 3.0
        gas = self.calculator.estimate_gas_cost("triangular")
        net_profit = gross_profit - fees - gas

        if net_profit <= 0:
            return None

        return TriangularPath(
            tokens=[token_a, token_b, token_c],
            pairs=[quote_ab.pair, quote_bc.pair, quote_ca.pair],
            dexes=[quote_ab.dex_name, quote_bc.dex_name, quote_ca.dex_name],
            input_amount=test_amount,
            output_amount=amount_a_out,
            gross_profit=gross_profit,
            net_profit=net_profit,
            total_fees=fees,
            total_gas=gas,
        )

    @staticmethod
    def _deduplicate_paths(paths: List[TriangularPath]) -> List[TriangularPath]:
        """Remove duplicate paths (same tokens, different order)."""
        seen = set()
        unique = []
        for path in paths:
            key = tuple(sorted(path.tokens))
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique


# ---------------------------------------------------------------------------
# DEX Arb Engine
# ---------------------------------------------------------------------------


class DEXArbEngine:
    """
    Main cross-DEX arbitrage engine coordinating monitoring, calculation,
    and execution.

    Parameters
    ----------
    min_profit_usd : float
        Minimum net profit to execute an opportunity.
    max_capital_usd : float
        Maximum capital to deploy per opportunity.
    network : str
        Target blockchain network.
    dry_run : bool
        If True, only simulate trades.
    """

    def __init__(
        self,
        min_profit_usd: float = 1.0,
        max_capital_usd: float = 10_000.0,
        network: str = "arbitrum",
        dry_run: bool = True,
    ) -> None:
        self.min_profit_usd = min_profit_usd
        self.max_capital_usd = max_capital_usd
        self.dry_run = dry_run

        self.monitor = DEXPriceMonitor()
        self.calculator = ArbitrageCalculator(
            min_profit_usd=min_profit_usd,
            network=network,
        )
        self.executor = AtomicArbExecutor(dry_run=dry_run)
        self.triangular_finder = TriangularArbFinder(
            min_profit_usd=min_profit_usd,
            calculator=self.calculator,
        )

        self._pairs: List[str] = []
        self._opportunities: List[ArbOpportunity] = []
        self._triangular_paths: List[TriangularPath] = []

        logger.info(
            "DEXArbEngine: min_profit=$%.2f max_capital=$%.0f network=%s dry_run=%s",
            min_profit_usd,
            max_capital_usd,
            network,
            dry_run,
        )

    def add_dex(self, name: str, quote_fn: Callable[[str], Optional[DEXQuote]]) -> None:
        """Register a DEX with the engine."""
        self.monitor.add_dex(name, quote_fn)

    def add_pair(self, pair: str) -> None:
        """Add a trading pair to monitor."""
        pair = pair.upper()
        if pair not in self._pairs:
            self._pairs.append(pair)
            logger.debug("Added pair: %s", pair)

    def scan_opportunities(
        self,
        min_profit_usd: Optional[float] = None,
    ) -> List[ArbOpportunity]:
        """
        Scan all monitored pairs for arbitrage opportunities.

        Parameters
        ----------
        min_profit_usd : float, optional
            Override minimum profit threshold.

        Returns
        -------
        list of ArbOpportunity
            All viable opportunities, sorted by net profit descending.
        """
        threshold = min_profit_usd if min_profit_usd is not None else self.min_profit_usd
        opportunities: List[ArbOpportunity] = []

        for pair in self._pairs:
            quotes = self.monitor.get_quotes(pair)
            if len(quotes) < 2:
                continue

            best_buy = min(quotes, key=lambda q: q.price)
            best_sell = max(quotes, key=lambda q: q.price)

            if best_buy.dex_name == best_sell.dex_name:
                continue

            if best_buy.is_stale() or best_sell.is_stale():
                continue

            opp = self.calculator.compute_arb_opportunity(best_buy, best_sell)
            if opp and opp.net_profit >= threshold:
                opportunities.append(opp)

        triangular = self.triangular_finder.find_triangular_paths(self._pairs, {
            p: self.monitor.get_quotes(p) for p in self._pairs
        })
        self._triangular_paths = triangular

        opportunities.sort(key=lambda o: o.net_profit, reverse=True)
        self._opportunities = opportunities
        return opportunities

    def execute_best_opportunity(self, capital: Optional[float] = None) -> ExecutionResult:
        """
        Find and execute the best available arbitrage opportunity.

        Parameters
        ----------
        capital : float, optional
            Capital to deploy. Defaults to max_capital_usd.

        Returns
        -------
        ExecutionResult
            Result of the execution attempt.
        """
        if not self._opportunities:
            self.scan_opportunities()

        if not self._opportunities:
            return ExecutionResult(
                success=False,
                tx_hash=None,
                actual_profit=0.0,
                actual_gas=0.0,
                opportunity=ArbOpportunity(
                    pair="", buy_dex="", sell_dex="",
                    buy_price=0.0, sell_price=0.0, spread_pct=0.0,
                    estimated_profit=0.0, estimated_gas=0.0,
                    net_profit=0.0, expiry_timestamp=datetime.now(timezone.utc),
                ),
                error="No opportunities found",
            )

        best = self._opportunities[0]
        amount = min(capital or self.max_capital_usd, self.max_capital_usd)

        bundle = self.executor.build_bundle(best, amount)
        sim = self.executor.simulate_bundle(bundle)

        if not sim.success:
            return ExecutionResult(
                success=False,
                tx_hash=None,
                actual_profit=0.0,
                actual_gas=0.0,
                opportunity=best,
                error=sim.failure_reason,
            )

        return self.executor.execute_bundle(bundle)

    async def run_continuous(
        self,
        min_profit: float = 1.0,
        interval: float = 1.0,
    ) -> AsyncIterator[ExecutionResult]:
        """
        Continuously scan and execute arbitrage opportunities.

        Parameters
        ----------
        min_profit : float
            Minimum profit threshold.
        interval : float
            Seconds between scan cycles.

        Yields
        ------
        ExecutionResult
            Result of each execution attempt.
        """
        logger.info(
            "Starting continuous scan: min_profit=$%.2f interval=%.1fs",
            min_profit,
            interval,
        )
        try:
            while True:
                opportunities = self.scan_opportunities(min_profit_usd=min_profit)

                if opportunities:
                    best = opportunities[0]
                    amount = self.max_capital_usd
                    bundle = self.executor.build_bundle(best, amount)
                    sim = self.executor.simulate_bundle(bundle)

                    if sim.success:
                        result = self.executor.execute_bundle(bundle)
                        yield result

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Continuous scan cancelled")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        return {
            "pairs_monitored": len(self._pairs),
            "dexes_connected": self.monitor.get_dex_count(),
            "opportunities_found": len(self._opportunities),
            "triangular_paths": len(self._triangular_paths),
            "execution_history": len(self.executor.get_execution_history()),
            "monitor_stats": self.monitor.get_stats(),
        }
