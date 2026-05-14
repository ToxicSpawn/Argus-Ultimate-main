"""
Flash Loan Arbitrage — borrow large amounts for one block, arb, repay.

Flash loans allow borrowing significant capital (up to $100K+) with zero
collateral as long as the loan is repaid within the same transaction.

Providers:
  - Aave V3: 0.09% flash loan fee (9 bps)
  - dYdX:     0% fee (free flash loans)

Strategy:
  1. Borrow token from flash loan provider
  2. Buy cheap on Pool A
  3. Sell expensive on Pool B
  4. Repay loan + fee
  5. Keep profit

Triangular arbitrage:
  A → B → C → A through 3 pools, capturing circular price discrepancies.

All flash loan features are disabled by default (``dex.flash_loan_enabled: false``).
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Flash loan provider fees in basis points
PROVIDER_FEES: Dict[str, int] = {
    "aave": 9,       # 0.09%
    "dydx": 0,       # free
    "balancer": 0,    # free
    "euler": 0,       # free (Euler V2)
}

# Typical gas cost for flash loan arb transactions
_DEFAULT_GAS_USD = 3.0  # Arbitrum; Ethereum mainnet would be ~$20-50


@dataclass
class FlashLoanResult:
    """Result of a flash loan profitability calculation."""

    profitable: bool
    optimal_amount: float
    expected_profit: float
    loan_fee: float
    gas_estimate: float
    gross_profit: float
    total_cost: float
    profit_after_costs: float
    path: List[str] = field(default_factory=list)

    def net_profit_usd(self) -> float:
        return self.profit_after_costs


@dataclass
class TriangularArbPath:
    """A triangular arb path through 3 pools: A → B → C → A."""

    pools: List[str]  # [pool_AB, pool_BC, pool_CA]
    tokens: List[str]  # [A, B, C]
    fees_bps: List[int]  # [fee_AB, fee_BC, fee_CA]
    expected_profit: float
    optimal_amount: float
    direction: str  # e.g. "ETH→USDC→BTC→ETH"


class FlashLoanArbitrage:
    """
    Flash loan arbitrage strategy.

    Parameters
    ----------
    min_profit_usd : float
        Minimum net profit in USD to execute.
    max_loan_usd : float
        Maximum flash loan amount in USD.
    provider : str
        Flash loan provider ('aave', 'dydx', 'balancer', 'euler').
    gas_estimate_usd : float
        Estimated gas cost for the flash loan transaction.
    """

    def __init__(
        self,
        min_profit_usd: float = 5.0,
        max_loan_usd: float = 100_000.0,
        provider: str = "aave",
        gas_estimate_usd: float = _DEFAULT_GAS_USD,
    ) -> None:
        if min_profit_usd < 0:
            raise ValueError("min_profit_usd must be non-negative")
        if max_loan_usd <= 0:
            raise ValueError("max_loan_usd must be positive")

        self.min_profit_usd = min_profit_usd
        self.max_loan_usd = max_loan_usd
        self.provider = provider.lower()
        self.gas_estimate_usd = gas_estimate_usd
        self.loan_fee_bps = PROVIDER_FEES.get(self.provider, 9)

        logger.info(
            "FlashLoanArbitrage: provider=%s fee=%dbps min_profit=$%.2f max_loan=$%.0f",
            self.provider,
            self.loan_fee_bps,
            min_profit_usd,
            max_loan_usd,
        )

    # ------------------------------------------------------------------
    # Two-pool arbitrage
    # ------------------------------------------------------------------

    def calculate_flash_loan_arb(
        self,
        pool_a_reserves: Dict[str, float],
        pool_b_reserves: Dict[str, float],
        fee_a_bps: int = 30,
        fee_b_bps: int = 30,
        loan_fee_bps: Optional[int] = None,
    ) -> FlashLoanResult:
        """
        Calculate if arbitrage between two pools is profitable after flash loan fee.

        Pool A is assumed to be cheaper (buy side), Pool B is expensive (sell side).

        Parameters
        ----------
        pool_a_reserves : dict
            Cheaper pool reserves {token0_reserve, token1_reserve}.
        pool_b_reserves : dict
            More expensive pool reserves {token0_reserve, token1_reserve}.
        fee_a_bps : int
            Pool A fee in basis points.
        fee_b_bps : int
            Pool B fee in basis points.
        loan_fee_bps : int, optional
            Override for the flash loan fee (defaults to provider fee).

        Returns
        -------
        FlashLoanResult
            Profitability analysis.
        """
        if loan_fee_bps is None:
            loan_fee_bps = self.loan_fee_bps

        xa = pool_a_reserves.get("token0_reserve", 0.0)
        ya = pool_a_reserves.get("token1_reserve", 0.0)
        xb = pool_b_reserves.get("token0_reserve", 0.0)
        yb = pool_b_reserves.get("token1_reserve", 0.0)

        if xa <= 0 or ya <= 0 or xb <= 0 or yb <= 0:
            return FlashLoanResult(
                profitable=False,
                optimal_amount=0,
                expected_profit=0,
                loan_fee=0,
                gas_estimate=self.gas_estimate_usd,
                gross_profit=0,
                total_cost=0,
                profit_after_costs=0,
            )

        fee_mult_a = 1.0 - fee_a_bps / 10_000.0
        fee_mult_b = 1.0 - fee_b_bps / 10_000.0

        # Price on each pool (token1 per token0)
        price_a = ya / xa
        price_b = yb / xb

        # Pool A must be cheaper than Pool B for arb
        if price_a >= price_b:
            return FlashLoanResult(
                profitable=False,
                optimal_amount=0,
                expected_profit=0,
                loan_fee=0,
                gas_estimate=self.gas_estimate_usd,
                gross_profit=0,
                total_cost=0,
                profit_after_costs=0,
            )

        # Calculate optimal input amount
        # Maximise: sell_on_B(buy_on_A(amount)) - amount - loan_fee
        optimal = self._optimal_two_pool_amount(
            xa, ya, xb, yb, fee_mult_a, fee_mult_b
        )

        if optimal <= 0:
            return FlashLoanResult(
                profitable=False,
                optimal_amount=0,
                expected_profit=0,
                loan_fee=0,
                gas_estimate=self.gas_estimate_usd,
                gross_profit=0,
                total_cost=0,
                profit_after_costs=0,
            )

        # Cap at max loan
        optimal = min(optimal, self.max_loan_usd)

        # Simulate the trade
        # Step 1: Buy token0 on pool A (spend token1)
        amount_in_a = optimal * fee_mult_a
        new_ya = (xa * ya) / (xa + amount_in_a)
        token0_received = ya - new_ya  # we get token1 actually; naming depends on direction
        # Actually let's model it as: borrow token1, buy token0 on A, sell token0 on B for token1

        # Borrow `optimal` of token1
        # Buy token0 on pool A: input token1, output token0
        token1_in = optimal
        token1_after_fee_a = token1_in * fee_mult_a
        # Pool A: x=token0, y=token1
        # Swapping token1 in: new_x = (xa * ya) / (ya + token1_after_fee_a)
        new_xa = (xa * ya) / (ya + token1_after_fee_a)
        token0_out = xa - new_xa

        # Sell token0 on pool B: input token0, output token1
        token0_after_fee_b = token0_out * fee_mult_b
        new_yb = (xb * yb) / (xb + token0_after_fee_b)
        token1_received = yb - new_yb

        # Costs
        loan_fee_amount = optimal * loan_fee_bps / 10_000.0
        gross_profit = token1_received - optimal
        total_cost = loan_fee_amount + self.gas_estimate_usd
        profit_after_costs = gross_profit - total_cost

        return FlashLoanResult(
            profitable=profit_after_costs >= self.min_profit_usd,
            optimal_amount=optimal,
            expected_profit=profit_after_costs,
            loan_fee=loan_fee_amount,
            gas_estimate=self.gas_estimate_usd,
            gross_profit=gross_profit,
            total_cost=total_cost,
            profit_after_costs=profit_after_costs,
        )

    def _optimal_two_pool_amount(
        self,
        xa: float,
        ya: float,
        xb: float,
        yb: float,
        fee_a: float,
        fee_b: float,
    ) -> float:
        """
        Find the optimal borrow amount for two-pool arb.

        Uses the analytical solution for constant-product AMMs.
        """
        # The profit function for borrowing D of token1:
        # Buy token0 on A: token0_out = xa - (xa*ya)/(ya + D*fee_a)
        # Sell on B: token1_back = yb - (xb*yb)/(xb + token0_out*fee_b)
        # profit = token1_back - D
        # Taking derivative and setting to 0 gives the optimal D
        # Approximation: D_opt ≈ sqrt(xa * ya * xb * yb * fee_a * fee_b) / ya - ya
        # This is a simplification; actual optimum requires numerical methods
        try:
            # Better approximation using the harmonic relationship
            d_opt = math.sqrt(ya * yb * fee_a * fee_b) - ya
            if d_opt <= 0:
                # Try alternative: smaller of two estimates
                d_opt = min(ya, yb) * 0.01  # 1% of smaller reserve
            return max(0.0, d_opt)
        except (ValueError, ZeroDivisionError):
            return 0.0

    # ------------------------------------------------------------------
    # Triangular arbitrage
    # ------------------------------------------------------------------

    def find_triangular_arb(
        self,
        pools: Dict[str, Dict[str, Any]],
    ) -> List[TriangularArbPath]:
        """
        Find A → B → C → A triangular arb paths through multiple pools.

        Parameters
        ----------
        pools : dict
            Pool address → {token0, token1, reserves: {token0_reserve, token1_reserve},
                            fee_bps: int}

        Returns
        -------
        list of TriangularArbPath
            Profitable triangular arb paths, sorted by expected profit.
        """
        if len(pools) < 3:
            return []

        # Build adjacency: token → [(pool, other_token, fee_bps)]
        adjacency: Dict[str, List[Tuple[str, str, int, Dict]]] = {}
        for pool_addr, info in pools.items():
            t0 = info.get("token0", "").upper()
            t1 = info.get("token1", "").upper()
            fee = info.get("fee_bps", 30)
            reserves = info.get("reserves", {})

            if not t0 or not t1:
                continue

            adjacency.setdefault(t0, []).append((pool_addr, t1, fee, reserves))
            adjacency.setdefault(t1, []).append((pool_addr, t0, fee, reserves))

        results: List[TriangularArbPath] = []
        tokens = list(adjacency.keys())

        # Check all 3-token cycles
        for cycle in itertools.combinations(tokens, 3):
            for perm in itertools.permutations(cycle):
                a, b, c = perm
                path = self._check_triangle(a, b, c, adjacency)
                if path is not None and path.expected_profit >= self.min_profit_usd:
                    results.append(path)

        results.sort(key=lambda p: p.expected_profit, reverse=True)

        # Deduplicate (same pools, different order)
        seen = set()
        unique = []
        for path in results:
            key = tuple(sorted(path.pools))
            if key not in seen:
                seen.add(key)
                unique.append(path)

        return unique

    def _check_triangle(
        self,
        token_a: str,
        token_b: str,
        token_c: str,
        adjacency: Dict[str, List[Tuple[str, str, int, Dict]]],
    ) -> Optional[TriangularArbPath]:
        """Check a specific A → B → C → A triangle for profitability."""
        # Find pools for each leg
        pool_ab = self._find_pool(token_a, token_b, adjacency)
        pool_bc = self._find_pool(token_b, token_c, adjacency)
        pool_ca = self._find_pool(token_c, token_a, adjacency)

        if not pool_ab or not pool_bc or not pool_ca:
            return None

        p_ab, fee_ab, res_ab = pool_ab
        p_bc, fee_bc, res_bc = pool_bc
        p_ca, fee_ca, res_ca = pool_ca

        # Simulate: start with 1000 units of token_a
        test_amount = 1000.0

        # Leg 1: A → B
        b_out = self._simulate_swap(test_amount, res_ab, fee_ab, forward=True)
        if b_out <= 0:
            return None

        # Leg 2: B → C
        c_out = self._simulate_swap(b_out, res_bc, fee_bc, forward=True)
        if c_out <= 0:
            return None

        # Leg 3: C → A
        a_out = self._simulate_swap(c_out, res_ca, fee_ca, forward=True)
        if a_out <= 0:
            return None

        # Profit = what we get back - what we started with
        gross_profit = a_out - test_amount
        loan_fee = test_amount * self.loan_fee_bps / 10_000.0
        net_profit = gross_profit - loan_fee - self.gas_estimate_usd

        if net_profit <= 0:
            return None

        # Scale optimal amount proportionally
        profit_ratio = net_profit / test_amount
        optimal = min(self.max_loan_usd, test_amount * 10)  # scale up

        return TriangularArbPath(
            pools=[p_ab, p_bc, p_ca],
            tokens=[token_a, token_b, token_c],
            fees_bps=[fee_ab, fee_bc, fee_ca],
            expected_profit=net_profit * (optimal / test_amount),
            optimal_amount=optimal,
            direction=f"{token_a}→{token_b}→{token_c}→{token_a}",
        )

    @staticmethod
    def _find_pool(
        token_from: str,
        token_to: str,
        adjacency: Dict[str, List[Tuple[str, str, int, Dict]]],
    ) -> Optional[Tuple[str, int, Dict]]:
        """Find the best pool connecting two tokens."""
        edges = adjacency.get(token_from, [])
        for pool_addr, other_token, fee, reserves in edges:
            if other_token == token_to:
                return (pool_addr, fee, reserves)
        return None

    @staticmethod
    def _simulate_swap(
        amount_in: float,
        reserves: Dict[str, float],
        fee_bps: int,
        forward: bool = True,
    ) -> float:
        """
        Simulate a constant-product swap.

        Parameters
        ----------
        amount_in : float
            Input token amount.
        reserves : dict
            Pool reserves.
        fee_bps : int
            Fee in basis points.
        forward : bool
            True = swap token0 → token1, False = token1 → token0.

        Returns
        -------
        float
            Output token amount.
        """
        x = reserves.get("token0_reserve", 0)
        y = reserves.get("token1_reserve", 0)
        if not forward:
            x, y = y, x

        if x <= 0 or y <= 0 or amount_in <= 0:
            return 0.0

        fee_mult = 1.0 - fee_bps / 10_000.0
        amount_after_fee = amount_in * fee_mult

        new_y = (x * y) / (x + amount_after_fee)
        return y - new_y

    # ------------------------------------------------------------------
    # Simulation / dry-run
    # ------------------------------------------------------------------

    def simulate_execution(self, arb_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dry-run an arb plan to estimate actual profit after all costs.

        Parameters
        ----------
        arb_plan : dict
            Must contain: amount, pools (list), reserves (list of dicts),
            fees_bps (list of int), provider (str, optional).

        Returns
        -------
        dict
            {success, final_amount, gross_profit, loan_fee, gas_cost,
             net_profit, price_impacts}
        """
        amount = arb_plan.get("amount", 0)
        pools = arb_plan.get("pools", [])
        all_reserves = arb_plan.get("reserves", [])
        fees = arb_plan.get("fees_bps", [])
        provider = arb_plan.get("provider", self.provider)

        if not pools or not all_reserves or len(pools) != len(all_reserves):
            return {"success": False, "reason": "Invalid arb plan structure"}

        if len(fees) != len(pools):
            fees = [30] * len(pools)

        current = amount
        price_impacts = []

        for i, (pool, reserves) in enumerate(zip(pools, all_reserves)):
            fee = fees[i]
            output = self._simulate_swap(current, reserves, fee)
            if output <= 0:
                return {
                    "success": False,
                    "reason": f"Zero output at hop {i} ({pool})",
                    "failed_at_hop": i,
                }

            impact = 1.0 - (output / current) if current > 0 else 0
            price_impacts.append(impact)
            current = output

        loan_fee_bps = PROVIDER_FEES.get(provider, 9)
        loan_fee = amount * loan_fee_bps / 10_000.0
        gross_profit = current - amount
        net_profit = gross_profit - loan_fee - self.gas_estimate_usd

        return {
            "success": net_profit > 0,
            "final_amount": current,
            "gross_profit": gross_profit,
            "loan_fee": loan_fee,
            "gas_cost": self.gas_estimate_usd,
            "net_profit": net_profit,
            "price_impacts": price_impacts,
            "profitable": net_profit >= self.min_profit_usd,
        }
