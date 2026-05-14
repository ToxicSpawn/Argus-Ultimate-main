"""Push 66 — Funding Rate Arbitrage strategy.

Market-neutral strategy: long spot + short perpetual.
Collects 8-hourly funding payments when funding rate is positive.

Expected return: 10-40% APY in neutral/trending crypto markets.
Risk: near-zero directional exposure; basis risk only.

Reference: Sharpe.ai (2025), BitMEX Q3 2025 derivatives report
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class FundingArbitrageState(str, Enum):
    FLAT = "FLAT"
    ENTERED = "ENTERED"
    EXITING = "EXITING"


@dataclass
class FundingArbitrageSignal:
    symbol: str
    action: str              # "ENTER" | "EXIT" | "HOLD"
    spot_side: str           # "buy"
    perp_side: str           # "sell"
    funding_rate: float      # current 8h funding rate
    annualised_apy: float    # funding_rate * 3 * 365
    min_apy_threshold: float
    reason: str


class FundingArbStrategy:
    """Funding rate arbitrage: long spot + short perpetual.

    Args:
        min_apy:          Minimum annualised funding yield to enter (e.g. 0.10 = 10%)
        exit_apy:         Exit when APY drops below this (e.g. 0.05 = 5%)
        max_basis_pct:    Max acceptable spot-perp basis spread to enter
        funding_interval: Hours between funding payments (Bybit = 8h)
    """

    name = "FundingArbStrategy"

    def __init__(
        self,
        min_apy: float = 0.10,
        exit_apy: float = 0.05,
        max_basis_pct: float = 0.003,
        funding_interval: int = 8,
    ):
        self.min_apy = min_apy
        self.exit_apy = exit_apy
        self.max_basis_pct = max_basis_pct
        self.funding_interval = funding_interval
        self._state: Dict[str, FundingArbitrageState] = {}
        self._total_funding_collected: float = 0.0

    def on_funding_tick(
        self,
        symbol: str,
        funding_rate: float,   # 8h funding rate e.g. 0.001 = 0.1%
        spot_price: float,
        perp_price: float,
    ) -> FundingArbitrageSignal:
        """Called every funding period. Returns action signal."""
        payments_per_year = (24 / self.funding_interval) * 365
        apy = funding_rate * payments_per_year
        basis_pct = abs(perp_price - spot_price) / spot_price
        state = self._state.get(symbol, FundingArbitrageState.FLAT)

        if state == FundingArbitrageState.FLAT:
            if apy >= self.min_apy and basis_pct <= self.max_basis_pct:
                self._state[symbol] = FundingArbitrageState.ENTERED
                return FundingArbitrageSignal(
                    symbol=symbol, action="ENTER",
                    spot_side="buy", perp_side="sell",
                    funding_rate=funding_rate, annualised_apy=apy,
                    min_apy_threshold=self.min_apy,
                    reason=f"APY={apy:.1%} > threshold={self.min_apy:.1%}",
                )

        elif state == FundingArbitrageState.ENTERED:
            self._total_funding_collected += funding_rate
            if apy < self.exit_apy or funding_rate < 0:
                self._state[symbol] = FundingArbitrageState.FLAT
                return FundingArbitrageSignal(
                    symbol=symbol, action="EXIT",
                    spot_side="sell", perp_side="buy",
                    funding_rate=funding_rate, annualised_apy=apy,
                    min_apy_threshold=self.min_apy,
                    reason=f"APY={apy:.1%} fell below exit={self.exit_apy:.1%}",
                )

        return FundingArbitrageSignal(
            symbol=symbol, action="HOLD",
            spot_side="hold", perp_side="hold",
            funding_rate=funding_rate, annualised_apy=apy,
            min_apy_threshold=self.min_apy,
            reason="Holding position",
        )

    @property
    def total_funding_collected(self) -> float:
        return self._total_funding_collected

    def reset(self, symbol: str) -> None:
        self._state[symbol] = FundingArbitrageState.FLAT
