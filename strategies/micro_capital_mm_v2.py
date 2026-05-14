"""
micro_capital_mm_v2.py — MicroCapitalMM v2: MEXC + BTC Markets extension.

Thin subclass of ``MicroCapitalMM`` (strategies/micro_capital_mm.py) that
adds first-class support for two new zero/negative-fee venues:

  * **MEXC** — 0% spot and futures maker fee.
  * **BTC Markets** — -0.05% maker rebate (exchange pays per fill).

Key changes from v1
-------------------
- Venue selection logic prefers BTC Markets (negative fee) → MEXC (zero fee)
  → Bybit (zero on select pairs) → Kraken / Coinbase (fallback only).
- ``_compute_min_viable_spread`` returns negative bps for BTC Markets,
  reflecting that any quote is inherently profitable.
- ``get_venue_stats()`` exposes per-venue fill counts, rebates earned, and
  zero-fee fill totals.

Configuration
-------------
Use ``MicroMMConfigV2`` (subclass of ``MicroMMConfig``) which adds::

    mexc_enabled: bool = True
    btcmarkets_enabled: bool = True
    btcmarkets_aud_usd_rate: float = 0.62
    prefer_rebate_venues: bool = True

Usage::

    config = MicroMMConfigV2(
        mexc_enabled=True,
        btcmarkets_enabled=True,
        exchanges=["mexc", "btcmarkets", "bybit"],
    )
    mm = MicroCapitalMMV2(config)
    await mm.run({
        "mexc":       mexc_client,
        "btcmarkets": btcm_client,
        "bybit":      bybit_client,
    })
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from strategies.micro_capital_mm import MicroCapitalMM, MicroMMConfig

log = logging.getLogger("argus.micro_mm_v2")

# ---------------------------------------------------------------------------
# Fee constants for new venues
# ---------------------------------------------------------------------------

_MEXC_MAKER_FEE: float = 0.0       # zero
_BTCM_MAKER_FEE: float = -0.0005   # negative rebate
_BYBIT_MAKER_FEE: float = 0.0      # zero on select spot pairs
_KRAKEN_MAKER_FEE: float = 0.0016
_COINBASE_MAKER_FEE: float = 0.0040

# Break-even spreads in basis points (round-trip: fee × 10_000 × 2)
# Negative means already profitable before capturing any spread.
_MIN_VIABLE_SPREAD_BPS: Dict[str, float] = {
    "btcmarkets": -5.0,   # rebate of 5 bps per side (−0.05% × 10_000)
    "mexc":        0.0,   # zero fee → zero break-even
    "bybit":       0.0,   # zero fee on select pairs
    "kraken":     32.0,   # 0.16% × 2 × 10_000 = 32 bps
    "coinbase":   80.0,   # 0.40% × 2 × 10_000 = 80 bps
}

# Venue priority for market-making (lower index = higher preference)
_VENUE_PRIORITY: List[str] = ["btcmarkets", "mexc", "bybit", "kraken", "coinbase"]


# ---------------------------------------------------------------------------
# MicroMMConfigV2
# ---------------------------------------------------------------------------

@dataclass
class MicroMMConfigV2(MicroMMConfig):
    """
    Extended configuration for MicroCapitalMMV2.

    Inherits all fields from ``MicroMMConfig`` and adds MEXC / BTC Markets
    specific settings.

    Attributes
    ----------
    mexc_enabled : bool
        Enable MEXC as a market-making venue.  Default True.
    btcmarkets_enabled : bool
        Enable BTC Markets as a market-making venue.  Default True.
    btcmarkets_aud_usd_rate : float
        AUD→USD conversion rate for BTC Markets price normalisation.
        Default 0.62.
    prefer_rebate_venues : bool
        When True (default), BTC Markets is always preferred over other venues
        if a pair is available there — the maker rebate is free revenue.
    """

    mexc_enabled: bool = True
    btcmarkets_enabled: bool = True
    btcmarkets_aud_usd_rate: float = 0.62
    prefer_rebate_venues: bool = True   # always prefer negative-fee venues

    def __post_init__(self) -> None:
        # Extend the default exchanges list with new venues if enabled,
        # without breaking the base class validation.
        new_venues: List[str] = []
        if self.btcmarkets_enabled and "btcmarkets" not in self.exchanges:
            new_venues.append("btcmarkets")
        if self.mexc_enabled and "mexc" not in self.exchanges:
            new_venues.append("mexc")
        # Prepend new venues so they appear before Bybit in the priority list
        self.exchanges = new_venues + [e for e in self.exchanges if e not in new_venues]


# ---------------------------------------------------------------------------
# MicroCapitalMMV2
# ---------------------------------------------------------------------------

class MicroCapitalMMV2(MicroCapitalMM):
    """
    Extended market maker supporting MEXC (0% fee) and BTC Markets (-0.05% rebate).

    Inherits the full Bybit / Kraken / Coinbase logic from ``MicroCapitalMM``
    and overrides two methods:

    1. ``_select_venue_for_pair`` — priority logic: BTC Markets → MEXC →
       Bybit → Kraken → Coinbase.
    2. ``_compute_min_viable_spread`` — returns -5 bps for BTC Markets,
       0 bps for MEXC / Bybit, positive bps for fee-paying venues.

    The constructor accepts ``MicroMMConfigV2`` (a strict subtype of
    ``MicroMMConfig``), so all existing orchestration code that passes a
    ``MicroMMConfig`` remains compatible via duck typing.

    Parameters
    ----------
    config : MicroMMConfigV2
        Extended configuration.  Passing a plain ``MicroMMConfig`` is allowed
        but will not enable MEXC / BTC Markets features.
    """

    def __init__(self, config: MicroMMConfigV2) -> None:
        super().__init__(config)
        # Track per-venue statistics
        self._venue_fill_counts: Dict[str, int] = {v: 0 for v in _VENUE_PRIORITY}
        self._venue_rebate_earned_usd: float = 0.0   # cumulative BTC Markets rebate
        self._venue_zero_fee_fills: int = 0           # fills on MEXC + Bybit (0% fee)

    # ------------------------------------------------------------------
    # Venue selection (overrides base class)
    # ------------------------------------------------------------------

    def _select_venue_for_pair(
        self,
        symbol: str,
        available_exchanges: List[str],
    ) -> str:
        """
        Select the best venue for a given symbol from the available exchanges.

        Priority
        --------
        1. BTC Markets — negative maker fee (-0.05%): profitable before spread.
        2. MEXC        — zero maker fee (0.00%): all spread is gross profit.
        3. Bybit       — zero fee on select spot pairs.
        4. Kraken      — 0.16% maker; only if no zero-fee venue available.
        5. Coinbase    — 0.40% maker; last resort.

        When ``prefer_rebate_venues=True`` (default in MicroMMConfigV2), BTC
        Markets is always preferred if the pair is available there.

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g. "BTCUSDT", "BTC-AUD").
        available_exchanges : list[str]
            Exchanges that have this pair available (from scanner results).

        Returns
        -------
        str
            Name of the selected exchange.

        Raises
        ------
        ValueError
            If ``available_exchanges`` is empty.
        """
        if not available_exchanges:
            raise ValueError(
                f"_select_venue_for_pair({symbol!r}): no exchanges provided"
            )

        lower_avail = [e.lower() for e in available_exchanges]

        cfg: MicroMMConfigV2 = self.config  # type: ignore[assignment]
        prefer_rebate = getattr(cfg, "prefer_rebate_venues", True)
        btcm_enabled = getattr(cfg, "btcmarkets_enabled", True)
        mexc_enabled = getattr(cfg, "mexc_enabled", True)

        # Walk priority list in order
        for venue in _VENUE_PRIORITY:
            if venue == "btcmarkets" and not (btcm_enabled and prefer_rebate):
                continue
            if venue == "mexc" and not mexc_enabled:
                continue
            if venue in lower_avail:
                log.debug(
                    "_select_venue_for_pair(%s): selected %s", symbol, venue
                )
                return venue

        # Fallback — return whatever was first provided
        log.warning(
            "_select_venue_for_pair(%s): no priority-matched venue found, "
            "using first available: %s",
            symbol,
            available_exchanges[0],
        )
        return available_exchanges[0]

    # ------------------------------------------------------------------
    # Minimum viable spread (overrides base class)
    # ------------------------------------------------------------------

    def _compute_min_viable_spread(self, exchange: str) -> float:
        """
        Return the minimum spread (bps) at which quoting on an exchange is
        worth doing.

        Values
        ------
        - BTC Markets: **-5.0 bps** — rebate generates revenue even at 0 spread;
          any spread is a bonus.
        - MEXC / Bybit: **0.0 bps** — zero fee means break-even is always met.
        - Kraken: **32.0 bps** — round-trip cost of 0.16% × 2 sides.
        - Coinbase: **80.0 bps** — round-trip cost of 0.40% × 2 sides.

        Parameters
        ----------
        exchange : str
            Exchange name (case-insensitive).

        Returns
        -------
        float
            Break-even spread in basis points.  Negative values indicate a
            rebate venue where quoting is profitable at any spread.
        """
        return _MIN_VIABLE_SPREAD_BPS.get(exchange.lower(), 0.0)

    # ------------------------------------------------------------------
    # Record fills (called by execution layer or subclass hooks)
    # ------------------------------------------------------------------

    def _record_fill(
        self,
        exchange: str,
        fill_size_usd: float,
        side: str,
    ) -> None:
        """
        Record a fill event and accumulate per-venue statistics.

        This method is intended to be called by the execution layer (or
        overriding code) after each confirmed fill.

        Parameters
        ----------
        exchange : str
            Exchange where the fill occurred.
        fill_size_usd : float
            USD value of the fill.
        side : str
            "bid" or "ask".
        """
        exch = exchange.lower()
        self._venue_fill_counts[exch] = self._venue_fill_counts.get(exch, 0) + 1

        if exch == "btcmarkets":
            # Rebate: exchange pays abs(BTCM_MAKER_FEE) × fill_size
            rebate = abs(_BTCM_MAKER_FEE) * fill_size_usd
            self._venue_rebate_earned_usd += rebate
            log.debug(
                "_record_fill: btcmarkets rebate +$%.4f (total $%.4f)",
                rebate,
                self._venue_rebate_earned_usd,
            )
        elif exch in ("mexc", "bybit"):
            self._venue_zero_fee_fills += 1

    # ------------------------------------------------------------------
    # Venue statistics
    # ------------------------------------------------------------------

    def get_venue_stats(self) -> Dict[str, Any]:
        """
        Return a breakdown of fills, rebates, and zero-fee activity by venue.

        Returns
        -------
        dict
            Keys:

            ``fills_per_venue`` : dict[str, int]
                Number of confirmed fills per exchange since startup.

            ``btcmarkets_rebate_earned_usd`` : float
                Cumulative USD value of BTC Markets maker rebates received.
                Positive means money earned from the rebate.

            ``zero_fee_fill_count`` : int
                Total fills on MEXC and Bybit combined (0% maker fee venues).

            ``total_fills`` : int
                Grand total fills across all venues.

            ``preferred_venue_pct`` : dict[str, float]
                Fraction of total fills that occurred on each venue.

        Examples
        --------
        >>> stats = mm.get_venue_stats()
        >>> print(stats["btcmarkets_rebate_earned_usd"])
        0.0425
        >>> print(stats["fills_per_venue"])
        {'btcmarkets': 85, 'mexc': 120, 'bybit': 30, 'kraken': 0, 'coinbase': 0}
        """
        total = sum(self._venue_fill_counts.values())
        pct: Dict[str, float] = {}
        for venue, count in self._venue_fill_counts.items():
            pct[venue] = round(count / total, 4) if total > 0 else 0.0

        return {
            "fills_per_venue": dict(self._venue_fill_counts),
            "btcmarkets_rebate_earned_usd": round(
                self._venue_rebate_earned_usd, 6
            ),
            "zero_fee_fill_count": self._venue_zero_fee_fills,
            "total_fills": total,
            "preferred_venue_pct": pct,
        }

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cfg: MicroMMConfigV2 = self.config  # type: ignore[assignment]
        mexc = getattr(cfg, "mexc_enabled", False)
        btcm = getattr(cfg, "btcmarkets_enabled", False)
        return (
            f"MicroCapitalMMV2("
            f"mexc={mexc}, btcmarkets={btcm}, "
            f"capital=${getattr(cfg, 'total_capital_usd', '?'):.0f}, "
            f"max_pairs={getattr(cfg, 'max_pairs', '?')})"
        )
