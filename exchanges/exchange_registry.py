"""
exchange_registry.py — Central registry of all exchanges and their fee structures.

Maps exchange identifiers to ExchangeProfile dataclasses that capture fee rates,
supported features, regulatory status, and strategic suitability for Argus strategies.

Supports MEXC and BTC Markets in addition to the original Bybit / Kraken / Coinbase.

Key fee facts
-------------
- MEXC spot maker:       0.00%   (zero — any spread is gross profit)
- MEXC futures maker:    0.00%   (zero — unique in the industry)
- BTC Markets maker:    -0.05%   (NEGATIVE — exchange pays you per fill)
- Bybit spot maker:      0.00%   (zero on select pairs)
- Bybit futures maker:   0.01%   (near-zero)
- Kraken maker:          0.16%   (needs ≥32 bps spread to break even)
- Coinbase maker:        0.40%   (needs ≥80 bps spread to break even)

Break-even spread formula (one-sided quote, round-trip):
    bps_needed = maker_fee_rate × 10_000 × 2

For negative maker fees (BTC Markets) the calculation yields a *negative*
break-even — meaning the maker rebate alone generates revenue before any
spread is captured.

Usage::

    from exchanges.exchange_registry import (
        EXCHANGE_REGISTRY,
        get_zero_fee_exchanges,
        get_rebate_exchanges,
        get_mm_preferred,
        get_aus_regulated,
        min_spread_to_profit,
        rank_exchanges_for_mm,
    )

    profile = EXCHANGE_REGISTRY["mexc"]
    print(profile.spot_maker_fee)     # 0.0
    print(profile.has_maker_rebate)   # False

    rebates = get_rebate_exchanges()  # [btcmarkets]
    best = rank_exchanges_for_mm()    # [(btcmarkets, -0.0005), (mexc, 0.0), ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ExchangeProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExchangeProfile:
    """
    Full descriptor for one exchange integration.

    Attributes
    ----------
    name : str
        Short identifier (lowercase, matches registry key).
    spot_maker_fee : float
        Fractional maker fee on spot markets (e.g. 0.001 = 0.1%).
        Negative values indicate a maker rebate (exchange pays you).
    spot_taker_fee : float
        Fractional taker fee on spot markets.
    futures_maker_fee : float | None
        Maker fee on perpetual/futures contracts. None if not offered.
    futures_taker_fee : float | None
        Taker fee on perpetual/futures contracts. None if not offered.
    has_maker_rebate : bool
        True when spot_maker_fee < 0 (maker activity earns revenue).
    min_order_usd : float
        Approximate minimum order value in USD.
    supports_post_only : bool
        Whether the exchange supports post-only order flag.
    supports_amend : bool
        Whether orders can be amended in-place (vs cancel-replace).
    is_aus_regulated : bool
        True if the exchange is registered with AUSTRAC (Australian regulated).
    ws_l2_feed_available : bool
        True if a WebSocket Level-2 order book feed is available.
    preferred_for_mm : bool
        True if maker fee <= 0 — any captured spread is net positive.
    preferred_for_funding_arb : bool
        True if futures funding rate arb is viable on this venue.
    client_class : str
        Name of the Python client class in exchanges/.
    base_currency : str
        Settlement / quote currency ("USDT" or "AUD").
    notes : str
        Human-readable strategic notes.
    """

    name: str
    spot_maker_fee: float
    spot_taker_fee: float
    futures_maker_fee: Optional[float]
    futures_taker_fee: Optional[float]
    has_maker_rebate: bool
    min_order_usd: float
    supports_post_only: bool
    supports_amend: bool
    is_aus_regulated: bool
    ws_l2_feed_available: bool
    preferred_for_mm: bool
    preferred_for_funding_arb: bool
    client_class: str
    base_currency: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry — all supported exchanges
# ---------------------------------------------------------------------------

EXCHANGE_REGISTRY: Dict[str, ExchangeProfile] = {

    # ── MEXC ────────────────────────────────────────────────────────────────
    # Unique proposition: 0% maker fee on BOTH spot and futures.
    # 1000+ spot pairs, many mid-cap altcoins with naturally wide spreads.
    # Not AUSTRAC-registered (offshore), but accessible to AU residents.
    "mexc": ExchangeProfile(
        name="mexc",
        spot_maker_fee=0.0,
        spot_taker_fee=0.0005,
        futures_maker_fee=0.0,
        futures_taker_fee=0.0002,
        has_maker_rebate=False,
        min_order_usd=1.0,
        supports_post_only=True,
        supports_amend=False,
        is_aus_regulated=False,
        ws_l2_feed_available=True,
        preferred_for_mm=True,         # 0% maker → every spread tick is profit
        preferred_for_funding_arb=True, # 0% futures maker is competitive
        client_class="MEXCClient",
        base_currency="USDT",
        notes=(
            "ZERO maker fee spot AND futures. Best venue for micro-capital MM. "
            "1000+ USDT pairs, many mid-cap alts with wide spreads. "
            "Lower liquidity than Bybit on individual pairs → less competition. "
            "MEXC_SPOT_MAKER_FEE=0.0, MEXC_FUTURES_MAKER_FEE=0.0."
        ),
    ),

    # ── BTC Markets ─────────────────────────────────────────────────────────
    # AUSTRAC-registered Australian exchange.
    # -0.05% maker rebate: exchange pays 5bps per fill.
    # AUD-quoted pairs → convert to USD for cross-venue comparison.
    # Less competition on AUD pairs → wider spreads typical.
    "btcmarkets": ExchangeProfile(
        name="btcmarkets",
        spot_maker_fee=-0.0005,        # NEGATIVE — rebate paid to maker
        spot_taker_fee=0.002,
        futures_maker_fee=None,        # no futures market
        futures_taker_fee=None,
        has_maker_rebate=True,         # spot_maker_fee < 0
        min_order_usd=1.0,
        supports_post_only=True,
        supports_amend=False,
        is_aus_regulated=True,         # AUSTRAC registered
        ws_l2_feed_available=True,
        preferred_for_mm=True,         # rebate makes it profitable even at 0 spread
        preferred_for_funding_arb=False,  # spot only
        client_class="BTCMarketsClient",
        base_currency="AUD",
        notes=(
            "AUSTRAC-registered Australian exchange. -0.05% maker REBATE — "
            "exchange pays you per fill. AUD-quoted pairs (BTC-AUD, ETH-AUD, etc.). "
            "Less competitive than USDT venues → wider spreads common. "
            "Ideal for AU-based bots. BTCM_MAKER_FEE=-0.0005."
        ),
    ),

    # ── Bybit ────────────────────────────────────────────────────────────────
    # Primary venue in the original Argus design.
    # Zero maker fee on select spot pairs; near-zero on perps.
    # Good for funding rate arbitrage via perpetual contracts.
    "bybit": ExchangeProfile(
        name="bybit",
        spot_maker_fee=0.001,          # standard; 0.0 on VIP / select pairs
        spot_taker_fee=0.001,
        futures_maker_fee=0.0001,      # 0.01% — near-zero
        futures_taker_fee=0.0006,
        has_maker_rebate=False,
        min_order_usd=1.0,
        supports_post_only=True,
        supports_amend=True,           # supports order amendment
        is_aus_regulated=False,
        ws_l2_feed_available=True,
        preferred_for_mm=True,         # zero fee on select spot pairs
        preferred_for_funding_arb=True,
        client_class="BybitClient",
        base_currency="USDT",
        notes=(
            "Primary original Argus venue. Zero maker fee on select spot pairs. "
            "Best perpetual funding arb venue (0.01% futures maker). "
            "Supports order amendment — efficient for cancel-replace loops. "
            "Large liquidity; BTC/ETH/SOL spreads compressed to 1-2 bps."
        ),
    ),

    # ── Kraken ───────────────────────────────────────────────────────────────
    # Reputable but expensive at micro-capital scale.
    # 0.16% maker → needs ≥32 bps spread to break even round-trip.
    # Disabled by default; only viable on very wide-spread pairs.
    "kraken": ExchangeProfile(
        name="kraken",
        spot_maker_fee=0.0016,
        spot_taker_fee=0.0026,
        futures_maker_fee=0.0002,
        futures_taker_fee=0.0005,
        has_maker_rebate=False,
        min_order_usd=5.0,             # Kraken min varies by asset; $5 conservative
        supports_post_only=True,
        supports_amend=False,
        is_aus_regulated=False,
        ws_l2_feed_available=True,
        preferred_for_mm=False,        # 0.16% too expensive at $1k scale
        preferred_for_funding_arb=False,
        client_class="KrakenClient",
        base_currency="USDT",
        notes=(
            "0.16% maker fee → needs ≥32 bps spread to break even. "
            "Not recommended at $1k scale; disabled by default. "
            "Only enable for very wide-spread pairs (>50 bps). "
            "Reputable, regulatory-compliant exchange."
        ),
    ),

    # ── Coinbase ─────────────────────────────────────────────────────────────
    # Most expensive venue — 0.40% maker.
    # Needs ≥80 bps spread just to break even.
    # Only included for completeness; disabled by default.
    "coinbase": ExchangeProfile(
        name="coinbase",
        spot_maker_fee=0.0040,
        spot_taker_fee=0.0060,
        futures_maker_fee=None,
        futures_taker_fee=None,
        has_maker_rebate=False,
        min_order_usd=1.0,
        supports_post_only=True,
        supports_amend=False,
        is_aus_regulated=False,
        ws_l2_feed_available=True,
        preferred_for_mm=False,        # 0.40% is prohibitive at $1k
        preferred_for_funding_arb=False,
        client_class="CoinbaseClient",
        base_currency="USDT",
        notes=(
            "0.40% maker fee → needs ≥80 bps spread to break even. "
            "Prohibitively expensive at $1k. Disabled by default. "
            "Regulatory-compliant (US-listed company). "
            "Only viable for extremely wide-spread, illiquid pairs."
        ),
    ),

    # ── WOO X ────────────────────────────────────────────────────────────────
    # Zero maker AND taker fee on 90+ spot pairs and 71+ futures pairs.
    # Near-zero cost makes it ideal for high-frequency market making.
    # WOO token holders get even deeper fee discounts.
    "woox": ExchangeProfile(
        name="woox",
        spot_maker_fee=0.0,            # ZERO on eligible pairs (90+ spot)
        spot_taker_fee=0.0,            # ZERO on eligible pairs
        futures_maker_fee=0.0,         # ZERO on eligible pairs (71+ perps)
        futures_taker_fee=0.0,         # ZERO on eligible pairs
        has_maker_rebate=False,
        min_order_usd=1.0,
        supports_post_only=True,       # post_only via order_type=POST_ONLY
        supports_amend=False,
        is_aus_regulated=False,
        ws_l2_feed_available=True,
        preferred_for_mm=True,         # zero fee → every spread tick is profit
        preferred_for_funding_arb=True,  # zero futures maker makes arb viable
        client_class="WOOXClient",
        base_currency="USDT",
        notes=(
            "Zero both sides spot+futures on eligible pairs. "
            "90+ zero-fee spot pairs, 71+ zero-fee perp pairs. "
            "Ideal for MM and funding arb at any capital scale. "
            "WOOX_SPOT_MAKER_FEE=0.0, WOOX_FUTURES_MAKER_FEE=0.0."
        ),
    ),

    # ── dYdX v4 ──────────────────────────────────────────────────────────────
    # Decentralised perps on CosmosSDK — self-custodial, no KYC required.
    # Near-zero maker fee (0.01%); taker 0.05%.
    # Funding arb is viable but requires on-chain transaction signing.
    # No spot market — perpetuals only.
    "dydx": ExchangeProfile(
        name="dydx",
        spot_maker_fee=0.0001,         # 0.01% — perps only (no spot market)
        spot_taker_fee=0.0005,         # 0.05%
        futures_maker_fee=0.0001,      # 0.01% maker
        futures_taker_fee=0.0005,      # 0.05% taker
        has_maker_rebate=False,
        min_order_usd=10.0,            # minimum order size varies by market
        supports_post_only=True,       # TIME_IN_FORCE_POST_ONLY supported
        supports_amend=False,          # cancel-replace required
        is_aus_regulated=False,
        ws_l2_feed_available=True,
        preferred_for_mm=False,        # 0.01% maker; not zero — needs 2 bps spread
        preferred_for_funding_arb=True,  # good funding rates, self-custodial
        client_class="DYDXClient",
        base_currency="USD",
        notes=(
            "Self-custodial perps. No KYC. 0.01% maker near-zero. "
            "Wallet-based auth (Cosmos private key) — not API key. "
            "Perpetuals only — no spot market. "
            "Decentralised: funds never leave wallet. "
            "DYDX_MAKER_FEE=0.0001, DYDX_TAKER_FEE=0.0005."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Minimum spread to profit (bps) — hardcoded strategic constants
# ---------------------------------------------------------------------------

# Break-even one-sided spread in basis points per exchange.
# Formula: round_trip_fee_bps = maker_fee × 10_000 × 2
# For rebates, the value is negative (already in profit before spread).
_MIN_SPREAD_BPS: Dict[str, float] = {
    "mexc":       0.0,    # 0% maker → 0 bps break-even
    "btcmarkets": -5.0,   # -0.05% rebate × 10_000 = -5 bps (already profitable)
    "bybit":      0.0,    # 0% maker on select pairs
    "kraken":     32.0,   # 0.16% × 2 sides × 10_000 bps = 32 bps
    "coinbase":   80.0,   # 0.40% × 2 sides × 10_000 bps = 80 bps
    "woox":       0.0,    # 0% maker AND taker on eligible pairs → 0 bps break-even
    "dydx":       2.0,    # 0.01% maker × 2 sides × 10_000 bps = 2 bps
}


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def get_zero_fee_exchanges() -> List[ExchangeProfile]:
    """
    Return exchanges where spot maker fee is zero or negative (≤ 0).

    These venues make any captured spread net-positive — no fee hurdle.

    Returns
    -------
    list[ExchangeProfile]
        Profiles with spot_maker_fee <= 0.0, sorted by fee ascending.
    """
    results = [
        p for p in EXCHANGE_REGISTRY.values()
        if p.spot_maker_fee <= 0.0
    ]
    return sorted(results, key=lambda p: p.spot_maker_fee)


def get_rebate_exchanges() -> List[ExchangeProfile]:
    """
    Return exchanges that pay the maker a rebate (spot_maker_fee < 0).

    A negative maker fee means posting liquidity earns revenue independent
    of the bid-ask spread captured.

    Returns
    -------
    list[ExchangeProfile]
        Profiles with has_maker_rebate=True.
    """
    return [p for p in EXCHANGE_REGISTRY.values() if p.has_maker_rebate]


def get_mm_preferred() -> List[ExchangeProfile]:
    """
    Return exchanges flagged as preferred for market-making strategies.

    Preference is set when maker_fee <= 0, making any spread profitable.

    Returns
    -------
    list[ExchangeProfile]
        Profiles with preferred_for_mm=True, sorted by maker fee ascending.
    """
    results = [
        p for p in EXCHANGE_REGISTRY.values()
        if p.preferred_for_mm
    ]
    return sorted(results, key=lambda p: p.spot_maker_fee)


def get_aus_regulated() -> List[ExchangeProfile]:
    """
    Return exchanges registered with AUSTRAC (Australian regulated).

    Useful for AU-domiciled operators with compliance requirements.

    Returns
    -------
    list[ExchangeProfile]
        Profiles with is_aus_regulated=True.
    """
    return [p for p in EXCHANGE_REGISTRY.values() if p.is_aus_regulated]


def min_spread_to_profit(exchange_name: str) -> float:
    """
    Return the minimum spread (bps) required to break even on a round-trip.

    For zero-fee venues (MEXC, Bybit) this is 0.0 — any spread is profit.
    For rebate venues (BTC Markets) this is negative — already in profit
    before any spread is captured.
    For fee-paying venues (Kraken, Coinbase) this is a positive hurdle.

    Parameters
    ----------
    exchange_name : str
        Exchange key (case-insensitive).

    Returns
    -------
    float
        Break-even spread in basis points. Negative = rebate venue.

    Raises
    ------
    KeyError
        If exchange_name is not in the registry.

    Examples
    --------
    >>> min_spread_to_profit("mexc")
    0.0
    >>> min_spread_to_profit("btcmarkets")
    -5.0
    >>> min_spread_to_profit("kraken")
    32.0
    """
    key = exchange_name.lower()
    if key not in EXCHANGE_REGISTRY:
        raise KeyError(
            f"Unknown exchange '{exchange_name}'. "
            f"Known exchanges: {list(EXCHANGE_REGISTRY.keys())}"
        )
    return _MIN_SPREAD_BPS.get(key, 0.0)


def rank_exchanges_for_mm() -> List[Tuple[str, float]]:
    """
    Rank all exchanges by attractiveness for market-making.

    Sort order: spot_maker_fee ascending (most negative / smallest first).
    BTC Markets (-0.05%) ranks first, then MEXC/Bybit (0%), then fee payers.

    Returns
    -------
    list[tuple[str, float]]
        List of (exchange_name, spot_maker_fee) tuples, sorted ascending.

    Examples
    --------
    >>> rank_exchanges_for_mm()
    [('btcmarkets', -0.0005), ('mexc', 0.0), ('bybit', 0.001), ...]
    """
    ranked = [
        (name, profile.spot_maker_fee)
        for name, profile in EXCHANGE_REGISTRY.items()
    ]
    return sorted(ranked, key=lambda t: t[1])


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------

def print_registry_summary() -> None:
    """Print a human-readable summary of all registered exchanges."""
    header = (
        f"{'Exchange':<14} {'SpotMaker':>10} {'SpotTaker':>10} "
        f"{'FutMaker':>10} {'Rebate':>7} {'AUS':>5} {'MMPref':>7} "
        f"{'MinSpread':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, p in EXCHANGE_REGISTRY.items():
        fut = f"{p.futures_maker_fee:.4f}" if p.futures_maker_fee is not None else "  N/A"
        ms = _MIN_SPREAD_BPS.get(name, 0.0)
        print(
            f"{name:<14} "
            f"{p.spot_maker_fee:>10.4f} "
            f"{p.spot_taker_fee:>10.4f} "
            f"{fut:>10} "
            f"{'YES' if p.has_maker_rebate else 'no':>7} "
            f"{'YES' if p.is_aus_regulated else 'no':>5} "
            f"{'YES' if p.preferred_for_mm else 'no':>7} "
            f"{ms:>9.1f}bps"
        )


if __name__ == "__main__":
    print_registry_summary()
    print()
    print("MM preferred:", [p.name for p in get_mm_preferred()])
    print("Rebate venues:", [p.name for p in get_rebate_exchanges()])
    print("AUS regulated:", [p.name for p in get_aus_regulated()])
    print("Ranked for MM:", rank_exchanges_for_mm())
