"""
test_exchange_registry.py — Unit tests for exchanges/exchange_registry.py.

Tests cover:
- All expected exchanges are present in EXCHANGE_REGISTRY.
- MEXC has exactly 0.0 spot maker fee.
- BTC Markets has a negative (< 0) spot maker fee.
- get_zero_fee_exchanges() includes mexc, bybit, and btcmarkets.
- get_rebate_exchanges() returns only btcmarkets.
- min_spread_to_profit() returns correct values per exchange.
- rank_exchanges_for_mm() ranks btcmarkets first (most negative fee).
"""

from __future__ import annotations

import pytest

from exchanges.exchange_registry import (
    EXCHANGE_REGISTRY,
    ExchangeProfile,
    get_aus_regulated,
    get_mm_preferred,
    get_rebate_exchanges,
    get_zero_fee_exchanges,
    min_spread_to_profit,
    rank_exchanges_for_mm,
)

# ---------------------------------------------------------------------------
# Registry presence tests
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """All five expected exchanges must be present in EXCHANGE_REGISTRY."""

    def test_registry_has_all_exchanges(self) -> None:
        """mexc, btcmarkets, bybit, kraken, and coinbase must all be registered."""
        required = {"mexc", "btcmarkets", "bybit", "kraken", "coinbase"}
        registered = set(EXCHANGE_REGISTRY.keys())
        missing = required - registered
        assert not missing, (
            f"Missing exchanges in EXCHANGE_REGISTRY: {sorted(missing)}"
        )

    def test_all_profiles_are_exchange_profile_instances(self) -> None:
        """Every value in EXCHANGE_REGISTRY must be an ExchangeProfile."""
        for name, profile in EXCHANGE_REGISTRY.items():
            assert isinstance(profile, ExchangeProfile), (
                f"EXCHANGE_REGISTRY[{name!r}] is {type(profile)}, expected ExchangeProfile"
            )

    def test_registry_names_match_keys(self) -> None:
        """profile.name must equal the dict key."""
        for key, profile in EXCHANGE_REGISTRY.items():
            assert profile.name == key, (
                f"Key {key!r} → profile.name {profile.name!r} mismatch"
            )


# ---------------------------------------------------------------------------
# Fee value tests
# ---------------------------------------------------------------------------


class TestFeeValues:
    """Verify specific fee values for MEXC and BTC Markets."""

    def test_mexc_zero_spot_maker_fee(self) -> None:
        """MEXC spot maker fee must be exactly 0.0 (no fee whatsoever)."""
        mexc = EXCHANGE_REGISTRY["mexc"]
        assert mexc.spot_maker_fee == 0.0, (
            f"Expected MEXC spot_maker_fee=0.0, got {mexc.spot_maker_fee}"
        )

    def test_mexc_zero_futures_maker_fee(self) -> None:
        """MEXC futures maker fee must be exactly 0.0."""
        mexc = EXCHANGE_REGISTRY["mexc"]
        assert mexc.futures_maker_fee == 0.0, (
            f"Expected MEXC futures_maker_fee=0.0, got {mexc.futures_maker_fee}"
        )

    def test_btcm_negative_spot_maker_fee(self) -> None:
        """BTC Markets spot maker fee must be negative (maker rebate)."""
        btcm = EXCHANGE_REGISTRY["btcmarkets"]
        assert btcm.spot_maker_fee < 0, (
            f"Expected btcmarkets spot_maker_fee < 0, got {btcm.spot_maker_fee}"
        )

    def test_btcm_maker_fee_is_correct_value(self) -> None:
        """BTC Markets maker fee must be -0.0005 (−0.05% rebate)."""
        btcm = EXCHANGE_REGISTRY["btcmarkets"]
        assert btcm.spot_maker_fee == pytest.approx(-0.0005), (
            f"Expected btcmarkets spot_maker_fee=-0.0005, got {btcm.spot_maker_fee}"
        )

    def test_btcm_has_rebate_flag_set(self) -> None:
        """BTC Markets has_maker_rebate must be True."""
        btcm = EXCHANGE_REGISTRY["btcmarkets"]
        assert btcm.has_maker_rebate is True

    def test_mexc_has_no_rebate_flag(self) -> None:
        """MEXC has_maker_rebate must be False (0% fee, not negative)."""
        mexc = EXCHANGE_REGISTRY["mexc"]
        assert mexc.has_maker_rebate is False

    def test_kraken_maker_fee(self) -> None:
        """Kraken spot maker fee must be 0.0016 (0.16%)."""
        kraken = EXCHANGE_REGISTRY["kraken"]
        assert kraken.spot_maker_fee == pytest.approx(0.0016)

    def test_coinbase_maker_fee(self) -> None:
        """Coinbase spot maker fee must be 0.0040 (0.40%)."""
        coinbase = EXCHANGE_REGISTRY["coinbase"]
        assert coinbase.spot_maker_fee == pytest.approx(0.0040)


# ---------------------------------------------------------------------------
# get_zero_fee_exchanges tests
# ---------------------------------------------------------------------------


class TestGetZeroFeeExchanges:
    """get_zero_fee_exchanges() returns exchanges with spot_maker_fee <= 0."""

    def test_includes_mexc(self) -> None:
        """MEXC (0% fee) must appear in zero-fee exchanges."""
        names = [p.name for p in get_zero_fee_exchanges()]
        assert "mexc" in names, f"mexc missing from zero-fee list: {names}"

    def test_includes_bybit(self) -> None:
        """Bybit (0% on select pairs) must appear in zero-fee exchanges."""
        names = [p.name for p in get_zero_fee_exchanges()]
        assert "bybit" in names, f"bybit missing from zero-fee list: {names}"

    def test_includes_btcmarkets(self) -> None:
        """BTC Markets (-0.05% rebate, which is <= 0) must appear."""
        names = [p.name for p in get_zero_fee_exchanges()]
        assert "btcmarkets" in names, (
            f"btcmarkets missing from zero-fee list: {names}"
        )

    def test_excludes_kraken(self) -> None:
        """Kraken (0.16% > 0) must NOT appear."""
        names = [p.name for p in get_zero_fee_exchanges()]
        assert "kraken" not in names, f"kraken should not be in zero-fee list: {names}"

    def test_excludes_coinbase(self) -> None:
        """Coinbase (0.40% > 0) must NOT appear."""
        names = [p.name for p in get_zero_fee_exchanges()]
        assert "coinbase" not in names, (
            f"coinbase should not be in zero-fee list: {names}"
        )

    def test_sorted_ascending(self) -> None:
        """Results must be sorted by spot_maker_fee ascending (negatives first)."""
        profiles = get_zero_fee_exchanges()
        fees = [p.spot_maker_fee for p in profiles]
        assert fees == sorted(fees), (
            f"get_zero_fee_exchanges() not sorted ascending: {fees}"
        )


# ---------------------------------------------------------------------------
# get_rebate_exchanges tests
# ---------------------------------------------------------------------------


class TestGetRebateExchanges:
    """get_rebate_exchanges() returns only exchanges with spot_maker_fee < 0."""

    def test_only_btcmarkets_is_rebate(self) -> None:
        """Only BTC Markets should be returned as a rebate exchange."""
        rebate = get_rebate_exchanges()
        names = [p.name for p in rebate]
        assert names == ["btcmarkets"], (
            f"Expected only ['btcmarkets'] in rebate list, got {names}"
        )

    def test_mexc_not_in_rebate(self) -> None:
        """MEXC (0% fee, not negative) must not be in the rebate list."""
        names = [p.name for p in get_rebate_exchanges()]
        assert "mexc" not in names

    def test_all_rebate_profiles_have_negative_fee(self) -> None:
        """Every returned profile must have spot_maker_fee < 0."""
        for p in get_rebate_exchanges():
            assert p.spot_maker_fee < 0, (
                f"{p.name} returned as rebate exchange but fee={p.spot_maker_fee}"
            )

    def test_all_rebate_profiles_have_flag_set(self) -> None:
        """Every returned profile must have has_maker_rebate=True."""
        for p in get_rebate_exchanges():
            assert p.has_maker_rebate is True, (
                f"{p.name} has_maker_rebate not set to True"
            )


# ---------------------------------------------------------------------------
# min_spread_to_profit tests
# ---------------------------------------------------------------------------


class TestMinSpreadToProfit:
    """min_spread_to_profit() returns correct break-even spreads in bps."""

    def test_min_spread_mexc_is_zero(self) -> None:
        """MEXC has 0% fee → break-even spread is 0.0 bps."""
        result = min_spread_to_profit("mexc")
        assert result == pytest.approx(0.0), (
            f"Expected 0.0 bps for MEXC, got {result}"
        )

    def test_min_spread_btcm_is_negative(self) -> None:
        """BTC Markets has negative rebate → break-even spread is < 0."""
        result = min_spread_to_profit("btcmarkets")
        assert result < 0, (
            f"Expected negative bps for BTC Markets (rebate), got {result}"
        )

    def test_min_spread_btcm_is_minus_five(self) -> None:
        """BTC Markets -0.05% rebate → exactly -5.0 bps break-even."""
        result = min_spread_to_profit("btcmarkets")
        assert result == pytest.approx(-5.0), (
            f"Expected -5.0 bps for BTC Markets, got {result}"
        )

    def test_min_spread_bybit_is_zero(self) -> None:
        """Bybit zero-fee spot pairs → break-even is 0.0 bps."""
        result = min_spread_to_profit("bybit")
        assert result == pytest.approx(0.0), (
            f"Expected 0.0 bps for Bybit, got {result}"
        )

    def test_min_spread_kraken_is_32(self) -> None:
        """Kraken 0.16% × 2 sides = 32 bps break-even."""
        result = min_spread_to_profit("kraken")
        assert result == pytest.approx(32.0), (
            f"Expected 32.0 bps for Kraken, got {result}"
        )

    def test_min_spread_coinbase_is_80(self) -> None:
        """Coinbase 0.40% × 2 sides = 80 bps break-even."""
        result = min_spread_to_profit("coinbase")
        assert result == pytest.approx(80.0), (
            f"Expected 80.0 bps for Coinbase, got {result}"
        )

    def test_unknown_exchange_raises_key_error(self) -> None:
        """Passing an unknown exchange name must raise KeyError."""
        with pytest.raises(KeyError):
            min_spread_to_profit("unknown_exchange_xyz")

    def test_case_insensitive(self) -> None:
        """Exchange name lookups should be case-insensitive."""
        assert min_spread_to_profit("MEXC") == min_spread_to_profit("mexc")
        assert min_spread_to_profit("BTCMarkets") == min_spread_to_profit("btcmarkets")


# ---------------------------------------------------------------------------
# rank_exchanges_for_mm tests
# ---------------------------------------------------------------------------


class TestRankExchangesForMM:
    """rank_exchanges_for_mm() sorts by spot_maker_fee ascending."""

    def test_btcmarkets_is_first(self) -> None:
        """BTC Markets (-0.05% fee) must be ranked first."""
        ranked = rank_exchanges_for_mm()
        assert ranked[0][0] == "btcmarkets", (
            f"Expected btcmarkets first in ranking, got {ranked[0][0]!r}"
        )

    def test_btcmarkets_fee_is_most_negative(self) -> None:
        """BTC Markets fee in the ranking must be negative."""
        ranked = rank_exchanges_for_mm()
        btcm_fee = dict(ranked)["btcmarkets"]
        assert btcm_fee < 0, (
            f"btcmarkets fee should be negative in ranking, got {btcm_fee}"
        )

    def test_kraken_ranked_after_zero_fee_venues(self) -> None:
        """Kraken (0.16%) must appear after all zero/negative-fee venues."""
        ranked = rank_exchanges_for_mm()
        names = [t[0] for t in ranked]
        kraken_idx = names.index("kraken")
        # All zero-fee venues (fee <= 0) should come before Kraken
        for name, fee in ranked[:kraken_idx]:
            assert fee <= 0, (
                f"{name} (fee={fee}) appears before Kraken but is not zero/negative"
            )

    def test_coinbase_is_last(self) -> None:
        """Coinbase (0.40% fee, highest) must be ranked last."""
        ranked = rank_exchanges_for_mm()
        assert ranked[-1][0] == "coinbase", (
            f"Expected coinbase last in ranking, got {ranked[-1][0]!r}"
        )

    def test_ranking_is_sorted_ascending(self) -> None:
        """Fees in the ranking tuple list must be non-decreasing."""
        ranked = rank_exchanges_for_mm()
        fees = [t[1] for t in ranked]
        assert fees == sorted(fees), (
            f"rank_exchanges_for_mm() not sorted ascending: {fees}"
        )

    def test_all_exchanges_present_in_ranking(self) -> None:
        """All five exchanges must appear in the ranking."""
        ranked = rank_exchanges_for_mm()
        names = {t[0] for t in ranked}
        expected = {"mexc", "btcmarkets", "bybit", "kraken", "coinbase"}
        assert names == expected, (
            f"Ranking names mismatch: expected {expected}, got {names}"
        )


# ---------------------------------------------------------------------------
# get_mm_preferred tests
# ---------------------------------------------------------------------------


class TestGetMMPreferred:
    """get_mm_preferred() returns venues flagged as preferred for market-making."""

    def test_mexc_is_mm_preferred(self) -> None:
        names = [p.name for p in get_mm_preferred()]
        assert "mexc" in names

    def test_btcmarkets_is_mm_preferred(self) -> None:
        names = [p.name for p in get_mm_preferred()]
        assert "btcmarkets" in names

    def test_bybit_is_mm_preferred(self) -> None:
        names = [p.name for p in get_mm_preferred()]
        assert "bybit" in names

    def test_kraken_not_mm_preferred(self) -> None:
        names = [p.name for p in get_mm_preferred()]
        assert "kraken" not in names

    def test_coinbase_not_mm_preferred(self) -> None:
        names = [p.name for p in get_mm_preferred()]
        assert "coinbase" not in names


# ---------------------------------------------------------------------------
# get_aus_regulated tests
# ---------------------------------------------------------------------------


class TestGetAUSRegulated:
    """get_aus_regulated() returns only AUSTRAC-registered exchanges."""

    def test_btcmarkets_is_aus_regulated(self) -> None:
        names = [p.name for p in get_aus_regulated()]
        assert "btcmarkets" in names

    def test_mexc_not_aus_regulated(self) -> None:
        names = [p.name for p in get_aus_regulated()]
        assert "mexc" not in names

    def test_bybit_not_aus_regulated(self) -> None:
        names = [p.name for p in get_aus_regulated()]
        assert "bybit" not in names
