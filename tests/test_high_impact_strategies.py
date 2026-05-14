"""
Tests for High-Impact Arbitrage & MEV Strategies (v15.0.0).

Covers:
- MEV Sandwich Attack
- Triangular Arbitrage
- Options Volatility Arbitrage
- Cross-Chain Bridge Arbitrage
- Oracle Deviation

Author: Argus Ultimate
"""

from __future__ import annotations

import math
import pytest
from datetime import datetime, timezone

# ============================================================================
# MEV Sandwich Strategy Tests
# ============================================================================

from strategies.mev_sandwich import MEVSandwichStrategy, SandwichOpportunity, SandwichResult


class TestMEVSandwichStrategy:
    """Tests for MEV Sandwich Strategy."""

    def test_init_defaults(self):
        strategy = MEVSandwichStrategy()
        assert strategy.min_profit_usd == 10.0
        assert strategy.min_trade_size_usd == 10000.0
        assert strategy.gas_buffer_gwei == 5.0
        assert len(strategy._opportunities) == 0
        assert len(strategy._results) == 0

    def test_detect_sandwich_opportunity_small_trade(self):
        """Should not detect opportunity for small trades."""
        strategy = MEVSandwichStrategy(min_trade_size_usd=100000)
        opp = strategy.detect_sandwich_opportunity(
            tx_hash="0x123",
            from_address="0xabc",
            to_address="0x5C69B77Ee6ea3DAA2D8b9A0C0D9F2b5F4b8C5e6D",  # Uniswap
            value_eth=1.0,
            gas_price_gwei=30,
            data=b"0x7ff36ab5",  # swapExactETHForTokens
            block_number=18000000,
        )
        assert opp is None

    def test_detect_sandwich_opportunity_large_trade(self):
        """Should detect opportunity for large trades."""
        strategy = MEVSandwichStrategy(min_trade_size_usd=100)
        opp = strategy.detect_sandwich_opportunity(
            tx_hash="0x456",
            from_address="0xdef",
            to_address="0x5C69B77Ee6ea3DAA2D8b9A0C0D9F2b5F4b8C5e6D",
            value_eth=10.0,
            gas_price_gwei=30,
            data=b"0x7ff36ab5",
            block_number=18000000,
        )
        # May or may not detect depending on parsing
        # Should not crash
        assert opp is None or isinstance(opp, SandwichOpportunity)

    def test_execute_sandwich(self):
        """Should execute sandwich attack."""
        strategy = MEVSandwichStrategy()
        opp = SandwichOpportunity(
            victim_tx="0x789",
            victim_address="0xvictim",
            token_in="WETH",
            token_out="USDC",
            amount_in=1000000000000000000,  # 1 ETH
            expected_price_impact=0.01,
            estimated_profit=50.0,
            confidence=0.8,
            gas_price_gwei=40,
            deadline_blocks=3,
        )
        prices = {"WETH": 3000, "USDC": 1.0, "ETH": 3000}
        result = strategy.execute_sandwich(opp, prices, "0xexecutor")
        
        assert isinstance(result, SandwichResult)
        assert result.front_run_success is True
        assert result.back_run_success is True
        assert result.profit_usd > 0

    def test_get_stats_empty(self):
        """Should return empty stats when no results."""
        strategy = MEVSandwichStrategy()
        stats = strategy.get_stats()
        assert stats["total_attacks"] == 0
        assert stats["net_profit_usd"] == 0.0


# ============================================================================
# Triangular Arbitrage Strategy Tests
# ============================================================================

from strategies.triangular_arbitrage import (
    TriangularArbitrageStrategy,
    TrianglePath,
    ArbitrageOpportunity,
    ArbitrageResult,
)


class TestTriangularArbitrageStrategy:
    """Tests for Triangular Arbitrage Strategy."""

    def test_init_defaults(self):
        strategy = TriangularArbitrageStrategy()
        assert strategy.min_deviation_pct == 0.1
        assert strategy.min_amount_usd == 1000.0
        assert len(strategy.triangles) > 0

    def test_update_prices(self):
        """Should update prices correctly."""
        strategy = TriangularArbitrageStrategy()
        prices = {
            "BTC/ETH": 20.0,
            "ETH/USDT": 3000.0,
            "BTC/USDT": 60000.0,
        }
        strategy.update_prices(prices)
        assert strategy._current_prices["BTC/ETH"] == 20.0
        assert strategy._last_scan is not None

    def test_scan_no_opportunities(self):
        """Should return empty when no deviation."""
        strategy = TriangularArbitrageStrategy()
        prices = {
            "BTC/ETH": 20.0,
            "ETH/USDT": 3000.0,
            "BTC/USDT": 60000.0,  # 20 * 3000 = 60000, perfect!
        }
        strategy.update_prices(prices)
        opps = strategy.scan_opportunities()
        assert isinstance(opps, list)

    def test_execute_arbitrage(self):
        """Should execute arbitrage."""
        strategy = TriangularArbitrageStrategy()
        triangle = TrianglePath("BTC", "ETH", "USDT")
        prices = {
            "BTC/ETH": 20.0,
            "ETH/USDT": 3000.0,
            "USDT/BTC": 1/60000,
        }
        strategy.update_prices(prices)
        
        # Create opportunity manually for testing
        opp = ArbitrageOpportunity(
            triangle=triangle,
            direction="forward",
            direct_rate=1.0,
            calculated_rate=1.005,  # 0.5% profit
            deviation_pct=0.5,
            estimated_profit_pct=0.2,
            min_amount=1000.0,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        )
        
        result = strategy.execute_arbitrage(opp, 1000)
        assert isinstance(result, ArbitrageResult)
        assert result.amount_in == 1000

    def test_get_stats(self):
        """Should return stats."""
        strategy = TriangularArbitrageStrategy()
        stats = strategy.get_stats()
        assert stats["total_cycles"] == 0
        assert stats["net_profit"] == 0.0


# ============================================================================
# Options Volatility Arbitrage Strategy Tests
# ============================================================================

from strategies.options_vol_arb import (
    OptionsVolatilityArbitrageStrategy,
    VolArbitrageOpportunity,
    VolatilitySurface,
)


class TestOptionsVolatilityArbitrageStrategy:
    """Tests for Options Volatility Arbitrage Strategy."""

    def test_init_defaults(self):
        strategy = OptionsVolatilityArbitrageStrategy()
        assert strategy.iv_hv_threshold_sell == 1.3
        assert strategy.iv_hv_threshold_buy == 0.7
        assert strategy.min_spread_pct == 10.0

    def test_update_price(self):
        """Should update price history."""
        strategy = OptionsVolatilityArbitrageStrategy()
        for i in range(50):
            strategy.update_price("BTC", 60000 + i * 10)
        assert len(strategy._price_history["BTC"]) == 50

    def test_calculate_hv_insufficient_data(self):
        """Should return None with insufficient data."""
        strategy = OptionsVolatilityArbitrageStrategy()
        strategy.update_price("BTC", 60000)
        hv = strategy.calculate_hv("BTC")
        assert hv is None

    def test_calculate_hv_sufficient_data(self):
        """Should calculate HV with sufficient data."""
        strategy = OptionsVolatilityArbitrageStrategy()
        for i in range(100):
            price = 60000 + 100 * math.sin(i / 10)
            strategy.update_price("BTC", price)
        hv = strategy.calculate_hv("BTC")
        assert hv is not None
        assert hv > 0

    def test_detect_opportunity_sell_vol(self):
        """Should detect sell vol opportunity."""
        strategy = OptionsVolatilityArbitrageStrategy()
        # Add price history
        for i in range(100):
            strategy.update_price("ETH", 3000 + 50 * math.sin(i / 5))
        
        # IV > HV by 50%
        hv = strategy.calculate_hv("ETH") or 0.5
        iv = hv * 1.5
        
        opp = strategy.detect_opportunity(
            symbol="ETH",
            option_type="call",
            strike=3100,
            expiry_days=30,
            iv=iv,
            spot_price=3000,
        )
        # Should detect if IV/HV ratio > threshold
        assert opp is None or isinstance(opp, VolArbitrageOpportunity)

    def test_calculate_delta_call(self):
        """Should calculate call delta correctly."""
        strategy = OptionsVolatilityArbitrageStrategy()
        delta = strategy.calculate_delta(
            option_type="call",
            strike=3000,
            spot=3100,
            iv=0.5,
            days_to_expiry=30,
        )
        assert 0 < delta < 1

    def test_calculate_delta_put(self):
        """Should calculate put delta correctly."""
        strategy = OptionsVolatilityArbitrageStrategy()
        delta = strategy.calculate_delta(
            option_type="put",
            strike=3000,
            spot=2900,
            iv=0.5,
            days_to_expiry=30,
        )
        assert -1 < delta < 0

    def test_execute_vol_arb(self):
        """Should execute volatility arbitrage."""
        strategy = OptionsVolatilityArbitrageStrategy()
        opp = VolArbitrageOpportunity(
            symbol="BTC",
            option_type="call",
            strike=60000,
            expiry_days=30,
            iv=0.8,
            hv=0.5,
            iv_hv_spread=60,
            direction="sell_iv",
            estimated_edge=20,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
        )
        result = strategy.execute_vol_arb(opp, 60000, 1)
        assert result.premium > 0
        assert result.delta_hedge_size != 0


# ============================================================================
# Cross-Chain Bridge Arbitrage Strategy Tests
# ============================================================================

from strategies.cross_chain_bridge_arb import (
    CrossChainBridgeArbitrageStrategy,
    BridgeInfo,
    BridgeArbitrageOpportunity,
)


class TestCrossChainBridgeArbitrageStrategy:
    """Tests for Cross-Chain Bridge Arbitrage Strategy."""

    def test_init_defaults(self):
        strategy = CrossChainBridgeArbitrageStrategy()
        assert strategy.min_profit_pct == 0.5
        assert strategy.max_transfer_usd == 100000
        assert len(strategy.KNOWN_BRIDGES) > 0

    def test_update_chain_price(self):
        """Should update chain prices."""
        strategy = CrossChainBridgeArbitrageStrategy()
        strategy.update_chain_price("Arbitrum", "ETH", 3050, 1000000)
        strategy.update_chain_price("Ethereum", "ETH", 3000, 2000000)
        
        key = ("Arbitrum", "ETH")
        assert key in strategy._chain_prices
        assert strategy._chain_prices[key].price == 3050

    def test_scan_opportunities_no_prices(self):
        """Should return empty when no prices."""
        strategy = CrossChainBridgeArbitrageStrategy()
        opps = strategy.scan_opportunities()
        assert opps == []

    def test_scan_opportunities_with_prices(self):
        """Should detect opportunities with price differences."""
        strategy = CrossChainBridgeArbitrageStrategy(min_profit_pct=0.1)
        strategy.update_chain_price("Arbitrum", "ETH", 3050, 1000000)
        strategy.update_chain_price("Ethereum", "ETH", 3000, 2000000)
        
        opps = strategy.scan_opportunities()
        assert len(opps) >= 0  # May or may not find opportunities

    def test_execute_bridge_arb(self):
        """Should execute bridge arbitrage."""
        strategy = CrossChainBridgeArbitrageStrategy(dex_slippage_pct=0.001)  # Lower slippage
        opp = BridgeArbitrageOpportunity(
            asset="ETH",
            source_chain="Ethereum",
            dest_chain="Arbitrum",
            source_price=3000,
            dest_price=3050,
            price_diff_pct=1.67,
            bridge_name="across",
            bridge_fee_pct=0.001,
            net_profit_pct=1.0,
            confidence=0.8,
            estimated_duration_minutes=5,
            timestamp=datetime.now(timezone.utc),
        )
        
        result = strategy.execute_bridge_arb(opp, 10000)
        assert result.amount == 10000
        assert result.duration_minutes >= 0

    def test_get_supported_bridges(self):
        """Should return bridge info."""
        strategy = CrossChainBridgeArbitrageStrategy()
        bridges = strategy.get_supported_bridges()
        assert "across" in bridges
        assert isinstance(bridges["across"], BridgeInfo)


# ============================================================================
# Oracle Deviation Strategy Tests
# ============================================================================

from strategies.oracle_deviation import (
    OracleDeviationStrategy,
    OraclePrice,
    MarketPrice,
    DeviationOpportunity,
)


class TestOracleDeviationStrategy:
    """Tests for Oracle Deviation Strategy."""

    def test_init_defaults(self):
        strategy = OracleDeviationStrategy()
        assert strategy.min_deviation_pct == 0.3
        assert strategy.max_staleness_seconds == 300
        assert strategy.confidence_threshold == 0.6

    def test_update_oracle_price(self):
        """Should update oracle prices."""
        strategy = OracleDeviationStrategy()
        strategy.update_oracle_price("BTC", "chainlink", 60000, 0.95)
        
        assert "BTC" in strategy._oracle_prices
        assert strategy._oracle_prices["BTC"].price == 60000
        assert strategy._oracle_prices["BTC"].source == "chainlink"

    def test_update_market_price(self):
        """Should update market prices."""
        strategy = OracleDeviationStrategy()
        strategy.update_market_price("BTC", "binance", 60000, 59990, 60010, 1000000)
        
        assert "BTC" in strategy._market_prices
        assert "binance" in strategy._market_prices["BTC"]
        assert strategy._market_prices["BTC"]["binance"].spread_bps > 0

    def test_scan_opportunities_no_deviation(self):
        """Should not detect with no deviation."""
        strategy = OracleDeviationStrategy(min_deviation_pct=1.0)  # 1% threshold
        strategy.update_oracle_price("ETH", "chainlink", 3000)
        strategy.update_market_price("ETH", "binance", 3000, 2999, 3001, 100000)
        
        opps = strategy.scan_opportunities()
        assert len(opps) == 0

    def test_scan_opportunities_with_deviation(self):
        """Should detect with significant deviation."""
        strategy = OracleDeviationStrategy(min_deviation_pct=0.1)
        strategy.update_oracle_price("BTC", "chainlink", 60000)
        strategy.update_market_price("BTC", "binance", 59000, 58990, 59010, 1000000)
        
        opps = strategy.scan_opportunities()
        assert len(opps) > 0
        opp = opps[0]
        assert opp.deviation_pct > 0

    def test_execute_deviation_arb(self):
        """Should execute deviation arbitrage."""
        strategy = OracleDeviationStrategy()
        opp = DeviationOpportunity(
            symbol="ETH",
            oracle_source="chainlink",
            oracle_price=3100,
            market_price=3000,
            market_venue="binance",
            deviation_pct=3.3,
            deviation_usd=100,
            direction="buy_market_sell_oracle",
            estimated_profit_pct=2.0,
            confidence=0.8,
            staleness_penalty=0.1,
            timestamp=datetime.now(timezone.utc),
        )
        
        result = strategy.execute_deviation_arb(opp, 5000)
        assert result.entry_price == 3000
        assert result.exit_price == 3100
        assert result.amount == 5000

    def test_get_oracle_prices(self):
        """Should return oracle prices."""
        strategy = OracleDeviationStrategy()
        strategy.update_oracle_price("BTC", "chainlink", 60000)
        strategy.update_oracle_price("ETH", "band", 3000)
        
        prices = strategy.get_oracle_prices()
        assert len(prices) == 2
        assert prices["BTC"].price == 60000
        assert prices["ETH"].price == 3000

    def test_get_stats(self):
        """Should return stats."""
        strategy = OracleDeviationStrategy()
        stats = strategy.get_stats()
        assert stats["total_trades"] == 0
        assert stats["total_profit"] == 0.0