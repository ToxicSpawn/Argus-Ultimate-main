"""
Tests for DEX HFT module — DEX connectors, DEX-CEX arb, flash loan arb,
mempool monitor, and gas optimizer.

Run with:  py -m pytest tests/test_dex_hft.py -v
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Guard module-level imports — these modules may not exist in all configurations
pytest.importorskip("core.connectors.dex_base")
pytest.importorskip("core.connectors.uniswap_v3")
pytest.importorskip("strategies.dex_cex_arb")
pytest.importorskip("strategies.flash_loan_arb")
pytest.importorskip("data.mempool_monitor")
pytest.importorskip("execution.gas_optimizer")

# ---------------------------------------------------------------------------
# DEXConnector base tests
# ---------------------------------------------------------------------------
from core.connectors.dex_base import DEXConnector, DEXConnectorError


class ConcreteDEX(DEXConnector):
    """Minimal concrete implementation for testing the ABC."""

    async def get_pool_price(self, pool_address):
        return 1.0

    async def get_pool_reserves(self, pool_address):
        return {"token0_reserve": 1000, "token1_reserve": 2000, "fee_tier": 3000}

    async def submit_swap(self, pool, token_in, token_out, amount, min_amount_out, deadline_seconds=30):
        return {"tx_hash": "0x123", "status": "ok"}

    async def get_pending_txs(self, pool_address):
        return []

    async def get_gas_price(self):
        return {"base_fee": 0.1, "priority_fee": 0.01, "estimated_cost_usd": 0.05}

    async def get_token_balance(self, token_address):
        return 100.0

    async def approve_token(self, token_address, spender, amount):
        return "0xabc"


class TestDEXConnectorBase:
    """Tests for DEXConnector abstract base class."""

    def test_init_requires_rpc_url(self):
        with pytest.raises(ValueError, match="rpc_url must not be empty"):
            ConcreteDEX("", "KEY_VAR", 1)

    def test_init_stores_chain_id(self):
        dex = ConcreteDEX("http://localhost:8545", "KEY_VAR", 42161)
        assert dex.chain_id == 42161
        assert dex.rpc_url == "http://localhost:8545"

    def test_chain_constants(self):
        assert DEXConnector.CHAIN_ETHEREUM == 1
        assert DEXConnector.CHAIN_ARBITRUM == 42161
        assert DEXConnector.CHAIN_BASE == 8453

    def test_has_private_key_false_when_not_set(self):
        dex = ConcreteDEX("http://localhost", "NONEXISTENT_KEY_12345", 1)
        assert dex.has_private_key() is False

    def test_has_private_key_true_when_set(self):
        os.environ["_TEST_DEX_KEY"] = "0xdeadbeef"
        try:
            dex = ConcreteDEX("http://localhost", "_TEST_DEX_KEY", 1)
            assert dex.has_private_key() is True
        finally:
            del os.environ["_TEST_DEX_KEY"]

    def test_get_private_key_raises_when_missing(self):
        dex = ConcreteDEX("http://localhost", "NONEXISTENT_KEY_999", 1)
        with pytest.raises(DEXConnectorError, match="not set"):
            dex._get_private_key()

    def test_estimate_price_impact_basic(self):
        reserves = {"token0_reserve": 100_000, "token1_reserve": 200_000}
        impact = DEXConnector.estimate_price_impact(reserves, 1000, 30)
        assert 0 < impact < 0.02  # small trade = small impact

    def test_estimate_price_impact_large_trade(self):
        reserves = {"token0_reserve": 10_000, "token1_reserve": 20_000}
        impact = DEXConnector.estimate_price_impact(reserves, 5000, 30)
        assert impact > 0.1  # 50% of reserve = large impact

    def test_estimate_price_impact_zero_reserves(self):
        assert DEXConnector.estimate_price_impact({"token0_reserve": 0, "token1_reserve": 100}, 10, 30) == 0.0
        assert DEXConnector.estimate_price_impact({"token0_reserve": 100, "token1_reserve": 0}, 10, 30) == 0.0

    def test_estimate_price_impact_zero_amount(self):
        reserves = {"token0_reserve": 1000, "token1_reserve": 2000}
        assert DEXConnector.estimate_price_impact(reserves, 0, 30) == 0.0

    def test_estimate_price_impact_empty_reserves(self):
        assert DEXConnector.estimate_price_impact({}, 100, 30) == 0.0

    def test_calculate_optimal_amount_basic(self):
        # Pool A is cheaper than pool B
        res_a = {"token0_reserve": 100_000, "token1_reserve": 200_000}  # price = 2.0
        res_b = {"token0_reserve": 100_000, "token1_reserve": 220_000}  # price = 2.2
        optimal = DEXConnector.calculate_optimal_amount(res_a, res_b, 30)
        assert optimal > 0

    def test_calculate_optimal_amount_same_price(self):
        res = {"token0_reserve": 100_000, "token1_reserve": 200_000}
        optimal = DEXConnector.calculate_optimal_amount(res, res, 30)
        assert optimal == 0.0

    def test_calculate_optimal_amount_wrong_direction(self):
        # Pool A is more expensive — no arb
        res_a = {"token0_reserve": 100_000, "token1_reserve": 220_000}  # price = 2.2
        res_b = {"token0_reserve": 100_000, "token1_reserve": 200_000}  # price = 2.0
        assert DEXConnector.calculate_optimal_amount(res_a, res_b, 30) == 0.0

    def test_calculate_optimal_amount_zero_reserves(self):
        assert DEXConnector.calculate_optimal_amount(
            {"token0_reserve": 0, "token1_reserve": 100},
            {"token0_reserve": 100, "token1_reserve": 200},
            30,
        ) == 0.0

    def test_next_rpc_id_increments(self):
        dex = ConcreteDEX("http://localhost", "KEY", 1)
        id1 = dex._next_rpc_id()
        id2 = dex._next_rpc_id()
        assert id2 == id1 + 1


# ---------------------------------------------------------------------------
# Uniswap V3 Connector tests
# ---------------------------------------------------------------------------
from core.connectors.uniswap_v3 import (
    FEE_TIERS,
    Q96,
    UniswapV3Connector,
)


class TestUniswapV3Connector:
    """Tests for the Uniswap V3 connector."""

    def test_init_requires_rpc(self):
        with pytest.raises(DEXConnectorError, match="No RPC URL"):
            # Ensure env var is not set
            with patch.dict(os.environ, {}, clear=True):
                UniswapV3Connector(rpc_url=None)

    def test_init_with_explicit_url(self):
        conn = UniswapV3Connector(rpc_url="http://localhost:8545")
        assert conn.rpc_url == "http://localhost:8545"
        assert conn.chain_id == DEXConnector.CHAIN_ARBITRUM

    def test_sqrt_price_x96_to_price_weth_usdc(self):
        # A realistic sqrtPriceX96 for WETH/USDC at ~$3000
        # price_raw = (sqrtPriceX96 / 2^96)^2
        # For WETH (18 dec) / USDC (6 dec), adjustment = 10^12
        # So sqrtPriceX96 = sqrt(3000 / 10^12) * 2^96
        target_price = 3000.0
        raw = target_price / (10 ** (18 - 6))
        sqrt_raw = math.sqrt(raw)
        sqrt_price_x96 = int(sqrt_raw * Q96)

        price = UniswapV3Connector.sqrt_price_x96_to_price(
            sqrt_price_x96, token0_decimals=18, token1_decimals=6
        )
        assert abs(price - target_price) / target_price < 0.01  # within 1%

    def test_sqrt_price_x96_zero(self):
        assert UniswapV3Connector.sqrt_price_x96_to_price(0) == 0.0

    def test_sqrt_price_roundtrip(self):
        """price → sqrtPriceX96 → price should roundtrip."""
        original_price = 2500.0
        sqrt_x96 = UniswapV3Connector.price_to_sqrt_price_x96(original_price, 18, 6)
        recovered = UniswapV3Connector.sqrt_price_x96_to_price(sqrt_x96, 18, 6)
        assert abs(recovered - original_price) / original_price < 0.001

    def test_price_to_sqrt_price_x96_zero(self):
        assert UniswapV3Connector.price_to_sqrt_price_x96(0) == 0

    def test_tick_to_price(self):
        # Tick 0 = price 1.0
        assert UniswapV3Connector.tick_to_price(0) == pytest.approx(1.0)
        # Positive tick = price > 1
        assert UniswapV3Connector.tick_to_price(100) > 1.0
        # Negative tick = price < 1
        assert UniswapV3Connector.tick_to_price(-100) < 1.0

    def test_price_to_tick(self):
        assert UniswapV3Connector.price_to_tick(1.0) == 0
        assert UniswapV3Connector.price_to_tick(0) == 0
        tick = UniswapV3Connector.price_to_tick(1.01)
        assert tick > 0

    def test_fee_tier_to_bps(self):
        assert UniswapV3Connector.fee_tier_to_bps(100) == 1
        assert UniswapV3Connector.fee_tier_to_bps(500) == 5
        assert UniswapV3Connector.fee_tier_to_bps(3000) == 30
        assert UniswapV3Connector.fee_tier_to_bps(10000) == 100

    def test_available_fee_tiers(self):
        tiers = UniswapV3Connector.available_fee_tiers()
        assert len(tiers) == 4
        assert all("fee" in t and "bps" in t for t in tiers)

    def test_register_pool(self):
        conn = UniswapV3Connector(rpc_url="http://localhost:8545")
        conn.register_pool("0xPool1", 18, 6, 3000, "WETH", "USDC")
        assert "0xpool1" in conn._pool_cache
        assert conn._pool_cache["0xpool1"]["token0_decimals"] == 18

    def test_estimate_multi_hop_output(self):
        conn = UniswapV3Connector(rpc_url="http://localhost:8545")
        reserves = {
            "pool_a": {"token0_reserve": 100_000, "token1_reserve": 200_000},
            "pool_b": {"token0_reserve": 50_000, "token1_reserve": 100_000},
        }
        path = [("pool_a", 3000), ("pool_b", 500)]
        output = conn.estimate_multi_hop_output(path, 1000, reserves)
        assert output > 0

    def test_estimate_multi_hop_empty_reserves(self):
        conn = UniswapV3Connector(rpc_url="http://localhost:8545")
        reserves = {"pool_a": {"token0_reserve": 0, "token1_reserve": 0}}
        path = [("pool_a", 3000)]
        assert conn.estimate_multi_hop_output(path, 1000, reserves) == 0.0

    def test_find_best_fee_tier(self):
        conn = UniswapV3Connector(rpc_url="http://localhost:8545")
        reserves_by_fee = {
            500: {"token0_reserve": 100_000, "token1_reserve": 200_000},
            3000: {"token0_reserve": 1_000_000, "token1_reserve": 2_000_000},
        }
        best = conn.find_best_fee_tier(reserves_by_fee, 1000)
        # The 3000 tier has deeper liquidity, so less impact = more output
        assert best == 3000


# ---------------------------------------------------------------------------
# DEX-CEX Arb tests
# ---------------------------------------------------------------------------
from strategies.dex_cex_arb import (
    ArbOpportunity,
    DEXCEXArbitrage,
    TradingSignal,
)


class TestDEXCEXArbitrage:
    """Tests for the DEX-CEX arbitrage strategy."""

    def test_init_validates(self):
        with pytest.raises(ValueError):
            DEXCEXArbitrage(min_profit_bps=-1)
        with pytest.raises(ValueError):
            DEXCEXArbitrage(max_gas_cost_usd=-1)
        with pytest.raises(ValueError):
            DEXCEXArbitrage(max_position_usd=0)

    def test_init_defaults(self):
        arb = DEXCEXArbitrage()
        assert arb.min_profit_bps == 10.0
        assert arb.max_gas_cost_usd == 5.0
        assert arb.max_position_usd == 500.0

    def test_update_cex_price(self):
        arb = DEXCEXArbitrage()
        arb.update_cex_price("ETH", 3000.0, "kraken")
        assert "kraken:ETH" in arb._cex_prices
        assert arb._cex_prices["kraken:ETH"].price == 3000.0

    def test_update_cex_price_rejects_zero(self):
        arb = DEXCEXArbitrage()
        arb.update_cex_price("ETH", 0, "kraken")
        assert len(arb._cex_prices) == 0

    def test_update_dex_price(self):
        arb = DEXCEXArbitrage()
        arb.update_dex_price("0xPool1", 3050.0, {"token0_reserve": 1000, "token1_reserve": 2000})
        assert "0xpool1" in arb._dex_prices

    def test_register_pool(self):
        arb = DEXCEXArbitrage()
        arb.register_pool("ETH", "0xPool1")
        assert "ETH" in arb._symbol_pools
        assert "0xpool1" in arb._pool_symbols

    def test_find_opportunities_no_data(self):
        arb = DEXCEXArbitrage()
        assert arb.find_opportunities() == []

    def test_find_opportunities_with_spread(self):
        arb = DEXCEXArbitrage(
            min_profit_bps=5,
            max_gas_cost_usd=1.0,
            dex_fee_bps=5,
            cex_fee_bps=4,
            slippage_buffer_bps=1,
        )
        arb.register_pool("ETH", "0xPool1")
        arb.update_gas_cost(0.5)

        # DEX is cheaper: 3000, CEX is 3100 → ~333 bps spread
        arb.update_dex_price("0xPool1", 3000.0, {"token0_reserve": 100000, "token1_reserve": 300000000})
        arb.update_cex_price("ETH", 3100.0, "kraken")

        opps = arb.find_opportunities()
        assert len(opps) >= 1
        assert opps[0].direction == "buy_dex_sell_cex"
        assert opps[0].profit_bps > 0

    def test_find_opportunities_filters_gas(self):
        arb = DEXCEXArbitrage(min_profit_bps=5, max_gas_cost_usd=0.01)
        arb.register_pool("ETH", "0xPool1")
        arb.update_gas_cost(100.0)  # gas too expensive

        arb.update_dex_price("0xPool1", 3000.0)
        arb.update_cex_price("ETH", 3100.0, "kraken")

        assert arb.find_opportunities() == []

    def test_find_opportunities_stale_prices(self):
        arb = DEXCEXArbitrage(min_profit_bps=1)
        arb.register_pool("ETH", "0xPool1")

        # Set very old timestamps
        arb.update_dex_price("0xPool1", 3000.0)
        arb.update_cex_price("ETH", 3100.0, "kraken")

        # Make prices stale
        arb._dex_prices["0xpool1"].timestamp = time.time() - 60
        arb._cex_prices["kraken:ETH"].timestamp = time.time() - 60

        assert arb.find_opportunities() == []

    def test_generate_signal_buy_dex(self):
        arb = DEXCEXArbitrage()
        opp = ArbOpportunity(
            symbol="ETH",
            direction="buy_dex_sell_cex",
            dex_price=3000.0,
            cex_price=3100.0,
            cex_exchange="kraken",
            pool_address="0xpool1",
            profit_bps=20.0,
            gas_cost_usd=0.5,
            optimal_size=500.0,
            confidence=0.67,
            timestamp=datetime.now(timezone.utc),
        )

        signal = arb.generate_signal(opp)
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "ETH"
        assert signal.action == "buy"
        assert signal.confidence == 0.67
        assert signal.entry_price == 3000.0
        assert "DEX-CEX arb" in signal.reasoning
        assert signal.metadata["strategy"] == "dex_cex_arb"

    def test_generate_signal_buy_cex(self):
        arb = DEXCEXArbitrage()
        opp = ArbOpportunity(
            symbol="ETH",
            direction="buy_cex_sell_dex",
            dex_price=3100.0,
            cex_price=3000.0,
            cex_exchange="coinbase",
            pool_address="0xpool1",
            profit_bps=15.0,
            gas_cost_usd=0.3,
            optimal_size=400.0,
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
        )

        signal = arb.generate_signal(opp)
        assert signal.action == "sell"
        assert signal.entry_price == 3000.0

    def test_get_stats(self):
        arb = DEXCEXArbitrage()
        stats = arb.get_stats()
        assert "cex_prices_tracked" in stats
        assert "dex_pools_tracked" in stats

    def test_arb_opportunity_expected_profit(self):
        opp = ArbOpportunity(
            symbol="ETH",
            direction="buy_dex_sell_cex",
            dex_price=3000,
            cex_price=3100,
            cex_exchange="kraken",
            pool_address="0x1",
            profit_bps=20,
            gas_cost_usd=0.5,
            optimal_size=500,
            confidence=0.5,
            timestamp=datetime.now(timezone.utc),
        )
        # 500 * 20/10000 - 0.5 = 1.0 - 0.5 = 0.5
        assert opp.expected_profit_usd(500) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Flash Loan Arb tests
# ---------------------------------------------------------------------------
from strategies.flash_loan_arb import FlashLoanArbitrage, PROVIDER_FEES


class TestFlashLoanArbitrage:
    """Tests for the flash loan arbitrage strategy."""

    def test_init_defaults(self):
        fla = FlashLoanArbitrage()
        assert fla.min_profit_usd == 5.0
        assert fla.max_loan_usd == 100_000.0
        assert fla.provider == "aave"
        assert fla.loan_fee_bps == 9

    def test_init_validates(self):
        with pytest.raises(ValueError):
            FlashLoanArbitrage(min_profit_usd=-1)
        with pytest.raises(ValueError):
            FlashLoanArbitrage(max_loan_usd=0)

    def test_provider_fees(self):
        assert PROVIDER_FEES["aave"] == 9
        assert PROVIDER_FEES["dydx"] == 0
        assert PROVIDER_FEES["balancer"] == 0

    def test_dydx_provider(self):
        fla = FlashLoanArbitrage(provider="dydx")
        assert fla.loan_fee_bps == 0

    def test_calculate_no_arb_same_price(self):
        fla = FlashLoanArbitrage(min_profit_usd=0.01)
        res = {"token0_reserve": 100_000, "token1_reserve": 200_000}
        result = fla.calculate_flash_loan_arb(res, res, 30, 30)
        assert result.profitable is False

    def test_calculate_arb_profitable(self):
        fla = FlashLoanArbitrage(min_profit_usd=0.01, provider="dydx", gas_estimate_usd=0.01)
        # Pool A: cheap (price = 2.0)
        res_a = {"token0_reserve": 1_000_000, "token1_reserve": 2_000_000}
        # Pool B: expensive (price = 2.2)
        res_b = {"token0_reserve": 1_000_000, "token1_reserve": 2_200_000}
        result = fla.calculate_flash_loan_arb(res_a, res_b, 30, 30, loan_fee_bps=0)
        assert result.gross_profit > 0  # should find positive gross profit
        assert result.optimal_amount > 0

    def test_calculate_arb_zero_reserves(self):
        fla = FlashLoanArbitrage()
        result = fla.calculate_flash_loan_arb(
            {"token0_reserve": 0, "token1_reserve": 100},
            {"token0_reserve": 100, "token1_reserve": 200},
            30, 30,
        )
        assert result.profitable is False
        assert result.optimal_amount == 0

    def test_calculate_arb_wrong_direction(self):
        fla = FlashLoanArbitrage()
        # Pool A is more expensive — no arb
        res_a = {"token0_reserve": 100_000, "token1_reserve": 220_000}
        res_b = {"token0_reserve": 100_000, "token1_reserve": 200_000}
        result = fla.calculate_flash_loan_arb(res_a, res_b, 30, 30)
        assert result.profitable is False

    def test_loan_fee_accounting(self):
        fla = FlashLoanArbitrage(provider="aave", min_profit_usd=0.01, gas_estimate_usd=0.01)
        res_a = {"token0_reserve": 1_000_000, "token1_reserve": 2_000_000}
        res_b = {"token0_reserve": 1_000_000, "token1_reserve": 2_200_000}
        result = fla.calculate_flash_loan_arb(res_a, res_b, 30, 30)
        if result.optimal_amount > 0:
            assert result.loan_fee == result.optimal_amount * 9 / 10_000.0

    def test_find_triangular_arb_insufficient_pools(self):
        fla = FlashLoanArbitrage()
        assert fla.find_triangular_arb({}) == []
        assert fla.find_triangular_arb({"p1": {"token0": "A", "token1": "B"}}) == []

    def test_find_triangular_arb_basic(self):
        fla = FlashLoanArbitrage(min_profit_usd=0.001, gas_estimate_usd=0.001)
        fla.loan_fee_bps = 0  # remove loan fee for easier testing

        pools = {
            "pool_ab": {
                "token0": "A",
                "token1": "B",
                "fee_bps": 5,
                "reserves": {"token0_reserve": 100_000, "token1_reserve": 200_000},
            },
            "pool_bc": {
                "token0": "B",
                "token1": "C",
                "fee_bps": 5,
                "reserves": {"token0_reserve": 200_000, "token1_reserve": 500_000},
            },
            "pool_ca": {
                "token0": "C",
                "token1": "A",
                "fee_bps": 5,
                "reserves": {"token0_reserve": 500_000, "token1_reserve": 120_000},
            },
        }
        results = fla.find_triangular_arb(pools)
        # May or may not find profitable path depending on price alignment
        assert isinstance(results, list)

    def test_simulate_execution_valid(self):
        fla = FlashLoanArbitrage(gas_estimate_usd=0.01)
        plan = {
            "amount": 1000,
            "pools": ["pool_a", "pool_b"],
            "reserves": [
                {"token0_reserve": 100_000, "token1_reserve": 200_000},
                {"token0_reserve": 50_000, "token1_reserve": 100_000},
            ],
            "fees_bps": [30, 30],
            "provider": "dydx",
        }
        result = fla.simulate_execution(plan)
        assert "success" in result
        assert "net_profit" in result

    def test_simulate_execution_invalid_plan(self):
        fla = FlashLoanArbitrage()
        result = fla.simulate_execution({"amount": 1000, "pools": [], "reserves": []})
        assert result["success"] is False

    def test_simulate_execution_zero_reserves(self):
        fla = FlashLoanArbitrage()
        plan = {
            "amount": 1000,
            "pools": ["pool_a"],
            "reserves": [{"token0_reserve": 0, "token1_reserve": 0}],
            "fees_bps": [30],
        }
        result = fla.simulate_execution(plan)
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Mempool Monitor tests
# ---------------------------------------------------------------------------
from data.mempool_monitor import (
    DecodedSwap,
    EXACT_INPUT_SINGLE,
    EXACT_INPUT_SINGLE_V2,
    MempoolMonitor,
    SWAP_SELECTORS,
)


class TestMempoolMonitor:
    """Tests for the mempool monitor."""

    def test_init_requires_url(self):
        with pytest.raises(ValueError, match="rpc_url must not be empty"):
            MempoolMonitor("")

    def test_init_stores_pools(self):
        mm = MempoolMonitor("wss://rpc.example.com", ["0xPool1", "0xPool2"])
        assert len(mm.watched_pools) == 2
        assert "0xpool1" in mm.watched_pools

    def test_decode_swap_tx_none(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        assert mm.decode_swap_tx("") is None
        assert mm.decode_swap_tx("0x123") is None
        assert mm.decode_swap_tx(None) is None

    def test_decode_swap_tx_unknown_selector(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        assert mm.decode_swap_tx("0xdeadbeef" + "0" * 500) is None

    def test_decode_swap_tx_exact_input_single(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        # Build a minimal exactInputSingle calldata
        # Selector + 7 words of 32 bytes each
        selector = EXACT_INPUT_SINGLE
        # Word 0: token_in (address, 20 bytes right-padded to 32)
        token_in = "000000000000000000000000" + "a" * 40
        # Word 1: token_out
        token_out = "000000000000000000000000" + "b" * 40
        # Word 2: fee
        fee = "0" * 64
        # Word 3: recipient
        recipient = "0" * 64
        # Word 4: deadline
        deadline = "0" * 56 + "ffffffff"
        # Word 5: amount_in (1 ETH = 10^18)
        amount_in = "0" * 48 + "0de0b6b3a7640000"  # 10^18
        # Word 6: min_amount_out
        min_out = "0" * 48 + "0de0b6b3a7640000"

        data = selector + token_in + token_out + fee + recipient + deadline + amount_in + min_out
        decoded = mm.decode_swap_tx(data)

        assert decoded is not None
        assert decoded.amount_in == pytest.approx(1.0, rel=0.01)

    def test_decode_swap_tx_short_data(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        # Valid selector but too short data
        result = mm.decode_swap_tx(EXACT_INPUT_SINGLE + "0" * 10)
        assert result is None

    def test_estimate_price_impact(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        swap = DecodedSwap(
            tx_hash="0x1",
            pool="0xpool",
            token_in="0xa",
            token_out="0xb",
            amount_in=10_000,
            min_amount_out=0,
            deadline=0,
            sender="0xsender",
            gas_price_gwei=0.1,
            selector="0x414bf389",
        )
        reserves = {"token0_reserve": 100_000, "token1_reserve": 200_000}
        impact = mm.estimate_price_impact(swap, reserves)
        assert 0 < impact < 1.0

    def test_estimate_price_impact_zero(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        swap = DecodedSwap(
            tx_hash="", pool="", token_in="", token_out="",
            amount_in=0, min_amount_out=0, deadline=0,
            sender="", gas_price_gwei=0, selector="",
        )
        assert mm.estimate_price_impact(swap, {"token0_reserve": 100, "token1_reserve": 200}) == 0.0

    def test_get_frontrun_opportunity_zero_reserves(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        swap = DecodedSwap(
            tx_hash="0x1", pool="", token_in="", token_out="",
            amount_in=1000, min_amount_out=0, deadline=0,
            sender="", gas_price_gwei=1.0, selector="",
        )
        result = mm.get_frontrun_opportunity(swap, {"token0_reserve": 0, "token1_reserve": 0})
        assert result.profitable is False

    def test_get_frontrun_opportunity_basic(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        swap = DecodedSwap(
            tx_hash="0x1", pool="0xpool", token_in="0xa", token_out="0xb",
            amount_in=50_000, min_amount_out=0, deadline=0,
            sender="0xwhale", gas_price_gwei=0.1, selector="",
        )
        reserves = {"token0_reserve": 1_000_000, "token1_reserve": 2_000_000}
        result = mm.get_frontrun_opportunity(swap, reserves)
        # Should return a valid analysis (may or may not be profitable)
        assert result.amount > 0
        assert result.pending_swap == swap

    def test_on_pending_swap_callback(self):
        mm = MempoolMonitor("wss://rpc.example.com")
        called = []
        mm.on_pending_swap(lambda swap: called.append(swap))
        assert len(mm._callbacks) == 1

    def test_add_watched_pool(self):
        mm = MempoolMonitor("wss://rpc.example.com", [])
        mm.add_watched_pool("0xNewPool")
        assert "0xnewpool" in mm.watched_pools
        mm.add_watched_pool("0xNewPool")  # duplicate
        assert mm.watched_pools.count("0xnewpool") == 1

    def test_get_stats(self):
        mm = MempoolMonitor("wss://rpc.example.com", ["0x1", "0x2"])
        stats = mm.get_stats()
        assert stats["watched_pools"] == 2
        assert stats["running"] is False

    def test_swap_selectors_set(self):
        assert len(SWAP_SELECTORS) >= 6


# ---------------------------------------------------------------------------
# Gas Optimizer tests
# ---------------------------------------------------------------------------
from execution.gas_optimizer import GasOptimizer


class TestGasOptimizer:
    """Tests for the gas fee optimizer."""

    def test_init_defaults(self):
        go = GasOptimizer()
        assert go.eth_price_usd == 3000.0
        assert go.gas_units_swap == 150_000

    def test_record_gas_price(self):
        go = GasOptimizer()
        go.record_gas_price(0.1, 0.01)
        assert len(go._gas_history) == 1

    def test_record_negative_ignored(self):
        go = GasOptimizer()
        go.record_gas_price(-1, 0.01)
        assert len(go._gas_history) == 0

    def test_predict_gas_price_no_history(self):
        go = GasOptimizer()
        pred = go.predict_gas_price()
        assert pred["confidence"] == 0.0
        assert pred["base_fee"] > 0

    def test_predict_gas_price_with_history(self):
        go = GasOptimizer()
        for i in range(30):
            go.record_gas_price(0.1 + i * 0.001, 0.01, time.time() - (30 - i))
        pred = go.predict_gas_price()
        assert pred["confidence"] > 0
        assert pred["base_fee"] > 0
        assert pred["total_fee"] > 0

    def test_predict_gas_price_trend(self):
        go = GasOptimizer()
        # Rising prices
        for i in range(20):
            go.record_gas_price(0.1 + i * 0.01, 0.01, time.time() - (20 - i))
        pred = go.predict_gas_price(blocks_ahead=5)
        assert "trend" in pred

    def test_optimal_priority_fee_no_history(self):
        go = GasOptimizer()
        assert go.optimal_priority_fee("low") == 0.005
        assert go.optimal_priority_fee("normal") == 0.01
        assert go.optimal_priority_fee("high") == 0.05
        assert go.optimal_priority_fee("critical") == 0.1

    def test_optimal_priority_fee_with_history(self):
        go = GasOptimizer()
        for i in range(100):
            go.record_gas_price(0.1, 0.001 * (i + 1))

        low = go.optimal_priority_fee("low")
        normal = go.optimal_priority_fee("normal")
        high = go.optimal_priority_fee("high")
        critical = go.optimal_priority_fee("critical")

        assert low <= normal <= high <= critical

    def test_is_gas_favorable(self):
        go = GasOptimizer(eth_price_usd=3000)
        # With default predictions (0.1 gwei), gas should be very cheap
        assert go.is_gas_favorable(max_cost_usd=5.0) is True

    def test_is_gas_favorable_expensive(self):
        go = GasOptimizer(eth_price_usd=3000)
        for _ in range(20):
            go.record_gas_price(100, 50)  # very high gas
        assert go.is_gas_favorable(max_cost_usd=0.001) is False

    def test_current_cost_usd(self):
        go = GasOptimizer(eth_price_usd=3000)
        cost = go.current_cost_usd()
        assert cost >= 0

    def test_get_stats_no_history(self):
        go = GasOptimizer()
        stats = go.get_stats()
        assert stats["history_size"] == 0
        assert stats["trend"] == "unknown"

    def test_get_stats_with_history(self):
        go = GasOptimizer()
        for i in range(20):
            go.record_gas_price(0.1, 0.01)
        stats = go.get_stats()
        assert stats["history_size"] == 20
        assert stats["avg"] > 0
        assert stats["median"] > 0
        assert stats["p95"] > 0
        assert stats["min"] > 0
        assert stats["max"] > 0

    def test_get_stats_trend_detection(self):
        go = GasOptimizer()
        # Falling gas prices
        for i in range(20):
            go.record_gas_price(1.0 - i * 0.04, 0.01)
        stats = go.get_stats()
        assert stats["trend"] in ("rising", "falling", "stable", "insufficient_data")

    def test_update_eth_price(self):
        go = GasOptimizer(eth_price_usd=3000)
        go.update_eth_price(4000)
        assert go.eth_price_usd == 4000
        go.update_eth_price(-100)  # invalid
        assert go.eth_price_usd == 4000  # unchanged


# ---------------------------------------------------------------------------
# Config validation test
# ---------------------------------------------------------------------------

class TestDEXConfig:
    """Tests for DEX config integration."""

    def test_dex_in_known_keys(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        assert "dex" in _KNOWN_TOP_LEVEL_KEYS

    def test_config_loads_dex_section(self):
        """Verify the unified_config.yaml has a dex section."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "unified_config.yaml",
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert "dex" in config
        assert config["dex"]["enabled"] is False
        assert config["dex"]["chain"] == "arbitrum"
        assert config["dex"]["max_gas_cost_usd"] == 5.0
        assert config["dex"]["min_arb_profit_bps"] == 10
        assert config["dex"]["flash_loan_enabled"] is False
        assert config["dex"]["mempool_monitoring"] is False
        assert config["dex"]["private_key_env"] == "DEX_PRIVATE_KEY"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_price_impact_tiny_reserves(self):
        """Very small reserves should produce large impact."""
        reserves = {"token0_reserve": 1, "token1_reserve": 1}
        impact = DEXConnector.estimate_price_impact(reserves, 1, 30)
        assert impact > 0.3

    def test_gas_optimizer_single_entry(self):
        go = GasOptimizer()
        go.record_gas_price(0.5, 0.02)
        pred = go.predict_gas_price()
        assert pred["base_fee"] == pytest.approx(0.5)

    def test_flash_loan_very_small_reserves(self):
        fla = FlashLoanArbitrage(min_profit_usd=0)
        res_a = {"token0_reserve": 1, "token1_reserve": 2}
        res_b = {"token0_reserve": 1, "token1_reserve": 3}
        result = fla.calculate_flash_loan_arb(res_a, res_b, 30, 30)
        # Should not crash
        assert isinstance(result.profitable, bool)

    def test_dex_cex_arb_stale_dex_fresh_cex(self):
        """Only DEX price stale — should find no opportunity."""
        arb = DEXCEXArbitrage(min_profit_bps=1)
        arb.register_pool("ETH", "0xPool1")
        arb.update_dex_price("0xPool1", 3000.0)
        arb.update_cex_price("ETH", 3100.0, "kraken")
        arb._dex_prices["0xpool1"].timestamp = time.time() - 120  # stale
        assert arb.find_opportunities() == []

    def test_gas_spike_blocks_arb(self):
        """Gas spike should prevent arb from being profitable."""
        arb = DEXCEXArbitrage(min_profit_bps=5, max_gas_cost_usd=1.0)
        arb.register_pool("ETH", "0xPool1")
        arb.update_gas_cost(50.0)  # $50 gas
        arb.update_dex_price("0xPool1", 3000.0)
        arb.update_cex_price("ETH", 3010.0, "kraken")
        assert arb.find_opportunities() == []

    def test_multiple_cex_venues(self):
        """Arb should check all CEX venues."""
        arb = DEXCEXArbitrage(min_profit_bps=5, dex_fee_bps=5, cex_fee_bps=2, slippage_buffer_bps=1)
        arb.register_pool("ETH", "0xPool1")
        arb.update_gas_cost(0.1)
        arb.update_dex_price("0xPool1", 3000.0, {"token0_reserve": 1e6, "token1_reserve": 3e9})
        arb.update_cex_price("ETH", 3050.0, "kraken")
        arb.update_cex_price("ETH", 3100.0, "coinbase")

        opps = arb.find_opportunities()
        # Should find opportunities on both exchanges
        assert len(opps) >= 1
        # Best opportunity should be coinbase (biggest spread)
        if len(opps) >= 2:
            assert opps[0].profit_bps >= opps[1].profit_bps
