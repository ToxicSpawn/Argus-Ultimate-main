"""Tests for data.onchain.etherscan_whales — EtherscanWhaleMonitor."""

from __future__ import annotations

import asyncio
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from data.onchain.etherscan_whales import EtherscanWhaleMonitor
from data.onchain.whale_tracker import WhaleTransaction


class TestEtherscanWhaleMonitor(unittest.TestCase):
    """Tests for EtherscanWhaleMonitor."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_default_construction(self) -> None:
        monitor = EtherscanWhaleMonitor(api_key="test_key")
        self.assertEqual(monitor._api_key, "test_key")
        self.assertEqual(monitor._min_value_usd, 100_000.0)
        self.assertEqual(monitor._poll_interval_s, 300.0)

    def test_no_api_key_returns_empty(self) -> None:
        """Returns empty list when no API key is set."""
        monitor = EtherscanWhaleMonitor(api_key="")
        result = self._run(monitor.poll_recent_whales())
        self.assertEqual(result, [])

    @patch.dict(os.environ, {"ETHERSCAN_API_KEY": ""}, clear=False)
    def test_no_env_api_key_returns_empty(self) -> None:
        """Returns empty list when env var is empty."""
        monitor = EtherscanWhaleMonitor()
        result = self._run(monitor.poll_recent_whales())
        self.assertEqual(result, [])

    def test_poll_interval_respected(self) -> None:
        """Second poll within interval returns cached results."""
        monitor = EtherscanWhaleMonitor(api_key="test", poll_interval_s=300)
        monitor._last_poll_ts = time.time()
        monitor._last_results = [MagicMock()]

        result = self._run(monitor.poll_recent_whales())
        self.assertEqual(len(result), 1)

    @patch("data.onchain.etherscan_whales.aiohttp")
    def test_successful_api_call(self, mock_aiohttp) -> None:
        """Parses a successful Etherscan API response into WhaleTransactions."""
        fake_transfers = [
            {
                "hash": "0xabc123",
                "contractAddress": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                "tokenSymbol": "USDT",
                "tokenDecimal": "6",
                "value": "500000000000",  # 500,000 USDT
                "from": "0x1234567890abcdef1234567890abcdef12345678",
                "to": "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance
                "timeStamp": str(int(time.time())),
            },
        ]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "status": "1",
            "result": fake_transfers,
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        monitor = EtherscanWhaleMonitor(api_key="test_key")
        result = self._run(monitor.poll_recent_whales())

        self.assertEqual(len(result), 1)
        tx = result[0]
        self.assertIsInstance(tx, WhaleTransaction)
        self.assertEqual(tx.asset, "USDT")
        self.assertAlmostEqual(tx.usd_value, 500_000.0)
        self.assertEqual(tx.signal, "INFLOW")  # going TO Binance
        self.assertEqual(tx.to_exchange, "binance")

    @patch("data.onchain.etherscan_whales.aiohttp")
    def test_min_value_filter(self, mock_aiohttp) -> None:
        """Transfers below min_value_usd are filtered out."""
        fake_transfers = [
            {
                "hash": "0xsmall",
                "contractAddress": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                "tokenSymbol": "USDT",
                "tokenDecimal": "6",
                "value": "50000000",  # 50 USDT — below threshold
                "from": "0x1111111111111111111111111111111111111111",
                "to": "0x2222222222222222222222222222222222222222",
                "timeStamp": str(int(time.time())),
            },
        ]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "status": "1",
            "result": fake_transfers,
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        monitor = EtherscanWhaleMonitor(api_key="test_key", min_value_usd=100_000)
        result = self._run(monitor.poll_recent_whales())
        self.assertEqual(len(result), 0)

    @patch("data.onchain.etherscan_whales.aiohttp")
    def test_api_error_returns_empty(self, mock_aiohttp) -> None:
        """API error (non-200) returns empty list."""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        monitor = EtherscanWhaleMonitor(api_key="test_key")
        result = self._run(monitor.poll_recent_whales())
        self.assertEqual(result, [])

    @patch("data.onchain.etherscan_whales.aiohttp")
    def test_network_exception_returns_empty(self, mock_aiohttp) -> None:
        """Network exception returns empty list."""
        mock_aiohttp.ClientSession = MagicMock(side_effect=Exception("connection refused"))
        mock_aiohttp.ClientTimeout = MagicMock()

        monitor = EtherscanWhaleMonitor(api_key="test_key")
        result = self._run(monitor.poll_recent_whales())
        self.assertEqual(result, [])

    def test_exchange_label_detection_outflow(self) -> None:
        """Transfer FROM exchange and TO non-exchange is OUTFLOW."""
        monitor = EtherscanWhaleMonitor(api_key="test")
        transfers = [
            {
                "hash": "0xoutflow",
                "contractAddress": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                "tokenSymbol": "USDT",
                "tokenDecimal": "6",
                "value": "1000000000000",  # 1M USDT
                "from": "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance
                "to": "0xdeadbeef00000000000000000000000000000000",
                "timeStamp": str(int(time.time())),
            },
        ]
        result = monitor._convert_to_whale_transactions(transfers, 100_000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].signal, "OUTFLOW")
        self.assertEqual(result[0].from_exchange, "binance")
        self.assertIsNone(result[0].to_exchange)

    def test_weth_value_estimation(self) -> None:
        """WETH transfers use rough ETH price estimate."""
        monitor = EtherscanWhaleMonitor(api_key="test")
        transfers = [
            {
                "hash": "0xweth",
                "contractAddress": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                "tokenSymbol": "WETH",
                "tokenDecimal": "18",
                "value": "100000000000000000000",  # 100 WETH
                "from": "0x1111111111111111111111111111111111111111",
                "to": "0x2222222222222222222222222222222222222222",
                "timeStamp": str(int(time.time())),
            },
        ]
        result = monitor._convert_to_whale_transactions(transfers, 100_000)
        self.assertEqual(len(result), 1)
        # 100 WETH * $3000 = $300K
        self.assertAlmostEqual(result[0].usd_value, 300_000.0)

    @patch("data.onchain.etherscan_whales.aiohttp")
    def test_min_value_override(self, mock_aiohttp) -> None:
        """poll_recent_whales accepts min_value_usd override."""
        fake_transfers = [
            {
                "hash": "0xmedium",
                "contractAddress": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                "tokenSymbol": "USDT",
                "tokenDecimal": "6",
                "value": "75000000000",  # 75,000 USDT
                "from": "0x1111111111111111111111111111111111111111",
                "to": "0x28c6c06298d514db089934071355e5743bf21d60",
                "timeStamp": str(int(time.time())),
            },
        ]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "1", "result": fake_transfers})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        monitor = EtherscanWhaleMonitor(api_key="test_key", min_value_usd=100_000)
        # Override to lower threshold
        result = self._run(monitor.poll_recent_whales(min_value_usd=50_000))
        self.assertEqual(len(result), 1)

    @patch("data.onchain.etherscan_whales.aiohttp")
    def test_api_status_not_1(self, mock_aiohttp) -> None:
        """API returning status != '1' returns empty list."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "0", "message": "NOTOK", "result": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = MagicMock()

        monitor = EtherscanWhaleMonitor(api_key="test_key")
        result = self._run(monitor.poll_recent_whales())
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
