"""
Tests for core/exchange_manager.py failover, health check fixes, and key rotation.

Covers:
- failover_to_backup does not raise AttributeError when exchange lacks .connected
- health_check_all uses health_check_symbol attribute when available
- rotate_api_keys: successful rotation with connectivity test
- rotate_api_keys: failed connectivity keeps old connector
- rotate_api_keys: unknown exchange raises KeyError
- rotate_api_keys: old connector is disconnected after swap
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFailoverNoConnectedAttr:
    def test_no_attribute_error_without_connected(self):
        """Exchange without .connected attribute must not crash failover."""
        from core.exchange_manager import ExchangeManager

        mgr = ExchangeManager()
        # spec=[] means no attributes at all — simulates missing .connected
        mock_ex = MagicMock(spec=[])
        mgr.exchanges["primary"] = mock_ex
        mgr.exchanges["backup"] = mock_ex
        mgr.active_exchanges = ["primary", "backup"]
        mgr.exchange_configs = {"backup": MagicMock(priority=1)}
        mgr.primary_exchange = "primary"

        result = asyncio.run(mgr.failover_to_backup("primary"))
        assert isinstance(result, bool)

    def test_failover_succeeds_when_backup_has_connected_true(self):
        """When backup exchange has connected=True, failover should return True."""
        from core.exchange_manager import ExchangeManager

        mgr = ExchangeManager()
        backup_ex = MagicMock()
        backup_ex.connected = True
        mgr.exchanges["primary"] = MagicMock(connected=False)
        mgr.exchanges["backup"] = backup_ex
        mgr.active_exchanges = ["primary", "backup"]
        mgr.exchange_configs = {"backup": MagicMock(priority=1)}
        mgr.primary_exchange = "primary"

        result = asyncio.run(mgr.failover_to_backup("primary"))
        assert result is True
        assert mgr.primary_exchange == "backup"

    def test_failover_returns_false_with_no_available_backups(self):
        """When all exchanges lack .connected, failover should return False gracefully."""
        from core.exchange_manager import ExchangeManager

        mgr = ExchangeManager()
        # No .connected attribute, so getattr(ex, "connected", False) → False
        mgr.exchanges["primary"] = MagicMock(spec=[])
        mgr.active_exchanges = ["primary"]
        mgr.exchange_configs = {}
        mgr.primary_exchange = "primary"

        result = asyncio.run(mgr.failover_to_backup("primary"))
        assert result is False


class TestHealthCheckUsesConfigurableSymbol:
    def test_health_check_uses_health_check_symbol_attr(self):
        """If an exchange has health_check_symbol attribute, it should be used."""
        from core.exchange_manager import ExchangeManager

        mgr = ExchangeManager()
        mock_ex = AsyncMock()
        mock_ex.health_check_symbol = "BTC/USDT"
        mock_ex.get_ticker = AsyncMock(return_value={"last": 50000.0})
        mgr.exchanges["kraken"] = mock_ex

        asyncio.run(mgr.health_check_all())
        mock_ex.get_ticker.assert_called_once_with("BTC/USDT")

    def test_health_check_falls_back_to_btc_usd(self):
        """Without health_check_symbol, health check should use BTC/USD."""
        from core.exchange_manager import ExchangeManager

        mgr = ExchangeManager()
        mock_ex = AsyncMock(spec=["get_ticker"])
        mock_ex.get_ticker = AsyncMock(return_value={"last": 50000.0})
        mgr.exchanges["coinbase"] = mock_ex

        asyncio.run(mgr.health_check_all())
        mock_ex.get_ticker.assert_called_once_with("BTC/USD")


class TestRotateApiKeys:
    """Tests for ExchangeManager.rotate_api_keys()."""

    def test_rotate_unknown_exchange_raises_key_error(self):
        """Rotating keys for an unregistered exchange must raise KeyError."""
        from core.exchange_manager import ExchangeManager

        mgr = ExchangeManager()
        with pytest.raises(KeyError, match="not registered"):
            asyncio.run(mgr.rotate_api_keys("nonexistent", "k", "s"))

    def test_rotate_successful_swaps_connector(self):
        """Successful rotation replaces the connector and disconnects the old one."""
        from core.exchange_manager import ExchangeManager, ExchangeConfig

        mgr = ExchangeManager()
        old_conn = AsyncMock()
        old_conn.health_check_symbol = "BTC/USD"
        old_conn.disconnect = AsyncMock()
        mgr.exchanges["coinbase"] = old_conn
        mgr.exchange_configs["coinbase"] = ExchangeConfig(
            name="coinbase", api_key="old-key", api_secret="old-secret"
        )

        new_conn = AsyncMock()
        new_conn.get_ticker = AsyncMock(return_value={"last": 95000.0})

        with patch.object(mgr, "_create_connector", return_value=new_conn):
            asyncio.run(mgr.rotate_api_keys("coinbase", "new-key", "new-secret"))

        # Old connector was disconnected
        old_conn.disconnect.assert_called_once()
        # New connector is now installed
        assert mgr.exchanges["coinbase"] is new_conn
        # Config updated
        assert mgr.exchange_configs["coinbase"].api_key == "new-key"
        assert mgr.exchange_configs["coinbase"].api_secret == "new-secret"

    def test_rotate_failed_connectivity_keeps_old(self):
        """If new credentials fail the ticker test, old connector is kept."""
        from core.exchange_manager import ExchangeManager, ExchangeConfig

        mgr = ExchangeManager()
        old_conn = AsyncMock()
        old_conn.health_check_symbol = "BTC/USD"
        old_conn.disconnect = AsyncMock()
        mgr.exchanges["coinbase"] = old_conn
        mgr.exchange_configs["coinbase"] = ExchangeConfig(
            name="coinbase", api_key="old-key", api_secret="old-secret"
        )

        bad_conn = AsyncMock()
        bad_conn.get_ticker = AsyncMock(return_value=None)  # connectivity fails

        with patch.object(mgr, "_create_connector", return_value=bad_conn):
            with pytest.raises(RuntimeError, match="no ticker data"):
                asyncio.run(mgr.rotate_api_keys("coinbase", "bad-key", "bad-secret"))

        # Old connector is still in place
        assert mgr.exchanges["coinbase"] is old_conn
        # Old connector was NOT disconnected
        old_conn.disconnect.assert_not_called()
        # Config unchanged
        assert mgr.exchange_configs["coinbase"].api_key == "old-key"

    def test_rotate_exception_during_ticker_keeps_old(self):
        """If get_ticker raises an exception, old connector is kept."""
        from core.exchange_manager import ExchangeManager, ExchangeConfig

        mgr = ExchangeManager()
        old_conn = AsyncMock()
        old_conn.health_check_symbol = "BTC/USD"
        mgr.exchanges["coinbase"] = old_conn
        mgr.exchange_configs["coinbase"] = ExchangeConfig(
            name="coinbase", api_key="old-key", api_secret="old-secret"
        )

        error_conn = AsyncMock()
        error_conn.get_ticker = AsyncMock(side_effect=ConnectionError("auth failed"))

        with patch.object(mgr, "_create_connector", return_value=error_conn):
            with pytest.raises(RuntimeError, match="Connectivity test failed"):
                asyncio.run(mgr.rotate_api_keys("coinbase", "err-key", "err-secret"))

        assert mgr.exchanges["coinbase"] is old_conn

    def test_rotate_uses_health_check_symbol_from_old_connector(self):
        """Rotation uses health_check_symbol from the existing connector."""
        from core.exchange_manager import ExchangeManager, ExchangeConfig

        mgr = ExchangeManager()
        old_conn = AsyncMock()
        old_conn.health_check_symbol = "ETH/AUD"
        old_conn.disconnect = AsyncMock()
        mgr.exchanges["kraken"] = old_conn
        mgr.exchange_configs["kraken"] = ExchangeConfig(
            name="kraken", api_key="old", api_secret="old"
        )

        new_conn = AsyncMock()
        new_conn.get_ticker = AsyncMock(return_value={"last": 4000.0})

        with patch.object(mgr, "_create_connector", return_value=new_conn):
            asyncio.run(mgr.rotate_api_keys("kraken", "new", "new"))

        # The ticker test should have used the old connector's health_check_symbol
        new_conn.get_ticker.assert_called_once_with("ETH/AUD")

    def test_rotate_falls_back_to_btc_usd_if_no_health_symbol(self):
        """If old connector has no health_check_symbol, fall back to BTC/USD."""
        from core.exchange_manager import ExchangeManager, ExchangeConfig

        mgr = ExchangeManager()
        old_conn = AsyncMock(spec=["disconnect", "get_ticker"])
        old_conn.disconnect = AsyncMock()
        mgr.exchanges["ex1"] = old_conn
        mgr.exchange_configs["ex1"] = ExchangeConfig(name="ex1")

        new_conn = AsyncMock()
        new_conn.get_ticker = AsyncMock(return_value={"last": 50000.0})

        with patch.object(mgr, "_create_connector", return_value=new_conn):
            asyncio.run(mgr.rotate_api_keys("ex1", "k", "s"))

        new_conn.get_ticker.assert_called_once_with("BTC/USD")

    def test_rotate_does_not_log_actual_keys(self):
        """Rotation logging must not contain the actual API key or secret."""
        from core.exchange_manager import ExchangeManager, ExchangeConfig

        mgr = ExchangeManager()
        old_conn = AsyncMock()
        old_conn.health_check_symbol = "BTC/USD"
        old_conn.disconnect = AsyncMock()
        mgr.exchanges["cb"] = old_conn
        mgr.exchange_configs["cb"] = ExchangeConfig(name="cb")

        new_conn = AsyncMock()
        new_conn.get_ticker = AsyncMock(return_value={"last": 1.0})

        secret_key = "SUPER_SECRET_KEY_12345"
        secret_val = "SUPER_SECRET_VAL_67890"

        with patch.object(mgr, "_create_connector", return_value=new_conn):
            with patch("core.exchange_manager.logger") as mock_logger:
                asyncio.run(mgr.rotate_api_keys("cb", secret_key, secret_val))

                # Check that no log call contains the actual secrets
                for call in mock_logger.info.call_args_list:
                    log_str = str(call)
                    assert secret_key not in log_str
                    assert secret_val not in log_str
