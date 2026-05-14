#!/usr/bin/env python3
"""
tests_unified/test_dual_node.py — Argus v6.4.0
================================================
Unit tests for the dual-node orchestration and state-sync layer.

Coverage:
  - NodeOrchestrator: role detection (env, hostname, default)
  - NodeOrchestrator: apply_role side-effects
  - NodeOrchestrator: get_node_info completeness
  - DuplicateOrderGuard: block and allow scenarios
  - GitHubStateSync: secret stripping from config snapshots
  - GitHubStateSync: capital_state.json schema validation
  - PaperTradeEngine: deterministic fill (prob=1.0)
  - PaperTradeEngine: slippage applied to fill price
  - PaperTradeEngine: expired order (prob=0.0)
  - PaperTradeEngine: ComparisonResult fields
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from core.node_orchestrator import (
    DuplicateOrderGuard,
    NodeConfig,
    NodeOrchestrator,
    NodeRole,
)
from core.github_state_sync import GitHubStateSync, SyncConfig, _strip_secrets
from core.paper_trade_engine import (
    ComparisonResult,
    OrderStatus,
    PaperConfig,
    PaperTradeEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orch(role_env: str = "", hostname_override: str = "") -> NodeOrchestrator:
    """Create a NodeOrchestrator with a temp state_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pass  # just to get a unique path string (we recreate below)

    tmpdir = tempfile.mkdtemp(prefix="argus_test_")
    cfg = NodeConfig(
        live_server_hostname="argus-r7525",
        state_dir=tmpdir,
        config_path="/nonexistent/path.yaml",
    )
    orch = NodeOrchestrator(cfg)
    if hostname_override:
        orch._hostname = hostname_override
    return orch


def _run(coro: Any) -> Any:
    """Run a coroutine synchronously in a fresh event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# NodeOrchestrator — role detection
# ---------------------------------------------------------------------------

class TestNodeRoleDetection:

    def test_node_role_env_live(self):
        """ARGUS_ROLE=live → NodeRole.LIVE_SERVER."""
        orch = _make_orch()
        with patch.dict(os.environ, {"ARGUS_ROLE": "live"}):
            role = orch.detect_role()
        assert role == NodeRole.LIVE_SERVER, (
            f"Expected LIVE_SERVER when ARGUS_ROLE=live, got {role}"
        )

    def test_node_role_env_paper(self):
        """ARGUS_ROLE=paper → NodeRole.PAPER_PC."""
        orch = _make_orch()
        with patch.dict(os.environ, {"ARGUS_ROLE": "paper"}):
            role = orch.detect_role()
        assert role == NodeRole.PAPER_PC, (
            f"Expected PAPER_PC when ARGUS_ROLE=paper, got {role}"
        )

    def test_node_role_default(self):
        """No ARGUS_ROLE env and no hostname match → NodeRole.STANDALONE."""
        orch = _make_orch()
        orch._hostname = "some-random-machine"
        # Ensure env variable is absent
        env = {k: v for k, v in os.environ.items() if k != "ARGUS_ROLE"}
        with patch.dict(os.environ, env, clear=True):
            role = orch.detect_role()
        assert role == NodeRole.STANDALONE, (
            f"Expected STANDALONE as default, got {role}"
        )

    def test_node_role_hostname_live(self):
        """Hostname matches live_server_hostname → LIVE_SERVER."""
        orch = _make_orch(hostname_override="argus-r7525")
        env = {k: v for k, v in os.environ.items() if k != "ARGUS_ROLE"}
        with patch.dict(os.environ, env, clear=True):
            role = orch.detect_role()
        assert role == NodeRole.LIVE_SERVER


# ---------------------------------------------------------------------------
# NodeOrchestrator — apply_role
# ---------------------------------------------------------------------------

class TestNodeApplyRole:

    def test_node_apply_live(self):
        """
        After apply_role() on a LIVE_SERVER node:
        - ARGUS_PAPER_TRADING env must be 'false'
        - is_live() must return True
        """
        orch = _make_orch()
        with patch.dict(os.environ, {"ARGUS_ROLE": "live"}):
            orch.detect_role()
            orch.apply_role()
            assert orch.is_live() is True
            assert os.environ.get("ARGUS_PAPER_TRADING", "").lower() == "false", (
                "ARGUS_PAPER_TRADING must be 'false' after apply_role for LIVE_SERVER"
            )

    def test_node_apply_paper(self):
        """
        After apply_role() on a PAPER_PC node:
        - ARGUS_PAPER_TRADING env must be 'true'
        - is_paper() must return True
        - ARGUS_LIVE_ORDERS must be 'false'
        """
        orch = _make_orch()
        with patch.dict(os.environ, {"ARGUS_ROLE": "paper"}):
            orch.detect_role()
            orch.apply_role()
            assert orch.is_paper() is True
            assert os.environ.get("ARGUS_PAPER_TRADING", "").lower() == "true", (
                "ARGUS_PAPER_TRADING must be 'true' after apply_role for PAPER_PC"
            )
            assert os.environ.get("ARGUS_LIVE_ORDERS", "").lower() == "false"


# ---------------------------------------------------------------------------
# NodeOrchestrator — node info
# ---------------------------------------------------------------------------

class TestNodeInfo:

    def test_node_info_keys(self):
        """get_node_info() must contain all required keys."""
        required_keys = {
            "node_id", "role", "hostname", "ip",
            "cpu_count", "ram_gb",
        }
        orch = _make_orch()
        info = orch.get_node_info()
        missing = required_keys - set(info.keys())
        assert not missing, f"get_node_info() missing keys: {missing}"

    def test_node_info_types(self):
        """node_id, role, hostname are non-empty strings; cpu_count ≥ 1; ram_gb > 0."""
        orch = _make_orch()
        info = orch.get_node_info()
        assert isinstance(info["node_id"], str) and info["node_id"]
        assert isinstance(info["role"], str) and info["role"]
        assert isinstance(info["hostname"], str) and info["hostname"]
        assert isinstance(info["cpu_count"], int) and info["cpu_count"] >= 1
        assert isinstance(info["ram_gb"], float) and info["ram_gb"] > 0.0


# ---------------------------------------------------------------------------
# DuplicateOrderGuard
# ---------------------------------------------------------------------------

class TestDuplicateOrderGuard:

    def _make_guard(self, remote_orders: list) -> DuplicateOrderGuard:
        """Create a guard with a pre-populated active_orders.json."""
        tmpdir = tempfile.mkdtemp(prefix="argus_guard_")
        orders_path = Path(tmpdir) / "active_orders.json"
        orders_path.write_text(json.dumps(remote_orders))
        return DuplicateOrderGuard(state_dir=tmpdir, local_node_id="local-node")

    def test_duplicate_guard_blocks(self):
        """
        If the remote node has a matching (symbol, side, exchange) order,
        check_order() must return False.
        """
        remote_orders = [
            {
                "node_id": "r7525-live",
                "symbol":  "BTC/USDT",
                "side":    "buy",
                "exchange": "binance",
                "status":  "open",
            }
        ]
        guard = self._make_guard(remote_orders)
        result = guard.check_order("BTC/USDT", "buy", "binance")
        assert result is False, (
            "DuplicateOrderGuard should block an order matching the remote node's active order"
        )

    def test_duplicate_guard_allows(self):
        """
        If the remote node has an order for a DIFFERENT symbol,
        check_order() must return True.
        """
        remote_orders = [
            {
                "node_id": "r7525-live",
                "symbol":  "ETH/USDT",
                "side":    "buy",
                "exchange": "binance",
                "status":  "open",
            }
        ]
        guard = self._make_guard(remote_orders)
        result = guard.check_order("BTC/USDT", "buy", "binance")
        assert result is True, (
            "DuplicateOrderGuard should allow an order for a different symbol"
        )

    def test_duplicate_guard_allows_own_order(self):
        """
        Orders belonging to the *local* node should not block new orders.
        """
        remote_orders = [
            {
                "node_id": "local-node",   # same as local_node_id
                "symbol":  "BTC/USDT",
                "side":    "buy",
                "exchange": "binance",
                "status":  "open",
            }
        ]
        guard = self._make_guard(remote_orders)
        result = guard.check_order("BTC/USDT", "buy", "binance")
        assert result is True


# ---------------------------------------------------------------------------
# GitHubStateSync — secret stripping
# ---------------------------------------------------------------------------

class TestGitHubStateSync:

    def test_sync_strips_secrets(self):
        """
        config_snapshot must not contain values for keys matching
        *_key, *_secret, *_mnemonic, *_password, *_token, *_passphrase, *_seed.
        """
        raw_config: Dict[str, Any] = {
            "exchange": "binance",
            "api_key": "SUPER_SECRET_KEY_12345",
            "api_secret": "SUPER_SECRET_VALUE",
            "wallet_mnemonic": "word1 word2 word3 word4 word5 word6 word7 word8",
            "jwt_token": "eyJhbGciOiJIUzI1NiJ9.test",
            "db_password": "hunter2",
            "paper_trading": True,
            "max_positions": 8,
        }
        sanitised = _strip_secrets(raw_config)

        secret_fields = ["api_key", "api_secret", "wallet_mnemonic", "jwt_token", "db_password"]
        for field in secret_fields:
            assert sanitised.get(field) == "<REDACTED>", (
                f"Field '{field}' should be redacted but got: {sanitised.get(field)!r}"
            )

        # Non-secret fields must pass through unchanged
        assert sanitised["exchange"] == "binance"
        assert sanitised["paper_trading"] is True
        assert sanitised["max_positions"] == 8

    def test_sync_strips_secrets_nested(self):
        """Secret stripping must recurse into nested dicts and lists."""
        raw_config = {
            "exchanges": {
                "binance": {
                    "api_key": "hidden",
                    "name": "Binance",
                }
            },
            "wallets": [
                {"seed": "private seed phrase", "address": "0x123"},
            ],
        }
        sanitised = _strip_secrets(raw_config)
        assert sanitised["exchanges"]["binance"]["api_key"] == "<REDACTED>"
        assert sanitised["exchanges"]["binance"]["name"] == "Binance"
        assert sanitised["wallets"][0]["seed"] == "<REDACTED>"
        assert sanitised["wallets"][0]["address"] == "0x123"

    def test_sync_state_schema(self):
        """
        capital_state.json written by push_state must contain:
        total_usd, available_usd, locked_usd, equity_usd, timestamp_utc.
        """
        required_keys = {"total_usd", "available_usd", "locked_usd", "equity_usd", "timestamp_utc"}

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SyncConfig(
                repo_path=".",
                state_dir=tmpdir,
                node_id="test-node",
            )
            sync = GitHubStateSync(cfg)

            state = {
                "capital": {
                    "total_usd":     620.0,
                    "available_usd": 600.0,
                    "locked_usd":    20.0,
                    "equity_usd":    625.0,
                },
                "pnl_history":  [],
                "positions":    [],
                "orders":       [],
                "health":       {},
                "config":       {},
            }

            # Manually call _write_state_files (avoids git)
            sync._write_state_files(tmpdir, state)

            capital_path = Path(tmpdir) / "capital_state.json"
            assert capital_path.exists(), "capital_state.json was not written"
            capital_data = json.loads(capital_path.read_text())

            missing = required_keys - set(capital_data.keys())
            assert not missing, (
                f"capital_state.json is missing required keys: {missing}"
            )


# ---------------------------------------------------------------------------
# PaperTradeEngine
# ---------------------------------------------------------------------------

class TestPaperTradeEngine:

    def test_paper_trade_fill(self):
        """
        With fill_probability=1.0, every market order must be FILLED.
        """
        cfg = PaperConfig(
            initial_capital_usd=620.0,
            fill_probability=1.0,
            slippage_bps=2.0,
        )
        engine = PaperTradeEngine(cfg)
        engine.update_mark_price("BTC/USDT", 65000.0)

        order = _run(engine.create_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            price=65000.0,
            exchange="binance",
        ))

        assert order.status == OrderStatus.FILLED, (
            f"Expected FILLED with fill_probability=1.0, got {order.status}"
        )
        assert order.fill_price is not None
        assert order.fill_quantity == pytest.approx(0.001, rel=1e-6)

    def test_paper_trade_slippage(self):
        """
        Fill price for a buy order must be > requested price by ~slippage_bps.
        """
        slippage_bps = 5.0
        ref_price    = 50000.0
        cfg = PaperConfig(
            initial_capital_usd=620.0,
            fill_probability=1.0,
            slippage_bps=slippage_bps,
        )
        engine = PaperTradeEngine(cfg)

        order = _run(engine.create_order(
            symbol="ETH/USDT",
            side="buy",
            order_type="market",
            quantity=0.01,
            price=ref_price,
            exchange="binance",
        ))

        expected_fill = ref_price * (1.0 + slippage_bps / 10_000)
        assert order.fill_price == pytest.approx(expected_fill, rel=1e-6), (
            f"Expected fill_price ≈ {expected_fill:.4f} (slippage={slippage_bps} bps), "
            f"got {order.fill_price}"
        )

    def test_paper_trade_sell_slippage(self):
        """
        Fill price for a sell order must be < requested price by ~slippage_bps.
        """
        slippage_bps = 3.0
        ref_price    = 3000.0
        cfg = PaperConfig(
            initial_capital_usd=10000.0,
            fill_probability=1.0,
            slippage_bps=slippage_bps,
        )
        engine = PaperTradeEngine(cfg)
        # Open a long first so we have something to sell
        _run(engine.create_order("ETH/USDT", "buy", "market", 1.0, ref_price, "binance"))

        order = _run(engine.create_order(
            symbol="ETH/USDT",
            side="sell",
            order_type="market",
            quantity=1.0,
            price=ref_price,
            exchange="binance",
        ))

        expected_fill = ref_price * (1.0 - slippage_bps / 10_000)
        assert order.fill_price == pytest.approx(expected_fill, rel=1e-6)

    def test_paper_trade_no_fill(self):
        """
        With fill_probability=0.0, every order must be EXPIRED.
        """
        cfg = PaperConfig(
            initial_capital_usd=620.0,
            fill_probability=0.0,
            slippage_bps=2.0,
        )
        engine = PaperTradeEngine(cfg)

        order = _run(engine.create_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=0.001,
            price=60000.0,
            exchange="bybit",
        ))

        assert order.status == OrderStatus.EXPIRED, (
            f"Expected EXPIRED with fill_probability=0.0, got {order.status}"
        )
        assert order.fill_price is None
        assert order.fill_quantity == 0.0

    def test_paper_comparison_result(self):
        """
        compare_to_live() must return a ComparisonResult with all required fields
        and correct types.
        """
        cfg = PaperConfig(
            initial_capital_usd=620.0,
            fill_probability=1.0,
            slippage_bps=2.0,
            track_live_comparison=True,
        )
        engine = PaperTradeEngine(cfg)
        engine.update_mark_price("BTC/USDT", 65000.0)

        # Execute a round-trip paper trade to generate P&L
        _run(engine.create_order("BTC/USDT", "buy",  "market", 0.01, 65000.0, "binance"))
        _run(engine.create_order("BTC/USDT", "sell", "market", 0.01, 66000.0, "binance"))

        # Simulate live P&L history (10 trades)
        live_history = [
            {"realised_pnl": 5.0 * (1 if i % 2 == 0 else -2.5)}
            for i in range(10)
        ]

        result = engine.compare_to_live(live_history)

        # Must be a ComparisonResult
        assert isinstance(result, ComparisonResult)

        # All required fields must be present and typed correctly
        assert isinstance(result.paper_pnl,         float)
        assert isinstance(result.live_pnl,          float)
        assert isinstance(result.difference,        float)
        assert isinstance(result.paper_win_rate,    float),  "paper_win_rate must be float"
        assert isinstance(result.live_win_rate,     float),  "live_win_rate must be float"
        assert isinstance(result.paper_trade_count, int),    "paper_trade_count must be int"
        assert isinstance(result.live_trade_count,  int),    "live_trade_count must be int"
        assert isinstance(result.recommendation,    str),    "recommendation must be str"
        assert isinstance(result.generated_at,      str),    "generated_at must be str"

        # Difference must equal paper - live
        assert result.difference == pytest.approx(
            result.paper_pnl - result.live_pnl, rel=1e-5
        )

        # live_trade_count must match the input history length
        assert result.live_trade_count == 10

        # to_dict() must serialise cleanly
        d = result.to_dict()
        assert set(d.keys()) >= {
            "paper_pnl", "live_pnl", "difference", "correlation",
            "paper_win_rate", "live_win_rate", "paper_trade_count",
            "live_trade_count", "recommendation", "generated_at",
        }
