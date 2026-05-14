"""
Tests for config activation changes:
  - Strategy enablement (cross_exchange_arb, futures_basis_arb, delta_neutral_perp_arb,
    liquidation_cascade, funding_rate_harvester)
  - Feature enablement (news_sentiment, fear_greed)
  - Execution routing weights configurable via unified_config.yaml
  - api/__init__.py exports
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "unified_config.yaml"


@pytest.fixture(scope="module")
def full_config() -> Dict[str, Any]:
    """Load unified_config.yaml once for the module."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# TASK 1 — Strategy router strategy enablement
# ===========================================================================

class TestStrategyRouterActivation:
    """Strategies with exchange connectors should be enabled."""

    def test_cross_exchange_arb_enabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["cross_exchange_arb"]["enabled"] is True

    def test_futures_basis_arb_enabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["futures_basis_arb"]["enabled"] is True

    def test_delta_neutral_perp_arb_enabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["delta_neutral_perp_arb"]["enabled"] is True

    def test_liquidation_cascade_enabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["liquidation_cascade"]["enabled"] is True

    def test_funding_rate_harvester_enabled(self, full_config):
        assert full_config["funding_rate_harvester"]["enabled"] is True

    # Strategies that should STAY disabled
    def test_deribit_options_still_disabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["deribit_options"]["enabled"] is False

    def test_scalping_still_disabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["scalping"]["enabled"] is False

    def test_volatility_arb_still_disabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["volatility_arb"]["enabled"] is False

    def test_market_maker_still_disabled(self, full_config):
        strats = full_config["strategy_router"]["strategies"]
        assert strats["market_maker"]["enabled"] is False


# ===========================================================================
# TASK 1 cont. — Top-level strategy section enablement
# ===========================================================================

class TestTopLevelStrategyActivation:
    """Top-level strategy config sections should match strategy_router."""

    def test_delta_neutral_perp_arb_section_enabled(self, full_config):
        assert full_config["delta_neutral_perp_arb"]["enabled"] is True

    def test_volatility_arb_section_still_disabled(self, full_config):
        assert full_config["volatility_arb"]["enabled"] is False


# ===========================================================================
# TASK 2 — Feature enablement where backing code exists
# ===========================================================================

class TestFeatureActivation:
    """Features with complete implementations should be enabled."""

    def test_news_sentiment_enabled(self, full_config):
        assert full_config["news_sentiment"]["enabled"] is True

    def test_fear_greed_enabled(self, full_config):
        assert full_config["fear_greed"]["enabled"] is True


# ===========================================================================
# TASK 3 — Execution routing weights in config
# ===========================================================================

class TestExecutionRoutingConfig:
    """execution_routing section should exist and have correct defaults."""

    def test_execution_routing_section_exists(self, full_config):
        assert "execution_routing" in full_config

    def test_execution_routing_enabled(self, full_config):
        assert full_config["execution_routing"]["enabled"] is True

    def test_weights_section_exists(self, full_config):
        weights = full_config["execution_routing"]["weights"]
        assert isinstance(weights, dict)

    def test_weights_keys(self, full_config):
        weights = full_config["execution_routing"]["weights"]
        expected = {"liquidity", "latency", "fees", "reliability"}
        assert set(weights.keys()) == expected

    def test_weights_sum_to_one(self, full_config):
        weights = full_config["execution_routing"]["weights"]
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_max_venues_default(self, full_config):
        assert full_config["execution_routing"]["max_venues"] == 3


class TestSmartRoutingAlgorithmConfig:
    """SmartRoutingAlgorithm reads weights from config."""

    def test_default_weights_when_no_config(self):
        from execution.smart_order_router import SmartRoutingAlgorithm
        algo = SmartRoutingAlgorithm()
        assert algo.optimization_weights == SmartRoutingAlgorithm.DEFAULT_WEIGHTS

    def test_custom_weights_from_config(self):
        from execution.smart_order_router import SmartRoutingAlgorithm
        cfg = {
            "execution_routing": {
                "weights": {
                    "liquidity": 0.5,
                    "latency": 0.1,
                    "fees": 0.3,
                    "reliability": 0.1,
                },
                "max_venues": 5,
            }
        }
        algo = SmartRoutingAlgorithm(config=cfg)
        assert algo.optimization_weights["liquidity"] == 0.5
        assert algo.optimization_weights["fees"] == 0.3
        assert algo.max_venues == 5

    def test_partial_config_uses_defaults_for_missing(self):
        from execution.smart_order_router import SmartRoutingAlgorithm
        cfg = {"execution_routing": {"weights": {"liquidity": 0.6}}}
        algo = SmartRoutingAlgorithm(config=cfg)
        assert algo.optimization_weights["liquidity"] == 0.6
        # Others should fall back to defaults
        assert algo.optimization_weights["latency"] == 0.2
        assert algo.optimization_weights["fees"] == 0.2
        assert algo.optimization_weights["reliability"] == 0.2

    def test_max_venues_default_is_three(self):
        from execution.smart_order_router import SmartRoutingAlgorithm
        algo = SmartRoutingAlgorithm()
        assert algo.max_venues == 3


class TestSmartOrderRouterConfig:
    """SmartOrderRouter passes config through to algorithm."""

    def test_router_accepts_config(self):
        from execution.smart_order_router import SmartOrderRouter
        cfg = {
            "execution_routing": {
                "weights": {"liquidity": 0.7, "latency": 0.1, "fees": 0.1, "reliability": 0.1},
                "max_venues": 2,
            }
        }
        router = SmartOrderRouter(config=cfg)
        algo = router.algorithms["smart_routing"]
        assert algo.optimization_weights["liquidity"] == 0.7
        assert algo.max_venues == 2
        assert router.max_venues == 2

    def test_router_default_config(self):
        from execution.smart_order_router import SmartOrderRouter
        router = SmartOrderRouter()
        algo = router.algorithms["smart_routing"]
        assert algo.optimization_weights == {
            "liquidity": 0.4, "latency": 0.2, "fees": 0.2, "reliability": 0.2
        }


# ===========================================================================
# TASK 3 cont. — execution_routing registered as known top-level key
# ===========================================================================

class TestConfigManagerKnownKeys:
    """execution_routing must be in _KNOWN_TOP_LEVEL_KEYS."""

    def test_execution_routing_is_known_key(self):
        from core.config_manager import _KNOWN_TOP_LEVEL_KEYS
        assert "execution_routing" in _KNOWN_TOP_LEVEL_KEYS


# ===========================================================================
# TASK 4 — api/__init__.py exports
# ===========================================================================

class TestAPIExports:
    """api package should export dashboard and signal service classes."""

    def test_all_attribute(self):
        import api
        assert hasattr(api, "__all__")
        expected = {"ArgusAPIServer", "SignalService", "SignalDatabase", "RateLimiter"}
        assert set(api.__all__) == expected

    def test_lazy_import_dashboard(self):
        from api import ArgusAPIServer
        assert ArgusAPIServer is not None
        assert callable(ArgusAPIServer)

    def test_lazy_import_signal_service(self):
        from api import SignalService
        assert SignalService is not None
        assert callable(SignalService)

    def test_lazy_import_signal_database(self):
        from api import SignalDatabase
        assert SignalDatabase is not None

    def test_lazy_import_rate_limiter(self):
        from api import RateLimiter
        assert RateLimiter is not None

    def test_bad_attribute_raises(self):
        import api
        with pytest.raises(AttributeError):
            _ = api.NonExistentClass
