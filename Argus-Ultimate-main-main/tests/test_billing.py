"""Tests for api/billing.py — BillingManager tier management and rate limiting."""
from __future__ import annotations

import os
import tempfile
import time

import pytest

from api.billing import (
    BillingManager,
    TIER_BASIC,
    TIER_PRO,
    TIER_ENTERPRISE,
    Tier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Provide a temporary SQLite path."""
    return str(tmp_path / "test_billing.db")


@pytest.fixture
def billing(db_path):
    return BillingManager(config=None, db_path=db_path)


# ---------------------------------------------------------------------------
# Tier definition tests
# ---------------------------------------------------------------------------

class TestTierDefinitions:
    def test_basic_tier(self):
        assert TIER_BASIC.name == "BASIC"
        assert TIER_BASIC.price_monthly_usd == 29.0
        assert TIER_BASIC.signals_per_day == 5
        assert TIER_BASIC.delay_seconds == 900
        assert TIER_BASIC.api_access is False

    def test_pro_tier(self):
        assert TIER_PRO.name == "PRO"
        assert TIER_PRO.price_monthly_usd == 99.0
        assert TIER_PRO.signals_per_day == 0  # unlimited
        assert TIER_PRO.delay_seconds == 900
        assert TIER_PRO.api_access is False

    def test_enterprise_tier(self):
        assert TIER_ENTERPRISE.name == "ENTERPRISE"
        assert TIER_ENTERPRISE.price_monthly_usd == 299.0
        assert TIER_ENTERPRISE.signals_per_day == 0  # unlimited
        assert TIER_ENTERPRISE.delay_seconds == 0  # real-time
        assert TIER_ENTERPRISE.api_access is True

    def test_tier_is_frozen(self):
        with pytest.raises(AttributeError):
            TIER_BASIC.name = "MODIFIED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tier assignment and retrieval
# ---------------------------------------------------------------------------

class TestTierAssignment:
    def test_default_tier_is_basic(self, billing):
        tier = billing.get_subscriber_tier("nonexistent")
        assert tier.name == "BASIC"

    def test_assign_pro(self, billing):
        billing.assign_tier("sub-1", "PRO")
        tier = billing.get_subscriber_tier("sub-1")
        assert tier.name == "PRO"
        assert tier.price_monthly_usd == 99.0

    def test_assign_enterprise(self, billing):
        billing.assign_tier("sub-2", "ENTERPRISE")
        tier = billing.get_subscriber_tier("sub-2")
        assert tier.name == "ENTERPRISE"

    def test_reassign_tier(self, billing):
        billing.assign_tier("sub-3", "BASIC")
        billing.assign_tier("sub-3", "ENTERPRISE")
        tier = billing.get_subscriber_tier("sub-3")
        assert tier.name == "ENTERPRISE"

    def test_invalid_tier_defaults_to_basic(self, billing):
        tier = billing.assign_tier("sub-4", "NONEXISTENT")
        assert tier.name == "BASIC"


# ---------------------------------------------------------------------------
# Signal allowance (rate limiting)
# ---------------------------------------------------------------------------

class TestSignalAllowance:
    def test_basic_allows_up_to_5(self, billing):
        billing.assign_tier("sub-basic", "BASIC")
        for i in range(5):
            assert billing.check_signal_allowance("sub-basic") is True
            billing.record_signal_delivery("sub-basic", f"sig-{i}")
        # 6th should be blocked
        assert billing.check_signal_allowance("sub-basic") is False

    def test_pro_unlimited(self, billing):
        billing.assign_tier("sub-pro", "PRO")
        for i in range(20):
            assert billing.check_signal_allowance("sub-pro") is True
            billing.record_signal_delivery("sub-pro", f"sig-{i}")
        # Still allowed
        assert billing.check_signal_allowance("sub-pro") is True

    def test_enterprise_unlimited(self, billing):
        billing.assign_tier("sub-ent", "ENTERPRISE")
        for i in range(50):
            assert billing.check_signal_allowance("sub-ent") is True
            billing.record_signal_delivery("sub-ent", f"sig-{i}")


# ---------------------------------------------------------------------------
# Billing summary
# ---------------------------------------------------------------------------

class TestBillingSummary:
    def test_summary_basic(self, billing):
        billing.assign_tier("sub-s", "BASIC")
        billing.record_signal_delivery("sub-s", "sig-1")
        billing.record_signal_delivery("sub-s", "sig-2")

        summary = billing.get_billing_summary("sub-s")
        assert summary["tier"] == "BASIC"
        assert summary["price_monthly_usd"] == 29.0
        assert summary["signals_today"] == 2
        assert summary["signals_remaining_today"] == 3  # 5 - 2
        assert summary["total_signals_delivered"] == 2
        assert summary["delay_seconds"] == 900

    def test_summary_pro_unlimited(self, billing):
        billing.assign_tier("sub-p", "PRO")
        summary = billing.get_billing_summary("sub-p")
        assert summary["signals_remaining_today"] == -1  # unlimited


# ---------------------------------------------------------------------------
# Stripe stub
# ---------------------------------------------------------------------------

class TestStripeStub:
    def test_checkout_url_without_key(self, billing, monkeypatch):
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
        url = billing.create_checkout_session("sub-1", "PRO")
        assert "stripe.com" in url
        assert "PRO" in url

    def test_list_tiers(self):
        tiers = BillingManager.list_tiers()
        assert len(tiers) == 3
        names = [t["name"] for t in tiers]
        assert "BASIC" in names
        assert "PRO" in names
        assert "ENTERPRISE" in names
